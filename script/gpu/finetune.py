#!/usr/bin/env python
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import inspect
import json
import logging
import math
import os
import sys
import threading
import time
import transformers
import copy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datetime import datetime
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from typing import Optional, List

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

CNVRG_LINE_FORMAT = '\"cnvrg_linechart_{} value: {}\"\n'


class CustomPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            if 'loss' in logs:
                print(CNVRG_LINE_FORMAT.format('loss', str(logs['loss'])))
            if 'learning_rate' in logs:
                print(CNVRG_LINE_FORMAT.format('learningrate', str(logs['learning_rate'])))
            if 'epoch' in logs:
                print(CNVRG_LINE_FORMAT.format('epoch', str(logs['epoch'])))


# ═════════════════════════════════════════════════════════════════════════════
# Data-transfer profiling: where does training time and data go
# across the two transfer boundaries — gradient AllReduce (network) and dataset
# loading (NFS/storage)?  
#
# Measurement notes
#  * Timers are closed with torch.cuda.synchronize() on GPU; CUDA is async and
#    without it the fractions would measure kernel-launch time.
#  * AllReduce time is the UNION of in-flight intervals seen by a DDP comm hook
#    (the clock runs while ≥1 bucket is outstanding). Summing per-bucket
#    durations double-counts overlapping buckets. DDP overlaps communication with
#    the backward pass, so this is an UPPER BOUND on exclusive comm time.
#  * Bytes: total_reduced_bytes is what the hook observed; per step it must equal
#    grad_size_bytes (= bytes of all parameters with requires_grad — only the
#    adapters under LoRA, every parameter in full fine-tuning). That equality is
#    reported as hook_coverage. total_allreduce_bytes is the ring-AllReduce
#    estimate 2(n−1)/n × reduced bytes — derived, not measured on the wire.
#  * Compute = on_step_begin → on_step_end. Dataload = previous on_step_end →
#    this on_step_begin (the Trainer fetches the batch in that gap). Logging and
#    in-training checkpoints also land in that gap, so keep save_strategy at
#    epoch/no and evaluate after training. Accurate with
#    gradient_accumulation_steps == 1 (the step callbacks fire per optimizer step).
#  * NFS bytes come from /proc/<pid>/io summed over this process and its
#    descendants (DataLoader workers do the reads). Rank 0's view only.
#  * The first --profile_warmup_steps steps are excluded from every aggregate.
# ═════════════════════════════════════════════════════════════════════════════
def _descendant_pids(root):
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return set()
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        children.setdefault(int(line.split()[1]), []).append(int(entry))
                        break
        except OSError:
            continue
    out, stack = set(), [root]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def read_tree_io():
    """Sum read_bytes/write_bytes over this process and all its descendants."""
    total_r = total_w = 0
    for pid in {os.getpid()} | _descendant_pids(os.getpid()):
        try:
            with open(f"/proc/{pid}/io") as f:
                fields = dict(line.strip().split(": ") for line in f)
            total_r += int(fields.get("read_bytes", 0))
            total_w += int(fields.get("write_bytes", 0))
        except (OSError, ValueError):
            continue
    return total_r, total_w


def dir_size_bytes(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class CommState:
    def __init__(self, world_size):
        self.world_size = world_size
        self.wire_factor = 2.0 * (world_size - 1) / world_size if world_size > 1 else 0.0
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.total_s = 0.0       # union of in-flight intervals
        self.bucket_sum_s = 0.0  # plain per-bucket sum (overlaps double-counted)
        self.reduced_bytes = 0
        self.wire_bytes = 0.0
        self.inflight = 0
        self.span_start = 0.0


def allreduce_hook(state, bucket):
    t0 = time.perf_counter()
    buf = bucket.buffer()
    nbytes = buf.numel() * buf.element_size()
    with state.lock:
        if state.inflight == 0:
            state.span_start = t0
        state.inflight += 1
        state.reduced_bytes += nbytes
        state.wire_bytes += nbytes * state.wire_factor
    buf.div_(state.world_size)  # average, as DDP's default AllReduce does
    fut = dist.all_reduce(buf, async_op=True).get_future()

    def done(f):
        now = time.perf_counter()
        with state.lock:
            state.bucket_sum_s += now - t0
            state.inflight -= 1
            if state.inflight == 0:
                state.total_s += now - state.span_start
        return f.value()[0]

    return fut.then(done)


class TransferProfileCallback(TrainerCallback):
    """Collects the transfer profile during trainer.train(); write it with
    finalize() after the final model save so checkpoint bytes are included."""

    def __init__(self, trainer, warmup_steps):
        self.trainer = trainer
        self.warmup = warmup_steps
        self.comm = CommState(1)
        self.world_size = 1
        self.use_cuda = False
        self.hook_installed = False
        self.grad_size_bytes = 0
        self.num_params = 0
        self.trainable_params = 0
        self.compute_times, self.dataload_times = [], []
        self.step_start = self.last_step_end = self.train_start = self.measured_end = None
        self.recording = False
        self.io0 = (0, 0)

    def _sync(self):
        if self.use_cuda:
            torch.cuda.synchronize()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        params = list(model.parameters())
        self.use_cuda = torch.cuda.is_available() and params[0].device.type == "cuda"
        self.num_params = sum(p.numel() for p in params)
        self.trainable_params = sum(p.numel() for p in params if p.requires_grad)
        self.grad_size_bytes = sum(p.numel() * p.element_size() for p in params if p.requires_grad)
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.comm = CommState(self.world_size)
        wrapped = self.trainer.model_wrapped  # DDP-wrapped before on_train_begin fires
        if self.world_size > 1 and isinstance(wrapped, DDP):
            wrapped.register_comm_hook(self.comm, allreduce_hook)
            self.hook_installed = True
        self.last_step_end = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if state.global_step == self.warmup and self.train_start is None:
            self._sync()
            self.comm.reset()
            self.io0 = read_tree_io()
            self.train_start = self.last_step_end  # this step's batch fetch started there
        self.recording = state.global_step >= self.warmup
        if self.recording:
            self.dataload_times.append(now - self.last_step_end)
        self.step_start = now

    def on_step_end(self, args, state, control, **kwargs):
        self._sync()
        now = time.perf_counter()
        if self.recording:
            self.compute_times.append(now - self.step_start)
            self.measured_end = now
        self.last_step_end = now

    def finalize(self, output_dir, data_source):
        """Return the transfer profile dict and write transfer_profile.json."""
        args = self.trainer.args
        steps = len(self.compute_times)
        total_s = (self.measured_end - self.train_start) if steps and self.train_start is not None else 0.0
        io1 = read_tree_io()
        nfs_read_b = max(0, io1[0] - self.io0[0])
        nfs_write_b = max(0, io1[1] - self.io0[1])
        total_dataload_s = sum(self.dataload_times)
        total_compute_s = sum(self.compute_times)
        samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.world_size
        total_samples = steps * samples_per_step
        wire_bytes = int(round(self.comm.wire_bytes))
        frac = lambda x: x / total_s if total_s > 0 else 0.0
        reduced_per_step = self.comm.reduced_bytes / steps if steps else 0
        hook_coverage = reduced_per_step / self.grad_size_bytes if (self.hook_installed and self.grad_size_bytes) else None

        profile = {
            "model": self.trainer.model.config._name_or_path if hasattr(self.trainer.model, "config") else "",
            "num_params": self.num_params,
            "trainable_params": self.trainable_params,
            "world_size": self.world_size,
            "data_source": data_source,
            "synthetic": data_source == "synthetic",
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "hook_installed": self.hook_installed,

            "total_training_s": total_s,
            "num_steps": steps,
            "warmup_steps": self.warmup,
            "total_samples": total_samples,
            "avg_step_s": total_s / steps if steps else 0.0,
            "avg_compute_s": total_compute_s / steps if steps else 0.0,
            "avg_dataload_s": total_dataload_s / steps if steps else 0.0,
            "total_allreduce_s": self.comm.total_s,
            "allreduce_bucket_sum_s": self.comm.bucket_sum_s,
            "allreduce_fraction": frac(self.comm.total_s),
            "dataload_fraction": frac(total_dataload_s),
            "compute_fraction": frac(total_compute_s),

            "grad_size_bytes": self.grad_size_bytes,
            "total_reduced_bytes": self.comm.reduced_bytes,
            "hook_coverage": hook_coverage,
            "total_allreduce_bytes": wire_bytes,
            "nfs_read_bytes": nfs_read_b,
            "nfs_write_bytes": nfs_write_b,
            "checkpoint_size_bytes": dir_size_bytes(output_dir),

            "allreduce_bandwidth_MBs": (wire_bytes / 1e6) / self.comm.total_s if self.comm.total_s > 0 else 0.0,
            "nfs_read_bandwidth_MBs": (nfs_read_b / 1e6) / total_dataload_s if total_dataload_s > 0 else 0.0,
            "nfs_write_bandwidth_MBs": (nfs_write_b / 1e6) / total_s if total_s > 0 else 0.0,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "transfer_profile.json"), "w") as f:
            json.dump(profile, f, indent=2)
        return profile


@dataclass
class ProfileArguments:
    """Arguments for the data-transfer profile (transfer_profile.json)."""
    profile_warmup_steps: int = field(
        default=5,
        metadata={"help": "Optimizer steps excluded from the transfer profile (CUDA init, autotune, allocator growth)."},
    )
    disable_transfer_profile: bool = field(
        default=False,
        metadata={"help": "Do not collect or write transfer_profile.json."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    validation_split_percentage: Optional[float] = field(
        default=0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep in memory the loaded dataset. Defaults to False."
        },
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and "
                    "the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load "
                    "processed dataset from this path."
        }
    )
    dataset_concatenation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to concatenate the sentence for more efficient training."
        }
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set to True, don’t download the data files. Instead, it streams the data progressively while "
                    "iterating on the dataset."
        }
    )
    prompt_with_input: str = field(
        default="Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.",
        metadata={"help": "The prompt string to use with an instruction that has an input/context string"}
    )
    prompt_without_input: str = field(
        default="Below is an instruction that describes a task. Write a response that appropriately completes the "
                "request.",
        metadata={"help": "The prompt string to use with an instruction that does not include an input/context string."}
    )
    synthetic_samples: int = field(
        default=0,
        metadata={
            "help": "If > 0, train on this many rows of random tokens instead of a dataset (no storage reads; "
                    "isolates AllReduce). Also selected by --dataset_name synthetic together with --max_train_samples."
        }
    )
    synthetic_seq_len: int = field(
        default=512,
        metadata={"help": "Sequence length of synthetic rows."}
    )
    pretokenized_path: str = field(
        default=None,
        metadata={
            "help": "Load an already tokenised dataset (datasets.save_to_disk layout with input_ids/attention_mask) "
                    "instead of downloading and tokenising. A --dataset_name that is an existing directory is "
                    "treated the same way."
        }
    )


@dataclass
class FinetuneArguments:
    """
    Arguments of finetune we are going to apply on the model.
    """
    use_lora: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use LoRA."
        },
    )

    lora_rank: int = field(
        default=8,
        metadata={
            "help": "Rank parameter in the LoRA method."
        },
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "Alpha parameter in the LoRA method."
        },
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout parameter in the LoRA method."
        },
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={
            "help": "Target modules for the LoRA method."
        },
    )


@dataclass
class QuantizationArguments:
    """
    Arguments used to do weights only post training quantization using the Intel Neural Compressor.
    """
    peft_model_dir: str = field(
        default=None,
        metadata={
            "help": "If not training the model, load a model that was previously trained using peft."
        },
    )
    do_quantize: bool = field(
        default=False,
        metadata={
            "help": "Perform weight only post training quantization."
        },
    )
    quantize_output_dir: str = field(
        default=None,
        metadata={
            "help": "If quantization is successful, the quantized model will be saved to this directory."
        },
    )
    woq_bits: int = field(
        default=8,
        metadata={
            "help": "Bits for weight only quantization, 1-8 bits."
        },
    )
    woq_group_size: int = field(
        default=-1,
        metadata={
            "help": "Group size for weight only quantization. Group_size=[1-N] indicates "
                    "splitting the input channel elements per group_size. -1 indicates "
                    "the per-channel quantization per output channel."
        },
    )
    woq_scheme: str = field(
        default="sym",
        metadata={
            "help": "Scheme for weight only quantization. Choose from 'sym' and 'asym'."
        },
    )
    woq_algo: str = field(
        default="RTN",
        metadata={
            "help": "Algorithm for weight only quantization. Choose from: RTN, AWQ, GPTQ, or TEQ"
        },
    )


@dataclass
class BenchmarkArguments:
    """
    Arguments used to benchmark the model using the Intel Neural Compressor.
    """
    do_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Benchmark the trained and/or quantized model."
        },
    )
    benchmark_warmup: int = field(
        default=10,
        metadata={
            "help": "The number of iterations to warmup before running performance tests."
        },
    )
    benchmark_iterations: int = field(
        default=100,
        metadata={
            "help": "The number of iterations to run performance tests."
        },
    )
    benchmark_cores_per_instance: int = field(
        default=None,
        metadata={
            "help": "The of CPU cores to use per instance"
        },
    )
    benchmark_num_instances: int = field(
        default=1,
        metadata={
            "help": "The number of instances to use for performance testing"
        },
    )


# Prompt dictionary with output used for chat models
CHAT_PROMPT_DICT = {
    "prompt_with_input": (
        "<s>[INST] <<SYS>>\n{prompt_with_input}<</SYS>>\n\n{instruction} {input} [/INST] "
        "{output} </s>"
    ),
    "prompt_without_input": (
        "<s>[INST] <<SYS>>\n{prompt_without_input}<</SYS>>\n\n{instruction} [/INST] {output} </s>"
    ),
}


# Prompt dictionary without output used for chat models
CHAT_PROMPT_DICT2 = {
    "prompt_with_input": (
        "<s>[INST] <<SYS>>\n{prompt_with_input}\n\n{instruction} {input} [/INST]"
    ),
    "prompt_without_input": (
        "<s>[INST] <<SYS>>\n{prompt_without_input}<</SYS>>\n\n{instruction} [/INST]"
    ),
}

# Prompt dictionary without output used for instruction tuning
STANDARD_PROMPT_DICT = {
    "prompt_with_input": (
        "{prompt_with_input}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    ),
    "prompt_without_input": (
        "{prompt_without_input}\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ),
}

# Prompt dictionary without output used for instruction tuning
STANDARD_PROMPT_DICT2 = {
    "prompt_with_input": (
        "{prompt_with_input}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_without_input": (
        "{prompt_without_input}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def create_prompts(examples, prompt_dict):
    prompts = []

    for example in examples:
        prompt_template = prompt_dict["prompt_with_input"] \
            if example["input"] != "" else prompt_dict["prompt_without_input"]
        prompt = prompt_template.format_map(example)
        prompts.append(prompt)
    return prompts


def create_system_turn(examples, prompt_dict):
    prompts = []

    for example in examples:
        prompt_template = prompt_dict["prompt_with_input"] \
            if example["input"] != "" else prompt_dict["prompt_without_input"]
        prompt = prompt_template.format_map(example)
        prompts.append(prompt)
    return prompts


def main():
    start_time = datetime.now()

    # Local smoke tests may run several ranks on one GPU (gloo backend); wrap
    # LOCAL_RANK onto the devices that exist before TrainingArguments picks the
    # device. A no-op on clusters with one process per GPU.
    if torch.cuda.is_available():
        local_rank, n_dev = int(os.environ.get("LOCAL_RANK", 0)), torch.cuda.device_count()
        if local_rank >= n_dev:
            os.environ["LOCAL_RANK"] = str(local_rank % n_dev)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, FinetuneArguments, ProfileArguments))
    model_args, data_args, training_args, finetune_args, profile_args, optim_args = \
        parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the summary
    b16 = training_args.fp16 or training_args.bf16
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}\ndistributed training: "
        f"{bool(training_args.local_rank != -1)}, 16-bits training: {b16}"
    )
    # Set the verbosity to info of the Transformers logger (main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
        "add_bos_token": False,
        "add_eos_token": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    # Create the dataset cache directory, if one is specified
    if data_args.dataset_cache_directory:
        os.makedirs(data_args.dataset_cache_directory, exist_ok=True)

    # Data modes:
    #   synthetic    — random tokens, no storage reads (thesis E1 series)
    #   pretokenized — datasets.load_from_disk, already tokenised (e.g. pretokenize.py output)
    #   text         — a plain-text corpus with a `text` column (e.g. WikiText-103)
    #   instruction  — the original instruction/input/output prompt pipeline
    if data_args.dataset_name == "synthetic" and not data_args.synthetic_samples:
        data_args.synthetic_samples = data_args.max_train_samples or 1000
    if data_args.dataset_name and not data_args.pretokenized_path and os.path.isdir(data_args.dataset_name):
        data_args.pretokenized_path = data_args.dataset_name

    if data_args.synthetic_samples > 0:
        data_mode = "synthetic"
        data_source = "synthetic"
        g = torch.Generator().manual_seed(data_args.dataset_seed)
        ids = torch.randint(0, config.vocab_size, (data_args.synthetic_samples, data_args.synthetic_seq_len), generator=g)
        raw_datasets = DatasetDict({"train": Dataset.from_dict({
            "input_ids": ids.tolist(),
            "attention_mask": torch.ones_like(ids).tolist(),
            "labels": ids.tolist(),
        })})
    elif data_args.pretokenized_path:
        data_mode = "pretokenized"
        data_source = data_args.pretokenized_path
        loaded = load_from_disk(data_args.pretokenized_path)
        raw_datasets = loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})
    elif data_args.dataset_name:
        # Downloading and loading a dataset from the hub.
        data_source = data_args.dataset_name
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.dataset_cache_directory,
            token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        cols = raw_datasets["train"].column_names
        data_mode = "text" if ("text" in cols and "instruction" not in cols) else "instruction"
    else:
        raise ValueError("Must provide dataset_name to download from the hub.")
    logger.info(f"Data mode: {data_mode} ({data_source})")


    # If no test data is there, validation_split_percentage will be used to divide the dataset.
    if "test" not in raw_datasets.keys() and training_args.do_eval:
        logger.info("Original dataset length: {}".format(len(raw_datasets["train"])))
        raw_datasets["train"] = raw_datasets["train"].shuffle(seed=data_args.dataset_seed)
        raw_datasets = raw_datasets["train"].train_test_split(test_size=data_args.validation_split_percentage)
        logger.info("Validation split percentage: {}".format(data_args.validation_split_percentage))
        logger.info("Train split length: {}".format(len(raw_datasets["train"])))
        logger.info("Test split length: {}".format(len(raw_datasets["test"])))

    if data_mode != "instruction" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if data_mode == "synthetic":
        tokenized_datasets = raw_datasets
    elif data_mode == "pretokenized":
        def add_labels(example):
            return {"labels": [t if m else -100 for t, m in zip(example["input_ids"], example["attention_mask"])]}

        with training_args.main_process_first(desc="pretokenized dataset labels"):
            tokenized_datasets = raw_datasets if "labels" in raw_datasets["train"].column_names \
                else raw_datasets.map(add_labels, load_from_cache_file=not data_args.overwrite_cache)
    elif data_mode == "text":
        def preprocess_text(examples):
            results = tokenizer(
                examples["text"], truncation=True, max_length=data_args.max_seq_length, padding=False,
            )
            results["labels"] = copy.deepcopy(results["input_ids"])
            return results

        with training_args.main_process_first(desc="text dataset map pre-processing"):
            raw_datasets = raw_datasets.filter(
                lambda ex: ex["text"].strip() != "", load_from_cache_file=not data_args.overwrite_cache,
            )
            tokenized_datasets = raw_datasets.map(
                preprocess_text, batched=True, remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Get prompt strings from the user provided arguments
    prompt_dict = CHAT_PROMPT_DICT if "chat" in model_args.model_name_or_path else STANDARD_PROMPT_DICT
    prompt_dict2 = CHAT_PROMPT_DICT2 if "chat" in model_args.model_name_or_path else STANDARD_PROMPT_DICT2
    for prompt_template in [prompt_dict, prompt_dict2]:
        for k, v in prompt_dict.items():
            prompt_template[k] = prompt_template[k].replace("{prompt_with_input}", data_args.prompt_with_input)
            prompt_template[k] = prompt_template[k].replace("{prompt_without_input}", data_args.prompt_without_input)
    logger.info(prompt_dict)
    logger.info(prompt_dict2)

    # Preprocessing the datasets (instruction mode only).
    if data_mode == "instruction":
        for key in raw_datasets:
            print(key)
            prompts = create_prompts(raw_datasets[key], prompt_dict)
            system_prompts = create_system_turn(raw_datasets[key], prompt_dict2)
            columns_to_be_removed = list(raw_datasets[key].features.keys())
            raw_datasets[key] = raw_datasets[key].add_column("prompts", prompts)
            raw_datasets[key] = raw_datasets[key].add_column("system_prompts", system_prompts)
            raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)

        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
    tokenizer.padding_side = "right"  # Allow batched inference

    def tokenize(prompt, system_prompt, add_eos_token=True):
        results = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        results_system = tokenizer(
            system_prompt,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        for i in range(len(results["input_ids"])):
            if (results["input_ids"][i][-1] != tokenizer.eos_token_id and
                    len(results["input_ids"][i]) < data_args.max_seq_length and add_eos_token):
                results["input_ids"][i].append(tokenizer.eos_token_id)
                results["attention_mask"][i].append(1)

        results["labels"] = copy.deepcopy(results["input_ids"])
        for i in range(len(results_system["input_ids"])):
            for j in range(len(results_system["input_ids"][i])):
                results["labels"][i][j] = -100

        return results

    def preprocess_function(examples):
        return tokenize(examples["prompts"], examples["system_prompts"])

    if data_mode == "instruction":
        with training_args.main_process_first(desc="dataset map pre-processing"):
            tokenized_datasets = raw_datasets.map(
                preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))

    if training_args.do_eval:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_eval requires a test dataset")
        eval_dataset = tokenized_datasets["test"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))

    # Data collator for randomly masking the tokens.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    logger.info("Using data collator of type {}".format(data_collator.__class__.__name__))

    # Load model
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
        )
        if data_mode == "instruction":
            model.generation_config.pad_token_id = 0
            model.generation_config.bos_token_id = 1
            model.generation_config.eos_token_id = 2
    else:
        raise ValueError("Must provide model_name_or_path to load a pretrained CausalLM model.")

    if training_args.do_train:
        if finetune_args.use_lora:
            # PEFT settings
            lora_config_dict = {
                "r": finetune_args.lora_rank,
                "lora_alpha": finetune_args.lora_alpha,
                "lora_dropout": finetune_args.lora_dropout,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
            }

            finetune_args.lora_target_modules = [x for x in finetune_args.lora_target_modules if x]
            if len(finetune_args.lora_target_modules) > 0:
                target_modules = []
                for tm in finetune_args.lora_target_modules:
                    target_modules += tm.split()
                lora_config_dict["target_modules"] = target_modules
            logger.info(lora_config_dict)
            peft_config = LoraConfig(**lora_config_dict)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Initialize Trainer
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            data_collator=data_collator,
        )
        # `tokenizer=` became `processing_class=` in transformers 4.46.
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Trainer(**trainer_kwargs)
        trainer.add_callback(CustomPrinterCallback)

        profiler = None
        if not profile_args.disable_transfer_profile:
            profiler = TransferProfileCallback(trainer, profile_args.profile_warmup_steps)
            trainer.add_callback(profiler)

        training_start = datetime.now()
        logger.info("Elapsed time: {}".format(str(training_start - start_time)))
        logger.info("Training start time: {}".format(str(training_start)))

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        training_end = datetime.now()
        logger.info("Total training time: {}".format(str(training_end - training_start)))
        logger.info("Training end time: {}".format(str(training_end)))

        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Get and log training metrics
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Transfer profile → transfer_profile.json (after the save so checkpoint
        # bytes are counted); key fractions are mirrored into all_results.json.
        if profiler is not None and trainer.is_world_process_zero():
            profile = profiler.finalize(training_args.output_dir, data_source)
            for key in ("allreduce_fraction", "dataload_fraction", "compute_fraction",
                        "grad_size_bytes", "total_allreduce_bytes", "nfs_read_bytes",
                        "allreduce_bandwidth_MBs", "nfs_read_bandwidth_MBs", "hook_coverage"):
                if profile[key] is not None:  # log_metrics cannot format None
                    train_metrics[key] = profile[key]
            logger.info("Transfer profile: " + json.dumps(profile, indent=2))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()

        # Get and log evaluation metrics
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        eval_metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()