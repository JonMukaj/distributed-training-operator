# Running the training scripts locally

Two instrumented training scripts live in this directory. Both write
`transfer_profile.json` (where training time and bytes go: AllReduce vs data
loading vs compute) next to the `all_results.json` the operator collects.

| Script | What it is | Use it for |
|---|---|---|
| `finetune.py` | The operator's training script (HuggingFace Trainer, LoRA or full fine-tuning) with a `TransferProfileCallback` added. Writes held-out `eval_loss` and `perplexity`, and with `num_train_epochs=1` does a fixed amount of work at every node count. | **All thesis runs** (Ch 5 operator evaluation and Ch 6 transfer matrix). |
| `script.py` | A minimal standalone DDP loop with the same instrumentation. No Trainer, no LoRA, pure bf16. | Reference implementation and quick experiments outside the operator. |

This page shows how to run them on one workstation GPU to check the
instrumentation before spending cloud time. Tested on an NVIDIA RTX 4060 Laptop
(8 GB) under WSL2 with Python 3.12.

## 1. Setup

```bash
cd script
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

The `--extra-index-url` is needed for the CUDA build of torch. Keep the `<5` pins:
transformers 5 needs torch ≥ 2.6, and datasets 5 pulls a huggingface-hub release that
transformers 4 rejects. If you move the venv directory afterwards, the scripts in
`venv/bin` keep the old path in their first line; use `python -m pip` / `python -m
torch.distributed.run` or fix the shebangs.

Model sizes that fit in 8 GB: Pythia-70M for two ranks of full fine-tuning,
Pythia-160M for one rank. Pythia-410M (the thesis model) needs the cluster GPUs.

## 2. finetune.py

Common flags: `--do_train --do_eval --bf16 --report_to none --save_strategy no`.
`--profile_warmup_steps N` (default 5) excludes the first N optimizer steps from the
profile. `--disable_transfer_profile` restores the original script behaviour.

### 2a. Two ranks sharing one GPU, synthetic data, full fine-tuning (checks the hook)

NCCL refuses two ranks on one device, so use gloo. The script wraps `LOCAL_RANK`
onto the GPUs that exist.

```bash
torchrun --standalone --nproc_per_node=2 finetune.py \
  --model_name_or_path EleutherAI/pythia-70m \
  --dataset_name synthetic --max_train_samples 400 --synthetic_seq_len 256 \
  --use_lora false --per_device_train_batch_size 4 --num_train_epochs 1 \
  --validation_split_percentage 0.2 --do_train --do_eval --bf16 \
  --ddp_backend gloo --save_strategy no --report_to none \
  --logging_steps 20 --profile_warmup_steps 3 --dataloader_num_workers 2 \
  --output_dir ./out/ft1
```

Timings are not meaningful here (one GPU, gloo), the run proves the instrumentation:

| Field in `transfer_profile.json` | Expected |
|---|---|
| `hook_installed` | `true` |
| `hook_coverage` | `1.0` — the hook saw every trainable gradient byte once per step |
| `grad_size_bytes` | trainable params × 4 (fp32 master weights under bf16 autocast) |
| `total_allreduce_bytes` | `total_reduced_bytes × 2(n−1)/n`, equal to it at n = 2 |
| `allreduce_fraction` | below 1.0; `allreduce_bucket_sum_s` is larger than `total_allreduce_s` |
| `allreduce_timing` | `cuda_events` (the CUDA path; `callback` only on CPU tensors) |
| `compute_excludes_allreduce` | `true`; `compute_fraction + allreduce_fraction + dataload_fraction` is about 1 (the remainder is Trainer bookkeeping) |
| `allreduce_timing_check` | `ok`: the hook's per-step time is within 0.5 to 2 times the blocking probe (`allreduce_probe_*`, `--profile_probe_mib`, default 256) |

Local runs share one GPU between both ranks through gloo, so absolute bandwidths mean nothing
(about 1.5 GB/s here), but hook and probe must agree. The real NCCL check happens on the
cluster: on 2 nodes the hook bandwidth must be near the probe bandwidth and near the link rate.

`--dataset_name synthetic` builds random-token rows (count from `--max_train_samples`,
or `--synthetic_samples`). The 80/20 validation split still applies, so
`train_samples` in `all_results.json` is 80 % of that count.

### 2b. One rank, plain-text corpus (WikiText), full fine-tuning

A dataset with a `text` column and no `instruction` column is tokenised directly
(labels = tokens, blank lines dropped). WikiText ships train/validation/test splits,
so no split is made.

```bash
torchrun --standalone --nproc_per_node=1 finetune.py \
  --model_name_or_path EleutherAI/pythia-160m \
  --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
  --max_train_samples 400 --max_eval_samples 50 --max_seq_length 256 \
  --use_lora false --per_device_train_batch_size 4 --num_train_epochs 1 \
  --do_train --do_eval --bf16 --save_strategy no --report_to none \
  --logging_steps 20 --profile_warmup_steps 3 --dataloader_num_workers 2 \
  --output_dir ./out/ft2
```

Expect `eval_loss` around 6 and `nfs_write_bytes` ≈ `checkpoint_size_bytes`.
`nfs_read_bytes` is near 0 locally because the Arrow cache was just written and sits
in the page cache. It only becomes meaningful against NFS on the cluster.

### 2c. One rank, the original LoRA instruction-tuning path (regression check)

```bash
torchrun --standalone --nproc_per_node=1 finetune.py \
  --model_name_or_path BEE-spoke-data/smol_llama-101M-GQA \
  --dataset_name medalpaca/medical_meadow_medical_flashcards \
  --max_train_samples 64 --max_eval_samples 16 --validation_split_percentage 0.2 \
  --use_lora true --lora_rank 4 --lora_alpha 8 --lora_dropout 0.1 \
  --lora_target_modules q_proj v_proj \
  --per_device_train_batch_size 4 --num_train_epochs 1 \
  --do_train --do_eval --bf16 --save_strategy epoch --report_to none \
  --logging_steps 5 --profile_warmup_steps 2 --output_dir ./out/ft3
```

Expect `eval_loss` around 1.7 and `grad_size_bytes` of a few hundred kB: under LoRA
only the adapters are reduced, which is why the transfer experiments use
`--use_lora false`.

### 2d. Pre-tokenised data

`pretokenize.py` writes a `save_to_disk` dataset; pass its directory as
`--pretokenized_path` (or as `--dataset_name`, a directory is detected):

```bash
python pretokenize.py --model_name EleutherAI/pythia-160m \
  --dataset wikitext:wikitext-2-raw-v1 --split 'train[:4000]' \
  --seq_len 256 --drop_empty --out ./wikitext2-tok
```

## 3. script.py (standalone reference)

```bash
# single GPU, synthetic
torchrun --standalone --nproc_per_node=1 script.py --model_name EleutherAI/pythia-160m \
  --synthetic --total_samples 160 --batch_size 4 --seq_len 256 --warmup_steps 3 --output_dir ./out/run1
# two ranks on one GPU
torchrun --standalone --nproc_per_node=2 script.py --model_name EleutherAI/pythia-160m \
  --synthetic --total_samples 160 --batch_size 4 --seq_len 256 --warmup_steps 3 \
  --ddp_backend gloo --output_dir ./out/run2
# pre-tokenised data + checkpoint
torchrun --standalone --nproc_per_node=1 script.py --model_name EleutherAI/pythia-160m \
  --data_path ./wikitext2-tok --total_samples 160 --batch_size 4 --seq_len 256 \
  --warmup_steps 3 --save_checkpoint --output_dir ./out/run3
```

`--total_samples` fixes the total work; `--steps` runs per-rank steps. The operator's
Trainer-style flags are accepted as aliases, and `TOTAL_SAMPLES`, `DATA_PATH`,
`SYNTHETIC` are read from env.

## 4. Launching finetune.py through the operator

The operator's torchrun builder already targets `finetune.py`; nothing in the CR
changes for the instruction workload. For the thesis runs:

| Series | CR settings |
|---|---|
| E1 (synthetic, AllReduce only) | `dataset.name: synthetic`, `optimization.lora.enabled: false`, `training.epochs: 1`; the row count comes from `--max_train_samples`, so pass it via `pytorchSpec.command` or set `--synthetic_samples` (default 1000) |
| E2 (NFS, WikiText-103) | `dataset.name: wikitext` with config `wikitext-103-raw-v1` (needs `--dataset_config_name`, so use a `pytorchSpec.command` override or pre-tokenise to the PVC and set `dataset.name` to that directory), `lora.enabled: false` |
| Ch 5 solver runs | any of the above with identical `epochs: 1` at every node count; W = `train_samples` is then constant |

Keep `dataset.name` different between series so they get separate config hashes
and α fits. Keep `gradAccumulationSteps: 1` and `saveStrategy: epoch` or `no`:
logging and in-training checkpoints land in the dataload gap of the profile.

The GPU image must carry a CUDA torch build plus transformers, datasets and peft.
The script runs on transformers ≥ 4.3x (it switches between `tokenizer=` and
`processing_class=` by inspecting the Trainer signature, and uses the `token=`
keyword rather than the removed `use_auth_token=`).

## 5. Known limits

- `total_allreduce_s` is the time at least one AllReduce is in flight: an upper
  bound on exclusive communication time because DDP overlaps it with backward.
- `total_allreduce_bytes` is the ring-AllReduce estimate, not a wire measurement;
  `hook_coverage` is the real instrumentation check.
- Dataload time is measured inside the Trainer's batch fetching (`get_batch_samples`,
  transformers ≥ 4.46), so it is exact for any `gradient_accumulation_steps` and excludes
  logging/checkpointing; `dataloader_timed` in the profile says which method was used.
- NFS byte counters cover rank 0's process tree only.
