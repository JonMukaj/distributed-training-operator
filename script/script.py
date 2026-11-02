"""Instrumented distributed-training script for the data-transfer analysis.

Measures, per run, where training time and bytes go across two transfer
boundaries — gradient AllReduce (network) and dataset loading from NFS
(storage) — and writes two files to the output directory:

  - transfer_profile.json : full time/size/bandwidth breakdown
  - all_results.json      : the subset the operator's collector already reads

Two ways to launch it:

  1. Directly (hand-written torchrun command):
       torchrun ... script.py --synthetic --total_samples 6400 --batch_size 4
       torchrun ... script.py --data_path /mnt/output/wikitext103-tok --total_samples 6400

  2. Through the operator's default torchrun builder, which emits HuggingFace
     Trainer-style flags. Those are accepted as aliases and every flag the
     script does not know is ignored (parse_known_args), so a plain
     DistributedTraining CR with spec.model / spec.dataset / spec.training can
     drive this script and still populate the operator's config hash:
       --model_name_or_path=…           → --model_name
       --dataset_name=synthetic         → --synthetic
       --dataset_name=/path/on/pvc      → --data_path (datasets.load_from_disk)
       --dataset_name=wikitext          → HF dataset name (tokenised on the fly)
       --per_device_train_batch_size=N  → --batch_size
       --learning_rate=X                → --lr
       --max_steps=N (>0)               → --steps
       --num_train_epochs=N             → full passes when nothing else fixes the work
       --output_dir=DIR                 → --output_dir
       --save_strategy=no               → do not write a checkpoint
     Knobs the CR cannot express are read from env (set via pytorchSpec.env):
       TOTAL_SAMPLES, DATA_PATH, SYNTHETIC=1.

How much work a run does (resolved in this order):
  * --total_samples S        → steps = ceil(S / (batch × world_size)). This is
    the mode the topology solver needs: it assumes a FIXED amount of work per
    workload (W = total samples), so time shrinks as nodes are added. Use the
    same S for every node count of a series.
  * --steps / --max_steps N  → N measured steps PER RANK. Total work then
    scales with world_size; fine for per-step fraction measurements, wrong for
    time/cost predictions.
  * --num_train_epochs E     → E full passes over the dataset (real data only).
  * default                  → 100 steps per rank.

Measurement-validity notes:
  * CUDA is asynchronous, so every wall-clock timer is closed with
    torch.cuda.synchronize(); without it the fractions measure kernel-launch
    time, not execution time.
  * AllReduce time comes from the DDP comm-hook as the UNION of in-flight
    intervals: the clock runs while at least one bucket's AllReduce is
    outstanding. Several buckets are in flight at once, so summing per-bucket
    durations instead would double-count (fractions > 1 were observed). Because
    DDP overlaps communication with the backward pass, total_allreduce_s is
    still an UPPER BOUND on exclusive communication time. The plain per-bucket
    sum is reported as allreduce_bucket_sum_s for reference.
  * Byte accounting: the hook observes the bytes handed to each AllReduce
    (total_reduced_bytes). Summed over one step this must equal
    grad_size_bytes = num_params × bytes-per-element — that equality is the
    instrumentation-coverage check. total_allreduce_bytes is the ring-AllReduce
    estimate of bytes each rank puts on the wire, 2(n−1)/n × reduced bytes; it
    is derived, not measured. With world_size == 1 no hook is installed and all
    communication fields are 0.
  * The real-data path keeps the Arrow-backed dataset memory-mapped and reads
    each batch on access, so NFS traffic happens INSIDE the measured window.
    Keep total_samples ≤ dataset size for NFS runs: a second pass would be
    served from the page cache, not from NFS.
  * NFS read bytes are summed over this process AND its descendants, because the
    DataLoader worker processes (num_workers > 0) are the ones that actually
    read the files; the main process's /proc/self/io would report ~0.
  * The first --warmup_steps steps (CUDA init, cuDNN autotune, allocator growth)
    are excluded from every aggregate.
  * Reported metrics are rank 0's view.
"""

import argparse
import json
import math
import os
import threading
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset
from transformers import AutoModelForCausalLM

DEFAULT_OUTPUT_DIR = "/mnt/output/checkpoints"


# ── process-tree /proc/<pid>/io ───────────────────────────────────────────────
def _descendant_pids(root):
    """Return all descendant PIDs of `root` by scanning /proc for PPid links."""
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
                ppid = None
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
        except OSError:
            continue
        if ppid is not None:
            children.setdefault(ppid, []).append(int(entry))

    out, stack = set(), [root]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def read_tree_io():
    """Sum read_bytes/write_bytes for this process and all its descendants.

    DataLoader workers are child processes that do the actual file reads, so
    summing the process tree is what captures NFS traffic (the main process's
    /proc/self/io alone reports ~0 when num_workers > 0).
    """
    total_r = total_w = 0
    pids = {os.getpid()} | _descendant_pids(os.getpid())
    for pid in pids:
        try:
            with open(f"/proc/{pid}/io") as f:
                fields = dict(line.strip().split(": ") for line in f)
            total_r += int(fields.get("read_bytes", 0))
            total_w += int(fields.get("write_bytes", 0))
        except (OSError, ValueError):
            continue
    return total_r, total_w


# ── AllReduce comm-hook ───────────────────────────────────────────────────────
class CommState:
    def __init__(self, world_size):
        self.world_size = world_size
        # Ring AllReduce: each rank sends 2(n−1)/n of the buffer.
        self.wire_factor = 2.0 * (world_size - 1) / world_size if world_size > 1 else 0.0
        self.reset()

    def reset(self):
        self.total_s = 0.0       # union of in-flight intervals (≤ wall time)
        self.bucket_sum_s = 0.0  # plain sum of per-bucket durations (overlaps double-counted)
        self.reduced_bytes = 0   # observed: bytes handed to AllReduce
        self.wire_bytes = 0.0    # derived:  reduced_bytes × 2(n−1)/n
        self.inflight = 0
        self.span_start = 0.0
        self.lock = threading.Lock()


def allreduce_hook(state: CommState, bucket):
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
        # Runs on the completion thread, hence the lock.
        now = time.perf_counter()
        with state.lock:
            state.bucket_sum_s += now - t0
            state.inflight -= 1
            if state.inflight == 0:
                state.total_s += now - state.span_start
        return f.value()[0]  # DDP expects the reduced tensor, not the list

    return fut.then(done)


# ── Checkpoint size ───────────────────────────────────────────────────────────
def checkpoint_size_bytes(directory):
    total = 0
    if not os.path.isdir(directory):
        return 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total


# ── Datasets ──────────────────────────────────────────────────────────────────
class LazyTokenizedDataset(Dataset):
    """Row-by-row view over an Arrow-backed HF dataset.

    The HF dataset is memory-mapped, so each __getitem__ touches only the
    pages for that row — the NFS reads happen when the DataLoader workers
    fetch a batch, inside the measured window. (Materialising the columns
    into one tensor would read the whole file at startup and need ~15 GB of
    RAM per rank for WikiText-103 at seq_len 512.)
    """

    def __init__(self, hf_dataset):
        hf_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return row["input_ids"], row["attention_mask"]


def build_synthetic_dataset(args, world_size, total_iters):
    """Synthetic tokens: no NFS reads, isolates AllReduce. Enough samples so
    each rank has total_iters × batch_size after the sampler splits."""
    n = total_iters * args.batch_size * world_size
    ids = torch.randint(0, 50000, (n, args.seq_len))
    attn = torch.ones_like(ids)
    return TensorDataset(ids, attn)


def build_real_dataset(args, rank):
    """Real data (NFS). Prefer a pre-tokenised dataset on disk; fall back to
    tokenising an HF dataset (slow: every rank tokenises the whole split, and
    that time lands in the operator's wall-clock training time)."""
    if args.data_path:
        from datasets import load_from_disk

        tokenized = load_from_disk(args.data_path)
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        if rank == 0:
            print(f"WARNING: tokenising {args.dataset_name} on every rank; "
                  f"pre-tokenise once and pass --data_path for measured runs.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        name, _, config = args.dataset_name.partition(":")
        raw = load_dataset(name, config or None, split=args.dataset_split)

        def tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True,
                max_length=args.seq_len, padding="max_length",
            )

        tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)

    return LazyTokenizedDataset(tokenized)


# ── CLI ───────────────────────────────────────────────────────────────────────
def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_name", "--model_name_or_path",
                   default="EleutherAI/pythia-410m")

    # Work: see module docstring for precedence.
    p.add_argument("--total_samples", type=int, default=env_int("TOTAL_SAMPLES", 0),
                   help="Fixed total samples across all ranks (constant work for the solver). "
                        "Env: TOTAL_SAMPLES.")
    p.add_argument("--steps", "--max_steps", type=int, default=0,
                   help="Measured steps PER RANK (work scales with world_size).")
    p.add_argument("--epochs", "--num_train_epochs", type=int, default=0,
                   help="Full passes over the dataset when neither total_samples nor steps is set.")
    p.add_argument("--warmup_steps", type=int, default=5,
                   help="Initial steps excluded from all aggregates.")

    p.add_argument("--batch_size", "--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader workers. Their file reads are captured via the process tree.")

    # Data source.
    p.add_argument("--synthetic", action="store_true",
                   default=os.environ.get("SYNTHETIC", "") not in ("", "0", "false"),
                   help="Use synthetic data (no NFS reads) to isolate AllReduce. Env: SYNTHETIC=1.")
    p.add_argument("--data_path", default=os.environ.get("DATA_PATH", ""),
                   help="Pre-tokenised dataset on disk (datasets.load_from_disk). Env: DATA_PATH.")
    p.add_argument("--dataset_name", default="wikitext:wikitext-103-raw-v1",
                   help="'synthetic', a directory (→ --data_path) or an HF dataset 'name[:config]'.")
    p.add_argument("--dataset_split", default="train")

    # Output.
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save_checkpoint", action="store_true",
                   help="Save the model after training to measure checkpoint write size/bandwidth.")
    p.add_argument("--save_strategy", default="",
                   help="Operator/Trainer flag: anything other than 'no' implies --save_checkpoint.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", "--learning_rate", type=float, default=1e-4)
    p.add_argument("--ddp_backend", default="",
                   help="nccl (default on GPU) or gloo (default on CPU).")
    p.add_argument("--no_cuda", default="False")

    args, unknown = p.parse_known_args()

    # Map the operator's dataset_name onto the script's data-source switches.
    if args.dataset_name == "synthetic":
        args.synthetic = True
    elif not args.data_path and os.path.isdir(args.dataset_name):
        args.data_path = args.dataset_name
    if args.save_strategy and args.save_strategy != "no":
        args.save_checkpoint = True
    args.use_cuda = torch.cuda.is_available() and str(args.no_cuda).lower() != "true"
    if not args.ddp_backend:
        args.ddp_backend = "nccl" if args.use_cuda else "gloo"
    return args, unknown


def resolve_steps(args, world_size, dataset_len):
    per_step = args.batch_size * world_size
    if args.total_samples > 0:
        return math.ceil(args.total_samples / per_step), "total_samples"
    if args.steps > 0:
        return args.steps, "steps"
    if args.epochs > 0 and dataset_len is not None:
        return args.epochs * math.ceil(dataset_len / per_step), "epochs"
    return 100, "default"


def main():
    args, unknown = parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────────
    dist.init_process_group(backend=args.ddp_backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.use_cuda:
        # Several ranks may share one GPU in a local smoke test (gloo backend);
        # wrap the local rank onto the devices that actually exist.
        local_rank %= torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    def sync():
        if args.use_cuda:
            torch.cuda.synchronize()

    if rank == 0 and unknown:
        print(f"ignoring unknown args: {unknown}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if args.use_cuda else None)

    num_params = sum(p.numel() for p in model.parameters())
    dtype_bytes = next(model.parameters()).element_size()
    grad_size_bytes = num_params * dtype_bytes  # raw gradient size per step

    comm_state = CommState(world_size)
    if world_size > 1:
        ddp_model.register_comm_hook(comm_state, allreduce_hook)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)

    # ── Dataset + work resolution ─────────────────────────────────────────────
    if args.synthetic:
        num_steps, steps_source = resolve_steps(args, world_size, None)
        total_iters = args.warmup_steps + num_steps
        dataset = build_synthetic_dataset(args, world_size, total_iters)
    else:
        dataset = build_real_dataset(args, rank)
        num_steps, steps_source = resolve_steps(args, world_size, len(dataset))
        total_iters = args.warmup_steps + num_steps
    total_samples = args.batch_size * world_size * num_steps

    if rank == 0:
        print(f"world_size={world_size} batch={args.batch_size} steps={num_steps} "
              f"(from {steps_source}) warmup={args.warmup_steps} total_samples={total_samples} "
              f"data={'synthetic' if args.synthetic else (args.data_path or args.dataset_name)}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=args.use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    epoch = 0
    sampler.set_epoch(epoch)
    loader_iter = iter(loader)

    # ── Training loop (warmup excluded from all aggregates) ────────────────────
    compute_times = []   # forward + backward + AllReduce + optimizer (GPU-synced)
    dataload_times = []  # wait for next batch from DataLoader / NFS
    loss_sum, loss_count = 0.0, 0
    io_read_0 = io_write_0 = 0
    train_start = None

    ddp_model.train()
    for step in range(total_iters):
        recording = step >= args.warmup_steps

        # At the warmup→measured boundary, zero the counters and start the clock.
        if step == args.warmup_steps:
            sync()
            comm_state.reset()
            io_read_0, io_write_0 = read_tree_io()
            train_start = time.perf_counter()

        # --- data loading (NFS or in-memory) ---------------------------------
        t_load = time.perf_counter()
        try:
            input_ids, attn = next(loader_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            input_ids, attn = next(loader_iter)
        if recording:
            dataload_times.append(time.perf_counter() - t_load)

        # --- compute (forward + backward + AllReduce + optimizer) ------------
        input_ids = input_ids.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        labels = input_ids.clone()
        labels[attn == 0] = -100  # don't train on padding

        t_compute = time.perf_counter()
        outputs = ddp_model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = outputs.loss
        loss.backward()        # AllReduce happens inside here
        optimizer.step()
        optimizer.zero_grad()
        sync()                 # close the timer on actual GPU completion
        if recording:
            compute_times.append(time.perf_counter() - t_compute)
            loss_sum += loss.item()
            loss_count += 1

        if rank == 0 and recording and (step - args.warmup_steps) % 10 == 0:
            print(f"step {step - args.warmup_steps}/{num_steps}  loss={loss.item():.4f}")

    sync()
    total_s = time.perf_counter() - train_start

    # ── Results (rank 0 only) ─────────────────────────────────────────────────
    if rank == 0:
        out_dir = args.output_dir

        # Optional checkpoint save (before the final IO snapshot so its writes
        # are counted in nfs_write_bytes). Save time is NOT in total_s.
        if args.save_checkpoint:
            os.makedirs(out_dir, exist_ok=True)
            ddp_model.module.save_pretrained(out_dir)

        io_read_1, io_write_1 = read_tree_io()
        nfs_read_b = max(0, io_read_1 - io_read_0)
        nfs_write_b = max(0, io_write_1 - io_write_0)

        total_dataload_s = sum(dataload_times)
        total_compute_s = sum(compute_times)

        allreduce_frac = comm_state.total_s / total_s if total_s > 0 else 0.0
        dataload_frac = total_dataload_s / total_s if total_s > 0 else 0.0
        compute_frac = total_compute_s / total_s if total_s > 0 else 0.0

        samples_per_sec = total_samples / total_s if total_s > 0 else 0.0

        wire_bytes = int(round(comm_state.wire_bytes))
        allreduce_bw_MBs = (wire_bytes / 1e6) / comm_state.total_s \
            if comm_state.total_s > 0 else 0.0
        nfs_read_bw_MBs = (nfs_read_b / 1e6) / total_dataload_s \
            if total_dataload_s > 0 else 0.0
        nfs_write_bw_MBs = (nfs_write_b / 1e6) / total_s if total_s > 0 else 0.0

        avg_loss = loss_sum / loss_count if loss_count else 0.0
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        ckpt_bytes = checkpoint_size_bytes(out_dir)

        # Instrumentation-coverage check: the hook must have seen every
        # gradient byte exactly once per step.
        reduced_per_step = comm_state.reduced_bytes / num_steps if num_steps else 0
        hook_coverage = reduced_per_step / grad_size_bytes if (grad_size_bytes and world_size > 1) else None

        # ── transfer_profile.json ─────────────────────────────────────────────
        transfer = {
            # model / run info
            "model": args.model_name,
            "num_params": num_params,
            "world_size": world_size,
            "synthetic": args.synthetic,
            "data_source": "synthetic" if args.synthetic else (args.data_path or args.dataset_name),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "steps_source": steps_source,

            # time breakdown
            "total_training_s": total_s,
            "num_steps": num_steps,
            "warmup_steps": args.warmup_steps,
            "total_samples": total_samples,
            "avg_step_s": total_s / num_steps if num_steps else 0.0,
            "avg_compute_s": total_compute_s / num_steps if num_steps else 0.0,
            "avg_dataload_s": total_dataload_s / num_steps if num_steps else 0.0,
            "total_allreduce_s": comm_state.total_s,          # union of in-flight intervals
            "allreduce_bucket_sum_s": comm_state.bucket_sum_s,  # per-bucket sum, for reference
            "allreduce_fraction": allreduce_frac,
            "dataload_fraction": dataload_frac,
            "compute_fraction": compute_frac,

            # size
            "grad_size_bytes": grad_size_bytes,              # num_params × bytes/elem (per step)
            "total_reduced_bytes": comm_state.reduced_bytes,  # observed by the hook, all steps
            "hook_coverage": hook_coverage,                   # reduced/step ÷ grad_size → expect 1.0
            "total_allreduce_bytes": wire_bytes,              # ring estimate: reduced × 2(n−1)/n
            "nfs_read_bytes": nfs_read_b,
            "nfs_write_bytes": nfs_write_b,
            "checkpoint_size_bytes": ckpt_bytes,

            # bandwidth (size / time)
            "allreduce_bandwidth_MBs": allreduce_bw_MBs,
            "nfs_read_bandwidth_MBs": nfs_read_bw_MBs,
            "nfs_write_bandwidth_MBs": nfs_write_bw_MBs,

            # quality
            "avg_loss": avg_loss,
            "perplexity": perplexity,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/transfer_profile.json", "w") as f:
            json.dump(transfer, f, indent=2)

        # ── all_results.json (operator collector reads this) ──────────────────
        all_results = {
            "train_samples_per_second": samples_per_sec,
            "train_loss": avg_loss,
            "perplexity": perplexity,
            "train_runtime": total_s,
            "train_samples": total_samples,
            # transfer summary (mirrored for convenience)
            "allreduce_fraction": allreduce_frac,
            "dataload_fraction": dataload_frac,
            "compute_fraction": compute_frac,
            "grad_size_bytes": grad_size_bytes,
            "total_allreduce_bytes": wire_bytes,
            "allreduce_bandwidth_MBs": allreduce_bw_MBs,
            "nfs_read_bytes": nfs_read_b,
            "nfs_write_bytes": nfs_write_b,
            "nfs_read_bandwidth_MBs": nfs_read_bw_MBs,
            "checkpoint_size_bytes": ckpt_bytes,
        }
        with open(f"{out_dir}/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(json.dumps(transfer, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
