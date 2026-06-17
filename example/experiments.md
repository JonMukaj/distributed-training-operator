# Baseline Experiment Plan — Distributed Job Operator vs Manual Workflow

## Purpose

This document defines the complete experiment plan for evaluating the `distributed-training-operator`
(Praktikum 2) against the manual infrastructure workflow used in Praktikum 1.

The **baseline** is the manual approach: a worker node pool provisioned via `gcloud`, a
PyTorchJob applied by hand, metrics collected via `kubectl cp`, and teardown done by
deleting the node pool. The **operator** automates all of these steps end-to-end through
a single `kubectl apply`.

The comparison is **not** about raw training quality (both approaches use identical hardware
so loss and throughput are expected to be the same). The comparison dimensions are:

| Dimension | Question |
|---|---|
| **Provisioning time** | How long does each method take to get nodes Ready? |
| **Total end-to-end time** | From "start" to "results in hand" ( provisioning + training + collection + teardown) |
| **Human effort** | How many manual commands are required? |
| **Cost** | Estimated USD per run at the same node count |
| **Solver accuracy** | Does the scaling overhead model predict time and cost correctly? |

---

## Workload (Fixed Across All Runs)

Every run (baseline and operator) uses exactly this workload so results are comparable.

| Parameter | Value |
|---|---|
| **Model** | `BEE-spoke-data/smol_llama-101M-GQA` |
| **Dataset** | `medalpaca/medical_meadow_medical_flashcards` (train split) |
| **Machine type** | `c4-highcpu-8` — $0.42/hr per node |
| **Training image** | `jmukaj/cpu-fine-tuner-hf:v1.0.0` |
| **Training script** | `/workspace/scripts/finetune.py` |
| **Batch size** | 4 per device |
| **Epochs** | 1 |
| **Learning rate** | 2e-5 |
| **Validation split** | 20% |
| **Grad accumulation steps** | 1 |
| **LoRA** | rank=4, alpha=8, dropout=0.1, targets: q/k/v/o/gate/up/down proj |
| **Mixed precision** | bf16 |
| **DDP backend** | gloo (CPU) |
| **Output PVC** | `distributed-training-output-pvc` |

> **Do not change any of these parameters between runs.** The operator's config hash is a
> SHA-256 of these fields — any change produces a different hash and the solver treats it
> as a new workload with zero history.

---

## Experiment Design

Both calibration points use **multi-node** configurations (2 and 3 nodes). This is
intentional since with two genuine distributed data points the solver fits the scaling overhead
parameter α from real parallelism measurements. The solver's prediction for O-OBJ is then
grounded in actual distributed training behaviour.

> **Solver α fitting behaviour:**
> - 1 distinct node count in history → α = 0, linear scaling assumed (no real fit possible)
> - **2 distinct node counts → closed-form α fit** ← what we use
> - 3+ node counts → golden-section search (more data = better, future work)

### Run Matrix — 8 runs total

| Run ID | Approach | Nodes | Mode | Purpose |
|---|---|---|---|---|
| **B-2** | Baseline (Manual) | 2 | Fixed | Manual workflow, 2-node distributed |
| **B-3** | Baseline (Manual) | 3 | Fixed | Manual workflow, 3-node distributed |
| **O-CAL-2** | Operator | 2 | Fixed topology | Calibration — writes 2-node history entry |
| **O-CAL-3** | Operator | 3 | Fixed topology | Calibration — writes 3-node history entry |
| **O-OBJ-A1** | Operator | Solver picks | Objective A (time-aggressive) | First objective run — 2-point closed-form α, writes 3rd history entry |
| **O-OBJ-A2** | Operator | Solver picks | Objective A (same as A1) | Second run at same objective — golden-section path unlocks once n_distinct ≥ 3 |
| **O-OBJ-A3** | Operator | Solver picks | Objective A (same as A1) | Third run at same objective — α and prediction further tighten |
| **O-OBJ-B** | Operator | Solver picks | Objective B (cost-bound) | Different objective — demonstrates solver adapts node count to SLA |

**Execution order:** B-2 → B-3 → O-CAL-2 → O-CAL-3 → O-OBJ-A1 → O-OBJ-A2 → O-OBJ-A3 → O-OBJ-B.

The objective runs must come after the calibration runs since they need both
`DistributedTrainingHistory` CRs to fit α before they can provision. Within the objective
runs, A1 → A2 → A3 → B is mandatory: each later run reads the history entries written
by the earlier ones, and the "solver improves with data" narrative only makes sense
in that order.

**Narrative this matrix supports:**

| Story | Evidence |
|---|---|
| Solver improves with data | A1 → A2 → A3 prediction-vs-actual at the same objective. Either α tightens (if A1 picks a new N) or throughput at the same N denoises across 3 measurements. |
| Solver adapts to SLA | B picks a different N than A1–A3 from the same workload + same α — something the baseline cannot formulate. |
| Variance at a single N | If A1–A3 all pick the same N, the 3 measurements give the only variance number in the study — directly answers "is the 30 s operator advantage noise?" |

**Critical setup step before O-OBJ-A1** — engineer Objective A's `targetTime` /
`maxCost` so the solver is structurally **forced** to pick N ∉ {2, 3}. Otherwise A1
collapses into an existing history bucket, golden-section never unlocks, and A1–A3
become an N=3 variance measurement at a known node count rather than an improvement
story. Preview `status.resolvedTopology` during A1's Pending phase before letting
it provision — see [O-OBJ-A1](#run-o-obj-a1--operator-objective-a-time-aggressive).

---

## Metrics to Record

Record these for every run. Baseline values are measured manually; operator values are
read from Grafana or `kubectl get dj`.

| Metric | Unit | Baseline: how to measure | Operator: where to read |
|---|---|---|---|
| `provisioning_seconds` | s | wall-clock from `gcloud container node-pools create` submit → all nodes `Ready` (written by `baseline_pool.sh`) | `distributedtraining_provisioning_seconds` (Grafana panel 5) |
| `training_seconds` | s | wall-clock from `kubectl apply` PyTorchJob → PyTorchJob `Succeeded` (written by `baseline_train.sh`). Includes pod scheduling + image pull, matching the operator. | `distributedtraining_training_seconds` (Grafana panel 4) |
| `train_runtime_s` | s | secondary value: `train_runtime` from `all_results.json` (inner training loop only). Not pushed to Grafana — recorded for sanity checks. | n/a |
| `eval_runtime_s` | s | secondary value: `eval_runtime` from `all_results.json` (post-train HF evaluation loop only). Not pushed to Grafana — recorded for sanity checks; explains the gap between `train_runtime_s` and the wall-clock `training_seconds`. | n/a |
| `collection_seconds` | s | wall-clock from PyTorchJob `Succeeded` → `kubectl cp` + parse done (written by `baseline_train.sh`) | ~0 automated |
| `teardown_seconds` | s | wall-clock from collection done → node pool deleted (written by `baseline_pool.sh teardown`) | operator async, confirmed in logs |
| `total_e2e_seconds` | s | sum of all four above | `provisioning_seconds + training_seconds` (approx) |
| `nodes` | count | `--nnodes=N` in PyTorchJob args (parsed by `baseline_train.sh`) | `distributedtraining_nodes_provisioned` (Grafana panel 2) |
| `loss` | float | `eval_loss` from `all_results.json` | `distributedtraining_training_metric{metric="loss"}` |
| `perplexity` | float | `exp(eval_loss)` | `distributedtraining_training_metric{metric="perplexity"}` |
| `samples_per_second` | float | `train_samples_per_second` from `all_results.json` | `distributedtraining_training_metric{metric="samplesPerSecond"}` |
| `estimated_cost_usd` | USD | manual formula below | `distributedtraining_cost_usd_actual` (Grafana panel 3) |
| `human_steps` | count | count every command typed + every manual file edit on the happy path (see [Counting `human_steps`](#counting-human_steps)) | 1 (`kubectl apply`) |

### Cost Formula

Same formula the operator uses internally:

```
cost = nodes × 0.42 × (provisioning_seconds + training_seconds) / 3600
```

---

## Setup (do once before any run)

### Required local tools

- `gcloud` (authenticated to the GCP project)
- `kubectl` (configured to the target cluster)
- `terraform` (only needed for the initial cluster + Pushgateway provisioning, not for baseline runs)
- `python3` (used by the result-parsing step)
- `helm` (only needed if you re-provision the monitoring stack)

### GCP / cluster prerequisites

The baseline scripts assume a GKE cluster already exists with these exact names
(hardcoded in [`scripts/baseline_pool.sh`](scripts/baseline_pool.sh#L18-L21)):

| Setting | Value |
|---|---|
| Project ID | `praktikum2-494215` |
| Cluster name | `praktikum2` |
| Zone | `us-east1-d` |
| Worker pool name (managed by the script) | `praktikum2-worker-node-pool` |
| Node IAM service account | `gke-cluster-sa@praktikum2-494215.iam.gserviceaccount.com` |

> **Reproducing on a different cluster:** edit the constants at the top of
> `scripts/baseline_pool.sh` and search/replace `praktikum2-worker-node-pool`
> in the PyTorchJob YAMLs below.

Authenticate and point kubectl at the cluster:

```bash
gcloud auth login
gcloud container clusters get-credentials praktikum2 \
  --zone us-east1-d --project praktikum2-494215

# Sanity check — must report the praktikum2 cluster
kubectl config current-context
```

### Initial one-time deploy (cluster + monitoring + NFS + kubeflow + operator)

Run the following sequence once. Subsequent baseline runs only invoke the
scripts described later in this doc; nothing here needs to be re-applied
between runs.

#### Step A — Provision GKE cluster + monitoring stack via Terraform

A single `terraform apply` provisions everything in [infra/main.tf](infra/main.tf),
[infra/network.tf](infra/network.tf), [infra/nfs.tf](infra/nfs.tf), and [infra/grafana.tf](infra/grafana.tf):

- VPC + subnet
- GKE cluster `praktikum2` (us-east1-d) with general-purpose node pool
- `gke-cluster-sa` IAM service account + role bindings + Workload Identity bindings for the operator
- Persistent disk for NFS (the NFS Deployment itself comes in Step C)
- Prometheus + Grafana (kube-prometheus-stack helm chart) in namespace `monitoring`
- Grafana dashboard ConfigMap for the DistributedTraining dashboard
- Prometheus Pushgateway in `monitoring`
- `distributed-training-system` namespace + the operator's metrics `Service` + `ServiceMonitor` so Prometheus scrapes the operator once it's deployed in Step E

```bash
cd /home/jmukaj/Semester4/praktikum2/infra
terraform init
terraform apply -var-file=operator-cpu-small-kubeflow.tfvars
```

Expected duration: ~10-15 min (the GKE cluster create dominates).

#### Step B — Point kubectl at the cluster

```bash
gcloud auth login
gcloud container clusters get-credentials praktikum2 \
  --zone us-east1-d --project praktikum2-494215

# Sanity check — must report the praktikum2 cluster
kubectl config current-context
kubectl get nodes
```

#### Step C — Deploy the NFS server and shared PVC

The persistent disk was created in Step A. Two manifests are needed here:

1. [nfs.yaml](nfs.yaml) — NFS server `Deployment` + `Service` (backed by the
   `nfs-disk` GCE disk created in Step A).
2. [operator/config/samples/pvc.yaml](operator/config/samples/pvc.yaml) — the
   `PersistentVolume` pointing at the NFS service plus the
   `distributed-training-output-pvc` `PersistentVolumeClaim` that every training
   pod mounts at `/mnt/output`.

Apply in order (the PVC's PV references the NFS service, so the NFS server
must exist first):

```bash
kubectl apply -f nfs.yaml
kubectl wait --for=condition=Ready pod -l role=nfs-server --timeout=2m

kubectl apply -f operator/config/samples/pvc.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/distributed-training-output-pvc --timeout=1m
```

#### Step D — Install Kubeflow training-operator (provides the PyTorchJob CRD)

Both the baseline and operator paths apply `PyTorchJob` resources, so the
Kubeflow training-operator must be running before any B-* or O-* run. Install
the standalone overlay directly from upstream (operator + CRDs in the
`kubeflow` namespace, no other Kubeflow components):

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"

# Verify
kubectl get crd pytorchjobs.kubeflow.org
kubectl wait --for=condition=Available deploy/training-operator -n kubeflow --timeout=2m
```

#### Step E — Deploy the distributed-training-operator (only needed before O-* runs)

The operator's GKE overlay at [operator/config/overlays/gke/manager_args_patch.yaml](operator/config/overlays/gke/manager_args_patch.yaml#L21-L35)
is already pinned to `--gcp-project=praktikum2-494215`, `--gcp-location=us-east1-d`,
`--cluster-name=praktikum2`, and `--node-service-account=gke-cluster-sa@…`,
so no edits are needed for this project.

```bash
cd operator
IMG=jmukaj/distributed-training-operator:latest make gke-build-deploy
cd ..

# Verify
kubectl wait --for=condition=Available \
  deploy/distributed-training-controller-manager -n distributed-training-system --timeout=2m
```

`gke-build-deploy` runs `docker-build` + `docker-push` + applies the GKE overlay
(`config/overlays/gke`), and
machine-costs ConfigMap. Use `kubectl apply -k config/overlays/gke/` instead if the
operator image hasn't changed and you only want to re-apply the overlay.

#### Step F — Warm the HuggingFace cache (mandatory before any measured run)

`training_seconds` is wall-clock from `kubectl apply` of the PyTorchJob to the
job reaching `Succeeded`, that includes the time HuggingFace spends downloading
the model and dataset on a cold cache. If you measure one run cold and the rest warm, the cold run
appears inflated by reasons unrelated to the workflow being studied.

**Rule:** every metric pushed to Grafana (B-2, B-3, O-CAL-2, O-CAL-3, O-OBJ)
must be from a warm-cache run.

The cache lives at `/exports/hf-cache` on the NFS server (same PVC, shared
across runs and persistent across node-pool teardowns). It is populated the
first time any training pod runs with this model + dataset.

**Procedure: use the first B-2 cycle as a throwaway warmup.**

```bash
# Warmup (these timings are NOT recorded)
./scripts/baseline_pool.sh provision B-2 2
# Save results/B-2-pytorchjob.yaml from B-2 Step 3 below
./scripts/baseline_train.sh B-2
./scripts/baseline_pool.sh teardown B-2

# Delete pytorchjob
kubectl delete pytorchjob baseline-b-2

# Discard the cold artifacts (the B-2 files will be regenerated
# by the warm re-run)
rm -f results/B-2-timings.txt \
      results/B-2-all_results.json \
      results/B-2-metrics.txt

# Confirm the cache is populated
kubectl exec $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- ls /exports/hf-cache
```

You should see the model and dataset directories. From this point on, every
B-* and O-* run will pick up the cache automatically. Do not delete
`/exports/hf-cache` between runs — only `/exports/checkpoints` (see Step 0 of
each Run section).

#### Final verification before any run

```bash
kubectl get nodes
kubectl get pod -l role=nfs-server
kubectl get pvc distributed-training-output-pvc   # must report STATUS=Bound
kubectl get crd pytorchjobs.kubeflow.org
kubectl get deploy -n kubeflow training-operator
kubectl get deploy -n distributed-training-system distributed-training-controller-manager   # only needed for O-* runs
kubectl get svc -n monitoring pushgateway-prometheus-pushgateway
```

If `distributed-training-output-pvc` is missing or `Pending`, pods will fail to
schedule with `persistentvolumeclaim "distributed-training-output-pvc" not found`.
Re-apply Step C.

### Workspace setup

```bash
cd /home/jmukaj/Semester4/praktikum2
mkdir -p results
```

### Scripts you'll use

| Script | Purpose |
|---|---|
| `scripts/baseline_pool.sh provision <run> <nodes>` | Create the worker node pool, wait for nodes Ready, record `provisioning_s` |
| `scripts/baseline_train.sh <run>` | Apply the PyTorchJob, wait for Succeeded, kubectl cp results, record `training_s` + `collection_s` |
| `scripts/push_baseline.sh ...` | Push metrics to Pushgateway so the run shows up in Grafana |
| `scripts/baseline_pool.sh teardown <run>` | Delete the worker node pool, record `teardown_s` |

---

## Run B-2 — Baseline, 2 Nodes

### Step 0 — Clean leftover checkpoints (keep hf-cache)

A previous run's `all_results.json` left on the PVC would silently overwrite
this run's metrics if the new training fails before writing the file. Clear
only `/exports/checkpoints`; never touch `/exports/hf-cache` (see Setup
Step F).

```bash
kubectl exec $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints
```

### Step 1 — Provision node pool

The worker pool is provisioned via `gcloud` (the `google_container_node_pool.worker`
resource in [infra/main.tf:147-195](infra/main.tf#L147-L195) is kept commented during baseline runs).

The script creates the pool, polls until both nodes show `Ready`, and writes
`t_provision_start`, `t_provision_end`, and `provisioning_s` into `results/B-2-timings.txt`:

```bash
cd /home/jmukaj/Semester4/praktikum2
./scripts/baseline_pool.sh provision B-2 2
```

### Step 2 — (folded into Step 1)

### Step 3 — Save the PyTorchJob YAML

Save the following as `results/B-2-pytorchjob.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: baseline-b-2
  namespace: default
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-nodepool: praktikum2-worker-node-pool
          tolerations:
            - key: reserved-pool
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: dshm
              emptyDir: {medium: Memory}
            - name: output
              persistentVolumeClaim: {claimName: distributed-training-output-pvc}
          containers:
            - name: pytorch
              image: jmukaj/cpu-fine-tuner-hf:v1.0.0
              imagePullPolicy: Always
              command: ["torchrun"]
              args:
                - --nproc_per_node=1
                - --nnodes=2
                - --node_rank=$(RANK)
                - --master_addr=$(MASTER_ADDR)
                - --master_port=23456
                - /workspace/scripts/finetune.py
                - --model_name_or_path=BEE-spoke-data/smol_llama-101M-GQA
                - --dataset_name=medalpaca/medical_meadow_medical_flashcards
                - --dataset_split=train
                - --dataset_cache_directory=/mnt/output/hf-cache/datasets
                - --output_dir=/mnt/output/checkpoints
                - --per_device_train_batch_size=4
                - --per_device_eval_batch_size=4
                - --learning_rate=2e-5
                - --num_train_epochs=1
                - --max_steps=-1
                - --gradient_accumulation_steps=1
                - --max_grad_norm=1.0
                - --validation_split_percentage=0.20
                - --warmup_steps=50
                - --logging_steps=10
                - --save_total_limit=2
                - --save_strategy=epoch
                - --overwrite_output_dir=True
                - --do_train=True
                - --do_eval=True
                - --bf16=True
                - --bf16_full_eval=True
                - --use_lora=True
                - --lora_rank=4
                - --lora_alpha=8
                - --lora_dropout=0.1
                - --lora_target_modules=q_proj
                - --lora_target_modules=v_proj
                - --lora_target_modules=k_proj
                - --lora_target_modules=o_proj
                - --lora_target_modules=gate_proj
                - --lora_target_modules=up_proj
                - --lora_target_modules=down_proj
                - --no_cuda=True
                - --ddp_backend=gloo
                - --ddp_find_unused_parameters=False
                - --use_fast_tokenizer=False
                - "--prompt_with_input=Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                - "--prompt_without_input=Below is an instruction that describes a task. Write a response that appropriately completes the request."
              env:
                - {name: TRANSFORMERS_CACHE,   value: /mnt/output/hf-cache}
                - {name: HF_DATASETS_CACHE,    value: /mnt/output/hf-cache/datasets}
                - {name: HF_HOME,              value: /mnt/output/hf-cache}
                - {name: HF_HUB_DISABLE_XET,   value: "1"}
                - {name: LOGLEVEL,             value: INFO}
                - {name: LD_PRELOAD,           value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"}
                - {name: CCL_WORKER_COUNT,     value: "1"}
              resources:
                requests: {cpu: "4", memory: "10Gi"}
                limits:   {cpu: "4", memory: "10Gi"}
              volumeMounts:
                - {name: dshm,   mountPath: /dev/shm}
                - {name: output, mountPath: /mnt/output}
              securityContext: {allowPrivilegeEscalation: false}
    Worker:
      replicas: 1                 # master=1 + worker=1 = 2 total nodes
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-nodepool: praktikum2-worker-node-pool
          tolerations:
            - key: reserved-pool
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: dshm
              emptyDir: {medium: Memory}
            - name: output
              persistentVolumeClaim: {claimName: distributed-training-output-pvc}
          containers:
            - name: pytorch
              image: jmukaj/cpu-fine-tuner-hf:v1.0.0
              imagePullPolicy: Always
              command: ["torchrun"]
              args:
                - --nproc_per_node=1
                - --nnodes=2
                - --node_rank=$(RANK)
                - --master_addr=$(MASTER_ADDR)
                - --master_port=23456
                - /workspace/scripts/finetune.py
                - --model_name_or_path=BEE-spoke-data/smol_llama-101M-GQA
                - --dataset_name=medalpaca/medical_meadow_medical_flashcards
                - --dataset_split=train
                - --dataset_cache_directory=/mnt/output/hf-cache/datasets
                - --output_dir=/mnt/output/checkpoints
                - --per_device_train_batch_size=4
                - --per_device_eval_batch_size=4
                - --learning_rate=2e-5
                - --num_train_epochs=1
                - --max_steps=-1
                - --gradient_accumulation_steps=1
                - --max_grad_norm=1.0
                - --validation_split_percentage=0.20
                - --warmup_steps=50
                - --logging_steps=10
                - --save_total_limit=2
                - --save_strategy=epoch
                - --overwrite_output_dir=True
                - --do_train=True
                - --do_eval=True
                - --bf16=True
                - --bf16_full_eval=True
                - --use_lora=True
                - --lora_rank=4
                - --lora_alpha=8
                - --lora_dropout=0.1
                - --lora_target_modules=q_proj
                - --lora_target_modules=v_proj
                - --lora_target_modules=k_proj
                - --lora_target_modules=o_proj
                - --lora_target_modules=gate_proj
                - --lora_target_modules=up_proj
                - --lora_target_modules=down_proj
                - --no_cuda=True
                - --ddp_backend=gloo
                - --ddp_find_unused_parameters=False
                - --use_fast_tokenizer=False
                - "--prompt_with_input=Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                - "--prompt_without_input=Below is an instruction that describes a task. Write a response that appropriately completes the request."
              env:
                - {name: TRANSFORMERS_CACHE,   value: /mnt/output/hf-cache}
                - {name: HF_DATASETS_CACHE,    value: /mnt/output/hf-cache/datasets}
                - {name: HF_HOME,              value: /mnt/output/hf-cache}
                - {name: HF_HUB_DISABLE_XET,   value: "1"}
                - {name: LOGLEVEL,             value: INFO}
                - {name: LD_PRELOAD,           value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"}
                - {name: CCL_WORKER_COUNT,     value: "1"}
              resources:
                requests: {cpu: "4", memory: "10Gi"}
                limits:   {cpu: "4", memory: "10Gi"}
              volumeMounts:
                - {name: dshm,   mountPath: /dev/shm}
                - {name: output, mountPath: /mnt/output}
              securityContext: {allowPrivilegeEscalation: false}
```

### Step 4 — Run training + collection

The script applies the YAML, waits for the master pod to appear, waits for the
PyTorchJob to reach `Succeeded`, copies `all_results.json` from the master pod,
parses metrics, and appends `t_training_*`, `training_s`, `t_collection_*`,
`collection_s`, and `train_runtime_s` to `results/B-2-timings.txt`. It prints
a ready-to-paste `push_baseline.sh` command when finished.

```bash
./scripts/baseline_train.sh B-2
```

### Step 5 — Push metrics to Grafana

Run the `push_baseline.sh` command that Step 4 printed (values are already
substituted). The command template is:

```bash
./scripts/push_baseline.sh \
  --run    baseline-b-2 \
  --nodes  2 \
  --prov   <provisioning_s   from results/B-2-timings.txt> \
  --train  <training_s       from results/B-2-timings.txt> \
  --loss   <loss             from results/B-2-metrics.txt> \
  --ppl    <perplexity       from results/B-2-metrics.txt> \
  --sps    <samplesPerSecond from results/B-2-metrics.txt> \
  --namespace default
```

You can run this any time, from any machine with `kubectl` access — it's
decoupled from the training run. Useful for re-publishing months later on a
fresh monitoring stack: keep `results/B-2-timings.txt` and `results/B-2-metrics.txt`
under version control and re-run with the same values.

After ~30 s, open Grafana and select `dj_name = baseline-b-2`.

### Step 6 — Tear down node pool

```bash
kubectl delete pytorchjob baseline-b-2
./scripts/baseline_pool.sh teardown B-2
```

This writes `t_teardown_start`, `t_teardown_end`, and `teardown_s` into
`results/B-2-timings.txt`.

### Step 7 — Fill in the manual fields of `results/B-2-timings.txt`

The script-written fields (`t_provision_*`, `provisioning_s`, `t_training_*`,
`training_s`, `t_collection_*`, `collection_s`, `train_runtime_s`,
`t_teardown_*`, `teardown_s`) are already present. Compute and append:

```
total_e2e_s        = provisioning_s + training_s + collection_s + teardown_s
human_steps        = <count of commands you typed>
```

---

## Run B-3 — Baseline, 3 Nodes

### Step 0 — Clean leftover checkpoints (keep hf-cache)

```bash
kubectl exec $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints
```

### Step 1 — Provision node pool

```bash
cd /home/jmukaj/Semester4/praktikum2
./scripts/baseline_pool.sh provision B-3 3
```

### Step 2 — (folded into Step 1)

### Step 3 — Save the PyTorchJob YAML

Save as `results/B-3-pytorchjob.yaml` — identical to B-2 except:
- `metadata.name: baseline-b-3`
- `--nnodes=3` in both Master and Worker args
- `Worker.replicas: 2` (master=1 + worker=2 = 3 total)

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: baseline-b-3
  namespace: default
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-nodepool: praktikum2-worker-node-pool
          tolerations:
            - key: reserved-pool
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: dshm
              emptyDir: {medium: Memory}
            - name: output
              persistentVolumeClaim: {claimName: distributed-training-output-pvc}
          containers:
            - name: pytorch
              image: jmukaj/cpu-fine-tuner-hf:v1.0.0
              imagePullPolicy: Always
              command: ["torchrun"]
              args:
                - --nproc_per_node=1
                - --nnodes=3
                - --node_rank=$(RANK)
                - --master_addr=$(MASTER_ADDR)
                - --master_port=23456
                - /workspace/scripts/finetune.py
                - --model_name_or_path=BEE-spoke-data/smol_llama-101M-GQA
                - --dataset_name=medalpaca/medical_meadow_medical_flashcards
                - --dataset_split=train
                - --dataset_cache_directory=/mnt/output/hf-cache/datasets
                - --output_dir=/mnt/output/checkpoints
                - --per_device_train_batch_size=4
                - --per_device_eval_batch_size=4
                - --learning_rate=2e-5
                - --num_train_epochs=1
                - --max_steps=-1
                - --gradient_accumulation_steps=1
                - --max_grad_norm=1.0
                - --validation_split_percentage=0.20
                - --warmup_steps=50
                - --logging_steps=10
                - --save_total_limit=2
                - --save_strategy=epoch
                - --overwrite_output_dir=True
                - --do_train=True
                - --do_eval=True
                - --bf16=True
                - --bf16_full_eval=True
                - --use_lora=True
                - --lora_rank=4
                - --lora_alpha=8
                - --lora_dropout=0.1
                - --lora_target_modules=q_proj
                - --lora_target_modules=v_proj
                - --lora_target_modules=k_proj
                - --lora_target_modules=o_proj
                - --lora_target_modules=gate_proj
                - --lora_target_modules=up_proj
                - --lora_target_modules=down_proj
                - --no_cuda=True
                - --ddp_backend=gloo
                - --ddp_find_unused_parameters=False
                - --use_fast_tokenizer=False
                - "--prompt_with_input=Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                - "--prompt_without_input=Below is an instruction that describes a task. Write a response that appropriately completes the request."
              env:
                - {name: TRANSFORMERS_CACHE,   value: /mnt/output/hf-cache}
                - {name: HF_DATASETS_CACHE,    value: /mnt/output/hf-cache/datasets}
                - {name: HF_HOME,              value: /mnt/output/hf-cache}
                - {name: HF_HUB_DISABLE_XET,   value: "1"}
                - {name: LOGLEVEL,             value: INFO}
                - {name: LD_PRELOAD,           value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"}
                - {name: CCL_WORKER_COUNT,     value: "1"}
              resources:
                requests: {cpu: "4", memory: "10Gi"}
                limits:   {cpu: "4", memory: "10Gi"}
              volumeMounts:
                - {name: dshm,   mountPath: /dev/shm}
                - {name: output, mountPath: /mnt/output}
              securityContext: {allowPrivilegeEscalation: false}
    Worker:
      replicas: 2                 # master=1 + worker=2 = 3 total nodes
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-nodepool: praktikum2-worker-node-pool
          tolerations:
            - key: reserved-pool
              operator: Equal
              value: "true"
              effect: NoSchedule
          volumes:
            - name: dshm
              emptyDir: {medium: Memory}
            - name: output
              persistentVolumeClaim: {claimName: distributed-training-output-pvc}
          containers:
            - name: pytorch
              image: jmukaj/cpu-fine-tuner-hf:v1.0.0
              imagePullPolicy: Always
              command: ["torchrun"]
              args:
                - --nproc_per_node=1
                - --nnodes=3
                - --node_rank=$(RANK)
                - --master_addr=$(MASTER_ADDR)
                - --master_port=23456
                - /workspace/scripts/finetune.py
                - --model_name_or_path=BEE-spoke-data/smol_llama-101M-GQA
                - --dataset_name=medalpaca/medical_meadow_medical_flashcards
                - --dataset_split=train
                - --dataset_cache_directory=/mnt/output/hf-cache/datasets
                - --output_dir=/mnt/output/checkpoints
                - --per_device_train_batch_size=4
                - --per_device_eval_batch_size=4
                - --learning_rate=2e-5
                - --num_train_epochs=1
                - --max_steps=-1
                - --gradient_accumulation_steps=1
                - --max_grad_norm=1.0
                - --validation_split_percentage=0.20
                - --warmup_steps=50
                - --logging_steps=10
                - --save_total_limit=2
                - --save_strategy=epoch
                - --overwrite_output_dir=True
                - --do_train=True
                - --do_eval=True
                - --bf16=True
                - --bf16_full_eval=True
                - --use_lora=True
                - --lora_rank=4
                - --lora_alpha=8
                - --lora_dropout=0.1
                - --lora_target_modules=q_proj
                - --lora_target_modules=v_proj
                - --lora_target_modules=k_proj
                - --lora_target_modules=o_proj
                - --lora_target_modules=gate_proj
                - --lora_target_modules=up_proj
                - --lora_target_modules=down_proj
                - --no_cuda=True
                - --ddp_backend=gloo
                - --ddp_find_unused_parameters=False
                - --use_fast_tokenizer=False
                - "--prompt_with_input=Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                - "--prompt_without_input=Below is an instruction that describes a task. Write a response that appropriately completes the request."
              env:
                - {name: TRANSFORMERS_CACHE,   value: /mnt/output/hf-cache}
                - {name: HF_DATASETS_CACHE,    value: /mnt/output/hf-cache/datasets}
                - {name: HF_HOME,              value: /mnt/output/hf-cache}
                - {name: HF_HUB_DISABLE_XET,   value: "1"}
                - {name: LOGLEVEL,             value: INFO}
                - {name: LD_PRELOAD,           value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"}
                - {name: CCL_WORKER_COUNT,     value: "1"}
              resources:
                requests: {cpu: "4", memory: "10Gi"}
                limits:   {cpu: "4", memory: "10Gi"}
              volumeMounts:
                - {name: dshm,   mountPath: /dev/shm}
                - {name: output, mountPath: /mnt/output}
              securityContext: {allowPrivilegeEscalation: false}
```

### Step 4 — Run training + collection

```bash
./scripts/baseline_train.sh B-3
```

### Step 5 — Push metrics to Grafana

Run the `push_baseline.sh` command that Step 4 printed. Template:

```bash
./scripts/push_baseline.sh \
  --run    baseline-b-3 \
  --nodes  3 \
  --prov   <provisioning_s   from results/B-3-timings.txt> \
  --train  <training_s       from results/B-3-timings.txt> \
  --loss   <loss             from results/B-3-metrics.txt> \
  --ppl    <perplexity       from results/B-3-metrics.txt> \
  --sps    <samplesPerSecond from results/B-3-metrics.txt> \
  --namespace default
```

(See B-2 Step 5 for notes on re-publishing later from saved files.)

### Step 6 — Tear down node pool

```bash
kubectl delete pytorchjob baseline-b-3
./scripts/baseline_pool.sh teardown B-3
```

### Step 7 — Fill in the manual fields of `results/B-3-timings.txt`

Same as B-2: append `total_e2e_s` and `human_steps`.

---

## Run O-CAL-2 — Operator, 2 Nodes (Calibration)

### Prerequisites (once before first operator run)

```bash
kubectl get deploy -n distributed-training-system          # operator running
kubectl get deploy -n kubeflow training-operator       # kubeflow running
kubectl get pvc distributed-training-output-pvc             # PVC exists

# Clean PVC checkpoints so results are not mixed with baseline runs
kubectl exec -it $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints
```

### CR to apply

Save as `results/O-CAL-2-distributedtraining.yaml`:

```yaml
apiVersion: training.distributedtraining.io/v1
kind: DistributedTraining
metadata:
  name: distributedtraining-cal-2node
  namespace: default
spec:
  backend: pytorch
  hardware:
    type: cpu
    machineType: "c4-highcpu-8"
  topology:
    nodes: 2
    processesPerNode: 1
  outputPVCName: distributed-training-output-pvc
  model:
    name: BEE-spoke-data/smol_llama-101M-GQA
    trainingScript: /workspace/scripts/finetune.py
  dataset:
    name: medalpaca/medical_meadow_medical_flashcards
    split: train
    promptWithInput: "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    promptWithoutInput: "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  training:
    batchSize: 4
    learningRate: "2e-5"
    epochs: 1
    maxSteps: -1
    gradAccumulationSteps: 1
    maxGradNorm: "1.0"
    validationSplit: "0.20"
    warmupSteps: 50
    loggingSteps: 10
    saveTotalLimit: 2
    saveStrategy: epoch
    overwriteOutputDir: true
    ddpBackend: gloo
    ddpFindUnusedParameters: false
    useFastTokenizer: false
  optimization:
    mixedPrecision: bf16
    bf16FullEval: true
    lora:
      enabled: true
      rank: 4
      alpha: 8
      dropout: "0.1"
      targetModules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
  pytorchSpec:
    image: jmukaj/cpu-fine-tuner-hf:v1.0.0
    resources:
      requests: {cpu: "4", memory: "10Gi"}
      limits:   {cpu: "4", memory: "10Gi"}
    env:
      - name: TRANSFORMERS_CACHE
        value: /mnt/output/hf-cache
      - name: HF_DATASETS_CACHE
        value: /mnt/output/hf-cache/datasets
      - name: HF_HOME
        value: /mnt/output/hf-cache
      - name: HF_HUB_DISABLE_XET
        value: "1"
      - name: LOGLEVEL
        value: INFO
      - name: LD_PRELOAD
        value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
      - name: CCL_WORKER_COUNT
        value: "1"
```

### Apply and watch

```bash
kubectl apply -f results/O-CAL-2-distributedtraining.yaml

# Watch phase transitions: Pending→Provisioning→Ready→Running→Collecting→Succeeded
kubectl get dj distributedtraining-cal-2node -w

# Detailed operator logs
kubectl logs -f -n distributed-training-system deploy/distributed-training-controller-manager
```

### Capture results

```bash
kubectl get dj distributedtraining-cal-2node -o yaml > results/O-CAL-2-status.yaml

# Verify DistributedTrainingHistory was written
kubectl get distributedtraininghistory

# Capture train_runtime + eval_runtime (sanity-check fields not in CR status).
# eval_runtime explains the gap between train_runtime and the wall-clock
# distributedtraining_training_seconds.
NFS_POD=$(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}')
kubectl cp default/${NFS_POD}:/exports/checkpoints/all_results.json \
  results/O-CAL-2-all_results.json
python3 -c "
import json
r = json.load(open('results/O-CAL-2-all_results.json'))
print(f\"train_runtime_s = {r.get('train_runtime', 0):.1f}\")
print(f\"eval_runtime_s  = {r.get('eval_runtime',  0):.1f}\")
" | tee results/O-CAL-2-runtime.txt
```

All timing metrics are in Grafana — select `dj_name = distributedtraining-cal-2node`.

---

## Run O-CAL-3 — Operator, 3 Nodes (Calibration)

### CR to apply

Save as `results/O-CAL-3-distributedtraining.yaml` — identical to O-CAL-2 except:

```yaml
metadata:
  name: distributedtraining-cal-3node
spec:
  topology:
    nodes: 3
    processesPerNode: 1
```

### Apply and watch

```bash
# Clean PVC checkpoints before this run
kubectl exec -it $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints

kubectl apply -f results/O-CAL-3-distributedtraining.yaml
kubectl get dj distributedtraining-cal-3node -w
```

### Capture results

```bash
kubectl get dj distributedtraining-cal-3node -o yaml > results/O-CAL-3-status.yaml

# Verify two history entries now exist with the same configHash
kubectl get distributedtraininghistory -o yaml | grep -E "configHash|nodes|throughput"

# Capture train_runtime + eval_runtime (sanity-check fields not in CR status).
NFS_POD=$(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}')
kubectl cp default/${NFS_POD}:/exports/checkpoints/all_results.json \
  results/O-CAL-3-all_results.json
python3 -c "
import json
r = json.load(open('results/O-CAL-3-all_results.json'))
print(f\"train_runtime_s = {r.get('train_runtime', 0):.1f}\")
print(f\"eval_runtime_s  = {r.get('eval_runtime',  0):.1f}\")
" | tee results/O-CAL-3-runtime.txt
```

> At this point the solver has 2 distinct multi-node data points and can fit α.

---

## Run O-OBJ-A1 — Operator, Objective A (Time-Aggressive)

This is the first of three runs at the same objective. The trio A1 → A2 → A3
demonstrates the "solver improves with data" narrative: each run writes a new
history entry, and the solver re-fits α with every additional datapoint.

### Choosing Objective A

The objective must structurally force the solver to pick a node count **other than
2 or 3**. Otherwise A1's history entry collapses into an existing bucket
(`len(distinct_n)` stays at 2), the golden-section solver path never unlocks
([solver.go:295-323](operator/internal/controller/solver.go#L295)), and A1–A3
become tautological repetitions.

Default suggestion — time-aggressive: forces high N to meet a tight wall-clock target.

```yaml
objective:
  targetTime: "10m"
  maxCost: "10"
  maxNodes: 10
```

This is a starting point. If the preview step below shows the solver picked
N=2 or N=3, tighten `targetTime` further (e.g. `"3m"`) and re-apply.

### CR to apply

Save as `results/O-OBJ-A1-distributedtraining.yaml`:

```yaml
apiVersion: training.distributedtraining.io/v1
kind: DistributedTraining
metadata:
  name: distributedtraining-obj-a1
  namespace: default
spec:
  backend: pytorch
  hardware:
    type: cpu
    machineType: "c4-highcpu-8"
  objective:
    targetTime: "10m"
    maxCost: "20"
    maxNodes: 10
  outputPVCName: distributed-training-output-pvc
  model:
    name: BEE-spoke-data/smol_llama-101M-GQA
    trainingScript: /workspace/scripts/finetune.py
  dataset:
    name: medalpaca/medical_meadow_medical_flashcards
    split: train
    promptWithInput: "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    promptWithoutInput: "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  training:
    batchSize: 4
    learningRate: "2e-5"
    epochs: 1
    maxSteps: -1
    gradAccumulationSteps: 1
    maxGradNorm: "1.0"
    validationSplit: "0.20"
    warmupSteps: 50
    loggingSteps: 10
    saveTotalLimit: 2
    saveStrategy: epoch
    overwriteOutputDir: true
    ddpBackend: gloo
    ddpFindUnusedParameters: false
    useFastTokenizer: false
  optimization:
    mixedPrecision: bf16
    bf16FullEval: true
    lora:
      enabled: true
      rank: 4
      alpha: 8
      dropout: "0.1"
      targetModules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
  pytorchSpec:
    image: jmukaj/cpu-fine-tuner-hf:v1.0.0
    resources:
      requests: {cpu: "4", memory: "10Gi"}
      limits:   {cpu: "4", memory: "10Gi"}
    env:
      - name: TRANSFORMERS_CACHE
        value: /mnt/output/hf-cache
      - name: HF_DATASETS_CACHE
        value: /mnt/output/hf-cache/datasets
      - name: HF_HOME
        value: /mnt/output/hf-cache
      - name: HF_HUB_DISABLE_XET
        value: "1"
      - name: LOGLEVEL
        value: INFO
      - name: LD_PRELOAD
        value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
      - name: CCL_WORKER_COUNT
        value: "1"
```

### Apply, preview solver pick, and re-tune if it collides

```bash
# Clean PVC checkpoints
kubectl exec -it $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints

kubectl apply -f results/O-OBJ-A1-distributedtraining.yaml

# CRITICAL: read the solver's pick during Pending — BEFORE the node pool is created.
# Poll until resolvedTopology is populated.
until kubectl get dj distributedtraining-obj-a1 \
  -o jsonpath='{.status.resolvedTopology.nodes}' 2>/dev/null | grep -qE '^[0-9]+$'; do
  sleep 2
done
kubectl get dj distributedtraining-obj-a1 -o jsonpath='{.status.resolvedTopology}' | python3 -m json.tool

# If solver picked N=2 or N=3, kill the run, tighten objective, re-apply:
#   kubectl delete dj distributedtraining-obj-a1
#   <edit results/O-OBJ-A1-distributedtraining.yaml: lower targetTime>
#   kubectl apply -f results/O-OBJ-A1-distributedtraining.yaml
# Repeat until the solver picks N >= 4. Save the working CR — A2 and A3 will reuse it.

# Once happy with the pick, let the run continue and watch it through:
kubectl get dj distributedtraining-obj-a1 -w
```

`status.resolvedTopology` fields:
- `nodes` — solver-selected node count
- `estimatedTime` — predicted training duration (Go duration string)
- `estimatedCost` — predicted cost (USD string)
- `masterReplicas` / `workerReplicas` — replica split

### Capture final results

```bash
kubectl get dj distributedtraining-obj-a1 -o yaml > results/O-OBJ-A1-status.yaml

NFS_POD=$(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}')
kubectl cp default/${NFS_POD}:/exports/checkpoints/all_results.json \
  results/O-OBJ-A1-all_results.json
python3 -c "
import json
r = json.load(open('results/O-OBJ-A1-all_results.json'))
print(f\"train_runtime_s = {r.get('train_runtime', 0):.1f}\")
print(f\"eval_runtime_s  = {r.get('eval_runtime',  0):.1f}\")
" | tee results/O-OBJ-A1-runtime.txt

# Confirm a new history entry was written, and note its node count.
kubectl get distributedtraininghistory -o yaml | grep -E "configHash|nodes|throughput"
```

---

## Run O-OBJ-A2 — Operator, Objective A (Second Repeat)

Identical objective to A1. After A1 completed there is now a 3rd `DistributedTrainingHistory`
entry — if A1's pick was a new N (not 2 or 3), the solver now sees 3 distinct
node counts and the **golden-section path unlocks** instead of closed-form.

### CR to apply

Save as `results/O-OBJ-A2-distributedtraining.yaml` — copy of A1 with only the name
changed:

```yaml
# ...identical to O-OBJ-A1-distributedtraining.yaml except:
metadata:
  name: distributedtraining-obj-a2
```

(All `spec.*` fields, including `objective`, are bit-for-bit identical to A1.)

### Apply and watch

```bash
kubectl exec -it $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints

kubectl apply -f results/O-OBJ-A2-distributedtraining.yaml

# Capture solver pick during Pending — compare against A1's pick.
until kubectl get dj distributedtraining-obj-a2 \
  -o jsonpath='{.status.resolvedTopology.nodes}' 2>/dev/null | grep -qE '^[0-9]+$'; do
  sleep 2
done
kubectl get dj distributedtraining-obj-a2 -o jsonpath='{.status.resolvedTopology}' | python3 -m json.tool

kubectl get dj distributedtraining-obj-a2 -w
```

> The pick may stay the same as A1 (likely if N=5 was clearly optimal). What
> matters is whether `estimatedTime` and `estimatedCost` shift — they will if α
> changed when A1's datapoint was incorporated. Record the predicted values for
> Table 7.

### Capture final results

```bash
kubectl get dj distributedtraining-obj-a2 -o yaml > results/O-OBJ-A2-status.yaml

NFS_POD=$(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}')
kubectl cp default/${NFS_POD}:/exports/checkpoints/all_results.json \
  results/O-OBJ-A2-all_results.json
python3 -c "
import json
r = json.load(open('results/O-OBJ-A2-all_results.json'))
print(f\"train_runtime_s = {r.get('train_runtime', 0):.1f}\")
print(f\"eval_runtime_s  = {r.get('eval_runtime',  0):.1f}\")
" | tee results/O-OBJ-A2-runtime.txt
```

---

## Run O-OBJ-A3 — Operator, Objective A (Third Repeat)

Same objective again. After A2, history has 4 entries (one of which may be a
repeat of A1's N). Throughput at the repeated N is now averaged across multiple
observations, denoising α further.

### CR to apply

Save as `results/O-OBJ-A3-distributedtraining.yaml` — identical to A1 except:

```yaml
metadata:
  name: distributedtraining-obj-a3
```

### Apply and watch

```bash
kubectl exec -it $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints

kubectl apply -f results/O-OBJ-A3-distributedtraining.yaml

until kubectl get dj distributedtraining-obj-a3 \
  -o jsonpath='{.status.resolvedTopology.nodes}' 2>/dev/null | grep -qE '^[0-9]+$'; do
  sleep 2
done
kubectl get dj distributedtraining-obj-a3 -o jsonpath='{.status.resolvedTopology}' | python3 -m json.tool

kubectl get dj distributedtraining-obj-a3 -w
```

### Capture final results

```bash
kubectl get dj distributedtraining-obj-a3 -o yaml > results/O-OBJ-A3-status.yaml

NFS_POD=$(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}')
kubectl cp default/${NFS_POD}:/exports/checkpoints/all_results.json \
  results/O-OBJ-A3-all_results.json
python3 -c "
import json
r = json.load(open('results/O-OBJ-A3-all_results.json'))
print(f\"train_runtime_s = {r.get('train_runtime', 0):.1f}\")
print(f\"eval_runtime_s  = {r.get('eval_runtime',  0):.1f}\")
" | tee results/O-OBJ-A3-runtime.txt
```

At this point the solver has 5 history entries (O-CAL-2, O-CAL-3, A1, A2, A3),
α is fit from at least 3 distinct node counts via golden-section search, and the
prediction-error trend across A1 → A2 → A3 is ready to plot for Table 7.

---

## Run O-OBJ-B — Operator, Objective B (Cost-Bound)

Different objective from A. Demonstrates that the same workload + same α produces
a different node-count decision when the SLA constraint changes. This is the
"solver adapts" / "multi-SLA" differentiator that the baseline workflow cannot
formulate at all (the baseline takes a hard-coded `--nnodes=N`; there's no notion
of an objective).

### Choosing Objective B

Should contrast with A. Suggested cost-bound objective: minimize cost subject
to a loose time bound. Likely pushes the solver toward fewer nodes than A
selected.

```yaml
objective:
  targetTime: "7m"
  maxCost: "10"
  maxNodes: 15
```

Same preview-and-retune discipline as A1 applies. If B happens to pick the
same N as A1–A3, the contrast is lost and the differentiator slide weakens —
retune `maxCost` (lower for fewer nodes) or `targetTime` (looser to allow
small N) until the pick differs from A.

### CR to apply

Save as `results/O-OBJ-B-distributedtraining.yaml` — identical to A1 except for the
name and the objective block:

```yaml
apiVersion: training.distributedtraining.io/v1
kind: DistributedTraining
metadata:
  name: distributedtraining-obj-b
  namespace: default
spec:
  # ...all other fields identical to O-OBJ-A1...
  objective:
    targetTime: "7m"
    maxCost: "10"
    maxNodes: 15
```

### Apply and watch

```bash
kubectl exec -it $(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}') \
  -- rm -rf /exports/checkpoints

kubectl apply -f results/O-OBJ-B-distributedtraining.yaml

until kubectl get dj distributedtraining-obj-b \
  -o jsonpath='{.status.resolvedTopology.nodes}' 2>/dev/null | grep -qE '^[0-9]+$'; do
  sleep 2
done
kubectl get dj distributedtraining-obj-b -o jsonpath='{.status.resolvedTopology}' | python3 -m json.tool

# Compare B's pick to A's pick — record both for the "solver adapts" slide.
kubectl get dj distributedtraining-obj-b -w
```

### Capture final results

```bash
kubectl get dj distributedtraining-obj-b -o yaml > results/O-OBJ-B-status.yaml

NFS_POD=$(kubectl get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}')
kubectl cp default/${NFS_POD}:/exports/checkpoints/all_results.json \
  results/O-OBJ-B-all_results.json
python3 -c "
import json
r = json.load(open('results/O-OBJ-B-all_results.json'))
print(f\"train_runtime_s = {r.get('train_runtime', 0):.1f}\")
print(f\"eval_runtime_s  = {r.get('eval_runtime',  0):.1f}\")
" | tee results/O-OBJ-B-runtime.txt
```

## Tearing down the entire stack

When all runs are finished and you want to release the cluster (stops every
running cost — node pools, persistent disks, load balancers, etc.):

```bash
cd /home/jmukaj/Semester4/praktikum2

# 1. Undeploy the operator (optional — terraform destroy will remove the namespace anyway)
cd operator && make undeploy ignore-not-found=true; cd ..

# 2. Delete any leftover baseline worker pool (if a run was interrupted)
gcloud container node-pools delete praktikum2-worker-node-pool \
  --cluster praktikum2 --zone us-east1-d --project praktikum2-494215 --quiet || true

# 3. Destroy everything provisioned by terraform (cluster, IAM, NFS disk,
#    monitoring stack, Pushgateway, ServiceMonitor)
cd infra && terraform destroy -var-file=operator-cpu-small-kubeflow.tfvars

# 4. Sanity check — no GKE clusters should remain in the project
gcloud container clusters list --project praktikum2-494215
```

`-refresh=false` can be added to `terraform destroy` if state has drifted
(e.g. resources deleted out of band).
