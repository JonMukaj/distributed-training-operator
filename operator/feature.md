# Operator Feature Extension

## Overview

The distributed-training operator has been extended with four features that shift the system from a static, user-specified topology model toward an **objective-driven** training infrastructure. Users can now declare what they want to achieve (a time target, a budget cap, or both) and the operator estimates a suitable cluster topology automatically, refining its predictions across runs using collected history.

---

## What Was Built

### 1. Run History Persistence (`DistributedTrainingHistory` CRD)

Previously, all run metrics (throughput, training time, node count) existed only in the `DistributedTraining` CR status and were permanently lost when the CR was deleted. A new Kubernetes CRD(`DistributedTrainingHistory`) now persists these metrics for every successful run, namespaced alongside the job.

Each history entry stores the node count used (`n_k`), measured throughput (`P_k` in samples/sec for PyTorch, records/sec for Spark), training duration (`T_k` in seconds), total work approximated as `W ≈ T_k × P_k`, and provisioning time. Retention is bounded to the last 10 entries per configuration, with the oldest deleted synchronously at write time.

#### Configuration Hash

The hash answers the question: **"have we seen this exact workload before?"** Without it, the solver has no way to know which history entries are relevant to the current job. If two different workloads shared the same history pool, p_baseline and α would be computed from meaningless averages and every prediction would be wrong.

The hash is computed from the fields that materially affect either how much total work W there is, or how fast P_k the job processes it — and it differs by backend:

```
# PyTorch
hash(backend | machineType | model.name | dataset.name | dataset.split |
     batchSize | gradAccumulationSteps | epochs | validationSplit)

# Spark
hash(backend | machineType | sparkSpec.image | sparkSpec.mainApplicationFile |
     hash(sparkSpec.arguments))
```

Each field is included for a specific reason. For PyTorch: `batchSize` and `gradAccumulationSteps` directly affect GPU utilisation and therefore P_k; `epochs` and `validationSplit` change the total training set size and therefore W. Changing any of these produces a new hash and resets history to zero, because mixing measurements from different regimes would corrupt the solver's predictions.

The hash is stored as a Kubernetes label on every history CR (`config-hash: <value>`), so the solver can load all relevant entries with a single label-filtered List query. History is scoped to the **namespace** — different teams in different namespaces do not share scaling data.

Critically, the hash is computed from the job's **content**, not its name or UID. This means history survives CR deletions and is shared across CR names:

```
"finetune-run-3" completes  →  history entry written with hash "a3f9..."
"finetune-run-3" deleted
"finetune-run-4" submitted with same config  →  hash "a3f9..."  →  solver finds previous history
```

This CRD is the prerequisite for everything that follows.

---

### 2. Cost Tracking in Job Results

A machine hourly cost `ConfigMap` (read via the Kubernetes client at reconcile time) now backs a cost estimate computed at the end of every successful run:

```
cost = nodeCount × machineHourlyCost × (provisioningSeconds + trainingSeconds) / 3600
```

The result is written as `estimatedCostUSD` in `JobResults`. If the machine type is absent from the ConfigMap the operator emits a Warning event and leaves the field empty rather than producing a misleading zero.

---

### 3. Checkpoint-Aware Resume

A new `spec.resumeFromCheckpoint` field allows users to provide an explicit checkpoint path that the operator injects into the torchrun command as `--resume_from_checkpoint`. This saves GPU cost on every re-run after a failure or preemption without any changes to the training script. The field is PyTorch-only; setting it on Spark or plain-Job backends emits a Warning and has no effect.

---

### 4. Objective-Driven Topology Auto-Sizing

The centerpiece of this extension. Users can now replace explicit topology declarations with a goal:

```yaml
spec:
  objective:
    targetTime: "2h"     # finish within this wall-clock duration
    maxCost: "8.00"      # or within this USD budget
    maxNodes: 5         # hard cap (required)
```

The operator runs a **topology solver** at the end of the Pending phase, before the node pool is provisioned, so the resolved node count is available when the GKE API is called. The solver is **idempotency-guarded**: a Kubernetes reconciler runs many times per job — on every status change, every requeue, every API server event. Without a guard the solver could re-run and pick a different node count between the Pending and Provisioning phases. The guard is simply `status.ResolvedTopology.Nodes == 0` — once the solver writes a value it is never re-run. `Nodes == 0` is a safe value because 0 is never a valid node count.

---

#### Scaling Model

The solver fits a parallel efficiency model to the collected history. In perfect linear scaling, doubling nodes doubles throughput. In reality, nodes spend time coordinating (sending gradients, syncing parameters), so efficiency degrades. The model captures this with a single parameter `α` representing communication overhead per additional node:

```
n_min          = smallest node count seen in history

p_baseline     = average(P_k\ at n_min) / n_min        # per-node baseline throughput
W              = average(T_k × P_k) at n_min           # total work approximation
T_provision_avg = average(T_provision across all history entries)

E(n) = 1 / (1 + α × (n − 1))        # scaling efficiency: 1.0 at n=1, degrades as n grows
P(n) = p_baseline × n × E(n)         # predicted throughput at n nodes
T(n) = W / P(n)                       # predicted training time
     = W / (p_baseline × n × E(n))

Cost(n) = (C_h / 3600) × n × (T(n) + T_provision_avg)
```

`p_baseline` and `W` are both computed from history entries at `n_min` specifically — averaging across all entries at the smallest observed node count. This ensures the model exactly reproduces the calibration run time at its own reference point. Using a single noisy entry, or mixing n values, would introduce bias into every subsequent prediction.

`T_provision_avg` is averaged across all history entries regardless of node count. GKE provisions nodes in parallel so provisioning time is roughly constant regardless of pool size, making cross-n averaging acceptable.

---

#### Estimating α

`α` is the only unknown the model needs to fit from history. At least two runs at **different** node counts are required:

- **Two distinct node counts** — exact closed-form solution:

```
E_observed = (P at n2 / n2) / (P at n1 / n1)
α = (1 − E_observed) / (E_observed × (n2 − 1))
```

- **Three or more distinct node counts** — no exact solution exists, so the solver minimizes the sum of squared residuals via a **1D golden-section search** over `α ∈ [0, 1.0]`:

```
minimize over α:  Σ (P_predicted(n) − P_observed(n))² (achieved by golden-section search)
```

Golden-section search is an iterative algorithm that finds the minimum of a unimodal function over an interval by repeatedly evaluating at two interior points and shrinking the interval, similar in spirit to binary search but for minimization. It requires no external libraries since there is only one scalar parameter. The upper bound `α = 1.0` is a hard cap: at α=1, E(4)=0.25, already representing severe degradation beyond any well-implemented distributed training workload. Values above 1.0 indicate measurement noise rather than real scaling behaviour.

---

#### Solver Algorithm

The solver runs in seven steps. The ordering is deliberate — W is checked before C_h because without W none of the equations can be evaluated regardless of cost availability:

| Step | What happens |
|---|---|
| 1 | **W check** — no history for this config hash → calibration run, skip all remaining steps |
| 2 | **C_h check** — machine type not in cost ConfigMap → fall back to calibration (cost-only) or skip cost steps (time+cost or time-only) |
| 3 | **Flat cost detection** — if `\|Cost(maxNodes) − Cost(2)\| / Cost(2) < 1%` the cost curve is numerically flat (occurs when α=0 and no provisioning history); binary search on cost would be meaningless, handled separately |
| 4 | **Time-only branch** — binary search for the smallest n_time where T(n) ≤ targetTime |
| 5 | **Cost-only branch** — binary search for the largest n_budget where Cost(n) ≤ maxCost |
| 6 | **Both constraints: find n_time** — same search as step 4, intermediate value only |
| 7 | **Both constraints: check budget** — if Cost(n_time) ≤ maxCost, both are satisfied; otherwise conflict resolution applies |

Steps 4, 5, and 6–7 are mutually exclusive — only one branch runs per job.

**Why binary search works:** Within the model, T(n) is non-increasing in n and Cost(n) is strictly increasing in n. This monotonicity holds by construction — it follows from constraining α ∈ [0, 1.0], which ensures E(n) degrades smoothly rather than creating non-monotone behaviour. Binary search is valid on the model's predicted values; it does not make claims about real-world throughput.

**Model assumption:** the scaling model assumes communication overhead degrades smoothly and continuously as n grows. In practice, real workloads can exhibit sudden efficiency drops at specific node counts — for example when crossing a network boundary or exceeding NVLink bandwidth. The model cannot predict these thresholds before observing them. When this happens the solver's selected n may be suboptimal, but the actual run's throughput is recorded in history and α is re-estimated on the next run, making future predictions more conservative. The solver is therefore best understood as a reasonable starting point that improves over runs, rather than a guaranteed optimal solution.

**Conflict resolution (step 7):** when n_time is the cheapest option that meets the time target but still exceeds the budget, no n satisfies both constraints simultaneously. The operator falls back to `n_budget` (max n within budget) and emits a Warning:
```
Warning: targetTime unreachable within maxCost — using 5 nodes, estimated time 2h45m.
Increase maxCost or relax targetTime.
```

---

#### Bootstrap

On the first run for a new configuration, W is unknown and the solver cannot run. The operator falls back to a configurable calibration topology (default: 2 nodes via `--default-calibration-nodes`), emits an informational event, and writes a history entry on completion. From the second run onward the solver has at least one data point and begins making estimates. A second run at a different node count enables α estimation, giving the model enough information to distinguish linear from degraded scaling. Predictions improve further as more distinct node counts accumulate in history.

---

#### End-to-End Example

**Setup:** PyTorch fine-tuning job, `n1-standard-8` machine at `$0.38/hr`, objective `targetTime: "2h"`, `maxCost: "5.00"`, `maxNodes: 8`.

---

**Run 1 — Calibration**

No history exists for this config hash. The solver cannot run.

```
Event: "No history found — running calibration topology (n=2)"
ResolvedTopology: { Nodes: 2, EstimatedTime: "unknown (calibration run)" }
```

The job runs on 2 nodes and completes successfully. The operator reads from pod logs:

```
train_samples_per_second = 1000
```

And from timing:
```
T_k          = 14400s   (training time)
T_provision  = 180s     (provisioning time)
P_k          = 1000 samples/sec
W            = T_k × P_k = 14,400,000
```

History entry written: `{ n=2, P_k=1000, T_k=14400, W=14,400,000, T_prov=180 }`

---

**Run 2 — Solver with one data point (α = 0)**

History has one entry at n=2. Only one distinct node count → `α = 0` assumed (perfect linear scaling).

```
p_baseline     = 1000 / 2 = 500 samples/sec/node
W              = 14,400,000
T_provision_avg = 180s
α              = 0  →  E(n) = 1.0 for all n
```

Predicted times:
```
T(2) = 14,400,000 / (500 × 2 × 1.0) = 14400s
T(3) = 14,400,000 / (500 × 3 × 1.0) = 9600s
T(4) = 14,400,000 / (500 × 4 × 1.0) = 7200s  ← exactly meets 7200s target
T(5) = 14,400,000 / (500 × 5 × 1.0) = 5760s  ← maxNodes
```

Binary search finds **n=4** (smallest n where T(n) ≤ 7200s).

Cost check:
```
Cost(4) = (0.38 / 3600) × 4 × (7200 + 180) = 0.000106 × 4 × 7380 = $3.11
$3.11 ≤ $5.00  →  both constraints satisfied
```

```
ResolvedTopology: { Nodes: 4, MasterReplicas: 1, WorkerReplicas: 3,
                    EstimatedTime: "2h0m0s", EstimatedCost: "3.1100" }
```

Job runs on 4 nodes. It finishes in 7600s (slightly over the predicted 7200s — because α=0 was an optimistic assumption, real communication overhead exists). Results written:

```
T_k = 7600s,  P_k = 1800 samples/sec,  T_provision = 185s
W   = 7600 × 1800 = 13,680,000
```

History now has two entries: `{ n=2, P_k=1000 }` and `{ n=4, P_k=1800 }`.

---

**Run 3 — Full solver with α estimated**

Two distinct node counts → closed-form α estimation.

```
p_baseline      = 1000 / 2 = 500
W               = 14,400,000     (averaged at n_min=2, only one entry there)
T_provision_avg = (180 + 185) / 2 = 182.5s

E_observed = (P_k(n=4) / 4) / (P_k(n=2) / 2)
           = (1800 / 4) / (1000 / 2)
           = 450 / 500
           = 0.90

α = (1 − 0.90) / (0.90 × (4 − 1)) = 0.10 / 2.70 = 0.037
```

Predicted times with α=0.037:
```
E(3) = 1 / (1 + 0.037×2) = 0.931   →  P(3) = 500×3×0.931 = 1397  →  T(3) = 10,308s
E(4) = 1 / (1 + 0.037×3) = 0.900   →  P(4) = 500×4×0.900 = 1800  →  T(4) = 8,000s
E(5) = 1 / (1 + 0.037×4) = 0.871   →  P(5) = 500×5×0.871 = 2178  →  T(5) = 6,612s  ✓
```

Binary search finds **n=5** (T(4)=8000s > 7200s, T(5)=6612s ≤ 7200s).

Cost check:
```
Cost(5) = (0.38 / 3600) × 5 × (6612 + 182.5) = 0.000106 × 5 × 6794.5 = $3.59
$3.59 ≤ $5.00  →  both constraints satisfied
```

```
ResolvedTopology: { Nodes: 5, MasterReplicas: 1, WorkerReplicas: 4,
                    EstimatedTime: "1h50m12s", EstimatedCost: "3.5900" }
```

The solver now has a calibrated model. Every subsequent run at a new node count adds another history entry and refines α further via golden-section search.

---

## Real-World Business Use Cases

The solver's core value is **automatic infrastructure sizing from SLA and budget constraints, learning progressively from historical runs**. This pattern appears in several production contexts:

**Nightly batch ML retraining** (closest fit): recommendation engines, fraud detection, and demand forecasting models retrain on fresh data nightly and must finish before morning traffic peaks. The solver calibrates on the first two runs and then satisfies the deadline (e.g., `targetTime: 5h`) at minimum cost automatically — replacing a platform engineer's manual node-count tuning.

**Financial risk computation**: banks run Monte Carlo simulations and VaR calculations tied to regulatory reporting deadlines (market close, T+1 settlement). `targetTime` maps directly to the reporting deadline; `maxCost` maps to the compute budget per trading day.

**Multi-tenant ML platform (SaaS)**: a company offering tiered training SLAs (Gold = 15 min, Silver = 30 min, Bronze = best effort) can back each tier with a different `targetTime`. The solver handles capacity planning automatically — no human decides which tier gets how many nodes.

**CI/CD fine-tune validation**: every model code change triggers a validation fine-tune. Teams want fast feedback (`targetTime: 20m`) without paying GPU prices for what is essentially a smoke test. The solver finds the minimum node count that meets the cycle time.

### Limitations and Extension Points

The scaling model assumes communication overhead scales linearly with n, which holds well for data-parallel training with gloo/NCCL but breaks down for pipeline parallelism or heterogeneous hardware. Replacing the single-α scaling model with a more expressive scaling law — for example Gustafson's law, or an empirical neural scaling law for LLMs — would extend accuracy at large node counts and represents a natural direction for further research.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| History written for all Succeeded runs (not only objective-mode) | Ensures prior explicit-topology runs seed the solver when a user later switches to objective mode |
| `ResolvedTopology` as a value type in status | Zero value (`Nodes == 0`) is a valid "not yet resolved" sentinel; no pointer indirection needed |
| History write skipped when `P_k == 0` | `W = T_k × P_k = 0` would cause division-by-zero in the solver; better to re-run calibration |

---