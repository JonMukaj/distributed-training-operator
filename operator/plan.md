# Operator Feature Plan: Objective-Driven Topology Auto-Sizing

## Overview

The operator currently requires users to manually specify topology (node count, master/worker split).
The goal is to let users declare **what they want to achieve** — target time, max cost — and have the
operator determine the optimal topology automatically, improving across runs using collected history.

---

## Features

### 1. DistributedTrainingHistory CRD

A new CRD that persists metrics from every completed **successful** run. History entries from failed
runs are excluded from solver inputs because partial throughput data would bias `p_baseline`.
Currently results only live in the `DistributedTraining` CR status. Once the CR is deleted, history is gone.

**Config hash key is backend-aware:**

The hash must differ by backend because the fields that identify a "same job" are different:

- **pytorch:** `hash(backend + machineType + model.name + dataset.name + dataset.split +
  batchSize + gradAccumulationSteps + epochs + validationSplit)`
- **spark:** `hash(backend + machineType + sparkSpec.image + sparkSpec.mainApplicationFile +
  hash(sparkSpec.arguments))`

Fields included and why (pytorch):
- `batchSize` + `gradAccumulationSteps` — directly affect `tokensPerSecond` (larger effective batch
  = better GPU utilization = higher throughput). Comparing across different values corrupts `p_baseline`.
- `epochs` — total work W is proportional to epochs. Changing epochs changes W and must trigger
  recalibration.
- `validationSplit` — determines training set fraction. Changing from 0.2 to 0.3 reduces the
  training set by 12.5%, changing W by the same proportion. Same impact as epochs — must be in hash.

For Spark, `model/dataset/epochs/batchSize` are empty strings and would cause all Spark jobs on
the same machine type to collide to the same hash, corrupting history. Spark hashes on its own
application identity fields instead.

**Namespace scope:** history CRs are namespaced to the same namespace as the `DistributedTraining`.
Cross-namespace history sharing is intentionally not supported, since different namespaces represent
different teams or environments and should not share scaling data.

**History is written for all Succeeded runs regardless of job mode:** objective and
explicit-topology alike. This ensures that if a user later switches a workload to objective mode,
the solver can benefit from throughput data collected during prior explicit-topology runs on the
same config hash. Implementors must not skip history writes for non-objective jobs.

**Stored per run:**
- `n_k` — actual node count used. Source depends on job mode:
  - objective mode: `status.ResolvedTopology.Nodes` (spec.pytorchSpec is empty in objective mode
    so masterReplicas + workerReplicas = 0 — must not be used)
  - explicit topology: `masterReplicas + workerReplicas` for pytorch, `topology.nodes` for spark
- `P_k` — throughput (`tokensPerSecond` or `throughputRecordsPerSec` from `JobResults.Metrics`)
- `T_k` — training time (from `JobResults.TrainingTime`, parsed with `time.ParseDuration` to seconds;
  return error and skip history write if parsing fails — never silently use 0)
- `W` — total work approximated as `T_k × P_k` (tokens or records processed — see W derivation below)
- `T_provision` — provisioning time (from `JobResults.ProvisioningTime`) for budget correction
- `machineType`, `backend`, `configHash` — for lookup

**Guard on P_k:** if `P_k` is missing or zero (log parsing regex found no match), skip writing the
history entry entirely and emit a Warning event. Storing `W = 0` would cause division by zero in
the solver.

**Retention:** keep the last 10 Succeeded entries per `configHash`. The operator deletes the oldest
entry (by `creationTimestamp`) synchronously before writing a new one when the count would exceed 10.
Triggered at history-write time inside the reconciler, not by a separate controller.
Two reconciler goroutines racing to write could produce 11 entries momentarily — this is acceptable
(off-by-one in a retention policy has no correctness impact) and not worth a distributed lock.

**Scheme registration:** `DistributedTrainingHistory` must be registered in `main.go` alongside
`DistributedTraining`, otherwise the controller-runtime client panics at startup.

**RBAC:** the operator's `ClusterRole` needs `create`, `get`, `list`, `watch`, `update`, `delete`
on `distributedtraininghistories`.

History CRs are namespaced and labeled with `config-hash` for efficient `List` queries.

---

### 2. Objective-Driven Topology Auto-Sizing

Users declare objectives instead of topology:

```yaml
spec:
  objective:
    targetTime: "2h"       # finish within this wall-clock duration  (at least one of
    maxCost: "8.00"        # or within this USD budget               these two required)
    maxNodes: 16           # hard cap on node count (required)
```

`totalWork` is **not** user-declared — it is approximated from history (see W derivation below).

**Validation rules (enforced at Pending phase or via webhook):**
- `objective` and explicit `topology` / `pytorchSpec.masterReplicas/workerReplicas` are mutually
  exclusive. Reject with a clear message if both are set.
- If `objective` is set, at least one of `targetTime` or `maxCost` must be present. An empty
  `objective: {}` block is rejected.
- `maxNodes` must be ≥ 2.

**Resolved topology goes to status, not spec.** The solver runs at the **end of the Pending phase**,
before transitioning to Provisioning, so that the resolved node count is available when the GKE
CreateNodePool API is called. Two type changes are required in `api/v1/distributedtraining_types.go`:

1. **`ObjectiveSpec` added as a pointer field in `DistributedTrainingSpec`:**

```go
type ObjectiveSpec struct {
    TargetTime string `json:"targetTime,omitempty"`
    MaxCost    string `json:"maxCost,omitempty"`
    // +kubebuilder:validation:Minimum=2
    MaxNodes   int32  `json:"maxNodes"`
}
```

Added to `DistributedTrainingSpec` as a pointer so nil = "not set":

```go
// Objective declares time/cost constraints for automatic topology sizing.
// Mutually exclusive with explicit topology fields.
// +optional
Objective *ObjectiveSpec `json:"objective,omitempty"`
```

Using a pointer allows a clean nil check (`spec.Objective != nil`) everywhere the code branches
between objective mode and explicit-topology mode. A value type would require fragile zero-value
checks and cannot distinguish "not set" from "set to zero values."

2. **`ResolvedTopology` added as a value type in `DistributedTrainingStatus`:**

```go
type ResolvedTopology struct {
    Nodes          int32  `json:"nodes"`
    MasterReplicas int32  `json:"masterReplicas,omitempty"`  // pytorch only
    WorkerReplicas int32  `json:"workerReplicas,omitempty"`  // pytorch only
    EstimatedTime  string `json:"estimatedTime"`             // always populated
    EstimatedCost  string `json:"estimatedCost,omitempty"`   // omitted if C_h unavailable
}
```

`ResolvedTopology` is a **value type** in the status struct (not a pointer). `Nodes == 0` is used
as the sentinel for "not yet resolved" since 0 is never a valid node count (minimum is 2).
Value type is correct here because status.ResolvedTopology always exists as a struct — the
zero value naturally represents "unresolved."

**Solver idempotency:** the reconciler runs multiple times per job. The solver must only run once.
At the start of the Pending phase, check `status.ResolvedTopology.Nodes > 0`. If already set,
skip the solver entirely and proceed with the existing value. This prevents re-solving on every
reconcile loop and ensures stable topology across retries.

**Bootstrap path — ResolvedTopology during calibration:** when no history exists and the solver
is skipped, `ResolvedTopology` must still be written before transitioning to Provisioning so the
Provisioning phase has a node count to work with:

```go
status.ResolvedTopology.Nodes         = defaultCalibrationNodes  // e.g. 2, from operator config
status.ResolvedTopology.EstimatedTime = "unknown (calibration run)"
// EstimatedCost left empty
```

This also gives users visibility via `kubectl get dj` into what topology is actually running.

**`EstimatedCost` population rule:** only write `EstimatedCost` to `ResolvedTopology` when `C_h`
is known (machine type found in the ConfigMap). If `C_h` is unavailable, leave `EstimatedCost`
empty and emit the existing Warning. Writing a zero or derived-from-zero cost would be misleading.

**Provisioning phase branching:** the Provisioning phase currently reads node count from spec.
With objective mode, it must branch:

```
if spec.objective is set:
    if status.ResolvedTopology.Nodes == 0:
        fail with error "ResolvedTopology not set before Provisioning — solver bug"
        // defensive guard: should never happen if solver always writes before transitioning
    nodeCount = status.ResolvedTopology.Nodes
else:
    nodeCount = spec.pytorchSpec.masterReplicas + spec.pytorchSpec.workerReplicas  // pytorch
              | spec.topology.nodes                                                 // spark
```

The defensive guard (`Nodes == 0`) makes bugs immediately visible with a clear error rather than
causing a cryptic GKE API failure when trying to create a 0-node pool.

This keeps existing explicit-topology behavior entirely unchanged.

The controller reads from `status.ResolvedTopology` when generating the backend manifest.
This avoids spec mutation and survives `kubectl apply` re-runs.

---

### 3. Checkpoint-Aware Resume

An explicit opt-in feature: the user provides a checkpoint path and the operator injects it into
the torchrun command. **Applies to the pytorch backend only.** Setting this field on a Spark or
job backend job should emit a validation Warning and the field should be ignored (not silently
swallowed — the user made a mistake and should know).

```yaml
spec:
  resumeFromCheckpoint: "checkpoints/checkpoint-500"  # explicit path, optional; pytorch only
```

If set, `--resume_from_checkpoint <path>` is appended to the torchrun command in the pytorch
backend command builder. Saves GPU cost on every re-run after failure or preemption.

**Future work:** automatic detection of the latest checkpoint in `outputPVC` without user input.

---

### 4. Cost Tracking in Results

Add `estimatedCostUSD` to `JobResults`. Computed in the Succeeded phase:

```
cost = nodeCount × machineHourlyCost × (provisioningSeconds + trainingSeconds) / 3600
```

Machine costs come from a `ConfigMap` mounted by the operator (e.g. `n1-standard-8: "0.38"`).
If the machine type is missing from the ConfigMap, emit a Warning event and skip cost tracking
rather than blocking the job.

---

## Topology Solver Logic

### W derivation (total work)

`DatasetSpec` only stores the dataset name and split, not token count. The operator cannot know
dataset size without loading it. Therefore W is not computed from spec fields.

**Actual approach:** approximate W from the calibration run:

```
W ≈ T_k × P_k    (training time × throughput ≈ total tokens or records processed)
```

This is a close approximation, not an exact value. `P_k` is parsed from log lines via regex and
may represent a point-in-time throughput value (e.g., the last logged rate) rather than the
time-weighted average. Training throughput varies across steps — the first steps are typically
slower due to JIT compilation and data-loader warm-up — so the approximation may slightly
overestimate or underestimate W. This is acceptable for the purpose of topology sizing.

W is stored in the history entry and reused for subsequent solver runs on the same config hash.
Because `epochs` and `validationSplit` are part of the hash, changing either always triggers a
new calibration rather than reusing a stale W.

**Which W to use:** when multiple history entries exist at the smallest n, use the **average W**
across all of them (`avg(T_k × P_k)` at n_min). This keeps W consistent with how p_baseline is
computed (p_baseline also averages P_k across all entries at n_min), ensuring the model recovers
the observed calibration time at its own reference point. Using a single entry's raw W while
p_baseline is averaged would cause the model to predict the wrong time for the calibration run.

For the very first run (no history), W is unknown — see bootstrap section.

### Scaling model

Define **effective throughput per node** from run k:

```
p_k = P_k / n_k
```

In perfect linear scaling `p_k` is constant. In reality it degrades with n due to communication
overhead. Model with a single parameter α (scaling overhead):

```
η(n)  = 1 / (1 + α × (n - 1))        scaling efficiency at n nodes
P(n)  = p_baseline × n × η(n)         predicted throughput at n nodes
```

`p_baseline` = (average P_k across all history entries at the smallest n) / n_min.

**Estimating α** from ≥2 runs at **different** node counts `(n_1, P_1)` and `(n_2, P_2)`,
where n_1 < n_2:

```
η_2_observed = (P_2 / n_2) / (P_1 / n_1)
α            = (1 - η_2_observed) / (η_2_observed × (n_2 - 1))
```

When multiple history entries exist at the same n value, average their P_k values before applying
the formula. This reduces measurement noise from run-to-run throughput variance.

**Precondition:** before estimating α, verify that history contains at least two entries with
distinct n values. If all history entries share the same n, fall back to α=0. This prevents
a degenerate formula where n_1 = n_2.

**Edge case — α < 0:** if a higher-n run shows superlinear throughput (cache warming, better
batch packing), `η_2_observed > 1` and α goes negative, breaking the model. Apply a floor:

```
α = max(0, computed_α)
```

**With ≥3 distinct n values:** minimize the sum of squared residuals over α ∈ [0, α_max] using
a 1D golden section search — straightforward to implement in Go with no external libraries since
there is only one scalar parameter. Apply the α ≥ 0 floor to the result.

**α_max = 1.0.** At α=1, η(2)=0.5 and η(4)=0.25 — already severe degradation beyond any
well-implemented distributed training workload. α > 1.0 indicates data noise rather than real
scaling behavior and should be capped. The golden section search range is therefore [0, 1.0].

With only 1 data point (or all entries at the same n), assume α=0 (linear scaling — optimistic
but safe as a first estimate).

### Time and cost equations

```
T(n)    = W / P(n)
        = W / (p_baseline × n × η(n))          ← non-increasing in n; strictly decreasing for α < 1
                                                   (at α=1: P(n)=p_baseline constant, T flat)

Cost(n) = training_cost(n) + provisioning_cost(n)

training_cost(n)     = C_h × n × T(n) / 3600
                     = C_h × W / (p_baseline × η(n)) / 3600    ← depends only on η, not n directly
                       (divide by 3600: C_h is $/node/hr, T(n) is in seconds)

provisioning_cost(n) = C_h × n × T_provision_avg / 3600        ← linear in n
                       T_provision_avg = average T_provision across all history entries, in seconds.
                       GKE provisions nodes in parallel so T_provision is roughly constant
                       regardless of n — averaging across entries of different n is acceptable.

Cost(n) = (C_h / 3600) × (W / (p_baseline × η(n))  +  n × T_provision_avg)
```

Note: all arithmetic uses float64. Integer node counts must be explicitly cast when passed into
these formulas (required in Go).

**Monotonicity of Cost(n):**
- For α > 0: `training_cost` increases with n (η degrades), `provisioning_cost` increases linearly → strictly increasing ✓
- For α = 0: `training_cost` is flat, `provisioning_cost` increases linearly → strictly increasing ✓
- Degenerate case (α=0 AND T_provision_avg=0): Cost is flat. Guard:
  `if Cost(2) == 0 OR |Cost(maxNodes) - Cost(2)| / Cost(2) < 0.01` → treat cost as unconstrained.
  The `Cost(2) == 0` check prevents division by zero if `C_h = 0.00` in the ConfigMap.
  The 1% relative threshold scales correctly with job size.

### Solver algorithm

The W check always runs first. The C_h check runs second. This ordering is critical: if W is
unknown the solver cannot run regardless of C_h availability, so W must be checked before
branching on C_h.

```
given: T_target (optional), B_max (optional), maxNodes, α, p_baseline, C_h (optional), W, T_provision_avg

Note: α and p_baseline are computed from the history List result for this configHash using the
scaling model above, before the solver steps begin. W is the average of (T_k × P_k) across all
entries at n_min. T_provision_avg is the average of all T_provision values across all entries.

Preconditions (validated before solver runs):
  - at least one of T_target or B_max is set (enforced at admission)
  - maxNodes ≥ 2
  - status.ResolvedTopology.Nodes == 0 (idempotency check — skip entire solver if already solved)

── Step 1: W availability (always first) ────────────────────────────────────────

1. If no history exists for this configHash (W unknown):
   write calibration ResolvedTopology (Nodes=defaultCalibrationNodes, EstimatedTime="unknown (calibration run)")
   transition to Provisioning; done.

── Step 2: C_h availability (only runs if W is known) ───────────────────────────

2. c_h_known = (machineType found in ConfigMap)
   if NOT c_h_known:
     if only maxCost given (no targetTime):
       emit Warning "machineType not in cost ConfigMap, cannot evaluate maxCost — falling back to calibration"
       write calibration ResolvedTopology; done.
     if targetTime is set (targetTime-only or both constraints):
       emit Warning "machineType not in cost ConfigMap — cost constraint skipped, using time-only path"
       skip step 3 (flat detection requires C_h), jump directly to step 4 (targetTime only),
       skip steps 5, 6, 7.

── Step 3: Detect flat Cost curve (only when c_h_known) ─────────────────────────

3. flat = (Cost(2) == 0) OR (|Cost(maxNodes) - Cost(2)| / Cost(2) < 0.01)

── Step 3→4/5/6 dispatch (branch on which constraints are set) ──────────────────

   After step 3, branch explicitly:
     if only targetTime given (no maxCost) → go to step 4
     if only maxCost given  (no targetTime) → go to step 5
     if both targetTime and maxCost given  → go to steps 6–7

── Step 4: targetTime only ──────────────────────────────────────────────────────

4. Find n_time = min n in [2, maxNodes] such that T(n) ≤ T_target
   → binary search over integers (T non-increasing in n)
   → if T(maxNodes) > T_target:
       emit Warning "targetTime unreachable at maxNodes, using maxNodes"
       n_time = maxNodes
   → write status.ResolvedTopology:
       Nodes=n_time, EstimatedTime=T(n_time)
       EstimatedCost=Cost(n_time) only if c_h_known, else omit
   → done.

── Step 5: maxCost only ─────────────────────────────────────────────────────────

5. If flat:
     if Cost(2) ≤ B_max: n_budget = maxNodes  // all equally cheap; pick fastest
     else: emit Warning "budget too small for any topology, using n=2"; n_budget = 2
   Otherwise (Cost strictly increasing):
     if Cost(2) > B_max:
       emit Warning "maxCost too low for minimum topology (n=2), using n=2"; n_budget = 2
     else:
       find n_budget = max n in [2, maxNodes] such that Cost(n) ≤ B_max
       (binary search; Cost strictly increasing → rightmost satisfying n)
   → write status.ResolvedTopology:
       Nodes=n_budget, EstimatedTime=T(n_budget), EstimatedCost=Cost(n_budget)
   → done.

── Steps 6–7: both targetTime and maxCost ───────────────────────────────────────

6. Find n_time using the same binary search logic as step 4, but do NOT write
   status yet — n_time is an intermediate value here.

7. Compute Cost(n_time):
   a. Cost(n_time) ≤ B_max → both satisfied.
      → write status.ResolvedTopology:
          Nodes=n_time, EstimatedTime=T(n_time), EstimatedCost=Cost(n_time)
      → done.

   b. Cost(n_time) > B_max → conflict.
      Since Cost is strictly increasing (or flat), n_time is the minimum-n option that
      meets targetTime and is therefore already the cheapest. No n satisfies both.

      Flat-cost sub-case: if flat AND Cost(2) > B_max →
        emit Warning "budget too low for any topology, using n=2"
        write status.ResolvedTopology: Nodes=2, EstimatedTime=T(2), EstimatedCost=Cost(2)
        done.

      Otherwise:
      → if Cost(2) > B_max:
          emit Warning "budget too low for minimum topology, using n=2"
          write status.ResolvedTopology: Nodes=2, EstimatedTime=T(2), EstimatedCost=Cost(2)
          done.
        else:
          find n_budget = max n in [2, maxNodes] such that Cost(n) ≤ B_max (binary search)
          emit Warning "targetTime unreachable within maxCost — using n_budget nodes,
            estimated time T(n_budget). Increase maxCost or relax targetTime."
          write status.ResolvedTopology:
            Nodes=n_budget, EstimatedTime=T(n_budget), EstimatedCost=Cost(n_budget)
          done.
```

Every code path writes `status.ResolvedTopology` before returning. The Provisioning phase reads
`status.ResolvedTopology.Nodes` unconditionally for objective jobs, with a defensive guard that
fails explicitly if Nodes is somehow 0 (see feature 2 Provisioning phase branching).

### Master/worker split after n is resolved

| Backend | Provisioned nodes | Master           | Workers                              |
|---------|-------------------|------------------|--------------------------------------|
| pytorch | n                 | masterReplicas=1 | workerReplicas=n-1                   |
| spark   | n                 | driver=1 (own pod, scheduled onto pool) | executors=n, `topology.nodes=n` |
| job     | 1                 | n always=1       | solver does not apply                |

**Spark note:** `TopologySpec.Nodes` represents executor node count, not total nodes including
driver. The driver pod is scheduled onto the same provisioned node pool but is not counted in
`topology.nodes`. Therefore the solver's n maps directly to `topology.nodes = n` (executor count),
not `n-1`. All n provisioned nodes are executor-eligible; the driver may share a node with an
executor depending on resource availability.

For Spark, `ResolvedTopology.MasterReplicas` and `ResolvedTopology.WorkerReplicas` are left empty;
the Spark backend reads only `ResolvedTopology.Nodes` and maps it to `topology.nodes`.

Minimum n=2 is enforced for pytorch and spark. At n=1 the scaling model degenerates (η=1,
zero communication overhead) and objective-driven mode adds no value over explicit topology.

### Bootstrap problem (no history for this job config)

W is unknown, `p_baseline` and α are unavailable on the first run.

- Operator uses a **configurable default topology** (e.g. 2 nodes, set in operator config).
- Writes `status.ResolvedTopology` with calibration values (see feature 2 bootstrap section).
- Emits Event: `"No history found — running calibration topology (n=2). Objective will be
  applied from next run."`
- On completion: if `P_k > 0`, write history entry storing `W ≈ T_k × P_k` with
  `n_k = status.ResolvedTopology.Nodes`; otherwise emit Warning and leave history empty
  (next run is another calibration attempt).
- From run 2 onwards: solver uses the calibration run as the first data point with α=0, until a
  second run at a **different** n provides enough data to estimate α.

Only one run per new job configuration is unoptimized (assuming log parsing succeeds).

---

## Machine Cost ConfigMap

Mounted by the operator. Contains per-node hourly cost per GCE machine type.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: machine-costs
  namespace: distributed-trainings-system
data:
  n1-standard-8: "0.38"
  n1-standard-16: "0.76"
  a2-highgpu-1g: "3.67"
  c4-highcpu-8: "0.42"
```

If a job's `hardware.machineType` is absent from the map, solver step 2 handles the fallback
(time-only optimization or calibration depending on which constraints are set).
Cost tracking in `JobResults` is also skipped with a Warning in this case.

---

## Implementation Order

1. **DistributedTrainingHistory CRD** — type definition, scheme registration, RBAC, write on
   Succeeded (not Failed) for ALL job modes (objective and explicit-topology alike), skip for
   `job` backend, n_k sourced from `status.ResolvedTopology.Nodes`
   for objective jobs and from spec for explicit-topology jobs, P_k=0 guard, T_k parse with
   `time.ParseDuration` (error on failure, do not silently use 0), backend-aware config hash
   (pytorch vs spark fields), namespace-scoped by design, retention cleanup (keep last 10 per
   configHash, delete oldest by creationTimestamp at write time, accept occasional off-by-one
   from concurrent reconcilers)
2. **Cost tracking** — `estimatedCostUSD` in `JobResults`, ConfigMap mount, missing-key Warning
3. **Checkpoint-aware resume** — `resumeFromCheckpoint` spec field, pytorch backend only,
   emit validation Warning and ignore if set on Spark or job backend, inject into torchrun builder
4. **Topology solver** — `ObjectiveSpec` pointer field (`*ObjectiveSpec`) added to `DistributedTrainingSpec`
   (nil = not set; enables clean nil check throughout), validation (mutual exclusion, at least one
   constraint, maxNodes≥2), `ResolvedTopology` value-type struct added to `DistributedTrainingStatus` with
   `Nodes==0` as not-yet-resolved sentinel, Provisioning phase branching (spec vs status.ResolvedTopology) with
   defensive guard (fail explicitly if objective set but Nodes==0), idempotency check (skip solver if
   `ResolvedTopology.Nodes > 0`), solver runs at end of Pending phase with W check first (step 1)
   then C_h check (step 2) then flat detection (step 3) then constraint branches (steps 4/5/6/7),
   step 2 C_h fallback jumps to step 4 directly (skipping step 3), every code path writes
   `ResolvedTopology`, `EstimatedCost` omitted when C_h unavailable, average W across all entries
   at n_min, P_k averaging across same-n entries before α estimation, α_max=1.0 for golden
   section search, flat-Cost guard with Cost(2)==0 division-by-zero protection and 1% relative
   threshold, step 6 computes n_time as intermediate value only (does not write status),
   Spark backend reads only `ResolvedTopology.Nodes`
5. **Bootstrap calibration** — configurable default topology, calibration `ResolvedTopology` write
   with `Nodes=defaultCalibrationNodes`, Warning event, P_k guard on history write with
   n_k=status.ResolvedTopology.Nodes, α=0 fallback when all history entries share the same n

Steps 1–3 are independent and can be parallelized. Step 4 depends on step 1. Step 5 is part of step 4.
