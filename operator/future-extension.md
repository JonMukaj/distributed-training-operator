# Future Extension — NodePool CR and Two-Operator Split

## Motivation

The current operator manages GKE node pool lifecycle directly inside the
`DistributedTraining` reconciler via the `cloud.Provider` interface. This works
correctly for a single-team, single-cloud deployment but has one structural
fragility: if the operator is deleted before the `DistributedTraining` CRs are
cleaned up, the finalizer `training.distributedtraining.io/cleanup` can never be
processed and the real GKE node pool is orphaned — continuing to run and bill.

The root cause is that the node pool exists only as a GKE resource, not as a
Kubernetes object. Kubernetes garbage collection cannot reach it.

### Actual blast radius

The risk is narrower than it first appears. The node pool lifecycle differs by
terminal phase:

- **Succeeded jobs**: the node pool is deleted during `reconcileCollecting`,
  before the CR ever reaches `PhaseSucceeded`. By the time the CR is deleted,
  there is nothing on GKE to clean up. The finalizer just removes itself. No
  cloud cost risk even if the operator is gone.

- **Failed jobs**: the node pool is only deleted via `reconcileDelete` when the
  CR is explicitly removed. If the operator is gone at that point, the finalizer
  blocks and the pool keeps running.

In practice the failed-job risk is further narrowed: a job that fails during
or before provisioning never had a fully running pool to begin with. The
concrete danger is a job that **fails mid-training** — pool fully created,
training crashes — and the operator is then deleted before the CR is cleaned up.
That specific window is where a live pool can be genuinely orphaned.

---

## Proposed Architecture

### Phase 1 — NodePool CR (single operator, no orphaning)

Introduce a `NodePool` CRD that represents one ephemeral node pool. The
`DistributedTraining` controller creates a `NodePool` CR with an owner reference
pointing to the `DistributedTraining`. A new `NodePoolReconciler` (in the same
operator binary) holds the finalizer and calls the cloud API.

```
DistributedTraining controller
  └── creates NodePool CR  (ownerRef → DistributedTraining)
        └── NodePoolReconciler
              └── calls cloud.Provider.CreateNodePool / DeleteNodePool
              └── writes NodePool.Status.Ready, ProvisioningTime
              └── holds the finalizer that deletes the real pool
```

When the `DistributedTraining` is deleted, Kubernetes GC automatically deletes the
owned `NodePool` CR. The `NodePoolReconciler` finalizer then deletes the real
pool — even if the `DistributedTraining` controller is gone.

**Work items:**

| Item | Estimated size |
|---|---|
| `api/v1/nodepool_types.go` — new CRD types | ~50 lines |
| `internal/controller/nodepool_controller.go` — new reconciler | ~150 lines |
| Refactor `reconcilePending` / `reconcileDelete` to create/watch `NodePool` CR instead of calling `r.Cloud` directly | ~80 lines changed |
| Status propagation: `DistributedTraining` waits for `NodePool.Status.Ready = true` before transitioning to `PhaseReady` | ~30 lines + `Watches` in `SetupWithManager` |
| `make manifests` to regenerate CRDs and RBAC | trivial |

**Effort:** ~2–3 days for someone familiar with controller-runtime and this codebase.

### Phase 2 — Two-Operator Split (multi-team, multi-cloud)

Split into two independent operators with separate image builds, RBAC, and
release cadences:

```
Infrastructure Operator              Training Operator
(NodePool CR)                        (DistributedTraining CR)
─────────────────────────────────    ────────────────────────────────
Create/delete GKE/EKS/AKS pools      Submit PyTorchJob / SparkApp
Poll cloud APIs for readiness        Collect metrics, write history
Hold finalizer on real infra         Run the topology solver
Owned by platform/infra team         Owned by ML platform team
```

The `NodePool` CRD schema becomes a shared contract between the two operators,
versioned independently (`v1alpha1` → `v1`) with explicit compatibility
guarantees.

**Effort:** ~3–4 additional weeks (second operator scaffold, shared CRD
versioning, separate CI/CD pipelines, integration testing across both).

---

## Multi-Cloud Impact

The `cloud.Provider` interface already provides the abstraction seam:

```go
type Provider interface {
    CreateNodePool(ctx, name, cfg) (string, error)
    DeleteNodePool(ctx, name) (string, error)
    GetNodePool(ctx, name) (*NodePoolInfo, error)
    Name() string
}
```

Each cloud gets its own `NodePoolReconciler` implementation that encapsulates
provider-specific quirks:

| Provider | Quirks to encapsulate |
|---|---|
| GKE | Operation polling, BCID warnings, pool warming time |
| EKS | CloudFormation stack waits, node group readiness, capacity type (ON_DEMAND/SPOT) |
| AKS | ARM deployment async, spot instance reclaim, vmSize + priority fields |

The `DistributedTraining` controller remains unchanged — it waits for
`NodePool.Status.Ready = true` regardless of which cloud produced it.

The `NodePool` spec would need to grow to carry provider-specific fields (GKE
uses `MachineType + AcceleratorType`, EKS uses `instanceType + capacityType`,
AKS uses `vmSize + priority`). This can be modelled as a union or as
provider-specific sub-specs similar to how `PytorchSpec` and `SparkSpec` are
handled today.
