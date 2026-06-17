package controller

import (
	"context"
	"fmt"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

const requeueAfter = 20 * time.Second

// ---------------------------------------------------------------------------
// Phase: Pending
// ---------------------------------------------------------------------------

func (r *DistributedTrainingReconciler) reconcilePending(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	log.FromContext(ctx).Info("phase: Pending — validating spec, will create ephemeral node pool")

	if err := validateSpec(job); err != nil {
		return r.setFailed(ctx, job, "invalid spec", err)
	}

	// Warn (non-blocking) when resumeFromCheckpoint is set on a non-pytorch backend.
	backend := job.Spec.Backend
	if backend == "" {
		backend = trainingv1.BackendPyTorch
	}
	if job.Spec.ResumeFromCheckpoint != "" && backend != trainingv1.BackendPyTorch {
		r.Recorder.Event(job, corev1.EventTypeWarning, "InvalidField",
			"spec.resumeFromCheckpoint is only supported for the pytorch backend and will be ignored")
	}

	if job.Status.StartTime == nil {
		patch := client.MergeFrom(job.DeepCopy())
		now := metav1.Now()
		job.Status.StartTime = &now
		if err := r.Status().Patch(ctx, job, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("setting StartTime: %w", err)
		}
	}

	// Run the topology solver when objective mode is active and topology is not yet resolved.
	if job.Spec.Objective != nil && job.Status.ResolvedTopology.Nodes == 0 {
		rt, err := r.runSolver(ctx, job)
		if err != nil {
			return r.setFailed(ctx, job, "topology solver failed", err)
		}
		patch := client.MergeFrom(job.DeepCopy())
		job.Status.ResolvedTopology = rt
		if err := r.Status().Patch(ctx, job, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("writing ResolvedTopology: %w", err)
		}
	}

	return r.setPhase(ctx, job, trainingv1.PhaseProvisioning,
		"spec validated, requesting node pool creation")
}

func validateSpec(job *trainingv1.DistributedTraining) error {
	backend := job.Spec.Backend
	if backend == "" {
		backend = trainingv1.BackendPyTorch
	}

	// Objective mode validation.
	if obj := job.Spec.Objective; obj != nil {
		if obj.TargetTime == "" && obj.MaxCost == "" {
			return fmt.Errorf("spec.objective requires at least one of targetTime or maxCost")
		}
		if obj.MaxNodes < 2 {
			return fmt.Errorf("spec.objective.maxNodes must be ≥ 2")
		}
		// Mutual exclusion: objective vs explicit topology.
		if backend == trainingv1.BackendSpark && job.Spec.Topology.Nodes > 0 {
			return fmt.Errorf("spec.objective and spec.topology.nodes are mutually exclusive — in objective mode the operator determines the topology")
		}
	}

	switch backend {
	case trainingv1.BackendPyTorch:
		ps := job.Spec.PytorchSpec
		if ps == nil {
			return fmt.Errorf("spec.pytorchSpec is required when backend is pytorch")
		}
		if ps.Image == "" {
			return fmt.Errorf("spec.pytorchSpec.image must not be empty")
		}

	case trainingv1.BackendSpark:
		if job.Spec.SparkSpec == nil {
			return fmt.Errorf("spec.sparkSpec is required when backend is spark")
		}
		// In objective mode topology.nodes is not required — the solver provides it.
		if job.Spec.Objective == nil && job.Spec.Topology.Nodes < 1 {
			return fmt.Errorf("spec.topology.nodes must be ≥ 1 for spark backend")
		}

	case trainingv1.BackendJob:
		// plain batch/v1 Job — no extra required fields

	default:
		return fmt.Errorf("unknown backend %q", backend)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Phase: Provisioning
// ---------------------------------------------------------------------------

func (r *DistributedTrainingReconciler) reconcileProvisioning(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	poolName := r.nodePoolName(job)
	cfg := r.nodePoolConfig(job)

	// Objective mode: node count comes from the solver-written ResolvedTopology.
	if job.Spec.Objective != nil {
		if job.Status.ResolvedTopology.Nodes == 0 {
			return r.setFailed(ctx, job,
				"ResolvedTopology not set before Provisioning — solver bug", nil)
		}
		cfg.NodeCount = job.Status.ResolvedTopology.Nodes
	}

	if job.Status.GKEOperationID == "" {
		logger.Info("phase: Provisioning — creating ephemeral node pool",
			"provider", r.Cloud.Name(),
			"pool", poolName,
			"nodes", cfg.NodeCount,
		)
	} else {
		logger.Info("phase: Provisioning — polling node pool creation",
			"provider", r.Cloud.Name(),
			"pool", poolName,
			"operation", job.Status.GKEOperationID,
		)
	}

	if job.Status.ProvisioningStartTime == nil {
		patch := client.MergeFrom(job.DeepCopy())
		now := metav1.Now()
		job.Status.ProvisioningStartTime = &now
		if err := r.Status().Patch(ctx, job, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("setting ProvisioningStartTime: %w", err)
		}
	}

	// If no operation is in flight, fire CreateNodePool and store the operation ID.
	if job.Status.GKEOperationID == "" {
		opID, err := r.Cloud.CreateNodePool(ctx, poolName, cfg)
		if err != nil {
			return r.setFailed(ctx, job, "node pool creation failed", err)
		}
		logger.Info("node pool creation started", "pool", poolName, "operation", opID)
		patch := client.MergeFrom(job.DeepCopy())
		job.Status.GKEOperationID = opID
		return ctrl.Result{RequeueAfter: requeueAfter}, r.Status().Patch(ctx, job, patch)
	}

	// Operation already in flight — poll for completion.
	done, err := r.Cloud.IsOperationDone(ctx, job.Status.GKEOperationID)
	if err != nil {
		// Creation failed — delete the (partially-created) pool before going terminal.
		if _, delErr := r.Cloud.DeleteNodePool(ctx, poolName); delErr != nil {
			logger.Error(delErr, "failed to delete node pool after failed creation", "pool", poolName)
		}
		return r.setFailed(ctx, job, "node pool creation operation failed", err)
	}
	if !done {
		logger.Info("waiting for node pool creation", "operation", job.Status.GKEOperationID)
		return ctrl.Result{RequeueAfter: requeueAfter}, nil
	}

	// Operation complete — count ready nodes. Keep GKEOperationID set until all
	// nodes are Ready so that intermediate requeueues poll IsOperationDone (which
	// returns true immediately) rather than attempting a second CreateNodePool.
	desired := cfg.NodeCount
	ready, err := r.countReadyNodesInPool(ctx, poolName)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("counting ready nodes: %w", err)
	}

	if ready < int(desired) {
		logger.Info("waiting for nodes to become Ready", "ready", ready, "desired", desired)
		patch := client.MergeFrom(job.DeepCopy())
		job.Status.NodePoolSize = int32(ready)
		_ = r.Status().Patch(ctx, job, patch)
		return ctrl.Result{RequeueAfter: requeueAfter}, nil
	}

	provDuration := time.Since(job.Status.ProvisioningStartTime.Time).Round(time.Second).String()
	logger.Info("node pool ready", "nodes", ready, "provisioningTime", provDuration)

	// All nodes ready — clear the operation ID and record results atomically.
	patch := client.MergeFrom(job.DeepCopy())
	job.Status.GKEOperationID = ""
	job.Status.NodePoolSize = int32(ready)
	if job.Status.Results == nil {
		job.Status.Results = &trainingv1.JobResults{}
	}
	job.Status.Results.ProvisioningTime = provDuration
	if err := r.Status().Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, err
	}

	return r.setPhase(ctx, job, trainingv1.PhaseReady,
		fmt.Sprintf("node pool %s has %d ready nodes", poolName, ready))
}

// countReadyNodesInPool lists nodes by the provider's label key and counts
// those in Ready condition.
func (r *DistributedTrainingReconciler) countReadyNodesInPool(
	ctx context.Context,
	poolName string,
) (int, error) {
	nodeList := &unstructured.UnstructuredList{}
	nodeList.SetGroupVersionKind(schema.GroupVersionKind{Version: "v1", Kind: "NodeList"})

	if err := r.List(ctx, nodeList,
		client.MatchingLabels{r.Cloud.NodePoolLabelKey(): poolName},
	); err != nil {
		return 0, err
	}

	ready := 0
	for _, node := range nodeList.Items {
		conditions, _, _ := unstructured.NestedSlice(node.Object, "status", "conditions")
		for _, c := range conditions {
			cmap, ok := c.(map[string]interface{})
			if !ok {
				continue
			}
			if cmap["type"] == "Ready" && cmap["status"] == "True" {
				ready++
			}
		}
	}
	return ready, nil
}

// ---------------------------------------------------------------------------
// Phase: Ready
// ---------------------------------------------------------------------------

// reconcileReady delegates manifest generation to the JobBackend (Strategy
// pattern) and applies the resulting object via server-side apply.
func (r *DistributedTrainingReconciler) reconcileReady(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("phase: Ready — generating job manifest", "backend", job.Spec.Backend)

	b, err := r.getBackend(job)
	if err != nil {
		return r.setFailed(ctx, job, "unsupported backend", err)
	}

	// Guarantee the generated manifest targets the ephemeral pool that was
	// created for this job. If hardware.nodePoolName is empty, fill it in so
	// both pytorch and spark backends use the correct node selector.
	if job.Spec.Hardware.NodePoolName == "" {
		job.Spec.Hardware.NodePoolName = r.nodePoolName(job)
	}

	obj, err := b.GenerateManifest(job)
	if err != nil {
		return r.setFailed(ctx, job, "failed to build job manifest", err)
	}

	if err := r.applyJob(ctx, job, obj); err != nil {
		if apierrors.IsInvalid(err) {
			return r.setFailed(ctx, job, "invalid job manifest", err)
		}
		return ctrl.Result{}, fmt.Errorf("failed to apply job manifest: %w", err)
	}

	patch := client.MergeFrom(job.DeepCopy())
	job.Status.JobName = obj.GetName()
	// Note: TrainingStartTime is NOT set here. It's set in reconcileRunning
	// from the backend's own start-time (e.g. PyTorchJob.status.startTime),
	// so `training_seconds` reflects pod runtime only and excludes image pull
	// and pod scheduling. See backend.JobBackend.GetStartTime.
	if err := r.Status().Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, err
	}

	return r.setPhase(ctx, job, trainingv1.PhaseRunning,
		fmt.Sprintf("%s job %s created", job.Spec.Backend, obj.GetName()))
}

// applyJob sets owner references on the generated object and applies it via
// server-side apply. This logic is backend-agnostic.
func (r *DistributedTrainingReconciler) applyJob(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
	obj *unstructured.Unstructured,
) error {
	obj.SetOwnerReferences([]metav1.OwnerReference{{
		APIVersion:         "training.distributedtraining.io/v1",
		Kind:               "DistributedTraining",
		Name:               job.GetName(),
		UID:                job.GetUID(),
		Controller:         boolPtr(true),
		BlockOwnerDeletion: boolPtr(true),
	}})

	data, err := obj.MarshalJSON()
	if err != nil {
		return fmt.Errorf("marshalling job manifest: %w", err)
	}

	force := true
	return r.Patch(ctx, obj, client.RawPatch(types.ApplyPatchType, data),
		&client.PatchOptions{FieldManager: "distributed-training-operator", Force: &force})
}

// ---------------------------------------------------------------------------
// Phase: Running
// ---------------------------------------------------------------------------

// reconcileRunning delegates status polling to the JobBackend.
func (r *DistributedTrainingReconciler) reconcileRunning(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.V(1).Info("phase: Running — checking job", "backend", job.Spec.Backend, "name", job.Status.JobName)

	b, err := r.getBackend(job)
	if err != nil {
		return r.setFailed(ctx, job, "unsupported backend", err)
	}

	phase, message, err := b.GetPhase(ctx, r.Client, job.Namespace, job.Status.JobName)
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, fmt.Errorf("getting job phase: %w", err)
		}
		// Backend job was deleted externally.
		if job.Spec.RestartPolicy == "OnFailure" {
			logger.Info("backend job missing — recreating (restartPolicy=OnFailure)", "job", job.Status.JobName)
			r.Recorder.Event(job, corev1.EventTypeWarning, "JobMissing",
				fmt.Sprintf("backend job %q not found — recreating per restartPolicy=OnFailure", job.Status.JobName))
			// Reset TrainingStartTime so the new run's duration is measured correctly.
			resetPatch := client.MergeFrom(job.DeepCopy())
			job.Status.TrainingStartTime = nil
			if err := r.Status().Patch(ctx, job, resetPatch); err != nil {
				return ctrl.Result{}, fmt.Errorf("resetting TrainingStartTime for restart: %w", err)
			}
			return r.reconcileReady(ctx, job)
		}
		return r.setFailed(ctx, job,
			fmt.Sprintf("backend job %q was deleted externally (set restartPolicy=OnFailure to auto-recreate)", job.Status.JobName),
			nil)
	}

	// Set TrainingStartTime from the backend's own start time the first time
	// we can read it. This excludes image pull and pod scheduling from
	// `training_seconds`, matching baseline_train.sh's measurement window
	// (Kubeflow PyTorchJob.status.startTime → status.completionTime).
	if job.Status.TrainingStartTime == nil {
		pjStart, startErr := b.GetStartTime(ctx, r.Client, job.Namespace, job.Status.JobName)
		if startErr != nil {
			logger.V(1).Info("could not fetch backend start time, will retry", "err", startErr)
		} else if !pjStart.IsZero() {
			t := metav1.NewTime(pjStart)
			patch := client.MergeFrom(job.DeepCopy())
			job.Status.TrainingStartTime = &t
			if patchErr := r.Status().Patch(ctx, job, patch); patchErr != nil {
				logger.Error(patchErr, "patching TrainingStartTime from backend start time")
			}
		}
	}

	switch phase {
	case "Succeeded":
		return r.setPhase(ctx, job, trainingv1.PhaseCollecting, "job completed — collecting results")
	case "Failed":
		return r.setFailed(ctx, job, "job failed: "+message, nil)
	default:
		logger.V(1).Info("job still running")
		return ctrl.Result{RequeueAfter: requeueAfter}, nil
	}
}

// ---------------------------------------------------------------------------
// Phase: Collecting
// ---------------------------------------------------------------------------

// reconcileCollecting delegates result collection to the JobBackend, then
// scales down the node pool and writes the final status.
func (r *DistributedTrainingReconciler) reconcileCollecting(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("phase: Collecting — reading metrics and tearing down")

	b, err := r.getBackend(job)
	if err != nil {
		logger.Error(err, "could not determine backend for result collection")
		b = nil
	}

	// Compute training duration from the backend's own start/end timestamps
	// so the metric exactly matches baseline_train.sh's
	// PJ.status.startTime → completionTime window. Falls back to time.Now()
	// if the backend doesn't expose an end time (e.g. Spark), and falls back
	// to the operator's own elapsed time if TrainingStartTime wasn't recorded.
	var trainingTimeStr string
	if job.Status.TrainingStartTime != nil {
		endTime := time.Now()
		if b != nil {
			if et, etErr := b.GetEndTime(ctx, r.Client, job.Namespace, job.Status.JobName); etErr == nil && !et.IsZero() {
				endTime = et
			} else if etErr != nil {
				logger.V(1).Info("could not fetch backend end time, falling back to time.Now()", "err", etErr)
			}
		}
		duration := endTime.Sub(job.Status.TrainingStartTime.Time)
		if duration < 0 {
			duration = 0
		}
		trainingTimeStr = duration.Round(time.Second).String()
	}

	// Trigger node-pool deletion in parallel with result collection. The PVC
	// reader pod runs against the NFS server, which lives on a separate node
	// pool — deleting the worker pool does not affect data availability. This
	// removes the collection cycle from billable worker-node time.
	poolName := r.nodePoolName(job)
	if opID, err := r.Cloud.DeleteNodePool(ctx, poolName); err != nil {
		logger.Error(err, "failed to delete node pool", "pool", poolName)
	} else {
		logger.Info("node pool deletion started", "pool", poolName, "operation", opID)
		deletePatch := client.MergeFrom(job.DeepCopy())
		job.Status.GKEOperationID = opID
		_ = r.Status().Patch(ctx, job, deletePatch)
	}

	collectionStart := time.Now()

	var results *trainingv1.JobResults
	if b != nil {
		results, err = b.CollectResults(ctx, r.Client, job)
		if err != nil {
			logger.Error(err, "result collection failed")
		}
	}
	if results == nil {
		results = &trainingv1.JobResults{}
	}

	// Preserve ProvisioningTime recorded in the Provisioning phase.
	if job.Status.Results != nil && results.ProvisioningTime == "" {
		results.ProvisioningTime = job.Status.Results.ProvisioningTime
	}
	results.TrainingTime = trainingTimeStr
	results.CollectionTime = time.Since(collectionStart).Round(time.Second).String()

	// --- Cost tracking ---
	// Collection runs in parallel with node-pool deletion (kicked off above),
	// so collection time is hidden inside the delete window and not billed
	// as training. Cost = (provisioning + training) × nodes × rate / 3600.
	nodeCount := job.Status.NodePoolSize // read before we zero it below
	machineType := r.effectiveMachineType(job)
	if ch, chKnown, costErr := r.getMachineCost(ctx, machineType); costErr != nil {
		logger.Error(costErr, "reading machine cost ConfigMap")
	} else if !chKnown {
		r.Recorder.Event(job, corev1.EventTypeWarning, "CostSkipped",
			fmt.Sprintf("machine type %q not in cost ConfigMap — estimatedCostUSD not recorded", machineType))
	} else {
		var trainSec, provSec float64
		if results.TrainingTime != "" {
			if d, parseErr := time.ParseDuration(results.TrainingTime); parseErr == nil {
				trainSec = d.Seconds()
			}
		}
		if results.ProvisioningTime != "" {
			if d, parseErr := time.ParseDuration(results.ProvisioningTime); parseErr == nil {
				provSec = d.Seconds()
			}
		}
		cost := float64(nodeCount) * ch * (provSec + trainSec) / 3600.0
		results.EstimatedCostUSD = strconv.FormatFloat(cost, 'f', 4, 64)
	}

	// --- Prometheus metrics ---
	ns, name := job.Namespace, job.Name
	if d, err := time.ParseDuration(results.TrainingTime); err == nil {
		trainingSecondsGauge.WithLabelValues(ns, name).Set(d.Seconds())
	}
	if d, err := time.ParseDuration(results.ProvisioningTime); err == nil {
		provisioningSecondsGauge.WithLabelValues(ns, name).Set(d.Seconds())
	}
	if d, err := time.ParseDuration(results.CollectionTime); err == nil {
		collectionSecondsGauge.WithLabelValues(ns, name).Set(d.Seconds())
	}
	if results.EstimatedCostUSD != "" {
		if cost, err := strconv.ParseFloat(results.EstimatedCostUSD, 64); err == nil {
			costUSDActualGauge.WithLabelValues(ns, name).Set(cost)
		}
	}
	nodesProvisionedGauge.WithLabelValues(ns, name).Set(float64(nodeCount))
	for k, v := range results.Metrics {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			trainingMetricGauge.WithLabelValues(ns, name, k).Set(f)
		}
	}

	// --- Write history (all Succeeded runs, pytorch/spark only) ---
	r.writeHistoryEntry(ctx, job, results)

	now := metav1.Now()
	patch := client.MergeFrom(job.DeepCopy())
	job.Status.Results = results
	job.Status.FinishTime = &now
	job.Status.NodePoolSize = 0
	if err := r.Status().Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, err
	}

	return r.setPhase(ctx, job, trainingv1.PhaseSucceeded,
		fmt.Sprintf("job complete — %d metrics collected", len(results.Metrics)))
}

// reconcileSucceeded polls the node pool delete operation until it completes,
// then logs confirmation and goes fully terminal.
func (r *DistributedTrainingReconciler) reconcileSucceeded(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if job.Status.GKEOperationID == "" {
		logger.Info("job is in terminal state", "phase", job.Status.Phase)
		return ctrl.Result{}, nil
	}

	done, err := r.Cloud.IsOperationDone(ctx, job.Status.GKEOperationID)
	if err != nil || !done {
		if err != nil {
			logger.Error(err, "polling node pool deletion operation", "operation", job.Status.GKEOperationID)
		}
		return ctrl.Result{RequeueAfter: requeueAfter}, nil
	}

	logger.Info("node pool deletion confirmed", "pool", r.nodePoolName(job))

	patch := client.MergeFrom(job.DeepCopy())
	job.Status.GKEOperationID = ""
	if err := r.Status().Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("clearing GKEOperationID: %w", err)
	}
	return ctrl.Result{}, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func boolPtr(b bool) *bool { return &b }
