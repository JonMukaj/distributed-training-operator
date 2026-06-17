// Package controller implements the DistributedTraining reconciliation loop.
package controller

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
	"github.com/JonMukaj/distributed-training-operator/internal/backend"
	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
)

const finalizerName = "training.distributedtraining.io/cleanup"

// DistributedTrainingReconciler reconciles DistributedTraining objects.
type DistributedTrainingReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder

	// Cloud is the provider-agnostic interface for node pool lifecycle.
	// Swap this out to support EKS, AKS, or any future cloud provider.
	Cloud cloud.Provider

	// Backends maps each BackendType to its JobBackend implementation.
	// The reconciler dispatches to the correct backend at runtime based on
	// spec.backend, so all registered backends are always available.
	Backends map[trainingv1.BackendType]backend.JobBackend

	// DefaultCPUMachineType is the VM type used for CPU jobs when
	// spec.hardware.machineType is not set.
	DefaultCPUMachineType string

	// DefaultGPUMachineType is the VM type used for GPU jobs when
	// spec.hardware.machineType is not set.
	DefaultGPUMachineType string

	// DefaultDiskSizeGb is the boot disk size for ephemeral node pool nodes.
	DefaultDiskSizeGb int32

	// DefaultCalibrationNodes is the node count used for the first calibration
	// run when no history exists for a job configuration.
	DefaultCalibrationNodes int32

	// MachineCostsConfigMapName is the name of the ConfigMap containing
	// per-machine-type hourly costs used by the solver and cost tracking.
	MachineCostsConfigMapName string

	// MachineCostsConfigMapNamespace is the namespace of the machine costs ConfigMap.
	MachineCostsConfigMapNamespace string

	// NodeServiceAccount is the IAM service account email attached to operator-created node pools.
	NodeServiceAccount string
}

// NewDistributedTrainingReconciler constructs a reconciler with the given cloud
// provider and job backends.
func NewDistributedTrainingReconciler(
	c client.Client,
	scheme *runtime.Scheme,
	recorder record.EventRecorder,
	provider cloud.Provider,
	backends map[trainingv1.BackendType]backend.JobBackend,
	defaultCPUMachineType, defaultGPUMachineType string,
	defaultDiskSizeGb int32,
	defaultCalibrationNodes int32,
	machineCostsConfigMapName, machineCostsConfigMapNamespace string,
	nodeServiceAccount string,
) *DistributedTrainingReconciler {
	return &DistributedTrainingReconciler{
		Client:                         c,
		Scheme:                         scheme,
		Recorder:                       recorder,
		Cloud:                          provider,
		Backends:                       backends,
		DefaultCPUMachineType:          defaultCPUMachineType,
		DefaultGPUMachineType:          defaultGPUMachineType,
		DefaultDiskSizeGb:              defaultDiskSizeGb,
		DefaultCalibrationNodes:        defaultCalibrationNodes,
		MachineCostsConfigMapName:      machineCostsConfigMapName,
		MachineCostsConfigMapNamespace: machineCostsConfigMapNamespace,
		NodeServiceAccount:             nodeServiceAccount,
	}
}

// getBackend returns the JobBackend for the job's spec.backend, defaulting to
// pytorch. Returns an error if no backend is registered for the requested type.
func (r *DistributedTrainingReconciler) getBackend(job *trainingv1.DistributedTraining) (backend.JobBackend, error) {
	key := job.Spec.Backend
	if key == "" {
		key = trainingv1.BackendPyTorch
	}
	b, ok := r.Backends[key]
	if !ok {
		return nil, fmt.Errorf("no backend registered for %q", key)
	}
	return b, nil
}

// +kubebuilder:rbac:groups=training.distributedtraining.io,resources=distributedtrainings,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=training.distributedtraining.io,resources=distributedtrainings/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=training.distributedtraining.io,resources=distributedtrainings/finalizers,verbs=update
// +kubebuilder:rbac:groups=training.distributedtraining.io,resources=distributedtraininghistories,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=kubeflow.org,resources=pytorchjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=sparkoperator.k8s.io,resources=sparkapplications,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=pods/log,verbs=get
// +kubebuilder:rbac:groups="",resources=pods/exec,verbs=create
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch

// Reconcile is the main entry point called by controller-runtime whenever a
// DistributedTraining object changes or a requeue is requested.
func (r *DistributedTrainingReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	job := &trainingv1.DistributedTraining{}
	if err := r.Get(ctx, req.NamespacedName, job); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("fetching DistributedTraining: %w", err)
	}

	// Handle deletion: delete the node pool before allowing Kubernetes to
	// remove the object. This guarantees no orphaned infrastructure.
	if !job.DeletionTimestamp.IsZero() {
		return r.reconcileDelete(ctx, job)
	}

	// Ensure the cleanup finalizer is registered before doing any real work.
	if !controllerutil.ContainsFinalizer(job, finalizerName) {
		patch := client.MergeFrom(job.DeepCopy())
		controllerutil.AddFinalizer(job, finalizerName)
		if err := r.Patch(ctx, job, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("adding finalizer: %w", err)
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Downgrade to V(1) during Running so poll noise doesn't flood logs.
	// All other phases log at Info so phase transitions remain visible.
	if job.Status.Phase == trainingv1.PhaseRunning {
		logger.V(1).Info("reconciling",
			"phase", job.Status.Phase,
			"name", job.Name,
			"backend", job.Spec.Backend,
			"provider", r.Cloud.Name(),
		)
	} else {
		logger.Info("reconciling",
			"phase", job.Status.Phase,
			"name", job.Name,
			"backend", job.Spec.Backend,
			"provider", r.Cloud.Name(),
		)
	}

	switch job.Status.Phase {
	case "", trainingv1.PhasePending:
		return r.reconcilePending(ctx, job)
	case trainingv1.PhaseProvisioning:
		return r.reconcileProvisioning(ctx, job)
	case trainingv1.PhaseReady:
		return r.reconcileReady(ctx, job)
	case trainingv1.PhaseRunning:
		return r.reconcileRunning(ctx, job)
	case trainingv1.PhaseCollecting:
		return r.reconcileCollecting(ctx, job)
	case trainingv1.PhaseSucceeded:
		return r.reconcileSucceeded(ctx, job)
	case trainingv1.PhaseFailed:
		logger.Info("job is in terminal state", "phase", job.Status.Phase)
		return ctrl.Result{}, nil
	default:
		logger.Info("unknown phase, resetting to Pending", "phase", job.Status.Phase)
		return r.setPhase(ctx, job, trainingv1.PhasePending, "unknown phase, resetting")
	}
}

// reconcileDelete deletes the ephemeral node pool and removes the finalizer,
// allowing Kubernetes to complete the deletion of the CR.
func (r *DistributedTrainingReconciler) reconcileDelete(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	poolName := r.nodePoolName(job)
	// If the job already succeeded, the node pool was deleted in the Collecting
	// phase — skip the GKE call entirely.
	if job.Status.Phase != trainingv1.PhaseSucceeded {
		logger.Info("deletion requested — deleting node pool", "job", job.Name, "pool", poolName)
		if _, err := r.Cloud.DeleteNodePool(ctx, poolName); err != nil {
			switch status.Code(err) {
			case codes.NotFound:
				// Pool already gone — safe to proceed with finalizer removal.
			case codes.FailedPrecondition:
				// Pool exists but is not in a deletable state — typically a
				// create operation is still in flight. Keep the finalizer,
				// requeue, and retry once the pool reaches RUNNING.
				logger.Info("delete deferred — pool has an operation in progress, retrying",
					"pool", poolName)
				return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
			default:
				logger.Error(err, "failed to delete node pool during CR deletion", "pool", poolName)
				return ctrl.Result{}, err
			}
		}
	} else {
		logger.Info("deletion requested — node pool already deleted (job succeeded)", "job", job.Name, "pool", poolName)
	}

	// Patch only metadata.finalizers so the API server does not re-validate
	// the spec (which may fail if e.g. spec.topology.nodes is 0).
	patch := client.MergeFrom(job.DeepCopy())
	controllerutil.RemoveFinalizer(job, finalizerName)
	if err := r.Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("removing finalizer: %w", err)
	}
	return ctrl.Result{}, nil
}

// SetupWithManager registers the controller with the manager.
func (r *DistributedTrainingReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&trainingv1.DistributedTraining{}).
		WithEventFilter(predicate.GenerationChangedPredicate{}).
		Complete(r)
}

// ---------------------------------------------------------------------------
// Helpers shared across phase handlers
// ---------------------------------------------------------------------------

func (r *DistributedTrainingReconciler) setPhase(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
	phase trainingv1.Phase,
	message string,
) (ctrl.Result, error) {
	patch := client.MergeFrom(job.DeepCopy())
	job.Status.Phase = phase
	job.Status.Message = message

	if err := r.Status().Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("patching status phase=%s: %w", phase, err)
	}
	phaseTransitionsTotal.WithLabelValues(job.Namespace, job.Name, string(phase)).Inc()
	r.Recorder.Event(job, corev1.EventTypeNormal, string(phase), message)
	return ctrl.Result{RequeueAfter: requeueAfter}, nil
}

func (r *DistributedTrainingReconciler) setFailed(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
	reason string,
	cause error,
) (ctrl.Result, error) {
	msg := reason
	if cause != nil {
		msg = fmt.Sprintf("%s: %v", reason, cause)
	}
	r.Recorder.Event(job, corev1.EventTypeWarning, "Failed", msg)

	patch := client.MergeFrom(job.DeepCopy())
	job.Status.Phase = trainingv1.PhaseFailed
	job.Status.Message = msg

	if err := r.Status().Patch(ctx, job, patch); err != nil {
		return ctrl.Result{}, fmt.Errorf("patching Failed status: %w", err)
	}
	return ctrl.Result{}, nil
}

// nodePoolName derives the ephemeral node pool name for a job.
// GKE pool names must be ≤ 40 characters.
func (r *DistributedTrainingReconciler) nodePoolName(job *trainingv1.DistributedTraining) string {
	if job.Spec.Hardware.NodePoolName != "" {
		return job.Spec.Hardware.NodePoolName
	}
	name := "dj-" + job.Name
	if len(name) > 40 {
		name = name[:40]
	}
	return name
}

// nodePoolConfig builds a cloud-agnostic NodePoolConfig from the job spec.
// For pytorch backend, hardware type and node count are inferred from pytorchSpec
// when spec.hardware is not explicitly set.
func (r *DistributedTrainingReconciler) nodePoolConfig(job *trainingv1.DistributedTraining) cloud.NodePoolConfig {
	machineType := job.Spec.Hardware.MachineType
	hwType := job.Spec.Hardware.Type

	// For pytorch: infer hardware type and node count from pytorchSpec when not set.
	nodeCount := job.Spec.Topology.Nodes
	if (job.Spec.Backend == trainingv1.BackendPyTorch || job.Spec.Backend == "") && job.Spec.PytorchSpec != nil {
		ps := job.Spec.PytorchSpec
		if hwType == "" {
			hwType = inferHardwareType(ps, job.Spec.Hardware)
		}
	}
	if nodeCount == 0 {
		nodeCount = 1
	}

	if machineType == "" {
		if hwType == trainingv1.HardwareGPU {
			machineType = r.DefaultGPUMachineType
		} else {
			machineType = r.DefaultCPUMachineType
		}
	}

	diskSizeGb := r.DefaultDiskSizeGb
	if job.Spec.Hardware.DiskSizeGb > 0 {
		diskSizeGb = job.Spec.Hardware.DiskSizeGb
	}

	cfg := cloud.NodePoolConfig{
		MachineType:        machineType,
		NodeCount:          nodeCount,
		DiskSizeGb:         diskSizeGb,
		NodeServiceAccount: r.NodeServiceAccount,
		Labels: map[string]string{
			"managed-by": "distributed-training-operator",
			"job-name":   job.Name,
			"namespace":  job.Namespace,
		},
	}

	if hwType == trainingv1.HardwareGPU && job.Spec.Hardware.GPUType != "" {
		cfg.AcceleratorType = job.Spec.Hardware.GPUType
		cfg.AcceleratorCount = int64(job.Spec.Hardware.GPUCount)
	}

	return cfg
}

// inferHardwareType inspects hardware.type and pytorchSpec resource limits to determine hardware type.
func inferHardwareType(ps *trainingv1.PytorchSpec, hw ...trainingv1.HardwareSpec) trainingv1.HardwareType {
	if len(hw) > 0 && hw[0].Type == trainingv1.HardwareGPU {
		return trainingv1.HardwareGPU
	}
	if ps == nil {
		return trainingv1.HardwareCPU
	}
	for k := range ps.Resources.Limits {
		if string(k) == "nvidia.com/gpu" {
			return trainingv1.HardwareGPU
		}
	}
	return trainingv1.HardwareCPU
}

// effectiveMachineType returns the resolved machine type for the job.
// Mirrors the logic in nodePoolConfig so that history and cost code uses the
// same machine type that was actually provisioned.
func (r *DistributedTrainingReconciler) effectiveMachineType(job *trainingv1.DistributedTraining) string {
	if job.Spec.Hardware.MachineType != "" {
		return job.Spec.Hardware.MachineType
	}
	hwType := job.Spec.Hardware.Type
	if (job.Spec.Backend == trainingv1.BackendPyTorch || job.Spec.Backend == "") && job.Spec.PytorchSpec != nil {
		if hwType == "" {
			hwType = inferHardwareType(job.Spec.PytorchSpec, job.Spec.Hardware)
		}
	}
	if hwType == trainingv1.HardwareGPU {
		return r.DefaultGPUMachineType
	}
	return r.DefaultCPUMachineType
}
