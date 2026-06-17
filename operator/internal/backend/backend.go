// Package backend defines the JobBackend interface that abstracts all
// workload-specific operations (manifest generation, status polling, result
// collection) needed by the operator controller.
//
// To add a new backend (e.g. RayJob) implement this interface in a new
// sub-package (e.g. internal/backend/ray) and register it in the factory
// in cmd/main.go. The controller itself has no backend-specific imports.
package backend

import (
	"context"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

// JobBackend abstracts the workload-specific operations the operator needs
// from the underlying distributed-training framework (PyTorchJob, SparkApplication, …).
type JobBackend interface {
	// GenerateManifest builds the backend-specific Kubernetes job object
	// (e.g. PyTorchJob, SparkApplication) from the DistributedTraining spec.
	GenerateManifest(job *trainingv1.DistributedTraining) (*unstructured.Unstructured, error)

	// JobName returns the deterministic name for the generated job object.
	// Must be stable across repeated calls for the same CR.
	JobName(job *trainingv1.DistributedTraining) string

	// GetPhase inspects the backend job object and returns the current phase
	// as one of "Succeeded", "Failed", or "Running", plus a human-readable message.
	GetPhase(ctx context.Context, c client.Client, namespace, jobName string) (phase string, message string, err error)

	// GetStartTime returns the wall-clock time at which the backend considers
	// the job to have started (e.g. PyTorchJob.status.startTime — set when all
	// replica pods reach Running). The controller uses this for TrainingStartTime
	// so that `training_seconds` excludes image pull and pod scheduling.
	//
	// Implementations may return a zero time if the start time is not yet
	// populated; the controller treats that as "retry later" and keeps polling.
	// Backends that don't expose a meaningful start time may return time.Now()
	// to fall back to an operator-perspective measurement.
	GetStartTime(ctx context.Context, c client.Client, namespace, jobName string) (time.Time, error)

	// GetEndTime returns the wall-clock time at which the backend considers
	// the job to have finished (e.g. PyTorchJob.status.completionTime). The
	// controller uses this as the right endpoint of `training_seconds` so the
	// metric matches baseline_train.sh's PJ.status.startTime → completionTime
	// window exactly (no operator reconcile-lag inflation).
	//
	// Implementations may return a zero time if not yet populated; the
	// controller falls back to its own reconcile time in that case.
	GetEndTime(ctx context.Context, c client.Client, namespace, jobName string) (time.Time, error)

	// CollectResults reads job metrics after completion and returns them as a
	// JobResults struct. The controller fills in TrainingTime and
	// ProvisioningTime; backends are responsible for populating Metrics.
	CollectResults(ctx context.Context, c client.Client, job *trainingv1.DistributedTraining) (*trainingv1.JobResults, error)
}
