// Package cloud defines the CloudProvider interface that abstracts all
// cloud-provider-specific operations needed by the operator controller.
//
// To add support for a new provider (e.g. AWS EKS, Azure AKS) implement this
// interface in a new sub-package (e.g. internal/cloud/eks) and register it in
// the factory in cmd/main.go.  The controller itself has no cloud-specific imports.
package cloud

import "context"

// NodePoolConfig is a cloud-agnostic description of a node pool to create.
// Each field maps to an equivalent concept on every major cloud provider.
type NodePoolConfig struct {
	// MachineType is the VM type, e.g. "n2-standard-8" (GKE), "m5.xlarge" (EKS).
	MachineType string

	// NodeCount is the number of nodes to provision.
	NodeCount int32

	// AcceleratorType is the GPU model, e.g. "nvidia-l4". Empty means CPU-only.
	AcceleratorType string

	// AcceleratorCount is the number of GPUs per node. Ignored when AcceleratorType is empty.
	AcceleratorCount int64

	// Labels are key/value pairs applied to every node in the pool.
	// The controller uses these to identify and count ready nodes.
	Labels map[string]string

	// DiskSizeGb is the boot disk size for each node in GB.
	// When 0 the cloud provider uses its default (100 GB on GKE).
	DiskSizeGb int32

	// NodeServiceAccount is the IAM service account email to attach to nodes.
	// When empty the cloud provider uses its default compute service account.
	NodeServiceAccount string
}

// Provider abstracts node-pool lifecycle operations for the operator.
// All mutating operations are non-blocking: they start the operation and
// return an opaque operation ID that the caller must poll with IsOperationDone.
type Provider interface {
	// CreateNodePool creates a new node pool with the given name and config.
	// Returns an operation ID to be polled with IsOperationDone.
	CreateNodePool(ctx context.Context, poolName string, cfg NodePoolConfig) (operationID string, err error)

	// DeleteNodePool deletes the named node pool entirely.
	// Returns an operation ID to be polled with IsOperationDone.
	DeleteNodePool(ctx context.Context, poolName string) (operationID string, err error)

	// IsOperationDone polls a previously started operation.
	// Returns (true, nil) on success, (false, nil) if still in progress,
	// (false, err) on terminal failure.
	IsOperationDone(ctx context.Context, operationID string) (bool, error)

	// NodePoolLabelKey returns the Kubernetes node label key that identifies
	// which node pool a node belongs to.
	//
	// GKE:  "cloud.google.com/gke-nodepool"
	// EKS:  "eks.amazonaws.com/nodegroup"
	// AKS:  "agentpool"
	NodePoolLabelKey() string

	// Name returns a short, human-readable provider identifier used in log
	// messages and metrics, e.g. "gke", "eks", "aks".
	Name() string
}
