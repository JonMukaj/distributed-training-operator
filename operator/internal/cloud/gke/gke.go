// Package gke implements the cloud.Provider interface for Google Kubernetes Engine.
package gke

import (
	"context"
	"fmt"
	"strings"

	container "cloud.google.com/go/container/apiv1"
	containerpb "cloud.google.com/go/container/apiv1/containerpb"
	"google.golang.org/api/option"

	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
)

// Compile-time assertion that *Provider satisfies cloud.Provider.
var _ cloud.Provider = (*Provider)(nil)

const nodePoolLabelKey = "cloud.google.com/gke-nodepool"

// Provider wraps the GKE Cluster Manager gRPC client.
type Provider struct {
	inner    *container.ClusterManagerClient
	project  string
	location string
	cluster  string
}

// New creates a GKE Provider authenticated via Application Default Credentials.
// When running inside GKE, ADC is provided automatically via the node service account.
func New(project, location, cluster string) (*Provider, error) {
	c, err := container.NewClusterManagerClient(context.Background(),
		option.WithScopes("https://www.googleapis.com/auth/cloud-platform"),
	)
	if err != nil {
		return nil, fmt.Errorf("creating GKE ClusterManagerClient: %w", err)
	}
	return &Provider{
		inner:    c,
		project:  project,
		location: location,
		cluster:  cluster,
	}, nil
}

// Name implements cloud.Provider.
func (p *Provider) Name() string { return "gke" }

// NodePoolLabelKey implements cloud.Provider.
func (p *Provider) NodePoolLabelKey() string { return nodePoolLabelKey }

// CreateNodePool implements cloud.Provider.
// It fires the GKE CreateNodePool API call and returns immediately with an
// operation ID. The caller must poll IsOperationDone to detect completion.
func (p *Provider) CreateNodePool(ctx context.Context, poolName string, cfg cloud.NodePoolConfig) (string, error) {
	nodePool := &containerpb.NodePool{
		Name: poolName,
		Config: &containerpb.NodeConfig{
			MachineType:    cfg.MachineType,
			Labels:         cfg.Labels,
			DiskSizeGb:     cfg.DiskSizeGb,
			ServiceAccount: cfg.NodeServiceAccount,
		},
		InitialNodeCount: cfg.NodeCount,
	}

	nodePool.Config.Taints = []*containerpb.NodeTaint{{
		Key:    "reserved-pool",
		Value:  "true",
		Effect: containerpb.NodeTaint_NO_SCHEDULE,
	}}

	if cfg.AcceleratorType != "" {
		nodePool.Config.Accelerators = []*containerpb.AcceleratorConfig{{
			AcceleratorType:  cfg.AcceleratorType,
			AcceleratorCount: cfg.AcceleratorCount,
		}}
	}

	op, err := p.inner.CreateNodePool(ctx, &containerpb.CreateNodePoolRequest{
		Parent:   p.clusterFQN(),
		NodePool: nodePool,
	})
	if err != nil {
		return "", fmt.Errorf("CreateNodePool %q: %w", poolName, err)
	}
	return op.GetName(), nil
}

// DeleteNodePool implements cloud.Provider.
// It fires the GKE DeleteNodePool API call and returns immediately with an
// operation ID. The caller must poll IsOperationDone to detect completion.
func (p *Provider) DeleteNodePool(ctx context.Context, poolName string) (string, error) {
	op, err := p.inner.DeleteNodePool(ctx, &containerpb.DeleteNodePoolRequest{
		Name: p.nodePoolFQN(poolName),
	})
	if err != nil {
		return "", fmt.Errorf("DeleteNodePool %q: %w", poolName, err)
	}
	return op.GetName(), nil
}

// IsOperationDone implements cloud.Provider.
// It polls the GKE Operations API for the given operation ID.
func (p *Provider) IsOperationDone(ctx context.Context, operationID string) (bool, error) {
	// GetOperation requires the fully-qualified name:
	// projects/{project}/locations/{location}/operations/{id}
	// CreateNodePool returns only the short ID so we construct the full path here.
	name := operationID
	if !strings.HasPrefix(operationID, "projects/") {
		name = fmt.Sprintf("projects/%s/locations/%s/operations/%s", p.project, p.location, operationID)
	}
	op, err := p.inner.GetOperation(ctx, &containerpb.GetOperationRequest{Name: name})
	if err != nil {
		return false, fmt.Errorf("polling operation %q: %w", operationID, err)
	}
	switch op.GetStatus() {
	case containerpb.Operation_DONE:
		if op.GetError() != nil {
			return false, fmt.Errorf("operation failed: %s", op.GetError().GetMessage())
		}
		return true, nil
	case containerpb.Operation_ABORTING:
		return false, fmt.Errorf("operation aborted: %s", op.GetError().GetMessage())
	default:
		return false, nil
	}
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (p *Provider) clusterFQN() string {
	return fmt.Sprintf("projects/%s/locations/%s/clusters/%s",
		p.project, p.location, p.cluster)
}

func (p *Provider) nodePoolFQN(poolName string) string {
	return fmt.Sprintf("%s/nodePools/%s", p.clusterFQN(), poolName)
}
