// Package hyperstack implements the cloud.Provider interface for Hyperstack
// (NexGen Cloud) managed Kubernetes clusters.
//
// Unlike GKE, Hyperstack has no long-running "operation" resource: creating or
// deleting a node group is a synchronous REST call and readiness is observed by
// polling node status. To fit the operation-polling model the controller
// expects, the opaque operation ID returned by CreateNodePool / DeleteNodePool
// encodes the intent and the node-group ID:
//
//	create:<node_group_id>:<desired_count>
//	delete:<node_group_id>
//
// IsOperationDone parses that string and polls the matching Hyperstack endpoint.
package hyperstack

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
)

// Compile-time assertion that *Provider satisfies cloud.Provider.
var _ cloud.Provider = (*Provider)(nil)

const (
	// DefaultAPIURL is the Hyperstack core API base URL.
	DefaultAPIURL = "https://infrahub-api.nexgencloud.com/v1"

	// DefaultNodePoolLabel is the Kubernetes node label assumed to carry the
	// node-group name. Hyperstack's docs do not specify this label, so it is
	// overridable via --hyperstack-nodepool-label and must be verified against
	// a live cluster.
	DefaultNodePoolLabel = "hyperstack.cloud/node-group"

	// maxNodeGroupNameLen is Hyperstack's hard limit on node-group names.
	maxNodeGroupNameLen = 20

	roleWorker   = "worker"
	statusActive = "ACTIVE"
)

// Provider talks to the Hyperstack core REST API over HTTP.
type Provider struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
	clusterID  int
	nodeLabel  string
}

// New creates a Hyperstack Provider. apiKey authenticates every request via the
// "api_key" header; clusterID identifies the managed cluster whose node groups
// this operator manages. nodePoolLabel is the Kubernetes node label the
// controller uses to count ready nodes per pool; pass "" for the default.
func New(apiURL, apiKey string, clusterID int, nodePoolLabel string) (*Provider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("hyperstack: API key is required (set HYPERSTACK_API_KEY)")
	}
	if clusterID == 0 {
		return nil, fmt.Errorf("hyperstack: --hyperstack-cluster-id is required")
	}
	if apiURL == "" {
		apiURL = DefaultAPIURL
	}
	if nodePoolLabel == "" {
		nodePoolLabel = DefaultNodePoolLabel
	}
	return &Provider{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		baseURL:    strings.TrimRight(apiURL, "/"),
		apiKey:     apiKey,
		clusterID:  clusterID,
		nodeLabel:  nodePoolLabel,
	}, nil
}

// Name implements cloud.Provider.
func (p *Provider) Name() string { return "hyperstack" }

// NodePoolLabelKey implements cloud.Provider.
func (p *Provider) NodePoolLabelKey() string { return p.nodeLabel }

// CreateNodePool implements cloud.Provider. It POSTs a new node group and
// returns an operation ID of the form "create:<node_group_id>:<count>".
func (p *Provider) CreateNodePool(ctx context.Context, poolName string, cfg cloud.NodePoolConfig) (string, error) {
	if len(poolName) > maxNodeGroupNameLen {
		return "", fmt.Errorf("hyperstack: node-group name %q exceeds %d-character limit", poolName, maxNodeGroupNameLen)
	}
	// Hyperstack bundles GPU into the flavor, so cfg.AcceleratorType/Count,
	// cfg.DiskSizeGb, cfg.Labels and cfg.NodeServiceAccount have no equivalent
	// in the create-node-group API and are intentionally ignored.
	body := map[string]any{
		"name":        poolName,
		"role":        roleWorker,
		"flavor_name": cfg.MachineType,
		"count":       cfg.NodeCount,
	}

	var resp struct {
		NodeGroup struct {
			ID int `json:"id"`
		} `json:"node_group"`
	}
	status, err := p.do(ctx, http.MethodPost,
		fmt.Sprintf("/core/clusters/%d/node-groups", p.clusterID), body, &resp)
	if err != nil {
		return "", fmt.Errorf("CreateNodePool %q: %w", poolName, err)
	}
	if status != http.StatusCreated && status != http.StatusOK {
		return "", fmt.Errorf("CreateNodePool %q: unexpected status %d", poolName, status)
	}
	if resp.NodeGroup.ID == 0 {
		return "", fmt.Errorf("CreateNodePool %q: API returned no node-group ID", poolName)
	}
	return fmt.Sprintf("create:%d:%d", resp.NodeGroup.ID, cfg.NodeCount), nil
}

// DeleteNodePool implements cloud.Provider. It resolves the node group's numeric
// ID from its name, issues the DELETE, and returns "delete:<node_group_id>".
func (p *Provider) DeleteNodePool(ctx context.Context, poolName string) (string, error) {
	id, err := p.resolveNodeGroupID(ctx, poolName)
	if err != nil {
		return "", fmt.Errorf("DeleteNodePool %q: %w", poolName, err)
	}
	if id == 0 {
		// Node group not found — already deleted. Signal "done" to the poller.
		return "delete:0", nil
	}

	status, err := p.do(ctx, http.MethodDelete,
		fmt.Sprintf("/core/clusters/%d/node-groups/%d", p.clusterID, id), nil, nil)
	if err != nil {
		return "", fmt.Errorf("DeleteNodePool %q: %w", poolName, err)
	}
	// 404 means the group is already gone, which is success for a delete.
	if status != http.StatusOK && status != http.StatusNoContent && status != http.StatusNotFound {
		return "", fmt.Errorf("DeleteNodePool %q: unexpected status %d", poolName, status)
	}
	return fmt.Sprintf("delete:%d", id), nil
}

// IsOperationDone implements cloud.Provider by parsing the encoded operation ID
// and polling the appropriate endpoint.
func (p *Provider) IsOperationDone(ctx context.Context, operationID string) (bool, error) {
	parts := strings.Split(operationID, ":")
	if len(parts) < 2 {
		return false, fmt.Errorf("hyperstack: malformed operation ID %q", operationID)
	}
	kind := parts[0]
	id, err := strconv.Atoi(parts[1])
	if err != nil {
		return false, fmt.Errorf("hyperstack: malformed operation ID %q: %w", operationID, err)
	}

	switch kind {
	case "create":
		if len(parts) < 3 {
			return false, fmt.Errorf("hyperstack: create operation ID %q missing count", operationID)
		}
		want, err := strconv.Atoi(parts[2])
		if err != nil {
			return false, fmt.Errorf("hyperstack: malformed operation ID %q: %w", operationID, err)
		}
		return p.nodeGroupReady(ctx, id, want)
	case "delete":
		if id == 0 {
			return true, nil // nothing to delete
		}
		return p.nodeGroupGone(ctx, id)
	default:
		return false, fmt.Errorf("hyperstack: unknown operation kind %q", kind)
	}
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// nodeGroupReady reports whether at least `want` worker nodes in the given node
// group have reached ACTIVE status.
func (p *Provider) nodeGroupReady(ctx context.Context, nodeGroupID, want int) (bool, error) {
	nodes, err := p.listNodes(ctx)
	if err != nil {
		return false, err
	}
	active := 0
	for _, n := range nodes {
		if n.Role == roleWorker && n.NodeGroupID != nil && *n.NodeGroupID == nodeGroupID && n.Status == statusActive {
			active++
		}
	}
	return active >= want, nil
}

// nodeGroupGone reports whether the given node group no longer exists.
func (p *Provider) nodeGroupGone(ctx context.Context, nodeGroupID int) (bool, error) {
	status, err := p.do(ctx, http.MethodGet,
		fmt.Sprintf("/core/clusters/%d/node-groups/%d", p.clusterID, nodeGroupID), nil, nil)
	if err != nil {
		return false, err
	}
	if status == http.StatusNotFound {
		return true, nil
	}
	if status == http.StatusOK {
		return false, nil
	}
	return false, fmt.Errorf("polling node-group %d deletion: unexpected status %d", nodeGroupID, status)
}

// resolveNodeGroupID looks up a node group's numeric ID by its name via the
// cluster nodes list. Returns 0 when no node currently reports that group name
// (i.e. the group does not exist or has no nodes).
func (p *Provider) resolveNodeGroupID(ctx context.Context, name string) (int, error) {
	nodes, err := p.listNodes(ctx)
	if err != nil {
		return 0, err
	}
	for _, n := range nodes {
		if n.NodeGroupName != nil && *n.NodeGroupName == name && n.NodeGroupID != nil {
			return *n.NodeGroupID, nil
		}
	}
	return 0, nil
}

type clusterNode struct {
	Role          string  `json:"role"`
	NodeGroupID   *int    `json:"node_group_id"`
	NodeGroupName *string `json:"node_group_name"`
	Status        string  `json:"status"`
}

// listNodes returns all nodes currently in the cluster.
func (p *Provider) listNodes(ctx context.Context) ([]clusterNode, error) {
	var resp struct {
		Nodes []clusterNode `json:"nodes"`
	}
	status, err := p.do(ctx, http.MethodGet,
		fmt.Sprintf("/core/clusters/%d/nodes", p.clusterID), nil, &resp)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("listing cluster nodes: unexpected status %d", status)
	}
	return resp.Nodes, nil
}

// do performs an HTTP request against the Hyperstack API. When out is non-nil
// and the response body is JSON, it is decoded into out. It returns the HTTP
// status code so callers can distinguish 404 (not found) from hard errors.
func (p *Provider) do(ctx context.Context, method, path string, body, out any) (int, error) {
	var reqBody io.Reader
	if body != nil {
		buf, err := json.Marshal(body)
		if err != nil {
			return 0, fmt.Errorf("encoding request body: %w", err)
		}
		reqBody = bytes.NewReader(buf)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.baseURL+path, reqBody)
	if err != nil {
		return 0, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("api_key", p.apiKey)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("%s %s: %w", method, path, err)
	}
	defer func() { _ = resp.Body.Close() }()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, fmt.Errorf("%s %s: reading response: %w", method, path, err)
	}

	// Surface API-level error messages on non-2xx (other than 404, which callers
	// handle explicitly).
	if resp.StatusCode >= 400 && resp.StatusCode != http.StatusNotFound {
		return resp.StatusCode, fmt.Errorf("%s %s: status %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	if out != nil && len(data) > 0 {
		if err := json.Unmarshal(data, out); err != nil {
			return resp.StatusCode, fmt.Errorf("%s %s: decoding response: %w", method, path, err)
		}
	}
	return resp.StatusCode, nil
}
