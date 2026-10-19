package hyperstack

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
)

const (
	testClusterID = 1617
	testAPIKey    = "test-key"
)

func TestNewValidation(t *testing.T) {
	if _, err := New(DefaultAPIURL, "", testClusterID, ""); err == nil {
		t.Fatal("expected error for missing API key")
	}
	if _, err := New(DefaultAPIURL, testAPIKey, 0, ""); err == nil {
		t.Fatal("expected error for missing cluster ID")
	}
	p, err := New("", testAPIKey, testClusterID, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.baseURL != DefaultAPIURL {
		t.Errorf("expected default API URL, got %q", p.baseURL)
	}
	if p.NodePoolLabelKey() != DefaultNodePoolLabel {
		t.Errorf("expected default node label, got %q", p.NodePoolLabelKey())
	}
	if p.Name() != "hyperstack" {
		t.Errorf("expected name hyperstack, got %q", p.Name())
	}
}

func TestCreateNodePoolNameTooLong(t *testing.T) {
	p, _ := New(DefaultAPIURL, testAPIKey, testClusterID, "")
	longName := strings.Repeat("a", maxNodeGroupNameLen+1)
	if _, err := p.CreateNodePool(context.Background(), longName, cloud.NodePoolConfig{}); err == nil {
		t.Fatal("expected error for over-long node-group name")
	}
}

func TestCreateNodePool(t *testing.T) {
	var gotPath, gotKey string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("api_key")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"status":true,"node_group":{"id":4501}}`))
	}))
	defer srv.Close()

	p, _ := New(srv.URL, testAPIKey, testClusterID, "")
	opID, err := p.CreateNodePool(context.Background(), "pool-a", cloud.NodePoolConfig{
		MachineType: "n1-cpu-small",
		NodeCount:   3,
	})
	if err != nil {
		t.Fatalf("CreateNodePool: %v", err)
	}
	if opID != "create:4501:3" {
		t.Errorf("unexpected operation ID %q", opID)
	}
	if gotPath != fmt.Sprintf("/core/clusters/%d/node-groups", testClusterID) {
		t.Errorf("unexpected path %q", gotPath)
	}
	if gotKey != testAPIKey {
		t.Errorf("api_key header not set, got %q", gotKey)
	}
	if gotBody["flavor_name"] != "n1-cpu-small" || gotBody["role"] != "worker" {
		t.Errorf("unexpected request body %v", gotBody)
	}
}

func TestIsOperationDoneCreate(t *testing.T) {
	// First poll: nodes still CREATING. Second poll: both ACTIVE.
	var calls int
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		statuses := []string{"CREATING", "CREATING"}
		if n > 1 {
			statuses = []string{"ACTIVE", "ACTIVE"}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"nodes":[
			{"role":"worker","node_group_id":4501,"status":%q},
			{"role":"worker","node_group_id":4501,"status":%q},
			{"role":"master","node_group_id":null,"status":"ACTIVE"}
		]}`, statuses[0], statuses[1])
	}))
	defer srv.Close()

	p, _ := New(srv.URL, testAPIKey, testClusterID, "")

	done, err := p.IsOperationDone(context.Background(), "create:4501:2")
	if err != nil {
		t.Fatalf("IsOperationDone: %v", err)
	}
	if done {
		t.Fatal("expected not done while nodes are CREATING")
	}

	done, err = p.IsOperationDone(context.Background(), "create:4501:2")
	if err != nil {
		t.Fatalf("IsOperationDone: %v", err)
	}
	if !done {
		t.Fatal("expected done once nodes are ACTIVE")
	}
}

func TestDeleteNodePoolAndPoll(t *testing.T) {
	deleted := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == fmt.Sprintf("/core/clusters/%d/nodes", testClusterID):
			// Name -> ID resolution.
			_, _ = w.Write([]byte(`{"nodes":[
				{"role":"worker","node_group_id":4501,"node_group_name":"pool-a","status":"ACTIVE"}
			]}`))
		case r.Method == http.MethodDelete:
			deleted = true
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":true}`))
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/node-groups/"):
			// Deletion poll — group is gone.
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"status":false}`))
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	p, _ := New(srv.URL, testAPIKey, testClusterID, "")
	opID, err := p.DeleteNodePool(context.Background(), "pool-a")
	if err != nil {
		t.Fatalf("DeleteNodePool: %v", err)
	}
	if opID != "delete:4501" {
		t.Errorf("unexpected operation ID %q", opID)
	}
	if !deleted {
		t.Error("DELETE was not issued")
	}

	done, err := p.IsOperationDone(context.Background(), opID)
	if err != nil {
		t.Fatalf("IsOperationDone: %v", err)
	}
	if !done {
		t.Error("expected deletion to report done on 404")
	}
}

func TestDeleteNodePoolAlreadyGone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// No node reports the requested group name.
		_, _ = w.Write([]byte(`{"nodes":[]}`))
	}))
	defer srv.Close()

	p, _ := New(srv.URL, testAPIKey, testClusterID, "")
	opID, err := p.DeleteNodePool(context.Background(), "pool-gone")
	if err != nil {
		t.Fatalf("DeleteNodePool: %v", err)
	}
	if opID != "delete:0" {
		t.Errorf("expected delete:0 for missing group, got %q", opID)
	}
	done, err := p.IsOperationDone(context.Background(), opID)
	if err != nil || !done {
		t.Errorf("expected done=true for delete:0, got done=%v err=%v", done, err)
	}
}

func TestIsOperationDoneMalformed(t *testing.T) {
	p, _ := New(DefaultAPIURL, testAPIKey, testClusterID, "")
	if _, err := p.IsOperationDone(context.Background(), "garbage"); err == nil {
		t.Error("expected error for malformed operation ID")
	}
}
