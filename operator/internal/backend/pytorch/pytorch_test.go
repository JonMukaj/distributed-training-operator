package pytorch

import (
	"encoding/json"
	"testing"
	"time"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

// sampleTransferProfile mirrors the JSON written by script.py's
// transfer_profile.json. It locks the script → operator field-name contract.
const sampleTransferProfile = `{
  "model": "EleutherAI/pythia-410m",
  "num_params": 405000000,
  "world_size": 2,
  "synthetic": true,
  "total_training_s": 50.0,
  "num_steps": 100,
  "total_allreduce_s": 12.5,
  "allreduce_fraction": 0.25,
  "dataload_fraction": 0.10,
  "compute_fraction": 0.65,
  "grad_size_bytes": 1620000000,
  "total_allreduce_bytes": 162000000000,
  "nfs_read_bytes": 2048,
  "nfs_write_bytes": 4096,
  "checkpoint_size_bytes": 820000000,
  "allreduce_bandwidth_MBs": 12960.0,
  "nfs_read_bandwidth_MBs": 0.5,
  "nfs_write_bandwidth_MBs": 0.08
}`

func TestTransferProfileUnmarshal(t *testing.T) {
	var tp transferProfile
	if err := json.Unmarshal([]byte(sampleTransferProfile), &tp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tp.AllReduceFraction != 0.25 {
		t.Errorf("AllReduceFraction = %v, want 0.25", tp.AllReduceFraction)
	}
	if tp.DataloadFraction != 0.10 {
		t.Errorf("DataloadFraction = %v, want 0.10", tp.DataloadFraction)
	}
	if tp.GradSizeBytes != 1620000000 {
		t.Errorf("GradSizeBytes = %d, want 1620000000", tp.GradSizeBytes)
	}
	if tp.TotalAllReduceBytes != 162000000000 {
		t.Errorf("TotalAllReduceBytes = %d, want 162000000000", tp.TotalAllReduceBytes)
	}
	if tp.AllReduceBandwidthMBs != 12960.0 {
		t.Errorf("AllReduceBandwidthMBs = %v, want 12960.0", tp.AllReduceBandwidthMBs)
	}
}

func TestFoldTransferMetrics(t *testing.T) {
	var tp transferProfile
	if err := json.Unmarshal([]byte(sampleTransferProfile), &tp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	metrics := make(map[string]string)
	foldTransferMetrics(tp, metrics)

	want := map[string]string{
		"allreduceFraction":     "0.250000",
		"dataloadFraction":      "0.100000",
		"computeFraction":       "0.650000",
		"totalAllreduceSeconds": "12.5000",
		"gradSizeBytes":         "1620000000",
		"totalAllreduceBytes":   "162000000000",
		"nfsReadBytes":          "2048",
		"nfsWriteBytes":         "4096",
		"checkpointSizeBytes":   "820000000",
		"allreduceBandwidthMBs": "12960.0000",
		"nfsReadBandwidthMBs":   "0.5000",
		"nfsWriteBandwidthMBs":  "0.0800",
	}
	for k, v := range want {
		if metrics[k] != v {
			t.Errorf("metrics[%q] = %q, want %q", k, metrics[k], v)
		}
	}
}

// TestBuildNodeSelectorUsesProviderLabel locks the multi-cloud contract: the
// replica nodeSelector must use the cloud provider's node-pool label, not a
// hard-coded GKE key, or pods never schedule on Hyperstack node groups.
func TestBuildNodeSelectorUsesProviderLabel(t *testing.T) {
	job := &trainingv1.DistributedTraining{}
	job.Spec.Hardware.Type = trainingv1.HardwareGPU
	job.Spec.Hardware.NodePoolName = "dj-gpu-cal-1"

	cases := []struct{ key, wantKey string }{
		{"", DefaultNodePoolLabelKey},
		{"hyperstack.cloud/node-group-name", "hyperstack.cloud/node-group-name"},
	}
	for _, c := range cases {
		sel, tol := New(c.key).buildNodeSelector(job)
		if got := sel[c.wantKey]; got != "dj-gpu-cal-1" {
			t.Errorf("New(%q): nodeSelector = %v, want %q=dj-gpu-cal-1", c.key, sel, c.wantKey)
		}
		if len(sel) != 1 {
			t.Errorf("New(%q): nodeSelector has %d keys, want 1", c.key, len(sel))
		}
		if len(tol) != 2 { // reserved-pool + nvidia.com/gpu
			t.Errorf("New(%q): %d tolerations, want 2 for GPU", c.key, len(tol))
		}
	}
}

// TestRunningSince locks the training-start definition: the Running condition's
// transition time, never status.startTime (job creation).
func TestRunningSince(t *testing.T) {
	cond := func(typ, status, ts string) map[string]any {
		return map[string]any{"type": typ, "status": status, "lastTransitionTime": ts}
	}
	got, err := runningSince([]any{
		cond("Created", "True", "2026-11-02T22:29:03Z"),
		cond("Running", "True", "2026-11-02T22:33:10Z"),
	})
	if err != nil || !got.Equal(time.Date(2026, 11, 2, 22, 33, 10, 0, time.UTC)) {
		t.Fatalf("got %v, %v; want 22:33:10Z", got, err)
	}
	// Not running yet → zero time, no error (controller retries).
	got, err = runningSince([]any{cond("Created", "True", "2026-11-02T22:29:03Z")})
	if err != nil || !got.IsZero() {
		t.Fatalf("expected zero time while not Running, got %v, %v", got, err)
	}
	got, err = runningSince([]any{cond("Running", "False", "2026-11-02T22:33:10Z")})
	if err != nil || !got.IsZero() {
		t.Fatalf("Running=False must not count, got %v, %v", got, err)
	}
	if _, err = runningSince([]any{cond("Running", "True", "yesterday")}); err == nil {
		t.Fatalf("expected parse error for a bad timestamp")
	}
}
