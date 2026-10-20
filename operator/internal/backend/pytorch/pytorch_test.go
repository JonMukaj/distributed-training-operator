package pytorch

import (
	"encoding/json"
	"testing"
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
