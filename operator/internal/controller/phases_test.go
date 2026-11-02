package controller

import "testing"

func nodeObj(ready string, gpu *string) map[string]any {
	status := map[string]any{
		"conditions": []any{
			map[string]any{"type": "MemoryPressure", "status": "False"},
			map[string]any{"type": "Ready", "status": ready},
		},
	}
	if gpu != nil {
		status["allocatable"] = map[string]any{"cpu": "28", "nvidia.com/gpu": *gpu}
	} else {
		status["allocatable"] = map[string]any{"cpu": "28"}
	}
	return map[string]any{"status": status}
}

func strp(s string) *string { return &s }

// TestNodeReadyForJob locks the GPU readiness gate: a kubelet-Ready node is not
// ready for a GPU job until the device plugin advertises nvidia.com/gpu.
func TestNodeReadyForJob(t *testing.T) {
	cases := []struct {
		name       string
		node       map[string]any
		requireGPU bool
		want       bool
	}{
		{"cpu job, node Ready", nodeObj("True", nil), false, true},
		{"cpu job, node NotReady", nodeObj("False", nil), false, false},
		{"gpu job, Ready but no GPU advertised yet", nodeObj("True", nil), true, false},
		{"gpu job, Ready, GPU 0", nodeObj("True", strp("0")), true, false},
		{"gpu job, Ready, GPU 1", nodeObj("True", strp("1")), true, true},
		{"gpu job, Ready, GPU 8", nodeObj("True", strp("8")), true, true},
		{"gpu job, NotReady but GPU 1", nodeObj("Unknown", strp("1")), true, false},
		{"gpu job, unparsable quantity", nodeObj("True", strp("one")), true, false},
		{"no status at all", map[string]any{}, true, false},
	}
	for _, c := range cases {
		if got := nodeReadyForJob(c.node, c.requireGPU); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}
