package collect

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestNewReaderPodWithoutOptionsHasNoAffinity(t *testing.T) {
	pod := newReaderPod("default", "pvc-reader-1", "out-pvc")
	if pod.Spec.Affinity != nil {
		t.Fatalf("expected no affinity, got %+v", pod.Spec.Affinity)
	}
	if got := pod.Spec.Volumes[0].PersistentVolumeClaim.ClaimName; got != "out-pvc" {
		t.Fatalf("claim = %q, want out-pvc", got)
	}
}

// TestAvoidNodesPinsReaderOffThePool locks the fix for the lost-metrics bug:
// the reader pod must carry a required NotIn anti-affinity on the pool label so
// it is never scheduled onto the node pool being torn down in parallel.
func TestAvoidNodesPinsReaderOffThePool(t *testing.T) {
	pod := newReaderPod("default", "pvc-reader-2", "out-pvc",
		AvoidNodes("hyperstack.cloud/node-group-name", "dj-gpu-cal-1"))

	aff := pod.Spec.Affinity
	if aff == nil || aff.NodeAffinity == nil || aff.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		t.Fatalf("expected required node affinity, got %+v", aff)
	}
	terms := aff.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	if len(terms) != 1 || len(terms[0].MatchExpressions) != 1 {
		t.Fatalf("expected one term with one expression, got %+v", terms)
	}
	expr := terms[0].MatchExpressions[0]
	if expr.Key != "hyperstack.cloud/node-group-name" || expr.Operator != corev1.NodeSelectorOpNotIn ||
		len(expr.Values) != 1 || expr.Values[0] != "dj-gpu-cal-1" {
		t.Fatalf("unexpected expression %+v", expr)
	}
}

func TestAvoidNodesIgnoresEmptyInputs(t *testing.T) {
	for _, c := range [][2]string{{"", "pool"}, {"key", ""}, {"", ""}} {
		pod := newReaderPod("default", "p", "pvc", AvoidNodes(c[0], c[1]))
		if pod.Spec.Affinity != nil {
			t.Errorf("AvoidNodes(%q,%q): expected no affinity", c[0], c[1])
		}
	}
}
