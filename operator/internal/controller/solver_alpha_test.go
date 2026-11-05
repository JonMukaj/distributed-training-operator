package controller

import (
	"math"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

// synthHistory builds one history entry per node count from the scaling model
// P(n) = p · n · η(n, α), so estimateAlpha must recover α.
func synthHistory(p, alpha float64, nodes []int32) (map[int32][]trainingv1.DistributedTrainingHistorySpec, []int32) {
	byN := map[int32][]trainingv1.DistributedTrainingHistorySpec{}
	for _, n := range nodes {
		thr := p * float64(n) * eta(float64(n), alpha)
		byN[n] = append(byN[n], trainingv1.DistributedTrainingHistorySpec{
			Nodes:      n,
			Throughput: resource.MustParse(formatFloat(thr)),
		})
	}
	return byN, nodes
}

func formatFloat(f float64) string {
	return resource.NewMilliQuantity(int64(f*1000), resource.DecimalSI).String()
}

// TestEstimateAlphaRecoversLargeAlpha locks the bracket fix: with three node
// counts and a workload that gets SLOWER with more nodes (α ≫ 1), the
// least-squares fit must not be clipped at 1.0.
func TestEstimateAlphaRecoversLargeAlpha(t *testing.T) {
	for _, want := range []float64{0.2, 0.9, 3.0, 14.0} {
		byN, ns := synthHistory(83.0, want, []int32{1, 2, 4})
		got := estimateAlpha(byN, ns, 83.0)
		if math.Abs(got-want) > 0.05*want+0.01 {
			t.Errorf("α=%v: estimateAlpha = %v", want, got)
		}
	}
}

// TestEstimateAlphaTwoPointsUnchanged: the closed-form path never clipped and
// must keep returning the exact value.
func TestEstimateAlphaTwoPointsUnchanged(t *testing.T) {
	byN, ns := synthHistory(83.0, 14.0, []int32{1, 4})
	if got := estimateAlpha(byN, ns, 83.0); math.Abs(got-14.0) > 0.05 {
		t.Errorf("two-point α = %v, want 14", got)
	}
}

// TestEstimateAlphaLinearScaling: perfect scaling still yields ≈ 0 with the
// widened bracket (no spurious drift toward the upper bound).
func TestEstimateAlphaLinearScaling(t *testing.T) {
	byN, ns := synthHistory(83.0, 0.0, []int32{1, 2, 4})
	if got := estimateAlpha(byN, ns, 83.0); got > 0.01 {
		t.Errorf("linear scaling α = %v, want ≈ 0", got)
	}
}
