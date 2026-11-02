package controller

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
)

// stubCloud is the minimal cloud.Provider needed by deleteNodesInPool: only
// the node-pool label key is consulted.
type stubCloud struct{ label string }

func (s *stubCloud) CreateNodePool(context.Context, string, cloud.NodePoolConfig) (string, error) {
	return "", nil
}
func (s *stubCloud) DeleteNodePool(context.Context, string) (string, error) { return "", nil }
func (s *stubCloud) IsOperationDone(context.Context, string) (bool, error)  { return true, nil }
func (s *stubCloud) NodePoolLabelKey() string                               { return s.label }
func (s *stubCloud) Name() string                                           { return "stub" }

// poolLabelKey is the node label the stub provider reports.
const poolLabelKey = "test.example.com/node-group-name"

// makeNode creates a Node carrying the pool label and sets its Ready condition.
func makeNode(ctx context.Context, name, pool string, ready corev1.ConditionStatus) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{
		Name:   name,
		Labels: map[string]string{poolLabelKey: pool},
	}}
	Expect(k8sClient.Create(ctx, node)).To(Succeed())
	node.Status.Conditions = []corev1.NodeCondition{{Type: corev1.NodeReady, Status: ready}}
	Expect(k8sClient.Status().Update(ctx, node)).To(Succeed())
}

var _ = Describe("deleteNodesInPool", func() {
	ctx := context.Background()

	It("removes the pool's non-Ready nodes and keeps Ready ones and other pools", func() {
		makeNode(ctx, "stale-gone-0", "dj-gone", corev1.ConditionUnknown) // VM deleted
		makeNode(ctx, "stale-gone-1", "dj-gone", corev1.ConditionFalse)
		makeNode(ctx, "live-gone-2", "dj-gone", corev1.ConditionTrue) // mislabelled live node
		makeNode(ctx, "other-pool-0", "dj-other", corev1.ConditionFalse)

		r := &DistributedTrainingReconciler{Client: k8sClient, Scheme: k8sClient.Scheme(), Cloud: &stubCloud{label: poolLabelKey}}
		r.deleteNodesInPool(ctx, "dj-gone")

		var n corev1.Node
		for _, name := range []string{"stale-gone-0", "stale-gone-1"} {
			err := k8sClient.Get(ctx, types.NamespacedName{Name: name}, &n)
			Expect(apierrors.IsNotFound(err)).To(BeTrue(), "expected %s to be deleted, got %v", name, err)
		}
		for _, name := range []string{"live-gone-2", "other-pool-0"} {
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name}, &n)).To(Succeed(), "%s must survive", name)
		}

		// Idempotent: a second pass finds nothing to do and does not error.
		r.deleteNodesInPool(ctx, "dj-gone")

		for _, name := range []string{"live-gone-2", "other-pool-0"} {
			Expect(k8sClient.Delete(ctx, &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})).To(Succeed())
		}
	})
})
