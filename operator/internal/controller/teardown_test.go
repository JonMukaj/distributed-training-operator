package controller

import (
	"context"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
)

// recordingCloud is a cloud.Provider whose operation outcomes the test controls
// and whose calls it records.
type recordingCloud struct {
	done        map[string]bool // operation ID → IsOperationDone result
	deleteCalls []string
}

func (c *recordingCloud) CreateNodePool(context.Context, string, cloud.NodePoolConfig) (string, error) {
	return "create:1", nil
}
func (c *recordingCloud) DeleteNodePool(_ context.Context, pool string) (string, error) {
	c.deleteCalls = append(c.deleteCalls, pool)
	return "delete:" + pool, nil
}
func (c *recordingCloud) IsOperationDone(_ context.Context, op string) (bool, error) {
	d, ok := c.done[op]
	if !ok {
		return false, fmt.Errorf("unknown operation %q", op)
	}
	return d, nil
}
func (c *recordingCloud) NodePoolLabelKey() string { return "test.example.com/node-group-name" }
func (c *recordingCloud) Name() string             { return "recording" }

// TestDeleteDuringProvisioning: deleting a CR while its node pool is still
// being CREATED must wait for the create, then DELETE the pool, then wait for
// the delete, then drop the finalizer. Regression for the bug where the CREATE
// operation in status.gkeOperationID was polled as if it were a deletion and the
// finalizer was removed without ever deleting the pool.
var _ = Describe("reconcileDelete during Provisioning", func() {
	ctx := context.Background()

	It("waits for the create, deletes the pool, waits for the delete, then removes the finalizer", func() {
		name := types.NamespacedName{Name: "dj-teardown-test", Namespace: "default"}
		job := &trainingv1.DistributedTraining{
			ObjectMeta: metav1.ObjectMeta{Name: name.Name, Namespace: name.Namespace, Finalizers: []string{finalizerName}},
			Spec: trainingv1.DistributedTrainingSpec{
				Backend:     trainingv1.BackendPyTorch,
				Hardware:    trainingv1.HardwareSpec{Type: trainingv1.HardwareGPU, NodePoolName: "dj-cal-x"},
				Topology:    trainingv1.TopologySpec{Nodes: 2, ProcessesPerNode: 1},
				PytorchSpec: &trainingv1.PytorchSpec{Image: "img"},
				Training:    &trainingv1.TrainingSpec{BatchSize: 4, LearningRate: "2e-5", Epochs: 1},
			},
		}
		Expect(k8sClient.Create(ctx, job)).To(Succeed())
		now := metav1.Now()
		job.Status.Phase = trainingv1.PhaseProvisioning
		job.Status.GKEOperationID = "create:1509:2"
		job.Status.ProvisioningStartTime = &now
		Expect(k8sClient.Status().Update(ctx, job)).To(Succeed())
		// Mark for deletion: the finalizer keeps the object around.
		Expect(k8sClient.Delete(ctx, job)).To(Succeed())
		Expect(k8sClient.Get(ctx, name, job)).To(Succeed())
		Expect(job.DeletionTimestamp).NotTo(BeNil())

		cloudStub := &recordingCloud{done: map[string]bool{"create:1509:2": false}}
		r := &DistributedTrainingReconciler{Client: k8sClient, Scheme: k8sClient.Scheme(), Cloud: cloudStub}

		// 1. Create still in flight → wait, no delete call, finalizer kept.
		res, err := r.reconcileDelete(ctx, job)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.RequeueAfter).To(BeNumerically(">", 0))
		Expect(cloudStub.deleteCalls).To(BeEmpty())
		Expect(k8sClient.Get(ctx, name, job)).To(Succeed())
		Expect(controllerutil.ContainsFinalizer(job, finalizerName)).To(BeTrue())

		// 2. Create finished → DeleteNodePool called, teardown op recorded, finalizer kept.
		cloudStub.done["create:1509:2"] = true
		cloudStub.done["delete:dj-cal-x"] = false
		res, err = r.reconcileDelete(ctx, job)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.RequeueAfter).To(BeNumerically(">", 0))
		Expect(cloudStub.deleteCalls).To(Equal([]string{"dj-cal-x"}))
		Expect(k8sClient.Get(ctx, name, job)).To(Succeed())
		Expect(job.Status.TeardownOperationID).To(Equal("delete:dj-cal-x"))
		Expect(controllerutil.ContainsFinalizer(job, finalizerName)).To(BeTrue())

		// 3. Delete still in flight → wait, no second delete call.
		res, err = r.reconcileDelete(ctx, job)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.RequeueAfter).To(BeNumerically(">", 0))
		Expect(cloudStub.deleteCalls).To(HaveLen(1))

		// 4. Delete confirmed → finalizer removed, object gone.
		cloudStub.done["delete:dj-cal-x"] = true
		_, err = r.reconcileDelete(ctx, job)
		Expect(err).NotTo(HaveOccurred())
		err = k8sClient.Get(ctx, name, job)
		Expect(err).To(HaveOccurred(), "object should be gone once the finalizer is removed")
	})

	It("removes the finalizer immediately when the pool was already deleted and confirmed (Succeeded)", func() {
		name := types.NamespacedName{Name: "dj-teardown-done", Namespace: "default"}
		job := &trainingv1.DistributedTraining{
			ObjectMeta: metav1.ObjectMeta{Name: name.Name, Namespace: name.Namespace, Finalizers: []string{finalizerName}},
			Spec: trainingv1.DistributedTrainingSpec{
				Backend:     trainingv1.BackendPyTorch,
				Hardware:    trainingv1.HardwareSpec{NodePoolName: "dj-done"},
				Topology:    trainingv1.TopologySpec{Nodes: 1, ProcessesPerNode: 1},
				PytorchSpec: &trainingv1.PytorchSpec{Image: "img"},
				Training:    &trainingv1.TrainingSpec{BatchSize: 4, LearningRate: "2e-5", Epochs: 1},
			},
		}
		Expect(k8sClient.Create(ctx, job)).To(Succeed())
		job.Status.Phase = trainingv1.PhaseSucceeded
		Expect(k8sClient.Status().Update(ctx, job)).To(Succeed())
		Expect(k8sClient.Delete(ctx, job)).To(Succeed())
		Expect(k8sClient.Get(ctx, name, job)).To(Succeed())

		cloudStub := &recordingCloud{done: map[string]bool{}}
		r := &DistributedTrainingReconciler{Client: k8sClient, Scheme: k8sClient.Scheme(), Cloud: cloudStub}
		_, err := r.reconcileDelete(ctx, job)
		Expect(err).NotTo(HaveOccurred())
		Expect(cloudStub.deleteCalls).To(BeEmpty())
		Expect(k8sClient.Get(ctx, name, job)).To(HaveOccurred())
	})
})
