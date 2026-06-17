/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

var _ = Describe("DistributedTraining Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}
		distributedtraining := &trainingv1.DistributedTraining{}

		BeforeEach(func() {
			By("creating the custom resource for the Kind DistributedTraining")
			err := k8sClient.Get(ctx, typeNamespacedName, distributedtraining)
			if err != nil && errors.IsNotFound(err) {
				resource := &trainingv1.DistributedTraining{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: trainingv1.DistributedTrainingSpec{
						Backend: trainingv1.BackendPyTorch,
						PytorchSpec: &trainingv1.PytorchSpec{
							Image: "python:3.11-slim",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("500m"),
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
						},
						Topology: trainingv1.TopologySpec{
							Nodes: 1,
						},
					},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			resource := &trainingv1.DistributedTraining{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			// Remove the finalizer before deletion so the API server can garbage-collect
			// the object without needing a running controller in the test environment.
			if controllerutil.ContainsFinalizer(resource, finalizerName) {
				patch := client.MergeFrom(resource.DeepCopy())
				controllerutil.RemoveFinalizer(resource, finalizerName)
				Expect(k8sClient.Patch(ctx, resource, patch)).To(Succeed())
			}

			By("Cleanup the specific resource instance DistributedTraining")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})

		It("should add the cleanup finalizer on the first reconcile", func() {
			By("Reconciling the created resource")
			controllerReconciler := NewDistributedTrainingReconciler(
				k8sClient,
				k8sClient.Scheme(),
				nil, // Recorder — not wired in unit tests
				nil, // Cloud provider — not reached on first reconcile (finalizer path returns early)
				nil, // Backends — not reached on first reconcile
				"e2-standard-4",
				"n1-standard-4",
				50,
				2,
				"machine-costs",
				"distributed-training-system",
				"",
			)

			// The first reconcile only registers the finalizer and requeues; it does
			// not attempt any cloud calls, so a nil provider is safe here.
			result, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Requeue).To(BeTrue(), "expected requeue after finalizer registration")

			By("Verifying the finalizer was registered on the resource")
			updated := &trainingv1.DistributedTraining{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, updated)).To(Succeed())
			Expect(controllerutil.ContainsFinalizer(updated, finalizerName)).To(BeTrue(),
				"expected finalizer %q to be present after first reconcile", finalizerName)
		})

		It("should leave the status phase empty after the finalizer-only reconcile", func() {
			By("Reconciling the created resource")
			controllerReconciler := NewDistributedTrainingReconciler(
				k8sClient,
				k8sClient.Scheme(),
				nil,
				nil,
				nil,
				"e2-standard-4",
				"n1-standard-4",
				50,
				2,
				"machine-costs",
				"distributed-training-system",
				"",
			)

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			By("Verifying status.phase is still empty — phase is set on the second reconcile")
			updated := &trainingv1.DistributedTraining{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, updated)).To(Succeed())
			Expect(updated.Status.Phase).To(BeEmpty(),
				"phase should not be set yet; the first reconcile only registers the finalizer")
		})
	})
})
