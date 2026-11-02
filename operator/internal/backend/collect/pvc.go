package collect

import (
	"bytes"
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"
	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	readerPodImage    = "busybox:1.36"
	readerPodTimeout  = 2 * time.Minute
	readerPodInterval = 3 * time.Second
)

// ReaderOption customises the short-lived reader pod.
type ReaderOption func(*corev1.Pod)

// AvoidNodes keeps the reader pod off every node carrying labelKey=value.
//
// The controller deletes a job's ephemeral node pool in parallel with result
// collection. Without this constraint the scheduler happily places the reader
// pod on one of those nodes — it is Ready and idle — and the exec into it
// times out when the VM disappears ("error dialing backend ... i/o timeout"),
// so the job ends with no metrics and no history entry. Expressed as a
// required node anti-affinity on the provider's pool label, which exists on
// every cloud the operator supports.
func AvoidNodes(labelKey, value string) ReaderOption {
	return func(pod *corev1.Pod) {
		if labelKey == "" || value == "" {
			return
		}
		pod.Spec.Affinity = &corev1.Affinity{
			NodeAffinity: &corev1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
					NodeSelectorTerms: []corev1.NodeSelectorTerm{{
						MatchExpressions: []corev1.NodeSelectorRequirement{{
							Key:      labelKey,
							Operator: corev1.NodeSelectorOpNotIn,
							Values:   []string{value},
						}},
					}},
				},
			},
		}
	}
}

// newReaderPod builds the reader pod spec. Kept separate so placement rules
// can be unit-tested without a cluster.
func newReaderPod(namespace, podName, pvcName string, opts ...ReaderOption) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name:    "reader",
				Image:   readerPodImage,
				Command: []string{"sleep", "infinity"},
				VolumeMounts: []corev1.VolumeMount{{
					Name:      "pvc",
					MountPath: "/mnt/pvc",
					ReadOnly:  true,
				}},
			}},
			Volumes: []corev1.Volume{{
				Name: "pvc",
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: pvcName,
						ReadOnly:  true,
					},
				},
			}},
		},
	}
	for _, opt := range opts {
		opt(pod)
	}
	return pod
}

// FromPVCFile reads a file from a PVC by creating a short-lived reader pod,
// execing cat on the target path, then deleting the pod.
// Returns the raw file bytes on success.
func FromPVCFile(ctx context.Context, cs *kubernetes.Clientset, namespace, pvcName, filePath string, opts ...ReaderOption) ([]byte, error) {
	podName := fmt.Sprintf("pvc-reader-%d", time.Now().UnixNano()%1_000_000)
	pod := newReaderPod(namespace, podName, pvcName, opts...)

	if _, err := cs.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("creating reader pod: %w", err)
	}
	defer func() {
		_ = cs.CoreV1().Pods(namespace).Delete(ctx, podName, metav1.DeleteOptions{
			GracePeriodSeconds: int64Ptr(0),
		})
	}()

	// Wait for pod to be Running.
	if err := wait.PollUntilContextTimeout(ctx, readerPodInterval, readerPodTimeout, true, func(ctx context.Context) (bool, error) {
		p, err := cs.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return p.Status.Phase == corev1.PodRunning, nil
	}); err != nil {
		return nil, fmt.Errorf("waiting for reader pod to be Running: %w", err)
	}

	// Exec cat on the target file.
	cfg, err := ctrl.GetConfig()
	if err != nil {
		return nil, fmt.Errorf("getting kubeconfig for exec: %w", err)
	}

	req := cs.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(podName).
		Namespace(namespace).
		SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: "reader",
			Command:   []string{"cat", "/mnt/pvc/" + filePath},
			Stdout:    true,
			Stderr:    true,
		}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(cfg, "POST", req.URL())
	if err != nil {
		return nil, fmt.Errorf("creating exec: %w", err)
	}

	var stdout, stderr bytes.Buffer
	if err := exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	}); err != nil {
		return nil, fmt.Errorf("exec cat %s: %w (stderr: %s)", filePath, err, stderr.String())
	}

	return stdout.Bytes(), nil
}

func int64Ptr(i int64) *int64 { return &i }
