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

// FromPVCFile reads a file from a PVC by creating a short-lived reader pod,
// execing cat on the target path, then deleting the pod.
// Returns the raw file bytes on success.
func FromPVCFile(ctx context.Context, cs *kubernetes.Clientset, namespace, pvcName, filePath string) ([]byte, error) {
	podName := fmt.Sprintf("pvc-reader-%d", time.Now().UnixNano()%1_000_000)

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
