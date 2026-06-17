// Package collect provides generic pod-log streaming and metric parsing
// shared across all JobBackend implementations.
//
// Each backend supplies backend-specific LogParser functions; this package
// handles the plumbing: clientset creation, log streaming, line scanning,
// and scanner error propagation.
package collect

import (
	"bufio"
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"
)

// LogParser is called once per log line. Implementations should parse the
// line and update the metrics map in-place. Parsers are stateless by convention;
// use a closure if state across lines is needed (e.g. tracking a running value).
type LogParser func(line string, metrics map[string]string)

// NewClientset creates a Kubernetes clientset from the in-cluster or
// kubeconfig credentials resolved by controller-runtime.
func NewClientset() (*kubernetes.Clientset, error) {
	cfg, err := ctrl.GetConfig()
	if err != nil {
		return nil, fmt.Errorf("getting kubeconfig: %w", err)
	}
	cs, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating clientset: %w", err)
	}
	return cs, nil
}

// FromPodLogs streams logs from the named pod and container, applies each
// parser to every line, and returns the populated metrics map.
// Scanner errors are propagated so callers can distinguish partial reads.
func FromPodLogs(
	ctx context.Context,
	cs *kubernetes.Clientset,
	namespace, podName, container string,
	parsers ...LogParser,
) (map[string]string, error) {
	stream, err := cs.CoreV1().Pods(namespace).GetLogs(podName, &corev1.PodLogOptions{
		Container: container,
	}).Stream(ctx)
	if err != nil {
		return nil, fmt.Errorf("streaming pod logs for %s/%s: %w", namespace, podName, err)
	}
	defer stream.Close()

	metrics := make(map[string]string)
	scanner := bufio.NewScanner(stream)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		for _, p := range parsers {
			p(line, metrics)
		}
	}
	if err := scanner.Err(); err != nil {
		return metrics, fmt.Errorf("reading pod logs: %w", err)
	}
	return metrics, nil
}
