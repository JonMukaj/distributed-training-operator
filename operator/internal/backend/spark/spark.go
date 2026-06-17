// Package spark implements the backend.JobBackend interface for the Spark Operator.
package spark

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
	"github.com/JonMukaj/distributed-training-operator/internal/backend"
	"github.com/JonMukaj/distributed-training-operator/internal/backend/collect"
)

// Compile-time assertion that *Backend satisfies backend.JobBackend.
var _ backend.JobBackend = (*Backend)(nil)

// Backend implements JobBackend for the Spark Operator (SparkApplication CRD).
type Backend struct{}

// New creates a Spark backend.
func New() *Backend {
	return &Backend{}
}

// ---------------------------------------------------------------------------
// JobBackend implementation
// ---------------------------------------------------------------------------

// JobName implements backend.JobBackend.
func (b *Backend) JobName(job *trainingv1.DistributedTraining) string {
	return fmt.Sprintf("spark-%s", job.Name)
}

// GenerateManifest implements backend.JobBackend.
// Produces a SparkApplication manifest for the Spark Operator.
func (b *Backend) GenerateManifest(job *trainingv1.DistributedTraining) (*unstructured.Unstructured, error) {
	spec := job.Spec
	name := b.JobName(job)

	sparkSpec, err := resolveSparkSpec(spec)
	if err != nil {
		return nil, err
	}

	driverCores, executorCores, executorInstances := resolveTopology(job, sparkSpec)
	driverMemory, executorMemory := resolveMemory(sparkSpec)
	nodeSel, tolerations := buildNodeSelector(job)

	appSpec := map[string]interface{}{
		"type":                sparkSpec.Type,
		"sparkVersion":        sparkSpec.SparkVersion,
		"mode":                "cluster",
		"image":               sparkSpec.Image,
		"imagePullPolicy":     "Always",
		"mainApplicationFile": sparkSpec.MainApplicationFile,
		"restartPolicy": map[string]interface{}{
			"type": "Never",
		},
		"driver": map[string]interface{}{
			"cores":          driverCores,
			"memory":         driverMemory,
			"nodeSelector":   nodeSel,
			"tolerations":    tolerations,
			"serviceAccount": sparkSpec.DriverServiceAccount,
			"volumeMounts":   buildVolumeMounts(job),
		},
		"executor": map[string]interface{}{
			"cores":        executorCores,
			"instances":    executorInstances,
			"memory":       executorMemory,
			"nodeSelector": nodeSel,
			"tolerations":  tolerations,
			"volumeMounts": buildVolumeMounts(job),
		},
		"volumes": buildVolumes(job),
	}
	if len(sparkSpec.Arguments) > 0 {
		appSpec["arguments"] = buildArguments(sparkSpec)
	}

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "sparkoperator.k8s.io/v1beta2",
			"kind":       "SparkApplication",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": job.Namespace,
				"labels": map[string]interface{}{
					"app":                                 name,
					"training.distributedtraining.io/job": job.Name,
				},
			},
			"spec": appSpec,
		},
	}
	return obj, nil
}

// GetPhase implements backend.JobBackend.
// Reads the SparkApplication status.applicationState.state field.
func (b *Backend) GetPhase(
	ctx context.Context,
	c client.Client,
	namespace, jobName string,
) (string, string, error) {
	app := &unstructured.Unstructured{}
	app.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "sparkoperator.k8s.io", Version: "v1beta2", Kind: "SparkApplication",
	})
	if err := c.Get(ctx, types.NamespacedName{Namespace: namespace, Name: jobName}, app); err != nil {
		return "", "", fmt.Errorf("fetching SparkApplication %s: %w", jobName, err)
	}
	phase := parseSparkPhase(app)
	return phase, "", nil
}

// GetStartTime implements backend.JobBackend.
//
// SparkApplication doesn't expose a clean "all executors running" timestamp
// (status.lastSubmissionAttemptTime is close but represents driver submission,
// not executor readiness). For now, return time.Now() so the controller
// records the operator-perspective start time on the first reconcile of the
// Running phase — preserves prior behavior for Spark workloads.
func (b *Backend) GetStartTime(
	ctx context.Context,
	c client.Client,
	namespace, jobName string,
) (time.Time, error) {
	return time.Now(), nil
}

// GetEndTime implements backend.JobBackend.
//
// SparkApplication doesn't expose a clean job-completion timestamp either.
// Returning a zero time tells the controller to fall back to its own
// reconcile-observation time — same as the prior behavior for this backend.
func (b *Backend) GetEndTime(
	ctx context.Context,
	c client.Client,
	namespace, jobName string,
) (time.Time, error) {
	return time.Time{}, nil
}

// CollectResults implements backend.JobBackend.
// Streams logs from the Spark driver pod and extracts metrics.
//
// Keys populated: "recordsProcessed", "processingTimeSeconds", "throughputRecordsPerSec"
func (b *Backend) CollectResults(
	ctx context.Context,
	c client.Client,
	job *trainingv1.DistributedTraining,
) (*trainingv1.JobResults, error) {
	cs, err := collect.NewClientset()
	if err != nil {
		return nil, err
	}
	jobName := b.JobName(job)
	driverPodName, err := findDriverPodName(ctx, c, job.Namespace, jobName)
	if err != nil {
		return nil, fmt.Errorf("locating driver pod: %w", err)
	}
	metrics, err := collect.FromPodLogs(ctx, cs, job.Namespace, driverPodName, "spark-kubernetes-driver",
		parseRecords, parseTimeElapsed, parseThroughput,
	)
	if err != nil {
		return nil, err
	}
	return &trainingv1.JobResults{Metrics: metrics}, nil
}

// ---------------------------------------------------------------------------
// Manifest helpers
// ---------------------------------------------------------------------------

func resolveSparkSpec(spec trainingv1.DistributedTrainingSpec) (*trainingv1.SparkSpec, error) {
	if spec.SparkSpec == nil {
		return nil, fmt.Errorf("sparkSpec is required when backend is spark")
	}
	s := spec.SparkSpec.DeepCopy()
	if s.Type == "" {
		s.Type = "Python"
	}
	if s.SparkVersion == "" {
		s.SparkVersion = "3.5.0"
	}
	return s, nil
}

func resolveTopology(job *trainingv1.DistributedTraining, s *trainingv1.SparkSpec) (int64, int64, int64) {
	spec := job.Spec
	driverCores := int64(1)
	executorCores := int64(spec.Topology.ProcessesPerNode)
	if executorCores == 0 {
		executorCores = 1
	}
	// In objective mode the solver writes the node count to ResolvedTopology.Nodes.
	var executorInstances int64
	if spec.Objective != nil {
		executorInstances = int64(job.Status.ResolvedTopology.Nodes)
	} else {
		executorInstances = int64(spec.Topology.Nodes)
	}
	if executorInstances == 0 {
		executorInstances = 1
	}
	return driverCores, executorCores, executorInstances
}

func resolveMemory(s *trainingv1.SparkSpec) (string, string) {
	driverMemory := s.DriverMemory
	if driverMemory == "" {
		driverMemory = "2g"
	}
	executorMemory := s.ExecutorMemory
	if executorMemory == "" {
		executorMemory = "4g"
	}
	return driverMemory, executorMemory
}

func buildArguments(s *trainingv1.SparkSpec) []interface{} {
	args := make([]interface{}, len(s.Arguments))
	for i, a := range s.Arguments {
		args[i] = a
	}
	return args
}

func buildVolumeMounts(job *trainingv1.DistributedTraining) []interface{} {
	mounts := []interface{}{}
	if job.Spec.OutputPVCName != "" {
		mounts = append(mounts, map[string]interface{}{
			"name":      "output",
			"mountPath": "/mnt/output",
		})
	}
	return mounts
}

func buildVolumes(job *trainingv1.DistributedTraining) []interface{} {
	volumes := []interface{}{}
	if job.Spec.OutputPVCName != "" {
		volumes = append(volumes, map[string]interface{}{
			"name":                  "output",
			"persistentVolumeClaim": map[string]interface{}{"claimName": job.Spec.OutputPVCName},
		})
	}
	return volumes
}

func buildNodeSelector(job *trainingv1.DistributedTraining) (map[string]interface{}, []interface{}) {
	poolName := job.Spec.Hardware.NodePoolName
	if poolName == "" {
		if job.Spec.Hardware.Type == trainingv1.HardwareGPU {
			poolName = "gpu-pool"
		} else {
			poolName = "cpu-pool"
		}
	}
	nodeSelector := map[string]interface{}{"cloud.google.com/gke-nodepool": poolName}
	tolerations := []interface{}{
		map[string]interface{}{"key": "reserved-pool", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
	}
	return nodeSelector, tolerations
}

// ---------------------------------------------------------------------------
// Phase parsing
// ---------------------------------------------------------------------------

// parseSparkPhase maps SparkApplication applicationState to the operator lifecycle phase.
// Spark states: SUBMITTED, RUNNING, COMPLETED, FAILED, SUBMISSION_FAILED, UNKNOWN.
func parseSparkPhase(app *unstructured.Unstructured) string {
	state, _, _ := unstructured.NestedString(app.Object, "status", "applicationState", "state")
	switch state {
	case "COMPLETED":
		return "Succeeded"
	case "FAILED", "SUBMISSION_FAILED":
		return "Failed"
	default:
		return "Running"
	}
}

// ---------------------------------------------------------------------------
// Result collection
// ---------------------------------------------------------------------------

func findDriverPodName(ctx context.Context, c client.Client, namespace, jobName string) (string, error) {
	podList := &unstructured.UnstructuredList{}
	podList.SetGroupVersionKind(schema.GroupVersionKind{Version: "v1", Kind: "PodList"})
	if err := c.List(ctx, podList,
		client.InNamespace(namespace),
		client.MatchingLabels{
			"spark-app-name": jobName,
			"spark-role":     "driver",
		},
	); err != nil {
		return "", err
	}
	if len(podList.Items) == 0 {
		return "", fmt.Errorf("no driver pod found for SparkApplication %s", jobName)
	}
	return podList.Items[0].GetName(), nil
}

// ---------------------------------------------------------------------------
// Metric parsers — each implements collect.LogParser
// ---------------------------------------------------------------------------

var (
	reRecords     = regexp.MustCompile(`records processed:\s*([\d]+)`)
	reTimeElapsed = regexp.MustCompile(`time elapsed:\s*([\d.]+)\s*s`)
	reThroughput  = regexp.MustCompile(`throughput:\s*([\d.eE+\-]+)\s*records/s`)
)

func parseRecords(line string, metrics map[string]string) {
	if m := reRecords.FindStringSubmatch(line); len(m) > 1 {
		metrics["recordsProcessed"] = m[1]
	}
}

func parseTimeElapsed(line string, metrics map[string]string) {
	if m := reTimeElapsed.FindStringSubmatch(line); len(m) > 1 {
		metrics["processingTimeSeconds"] = m[1]
	}
}

func parseThroughput(line string, metrics map[string]string) {
	if m := reThroughput.FindStringSubmatch(line); len(m) > 1 {
		if f, err := strconv.ParseFloat(m[1], 64); err == nil {
			metrics["throughputRecordsPerSec"] = fmt.Sprintf("%.2f", f)
		}
	}
}
