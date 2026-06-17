// Package pytorch implements the backend.JobBackend interface for Kubeflow PyTorchJob.
package pytorch

import (
	"context"
	"encoding/json"
	"fmt"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"math"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"time"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
	"github.com/JonMukaj/distributed-training-operator/internal/backend"
	"github.com/JonMukaj/distributed-training-operator/internal/backend/collect"
)

// Compile-time assertion that *Backend satisfies backend.JobBackend.
var _ backend.JobBackend = (*Backend)(nil)

// Backend implements JobBackend for Kubeflow PyTorchJob.
type Backend struct{}

// New creates a PyTorch backend.
func New() *Backend {
	return &Backend{}
}

// ---------------------------------------------------------------------------
// JobBackend implementation
// ---------------------------------------------------------------------------

// JobName implements backend.JobBackend.
func (b *Backend) JobName(job *trainingv1.DistributedTraining) string {
	return fmt.Sprintf("pj-%s", job.Name)
}

// GenerateManifest implements backend.JobBackend.
// Uses pytorchSpec directly for infrastructure (image, replicas, resources).
// If pytorchSpec.Command is set, uses it as-is (general distributed training).
// If pytorchSpec.Command is empty and model/dataset are set, builds a torchrun
// command from the LLM fine-tuning overlay fields.
func (b *Backend) GenerateManifest(job *trainingv1.DistributedTraining) (*unstructured.Unstructured, error) {
	ps := job.Spec.PytorchSpec
	if ps == nil {
		return nil, fmt.Errorf("pytorchSpec is required for pytorch backend")
	}

	name := b.JobName(job)
	masterReplicas := int64(ps.MasterReplicas)
	if masterReplicas == 0 {
		masterReplicas = 1
	}
	var workerReplicas int64
	// In objective mode the solver writes the replica split to ResolvedTopology.
	if job.Spec.Objective != nil {
		masterReplicas = int64(job.Status.ResolvedTopology.MasterReplicas)
		workerReplicas = int64(job.Status.ResolvedTopology.WorkerReplicas)
	} else if job.Spec.Topology.Nodes > 1 {
		workerReplicas = int64(job.Spec.Topology.Nodes) - masterReplicas
	}

	command, args := buildCommandAndArgs(job)
	envVars := buildEnvVars(job)
	volumes, volumeMounts := buildVolumes(job)
	resources := buildResources(job)
	nodeSel, tolerations := buildNodeSelector(job)

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "kubeflow.org/v1",
			"kind":       "PyTorchJob",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": job.Namespace,
				"labels": map[string]interface{}{
					"app":                                 name,
					"training.distributedtraining.io/job": job.Name,
				},
			},
			"spec": map[string]interface{}{
				"pytorchReplicaSpecs": buildReplicaSpecs(masterReplicas, workerReplicas, ps.Image, command, args, envVars, volumes, volumeMounts, resources, nodeSel, tolerations),
			},
		},
	}
	return obj, nil
}

// GetPhase implements backend.JobBackend.
func (b *Backend) GetPhase(
	ctx context.Context,
	c client.Client,
	namespace, jobName string,
) (string, string, error) {
	pj := &unstructured.Unstructured{}
	pj.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "kubeflow.org", Version: "v1", Kind: "PyTorchJob",
	})
	if err := c.Get(ctx, types.NamespacedName{Namespace: namespace, Name: jobName}, pj); err != nil {
		return "", "", fmt.Errorf("fetching PyTorchJob %s: %w", jobName, err)
	}
	phase, msg := parsePyTorchJobPhase(pj)
	return phase, msg, nil
}

// GetStartTime implements backend.JobBackend.
//
// Returns PyTorchJob.status.startTime — the wall-clock moment Kubeflow's
// training-operator considers all replica pods to have reached Running.
// This excludes image pull and pod scheduling from the operator's
// TrainingStartTime, which means `distributedtraining_training_seconds` ends up
// measuring the same window as baseline_train.sh's `training_s`.
//
// Returns a zero time (with nil error) if status.startTime is not yet
// populated — the controller treats that as "retry later."
func (b *Backend) GetStartTime(
	ctx context.Context,
	c client.Client,
	namespace, jobName string,
) (time.Time, error) {
	return pytorchJobStatusTime(ctx, c, namespace, jobName, "startTime")
}

// GetEndTime implements backend.JobBackend.
//
// Returns PyTorchJob.status.completionTime — the wall-clock moment Kubeflow's
// training-operator marked the job complete. Used as the right endpoint of
// `training_seconds` so the metric exactly matches baseline_train.sh's
// PJ.status.startTime → completionTime window (no reconcile-lag inflation).
//
// Returns a zero time (with nil error) if completionTime is not yet populated.
func (b *Backend) GetEndTime(
	ctx context.Context,
	c client.Client,
	namespace, jobName string,
) (time.Time, error) {
	return pytorchJobStatusTime(ctx, c, namespace, jobName, "completionTime")
}

// pytorchJobStatusTime reads an RFC3339 timestamp from
// PyTorchJob.status.<field>. Returns zero time if absent or empty.
func pytorchJobStatusTime(
	ctx context.Context,
	c client.Client,
	namespace, jobName, field string,
) (time.Time, error) {
	pj := &unstructured.Unstructured{}
	pj.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "kubeflow.org", Version: "v1", Kind: "PyTorchJob",
	})
	if err := c.Get(ctx, types.NamespacedName{Namespace: namespace, Name: jobName}, pj); err != nil {
		return time.Time{}, fmt.Errorf("fetching PyTorchJob %s: %w", jobName, err)
	}
	str, found, err := unstructured.NestedString(pj.Object, "status", field)
	if err != nil || !found || str == "" {
		return time.Time{}, nil
	}
	t, parseErr := time.Parse(time.RFC3339, str)
	if parseErr != nil {
		return time.Time{}, fmt.Errorf("parsing PyTorchJob status.%s %q: %w", field, str, parseErr)
	}
	return t, nil
}

// allResults maps the subset of HuggingFace Trainer's all_results.json we care about.
type allResults struct {
	TrainSamplesPerSecond float64 `json:"train_samples_per_second"`
	TrainLoss             float64 `json:"train_loss"`
	EvalLoss              float64 `json:"eval_loss"`
	Perplexity            float64 `json:"perplexity"`
	TrainRuntime          float64 `json:"train_runtime"`
	TrainSamples          int     `json:"train_samples"`
}

// CollectResults implements backend.JobBackend.
//
// Reads all_results.json written by the HuggingFace Trainer to
// /mnt/output/checkpoints/all_results.json on the configured output PVC.
// OutputPVCName is required — the log-scraping fallback was removed so the
// operator can delete the worker node pool in parallel with result collection
// (the NFS server lives on a different pool, so PVC reads survive the worker
// teardown).
//
// Keys populated: "loss", "perplexity", "samplesPerSecond"
func (b *Backend) CollectResults(
	ctx context.Context,
	c client.Client,
	job *trainingv1.DistributedTraining,
) (*trainingv1.JobResults, error) {
	if job.Spec.OutputPVCName == "" {
		return nil, fmt.Errorf("pytorch backend requires spec.outputPVCName to collect results")
	}
	cs, err := collect.NewClientset()
	if err != nil {
		return nil, err
	}
	metrics, err := collectFromPVC(ctx, cs, job.Namespace, job.Spec.OutputPVCName)
	if err != nil {
		return nil, fmt.Errorf("reading all_results.json from PVC %s: %w", job.Spec.OutputPVCName, err)
	}
	return &trainingv1.JobResults{Metrics: metrics}, nil
}

func collectFromPVC(ctx context.Context, cs *kubernetes.Clientset, namespace, pvcName string) (map[string]string, error) {
	raw, err := collect.FromPVCFile(ctx, cs, namespace, pvcName, "checkpoints/all_results.json")
	if err != nil {
		return nil, err
	}
	var r allResults
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, fmt.Errorf("parsing all_results.json: %w", err)
	}
	metrics := make(map[string]string)
	if r.TrainSamplesPerSecond > 0 {
		metrics["samplesPerSecond"] = fmt.Sprintf("%.4f", r.TrainSamplesPerSecond)
	}
	// Store total samples processed for wall-clock-effective throughput
	// computation in history (W = totalSamples, P = totalSamples/T_wall).
	// This removes the inner-loop-vs-wall-clock throughput mismatch from the
	// solver fit.
	if r.TrainSamples > 0 {
		metrics["totalSamples"] = fmt.Sprintf("%d", r.TrainSamples)
	}
	loss := r.EvalLoss
	if loss == 0 {
		loss = r.TrainLoss
	}
	if loss > 0 {
		metrics["loss"] = fmt.Sprintf("%.4f", loss)
		ppl := r.Perplexity
		if ppl == 0 {
			ppl = math.Exp(loss)
		}
		metrics["perplexity"] = fmt.Sprintf("%.4f", ppl)
	}
	return metrics, nil
}

// ---------------------------------------------------------------------------
// Manifest sub-builders
// ---------------------------------------------------------------------------

// buildCommandAndArgs returns the container command and args.
// If pytorchSpec.Command is set → use first element as command, rest as args.
// Otherwise build torchrun from the LLM fine-tuning overlay.
func buildCommandAndArgs(job *trainingv1.DistributedTraining) ([]interface{}, []interface{}) {
	ps := job.Spec.PytorchSpec

	if len(ps.Command) > 0 {
		cmd := []interface{}{ps.Command[0]}
		args := make([]interface{}, len(ps.Command)-1)
		for i, a := range ps.Command[1:] {
			args[i] = a
		}
		return cmd, args
	}

	// Build torchrun command for LLM fine-tuning workload.
	torchrunArgs := buildTorchrunArgs(job)
	return []interface{}{"torchrun"}, torchrunArgs
}

func buildTorchrunArgs(job *trainingv1.DistributedTraining) []interface{} {
	spec := job.Spec
	ps := spec.PytorchSpec

	processesPerNode := int32(1)
	if spec.Topology.ProcessesPerNode > 0 {
		processesPerNode = spec.Topology.ProcessesPerNode
	}

	trainingScript := "/workspace/finetune.py"
	if spec.Model != nil && spec.Model.TrainingScript != "" {
		trainingScript = spec.Model.TrainingScript
	}

	args := []string{
		fmt.Sprintf("--nproc_per_node=%d", processesPerNode),
		"--nnodes=$(WORLD_SIZE)",
		"--node_rank=$(RANK)",
		"--master_addr=$(MASTER_ADDR)",
		"--master_port=23456",
		trainingScript,
	}

	if spec.Model != nil && spec.Model.Name != "" {
		args = append(args, fmt.Sprintf("--model_name_or_path=%s", spec.Model.Name))
	}
	if spec.Dataset != nil && spec.Dataset.Name != "" {
		args = append(args, fmt.Sprintf("--dataset_name=%s", spec.Dataset.Name))
		if spec.Dataset.Split != "" {
			args = append(args, fmt.Sprintf("--dataset_split=%s", spec.Dataset.Split))
		}
		datasetCacheDir := spec.Dataset.CacheDirectory
		if datasetCacheDir == "" {
			cacheBase := "/mnt/output/hf-cache"
			if spec.Model != nil && spec.Model.CacheDir != "" {
				cacheBase = spec.Model.CacheDir
			}
			datasetCacheDir = cacheBase + "/datasets"
		}
		args = append(args, fmt.Sprintf("--dataset_cache_directory=%s", datasetCacheDir))
		if spec.Dataset.PromptWithInput != "" {
			args = append(args, fmt.Sprintf("--prompt_with_input=%s", spec.Dataset.PromptWithInput))
		}
		if spec.Dataset.PromptWithoutInput != "" {
			args = append(args, fmt.Sprintf("--prompt_without_input=%s", spec.Dataset.PromptWithoutInput))
		}
	}

	args = append(args, "--output_dir=/mnt/output/checkpoints")

	if spec.Training.BatchSize > 0 {
		args = append(args, fmt.Sprintf("--per_device_train_batch_size=%d", spec.Training.BatchSize))
	}
	evalBatch := spec.Training.EvalBatchSize
	if evalBatch == 0 {
		evalBatch = spec.Training.BatchSize
	}
	if evalBatch > 0 {
		args = append(args, fmt.Sprintf("--per_device_eval_batch_size=%d", evalBatch))
	}
	if spec.Training.LearningRate != "" {
		args = append(args, fmt.Sprintf("--learning_rate=%s", spec.Training.LearningRate))
	}
	if spec.Training.Epochs > 0 {
		args = append(args, fmt.Sprintf("--num_train_epochs=%d", spec.Training.Epochs))
	}
	args = append(args, fmt.Sprintf("--max_steps=%d", spec.Training.MaxSteps))
	if spec.Training.GradAccumulationSteps > 0 {
		args = append(args, fmt.Sprintf("--gradient_accumulation_steps=%d", spec.Training.GradAccumulationSteps))
	}
	if spec.Training.MaxGradNorm != "" {
		args = append(args, fmt.Sprintf("--max_grad_norm=%s", spec.Training.MaxGradNorm))
	}
	if spec.Training.ValidationSplit != "" {
		args = append(args, fmt.Sprintf("--validation_split_percentage=%s", spec.Training.ValidationSplit))
	}
	if spec.Training.WarmupSteps > 0 {
		args = append(args, fmt.Sprintf("--warmup_steps=%d", spec.Training.WarmupSteps))
	}
	if spec.Training.LoggingSteps > 0 {
		args = append(args, fmt.Sprintf("--logging_steps=%d", spec.Training.LoggingSteps))
	}
	if spec.Training.SaveTotalLimit > 0 {
		args = append(args, fmt.Sprintf("--save_total_limit=%d", spec.Training.SaveTotalLimit))
	}
	if spec.Training.SaveStrategy != "" {
		args = append(args, fmt.Sprintf("--save_strategy=%s", spec.Training.SaveStrategy))
	}
	if spec.Training.OverwriteOutputDir {
		args = append(args, "--overwrite_output_dir=True")
	}

	args = append(args, "--do_train=True", "--do_eval=True")

	switch spec.Optimization.MixedPrecision {
	case trainingv1.MixedPrecisionBF16:
		args = append(args, "--bf16=True")
		if spec.Optimization.BF16FullEval {
			args = append(args, "--bf16_full_eval=True")
		}
	case trainingv1.MixedPrecisionFP32:
		args = append(args, "--bf16=False", "--fp16=False")
	}

	if spec.Optimization.LoRA.Enabled {
		lora := spec.Optimization.LoRA
		args = append(args,
			"--use_lora=True",
			fmt.Sprintf("--lora_rank=%d", lora.Rank),
			fmt.Sprintf("--lora_alpha=%d", lora.Alpha),
			fmt.Sprintf("--lora_dropout=%s", lora.Dropout),
		)
		for _, m := range lora.TargetModules {
			args = append(args, fmt.Sprintf("--lora_target_modules=%s", m))
		}
	}

	// Detect GPU from pytorchSpec resources or hardware.type.
	gpu := hasGPUFromSpec(ps, spec.Hardware)
	if gpu {
		args = append(args, "--no_cuda=False")
	} else {
		args = append(args, "--no_cuda=True")
	}

	// DDP backend: explicit > default-by-hardware (gloo for CPU, nccl for GPU).
	ddpBackend := spec.Training.DDPBackend
	if ddpBackend == "" {
		if gpu {
			ddpBackend = "nccl"
		} else {
			ddpBackend = "gloo"
		}
	}
	args = append(args, fmt.Sprintf("--ddp_backend=%s", ddpBackend))

	if spec.Training.DDPFindUnusedParameters != nil {
		if *spec.Training.DDPFindUnusedParameters {
			args = append(args, "--ddp_find_unused_parameters=True")
		} else {
			args = append(args, "--ddp_find_unused_parameters=False")
		}
	}

	if spec.Training.UseFastTokenizer != nil {
		if *spec.Training.UseFastTokenizer {
			args = append(args, "--use_fast_tokenizer=True")
		} else {
			args = append(args, "--use_fast_tokenizer=False")
		}
	}

	if spec.ResumeFromCheckpoint != "" {
		args = append(args, fmt.Sprintf("--resume_from_checkpoint=%s", spec.ResumeFromCheckpoint))
	}

	result := make([]interface{}, len(args))
	for i, a := range args {
		result[i] = a
	}
	return result
}

func buildEnvVars(job *trainingv1.DistributedTraining) []interface{} {
	vars := []interface{}{}

	// HF_TOKEN is the only env the operator injects dynamically — it wires a
	// Kubernetes Secret reference (valueFrom.secretKeyRef) that can't be
	// expressed cleanly via the static value-only pytorchSpec.env path below.
	// All other env vars (HF cache paths, LD_PRELOAD, etc.) must be set on
	// the DistributedTraining CR via spec.pytorchSpec.env so the operator stays
	// image- and workload-agnostic.
	if job.Spec.Model != nil {
		if hft := job.Spec.Model.HFTokenSecret; hft != nil {
			vars = append(vars, map[string]interface{}{
				"name": "HF_TOKEN",
				"valueFrom": map[string]interface{}{
					"secretKeyRef": map[string]interface{}{"name": hft.Name, "key": hft.Key},
				},
			})
		}
	}

	// Append user-defined env vars from pytorchSpec.
	for _, e := range job.Spec.PytorchSpec.Env {
		entry := map[string]interface{}{"name": e.Name, "value": e.Value}
		vars = append(vars, entry)
	}

	return vars
}

func buildVolumes(job *trainingv1.DistributedTraining) ([]interface{}, []interface{}) {
	volumes := []interface{}{
		map[string]interface{}{"name": "dshm", "emptyDir": map[string]interface{}{"medium": "Memory"}},
	}
	mounts := []interface{}{
		map[string]interface{}{"name": "dshm", "mountPath": "/dev/shm"},
	}
	if job.Spec.OutputPVCName != "" {
		volumes = append(volumes, map[string]interface{}{
			"name":                  "output",
			"persistentVolumeClaim": map[string]interface{}{"claimName": job.Spec.OutputPVCName},
		})
		mounts = append(mounts, map[string]interface{}{"name": "output", "mountPath": "/mnt/output"})
	}
	return volumes, mounts
}

// buildResources converts pytorchSpec.Resources into an unstructured map.
// Falls back to GPU resource limit derived from hardware spec if resources are empty.
func buildResources(job *trainingv1.DistributedTraining) map[string]interface{} {
	ps := job.Spec.PytorchSpec
	if ps == nil {
		return map[string]interface{}{}
	}

	// If pytorchSpec.Resources has limits set, convert them directly.
	if len(ps.Resources.Limits) > 0 || len(ps.Resources.Requests) > 0 {
		res := map[string]interface{}{}
		if len(ps.Resources.Limits) > 0 {
			limits := map[string]interface{}{}
			for k, v := range ps.Resources.Limits {
				limits[string(k)] = v.String()
			}
			res["limits"] = limits
		}
		if len(ps.Resources.Requests) > 0 {
			requests := map[string]interface{}{}
			for k, v := range ps.Resources.Requests {
				requests[string(k)] = v.String()
			}
			res["requests"] = requests
		}
		return res
	}

	// Fallback: if hardware spec says GPU, add a default GPU limit.
	if job.Spec.Hardware.Type == trainingv1.HardwareGPU {
		gpuCount := job.Spec.Hardware.GPUCount
		if gpuCount == 0 {
			gpuCount = 1
		}
		return map[string]interface{}{
			"limits": map[string]interface{}{"nvidia.com/gpu": fmt.Sprintf("%d", gpuCount)},
		}
	}

	return map[string]interface{}{}
}

func buildNodeSelector(job *trainingv1.DistributedTraining) (map[string]interface{}, []interface{}) {
	poolName := job.Spec.Hardware.NodePoolName
	if poolName == "" {
		if hasGPUFromSpec(job.Spec.PytorchSpec, job.Spec.Hardware) {
			poolName = "gpu-pool"
		} else {
			poolName = "cpu-pool"
		}
	}
	nodeSelector := map[string]interface{}{"cloud.google.com/gke-nodepool": poolName}
	tolerations := []interface{}{
		map[string]interface{}{"key": "reserved-pool", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
	}
	if hasGPUFromSpec(job.Spec.PytorchSpec, job.Spec.Hardware) {
		tolerations = append(tolerations, map[string]interface{}{
			"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule",
		})
	}
	return nodeSelector, tolerations
}

// hasGPU reports whether the job requires GPU hardware.
// Checks both the explicit hardware.type field and the nvidia.com/gpu resource limit.
func hasGPU(ps *trainingv1.PytorchSpec) bool {
	if ps == nil {
		return false
	}
	for k := range ps.Resources.Limits {
		if string(k) == "nvidia.com/gpu" {
			return true
		}
	}
	return false
}

// hasGPUFromSpec checks both pytorchSpec resource limits and hardware.type.
func hasGPUFromSpec(ps *trainingv1.PytorchSpec, hw trainingv1.HardwareSpec) bool {
	if hw.Type == trainingv1.HardwareGPU {
		return true
	}
	return hasGPU(ps)
}

func buildReplicaSpecs(
	masterReplicas, workerReplicas int64,
	image string,
	command, args, envVars, volumes, volumeMounts []interface{},
	resources map[string]interface{},
	nodeSelector map[string]interface{},
	tolerations []interface{},
) map[string]interface{} {
	specs := map[string]interface{}{
		"Master": map[string]interface{}{
			"replicas":      masterReplicas,
			"restartPolicy": "OnFailure",
			"template":      replicaTemplate(image, command, args, envVars, volumes, volumeMounts, resources, nodeSelector, tolerations),
		},
	}
	if workerReplicas > 0 {
		specs["Worker"] = map[string]interface{}{
			"replicas":      workerReplicas,
			"restartPolicy": "OnFailure",
			"template":      replicaTemplate(image, command, args, envVars, volumes, volumeMounts, resources, nodeSelector, tolerations),
		}
	}
	return specs
}

func replicaTemplate(
	image string,
	command, args, envVars, volumes, volumeMounts []interface{},
	resources map[string]interface{},
	nodeSelector map[string]interface{},
	tolerations []interface{},
) map[string]interface{} {
	return map[string]interface{}{
		"spec": map[string]interface{}{
			"restartPolicy": "Never",
			"nodeSelector":  nodeSelector,
			"tolerations":   tolerations,
			"volumes":       volumes,
			"containers": []interface{}{
				map[string]interface{}{
					"name":            "pytorch",
					"image":           image,
					"imagePullPolicy": "Always",
					"command":         command,
					"args":            args,
					"env":             envVars,
					"volumeMounts":    volumeMounts,
					"resources":       resources,
					"securityContext": map[string]interface{}{"allowPrivilegeEscalation": false},
				},
			},
		},
	}
}

// ---------------------------------------------------------------------------
// Phase parsing
// ---------------------------------------------------------------------------

func parsePyTorchJobPhase(pj *unstructured.Unstructured) (string, string) {
	conditions, _, _ := unstructured.NestedSlice(pj.Object, "status", "conditions")
	for _, c := range conditions {
		cmap, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		if cmap["status"] != "True" {
			continue
		}
		switch cmap["type"] {
		case "Succeeded":
			return "Succeeded", fmt.Sprint(cmap["reason"])
		case "Failed":
			return "Failed", fmt.Sprint(cmap["reason"])
		}
	}
	return "Running", ""
}
