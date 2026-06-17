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

package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ---------------------------------------------------------------------------
// Backend constants
// ---------------------------------------------------------------------------

// BackendType selects the distributed-training framework that backs the job.
// +kubebuilder:validation:Enum=pytorch;spark;job
type BackendType string

const (
	// BackendPyTorch uses Kubeflow PyTorchJob (default).
	BackendPyTorch BackendType = "pytorch"

	// BackendSpark uses the Spark Operator (SparkApplication) for general
	// distributed data processing workloads.
	BackendSpark BackendType = "spark"

	// BackendJob uses a plain Kubernetes batch/v1 Job for single-node workloads.
	BackendJob BackendType = "job"
)

// ---------------------------------------------------------------------------
// Phase constants
// ---------------------------------------------------------------------------

// Phase describes the current lifecycle stage of a DistributedTraining.
type Phase string

const (
	// PhasePending: CR created, spec is being validated.
	PhasePending Phase = "Pending"

	// PhaseProvisioning: controller is scaling the node pool.
	PhaseProvisioning Phase = "Provisioning"

	// PhaseReady: nodes are Ready; backend job manifest is being applied.
	PhaseReady Phase = "Ready"

	// PhaseRunning: backend job submitted; controller is monitoring pods.
	PhaseRunning Phase = "Running"

	// PhaseCollecting: training complete; reading metrics and tearing down nodes.
	PhaseCollecting Phase = "Collecting"

	// PhaseSucceeded: results written to status; node pool scaled down.
	PhaseSucceeded Phase = "Succeeded"

	// PhaseFailed: an unrecoverable error occurred.
	PhaseFailed Phase = "Failed"
)

// ---------------------------------------------------------------------------
// Spec sub-types
// ---------------------------------------------------------------------------

// PytorchSpec holds the PyTorch job infrastructure configuration.
// Used when backend is "pytorch". Replaces the hardware/topology pattern
// for PyTorch workloads — the spec shape is the type (Kubernetes volume style).
type PytorchSpec struct {
	// MasterReplicas is the number of master replicas. Defaults to 1.
	// +optional
	// +kubebuilder:default=1
	MasterReplicas int32 `json:"masterReplicas,omitempty"`

	// Image is the container image for both master and worker pods.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// Command overrides the default torchrun entrypoint.
	// When set, it is used verbatim and model/dataset/training fields are ignored.
	// When omitted, the operator builds a torchrun command from model/dataset/training/optimization.
	// +optional
	Command []string `json:"command,omitempty"`

	// Resources specifies the compute resources for each replica pod.
	// +optional
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// Env is a list of environment variables to set in every replica container.
	// These are merged with the operator-generated HF token / cache variables.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// ModelSpec identifies the model and optional Hugging Face credentials.
// Used primarily for LLM fine-tuning workloads; ignored by backends that do
// not require a pretrained model.
type ModelSpec struct {
	// Name is the Hugging Face model identifier, e.g. "meta-llama/Llama-3.2-1B".
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// HFTokenSecret references the Kubernetes Secret key that holds the
	// Hugging Face access token. Required for gated models.
	// +optional
	HFTokenSecret *corev1.SecretKeySelector `json:"hfTokenSecret,omitempty"`

	// CacheDir is the path inside the training container where Hugging Face
	// caches downloaded model weights. Defaults to /mnt/output/hf-cache (persisted on PVC).
	// +optional
	// +kubebuilder:default="/mnt/output/hf-cache"
	CacheDir string `json:"cacheDir,omitempty"`

	// TrainingScript is the path to the fine-tuning script inside the container.
	// Defaults to /workspace/finetune.py.
	// +optional
	// +kubebuilder:default="/workspace/finetune.py"
	TrainingScript string `json:"trainingScript,omitempty"`
}

// DatasetSpec identifies the training dataset.
// Used primarily for LLM fine-tuning workloads.
type DatasetSpec struct {
	// Name is the Hugging Face dataset identifier.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Split controls which portion of the dataset is used for training.
	// Defaults to "train".
	// +optional
	// +kubebuilder:default="train"
	Split string `json:"split,omitempty"`

	// CacheDirectory is the path inside the container where the dataset cache is stored.
	// +optional
	CacheDirectory string `json:"cacheDirectory,omitempty"`

	// PromptWithInput overrides the prompt template used when an instruction has an input field.
	// +optional
	PromptWithInput string `json:"promptWithInput,omitempty"`

	// PromptWithoutInput overrides the prompt template used when an instruction has no input field.
	// +optional
	PromptWithoutInput string `json:"promptWithoutInput,omitempty"`
}

// HardwareType selects the broad hardware class for the job.
// +kubebuilder:validation:Enum=cpu;gpu
type HardwareType string

const (
	HardwareCPU HardwareType = "cpu"
	HardwareGPU HardwareType = "gpu"
)

// HardwareSpec controls which GKE node pool and machine type to use.
// Used by the Spark backend and optional for PyTorch (hardware type is
// inferred from pytorchSpec.resources when not set).
type HardwareSpec struct {
	// Type selects cpu or gpu hardware.
	// +kubebuilder:validation:Enum=cpu;gpu
	// +optional
	Type HardwareType `json:"type,omitempty"`

	// NodePoolName overrides the default node pool selected by the operator.
	// +optional
	NodePoolName string `json:"nodePoolName,omitempty"`

	// MachineType overrides the GCE machine type, e.g. "c4-highcpu-8".
	// +optional
	MachineType string `json:"machineType,omitempty"`

	// GPUType is the accelerator type when hardware.type=gpu, e.g. "nvidia-l4".
	// +optional
	GPUType string `json:"gpuType,omitempty"`

	// GPUCount is the number of GPUs per node. Defaults to 1.
	// +optional
	// +kubebuilder:default=1
	GPUCount int32 `json:"gpuCount,omitempty"`

	// DiskSizeGb overrides the boot disk size (in GB) for each node in the
	// ephemeral node pool the operator provisions. When 0 (the default),
	// the operator falls back to its `--default-disk-size-gb` flag.
	// +optional
	// +kubebuilder:validation:Minimum=10
	DiskSizeGb int32 `json:"diskSizeGb,omitempty"`
}

// TopologySpec defines the distributed job topology.
// Used by the Spark backend. For PyTorch the topology is derived from
// pytorchSpec.masterReplicas + workerReplicas.
type TopologySpec struct {
	// Nodes is the total number of worker nodes.
	// Zero is valid in objective mode (solver fills this in via status.resolvedTopology).
	// +optional
	// +kubebuilder:validation:Minimum=0
	Nodes int32 `json:"nodes,omitempty"`

	// ProcessesPerNode is the number of processes (ranks) per node.
	// +optional
	// +kubebuilder:default=1
	ProcessesPerNode int32 `json:"processesPerNode,omitempty"`
}

// MixedPrecision selects the floating-point format.
// +kubebuilder:validation:Enum=bf16;fp32
type MixedPrecision string

const (
	MixedPrecisionBF16 MixedPrecision = "bf16"
	MixedPrecisionFP32 MixedPrecision = "fp32"
)

// LoRASpec configures Low-Rank Adaptation fine-tuning.
type LoRASpec struct {
	// Enabled toggles LoRA. When false all other fields are ignored.
	Enabled bool `json:"enabled"`

	// Rank is the LoRA rank dimension. Defaults to 4.
	// +optional
	// +kubebuilder:default=4
	Rank int32 `json:"rank,omitempty"`

	// Alpha is the LoRA scaling factor. Defaults to 8.
	// +optional
	// +kubebuilder:default=8
	Alpha int32 `json:"alpha,omitempty"`

	// Dropout probability for LoRA layers. Defaults to "0.1".
	// +optional
	// +kubebuilder:default="0.1"
	Dropout string `json:"dropout,omitempty"`

	// TargetModules lists the model module names to apply LoRA to.
	// +optional
	TargetModules []string `json:"targetModules,omitempty"`
}

// OptimizationSpec groups all training-efficiency options.
type OptimizationSpec struct {
	// MixedPrecision selects bf16 or fp32 training. Defaults to bf16.
	// +optional
	// +kubebuilder:default="bf16"
	MixedPrecision MixedPrecision `json:"mixedPrecision,omitempty"`

	// LoRA configures Low-Rank Adaptation.
	// +optional
	LoRA LoRASpec `json:"lora,omitempty"`

	// BF16FullEval enables bf16 precision during evaluation (CPU training).
	// +optional
	BF16FullEval bool `json:"bf16FullEval,omitempty"`
}

// TrainingSpec holds the standard hyperparameters.
type TrainingSpec struct {
	// BatchSize is the per-device training batch size.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=4
	BatchSize int32 `json:"batchSize"`

	// LearningRate as a decimal string, e.g. "2e-5".
	// +kubebuilder:default="2e-5"
	LearningRate string `json:"learningRate"`

	// Epochs is the number of training epochs.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	Epochs int32 `json:"epochs"`

	// GradAccumulationSteps accumulates gradients over N steps.
	// +optional
	// +kubebuilder:default=1
	GradAccumulationSteps int32 `json:"gradAccumulationSteps,omitempty"`

	// ValidationSplit fraction of the dataset reserved for evaluation.
	// +optional
	// +kubebuilder:default="0.2"
	ValidationSplit string `json:"validationSplit,omitempty"`

	// WarmupSteps is the number of linear warmup steps for the learning rate scheduler.
	// +optional
	// +kubebuilder:default=0
	WarmupSteps int32 `json:"warmupSteps,omitempty"`

	// MaxSteps overrides epochs-based training. -1 means train for full epochs.
	// +optional
	// +kubebuilder:default=-1
	MaxSteps int32 `json:"maxSteps,omitempty"`

	// MaxGradNorm is the gradient clipping max norm. Defaults to 1.0.
	// +optional
	MaxGradNorm string `json:"maxGradNorm,omitempty"`

	// LoggingSteps is the number of update steps between log outputs.
	// +optional
	// +kubebuilder:default=10
	LoggingSteps int32 `json:"loggingSteps,omitempty"`

	// SaveTotalLimit caps the number of saved checkpoints.
	// +optional
	// +kubebuilder:default=2
	SaveTotalLimit int32 `json:"saveTotalLimit,omitempty"`

	// SaveStrategy controls when checkpoints are saved (epoch or steps).
	// +optional
	// +kubebuilder:default="epoch"
	// +kubebuilder:validation:Enum=epoch;steps;no
	SaveStrategy string `json:"saveStrategy,omitempty"`

	// EvalBatchSize is the per-device evaluation batch size.
	// When 0, defaults to the training batch size.
	// +optional
	EvalBatchSize int32 `json:"evalBatchSize,omitempty"`

	// DDPBackend is the PyTorch distributed backend (gloo, nccl, mpi).
	// Defaults to "gloo" for CPU jobs and "nccl" for GPU jobs when unset.
	// +optional
	// +kubebuilder:validation:Enum=gloo;nccl;mpi
	DDPBackend string `json:"ddpBackend,omitempty"`

	// DDPFindUnusedParameters sets find_unused_parameters in DDP.
	// +optional
	DDPFindUnusedParameters *bool `json:"ddpFindUnusedParameters,omitempty"`

	// OverwriteOutputDir overwrites the output directory if it already exists.
	// +optional
	OverwriteOutputDir bool `json:"overwriteOutputDir,omitempty"`

	// UseFastTokenizer controls whether to use the fast HuggingFace tokenizer.
	// When unset, the training script default applies (true).
	// Set to false for models where the fast tokenizer produces incorrect output.
	// +optional
	UseFastTokenizer *bool `json:"useFastTokenizer,omitempty"`
}

// SparkSpec holds configuration for the Spark Operator backend.
// Only used when backend is "spark".
type SparkSpec struct {
	// Type is the Spark application type. One of Python, Scala, Java, R.
	// +kubebuilder:validation:Enum=Python;Scala;Java;R
	// +kubebuilder:default="Python"
	Type string `json:"type,omitempty"`

	// SparkVersion is the version of Spark to use, e.g. "3.5.0".
	// +kubebuilder:default="3.5.0"
	SparkVersion string `json:"sparkVersion,omitempty"`

	// Image is the container image for the driver and executor pods.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// MainApplicationFile is the path to the application entry point
	// (e.g. "local:///opt/spark/app/job.py").
	// +kubebuilder:validation:MinLength=1
	MainApplicationFile string `json:"mainApplicationFile"`

	// Arguments is the list of arguments to pass to the main application.
	// +optional
	Arguments []string `json:"arguments,omitempty"`

	// DriverServiceAccount is the Kubernetes service account for the Spark driver pod.
	// The SA must already exist and have permission to create executor pods.
	// Defaults to "spark".
	// +optional
	// +kubebuilder:default="spark"
	DriverServiceAccount string `json:"driverServiceAccount,omitempty"`

	// DriverMemory is the memory request for the driver pod (e.g. "2g").
	// +optional
	// +kubebuilder:default="2g"
	DriverMemory string `json:"driverMemory,omitempty"`

	// ExecutorMemory is the memory request per executor pod (e.g. "4g").
	// +optional
	// +kubebuilder:default="4g"
	ExecutorMemory string `json:"executorMemory,omitempty"`
}

// ---------------------------------------------------------------------------
// Main spec
// ---------------------------------------------------------------------------

// ObjectiveSpec declares time and cost constraints for automatic topology sizing.
// Mutually exclusive with explicit topology fields.
type ObjectiveSpec struct {
	// TargetTime is the desired maximum wall-clock duration (e.g. "2h", "90m").
	// At least one of TargetTime or MaxCost must be set.
	// +optional
	TargetTime string `json:"targetTime,omitempty"`

	// MaxCost is the maximum allowed spend in USD (e.g. "8.00").
	// At least one of TargetTime or MaxCost must be set.
	// +optional
	MaxCost string `json:"maxCost,omitempty"`

	// MaxNodes is a hard cap on the node count the solver may select.
	// +kubebuilder:validation:Minimum=2
	MaxNodes int32 `json:"maxNodes"`
}

// DistributedTrainingSpec defines the desired state of a DistributedTraining.
// +kubebuilder:validation:XValidation:rule="has(self.objective) || self.topology.nodes >= 1",message="spec.topology.nodes must be >= 1 when spec.objective is not set"
type DistributedTrainingSpec struct {
	// Backend selects the distributed-training framework. Defaults to pytorch.
	// +optional
	// +kubebuilder:default="pytorch"
	// +kubebuilder:validation:Enum=pytorch;spark;job
	Backend BackendType `json:"backend,omitempty"`

	// PytorchSpec configures the PyTorch job infrastructure.
	// Required when backend is "pytorch"; ignored otherwise.
	// +optional
	PytorchSpec *PytorchSpec `json:"pytorchSpec,omitempty"`

	// SparkSpec configures the Spark Operator backend.
	// Required when backend is "spark"; ignored otherwise.
	// +optional
	SparkSpec *SparkSpec `json:"sparkSpec,omitempty"`

	// Model identifies the language model to fine-tune.
	// Optional LLM overlay for pytorch; ignored by spark backend.
	// +optional
	Model *ModelSpec `json:"model,omitempty"`

	// Dataset identifies the Hugging Face dataset to train on.
	// Optional LLM overlay; ignored by generic backends.
	// +optional
	Dataset *DatasetSpec `json:"dataset,omitempty"`

	// Hardware selects the GKE node pool and machine type.
	// Required for spark backend. For pytorch, hardware type is inferred
	// from pytorchSpec.resources when not set.
	// +optional
	Hardware HardwareSpec `json:"hardware,omitempty"`

	// Topology configures the distributed job layout.
	// Required for spark backend (unless objective is set). For pytorch, derived from
	// pytorchSpec.masterReplicas + workerReplicas.
	// +optional
	Topology TopologySpec `json:"topology,omitempty"`

	// Optimization groups mixed-precision and LoRA settings.
	// Optional LLM overlay for pytorch backend.
	// +optional
	Optimization OptimizationSpec `json:"optimization,omitempty"`

	// Training holds the standard hyperparameters.
	// Optional LLM overlay for pytorch backend.
	// +optional
	Training *TrainingSpec `json:"training,omitempty"`

	// OutputPVCName is the name of the PVC for checkpoints and result files.
	// Must exist before the job is created.
	// +optional
	OutputPVCName string `json:"outputPVCName,omitempty"`

	// Objective declares time/cost constraints for automatic topology sizing.
	// Mutually exclusive with explicit topology fields.
	// +optional
	Objective *ObjectiveSpec `json:"objective,omitempty"`

	// ResumeFromCheckpoint is an explicit checkpoint path injected into the
	// torchrun command (e.g. "checkpoints/checkpoint-500"). PyTorch backend only.
	// Setting this on Spark or job backends emits a Warning and has no effect.
	// +optional
	ResumeFromCheckpoint string `json:"resumeFromCheckpoint,omitempty"`

	// RestartPolicy controls what happens when the underlying backend job
	// disappears unexpectedly (e.g. manually deleted) while the DistributedTraining
	// is still Running.
	// Never (default): mark the DistributedTraining as Failed.
	// OnFailure: recreate the backend job and continue from the last checkpoint if one exists.
	// +optional
	// +kubebuilder:default="Never"
	// +kubebuilder:validation:Enum=Never;OnFailure
	RestartPolicy string `json:"restartPolicy,omitempty"`
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

// JobResults holds metrics collected after a completed job run.
type JobResults struct {
	// TrainingTime is the wall-clock duration of pod runtime, computed from
	// the backend's own start/end timestamps (e.g.
	// PyTorchJob.status.startTime → PyTorchJob.status.completionTime).
	// Excludes image pull, pod scheduling, the operator-side result-collection
	// cycle, and any reconcile lag — matches the window measured by
	// baseline_train.sh and the solver's estimatedTime prediction.
	TrainingTime string `json:"trainingTime,omitempty"`

	// ProvisioningTime is the wall-clock duration of node-pool provisioning.
	ProvisioningTime string `json:"provisioningTime,omitempty"`

	// CollectionTime is the wall-clock duration spent in the Collecting phase
	// reading result files from the output PVC via a reader pod. It runs in
	// parallel with node-pool deletion, so it is NOT included in the cost
	// calculation — exposed for diagnostic transparency only.
	// +optional
	CollectionTime string `json:"collectionTime,omitempty"`

	// EstimatedCostUSD is the estimated total cost of the run in USD,
	// computed as nodes × hourly_rate × (provisioning + training) / 3600.
	// Collection time is excluded because the operator deletes the worker
	// node pool in parallel with result collection.
	// Omitted when the machine type is not present in the cost ConfigMap.
	// +optional
	EstimatedCostUSD string `json:"estimatedCostUSD,omitempty"`

	// Metrics is a free-form map of metric names to string values.
	//
	// PyTorch backend keys: "loss", "perplexity", "samplesPerSecond", "tokensPerSecond"
	// Spark backend keys:   "recordsProcessed", "processingTimeSeconds", "throughputRecordsPerSec"
	// +optional
	Metrics map[string]string `json:"metrics,omitempty"`
}

// ---------------------------------------------------------------------------
// Main status
// ---------------------------------------------------------------------------

// ResolvedTopology holds the topology determined by the objective solver.
// Nodes == 0 is the sentinel value for "not yet resolved".
// Value type in status — the zero value naturally represents "unresolved".
type ResolvedTopology struct {
	// Nodes is the solver-selected node count.
	Nodes int32 `json:"nodes"`

	// MasterReplicas is the master replica count (pytorch only).
	// +optional
	MasterReplicas int32 `json:"masterReplicas,omitempty"`

	// WorkerReplicas is the worker replica count (pytorch only).
	// +optional
	WorkerReplicas int32 `json:"workerReplicas,omitempty"`

	// EstimatedTime is the solver's predicted training duration.
	EstimatedTime string `json:"estimatedTime"`

	// EstimatedCost is the solver's predicted cost in USD.
	// Omitted when the machine type is not in the cost ConfigMap.
	// +optional
	EstimatedCost string `json:"estimatedCost,omitempty"`
}

// DistributedTrainingStatus defines the observed state of a DistributedTraining.
type DistributedTrainingStatus struct {
	// Phase is the current lifecycle stage of the job.
	// +optional
	Phase Phase `json:"phase,omitempty"`

	// Message is a human-readable description of the current state or last error.
	// +optional
	Message string `json:"message,omitempty"`

	// StartTime records when the controller first processed the job.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// ProvisioningStartTime records when node-pool provisioning began.
	// +optional
	ProvisioningStartTime *metav1.Time `json:"provisioningStartTime,omitempty"`

	// TrainingStartTime records when the backend job actually started running,
	// as reported by the backend itself (e.g. PyTorchJob.status.startTime for
	// Kubeflow). This is set on the first reconcile of the Running phase, NOT
	// when the operator submitted the backend job CR — so `training_seconds`
	// excludes image pull and pod scheduling overhead and matches the window
	// baseline_train.sh measures (status.startTime → status.completionTime).
	// +optional
	TrainingStartTime *metav1.Time `json:"trainingStartTime,omitempty"`

	// FinishTime records when the job reached Succeeded or Failed.
	// +optional
	FinishTime *metav1.Time `json:"finishTime,omitempty"`

	// JobName is the name of the generated backend job resource.
	// +optional
	JobName string `json:"jobName,omitempty"`

	// NodePoolSize is the number of nodes currently provisioned.
	// +optional
	NodePoolSize int32 `json:"nodePoolSize,omitempty"`

	// Results holds the collected job metrics. Populated in the Collecting phase.
	// +optional
	Results *JobResults `json:"results,omitempty"`

	// GKEOperationID is the name of the in-flight GKE long-running operation.
	// +optional
	GKEOperationID string `json:"gkeOperationID,omitempty"`

	// Conditions provides machine-readable status conditions.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ResolvedTopology holds the topology the solver selected in objective mode.
	// Nodes == 0 means the solver has not run yet.
	// +optional
	ResolvedTopology ResolvedTopology `json:"resolvedTopology,omitempty"`
}

// ---------------------------------------------------------------------------
// Root object
// ---------------------------------------------------------------------------

// DistributedTraining is the Schema for the distributedtrainings API.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dj;distjob,categories=kubeflow
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.spec.backend`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type DistributedTraining struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DistributedTrainingSpec   `json:"spec,omitempty"`
	Status DistributedTrainingStatus `json:"status,omitempty"`
}

// DistributedTrainingList contains a list of DistributedTraining.
//
// +kubebuilder:object:root=true
type DistributedTrainingList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DistributedTraining `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DistributedTraining{}, &DistributedTrainingList{})
}
