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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DistributedTrainingHistorySpec stores throughput metrics from a single completed run.
// Entries are written for every Succeeded run (objective and explicit-topology alike).
// Failed runs are excluded — partial throughput data would bias the solver.
type DistributedTrainingHistorySpec struct {
	// Backend is the job backend that produced this entry (pytorch or spark).
	Backend BackendType `json:"backend"`

	// MachineType is the GCE machine type used for this run.
	MachineType string `json:"machineType"`

	// ConfigHash is the backend-aware hash of the job configuration fields
	// that affect throughput. Used to group entries for the same logical workload.
	ConfigHash string `json:"configHash"`

	// Nodes is the actual node count used in this run (n_k).
	Nodes int32 `json:"nodes"`

	// Throughput is the measured throughput P_k — tokens/sec for PyTorch, records/sec for Spark.
	Throughput resource.Quantity `json:"throughput"`

	// TrainingSeconds is the training duration T_k in seconds.
	TrainingSeconds resource.Quantity `json:"trainingSeconds"`

	// TotalWork is the approximated total work W ≈ T_k × P_k.
	TotalWork resource.Quantity `json:"totalWork"`

	// ProvisioningSeconds is the node pool provisioning time T_provision in seconds.
	ProvisioningSeconds resource.Quantity `json:"provisioningSeconds"`

	// ActualCostUSD is the observed run cost in USD at the time of execution.
	// Stored for observability only — the solver recomputes cost from current pricing.
	// +optional
	ActualCostUSD string `json:"actualCostUSD,omitempty"`
}

// DistributedTrainingHistory is the Schema for per-run throughput history used by the topology solver.
//
// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=djh
type DistributedTrainingHistory struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec DistributedTrainingHistorySpec `json:"spec,omitempty"`
}

// DistributedTrainingHistoryList contains a list of DistributedTrainingHistory.
//
// +kubebuilder:object:root=true
type DistributedTrainingHistoryList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DistributedTrainingHistory `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DistributedTrainingHistory{}, &DistributedTrainingHistoryList{})
}
