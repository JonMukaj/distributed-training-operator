variable "cluster_name" {
  description = "Name of the Hyperstack Kubernetes cluster."
  type        = string
  default     = "distributed-training"
}

variable "environment_name" {
  description = "Name of an existing Hyperstack environment to deploy the cluster into. Must be created beforehand (console or API)."
  type        = string
}

variable "keypair_name" {
  description = "Name of an existing Hyperstack SSH keypair for node access. Must be created beforehand (console or API)."
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster."
  type        = string
  default     = "1.36.1"
}

variable "master_flavor_name" {
  description = "Flavor for the master node(s). Masters must use a CPU-only flavor."
  type        = string
  default     = "n1-cpu-medium"
}

variable "node_flavor_name" {
  description = "Flavor for the initial worker node group (the operator provisions additional node groups per training job)."
  type        = string
  default     = "n1-cpu-medium"
}

variable "master_count" {
  description = "Number of master nodes.(2-3)"
  type        = number
  default     = 2
}


variable "node_count" {
  description = "Number of worker nodes in the initial (default) node group. The operator adds ephemeral node groups on top of this."
  type        = number
  default     = 2
}
