variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
  default     = "my-gke-cluster"
}

variable "region" {
  description = "The default location for project resources"
  type        = string
  default     = "us-west1"
}

variable "zone" {
  description = "The zone to deploy the cluster in"
  type        = string
  default     = "us-west1-a"
}

variable "environment" {
  description = "Environment name for tagging"
  type        = string
  default     = "development"
}

variable "node_count" {
  description = "Number of nodes per zone"
  type        = number
  default     = 1
}

variable "min_node_count" {
  description = "Minimum number of nodes in the node pool"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes in the node pool"
  type        = number
  default     = 3
}

variable "disk_size_gb" {
  description = "Disk size for nodes"
  type        = number
  default     = 64
}

variable "machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "e2-medium"
}


variable "worker_machine_type" {
  description = "Machine type for worker nodes"
  type        = string
  default     = "e2-medium"
}

variable "worker_disk_type" {
  type = string
  default = "pd-ssd"
}

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/20"
}


variable "assignment" {
  description = "Master assignment cluster belongs to"
  type = string
  default = "CC"
}

variable "use_gpu" {
  description = "Add accelerator to the cluster"
  type = bool
  default = false
}

variable "gpu_type" {
  description = "Type of GPU to attach to nodes"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach to nodes"
  type        = number
  default     = 1
}

variable "worker_node_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 2
}