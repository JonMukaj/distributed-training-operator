terraform {
  required_version = ">= 1.3"
  required_providers {
    hyperstack = {
      source  = "NexGenCloud/hyperstack"
      version = "1.56.0-alpha"
    }
  }
}

# Authentication is read from the HYPERSTACK_API_KEY environment variable:
#   export HYPERSTACK_API_KEY=your-api-key-here
# (the same key the operator's hyperstack provider uses). The provider block is
# intentionally empty — do not put the key in Terraform state or source.
#
# Scope: this module manages the cloud resource only (the cluster). Everything
# inside the cluster — CSI drivers, NFS, Kubeflow, the operator, monitoring — is
# installed with kubectl/helm from operator/ (see README and the Makefile
# `monitoring-*` targets), so no Kubernetes/Helm providers are configured here
# and a single `terraform apply` suffices.
provider "hyperstack" {}
