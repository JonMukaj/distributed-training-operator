# The numeric cluster ID — feed this to the operator as --hyperstack-cluster-id
# (config/overlays/hyperstack/manager_args_patch.yaml, or `make hs-configure`).
output "cluster_id" {
  description = "Hyperstack cluster ID, used by the operator's --hyperstack-cluster-id flag."
  value       = hyperstack_core_cluster.this.id
}

output "cluster_name" {
  description = "Name of the created Hyperstack cluster."
  value       = hyperstack_core_cluster.this.name
}

# The cluster's kubeconfig (client-certificate auth), decoded. Write it to a
# gitignored file for kubectl/helm:
#   terraform output -raw kubeconfig > kubeconfig.yaml
locals {
  # The provider hands kube_config through as the API returns it (base64);
  # fall back to raw YAML should a future provider version decode it.
  kubeconfig_yaml = try(
    base64decode(hyperstack_core_cluster.this.kube_config),
    hyperstack_core_cluster.this.kube_config,
  )
}

output "kubeconfig" {
  description = "Decoded kubeconfig for the cluster (sensitive)."
  value       = local.kubeconfig_yaml
  sensitive   = true
}
