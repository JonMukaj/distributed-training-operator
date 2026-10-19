# The numeric cluster ID — feed this to the operator as --hyperstack-cluster-id
# (config/overlays/hyperstack/manager_args_patch.yaml).
output "cluster_id" {
  description = "Hyperstack cluster ID, used by the operator's --hyperstack-cluster-id flag."
  value       = hyperstack_core_cluster.this.id
}

output "cluster_name" {
  description = "Name of the created Hyperstack cluster."
  value       = hyperstack_core_cluster.this.name
}
