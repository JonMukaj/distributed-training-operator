# A Hyperstack (NexGen Cloud) managed Kubernetes cluster.
#
# This provisions the base cluster only — a control plane plus a small default
# worker node group. The distributed-training-operator then attaches ephemeral
# node groups to this cluster per training job (see the operator's hyperstack
# cloud provider and config/overlays/hyperstack), using the cluster_id output
# below as --hyperstack-cluster-id.
#
# Prerequisites that are NOT managed by this module (create them first via the
# Hyperstack console or API, then pass their names in):
#   - a Hyperstack environment  -> var.environment_name
#   - a Hyperstack SSH keypair   -> var.keypair_name
resource "hyperstack_core_cluster" "this" {
  name               = var.cluster_name
  environment_name   = var.environment_name
  keypair_name       = var.keypair_name
  kubernetes_version = var.kubernetes_version
  image_name         = var.image_name
  master_flavor_name = var.master_flavor_name
  node_flavor_name   = var.node_flavor_name
  node_count         = var.node_count
}
