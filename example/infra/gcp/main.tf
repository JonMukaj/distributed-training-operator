
resource "google_service_account" "gke_service_account" {
  account_id   = "gke-cluster-sa"
  display_name = "GKE Cluster Service Account"
  description  = "Service account for GKE cluster"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/container.admin",
    "roles/compute.viewer",
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.gke_service_account.email}"
}

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  remove_default_node_pool = true
  initial_node_count       = 1

  # node_config {
  #   disk_size_gb = var.disk_size_gb
  #   machine_type = var.machine_type
  # }

  network                  = google_compute_network.vpc.name
  subnetwork               = google_compute_subnetwork.subnet.name
  enable_l4_ilb_subsetting = true
  datapath_provider        = "ADVANCED_DATAPATH"

  resource_labels = {
    environment = var.environment
    assignment  = var.assignment
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pod-range"
    services_secondary_range_name = "services-range"
    stack_type                    = "IPV4_IPV6"
  }


  # Workload Identity configuration
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Cluster addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = true
    }
  }

  monitoring_config {
    managed_prometheus {
      enabled = false
    }
    enable_components = ["SYSTEM_COMPONENTS"]
  }

  # Vertical Pod Autoscaling
  vertical_pod_autoscaling {
    enabled = false
  }

  deletion_protection = false
  depends_on = [google_project_iam_member.gke_sa_roles]

  lifecycle {
    ignore_changes = [node_config]
  }
}

# Allow the operator's Kubernetes ServiceAccount to impersonate the GKE SA.
# This is the Workload Identity binding — no key file or secret needed.
# KSA: distributed-training-system/distributed-training-controller-manager (namePrefix applied by kustomize)
resource "google_service_account_iam_member" "operator_wi" {
  service_account_id = google_service_account.gke_service_account.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[distributed-training-system/distributed-training-controller-manager]"
}

# The operator creates node pools that attach gke-cluster-sa as the node service account.
# GKE requires iam.serviceAccountUser on the target SA even when it's the same SA as the caller.
resource "google_service_account_iam_member" "operator_sa_user" {
  service_account_id = google_service_account.gke_service_account.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.gke_service_account.email}"
}


# General purpose Node Pool
resource "google_container_node_pool" "general" {
  name       = "${var.cluster_name}-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count

  node_config {

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
    }

    machine_type = var.machine_type
    disk_type    = "pd-ssd"
    disk_size_gb = var.disk_size_gb

    tags = ["gke-node", "${var.project_id}-gke"]

    # Workload Identity
    service_account = google_service_account.gke_service_account.email
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }
  lifecycle {
    ignore_changes = [
      node_config[0].resource_labels,
      node_config[0].kubelet_config,
      node_count,
    ]
  }
}

# resource "google_container_node_pool" "worker" {
#  name       = "${var.cluster_name}-worker-node-pool"
#  location   = var.zone
#  cluster    = google_container_cluster.primary.name
#  node_count = var.worker_node_count

#  node_config {
#    dynamic "guest_accelerator" {
#      for_each = var.use_gpu ? [1] : []
#      content {
#        type  = var.gpu_type
#        count = var.worker_node_count
#      }
#    }
#    oauth_scopes = [
#      "https://www.googleapis.com/auth/cloud-platform"
#    ]

#    labels = {
#      environment = var.environment
#      role        = "worker"
#    }

#    machine_type = var.worker_machine_type
#   #  disk_type    = var.worker_disk_type
#    disk_size_gb = 50

#    tags = ["gke-node", "${var.project_id}-gke"]

#    # Workload Identity
#    service_account = google_service_account.gke_service_account.email
#    workload_metadata_config {
#      mode = "GKE_METADATA"
#    }

#    taint {
#      key    = "reserved-pool"
#      value  = "true"
#      effect = "NO_SCHEDULE"
#    }
#  }

#  lifecycle {
#    ignore_changes = [
#      node_config[0].resource_labels,
#      node_config[0].kubelet_config
#    ]
#  }
# }