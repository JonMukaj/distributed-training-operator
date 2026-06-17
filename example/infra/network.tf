# VPC Network
resource "google_compute_network" "vpc" {
  name                            = "${var.cluster_name}-vpc"
  auto_create_subnetworks         = false
  routing_mode                    = "REGIONAL"
  enable_ula_internal_ipv6        = true
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.name

  stack_type       = "IPV4_IPV6"
  ipv6_access_type = "INTERNAL"
  secondary_ip_range {
    range_name    = "pod-range"
    ip_cidr_range = "10.4.0.0/14" 
  }

  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "10.1.0.0/20"
  }
}