project_id     = "praktikum-475820"
# project_id     = "jmukaj-praktikum1"
cluster_name   = "praktikum1"
region         = "us-east1"
zone           = "us-east1-d"
environment    = "production"
node_count     = 2
min_node_count = 2
max_node_count = 2
disk_size_gb   = 100
machine_type   = "e2-standard-2"
assignment     = "praktikum1"
worker_machine_type = "c4-highcpu-8"
worker_node_count = 2
worker_disk_type = "hyperdisk-balanced"
# e2-highmem-2 -> 0.41
# c4-highmem-4 -> 0.91
# e2-highmem-4 -> 0.68
# e2-highmem-8 -> 1.23
# "c4-highmem-2" # 0.52
# "c4-highcpu-8" -> 0.8
# "g2-standard-8" -> gpu