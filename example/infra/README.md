# Infrastructure

Terraform for the clusters the `distributed-training-operator` runs on. Two
independent stacks, one per cloud:

| Subfolder | Cloud | What it creates |
|---|---|---|
| [`gcp/`](gcp/) | Google Cloud (GKE) | GKE cluster, VPC/subnet, node pool, NFS disk, Workload Identity service account, and the Grafana/Prometheus stack |
| [`hyperstack/`](hyperstack/) | Hyperstack (NexGen Cloud) | A managed Kubernetes cluster via the official `hyperstack_core_cluster` Terraform resource |

Each subfolder is a standalone Terraform root — `cd` into one and run
`terraform init/plan/apply` there. Pick the stack that matches where you want the
operator to provision training node pools; the operator targets one cluster at a
time via its `--cloud-provider` flag (`gke` or `hyperstack`).

See each subfolder's own notes for prerequisites and variables
([`hyperstack/README.md`](hyperstack/README.md); the `gcp/` stack is driven by the
`*.tfvars` files in that folder).
