# Hyperstack cluster (Terraform)

Provisions a [Hyperstack](https://www.hyperstack.cloud/) (NexGen Cloud) managed
Kubernetes cluster that the `distributed-training-operator` runs on. The operator
attaches ephemeral node groups to this cluster per training job via its Hyperstack
cloud provider.

Uses the official (alpha) [`NexGenCloud/hyperstack`](https://registry.terraform.io/providers/NexGenCloud/hyperstack)
Terraform provider and its `hyperstack_core_cluster` resource.

> **Note:** the Hyperstack Terraform provider is in **alpha** (`1.56.0-alpha`).
> Some resources may not be fully stable; check `terraform plan` output carefully.

## Prerequisites (not managed by this module)

Create these first in the Hyperstack console or API, then reference them by name:

- **API key** — generate at <https://console.hyperstack.cloud/api-keys>.
- **Environment** — pass its name as `environment_name`.
- **SSH keypair** — pass its name as `keypair_name`.

## Usage

```sh
# 1. Authenticate (the provider reads this env var; the same key the operator uses)
export HYPERSTACK_API_KEY=your-api-key-here

# 2. Configure
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: environment_name, keypair_name, cluster_name

# 3. Deploy the cluster
terraform init
terraform apply

# 4. Kubeconfig for kubectl/helm (sensitive output, file is gitignored)
terraform output -raw kubeconfig > kubeconfig.yaml
export KUBECONFIG=$PWD/kubeconfig.yaml
```

This module manages the cluster only. In-cluster components (CSI drivers, NFS, Kubeflow,
the operator, Prometheus/Grafana via `make monitoring-install`) are installed with
kubectl/helm from `operator/` — see the root README, Hyperstack section. Run
`make monitoring-uninstall` and delete the NFS workload before `terraform destroy` so the
CSI volumes are detached (NFS disk: deleted; Prometheus: retained) while the nodes still
exist — a volume still attached when its VM disappears gets stuck RESERVED.

## Wiring the cluster to the operator

After `apply`, Terraform prints the cluster ID:

```sh
terraform output cluster_id
```

Use that value for the operator's `--hyperstack-cluster-id` flag in
[`operator/config/overlays/hyperstack/manager_args_patch.yaml`](../../../operator/config/overlays/hyperstack/manager_args_patch.yaml),
and create the API-key secret in the operator's namespace:

```sh
cd ../../../operator
make hs-configure HYPERSTACK_CLUSTER_ID=$(terraform -chdir=../example/infra/hyperstack output -raw cluster_id)
make hs-secret   # reads $HYPERSTACK_API_KEY, creates the Secret in distributed-training-system
```

Then deploy the operator with `make hs-build-deploy IMG=<image:tag>` (or
`kubectl apply -k config/overlays/hyperstack` to deploy the image already pinned in
`config/manager/kustomization.yaml`). See the root README, Hyperstack Steps 4–6.
