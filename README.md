# distributed-training-operator

A Kubernetes operator for running distributed machine-learning training jobs on GKE
or Hyperstack. It provisions ephemeral node pools, submits PyTorch (Kubeflow) or Spark
workloads, monitors their execution, and tears down the infrastructure once training
completes.

## Table of Contents

- [Description](#description)
- [Business Use Case](#business-use-case)
- [Job Lifecycle](#job-lifecycle)
- [History Tracking](#history-tracking)
- [Topology Solver](#topology-solver)
- [Getting Started](#getting-started)
- [Project Distribution](#project-distribution)
- [Development and Deployment on GKE](#development-and-deployment-on-gke)
- [Development and Deployment on Hyperstack](#development-and-deployment-on-hyperstack)
- [Contributing](#contributing)
- [License](#license)

---

## Description

`distributed-training-operator` manages the full lifecycle of a `DistributedTraining` custom resource.
Users declare what they want — a model, a dataset, and optionally a time or cost objective —
and the operator handles provisioning, scheduling, monitoring, and cleanup automatically.

Supported training backends:

| Backend | Resource created | Use case |
| --- | --- | --- |
| `pytorch` (default) | Kubeflow `PyTorchJob` | LLM fine-tuning, distributed deep learning |
| `spark` | Spark Operator `SparkApplication` | Large-scale data processing |
| `job` | Kubernetes `batch/v1 Job` | Single-node workloads |

---

## Business Use Case

Running distributed ML training on cloud infrastructure involves two hard problems:

1. **Right-sizing** — too few nodes means missing deadlines; too many wastes money on
   diminishing returns from parallelism.
2. **Infrastructure lifecycle** — manually spinning up node pools, submitting jobs, monitoring
   progress, collecting results, and tearing down nodes is error-prone and repetitive.

`distributed-training-operator` solves both. Teams describe their training workload as a
`DistributedTraining` CR and express constraints like `targetTime: 2h` or `maxCost: "8.00"`.
The operator provisions exactly the right cluster size, runs the job end-to-end, writes
cost and throughput metrics back to the CR status, and deletes the node pool when done —
no manual steps required.

Over multiple runs the operator learns how a given workload scales and improves its
node-count predictions, reducing both cost and wall-clock time across the team's
training iterations.

---

## Job Lifecycle

Each `DistributedTraining` moves through the following phases:

| Phase | What the operator does |
| --- | --- |
| **Pending** | Validates the spec; runs the topology solver if `spec.objective` is set |
| **Provisioning** | Creates an ephemeral GKE node pool; polls until nodes are `Ready` |
| **Ready** | Submits the backend job manifest (PyTorchJob / SparkApplication / Job) |
| **Running** | Monitors pod logs and backend job conditions |
| **Collecting** | Scrapes metrics (loss, throughput, cost); writes a `DistributedTrainingHistory` CR; deletes the node pool |
| **Succeeded** | Records final results in `status.results`; node pool already gone |
| **Failed** | Terminal state; node pool deleted; error written to `status.message` |

---

## History Tracking

After every successful job the operator writes a `DistributedTrainingHistory` CR in the same
namespace. These records are the training data for the topology solver.

### What is recorded

| Field | Description |
| --- | --- |
| `nodes` | Number of nodes used in this run |
| `throughput` | Samples/sec (PyTorch) or records/sec (Spark) |
| `trainingSeconds` | Wall-clock training duration |
| `totalWork` | `trainingSeconds × throughput` — the workload volume W |
| `provisioningSeconds` | Time spent waiting for the node pool to become ready |
| `actualCostUSD` | Estimated cost based on the machine-costs ConfigMap |
| `machineType` | GCE machine type used |
| `configHash` | SHA-256 fingerprint of the job configuration (see below) |

### Config hash

Runs of the "same job" are grouped by a SHA-256 hash of the fields that define the
workload, so the solver only learns from comparable runs:

- **PyTorch:** backend, machine type, model name, dataset name/split, batch size, gradient
  accumulation steps, epochs, validation split.
- **Spark:** backend, machine type, container image, main application file, arguments.

### Retention

At most **10** history entries are kept per config hash. When a new entry would exceed
this limit, the oldest entry for that hash is deleted first.

---

## Topology Solver

When `spec.objective` is set, users declare constraints instead of an explicit node count:

```yaml
spec:
  objective:
    targetTime: "2h"      # maximum acceptable wall-clock duration
    maxCost: "8.00"       # maximum spend in USD
    maxNodes: 8           # hard upper bound on node count
```

The solver runs at the end of the **Pending** phase and writes its result to
`status.resolvedTopology` before provisioning begins.

### Calibration run

If no history exists for the job's config hash, the solver cannot estimate the scaling
model yet. It schedules a **calibration run** using `defaultCalibrationNodes` (configured
via the `--default-calibration-nodes` flag, default `2`). The objective is applied from
the second run onward, once real throughput data is available.

### Scaling model

The solver fits a **scaling overhead** model to the recorded throughput values:

```text
η(n) = 1 / (1 + α(n − 1))          # parallel efficiency
P(n) = p_baseline × n × η(n)        # predicted throughput at n nodes
T(n) = W / P(n)                     # predicted training time
Cost(n) = C_h × n × (T(n) + T_prov_avg) / 3600   # predicted cost in USD
```

| Symbol | Meaning |
| --- | --- |
| `α` | Serial fraction — how quickly efficiency drops as nodes are added |
| `p_baseline` | Throughput per node at the smallest observed node count |
| `W` | Total work (`T × P` averaged at the minimum node count) |
| `T_prov_avg` | Average node-pool provisioning time across all history entries |
| `C_h` | Hourly cost per node, read from the `machine-costs` ConfigMap |

### Estimating α

The serial fraction `α` is fitted from historical throughput measurements grouped by node
count:

- **1 distinct node count in history** — `α = 0` (linear scaling assumed).
- **2 distinct node counts** — closed-form solution from the observed efficiency ratio.
- **3 or more distinct node counts** — 1-D golden-section search that minimises the
  sum of squared residuals between `P(n)` and the measured throughput averages.

### Constraint branches

| `spec.objective` | Solver strategy |
| --- | --- |
| `targetTime` only | Binary-search for the minimum `n` such that `T(n) ≤ targetTime` |
| `maxCost` only | Binary-search for the maximum `n` such that `Cost(n) ≤ maxCost`; if cost is flat across the range, pick `maxNodes` (fastest) |
| Both | Find `n_time` first; if `Cost(n_time) ≤ maxCost` both constraints are satisfied; otherwise fall back to the largest affordable `n` and emit a warning |

If `C_h` is not in the cost ConfigMap, cost constraints are skipped and only the time
constraint is evaluated.

### Machine costs ConfigMap

The solver reads hourly prices from a ConfigMap in the operator's namespace:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: machine-costs
  namespace: distributed-training-system
data:
  e2-standard-4: "0.134"
  n1-standard-4: "0.150"
  n1-standard-8: "0.300"
```

---

## Getting Started

### Prerequisites

- go version v1.25.3+
- docker version 17.03+.
- kubectl version v1.11.3+.
- Access to a GKE cluster with Workload Identity enabled.

### To Deploy on the cluster

**Build and push your image to the location specified by `IMG`:**

```sh
make docker-build docker-push IMG=<some-registry>/operator:tag
```

**NOTE:** This image ought to be published in the personal registry you specified.
And it is required to have access to pull the image from the working environment.
Make sure you have the proper permission to the registry if the above commands don't work.

**Install the CRDs into the cluster:**

```sh
make install
```

**Deploy the Manager to the cluster with the image specified by `IMG`:**

```sh
make deploy IMG=<some-registry>/operator:tag
```

> **NOTE**: If you encounter RBAC errors, you may need to grant yourself cluster-admin
privileges or be logged in as admin.

### Deploying on GKE

The operator uses a Kustomize overlay at `config/overlays/gke/` that injects GKE-specific
flags and wires up Workload Identity so no credential files are needed inside the pod.

**1. Enable Workload Identity on your cluster** (if not already enabled):

```sh
gcloud container clusters update YOUR_CLUSTER_NAME \
  --workload-pool=YOUR_PROJECT.svc.id.goog \
  --region=us-east1
```

**2. Create a GCP service account with node pool permissions:**

```sh
gcloud iam service-accounts create distributed-training-operator \
  --display-name="Distributed Job Operator"

gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --role=roles/container.nodeAdmin \
  --member=serviceAccount:distributed-training-operator@YOUR_PROJECT.iam.gserviceaccount.com
```

**3. Bind the GCP service account to the Kubernetes service account:**

```sh
gcloud iam service-accounts add-iam-policy-binding \
  distributed-training-operator@YOUR_PROJECT.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:YOUR_PROJECT.svc.id.goog[operator-system/operator-controller-manager]"
```

**4. Fill in your project and cluster values** in the overlay patch files:

- `config/overlays/gke/manager_args_patch.yaml` — set `--gcp-project` and `--cluster-name`

**5. Deploy using the GKE overlay:**

```sh
# Preview what will be applied
kubectl kustomize config/overlays/gke

# Apply
kubectl apply -k config/overlays/gke
```

#### Deploying on Hyperstack

The operator also supports [Hyperstack](https://www.hyperstack.cloud/) (NexGen Cloud)
managed Kubernetes clusters. Instead of GKE node pools it manages Hyperstack
**node groups** via the core REST API.

**1. Create an API key** in the [Hyperstack console](https://console.hyperstack.cloud/api-keys)
and store it as a Kubernetes Secret (the operator reads it from `HYPERSTACK_API_KEY`,
never from a command-line flag):

```sh
kubectl create secret generic hyperstack-api \
  --from-literal=api-key=YOUR_HYPERSTACK_API_KEY \
  -n operator-system
```

**2. Set your numeric cluster ID** in `config/overlays/hyperstack/manager_args_patch.yaml`
(the `--hyperstack-cluster-id` flag).

**3. Deploy using the Hyperstack overlay:**

```sh
kubectl kustomize config/overlays/hyperstack   # preview
kubectl apply -k config/overlays/hyperstack
```

Notes and differences from GKE:

- **Machine types are Hyperstack flavor names** (e.g. `n1-cpu-small`); a flavor bundles
  its GPU, so `spec.hardware.gpuType`/`gpuCount` are ignored on Hyperstack.
- Node-group names are limited to **20 characters**.
- Readiness is observed by polling node status (`ACTIVE`) through the Hyperstack API.
  The controller additionally counts Ready Kubernetes nodes using the
  `--hyperstack-nodepool-label` label (default `hyperstack.cloud/node-group`) — verify
  this against `kubectl get nodes --show-labels` on your cluster and override if needed.

#### Adding support for another cloud provider

The overlay pattern is designed to extend. To add a new provider (e.g. EKS, AKS):

1. Implement `cloud.Provider` in `internal/cloud/<provider>/`
2. Add provider flags and a `case` in `cmd/main.go`
3. Create `config/overlays/<provider>/` with:
   - `kustomization.yaml` referencing `../../default`
   - `manager_args_patch.yaml` with `--cloud-provider=<provider>` and its flags
   - a service-account / credentials patch for the provider's auth (e.g. Workload
     Identity on GKE, or an API-key Secret env var on Hyperstack)

#### Create instances of your solution

You can apply the samples (examples) from the config/sample:

```sh
kubectl apply -k config/samples/
```

> **NOTE**: Ensure that the samples has default values to test it out.

### To Uninstall

**Delete the instances (CRs) from the cluster:**

```sh
kubectl delete -k config/samples/
```

**Delete the APIs(CRDs) from the cluster:**

```sh
make uninstall
```

**UnDeploy the controller from the cluster:**

```sh
make undeploy
```

---

## Project Distribution

Following the options to release and provide this solution to the users.

### By providing a bundle with all YAML files

1. Build the installer for the image built and published in the registry:

```sh
make build-installer IMG=<some-registry>/operator:tag
```

**NOTE:** The makefile target mentioned above generates an `install.yaml`
file in the dist directory. This file contains all the resources built
with Kustomize, which are necessary to install this project without its
dependencies.

1. Using the installer

Users can just run `kubectl apply -f <YAML-BUNDLE-URL>` to install the project, i.e.:

```sh
kubectl apply -f https://raw.githubusercontent.com/<org>/operator/<tag or branch>/dist/install.yaml
```

### By providing a Helm Chart

1. Build the chart using the optional helm plugin

```sh
kubebuilder edit --plugins=helm/v2-alpha
```

1. See that a chart was generated under `dist/chart`, and users
can obtain this solution from there.

**NOTE:** If you change the project, you need to update the Helm Chart
using the same command above to sync the latest changes. Furthermore,
if you create webhooks, you need to use the above command with
the '--force' flag and manually ensure that any custom configuration
previously added to `dist/chart/values.yaml` or `dist/chart/manager/manager.yaml`
is manually re-applied afterwards.

---

## Development and Deployment on GKE

This section covers the operator development workflow assuming a GKE cluster
is already running and `kubectl` is pointed at it.

---

### Step 1 — Authenticate with GCP

```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project-id>
```

---

### Step 2 — Connect kubectl to the cluster

Install the GKE auth plugin if not already present:

```sh
gcloud components install gke-gcloud-auth-plugin
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
```

Fetch credentials:

```sh
gcloud container clusters get-credentials <your-cluster-name> \
  --zone=<your-gcp-zone> \
  --project=<your-project-id>
```

Verify:

```sh
kubectl get nodes
```

---

### Step 3 — Configure the GKE overlay

```sh
cp config/overlays/gke/values.env.example config/overlays/gke/values.env
# edit values.env with your real values, then:
source config/overlays/gke/values.env
sed -i "s/YOUR_PROJECT_ID/$GCP_PROJECT/g; s/YOUR_CLUSTER_NAME/$CLUSTER_NAME/g; \
        s/YOUR_GCP_ZONE/$GCP_ZONE/g; s/YOUR_SERVICE_ACCOUNT/$SERVICE_ACCOUNT/g" \
  config/overlays/gke/manager_args_patch.yaml \
  config/overlays/gke/serviceaccount_wi_patch.yaml
```

---

### Step 4 — Install CRDs and dependencies

```sh
make install
```

Install the Kubeflow Training Operator (required for `PyTorchJob` support):

```sh
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
kubectl wait --for=condition=Available deployment/training-operator -n kubeflow --timeout=120s
```

---

### Step 5 — Deploy the controller

The `gke-rebuild` Makefile target handles namespace creation, image build/push, and deployment in one command:

```sh
make gke-rebuild IMG=jmukaj/distributed-training-operator:<tag>
```

This runs the following steps internally:

1. Creates the `operator-system` namespace (idempotent)
2. Builds and pushes the Docker image
3. Applies `config/overlays/gke/` via kubectl
4. Rolls out the deployment and waits for it to be ready
5. Deletes and re-applies the sample `DistributedTraining` CR

The controller authenticates to GCP via **Workload Identity** — the GKE metadata server issues
tokens automatically using the binding created in the [Deploying on GKE](#deploying-on-gke) section.
No credential files are mounted inside the pod.

> **ADC fallback:** If Workload Identity is not available on your cluster, the `gke-init` target
> can mount your local ADC credentials as a Kubernetes secret instead:
>
> ```sh
> make gke-init   # creates the gcp-adc secret from ~/.config/gcloud/application_default_credentials.json
> ```
>
> Then uncomment the `env` / `volumeMounts` / `volumes` blocks in
> `config/overlays/gke/manager_args_patch.yaml` before deploying.

---

### Step 6 — Watch the operator

```sh
# Watch phase transitions
kubectl get dj -w

# Tail controller logs
kubectl logs -f -n operator-system deploy/operator-controller-manager
```

Expected phase sequence: `Pending → Provisioning → Ready → Running → Collecting → Succeeded`

Check the node pool being created on GKE:

```sh
gcloud container node-pools list \
  --cluster=<your-cluster-name> \
  --zone=<your-gcp-zone> \
  --project=<your-project-id>
```

---

### Teardown

To delete the controller, CRDs, and all CRs:

```sh
make gke-clean
```

---

### Updating the GCP project ID

If you need to point the controller at a different GCP project, update the overlay and redeploy:

```sh
sed -i 's/--gcp-project=.*/--gcp-project=<new-project-id>/' config/overlays/gke/manager_args_patch.yaml
make gke-rebuild IMG=jmukaj/distributed-training-operator:<tag>
```

---

## Development and Deployment on Hyperstack

This section covers the operator development workflow on a
[Hyperstack](https://www.hyperstack.cloud/) (NexGen Cloud) managed Kubernetes cluster.
The operator manages Hyperstack **node groups** (instead of GKE node pools) via its
`hyperstack` cloud provider.

### Prerequisites

- **Tooling:** `go` v1.25.3+, `docker` 17.03+, `kubectl` v1.11.3+, `helm` 3+,
  `terraform` v1.0+, and `curl` + `jq` (to fetch the cluster kubeconfig).
- **Hyperstack account** with an [API key](https://console.hyperstack.cloud/api-keys).
  The same key is used by the operator, the CSI driver, and Terraform — export it once:

  ```sh
  export HYPERSTACK_API_KEY=<your-api-key>
  ```
- **A Hyperstack environment and an SSH keypair** (created beforehand in the console or
  API) — the cluster is provisioned into them.
- **CSI note:** to use dynamic block storage (for the NFS server disk) the cluster must
  have been created on or after **2025-09-04** (or scaled by one node since), and the API
  key needs the `volume:*` and `virtual-machine:volume-attach/detach` permissions.

---

### Step 1 — Provision the cluster (Terraform)

```sh
cd example/infra/hyperstack
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: environment_name, keypair_name, cluster_name
terraform init
terraform apply
```

Record the cluster ID for the operator overlay:

```sh
terraform output cluster_id
```

---

### Step 2 — Connect kubectl to the cluster

The cluster's `kube_config` field (on the cluster-detail response) holds a
base64-encoded kubeconfig. Retrieve it, decode it, and point `kubectl` at it — replace
`<cluster_id>` with the ID from Step 1:

```sh
B64_KUBECONFIG=$(curl -s https://infrahub-api.nexgencloud.com/v1/core/clusters/<cluster_id> \
  -H "api_key: $HYPERSTACK_API_KEY" | jq -r '.cluster.kube_config')

echo "$B64_KUBECONFIG" | base64 -d > kubeconfig.yaml
export KUBECONFIG=$PWD/kubeconfig.yaml
kubectl get nodes
```

---

### Step 3 — Install cluster dependencies

**CSI driver** (dynamic block storage for the NFS server):

```sh
helm repo add nexgencloud https://nexgencloud.github.io/csi-hyperstack
helm repo update
helm install csi-hyperstack nexgencloud/csi-hyperstack \
  --namespace kube-system --set hyperstack.apiKey=$HYPERSTACK_API_KEY
```

**NFS server** (re-exports an RWO volume as the `ReadWriteMany` output PVC the jobs use):

```sh
kubectl apply -f example/infra/demo/nfs-hyperstack.yaml
```

**Kubeflow Training Operator** (required for `PyTorchJob` support):

```sh
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
kubectl wait --for=condition=Available deployment/training-operator -n kubeflow --timeout=120s
```

---

### Step 4 — Configure the Hyperstack overlay

Create the namespace and the API-key secret the operator reads:

```sh
kubectl create namespace operator-system
kubectl create secret generic hyperstack-api \
  --from-literal=api-key=$HYPERSTACK_API_KEY \
  -n operator-system
```

Set your cluster ID (from Step 1) in
`config/overlays/hyperstack/manager_args_patch.yaml` (the `--hyperstack-cluster-id` flag).

> **Node label:** the operator counts Ready nodes using the
> `--hyperstack-nodepool-label` flag (default `hyperstack.cloud/node-group`). Verify it
> against your cluster with `kubectl get nodes --show-labels` and override it in the
> overlay if the node-group label differs.

---

### Step 5 — Build and deploy the operator

Install the CRDs, build/push the image, and apply the Hyperstack overlay:

```sh
# Regenerate manifests/deepcopy after any API change
make manifests generate

# Install CRDs
make install

# Build and push the controller image
make docker-build docker-push IMG=jmukaj/distributed-training-operator:<tag>

# Deploy with the Hyperstack overlay
kubectl apply -k config/overlays/hyperstack
kubectl rollout status -n operator-system deploy/operator-controller-manager
```

> **Local dev loop:** to iterate without building an image, run the controller locally
> against the cluster (uses your current kubeconfig). `make run` takes no flags, so pass
> them to `go run` directly:
>
> ```sh
> # HYPERSTACK_API_KEY must be exported in your shell
> go run ./cmd/main.go --cloud-provider=hyperstack --hyperstack-cluster-id=<id>
> ```

After editing `*.go` files run `make lint` and `make test` before committing.

---

### Step 6 — Watch the operator

```sh
# Watch phase transitions
kubectl get dj -w

# Tail controller logs
kubectl logs -f -n operator-system deploy/operator-controller-manager
```

Expected phase sequence: `Pending → Provisioning → Ready → Running → Collecting → Succeeded`

---

### Teardown

Delete the storage workload **before** destroying the cluster. If the node VMs are
deleted while the CSI volume is still attached, the volume gets stuck in Hyperstack's
`RESERVED` state — still billed, and recoverable only via the console or support.

```sh
# 1. Remove workloads so the CSI driver detaches the volume (→ AVAILABLE) while the
#    nodes still exist.
kubectl delete -f example/infra/demo/nfs-hyperstack.yaml         # NFS server + PVC
kubectl delete -k config/overlays/hyperstack       # controller
make uninstall                                      # CRDs

# 2. Destroy the cluster.
cd example/infra/hyperstack && terraform destroy
```

> **Retain note:** the `nfs-retain` StorageClass uses `reclaimPolicy: Retain`, so
> deleting the PVC **detaches** the Hyperstack volume but does **not** delete it — the
> volume (with your dataset) stays on the
> [Volumes page](https://console.hyperstack.cloud/volumes) and keeps billing. Delete it
> there when you no longer need the data, or set the StorageClass to
> `reclaimPolicy: Delete` to have the CSI driver remove it automatically with the PVC.

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch from `main`.
2. Run `make test` and `make lint` before opening a PR — both must pass.
3. Keep PRs focused: one feature or fix per PR.
4. Add or update tests for any changed behaviour.
5. Open a pull request with a clear description of what changed and why.

**NOTE:** Run `make help` for more information on all potential `make` targets

More information can be found via the [Kubebuilder Documentation](https://book.kubebuilder.io/introduction.html)

---

## License

Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
