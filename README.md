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
  --member="serviceAccount:YOUR_PROJECT.svc.id.goog[distributed-training-system/distributed-training-controller-manager]"
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
export HYPERSTACK_API_KEY=YOUR_HYPERSTACK_API_KEY
make hs-secret        # creates distributed-training-system + the hyperstack-api Secret
```

**2. Set your numeric cluster ID** in `config/overlays/hyperstack/manager_args_patch.yaml`:

```sh
make hs-configure HYPERSTACK_CLUSTER_ID=<id>
```

**3. Deploy using the Hyperstack overlay:**

```sh
kubectl kustomize config/overlays/hyperstack   # preview
kubectl apply -k config/overlays/hyperstack
```

This deploys the controller image pinned in `config/manager/kustomization.yaml`. To build
and deploy your own image in one go use `make hs-build-deploy IMG=<image:tag>` — see
[Step 5 — Build and deploy the operator](#step-5--build-and-deploy-the-operator)
in the Hyperstack development section.

Notes and differences from GKE:

- **Machine types are Hyperstack flavor names** (e.g. `n1-cpu-small`); a flavor bundles
  its GPU, so `spec.hardware.gpuType`/`gpuCount` are ignored on Hyperstack.
- Node-group names are limited to **20 characters**.
- Readiness is observed by polling node status (`ACTIVE`) through the Hyperstack API.
  The controller additionally counts Ready Kubernetes nodes using the
  `--hyperstack-nodepool-label` label (default `hyperstack.cloud/node-group-name`, verified
  on a live cluster; re-check with `kubectl get nodes --show-labels` after platform upgrades).
  **The same label is used as the replica pods' `nodeSelector`**, so if it is wrong the
  node group comes up but the PyTorchJob pods stay `Pending`.
- Node-group names must be ≤ 20 characters. The operator derives `dj-<CR name>`; for a
  longer CR name set `spec.hardware.nodePoolName` explicitly (see the GPU sample).
- Hyperstack does not remove the Kubernetes `Node` objects of a deleted node group (they
  linger as `NotReady,SchedulingDisabled`; GKE's cloud controller does this for you). The
  operator therefore deletes the pool's non-Ready Node objects itself once the cloud has
  confirmed the pool is gone — after a successful run and when a CR is deleted. Deleting
  a CR consequently keeps its finalizer until the node group is confirmed deleted
  (`kubectl delete dj … ` returns when the pool is gone).
- Result collection runs in parallel with node-group deletion. The reader pod that cats
  `all_results.json` off the PVC carries a node anti-affinity on the pool label so it is
  never scheduled onto the node being torn down; Hyperstack deletes VMs fast enough that
  this race lost a whole run's metrics before the fix (symptom: `job complete — 0 metrics
  collected` and a `HistorySkipped` event while the files exist on the NFS export).

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
> This applies the PVC and the CPU (GKE) sample only. The GPU sample targets
> Hyperstack (`config/samples/training_v1_distributedtraining_hyperstack_gpu.yaml`)
> and is applied on its own — see
> [Step 7 — Run the GPU fine-tuning sample](#step-7--run-the-gpu-fine-tuning-sample)
> in the Hyperstack section.

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

The `gke-rebuild` Makefile target handles image build/push, deployment and a sample run in
one command (it is `make cloud-rebuild CLOUD=gke`; every `gke-*` target has an `hs-*` twin
for Hyperstack and a generic `cloud-*` form, see `make help`):

```sh
make gke-rebuild IMG=jmukaj/distributed-training-operator:<tag>
```

This runs the following steps internally:

1. Checks the overlay for unfilled `YOUR_*` placeholders
2. Builds and pushes the Docker image and writes it into `config/manager/kustomization.yaml`
3. Applies `config/overlays/gke/` (which creates the `distributed-training-system` namespace) and the machine-costs ConfigMap
4. Waits for the deployment rollout
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
kubectl logs -f -n distributed-training-system deploy/distributed-training-controller-manager
```

(or `make gke-watch` / `make gke-logs`.)

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

> **Working directory.** Steps 1–2 run from `example/infra/hyperstack/` (Terraform).
> From Step 3 on, every command runs from `operator/`, where the Makefile lives; paths
> into `example/` are therefore written as `../example/...`.

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
cd example/infra/hyperstack          # from the repository root
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: environment_name, keypair_name, cluster_name
terraform init
terraform apply
```

Terraform manages the cloud resource only. Everything inside the cluster (CSI drivers,
NFS, Kubeflow, the operator, monitoring) is installed with `kubectl`/`helm` from
`operator/` in the following steps.

Record the cluster ID for the operator overlay:

```sh
terraform output cluster_id
```

---

### Step 2 — Connect kubectl to the cluster

Terraform already holds the cluster's kubeconfig (client-certificate auth) as a sensitive
output; write it to a file and point `kubectl` at it:

```sh
terraform output -raw kubeconfig > kubeconfig.yaml      # gitignored
export KUBECONFIG=$PWD/kubeconfig.yaml
kubectl get nodes
```

<details>
<summary>Without Terraform: fetch it from the API</summary>

```sh
B64_KUBECONFIG=$(curl -s https://infrahub-api.nexgencloud.com/v1/core/clusters/<cluster_id> \
  -H "api_key: $HYPERSTACK_API_KEY" | jq -r '.cluster.kube_config')
echo "$B64_KUBECONFIG" | base64 -d > kubeconfig.yaml
```
</details>

---

### Step 3 — Install cluster dependencies

```sh
cd ../../../operator                 # all remaining steps run from operator/
```

**CSI driver** (dynamic block storage for the NFS server):

```sh
helm repo add nexgencloud https://nexgencloud.github.io/csi-hyperstack
helm repo update
helm install csi-hyperstack nexgencloud/csi-hyperstack --namespace kube-system --set hyperstack.apiKey=$HYPERSTACK_API_KEY
# Wait until the node plugin has registered on every node; applying the NFS server
# before that fails its first attach with "CSINode ... does not contain driver
# hyperstack.csi.nexgencloud.com" (it would retry and recover, but the wait avoids the noise).
kubectl rollout status -n kube-system ds/csi-hyperstack-node --timeout=180s
kubectl get csinode -o custom-columns='NODE:.metadata.name,DRIVERS:.spec.drivers[*].name'   # expect hyperstack.csi.nexgencloud.com on the workers
```

**NFS server** (re-exports an RWO volume as `ReadWriteMany`):

```sh
kubectl apply -f ../example/infra/demo/nfs-hyperstack.yaml
kubectl wait --for=condition=Available deployment/nfs-server --timeout=180s
```

**NFS CSI driver and the shared output PVC** every job references as `spec.outputPVCName`.
Hyperstack node images have no NFS client, so the in-tree `nfs:` volume used on GKE
(`config/samples/pvc.yaml`) fails there with `bad option; ... you might need a
/sbin/mount.nfs helper`. The CSI driver mounts from inside its own pod instead:

```sh
helm repo add csi-driver-nfs https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/master/charts
helm repo update
helm install csi-driver-nfs csi-driver-nfs/csi-driver-nfs --namespace kube-system
kubectl rollout status -n kube-system ds/csi-nfs-node --timeout=180s
kubectl apply -f config/samples/pvc-nfs-csi.yaml       # NOT pvc.yaml on Hyperstack
```

> Switching an existing cluster from `pvc.yaml`: delete any running `DistributedTraining`
> first (its pod holds the claim), then `kubectl delete -f config/samples/pvc.yaml` and
> apply `pvc-nfs-csi.yaml`. Data on the NFS server is untouched; only the PV object changes.

**Kubeflow Training Operator** (required for `PyTorchJob` support):

```sh
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
kubectl wait --for=condition=Available deployment/training-operator -n kubeflow --timeout=120s
```

---

### Step 3b — Monitoring: Prometheus, Grafana, Pushgateway

With the CSI driver in place (the Prometheus volume comes from it):

```sh
make monitoring-install CLOUD=hyperstack GRAFANA_PASSWORD=<password>   # default admin
kubectl -n monitoring get pods
```

This applies the `prometheus-retain` StorageClass, installs kube-prometheus-stack and the
Pushgateway into `monitoring` with `config/monitoring/values-hyperstack.yaml`, publishes
the operator dashboard (`kubectl apply -k config/grafana`) and creates the ServiceMonitor
(`config/monitoring/servicemonitor.yaml`) that scrapes the operator's plain-HTTP metrics
Service on port 8080 — that Service is part of the Hyperstack overlay
(`config/prometheus-http`), so it arrives with the operator in Step 5.

Design points, all in `config/monitoring/`:

- Prometheus/Grafana/Pushgateway are pinned to the base worker group via
  `hyperstack.cloud/node-group-name: default`. Ephemeral GPU node groups carry no taint on
  Hyperstack, so without the pin Prometheus could land on a GPU node and die, volume
  attached, when the operator deletes that group.
- The TSDB lives on a Hyperstack `Cloud-SSD` volume (`prometheus-retain`,
  `reclaimPolicy: Retain`, 10Gi, retention 90d). Retain means the volume outlives the PVC —
  `make monitoring-uninstall` deletes the PVC so it is detached before a cluster teardown.

Open Grafana:

```sh
make monitoring-grafana          # http://localhost:3000 → dashboard "DistributedTraining Operator"
```

---

### Step 4 — Configure the Hyperstack overlay

Write the cluster ID (from Step 1) into the overlay, then create the namespace and the
API-key Secret (read from `$HYPERSTACK_API_KEY`):

```sh
make hs-configure HYPERSTACK_CLUSTER_ID=<id> HYPERSTACK_GPU_FLAVOR=n3-L40x1
make hs-init
```

`hs-configure` edits `config/overlays/hyperstack/manager_args_patch.yaml` in place
(`--hyperstack-cluster-id`, `--default-gpu-machine-type`); `hs-init` creates
`distributed-training-system` and the `hyperstack-api` Secret in it.

The node label the operator uses to find a node group's nodes is the platform label
`hyperstack.cloud/node-group-name`, which Hyperstack puts on every worker. It is the
built-in default, so nothing to configure. Should a platform upgrade ever rename it,
find the new key and pass it once as `HYPERSTACK_NODE_LABEL=<key>`:

```sh
# list a worker's hyperstack.cloud/* labels; the key whose value is the group name
# (e.g. "default") is the one the operator needs
kubectl get node $(kubectl get nodes -l node-role.kubernetes.io/worker -o jsonpath='{.items[0].metadata.name}') \
  -o jsonpath='{.metadata.labels}' | tr ',' '\n' | grep hyperstack

kubectl get nodes -L hyperstack.cloud/node-group-name
```

> **Node label matters twice.** The operator counts Ready nodes with it **and** pins the
> training pods to the node group with a `nodeSelector` on it. With a wrong label the
> node group comes up and the PyTorchJob pods stay `Pending`.
>
> The manager runs in `distributed-training-system` (set by `config/default/kustomization.yaml`,
> which also prefixes every resource with `distributed-training-`). The Secret must live in
> that namespace or the pod fails with `CreateContainerConfigError`.

---

### Step 5 — Build and deploy the operator

Install the CRDs, then build, push and deploy in one step:

```sh
# Regenerate manifests/deepcopy after any API change
make manifests generate

# Install CRDs
make install

# Build + push the controller image, point the manifests at it, deploy the Hyperstack
# overlay and the machine-costs ConfigMap, wait for the rollout
make hs-build-deploy IMG=jmukaj/distributed-training-operator:<tag>
```

`hs-build-deploy` refuses to run while the overlay still has `YOUR_*` placeholders
(`make cloud-check CLOUD=hyperstack` runs the same test). To change only overlay flags
without rebuilding, use `make hs-apply-config`; note it also resets the sample CR, which on
Hyperstack is the paid GPU sample.

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
make hs-watch     # kubectl get dj -A -w
make hs-logs      # kubectl logs -f -n distributed-training-system deploy/distributed-training-controller-manager
```

Expected phase sequence: `Pending → Provisioning → Ready → Running → Collecting → Succeeded`

With Step 3b in place the same run shows up in Grafana (provisioning/training seconds,
nodes, cost, solver α and estimates, training metrics incl. the transfer fractions):

```sh
make monitoring-grafana
```

---

### Step 7 — Run the GPU fine-tuning sample

`config/samples/training_v1_distributedtraining_hyperstack_gpu.yaml` is the GPU
counterpart of the CPU sample: the same smol_llama-101M-GQA LoRA workload on an
ephemeral **`n3-L40x1`** node group (one NVIDIA L40 48 GB per node, 28 vCPU, 58 GB RAM,
region CANADA-1; `n3-A100x1` is the commented alternative in the file), image
`jmukaj/gpu-fine-tuner-hf:v2.1.0`. It is
deliberately *not* in `config/samples/kustomization.yaml`, so `kubectl apply -k
config/samples/` never starts a GPU job by accident.

**Prerequisites (once per cluster)**

1. Hyperstack CSI driver, NFS server, NFS CSI driver and the shared PVC from Step 3
   (`../example/infra/demo/nfs-hyperstack.yaml`, `config/samples/pvc-nfs-csi.yaml`), and the
   Kubeflow training operator.
2. The flavor's hourly price in the machine-costs ConfigMap — needed only for
   objective mode. `config/samples/machine-costs-configmap.yaml` lists
   `n3-L40x1: "1.00"` and `n3-A100x1: "1.35"` (console on-demand prices);
   `hs-build-deploy` applies the ConfigMap. Without a price for the flavor in use,
   objective-mode CRs fall back to a calibration run.
3. The node-group label (`hyperstack.cloud/node-group-name`, the built-in default; see
   Step 4). The sample pins pods to the node group with it; a wrong label leaves the pods
   `Pending`.
4. For gated models only: `kubectl create secret generic hf-token-secret
   --from-literal=token=<hf_...>` and uncomment `model.hfTokenSecret`. Never commit
   the token.

NVIDIA drivers and the device plugin are preinstalled on Hyperstack clusters; nothing
to apply. Hyperstack flavors bundle their GPU, so `hardware.gpuType`/`gpuCount` are
ignored and the sample pins the GPU with the `nvidia.com/gpu: "1"` limit.

**Run sequence** — the solver needs two node counts in history before it can fit α:

```sh
# Calibration run 1 (nodes: 1 as shipped)
kubectl apply -f config/samples/training_v1_distributedtraining_hyperstack_gpu.yaml
kubectl get dj distributedtraining-llm-gpu-1 -w        # wait for Succeeded

# Calibration run 2: set topology.nodes: 2, a new metadata.name and a new
# hardware.nodePoolName (≤ 20 chars), apply again.

# Confirm both history entries exist
kubectl get distributedtraininghistories

# Objective run: uncomment the second document in the file (solver picks the node
# count), apply, and check the chosen topology before the node group is provisioned.
kubectl get dj distributedtraining-llm-gpu-obj-1 -o jsonpath='{.status.resolvedTopology}{"\n"}'
```

All three CRs must keep identical `model`, `dataset` and `training` fields: they form
the config hash that groups history entries (see [Config hash](#config-hash)).
Watch the node group appear and disappear in the Hyperstack console under the
cluster's **Node groups** tab.

**Reading the results**

```sh
kubectl get dj distributedtraining-llm-gpu-1 -o jsonpath='{.status.results}' | jq .
kubectl get distributedtraininghistories -o yaml      # throughput, α inputs, cost per run
```

Raw files (`all_results.json`, `transfer_profile.json`, `train_results.json`,
`eval_results.json`, `trainer_state.json` with the per-step loss curve; the `v2.1.0` image
carries the instrumented `script/gpu/finetune.py`) land under `/exports/checkpoints` on the
NFS server — and **every job overwrites them**, since all jobs share that output directory.
Archive each run before starting the next:

```sh
make hs-archive-results CR=distributedtraining-llm-gpu-3           # RUN=<dir name> defaults to the CR name
```

This copies the JSON files to `/exports/results/<RUN>/` on the PVC (outside the path the next
job overwrites), pulls them to `../results/<RUN>/` on your machine (`RESULTS_DIR` to change),
and dumps the CR (`cr.yaml`) and its history entry (`history.yaml`) next to them.

The first run on a fresh PVC downloads the model and dataset into `/mnt/output/hf-cache`
(`HF_HOME` in the sample), so its training time is inflated. Treat it as a warm-up and
do not use it as a measured run.

> **How the operator attributes time.** `provisioning_seconds` runs from the node-pool
> request until every node is kubelet-Ready **and, for GPU jobs, advertises an allocatable
> `nvidia.com/gpu`** — the device plugin needs ~2 min after a fresh node joins, and that
> wait belongs to provisioning. `training_seconds` runs from the PyTorchJob's `Running`
> condition (all replica pods running) to its completion, so pod scheduling and image pull
> sit between the two and in neither. Provisioning fails after 30 min if the nodes never
> become usable. (Before these two changes the first Hyperstack runs booked a 173 s
> training loop as 399 s of training time and the solver fitted α≈1 to infrastructure
> overhead.)

Delete a finished CR when you no longer need its status (the node group is already
torn down by the operator):

```sh
kubectl delete dj distributedtraining-llm-gpu-1
```

---

### Teardown

Delete the storage workload **before** destroying the cluster. If the node VMs are
deleted while the CSI volume is still attached, the volume gets stuck in Hyperstack's
`RESERVED` state — still billed, and recoverable only via the console or support.

```sh
# 1. Remove workloads so the CSI driver detaches the volumes (→ AVAILABLE) while the
#    nodes still exist.
make hs-clean                                       # CRs (waits for node groups to go), controller, CRDs
kubectl delete -f config/samples/pvc-nfs-csi.yaml   # shared output PVC/PV
kubectl delete -f ../example/infra/demo/nfs-hyperstack.yaml      # NFS server + CSI-backed disk PVC

# 2. Monitoring: Helm does not delete the Prometheus StatefulSet's PVC; this target
#    uninstalls the charts AND deletes the PVC so the Retain volume is detached before
#    the node VMs go (otherwise it gets stuck RESERVED).
make monitoring-uninstall

# 3. Destroy the cluster.
cd ../example/infra/hyperstack && terraform destroy
```

> **Volume lifecycle.** The NFS disk's `nfs-hyperstack` StorageClass uses
> `reclaimPolicy: Delete`: deleting the `nfs-disk` PVC deletes the Hyperstack volume, so
> nothing keeps billing — and the HF cache, checkpoints and anything under
> `/exports/results` go with it. Archive every run first (`make hs-archive-results`).
> The Prometheus volume (`prometheus-retain`) is kept on purpose; after
> `make monitoring-uninstall` it sits detached on the
> [Volumes page](https://console.hyperstack.cloud/volumes) until you delete it there.
>
> Switching an existing cluster from the former `nfs-retain` class: StorageClasses are
> immutable, so `kubectl delete sc nfs-retain`, re-apply `nfs-hyperstack.yaml`, and patch
> the already-provisioned PV, whose policy was fixed when it was created:
>
> ```sh
> kubectl patch pv $(kubectl get pvc nfs-disk -o jsonpath='{.spec.volumeName}') \
>   -p '{"spec":{"persistentVolumeReclaimPolicy":"Delete"}}'
> ```

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
