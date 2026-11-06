#!/usr/bin/env bash
# Bootstrap a Hyperstack cluster and the distributed-training operator, end to end.
#
# Automates README "Development and Deployment on Hyperstack", Steps 1-5 and 3b:
#   1  Terraform: create the cluster            (example/infra/hyperstack)
#   2  Write kubeconfig, wait for nodes
#   3  CSI driver, NFS server, NFS CSI, shared PVC, Kubeflow training operator
#   3b Prometheus / Grafana / Pushgateway       (make monitoring-install)
#   4  Hyperstack overlay: cluster ID, namespace, API-key Secret
#   5  CRDs, manifests, build/deploy the operator
#
# It stops before Step 7: no GPU node group is started, so nothing billable
# runs for training. Start experiments with `make run-experiment EXP=...`.
#
# Usage (from anywhere):
#   export HYPERSTACK_API_KEY=...
#   export GRAFANA_PASSWORD=...                # optional; default Grafana login is admin/admin
#   [IMG=jmukaj/distributed-training-operator:<tag>] script/hyperstack-bootstrap.sh
#
# Resume at a later step (earlier steps are skipped; each step is idempotent):
#   START_STEP=4 script/hyperstack-bootstrap.sh     # runs steps 4 and 5 only
#   START_STEP=3 script/hyperstack-bootstrap.sh     # runs steps 3, 3b, 4 and 5
#
# Environment:
#   HYPERSTACK_API_KEY      required; used by the operator, CSI driver and Terraform
#   GRAFANA_PASSWORD        optional; unset = the Makefile default (admin)
#   IMG                     image to build and push (needs `docker login`). Unset =
#                           deploy the image already pinned in config/manager/kustomization.yaml
#   HYPERSTACK_GPU_FLAVOR   default n3-L40x1
#   SKIP_MONITORING=1       skip Step 3b
#   START_STEP=N            resume at step N (1-5, 3b is run with 3); each step is idempotent
#
# Prerequisites: terraform, kubectl, helm 3, jq, curl, make, go (and docker if IMG is set),
# a populated example/infra/hyperstack/terraform.tfvars, and the Hyperstack environment
# and SSH keypair named in it.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$REPO_ROOT/example/infra/hyperstack"
OPERATOR_DIR="$REPO_ROOT/operator"
KUBECONFIG_FILE="$INFRA_DIR/kubeconfig.yaml"   # gitignored in example/infra/hyperstack

IMG="${IMG:-}"
GPU_FLAVOR="${HYPERSTACK_GPU_FLAVOR:-n3-L40x1}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-}"
SKIP_MONITORING="${SKIP_MONITORING:-0}"
START_STEP="${START_STEP:-1}"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
# want N: succeeds when step N should run, i.e. START_STEP <= N.
# Two usage patterns:
#   if want 1; then ...block...; fi     # run a block only when resuming at or before step 1
#   want 3 || return 0                  # skip the rest of a function when resuming past step 3
# Example: START_STEP=4 skips steps 1-3 and 3b, and runs 4 and 5.
want() { (( START_STEP <= $1 )); }

# ---------------------------------------------------------------- preflight
preflight() {
  log "Preflight"
  local tool
  for tool in terraform kubectl helm jq curl make go; do
    command -v "$tool" >/dev/null || die "missing tool: $tool"
  done
  if [[ -n "$IMG" ]]; then
    command -v docker >/dev/null || die "IMG is set but docker is not installed"
  fi
  [[ -n "${HYPERSTACK_API_KEY:-}" ]] || die "export HYPERSTACK_API_KEY first"
  [[ -f "$INFRA_DIR/terraform.tfvars" ]] || \
    die "$INFRA_DIR/terraform.tfvars is missing: cp terraform.tfvars.example terraform.tfvars and fill it in"
  echo "ok (START_STEP=$START_STEP, GPU flavor=$GPU_FLAVOR, IMG=${IMG:-<pinned in config/manager>})"
}

# ---------------------------------------------------------------- step 1 + 2
step_cluster() {
  if want 1; then
    log "Step 1: Terraform (cluster)"
    (
      cd "$INFRA_DIR"
      terraform init -input=false
      terraform apply
    )
  fi

  log "Step 2: kubeconfig"
  local cluster_id
  cluster_id="$(cd "$INFRA_DIR" && terraform output -raw cluster_id)"
  [[ -n "$cluster_id" ]] || die "terraform output cluster_id is empty: has the cluster been applied?"
  (cd "$INFRA_DIR" && terraform output -raw kubeconfig > "$KUBECONFIG_FILE")
  chmod 600 "$KUBECONFIG_FILE"
  export KUBECONFIG="$KUBECONFIG_FILE"
  echo "cluster_id=$cluster_id  kubeconfig=$KUBECONFIG_FILE"
  kubectl wait --for=condition=Ready nodes --all --timeout=600s
  kubectl get nodes
  CLUSTER_ID="$cluster_id"
}

# ---------------------------------------------------------------- step 3
step_dependencies() {
  want 3 || return 0
  export KUBECONFIG="$KUBECONFIG_FILE"
  cd "$OPERATOR_DIR"

  log "Step 3: Hyperstack CSI driver"
  helm repo add nexgencloud https://nexgencloud.github.io/csi-hyperstack --force-update
  helm repo update nexgencloud
  helm upgrade --install csi-hyperstack nexgencloud/csi-hyperstack \
    --namespace kube-system --set hyperstack.apiKey="$HYPERSTACK_API_KEY"
  kubectl rollout status -n kube-system ds/csi-hyperstack-node --timeout=180s
  kubectl get csinode -o custom-columns='NODE:.metadata.name,DRIVERS:.spec.drivers[*].name'

  log "Step 3: NFS server"
  kubectl apply -f "$REPO_ROOT/example/infra/demo/nfs-hyperstack.yaml"
  kubectl wait --for=condition=Available deployment/nfs-server --timeout=180s

  log "Step 3: NFS CSI driver and shared output PVC"
  helm repo add csi-driver-nfs https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/master/charts --force-update
  helm repo update csi-driver-nfs
  helm upgrade --install csi-driver-nfs csi-driver-nfs/csi-driver-nfs --namespace kube-system
  kubectl rollout status -n kube-system ds/csi-nfs-node --timeout=180s
  kubectl apply -f config/samples/pvc-nfs-csi.yaml

  log "Step 3: Kubeflow training operator"
  kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
  kubectl wait --for=condition=Available deployment/training-operator -n kubeflow --timeout=120s
}

# ---------------------------------------------------------------- step 3b
step_monitoring() {
  want 3 || return 0
  [[ "$SKIP_MONITORING" == "1" ]] && { log "Step 3b: skipped (SKIP_MONITORING=1)"; return 0; }
  export KUBECONFIG="$KUBECONFIG_FILE"
  log "Step 3b: monitoring"
  if [[ -n "$GRAFANA_PASSWORD" ]]; then
    make -C "$OPERATOR_DIR" monitoring-install CLOUD=hyperstack GRAFANA_PASSWORD="$GRAFANA_PASSWORD"
  else
    make -C "$OPERATOR_DIR" monitoring-install CLOUD=hyperstack   # Makefile default: admin/admin
  fi
  kubectl -n monitoring get pods
}

# ---------------------------------------------------------------- step 4 + 5
step_operator() {
  export KUBECONFIG="$KUBECONFIG_FILE"
  cd "$OPERATOR_DIR"

  if want 4; then
    log "Step 4: configure Hyperstack overlay"
    make hs-configure HYPERSTACK_CLUSTER_ID="$CLUSTER_ID" HYPERSTACK_GPU_FLAVOR="$GPU_FLAVOR"
    make hs-init
  fi

  if want 5; then
    log "Step 5: CRDs and manifests"
    make manifests generate
    make install

    if [[ -n "$IMG" ]]; then
      log "Step 5: build, push and deploy $IMG"
      make hs-build-deploy IMG="$IMG"
    else
      log "Step 5: deploy (no build; image pinned in config/manager)"
      make hs-deploy
    fi
  fi
}

# ---------------------------------------------------------------- main
preflight
step_cluster
step_dependencies
step_monitoring
step_operator

log "Done"
cat <<EOF
Cluster ${CLUSTER_ID} is up with the operator deployed. No GPU node group has been started.

  export KUBECONFIG=$KUBECONFIG_FILE
  make -C $OPERATOR_DIR hs-watch          # watch DistributedTraining phases
  make -C $OPERATOR_DIR monitoring-grafana # http://localhost:3000

Next: Step 7 in README.md (calibration runs, then the thesis series via make run-experiment EXP=...).
Teardown: make -C $OPERATOR_DIR monitoring-uninstall BEFORE destroying the cluster.
EOF
