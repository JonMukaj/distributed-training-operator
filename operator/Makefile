# Image URL to use all building/pushing image targets
IMG ?= controller:latest

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# CONTAINER_TOOL defines the container tool to be used for building images.
# Be aware that the target commands are only tested with Docker which is
# scaffolded by default. However, you might want to replace it to use other
# tools. (i.e. podman)
CONTAINER_TOOL ?= docker

# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk command is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: manifests
manifests: controller-gen ## Generate WebhookConfiguration, ClusterRole and CustomResourceDefinition objects.
	"$(CONTROLLER_GEN)" rbac:roleName=manager-role crd webhook paths="./..." output:crd:artifacts:config=config/crd/bases

.PHONY: generate
generate: controller-gen ## Generate code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	"$(CONTROLLER_GEN)" object:headerFile="hack/boilerplate.go.txt" paths="./..."

.PHONY: fmt
fmt: ## Run go fmt against code.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...

.PHONY: test
test: manifests generate fmt vet setup-envtest ## Run tests.
	KUBEBUILDER_ASSETS="$(shell "$(ENVTEST)" use $(ENVTEST_K8S_VERSION) --bin-dir "$(LOCALBIN)" -p path)" go test $$(go list ./... | grep -v /e2e) -coverprofile cover.out

# TODO(user): To use a different vendor for e2e tests, modify the setup under 'tests/e2e'.
# The default setup assumes Kind is pre-installed and builds/loads the Manager Docker image locally.
# CertManager is installed by default; skip with:
# - CERT_MANAGER_INSTALL_SKIP=true
KIND_CLUSTER ?= operator-test-e2e

.PHONY: setup-test-e2e
setup-test-e2e: ## Set up a Kind cluster for e2e tests if it does not exist
	@command -v $(KIND) >/dev/null 2>&1 || { \
		echo "Kind is not installed. Please install Kind manually."; \
		exit 1; \
	}
	@case "$$($(KIND) get clusters)" in \
		*"$(KIND_CLUSTER)"*) \
			echo "Kind cluster '$(KIND_CLUSTER)' already exists. Skipping creation." ;; \
		*) \
			echo "Creating Kind cluster '$(KIND_CLUSTER)'..."; \
			$(KIND) create cluster --name $(KIND_CLUSTER) ;; \
	esac

.PHONY: test-e2e
test-e2e: setup-test-e2e manifests generate fmt vet ## Run the e2e tests. Expected an isolated environment using Kind.
	KIND=$(KIND) KIND_CLUSTER=$(KIND_CLUSTER) go test -tags=e2e ./test/e2e/ -v -ginkgo.v
	$(MAKE) cleanup-test-e2e

.PHONY: cleanup-test-e2e
cleanup-test-e2e: ## Tear down the Kind cluster used for e2e tests
	@$(KIND) delete cluster --name $(KIND_CLUSTER)

.PHONY: lint
lint: golangci-lint ## Run golangci-lint linter
	"$(GOLANGCI_LINT)" run

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes
	"$(GOLANGCI_LINT)" run --fix

.PHONY: lint-config
lint-config: golangci-lint ## Verify golangci-lint linter configuration
	"$(GOLANGCI_LINT)" config verify

##@ Build

.PHONY: build
build: manifests generate fmt vet ## Build manager binary.
	go build -o bin/manager cmd/main.go

.PHONY: run
run: manifests generate fmt vet ## Run a controller from your host.
	go run ./cmd/main.go

# If you wish to build the manager image targeting other platforms you can use the --platform flag.
# (i.e. docker build --platform linux/arm64). However, you must enable docker buildKit for it.
# More info: https://docs.docker.com/develop/develop-images/build_enhancements/
.PHONY: docker-build
docker-build: ## Build docker image with the manager.
	$(CONTAINER_TOOL) build -t ${IMG} .

.PHONY: docker-push
docker-push: ## Push docker image with the manager.
	$(CONTAINER_TOOL) push ${IMG}

# PLATFORMS defines the target platforms for the manager image be built to provide support to multiple
# architectures. (i.e. make docker-buildx IMG=myregistry/mypoperator:0.0.1). To use this option you need to:
# - be able to use docker buildx. More info: https://docs.docker.com/build/buildx/
# - have enabled BuildKit. More info: https://docs.docker.com/develop/develop-images/build_enhancements/
# - be able to push the image to your registry (i.e. if you do not set a valid value via IMG=<myregistry/image:<tag>> then the export will fail)
# To adequately provide solutions that are compatible with multiple platforms, you should consider using this option.
PLATFORMS ?= linux/arm64,linux/amd64,linux/s390x,linux/ppc64le
.PHONY: docker-buildx
docker-buildx: ## Build and push docker image for the manager for cross-platform support
	# copy existing Dockerfile and insert --platform=${BUILDPLATFORM} into Dockerfile.cross, and preserve the original Dockerfile
	sed -e '1 s/\(^FROM\)/FROM --platform=\$$\{BUILDPLATFORM\}/; t' -e ' 1,// s//FROM --platform=\$$\{BUILDPLATFORM\}/' Dockerfile > Dockerfile.cross
	- $(CONTAINER_TOOL) buildx create --name operator-builder
	$(CONTAINER_TOOL) buildx use operator-builder
	- $(CONTAINER_TOOL) buildx build --push --platform=$(PLATFORMS) --tag ${IMG} -f Dockerfile.cross .
	- $(CONTAINER_TOOL) buildx rm operator-builder
	rm Dockerfile.cross

.PHONY: build-installer
build-installer: manifests generate kustomize ## Generate a consolidated YAML with CRDs and deployment.
	mkdir -p dist
	cd config/manager && "$(KUSTOMIZE)" edit set image controller=${IMG}
	"$(KUSTOMIZE)" build config/default > dist/install.yaml

##@ Cloud Deployment (GKE or Hyperstack)

# One set of targets for both clouds. Pick the cloud with CLOUD=gke (default)
# or CLOUD=hyperstack; the overlay and sample CR follow from it. The gke-* and
# hs-* targets below are shorthands that pin CLOUD, so
#   make hs-build-deploy IMG=...   ==   make cloud-build-deploy CLOUD=hyperstack IMG=...
CLOUD          ?= gke
OVERLAY        ?= config/overlays/$(CLOUD)
OPERATOR_NS    ?= distributed-training-system
MANAGER_DEPLOY ?= distributed-training-controller-manager
ifeq ($(CLOUD),hyperstack)
SAMPLE_CR      ?= config/samples/training_v1_distributedtraining_hyperstack_gpu.yaml
else
SAMPLE_CR      ?= config/samples/training_v1_distributedtraining.yaml
endif
ADC_CREDS_FILE ?= $(HOME)/.config/gcloud/application_default_credentials.json

.PHONY: cloud-check
cloud-check: ## Fail if $(OVERLAY) still contains YOUR_* placeholders (cluster ID, project, ...).
	@test -d $(OVERLAY) || { echo "unknown CLOUD=$(CLOUD): $(OVERLAY) not found"; exit 1; }
	@if grep -v '^\s*#' $(OVERLAY)/manager_args_patch.yaml | grep -q 'YOUR_'; then \
		echo "$(OVERLAY)/manager_args_patch.yaml still has placeholders:"; \
		grep -v '^\s*#' $(OVERLAY)/manager_args_patch.yaml | grep -n 'YOUR_'; \
		echo "(Hyperstack: run 'make hs-configure HYPERSTACK_CLUSTER_ID=<id>')"; exit 1; fi

.PHONY: cloud-init
cloud-init: ## Create the operator namespace (and, on Hyperstack, the API-key Secret from the HYPERSTACK_API_KEY env var).
	kubectl create namespace $(OPERATOR_NS) --dry-run=client -o yaml | kubectl apply -f -
ifeq ($(CLOUD),hyperstack)
	$(MAKE) hs-secret
endif
# 	kubectl create secret generic gcp-adc \
# 		--from-file=application_default_credentials.json=$(ADC_CREDS_FILE) \
# 		--namespace=$(OPERATOR_NS) \
# 		--dry-run=client -o yaml | kubectl apply -f -

.PHONY: cloud-build-deploy
cloud-build-deploy: cloud-check docker-build docker-push ## Build + push $(IMG), point manifests at it, apply $(OVERLAY) and the machine-costs ConfigMap.
	cd config/manager && "$(KUSTOMIZE)" edit set image controller=${IMG}
	kubectl apply -k $(OVERLAY)
	kubectl apply -f config/samples/machine-costs-configmap.yaml
	kubectl rollout status -n $(OPERATOR_NS) deploy/$(MANAGER_DEPLOY) --timeout=180s

.PHONY: cloud-apply-config
cloud-apply-config: cloud-check cloud-init ## Re-apply $(OVERLAY) without rebuilding, wait for the controller, then reset the sample CR.
	kubectl apply -k $(OVERLAY)
	kubectl wait --for=condition=Available deployment/$(MANAGER_DEPLOY) -n $(OPERATOR_NS) --timeout=120s
	$(MAKE) cloud-reset-cr CLOUD=$(CLOUD)

.PHONY: cloud-reset-cr
cloud-reset-cr: ## Delete all DistributedTraining CRs and apply $(SAMPLE_CR). Hyperstack: that is the GPU sample (provisions a paid node group).
	kubectl delete distributedtraining --all --all-namespaces --ignore-not-found=true
	kubectl apply -f config/samples/machine-costs-configmap.yaml
	kubectl apply -f $(SAMPLE_CR)

.PHONY: cloud-rebuild
cloud-rebuild: cloud-build-deploy cloud-reset-cr ## Full redeploy: build, push, deploy controller, reset sample CR.

.PHONY: cloud-watch
cloud-watch: ## Watch DistributedTraining phase transitions.
	kubectl get dj -A -w

.PHONY: cloud-logs
cloud-logs: ## Tail the controller logs.
	kubectl logs -f -n $(OPERATOR_NS) deploy/$(MANAGER_DEPLOY)

.PHONY: cloud-clean
cloud-clean: ## Delete all CRs (waits for node pools/groups to go), the controller ($(OVERLAY)) and the CRDs.
	kubectl delete distributedtraining --all --all-namespaces --ignore-not-found=true --wait=true
	kubectl delete -k $(OVERLAY) --ignore-not-found=true
	$(MAKE) uninstall ignore-not-found=true

# ---------------------------------------------------------------------------
# Archiving a run. Every job writes to the same /mnt/output/checkpoints, so the
# raw JSON files (all_results.json, transfer_profile.json, train/eval_results,
# trainer_state.json with the per-step loss curve) are overwritten by the next
# run. Only the history CR persists the key numbers. Run this after each job.
# ---------------------------------------------------------------------------
NFS_NS      ?= default
RESULTS_DIR ?= ../results
RUN         ?= $(CR)

.PHONY: archive-results
archive-results: ## Copy the last run's JSON files off the PVC + dump its CR and history to $(RESULTS_DIR)/$(RUN)/. Usage: make archive-results CR=<DistributedTraining name> [RUN=<dir name>]
	@test -n "$(CR)" || { echo "usage: make archive-results CR=<DistributedTraining name> [RUN=<dir name, default CR>] [RESULTS_DIR=$(RESULTS_DIR)]"; exit 1; }
	@NFS=$$(kubectl -n $(NFS_NS) get pod -l role=nfs-server -o jsonpath='{.items[0].metadata.name}'); \
	test -n "$$NFS" || { echo "no nfs-server pod (label role=nfs-server) in namespace $(NFS_NS)"; exit 1; }; \
	echo "archiving /exports/checkpoints/*.json from $$NFS into /exports/results/$(RUN)/ and $(RESULTS_DIR)/$(RUN)/"; \
	kubectl -n $(NFS_NS) exec $$NFS -- sh -c 'mkdir -p /exports/results/$(RUN) && cp /exports/checkpoints/*.json /exports/results/$(RUN)/ && ls -la /exports/results/$(RUN)'; \
	mkdir -p $(RESULTS_DIR)/$(RUN); \
	kubectl -n $(NFS_NS) exec $$NFS -- tar cf - -C /exports/results/$(RUN) . | tar xf - -C $(RESULTS_DIR)/$(RUN)
	@kubectl get distributedtraining $(CR) -o yaml > $(RESULTS_DIR)/$(RUN)/cr.yaml 2>/dev/null \
		&& echo "wrote cr.yaml" || echo "CR $(CR) not found — skipped cr.yaml"
	@kubectl get distributedtraininghistories -A -o name 2>/dev/null | grep "djh-$(CR)-" | \
		xargs -r kubectl get -o yaml > $(RESULTS_DIR)/$(RUN)/history.yaml; \
		test -s $(RESULTS_DIR)/$(RUN)/history.yaml && echo "wrote history.yaml" || { rm -f $(RESULTS_DIR)/$(RUN)/history.yaml; echo "no history entry for $(CR) (HistorySkipped?)"; }
	@ls -la $(RESULTS_DIR)/$(RUN)

##@ GKE shorthands (CLOUD=gke)

.PHONY: gke-init gke-build-deploy gke-apply-config gke-reset-cr gke-rebuild gke-clean gke-watch gke-logs gke-archive-results
gke-init: ## = cloud-init CLOUD=gke
	$(MAKE) cloud-init CLOUD=gke
gke-build-deploy: ## = cloud-build-deploy CLOUD=gke
	$(MAKE) cloud-build-deploy CLOUD=gke IMG=$(IMG)
gke-apply-config: ## = cloud-apply-config CLOUD=gke
	$(MAKE) cloud-apply-config CLOUD=gke
gke-reset-cr: ## = cloud-reset-cr CLOUD=gke
	$(MAKE) cloud-reset-cr CLOUD=gke
gke-rebuild: ## = cloud-rebuild CLOUD=gke
	$(MAKE) cloud-rebuild CLOUD=gke IMG=$(IMG)
gke-clean: ## = cloud-clean CLOUD=gke
	$(MAKE) cloud-clean CLOUD=gke
gke-watch: ## = cloud-watch
	$(MAKE) cloud-watch
gke-logs: ## = cloud-logs
	$(MAKE) cloud-logs
gke-archive-results: ## = archive-results CR=<name>
	$(MAKE) archive-results CR=$(CR) RUN=$(RUN)

##@ Hyperstack shorthands (CLOUD=hyperstack)

HYPERSTACK_PATCH ?= config/overlays/hyperstack/manager_args_patch.yaml

.PHONY: hs-configure
hs-configure: ## Write HYPERSTACK_CLUSTER_ID (required), HYPERSTACK_NODE_LABEL and HYPERSTACK_GPU_FLAVOR (optional) into the Hyperstack overlay.
	@test -n "$(HYPERSTACK_CLUSTER_ID)" || { echo "usage: make hs-configure HYPERSTACK_CLUSTER_ID=<numeric id> [HYPERSTACK_NODE_LABEL=<node label>] [HYPERSTACK_GPU_FLAVOR=<flavor>]"; exit 1; }
	sed -i 's#--hyperstack-cluster-id=.*#--hyperstack-cluster-id=$(HYPERSTACK_CLUSTER_ID)#' $(HYPERSTACK_PATCH)
	@if [ -n "$(HYPERSTACK_NODE_LABEL)" ]; then sed -i 's#--hyperstack-nodepool-label=.*#--hyperstack-nodepool-label=$(HYPERSTACK_NODE_LABEL)#' $(HYPERSTACK_PATCH); fi
	@if [ -n "$(HYPERSTACK_GPU_FLAVOR)" ]; then sed -i 's#--default-gpu-machine-type=.*#--default-gpu-machine-type=$(HYPERSTACK_GPU_FLAVOR)#' $(HYPERSTACK_PATCH); fi
	@grep -n 'hyperstack-cluster-id\|hyperstack-nodepool-label\|default-gpu-machine-type' $(HYPERSTACK_PATCH)

.PHONY: hs-secret
hs-secret: ## Create/update the hyperstack-api Secret in the operator namespace from the HYPERSTACK_API_KEY env var.
	@test -n "$$HYPERSTACK_API_KEY" || { echo "HYPERSTACK_API_KEY is not exported"; exit 1; }
	kubectl create namespace $(OPERATOR_NS) --dry-run=client -o yaml | kubectl apply -f -
	@kubectl create secret generic hyperstack-api --from-literal=api-key="$$HYPERSTACK_API_KEY" \
		-n $(OPERATOR_NS) --dry-run=client -o yaml | kubectl apply -f -

.PHONY: hs-init hs-build-deploy hs-apply-config hs-reset-cr hs-rebuild hs-clean hs-watch hs-logs hs-archive-results
hs-init: ## = cloud-init CLOUD=hyperstack (namespace + API-key Secret)
	$(MAKE) cloud-init CLOUD=hyperstack
hs-build-deploy: ## = cloud-build-deploy CLOUD=hyperstack
	$(MAKE) cloud-build-deploy CLOUD=hyperstack IMG=$(IMG)
hs-apply-config: ## = cloud-apply-config CLOUD=hyperstack
	$(MAKE) cloud-apply-config CLOUD=hyperstack
hs-reset-cr: ## = cloud-reset-cr CLOUD=hyperstack (applies the paid GPU sample)
	$(MAKE) cloud-reset-cr CLOUD=hyperstack
hs-rebuild: ## = cloud-rebuild CLOUD=hyperstack
	$(MAKE) cloud-rebuild CLOUD=hyperstack IMG=$(IMG)
hs-clean: ## = cloud-clean CLOUD=hyperstack
	$(MAKE) cloud-clean CLOUD=hyperstack
hs-watch: ## = cloud-watch
	$(MAKE) cloud-watch
hs-logs: ## = cloud-logs
	$(MAKE) cloud-logs
hs-archive-results: ## = archive-results CR=<name>
	$(MAKE) archive-results CR=$(CR) RUN=$(RUN)

##@ Monitoring (Prometheus + Grafana + Pushgateway)

# kube-prometheus-stack + Pushgateway via Helm, the operator dashboard and the
# operator ServiceMonitor via kubectl. Cloud-specific values live in
# config/monitoring/values-$(CLOUD).yaml (Hyperstack provided; GKE keeps its
# Terraform-managed stack in example/infra/gcp/grafana.tf).
MONITORING_NS      ?= distributed-training-system
MONITORING_VALUES  ?= config/monitoring/values-$(CLOUD).yaml
MONITORING_SC      ?= config/monitoring/storageclass-$(CLOUD).yaml
GRAFANA_PASSWORD   ?= admin

.PHONY: monitoring-install
monitoring-install: ## Install/upgrade Prometheus, Grafana, Pushgateway, the operator dashboard and ServiceMonitor (CLOUD=hyperstack).
	@test -f $(MONITORING_VALUES) || { echo "no $(MONITORING_VALUES) for CLOUD=$(CLOUD); on GKE the stack is Terraform-managed (example/infra/gcp/grafana.tf)"; exit 1; }
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
	helm repo update prometheus-community >/dev/null
	@if [ -f $(MONITORING_SC) ]; then kubectl apply -f $(MONITORING_SC); fi
	helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
		--namespace monitoring --create-namespace --timeout 10m --wait \
		-f $(MONITORING_VALUES) --set grafana.adminPassword=$(GRAFANA_PASSWORD)
	helm upgrade --install pushgateway prometheus-community/prometheus-pushgateway \
		--namespace monitoring --wait -f config/monitoring/values-pushgateway.yaml
	kubectl apply -k config/grafana
	kubectl apply -f config/monitoring/servicemonitor.yaml
	@echo "Grafana: make monitoring-grafana   (user admin / $(GRAFANA_PASSWORD))"

.PHONY: monitoring-grafana
monitoring-grafana: ## Port-forward Grafana to http://localhost:3000.
	kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80

.PHONY: monitoring-uninstall
monitoring-uninstall: ## Remove the monitoring stack AND the Prometheus PVC (detaches the Retain volume — run before destroying the cluster).
	-kubectl delete -f config/monitoring/servicemonitor.yaml --ignore-not-found
	-kubectl delete -k config/grafana --ignore-not-found
	-helm uninstall pushgateway --namespace monitoring
	-helm uninstall monitoring --namespace monitoring
	kubectl -n monitoring delete pvc --all --ignore-not-found
	kubectl -n monitoring wait --for=delete pvc --all --timeout=180s || true
	@echo "Retain volume is detached; delete it on the Hyperstack Volumes page when the data is no longer needed."

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: manifests kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	@out="$$( "$(KUSTOMIZE)" build config/crd 2>/dev/null || true )"; \
	if [ -n "$$out" ]; then echo "$$out" | "$(KUBECTL)" apply -f -; else echo "No CRDs to install; skipping."; fi

.PHONY: uninstall
uninstall: manifests kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	@out="$$( "$(KUSTOMIZE)" build config/crd 2>/dev/null || true )"; \
	if [ -n "$$out" ]; then echo "$$out" | "$(KUBECTL)" delete --ignore-not-found=$(ignore-not-found) -f -; else echo "No CRDs to delete; skipping."; fi

.PHONY: deploy
deploy: manifests kustomize ## Deploy controller to the K8s cluster specified in ~/.kube/config.
	cd config/manager && "$(KUSTOMIZE)" edit set image controller=${IMG}
	"$(KUSTOMIZE)" build config/default | "$(KUBECTL)" apply -f -

.PHONY: undeploy
undeploy: kustomize ## Undeploy controller from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	"$(KUSTOMIZE)" build config/default | "$(KUBECTL)" delete --ignore-not-found=$(ignore-not-found) -f -

##@ Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p "$(LOCALBIN)"

## Tool Binaries
KUBECTL ?= kubectl
KIND ?= kind
KUSTOMIZE ?= $(LOCALBIN)/kustomize
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
ENVTEST ?= $(LOCALBIN)/setup-envtest
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint

## Tool Versions
KUSTOMIZE_VERSION ?= v5.8.1
CONTROLLER_TOOLS_VERSION ?= v0.20.1

#ENVTEST_VERSION is the version of controller-runtime release branch to fetch the envtest setup script (i.e. release-0.20)
ENVTEST_VERSION ?= $(shell v='$(call gomodver,sigs.k8s.io/controller-runtime)'; \
  [ -n "$$v" ] || { echo "Set ENVTEST_VERSION manually (controller-runtime replace has no tag)" >&2; exit 1; }; \
  printf '%s\n' "$$v" | sed -E 's/^v?([0-9]+)\.([0-9]+).*/release-\1.\2/')

#ENVTEST_K8S_VERSION is the version of Kubernetes to use for setting up ENVTEST binaries (i.e. 1.31)
ENVTEST_K8S_VERSION ?= $(shell v='$(call gomodver,k8s.io/api)'; \
  [ -n "$$v" ] || { echo "Set ENVTEST_K8S_VERSION manually (k8s.io/api replace has no tag)" >&2; exit 1; }; \
  printf '%s\n' "$$v" | sed -E 's/^v?[0-9]+\.([0-9]+).*/1.\1/')

GOLANGCI_LINT_VERSION ?= v2.8.0
.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	$(call go-install-tool,$(KUSTOMIZE),sigs.k8s.io/kustomize/kustomize/v5,$(KUSTOMIZE_VERSION))

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

.PHONY: setup-envtest
setup-envtest: envtest ## Download the binaries required for ENVTEST in the local bin directory.
	@echo "Setting up envtest binaries for Kubernetes version $(ENVTEST_K8S_VERSION)..."
	@"$(ENVTEST)" use $(ENVTEST_K8S_VERSION) --bin-dir "$(LOCALBIN)" -p path || { \
		echo "Error: Failed to set up envtest binaries for version $(ENVTEST_K8S_VERSION)."; \
		exit 1; \
	}

.PHONY: envtest
envtest: $(ENVTEST) ## Download setup-envtest locally if necessary.
$(ENVTEST): $(LOCALBIN)
	$(call go-install-tool,$(ENVTEST),sigs.k8s.io/controller-runtime/tools/setup-envtest,$(ENVTEST_VERSION))

.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT) ## Download golangci-lint locally if necessary.
$(GOLANGCI_LINT): $(LOCALBIN)
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/v2/cmd/golangci-lint,$(GOLANGCI_LINT_VERSION))
	@test -f .custom-gcl.yml && { \
		echo "Building custom golangci-lint with plugins..." && \
		$(GOLANGCI_LINT) custom --destination $(LOCALBIN) --name golangci-lint-custom && \
		mv -f $(LOCALBIN)/golangci-lint-custom $(GOLANGCI_LINT); \
	} || true

# go-install-tool will 'go install' any package with custom target and name of binary, if it doesn't exist
# $1 - target path with name of binary
# $2 - package url which can be installed
# $3 - specific version of package
define go-install-tool
@[ -f "$(1)-$(3)" ] && [ "$$(readlink -- "$(1)" 2>/dev/null)" = "$(1)-$(3)" ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
rm -f "$(1)" ;\
GOBIN="$(LOCALBIN)" go install $${package} ;\
mv "$(LOCALBIN)/$$(basename "$(1)")" "$(1)-$(3)" ;\
} ;\
ln -sf "$$(realpath "$(1)-$(3)")" "$(1)"
endef

define gomodver
$(shell go list -m -f '{{if .Replace}}{{.Replace.Version}}{{else}}{{.Version}}{{end}}' $(1) 2>/dev/null)
endef
