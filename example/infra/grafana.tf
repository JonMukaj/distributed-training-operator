variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
  default     = "admin"
}

locals {
  dashboard_json = file("${path.module}/operator/config/grafana/distributedtraining-dashboard.json")
}

# Custom storage class with Retain reclaim policy so the underlying GCE
# persistent disk survives PVC deletion and cluster destruction. The disk
# becomes orphaned (Released state) in the GCP project and persists until
# manually deleted. Restoring later requires creating a static PV pointing
# at the existing disk and binding a new PVC to it — see the RESTORATION.MD
resource "kubernetes_storage_class_v1" "prometheus_retain" {
  metadata {
    name = "prometheus-retain"
  }
  storage_provisioner    = "pd.csi.storage.gke.io"
  reclaim_policy         = "Retain"
  volume_binding_mode    = "WaitForFirstConsumer"
  allow_volume_expansion = true
  parameters = {
    type                          = "pd-balanced"
    "csi.storage.k8s.io/fstype"   = "ext4"
  }
}


# kube-prometheus-stack: Prometheus + Grafana + alerting stack.
# Alertmanager disabled to keep resource usage low on the general node pool.
# The sidecar watches for ConfigMaps labelled grafana_dashboard="1" and hot-loads
# dashboards without restarting Grafana; the distributedtraining dashboard is injected
# via grafana.dashboards below.
resource "helm_release" "monitoring" {
  name             = "monitoring"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  timeout          = 600

  depends_on = [kubernetes_storage_class_v1.prometheus_retain]

  # All chart configuration in a single values block — avoids set{} blocks
  # that the LSP flags as unknown until `terraform init` downloads the schema.
  values = [
    yamlencode({
      alertmanager = {
        enabled = false
      }
      grafana = {
        adminPassword = var.grafana_admin_password
        sidecar = {
          dashboards = {
            enabled = true
            label   = "grafana_dashboard"
          }
        }
        imageRenderer = {
          enabled = true
        }
      }
      prometheus = {
        prometheusSpec = {
          serviceMonitorSelectorNilUsesHelmValues = false
          podMonitorSelectorNilUsesHelmValues     = false
          enableAdminAPI                          = true
          # Persistent storage for the TSDB. Uses the `prometheus-retain`
          # StorageClass (reclaimPolicy=Retain) so the underlying GCE disk
          # outlives PVC deletion and cluster destruction. The disk persists
          # in the GCP project until manually deleted via
          # `gcloud compute disks delete`.
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                accessModes      = ["ReadWriteOnce"]
                storageClassName = "prometheus-retain"
                resources = {
                  requests = {
                    storage = "5Gi"
                  }
                }
              }
            }
          }
        }
      }
    })
  ]
}

# Dashboard ConfigMap with the label the Grafana sidecar actually watches for.
# grafana.dashboards in helm values creates ConfigMaps without this label, so
# the sidecar ignores them. Creating it directly here guarantees the right label.
resource "kubectl_manifest" "dashboard_configmap" {
  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "ConfigMap"
    metadata = {
      name      = "distributedtraining-operator-dashboard"
      namespace = "monitoring"
      labels = {
        grafana_dashboard = "1"
      }
    }
    data = {
      "distributedtraining-dashboard.json" = local.dashboard_json
    }
  })

  depends_on = [helm_release.monitoring]
}

# ---------------------------------------------------------------------------
# Prometheus Pushgateway
# Allows baseline (manual) experiment results to be pushed into Prometheus
# so they appear alongside operator-emitted metrics in the Grafana dashboard.
# After each baseline run, run scripts/push_baseline.sh to inject the metrics.
# ---------------------------------------------------------------------------
resource "helm_release" "pushgateway" {
  name             = "pushgateway"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "prometheus-pushgateway"
  namespace        = "monitoring"
  create_namespace = false

  values = [
    yamlencode({
      serviceMonitor = {
        enabled = true
        # Must match kube-prometheus-stack's serviceMonitorSelector label
        # so the Prometheus CR picks it up automatically.
        additionalLabels = {
          release = "monitoring"
        }
      }
    })
  ]

  depends_on = [helm_release.monitoring]
}

# Ensures the operator namespace exists before the Service is created.
# Idempotent: if kustomize already created it this is a no-op update.
resource "kubectl_manifest" "operator_namespace" {
  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "Namespace"
    metadata = {
      name = "distributed-training-system"
    }
  })
}

# ClusterIP Service that exposes the operator's HTTP metrics port (8080).
# The default kubebuilder metrics_service.yaml targets port 8443 (HTTPS); this
# companion Service targets the HTTP endpoint added in the GKE overlay.
resource "kubectl_manifest" "operator_metrics_service" {
  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "Service"
    metadata = {
      name      = "distributed-training-operator-metrics"
      namespace = "distributed-training-system"
      labels = {
        "control-plane" = "controller-manager"
      }
    }
    spec = {
      selector = {
        "control-plane" = "controller-manager"
      }
      ports = [
        {
          name       = "http"
          port       = 8080
          targetPort = 8080
          protocol   = "TCP"
        }
      ]
    }
  })

  depends_on = [kubectl_manifest.operator_namespace]
}

# ServiceMonitor tells Prometheus (deployed by kube-prometheus-stack) where to
# scrape the operator's /metrics endpoint.
resource "kubectl_manifest" "operator_service_monitor" {
  yaml_body = yamlencode({
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "distributed-training-operator"
      namespace = "monitoring"
      labels = {
        # Must match the helm release name so kube-prometheus-stack's Prometheus
        # CR picks it up via its serviceMonitorSelector.
        release = "monitoring"
      }
    }
    spec = {
      endpoints = [
        {
          port     = "http"
          path     = "/metrics"
          interval = "30s"
          # honorLabels=true keeps the operator's own `namespace` label
          # (set from the DistributedTraining CR's namespace, e.g. "default") and
          # prevents Prometheus's auto-injected scrape-target namespace
          # ("distributed-training-system") from overwriting it. Without this,
          # all operator metrics carry namespace="distributed-training-system",
          # forcing baselines pushed via Pushgateway to use the same label
          # value just to land in the same Grafana dropdown bucket.
          honorLabels = true
        }
      ]
      selector = {
        matchLabels = {
          "control-plane" = "controller-manager"
        }
      }
      namespaceSelector = {
        matchNames = ["distributed-training-system"]
      }
    }
  })

  depends_on = [
    kubectl_manifest.operator_metrics_service,
    helm_release.monitoring,
  ]
}
