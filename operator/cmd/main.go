/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"crypto/tls"
	"flag"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
	"github.com/JonMukaj/distributed-training-operator/internal/backend"
	"github.com/JonMukaj/distributed-training-operator/internal/backend/pytorch"
	"github.com/JonMukaj/distributed-training-operator/internal/backend/spark"
	"github.com/JonMukaj/distributed-training-operator/internal/cloud"
	"github.com/JonMukaj/distributed-training-operator/internal/cloud/gke"
	"github.com/JonMukaj/distributed-training-operator/internal/controller"
	// +kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(trainingv1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

// nolint:gocyclo
func main() {
	var (
		metricsAddr                                      string
		metricsCertPath, metricsCertName, metricsCertKey string
		webhookCertPath, webhookCertName, webhookCertKey string
		enableLeaderElection                             bool
		probeAddr                                        string
		secureMetrics                                    bool
		enableHTTP2                                      bool

		cloudProvider         string
		defaultCPUMachineType string
		defaultGPUMachineType string
		defaultDiskSizeGb     int

		defaultCalibrationNodes        int
		machineCostsConfigMapName      string
		machineCostsConfigMapNamespace string
		nodeServiceAccount             string

		// GKE-specific flags — only consumed when --cloud-provider=gke.
		gcpProject  string
		gcpLocation string
		clusterName string
	)

	flag.StringVar(&metricsAddr, "metrics-bind-address", "0",
		"The address the metrics endpoint binds to. Use :8443 for HTTPS or :8080 for HTTP, or leave as 0 to disable.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081",
		"The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager.")
	flag.BoolVar(&secureMetrics, "metrics-secure", true,
		"If set, the metrics endpoint is served securely via HTTPS.")
	flag.StringVar(&webhookCertPath, "webhook-cert-path", "", "The directory that contains the webhook certificate.")
	flag.StringVar(&webhookCertName, "webhook-cert-name", "tls.crt", "The name of the webhook certificate file.")
	flag.StringVar(&webhookCertKey, "webhook-cert-key", "tls.key", "The name of the webhook key file.")
	flag.StringVar(&metricsCertPath, "metrics-cert-path", "", "The directory that contains the metrics server certificate.")
	flag.StringVar(&metricsCertName, "metrics-cert-name", "tls.crt", "The name of the metrics server certificate file.")
	flag.StringVar(&metricsCertKey, "metrics-cert-key", "tls.key", "The name of the metrics server key file.")
	flag.BoolVar(&enableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers.")

	flag.StringVar(&cloudProvider, "cloud-provider", "gke",
		"Cloud provider for node pool management. Supported: gke")
	flag.StringVar(&defaultCPUMachineType, "default-cpu-machine-type", "e2-standard-4",
		"Default GCE machine type for CPU jobs when spec.hardware.machineType is unset.")
	flag.StringVar(&defaultGPUMachineType, "default-gpu-machine-type", "n1-standard-4",
		"Default GCE machine type for GPU jobs when spec.hardware.machineType is unset.")
	flag.IntVar(&defaultDiskSizeGb, "default-disk-size-gb", 50,
		"Default boot disk size in GB for ephemeral node pool nodes.")
	flag.IntVar(&defaultCalibrationNodes, "default-calibration-nodes", 2,
		"Node count used for the first calibration run when no history exists for a job configuration.")
	flag.StringVar(&machineCostsConfigMapName, "machine-costs-configmap", "machine-costs",
		"Name of the ConfigMap containing per-machine-type hourly costs (used by the topology solver and cost tracking).")
	flag.StringVar(&machineCostsConfigMapNamespace, "machine-costs-namespace", "distributed-training-system",
		"Namespace of the machine costs ConfigMap.")
	flag.StringVar(&nodeServiceAccount, "node-service-account", "",
		"IAM service account email to attach to operator-created node pools. Leave empty to use the cloud provider default.")

	// GKE flags.
	flag.StringVar(&gcpProject, "gcp-project", "",
		"[gke] GCP project ID. Falls back to GCP_PROJECT env var.")
	flag.StringVar(&gcpLocation, "gcp-location", "us-east1",
		"[gke] GCP region or zone of the GKE cluster.")
	flag.StringVar(&clusterName, "cluster-name", "",
		"[gke] Name of the GKE cluster.")

	opts := zap.Options{Development: false}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	// Disable HTTP/2 by default to avoid CVE-2023-44487 and CVE-2023-39325.
	var tlsOpts []func(*tls.Config)
	disableHTTP2 := func(c *tls.Config) {
		setupLog.Info("Disabling HTTP/2")
		c.NextProtos = []string{"http/1.1"}
	}
	if !enableHTTP2 {
		tlsOpts = append(tlsOpts, disableHTTP2)
	}

	webhookServerOptions := webhook.Options{TLSOpts: tlsOpts}
	if len(webhookCertPath) > 0 {
		setupLog.Info("Initializing webhook certificate watcher",
			"webhook-cert-path", webhookCertPath,
			"webhook-cert-name", webhookCertName,
			"webhook-cert-key", webhookCertKey)
		webhookServerOptions.CertDir = webhookCertPath
		webhookServerOptions.CertName = webhookCertName
		webhookServerOptions.KeyName = webhookCertKey
	}
	webhookServer := webhook.NewServer(webhookServerOptions)

	metricsServerOptions := metricsserver.Options{
		BindAddress:   metricsAddr,
		SecureServing: secureMetrics,
		TLSOpts:       tlsOpts,
	}
	if secureMetrics {
		metricsServerOptions.FilterProvider = filters.WithAuthenticationAndAuthorization
	}
	if len(metricsCertPath) > 0 {
		setupLog.Info("Initializing metrics certificate watcher",
			"metrics-cert-path", metricsCertPath,
			"metrics-cert-name", metricsCertName,
			"metrics-cert-key", metricsCertKey)
		metricsServerOptions.CertDir = metricsCertPath
		metricsServerOptions.CertName = metricsCertName
		metricsServerOptions.KeyName = metricsCertKey
	}

	// Build the cloud provider — the only place in the binary with provider-specific logic.
	provider, err := buildProvider(cloudProvider, gcpProject, gcpLocation, clusterName)
	if err != nil {
		setupLog.Error(err, "unable to initialise cloud provider", "provider", cloudProvider)
		os.Exit(1)
	}
	setupLog.Info("cloud provider ready", "provider", provider.Name())

	// Register all job backends. Adding a new backend means implementing
	// backend.JobBackend and adding an entry here.
	backends := map[trainingv1.BackendType]backend.JobBackend{
		trainingv1.BackendPyTorch: pytorch.New(),
		trainingv1.BackendSpark:   spark.New(),
	}
	setupLog.Info("job backends registered", "backends", []string{"pytorch", "spark"})

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsServerOptions,
		WebhookServer:          webhookServer,
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "338fdebe.distributedtraining.io",
	})
	if err != nil {
		setupLog.Error(err, "Failed to start manager")
		os.Exit(1)
	}

	reconciler := controller.NewDistributedTrainingReconciler(
		mgr.GetClient(),
		mgr.GetScheme(),
		mgr.GetEventRecorderFor("distributedtraining-controller"),
		provider,
		backends,
		defaultCPUMachineType,
		defaultGPUMachineType,
		int32(defaultDiskSizeGb),
		int32(defaultCalibrationNodes),
		machineCostsConfigMapName,
		machineCostsConfigMapNamespace,
		nodeServiceAccount,
	)

	if err = reconciler.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to set up DistributedTraining controller")
		os.Exit(1)
	}
	// +kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "Failed to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "Failed to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager",
		"provider", provider.Name(),
		"defaultCPUMachineType", defaultCPUMachineType,
		"defaultGPUMachineType", defaultGPUMachineType,
	)
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "Failed to run manager")
		os.Exit(1)
	}
}

// buildProvider is the cloud provider factory. To add a new cloud provider:
//  1. Create internal/cloud/<name>/<name>.go implementing cloud.Provider
//  2. Add a case below
//  3. Add the provider-specific flags above
func buildProvider(name, gcpProject, gcpLocation, clusterName string) (cloud.Provider, error) {
	switch name {
	case "gke":
		if gcpProject == "" {
			gcpProject = os.Getenv("GCP_PROJECT")
		}
		if gcpProject == "" {
			return nil, fmt.Errorf("--gcp-project or GCP_PROJECT env var is required for the gke provider")
		}
		return gke.New(gcpProject, gcpLocation, clusterName)

	// case "eks":
	//     return eks.New(awsRegion, clusterName)

	// case "aks":
	//     return aks.New(subscriptionID, resourceGroup, clusterName)

	default:
		return nil, fmt.Errorf("unsupported cloud provider %q — supported: gke", name)
	}
}
