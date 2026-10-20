package controller

import (
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var (
	phaseTransitionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "distributedtraining_phase_transitions_total",
			Help: "Total phase transitions per job.",
		},
		[]string{"namespace", "dj_name", "phase"},
	)

	trainingSecondsGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_training_seconds",
			Help: "Actual training duration in seconds.",
		},
		[]string{"namespace", "dj_name"},
	)

	provisioningSecondsGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_provisioning_seconds",
			Help: "Actual node pool provisioning duration in seconds.",
		},
		[]string{"namespace", "dj_name"},
	)

	collectionSecondsGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_collection_seconds",
			Help: "Wall-clock duration of the operator-side result-collection cycle (reader pod mount + read + teardown). Worker nodes are still billed during this window.",
		},
		[]string{"namespace", "dj_name"},
	)

	costUSDActualGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_cost_usd_actual",
			Help: "Actual estimated cost in USD for the completed job.",
		},
		[]string{"namespace", "dj_name"},
	)

	nodesProvisionedGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_nodes_provisioned",
			Help: "Number of nodes provisioned for the job.",
		},
		[]string{"namespace", "dj_name"},
	)

	solverEstimatedTimeSecondsGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_solver_estimated_time_seconds",
			Help: "Solver-predicted training duration in seconds.",
		},
		[]string{"namespace", "dj_name"},
	)

	solverEstimatedCostUSDGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_solver_estimated_cost_usd",
			Help: "Solver-predicted cost in USD.",
		},
		[]string{"namespace", "dj_name"},
	)

	solverNodesSelectedGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_solver_nodes_selected",
			Help: "Node count selected by the topology solver.",
		},
		[]string{"namespace", "dj_name"},
	)

	solverAlphaGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_solver_alpha",
			Help: "Scaling overhead parameter α estimated from historical runs.",
		},
		[]string{"namespace", "dj_name"},
	)

	trainingMetricGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_training_metric",
			Help: "Training metrics collected from job logs (loss, samplesPerSecond, etc.).",
		},
		[]string{"namespace", "dj_name", "metric"},
	)

	objectiveTargetTimeSecondsGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_objective_target_time_seconds",
			Help: "Objective targetTime constraint in seconds (0 if not set).",
		},
		[]string{"namespace", "dj_name"},
	)

	objectiveMaxCostUSDGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_objective_max_cost_usd",
			Help: "Objective maxCost constraint in USD (0 if not set).",
		},
		[]string{"namespace", "dj_name"},
	)

	// --- Data-transfer gauges (set only for instrumented runs) ---

	allreduceFractionGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_allreduce_fraction",
			Help: "Fraction of training time spent in gradient AllReduce (upper bound).",
		},
		[]string{"namespace", "dj_name"},
	)

	dataloadFractionGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_dataload_fraction",
			Help: "Fraction of training time spent waiting for data loading.",
		},
		[]string{"namespace", "dj_name"},
	)

	computeFractionGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_compute_fraction",
			Help: "Fraction of training time spent in compute.",
		},
		[]string{"namespace", "dj_name"},
	)

	allreduceBandwidthGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_allreduce_bandwidth_mbs",
			Help: "Measured AllReduce bandwidth in MB/s.",
		},
		[]string{"namespace", "dj_name"},
	)

	nfsReadBandwidthGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "distributedtraining_nfs_read_bandwidth_mbs",
			Help: "Measured NFS read bandwidth in MB/s.",
		},
		[]string{"namespace", "dj_name"},
	)
)

func init() {
	metrics.Registry.MustRegister(
		phaseTransitionsTotal,
		trainingSecondsGauge,
		provisioningSecondsGauge,
		collectionSecondsGauge,
		costUSDActualGauge,
		nodesProvisionedGauge,
		solverEstimatedTimeSecondsGauge,
		solverEstimatedCostUSDGauge,
		solverNodesSelectedGauge,
		solverAlphaGauge,
		trainingMetricGauge,
		objectiveTargetTimeSecondsGauge,
		objectiveMaxCostUSDGauge,
		allreduceFractionGauge,
		dataloadFractionGauge,
		computeFractionGauge,
		allreduceBandwidthGauge,
		nfsReadBandwidthGauge,
	)
}
