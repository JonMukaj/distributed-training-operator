package controller

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

// ---------------------------------------------------------------------------
// Solver entry point
// ---------------------------------------------------------------------------

// runSolver runs the topology solver at the end of the Pending phase.
// It loads history, estimates the scaling model, and selects an optimal node count
// that satisfies the declared objective constraints.
// Every code path writes to status.ResolvedTopology before returning.
func (r *DistributedTrainingReconciler) runSolver(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
) (trainingv1.ResolvedTopology, error) {
	logger := log.FromContext(ctx)
	obj := job.Spec.Objective
	maxNodes := obj.MaxNodes

	configHash := computeConfigHash(job)
	history, err := r.loadHistory(ctx, job.Namespace, configHash)
	if err != nil {
		return trainingv1.ResolvedTopology{}, fmt.Errorf("loading history: %w", err)
	}

	// ── Step 1: W availability ────────────────────────────────────────────────
	if len(history) == 0 {
		r.Recorder.Event(job, corev1.EventTypeNormal, "Calibration",
			fmt.Sprintf("No history found — running calibration topology (n=%d). Objective will be applied from next run.",
				r.DefaultCalibrationNodes))
		logger.Info("no history — calibration run", "nodes", r.DefaultCalibrationNodes)
		return trainingv1.ResolvedTopology{
			Nodes:          r.DefaultCalibrationNodes,
			MasterReplicas: 1,
			WorkerReplicas: r.DefaultCalibrationNodes - 1,
			EstimatedTime:  "unknown (calibration run)",
		}, nil
	}

	// Record objective constraints as metrics so the dashboard can show them.
	if obj.TargetTime != "" {
		if secs, err := parseDurationSeconds(obj.TargetTime); err == nil {
			objectiveTargetTimeSecondsGauge.WithLabelValues(job.Namespace, job.Name).Set(secs)
		}
	}
	if obj.MaxCost != "" {
		if cost, err := strconv.ParseFloat(obj.MaxCost, 64); err == nil {
			objectiveMaxCostUSDGauge.WithLabelValues(job.Namespace, job.Name).Set(cost)
		}
	}

	// Compute model parameters from history.
	mp := estimateModel(history)
	logger.Info("scaling model estimated", "alpha", mp.alpha, "pBaseline", mp.pBaseline, "W", mp.w, "tProvAvg", mp.tProvAvg)
	solverAlphaGauge.WithLabelValues(job.Namespace, job.Name).Set(mp.alpha)

	// ── Step 2: C_h availability ──────────────────────────────────────────────
	machineType := r.effectiveMachineType(job)
	ch, chKnown, err := r.getMachineCost(ctx, machineType)
	if err != nil {
		logger.Error(err, "error reading machine cost ConfigMap — treating as not found")
		chKnown = false
	}

	if !chKnown {
		if obj.TargetTime == "" {
			// Only maxCost given — cannot evaluate without C_h.
			r.Recorder.Event(job, corev1.EventTypeWarning, "SolverFallback",
				fmt.Sprintf("machine type %q not in cost ConfigMap, cannot evaluate maxCost — falling back to calibration", machineType))
			return trainingv1.ResolvedTopology{
				Nodes:          r.DefaultCalibrationNodes,
				MasterReplicas: 1,
				WorkerReplicas: r.DefaultCalibrationNodes - 1,
				EstimatedTime:  "unknown (calibration run)",
			}, nil
		}
		// targetTime is set — skip cost steps, go directly to time-only path.
		r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
			fmt.Sprintf("machine type %q not in cost ConfigMap — cost constraint skipped, using time-only path", machineType))
		return r.solveTimeOnly(job, mp, maxNodes)
	}

	// ── Step 3: Detect flat cost curve ───────────────────────────────────────
	flat := isFlatCost(mp, ch, maxNodes)

	// ── Dispatch to constraint branches ──────────────────────────────────────
	hasTime := obj.TargetTime != ""
	hasCost := obj.MaxCost != ""

	switch {
	case hasTime && !hasCost:
		return r.solveTimeOnlyWithCost(job, mp, ch, maxNodes, chKnown)
	case !hasTime && hasCost:
		return r.solveCostOnly(job, mp, ch, flat, maxNodes)
	default: // both
		return r.solveBoth(job, mp, ch, flat, maxNodes)
	}
}

// ---------------------------------------------------------------------------
// Solver branches
// ---------------------------------------------------------------------------

func (r *DistributedTrainingReconciler) solveTimeOnly(
	job *trainingv1.DistributedTraining,
	mp modelParams,
	maxNodes int32,
) (trainingv1.ResolvedTopology, error) {
	tTarget, err := parseDurationSeconds(job.Spec.Objective.TargetTime)
	if err != nil {
		return trainingv1.ResolvedTopology{}, fmt.Errorf("parsing targetTime: %w", err)
	}

	nTime := binarySearchMinN(2, maxNodes, tTarget, mp)
	if predT(nTime, mp) > tTarget {
		r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
			fmt.Sprintf("targetTime %s unreachable at maxNodes=%d — using maxNodes", job.Spec.Objective.TargetTime, maxNodes))
	}
	return r.buildTopology(job, nTime, mp, 0, false), nil
}

func (r *DistributedTrainingReconciler) solveTimeOnlyWithCost(
	job *trainingv1.DistributedTraining,
	mp modelParams,
	ch float64,
	maxNodes int32,
	chKnown bool,
) (trainingv1.ResolvedTopology, error) {
	tTarget, err := parseDurationSeconds(job.Spec.Objective.TargetTime)
	if err != nil {
		return trainingv1.ResolvedTopology{}, fmt.Errorf("parsing targetTime: %w", err)
	}

	nTime := binarySearchMinN(2, maxNodes, tTarget, mp)
	if predT(nTime, mp) > tTarget {
		r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
			fmt.Sprintf("targetTime %s unreachable at maxNodes=%d — using maxNodes", job.Spec.Objective.TargetTime, maxNodes))
	}
	return r.buildTopology(job, nTime, mp, ch, chKnown), nil
}

func (r *DistributedTrainingReconciler) solveCostOnly(
	job *trainingv1.DistributedTraining,
	mp modelParams,
	ch float64,
	flat bool,
	maxNodes int32,
) (trainingv1.ResolvedTopology, error) {
	bMax, err := strconv.ParseFloat(job.Spec.Objective.MaxCost, 64)
	if err != nil {
		return trainingv1.ResolvedTopology{}, fmt.Errorf("parsing maxCost: %w", err)
	}

	var nBudget int32
	if flat {
		if predCost(2, mp, ch) <= bMax {
			nBudget = maxNodes // all equally cheap — pick fastest
		} else {
			r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
				"budget too small for any topology — using n=2")
			nBudget = 2
		}
	} else {
		if predCost(2, mp, ch) > bMax {
			r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
				"maxCost too low for minimum topology (n=2) — using n=2")
			nBudget = 2
		} else {
			nBudget = binarySearchMaxN(2, maxNodes, bMax, mp, ch)
		}
	}
	return r.buildTopology(job, nBudget, mp, ch, true), nil
}

func (r *DistributedTrainingReconciler) solveBoth(
	job *trainingv1.DistributedTraining,
	mp modelParams,
	ch float64,
	flat bool,
	maxNodes int32,
) (trainingv1.ResolvedTopology, error) {
	tTarget, err := parseDurationSeconds(job.Spec.Objective.TargetTime)
	if err != nil {
		return trainingv1.ResolvedTopology{}, fmt.Errorf("parsing targetTime: %w", err)
	}
	bMax, err := strconv.ParseFloat(job.Spec.Objective.MaxCost, 64)
	if err != nil {
		return trainingv1.ResolvedTopology{}, fmt.Errorf("parsing maxCost: %w", err)
	}

	// Step 6: find n_time (intermediate — do not write status yet)
	nTime := binarySearchMinN(2, maxNodes, tTarget, mp)
	if predT(nTime, mp) > tTarget {
		r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
			fmt.Sprintf("targetTime %s unreachable at maxNodes=%d", job.Spec.Objective.TargetTime, maxNodes))
	}

	// Step 7: check Cost(n_time)
	costAtNTime := predCost(nTime, mp, ch)
	if costAtNTime <= bMax {
		// Both satisfied.
		return r.buildTopology(job, nTime, mp, ch, true), nil
	}

	// Conflict: n_time is the cheapest option for the time constraint, but still over budget.
	if flat && predCost(2, mp, ch) > bMax {
		r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
			"budget too low for any topology — using n=2")
		return r.buildTopology(job, 2, mp, ch, true), nil
	}

	if predCost(2, mp, ch) > bMax {
		r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
			"budget too low for minimum topology — using n=2")
		return r.buildTopology(job, 2, mp, ch, true), nil
	}

	nBudget := binarySearchMaxN(2, maxNodes, bMax, mp, ch)
	r.Recorder.Event(job, corev1.EventTypeWarning, "SolverWarning",
		fmt.Sprintf("targetTime unreachable within maxCost — using %d nodes (estimated time %s). Increase maxCost or relax targetTime.",
			nBudget, formatSeconds(predT(nBudget, mp))))
	return r.buildTopology(job, nBudget, mp, ch, true), nil
}

// ---------------------------------------------------------------------------
// Model estimation
// ---------------------------------------------------------------------------

type modelParams struct {
	alpha     float64 // scaling overhead
	pBaseline float64 // base throughput
	w         float64 // average Work at n_min
	tProvAvg  float64 // average provisioning seconds
}

// estimateModel computes alpha, p_baseline, W, and T_provision_avg from history entries.
func estimateModel(history []trainingv1.DistributedTrainingHistorySpec) modelParams {
	if len(history) == 0 {
		return modelParams{}
	}

	// Group entries by node count.
	byN := groupByN(history)
	ns := sortedKeys(byN)
	nMin := ns[0]

	// p_baseline = average P_k at n_min / n_min
	avgPAtNMin := average(mapValues(byN[nMin], func(e trainingv1.DistributedTrainingHistorySpec) float64 {
		return e.Throughput.AsApproximateFloat64()
	}))
	pBaseline := avgPAtNMin / float64(nMin)

	// W = average(T_k × P_k) at n_min
	wAtNMin := average(mapValues(byN[nMin], func(e trainingv1.DistributedTrainingHistorySpec) float64 {
		return e.TotalWork.AsApproximateFloat64()
	}))

	// T_provision_avg across all entries
	tProvAvg := average(mapValues(history, func(e trainingv1.DistributedTrainingHistorySpec) float64 {
		return e.ProvisioningSeconds.AsApproximateFloat64()
	}))

	// Estimate alpha
	alpha := estimateAlpha(byN, ns, pBaseline)

	return modelParams{
		alpha:     alpha,
		pBaseline: pBaseline,
		w:         wAtNMin,
		tProvAvg:  tProvAvg,
	}
}

// estimateAlpha computes the scaling overhead parameter.
// With 1 distinct n: α = 0 (linear scaling assumption).
// With 2 distinct n: closed-form formula.
// With ≥3 distinct n: 1D golden section search.
func estimateAlpha(byN map[int32][]trainingv1.DistributedTrainingHistorySpec, ns []int32, pBaseline float64) float64 {
	if len(ns) < 2 {
		return 0
	}

	if len(ns) == 2 {
		n1, n2 := ns[0], ns[1]
		avgP1 := average(mapValues(byN[n1], func(e trainingv1.DistributedTrainingHistorySpec) float64 { return e.Throughput.AsApproximateFloat64() }))
		avgP2 := average(mapValues(byN[n2], func(e trainingv1.DistributedTrainingHistorySpec) float64 { return e.Throughput.AsApproximateFloat64() }))
		eta2Obs := (avgP2 / float64(n2)) / (avgP1 / float64(n1))
		if eta2Obs <= 0 {
			return 0
		}
		alpha := (1 - eta2Obs) / (eta2Obs * float64(n2-1))
		return math.Max(0, alpha)
	}

	// ≥3 distinct n: minimize sum-of-squared residuals via golden section search.
	residualFn := func(alpha float64) float64 {
		total := 0.0
		for _, n := range ns {
			avgP := average(mapValues(byN[n], func(e trainingv1.DistributedTrainingHistorySpec) float64 { return e.Throughput.AsApproximateFloat64() }))
			predicted := pBaseline * float64(n) * eta(float64(n), alpha)
			diff := predicted - avgP
			total += diff * diff
		}
		return total
	}

	return goldenSectionSearch(residualFn, 0, 1.0, 1e-6)
}

// ---------------------------------------------------------------------------
// Scaling model formulas
// ---------------------------------------------------------------------------

func eta(n, alpha float64) float64 {
	return 1.0 / (1.0 + alpha*(n-1))
}

func predP(n int32, mp modelParams) float64 {
	return mp.pBaseline * float64(n) * eta(float64(n), mp.alpha)
}

func predT(n int32, mp modelParams) float64 {
	p := predP(n, mp)
	if p <= 0 {
		return math.MaxFloat64
	}
	return mp.w / p
}

func predCost(n int32, mp modelParams, ch float64) float64 {
	trainingCost := ch * float64(n) * predT(n, mp) / 3600.0
	provisioningCost := ch * float64(n) * mp.tProvAvg / 3600.0
	return trainingCost + provisioningCost
}

func isFlatCost(mp modelParams, ch float64, maxNodes int32) bool {
	c2 := predCost(2, mp, ch)
	if c2 == 0 {
		return true
	}
	cMax := predCost(maxNodes, mp, ch)
	return math.Abs(cMax-c2)/c2 < 0.01
}

// ---------------------------------------------------------------------------
// Binary search helpers
// ---------------------------------------------------------------------------

// binarySearchMinN finds the minimum n in [lo, maxNodes] such that T(n) <= tTarget.
// T is non-increasing in n. Returns maxNodes if T(maxNodes) > tTarget.
func binarySearchMinN(lo, maxNodes int32, tTarget float64, mp modelParams) int32 {
	if predT(maxNodes, mp) > tTarget {
		return maxNodes
	}
	hi := maxNodes
	for lo < hi {
		mid := (lo + hi) / 2
		if predT(mid, mp) <= tTarget {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return lo
}

// binarySearchMaxN finds the maximum n in [lo, maxNodes] such that Cost(n) <= bMax.
// Cost is strictly increasing. Returns lo if Cost(lo) > bMax.
func binarySearchMaxN(lo, maxNodes int32, bMax float64, mp modelParams, ch float64) int32 {
	if predCost(lo, mp, ch) > bMax {
		return lo
	}
	hi := maxNodes
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if predCost(mid, mp, ch) <= bMax {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// ---------------------------------------------------------------------------
// Topology construction
// ---------------------------------------------------------------------------

// buildTopology assembles a ResolvedTopology with master/worker split for pytorch.
func (r *DistributedTrainingReconciler) buildTopology(
	job *trainingv1.DistributedTraining,
	n int32,
	mp modelParams,
	ch float64,
	chKnown bool,
) trainingv1.ResolvedTopology {
	rt := trainingv1.ResolvedTopology{
		Nodes:         n,
		EstimatedTime: formatSeconds(predT(n, mp)),
	}

	backend := job.Spec.Backend
	if backend == "" {
		backend = trainingv1.BackendPyTorch
	}
	if backend == trainingv1.BackendPyTorch {
		rt.MasterReplicas = 1
		rt.WorkerReplicas = n - 1
	}

	if chKnown && ch > 0 {
		rt.EstimatedCost = fmt.Sprintf("%.4f", predCost(n, mp, ch))
	}

	solverNodesSelectedGauge.WithLabelValues(job.Namespace, job.Name).Set(float64(n))
	if t := predT(n, mp); t < math.MaxFloat64/2 {
		solverEstimatedTimeSecondsGauge.WithLabelValues(job.Namespace, job.Name).Set(t)
	}
	if chKnown && ch > 0 {
		solverEstimatedCostUSDGauge.WithLabelValues(job.Namespace, job.Name).Set(predCost(n, mp, ch))
	}

	return rt
}

// ---------------------------------------------------------------------------
// Machine cost lookup
// ---------------------------------------------------------------------------

// getMachineCost reads the hourly cost per node for machineType from the
// operator's cost ConfigMap. Returns (0, false, nil) when the ConfigMap does not
// exist or the machine type is absent — callers must check the bool.
func (r *DistributedTrainingReconciler) getMachineCost(ctx context.Context, machineType string) (float64, bool, error) {
	cm := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{
		Namespace: r.MachineCostsConfigMapNamespace,
		Name:      r.MachineCostsConfigMapName,
	}, cm)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			return 0, false, nil
		}
		return 0, false, err
	}

	val, ok := cm.Data[machineType]
	if !ok {
		return 0, false, nil
	}
	cost, err := strconv.ParseFloat(val, 64)
	if err != nil {
		return 0, false, fmt.Errorf("parsing cost for %q in ConfigMap: %w", machineType, err)
	}
	return cost, true, nil
}

// ---------------------------------------------------------------------------
// History loading
// ---------------------------------------------------------------------------

func (r *DistributedTrainingReconciler) loadHistory(
	ctx context.Context,
	namespace, configHash string,
) ([]trainingv1.DistributedTrainingHistorySpec, error) {
	list := &trainingv1.DistributedTrainingHistoryList{}
	if err := r.List(ctx, list,
		client.InNamespace(namespace),
		client.MatchingLabels{"config-hash": configHash},
	); err != nil {
		return nil, err
	}

	specs := make([]trainingv1.DistributedTrainingHistorySpec, len(list.Items))
	for i, item := range list.Items {
		specs[i] = item.Spec
	}
	return specs, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func parseDurationSeconds(s string) (float64, error) {
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, err
	}
	return d.Seconds(), nil
}

func formatSeconds(s float64) string {
	if s >= math.MaxFloat64/2 {
		return "unknown"
	}
	// Convert via nanoseconds so the fractional second is preserved before
	// rounding — otherwise `time.Duration(s)` truncates 540.9 → 540, making
	// the formatted status string disagree with the Grafana gauge that holds
	// the unrounded float.
	d := time.Duration(s * float64(time.Second))
	return d.Round(time.Second).String()
}

func groupByN(entries []trainingv1.DistributedTrainingHistorySpec) map[int32][]trainingv1.DistributedTrainingHistorySpec {
	m := make(map[int32][]trainingv1.DistributedTrainingHistorySpec)
	for _, e := range entries {
		m[e.Nodes] = append(m[e.Nodes], e)
	}
	return m
}

func sortedKeys(m map[int32][]trainingv1.DistributedTrainingHistorySpec) []int32 {
	keys := make([]int32, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func mapValues(entries []trainingv1.DistributedTrainingHistorySpec, fn func(trainingv1.DistributedTrainingHistorySpec) float64) []float64 {
	out := make([]float64, len(entries))
	for i, e := range entries {
		out[i] = fn(e)
	}
	return out
}

func average(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// goldenSectionSearch minimizes f over [a, b] to within tol.
func goldenSectionSearch(f func(float64) float64, a, b, tol float64) float64 {
	const phi = 0.6180339887498949 // (√5 - 1) / 2
	c := b - phi*(b-a)
	d := a + phi*(b-a)
	for math.Abs(b-a) > tol {
		if f(c) < f(d) {
			b = d
		} else {
			a = c
		}
		c = b - phi*(b-a)
		d = a + phi*(b-a)
	}
	return (a + b) / 2
}
