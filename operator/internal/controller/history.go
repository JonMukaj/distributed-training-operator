package controller

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	corev1 "k8s.io/api/core/v1"

	trainingv1 "github.com/JonMukaj/distributed-training-operator/api/v1"
)

const historyRetentionLimit = 10

// writeHistoryEntry records a completed run in a DistributedTrainingHistory CR.
// Called at the end of reconcileCollecting for every Succeeded job (pytorch and spark;
// never for the "job" backend since the solver does not apply to single-node jobs).
// Skipped with a Warning event when P_k == 0 or T_k cannot be parsed.
func (r *DistributedTrainingReconciler) writeHistoryEntry(
	ctx context.Context,
	job *trainingv1.DistributedTraining,
	results *trainingv1.JobResults,
) {
	logger := log.FromContext(ctx)

	backend := job.Spec.Backend
	if backend == "" {
		backend = trainingv1.BackendPyTorch
	}
	if backend == trainingv1.BackendJob {
		return
	}

	// --- Parse T_k (training time) — needed first because effective
	// throughput is computed as totalSamples / T_k below. ---
	if results.TrainingTime == "" {
		r.Recorder.Event(job, corev1.EventTypeWarning, "HistorySkipped",
			"skipping history write: trainingTime is empty")
		return
	}
	trainingDur, err := time.ParseDuration(results.TrainingTime)
	if err != nil {
		r.Recorder.Event(job, corev1.EventTypeWarning, "HistorySkipped",
			fmt.Sprintf("skipping history write: cannot parse trainingTime %q: %v", results.TrainingTime, err))
		return
	}
	tk := trainingDur.Seconds()

	// --- Determine P_k (throughput) ---
	// Prefer wall-clock-effective throughput (totalSamples / T_k) when the
	// backend reports total samples processed. This makes W = T_k × P_k
	// collapse to total samples — constant per workload, and removes the
	// inner-loop vs wall-clock mismatch from the solver's fit.
	pk, err := extractThroughput(backend, results, tk)
	if err != nil || pk == 0 {
		r.Recorder.Event(job, corev1.EventTypeWarning, "HistorySkipped",
			"skipping history write: throughput (P_k) is missing or zero — W = 0 would corrupt the solver")
		return
	}

	// --- Parse T_provision (provisioning time, best-effort) ---
	var tProvision float64
	if results.ProvisioningTime != "" {
		if d, err2 := time.ParseDuration(results.ProvisioningTime); err2 == nil {
			tProvision = d.Seconds()
		}
	}

	// --- Determine n_k ---
	nk := nodeCountForHistory(job)
	if nk == 0 {
		r.Recorder.Event(job, corev1.EventTypeWarning, "HistorySkipped",
			"skipping history write: could not determine node count (n_k == 0)")
		return
	}

	// --- Compute config hash ---
	configHash := computeConfigHash(job)

	// --- Effective machine type ---
	machineType := r.effectiveMachineType(job)

	// --- Retention: keep last historyRetentionLimit entries ---
	if err := r.pruneHistory(ctx, job.Namespace, configHash); err != nil {
		logger.Error(err, "history pruning failed — proceeding with write anyway")
	}

	// --- Create the history CR ---
	prefix := "djh-" + job.Name + "-"
	if len(prefix) > 57 {
		prefix = prefix[:57]
	}
	hist := &trainingv1.DistributedTrainingHistory{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: prefix,
			Namespace:    job.Namespace,
			Labels: map[string]string{
				"config-hash": configHash,
			},
		},
		Spec: trainingv1.DistributedTrainingHistorySpec{
			Backend:             backend,
			MachineType:         machineType,
			ConfigHash:          configHash,
			Nodes:               nk,
			Throughput:          quantityFromFloat(pk),
			TrainingSeconds:     quantityFromFloat(tk),
			TotalWork:           quantityFromFloat(tk * pk),
			ProvisioningSeconds: quantityFromFloat(tProvision),
			ActualCostUSD: func() string {
				if job.Status.Results != nil {
					return job.Status.Results.EstimatedCostUSD
				}
				return ""
			}(),
		},
	}

	if err := r.Create(ctx, hist); err != nil {
		logger.Error(err, "failed to create DistributedTrainingHistory CR")
	} else {
		logger.Info("history entry written", "configHash", configHash, "nodes", nk, "throughput", pk)
	}
}

func quantityFromFloat(f float64) resource.Quantity {
	return resource.MustParse(strconv.FormatFloat(f, 'f', -1, 64))
}

// pruneHistory deletes the oldest entries if the count for configHash is at or above the limit.
func (r *DistributedTrainingReconciler) pruneHistory(ctx context.Context, namespace, configHash string) error {
	list := &trainingv1.DistributedTrainingHistoryList{}
	if err := r.List(ctx, list,
		client.InNamespace(namespace),
		client.MatchingLabels{"config-hash": configHash},
	); err != nil {
		return err
	}

	if len(list.Items) < historyRetentionLimit {
		return nil
	}

	// Sort oldest first by creationTimestamp.
	sort.Slice(list.Items, func(i, j int) bool {
		return list.Items[i].CreationTimestamp.Before(&list.Items[j].CreationTimestamp)
	})

	toDelete := len(list.Items) - historyRetentionLimit + 1 // +1 to make room for the new entry
	for i := 0; i < toDelete; i++ {
		item := list.Items[i]
		if err := r.Delete(ctx, &item); err != nil {
			return fmt.Errorf("deleting old history entry %s: %w", item.Name, err)
		}
	}
	return nil
}

// nodeCountForHistory returns n_k for the history entry.
// For objective jobs, uses status.ResolvedTopology.Nodes.
// For explicit-topology jobs, reads from spec.
func nodeCountForHistory(job *trainingv1.DistributedTraining) int32 {
	if job.Spec.Objective != nil {
		return job.Status.ResolvedTopology.Nodes
	}

	backend := job.Spec.Backend
	if backend == "" {
		backend = trainingv1.BackendPyTorch
	}

	switch backend {
	case trainingv1.BackendPyTorch:
		if job.Spec.PytorchSpec != nil {
			return int32(job.Spec.Topology.Nodes)
		}
		return 1
	case trainingv1.BackendSpark:
		return job.Spec.Topology.Nodes
	}
	return 0
}

// extractThroughput returns P_k from the results for the given backend.
//
// Preferred path: if the backend reports total samples/records processed,
// compute wall-clock-effective throughput as totalSamples / tk. This makes
// W = T_k × P_k collapse to a constant samples-per-workload quantity in
// history, so the solver's W parameter is no longer biased by the
// fixed-overhead fraction of wall-clock time (which differs across n).
//
// Fallback path: use the backend's inner-loop throughput metric — preserves
// behavior for backends or runs where totalSamples isn't recorded.
func extractThroughput(backend trainingv1.BackendType, results *trainingv1.JobResults, tk float64) (float64, error) {
	if results == nil || results.Metrics == nil {
		return 0, fmt.Errorf("no metrics")
	}

	// Effective-throughput path (preferred).
	var totalKey string
	switch backend {
	case trainingv1.BackendPyTorch:
		totalKey = "totalSamples"
	case trainingv1.BackendSpark:
		totalKey = "recordsProcessed"
	}
	if totalKey != "" && tk > 0 {
		if val, ok := results.Metrics[totalKey]; ok && val != "" {
			if total, err := strconv.ParseFloat(val, 64); err == nil && total > 0 {
				return total / tk, nil
			}
		}
	}

	// Fallback: backend-reported inner-loop throughput.
	var key string
	switch backend {
	case trainingv1.BackendPyTorch:
		key = "samplesPerSecond"
	case trainingv1.BackendSpark:
		key = "throughputRecordsPerSec"
	default:
		return 0, fmt.Errorf("unsupported backend %q for throughput extraction", backend)
	}
	val, ok := results.Metrics[key]
	if !ok || val == "" {
		return 0, fmt.Errorf("metric %q not found", key)
	}
	f, err := strconv.ParseFloat(val, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing %q=%q: %w", key, val, err)
	}
	return f, nil
}

// computeConfigHash returns a backend-aware SHA256 hash (63 hex chars) that
// identifies a "same job" configuration for history grouping.
func computeConfigHash(job *trainingv1.DistributedTraining) string {
	backend := job.Spec.Backend
	if backend == "" {
		backend = trainingv1.BackendPyTorch
	}
	machineType := job.Spec.Hardware.MachineType

	var parts []string
	switch backend {
	case trainingv1.BackendPyTorch:
		var modelName, datasetName, datasetSplit string
		if job.Spec.Model != nil {
			modelName = job.Spec.Model.Name
		}
		if job.Spec.Dataset != nil {
			datasetName = job.Spec.Dataset.Name
			datasetSplit = job.Spec.Dataset.Split
		}
		var tr trainingv1.TrainingSpec
		if job.Spec.Training != nil {
			tr = *job.Spec.Training
		}
		parts = []string{
			string(backend),
			machineType,
			modelName,
			datasetName,
			datasetSplit,
			strconv.Itoa(int(tr.BatchSize)),
			strconv.Itoa(int(tr.GradAccumulationSteps)),
			strconv.Itoa(int(tr.Epochs)),
			tr.ValidationSplit,
		}
	case trainingv1.BackendSpark:
		ss := job.Spec.SparkSpec
		var image, mainFile, argsHash string
		if ss != nil {
			image = ss.Image
			mainFile = ss.MainApplicationFile
			h := sha256.Sum256([]byte(strings.Join(ss.Arguments, "|")))
			argsHash = fmt.Sprintf("%x", h)
		}
		parts = []string{string(backend), machineType, image, mainFile, argsHash}
	default:
		parts = []string{string(backend), machineType}
	}

	h := sha256.Sum256([]byte(strings.Join(parts, "|")))
	full := fmt.Sprintf("%x", h)
	if len(full) > 63 {
		return full[:63]
	}
	return full
}
