# Experiment runbook — Hyperstack campaign

Checklist to open at the start of every experiment session. It covers the Chapter 5
(operator evaluation) and Chapter 6 (transfer analysis) runs on the Hyperstack cluster.
Setup itself is in `README.md` → *Development and Deployment on Hyperstack*. Every thesis
measurement is taken on Hyperstack (one provider, one region, one GPU flavor). The earlier
GKE campaign in `example/experiments.md` is cited as prior validation of the GKE provider
only; it is not part of the evaluation and is not re-run.

All commands run from `operator/` with `KUBECONFIG` pointing at the cluster. Project state and
scope decisions: `PROJECT.md`.

---

## 0. Before the first run of the day (5 min)

```sh
export KUBECONFIG=../example/infra/hyperstack/kubeconfig.yaml
export HYPERSTACK_API_KEY=…                       # only needed for API checks

kubectl get nodes -L hyperstack.cloud/node-group-name   # expect: masters + default workers only
kubectl get dj -A                                        # expect: nothing running
kubectl get distributedtraininghistories -A              # know what the solver will see
```

| Check | Command | Expect |
|---|---|---|
| Operator image is the one you think | `kubectl -n distributed-training-system get deploy distributed-training-controller-manager -o jsonpath='{.spec.template.spec.containers[0].image}'` | ≥ `v3.5.0` (α search no longer capped at 1; plus v3.4.0's cleanup/teardown/GPU-gate fixes) |
| Training image in the sample file matches what is pushed | `grep -c 'gpu-fine-tuner-hf:v2.6.0' config/samples/training_v1_distributedtraining_hyperstack_gpu.yaml` | 17 (one per CR, the commented block and the header); v2.6.0 = compute fraction excludes AllReduce, on top of v2.5.0's CUDA-event AllReduce timer + probe (v2.3/v2.4 under-report AllReduce on NCCL), on top of the split layers and DataLoader-timed profiler (needed for grad-accum runs) |
| Inter-node bandwidth known | `make netprobe` (base workers) or `make netprobe NODE_GROUP=dj-e1-n2` while a 2-node job's nodes exist | a Gbit/s figure for the thesis; ~3 Gbit/s was inferred from AllReduce step times |
| Storage | `kubectl get pv,pvc` | `distributed-training-output-pv` **Bound**, CSI driver `nfs.csi.k8s.io`; NFS server pod Running |
| Monitoring | `kubectl -n monitoring get pods` | prometheus, grafana, kube-state-metrics, pushgateway Running (no node-exporter — disabled on Hyperstack) |
| Prometheus sees the operator | `kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 &` → http://localhost:9090/targets | `distributed-training-operator` target **UP** |
| Costs ConfigMap | `kubectl -n distributed-training-system get cm machine-costs -o jsonpath='{.data}'` | `n3-L40x1: 1.00`, `n3-A100x1: 1.35` |
| No leftovers in Hyperstack | console → cluster → **Node groups** | only `default` |

**History hygiene.** The solver fits α from every history entry with the same config
hash. Entries measured before the time-attribution fix (training time counted from job
creation) are poison. If any entry predates operator v3.2.0, delete all history before a
calibration series: `kubectl delete distributedtraininghistories --all`.

**Restoring history on a new cluster.** `make restore-history DRY=1` (from `operator/`) shows what
would be created from `results/*/history.yaml`; without `DRY=1` it creates the entries. Run it after
`make hs-deploy` (the CRDs must exist). It restores the newest entry of every archive, skips
`*-warmup` runs and duplicate archives, refuses a config-hash group whose runs did different work,
and can be repeated safely. `ONLY="e2-n1 e2-n10"` and `EXCLUDE="…"` pick runs. Needed when objective
runs follow E-series runs across a cluster rebuild; not needed otherwise.

---

## 1. The run matrix

| Label (`experiment=`) | Chapter | Workload | Nodes | Purpose | Expected loop |
|---|---|---|---|---|---|
| `lora-cal` | 5 | smol_llama-101M LoRA, medical flashcards | 1, then 2 (edit `nodes`, `metadata.name`, `nodePoolName`) | solver calibration points | ~3 min / ~2 min |
| objective block (commented in the file) | 5 | same | solver picks | pilot: solver on the small model | ~3 min |
| `e1-n1` `e1-n2` `e1-n5` `e1-n7` `e1-n10` | 6 | Pythia-410M **full** FT, synthetic 40k samples, grad-accum 64 | 1 / 2 / 5 / 7 / 10 | AllReduce overhead in isolation | ~18 / ~15 / shorter each step up / … min |
| `e2-n1` `e2-n2` `e2-n5` `e2-n7` `e2-n10` | 6 | Pythia-410M full FT, WikiText-103 (NFS), 40k packed 512-token blocks (same token count as E1), grad-accum 64 | 1 / 2 / 5 / 7 / 10 | AllReduce + NFS data loading; `e2-n10` is the naive baseline | similar; first run also tokenises |
| `obj-e2-2pt` | 5 | E2 workload, `objective.targetTime`, no topology | solver picks from 2..10 (no pick assumed) | solver accuracy from a **two-point** fit on n = 1 and 10; picks a node count it never measured | ~15 min |
| `obj-e2-tight` | 5 | same | solver picks (expect 10) | five-point fit, tight target: solver agrees with the naive maximum | ~9 min |
| `obj-e2-mid` | 5 | same | solver picks (an unmeasured count between measured ones) | five-point fit, intermediate target: topology selection and **cost saving vs naive** (`e2-n10`) | ~11 min |
| `e2-n2-rep` | 5/6 | identical to `e2-n2` | 2 | run-to-run variance (noise floor for every comparison) | ~15 min |

The four rows above (three objective runs and the repeat) use the E2 workload on purpose: its config hash is the one the E2
history entries carry, so the objective CRs read real history instead of a synthetic
calibration. The objective CRs use `maxNodes: 10`, and the node set 1, 2, 5, 7, 10 is chosen so
the solver interpolates between measured counts instead of extrapolating past them.
**Naive baseline = `e2-n10`**: provisioning the maximum allowed nodes for the same fixed work.
No separate naive run is needed; the cost saving is the objective run's actual cost against
`e2-n10`'s. The solver never picks fewer than 2 nodes, so a 1-node answer is not part of the
selection space.
**Before the first 10-node run,** check the account's GPU stock in the Hyperstack console:
the docs describe stock as a shared regional pool and do not state an account quota.
**Why gradient accumulation 64 (2026-11-05).** Without it, every 4-sample micro-batch
triggered a 1.6 GB AllReduce; on this ~3 Gbit/s interconnect that made 4 nodes **45× slower
per step** than 1 (5.9 s vs 0.13 s; η ≈ 0.02, α ≈ 14) and each multi-node run 4–6 h. One
finding, repeated. With 64 micro-batches per optimizer step the per-node compute (~7 s)
is of the same order as the AllReduce (~4 s at 2 nodes, ~6 s at 4), so η varies with n and
data source and the solver/classifier have something to discriminate. Optimizer steps per
run: 156 / 78 / 39 at 1 / 2 / 4 nodes (LR warm-up is `--warmup_ratio=0.1`). Entries recorded
before this change (accum 1, `find_unused_parameters` on) are a different series: delete
their history before fitting.

Order (the position of the objective runs matters: it fixes what history each one sees):

1. `lora-cal` ×2 → the commented lora objective block (pilot).
2. `e1-n1` (warm-up) → `e1-n1` → `e1-n2` → `e1-n5` → `e1-n7` → `e1-n10`.
3. `e2-n1` (warm-up) → `e2-n1` → `e2-n10` (naive baseline, and the largest point of the history).
4. `obj-e2-2pt`: history holds n=1 and n=10 only, so it is a two-point, held-out prediction.
5. `e2-n2`. Now history has n = 1, 2, 10.
6. `e2-n5` → `e2-n7`. History now has all five node counts.
7. `obj-e2-tight` → `obj-e2-mid` (five-point fit).
8. `e2-n2-rep` last, so it cannot change the fit the solver runs used.

Before steps 4 and 7 set each `targetTime` from the measured history, not from the
provisional value in the file:

```sh
python3 ../script/solver_whatif.py                                   # list config hashes found in results/
python3 ../script/solver_whatif.py --hash <e2 hash> --subset 1,10 --target <T>   # before obj-e2-2pt
python3 ../script/solver_whatif.py --hash <e2 hash> --target 12m                # before tight / mid
```

The first prints predicted training time and cost per node count and which node count the
solver would pick for that target. It mirrors the operator's estimator, so what it prints is
what the operator will decide. The target that matters is the one you commit before the run:
write it, the predicted time and the predicted cost into your notes first, then apply the CR.

**Why a warm-up per series:** the first run of a new model/dataset downloads weights into
`/mnt/output/hf-cache` and (E2) tokenises WikiText-103 into the PVC cache; its training
time is not comparable. Archive it anyway, named `*-warmup`, then **delete its history
entry** (`kubectl get distributedtraininghistories`, delete the newest one): the operator
records every successful run, and a warm-up entry at n=1 would be averaged into the baseline
the solver fits. `solver_whatif.py` already skips `*-warmup` directories, so offline and
on-cluster fits stay identical.

**Fixed work.** Every run of a series processes the same 40k samples (`--synthetic_samples`
/ `--max_train_samples`). If the first E1 run's loop is far from 20 min, change that number
identically in all six E documents before continuing — never mid-series.

---

## 2. Per-run procedure

```sh
# 1. start exactly one CR (refuses if the label matches ≠ 1 documents)
make run-experiment EXP=e1-n2

# 2. watch the phases: Pending → Provisioning → Ready → Running → Collecting → Succeeded
make hs-watch
make hs-logs            # in a second terminal if something looks slow

# 3. when Succeeded: verify (section 3), then archive
make hs-archive-results CR=e1-n2-pythia410m RUN=e1-n2          # RUN=…-warmup for warm-ups

# 4. warm-ups only: delete the newest history entry (see §1, "Why a warm-up per series")
kubectl get distributedtraininghistories
kubectl delete distributedtraininghistory <newest djh-… name>

# 5. confirm teardown
kubectl get nodes | grep -v -e default- -e master-               # nothing
#   Hyperstack console → Node groups: only `default`

# 6. log the run: regenerate the run log; add anomalies (if any) first
#    echo "pod pending 4 min" >> ../results/e1-n2/notes.txt      # optional, one line per anomaly
make runlog                                                      # from operator/: writes results/RUNLOG.md

# 7. only then start the next run
```

The 10-node runs (`e1-n10`, `e2-n10`) take the longest to provision, and the Hyperstack console
is the only place to check stock for ten L40s before the first one. Check it before step 1.

Typical timings on `n3-L40x1`: Provisioning 7–8 min (VM ~5 min, GPU plugin ~2 min),
image pull 1–3 min (between Ready and Running), teardown ~1 min, stale-node cleanup
≤ 2 min after that.

**Deleting a run early:** `kubectl delete dj <name>` is safe in any phase with operator
≥ v3.3.0; the CR disappears only when the node group is confirmed gone. If it is still
Provisioning, that can take until the create finishes (clouds refuse to delete a pool
mid-creation).

---

## 3. What to verify on every run (2 min)

```sh
NAME=e1-n2-pythia410m
kubectl get dj $NAME -o jsonpath='{.status.phase} prov={.status.results.provisioningTime} train={.status.results.trainingTime} cost={.status.results.estimatedCostUSD} bottleneck={.status.results.bottleneck}{"\n"}'
kubectl get events --field-selector involvedObject.name=$NAME -o custom-columns='T:.lastTimestamp,R:.reason,M:.message' | grep -E "Bottleneck|History|Collect"
kubectl get distributedtraininghistories -o custom-columns='NAME:.metadata.name,N:.spec.nodes,P:.spec.throughput,T:.spec.trainingSeconds,AR:.spec.allReduceFraction,DL:.spec.dataloadFraction,CP:.spec.computeFraction'
```

| Check | Where | Pass |
|---|---|---|
| metrics were collected | Succeeded event | *"N metrics collected"* with N > 0 — **not** *"0 metrics collected"* |
| history written | events | no `HistorySkipped`; a new `djh-<name>-…` entry exists |
| fractions sane | `transfer_profile.json` | `dataloader_timed: true`, `compute_excludes_allreduce: true`; `compute + allreduce + dataload` ≈ 0.95–1 (remainder = Trainer bookkeeping). A sum near 1.4 means an image older than v2.6.0 |
| **AllReduce timer verified** (multi-node) | `transfer_profile.json` | `allreduce_timing: cuda_events` and `allreduce_timing_check: ok` (hook time per step within 0.5 to 2 times the blocking probe, `allreduce_vs_probe_ratio` between about 0.7 and 1.4; the first real 2-node run gave 0.79, the probe being the slower of the two). `allreduce_bandwidth_MBs` must be below the link rate (~400 to 1000 MB/s here; `make netprobe`). **A bandwidth in GB/s means the timer measured kernel launch: discard the run's fractions.** Cross-check with the step times: AllReduce per step is about `avg_step_s(n) − avg_step_s(1)` |
| instrumentation covers every gradient byte | `transfer_profile.json` | `hook_installed: true` and `hook_coverage: 1.0` on multi-node runs; `false`/`null` at 1 node |
| gradient size is what the model implies | `grad_size_bytes` | full FT Pythia-410M ≈ 1.6 GB (405M × 4 B); LoRA smol_llama ≈ 369 kB |
| NFS reads happened (E2) | `nfs_read_bytes` | > 0, and about the same for all n (fixed work, split across nodes; rank 0's share shrinks with n). 0 means page cache or synthetic data |
| time attribution | `trainingTime` vs `train_runtime` | gap ≈ model load + eval + save (~1–2 min); a gap of 4–5 min means image pull / GPU lag leaked in (old operator) |
| teardown clean | `kubectl get nodes`, console | no `dj-*` nodes, no `dj-*` node group |

**Run log:** do not type it by hand. `make runlog` (from `operator/`) writes
`results/RUNLOG.md` from the archives: one row per run with date, label, nodes, provisioning,
training, inner loop, throughput, α (analysis, wall clock), bottleneck, eval loss and cost.
The only manual input is `results/<RUN>/notes.txt` for anomalies. Re-run `make runlog` after each
archive. Do not edit `RUNLOG.md` by hand; it is overwritten.

---

## 4. Where each thesis number lives

| Quantity | Source of truth | Prometheus metric | Grafana panel |
|---|---|---|---|
| provisioning time | history `provisioningSeconds`; CR `results.provisioningTime` | `distributedtraining_provisioning_seconds` | Training Overview |
| training time (Running→completion) | history `trainingSeconds` | `distributedtraining_training_seconds` | Training Overview |
| inner-loop time | `all_results.json` `train_runtime` | — (not exported; `samplesPerSecond` is its reciprocal × samples) | Scaling Summary → inner-loop bar |
| throughput used by the solver (W / T) | history `throughput` | `training_metric{metric="totalSamples"} / training_seconds` | Scaling Summary → Throughput per run |
| inner-loop throughput | `all_results.json` `train_samples_per_second` | `training_metric{metric="samplesPerSecond"}` | Training Metrics |
| α, predicted time/cost, chosen nodes | solver event on the objective CR; CR status | `distributedtraining_solver_alpha`, `_solver_estimated_time_seconds`, `_solver_estimated_cost_usd`, `_solver_nodes_selected` | Solver Accuracy / Learning |
| actual cost | history `actualCostUSD`; CR status | `distributedtraining_cost_usd_actual` | Training Overview |
| AllReduce / dataload / compute fractions | `transfer_profile.json`; history | `distributedtraining_allreduce_fraction`, `_dataload_fraction`, `_compute_fraction` | Transfer Analysis → Transfer fractions per run |
| AllReduce & NFS bandwidth | `transfer_profile.json`; history | `distributedtraining_allreduce_bandwidth_mbs`, `_nfs_read_bandwidth_mbs` | Transfer Analysis → bandwidth panels |
| gradient / wire / NFS bytes, hook coverage | `transfer_profile.json`; history (bytes) | `training_metric{metric=…}` catch-all | — |
| bottleneck class | CR `status.results.bottleneck`; `Bottleneck` event | derived from the three fractions (same rule as the operator) | Transfer Analysis → Bottleneck class per run |
| eval loss / perplexity | `all_results.json`, `eval_results.json` | `training_metric{metric="loss"|"perplexity"}` | Training Metrics → Loss; Quality & Lifecycle → Perplexity |
| per-step loss curve | `trainer_state.json` | — | — |

Gauges hold the **latest value per CR name**; Prometheus retention is 90 days. The history
CRs and the archived JSON are the durable record; Grafana is the live view and the
screenshot source.

**Ad-hoc PromQL** (Grafana → Explore) — the same expressions back the new panels; useful for one-off variants:

```promql
# transfer fractions per run
distributedtraining_allreduce_fraction{namespace="default"}
distributedtraining_dataload_fraction{namespace="default"}
distributedtraining_compute_fraction{namespace="default"}

# bottleneck class as a number: 1 compute-bound, 2 communication-bound, 3 data-bound
  1 * (distributedtraining_compute_fraction >= bool 0.6)
+ 2 * ((distributedtraining_compute_fraction < bool 0.6) * (distributedtraining_allreduce_fraction >= bool distributedtraining_dataload_fraction))
+ 3 * ((distributedtraining_compute_fraction < bool 0.6) * (distributedtraining_allreduce_fraction <  bool distributedtraining_dataload_fraction))

# scaling efficiency η = per-node wall-clock throughput, normalised later by the 1-node value
(distributedtraining_training_metric{metric="totalSamples"} / distributedtraining_training_seconds) / on(namespace,dj_name) distributedtraining_nodes_provisioned

# α the solver fitted (objective runs only)
distributedtraining_solver_alpha
```

Grafana: `make monitoring-grafana` → http://localhost:3000 (admin / your password) →
dashboard **DistributedTraining Operator**; variables `namespace=default`, `dj_name` = regex
over the CR names of one series, e.g. `e1-n.*`. The *Per-run scaling summary* table is the
thesis table (§6.6/§6.7) in one panel. After editing the dashboard JSON in the repo, reload
it with `kubectl apply -k config/grafana` (the sidecar picks it up within a minute).

---

## 5. Reading the results

- **α two ways.** The solver's α comes from wall-clock training time; recompute it from
  `train_runtime` as well (η = T₁/(n·Tₙ), α = (1−η)/(η(n−1))). The difference is the
  serial remainder (model load, eval, save) — report both (§3.6/§5.5).
- **AllReduce fraction vs α (§6.6).** One point per run: x = `allreduce_fraction`, y = α
  fitted by the solver after that run, or the recomputed α. Expect monotone growth with n
  for E1; E2 adds the dataload fraction.
- **Bytes check (§6.4).** `hook_coverage` = 1.0 is the instrumentation validation;
  `total_allreduce_bytes` = `total_reduced_bytes` × 2(n−1)/n is derived, say so.
- **Bottleneck table (§6.7).** One row per run: fractions + class. LoRA runs are the
  known-compute-bound control.
- **One command for all of it.** `make analyze` (from `operator/`; `NETPROBE_GBIT=3.1` adds the
  iperf3 rate to the bandwidth figure) reads `results/` and writes `analysis_out/tables.md`, one
  CSV per table and the figures. It skips `*-warmup` runs, and skips any table or figure whose data
  does not exist yet, so it can be run after every archived run.
- **Cost (§5.4).** actual = nodes × price × (provisioning + training)/3600; the solver's
  estimate is in the objective CR's status.
- **Solver accuracy (§5.2).** Two sources, reported separately. (1) The four live
  predictions: for `obj-e2-2pt`, `obj-e2-tight`, `obj-e2-mid` (and the lora pilot) take the
  solver's predicted training time from the CR status and compare with the actual
  `trainingSeconds`; give the signed error per run, not only a mean. (2) Leave-one-out on
  the archived history, which costs no GPU time:
  `python3 ../script/solver_whatif.py --hash <e2 hash> --loo`. State plainly that the sample is
  small (five node counts, each held-out fit uses the other four) and that the held-out points are 1, 2, 5, 7 and 10 nodes.
- **Two-point vs five-point fit (§5.2).** `obj-e2-2pt` is the live two-point prediction.
  Reproduce it offline with `--subset 1,10`, and show the same prediction after `e2-n2`
  joined the history (`--subset 1,2,10`) and after `e2-n5`/`e2-n7` (`--subset 1,2,5,7,10`).
  The gap between these is the value of the extra points.
- **Topology selection (§5.3).** Table of objective run, target, predicted n, chosen n,
  whether the target was met. The solver's choice set is n in {2, ..., 10}.
- **Cost saving vs naive (§5.4).** For each objective run that met its target:
  saving = actual cost of `e2-n10` (naive maximum, same fixed work) minus the run's actual cost.
  For `obj-e2-tight` the saving is expected to be zero; that is the correct answer, say so.
- **Variance.** `e2-n2` vs `e2-n2-rep` (and any objective run that chose 2 nodes) gives
  the relative run-to-run difference in training time and in each fraction. Quote it next to
  every claim that compares two runs; differences smaller than it are not findings.
- **Lifecycle overhead (§5.5).** Provisioning and teardown times of all 19 runs form the
  distribution (median, min, max); the collection overhead is `Collecting` → `Succeeded`.

---

## 6. When something goes wrong

| Symptom | Cause | Do |
|---|---|---|
| pod `Pending`, *Insufficient nvidia.com/gpu* | GPU plugin not up yet on the new node | operator ≥ v3.2.0 waits for it in Provisioning; with older images just wait ~2 min |
| *failed to extract layer … context canceled* | containerd cancels pulls that stall > ~1 min while unpacking a big layer | `v2.1.0`: wait for the kubelet's retries (2–3 attempts); fix = image `v2.2.0` (split layers) |
| *MountVolume.SetUp failed … mount.nfs helper* | PV is the in-tree NFS one; nodes have no NFS client | use `config/samples/pvc-nfs-csi.yaml` (README Step 3) |
| *CSINode … does not contain driver hyperstack.csi* | CSI node plugin not registered yet | transient; it retries |
| Succeeded with *0 metrics collected*, `HistorySkipped` | reader pod landed on the node being deleted | fixed ≥ v3.1.0 (anti-affinity); on older: files are on NFS, history is lost → rerun |
| CR gone but `dj-*` node group still in Hyperstack | deleted during Provisioning with operator < v3.3.0 | README → *Remediation — a node group the operator no longer tracks* |
| `NotReady` `dj-*` node lingers | kubelet heartbeat outlived the VM; operator < v3.4.0 skipped it once | `kubectl delete node <name>`; ≥ v3.4.0 retries by itself |
| *targetTime … unreachable at maxNodes* | with the current α no node count meets the target (training time only, provisioning excluded) | pick a reachable target from `predT(n) = W / (p·n·η(n))`; check the history isn't mixing old/new attribution |
| solver picked 1 node for an objective run | cheapest configuration that meets the target; correct behaviour | tighten `targetTime` if you want to see it scale |
| Grafana shows no new run | operator target down, or `honorLabels` lost the CR namespace | check Prometheus targets; `kubectl -n distributed-training-system get svc distributed-training-controller-manager-metrics-http` |

---

## 7. Viewing results without the cluster (offline Grafana)

The dashboard is per-run gauges, and every value is in `results/<RUN>/`. A local stack
renders the same dashboard from the archives — no GPU, no cluster:

```sh
make replay-up                         # Prometheus + Pushgateway + Grafana in Docker (from operator/)
make replay-results                    # push all archived runs; ONLY="e1-n1 e1-n2" to restrict
# Grafana: http://localhost:3000  (admin / admin) → DistributedTraining Operator
make replay-down                       # stop; data under example/local-grafana/data/ is kept
```

`replay-results` is a **sync** with `results/`: a run shows up when its directory (with
`cr.yaml`) exists and disappears from the dashboard when the directory is moved out and
the command is re-run (`--no-prune` on the script keeps stale runs; `ONLY=…` never prunes).
Runs are keyed by the CR name inside `cr.yaml`, so two archives of the same CR name
overwrite each other — give warm-up runs their own CR name if both must stay visible.

What replays: every operator gauge (durations, cost, nodes, the full `training_metric`
set, the five transfer gauges, objective and resolved-topology values) plus the phase
counters the dashboard variables are built from. **α** is not stored anywhere, so for
objective runs it is **re-fitted** with the operator's own estimator from the archived
history entries of the same config hash created before that run — the value the solver
actually saw; fixed-topology runs get no α, exactly as on the cluster. What does not
replay: time series shapes (everything is pushed "now").

For the real time series, **snapshot the cluster's Prometheus before teardown**:

```sh
make prometheus-snapshot               # → results/prometheus-snapshots/<name>/ (TSDB blocks)
# later, offline:
cp -r ../results/prometheus-snapshots/<name>/* ../example/local-grafana/data/prometheus/
make replay-up                         # Prometheus serves the blocks with original timestamps
```

Add `make prometheus-snapshot` to the end-of-campaign checklist below; it needs the
Prometheus pod and takes seconds. The dashboard JSON is mounted from the repo, so panel
edits are tested on the local stack first and then pushed with `kubectl apply -k config/grafana`.

## 8. End of session

```sh
kubectl get dj -A                       # nothing Running/Provisioning
kubectl get nodes | grep dj-            # nothing
# Hyperstack console: Node groups = default only; Volumes: nothing attached to dead VMs
```

Before a campaign-ending teardown: `make prometheus-snapshot`, and make sure every run has
a `results/<RUN>/` directory (section 7 replays from them).

The base cluster (2 masters + 1–2 CPU workers), the NFS disk and the Prometheus volume bill
while they exist. Full teardown is README → Hyperstack *Teardown* (`make monitoring-uninstall`
first, then the NFS workload, then `terraform destroy`). Results live in `results/<RUN>/`
and in the history CRs. Both survive an operator redeploy. The history CRs live in the cluster, so a
cluster rebuild deletes them; the archive (`results/<RUN>/history.yaml`) survives and
`make restore-history` recreates them. Archive before any teardown.
