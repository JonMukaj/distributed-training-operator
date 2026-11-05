#!/usr/bin/env python3
"""Replay archived experiment runs into a Prometheus Pushgateway so the operator's
Grafana dashboard can be viewed without a cluster.

Reads every results/<RUN>/ directory written by `make hs-archive-results`
(cr.yaml, history.yaml, transfer_profile.json, all_results.json) and pushes the
SAME metric names and labels the operator exports (operator/internal/controller/
metrics.go), one Pushgateway group per run:

  distributedtraining_{training,provisioning,collection}_seconds
  distributedtraining_cost_usd_actual, _nodes_provisioned
  distributedtraining_training_metric{metric=…}          (every key of status.results.metrics)
  distributedtraining_{allreduce,dataload,compute}_fraction,
  distributedtraining_{allreduce,nfs_read}_bandwidth_mbs
  distributedtraining_solver_{alpha,nodes_selected,estimated_time_seconds,estimated_cost_usd}
  distributedtraining_objective_{target_time_seconds,max_cost_usd}
  distributedtraining_phase_transitions_total{phase=…}   (dashboard variables are built from it)

α is not stored in the CR or history, and the operator only exports it for
OBJECTIVE runs (the solver does not run for fixed-topology CRs). The replay mirrors
that: solver_alpha is pushed only for runs with spec.objective, re-fitted with the
operator's own method (closed form for two node counts, least-squares grid search
for three or more) from the archived history entries of the same config hash that
were created BEFORE that run's CR — i.e. the α the solver actually saw.

The push is a SYNC: a full replay (no --only) also deletes Pushgateway groups whose
run directory no longer exists, so removing a run from results/ and re-running
removes it from the dashboard. --no-prune keeps stale groups.

Usage:
  python3 script/replay_metrics.py [--results ../results] [--gateway http://localhost:9091]
                                   [--only RUN ...] [--no-prune] [--dry-run]
Requires PyYAML (python3-yaml) and nothing else.
"""
import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import yaml

PHASES = ["Pending", "Provisioning", "Ready", "Running", "Collecting"]

# status.results.metrics key → dedicated gauge name (mirrors phases.go)
FRACTION_GAUGES = {
    "allreduceFraction": "distributedtraining_allreduce_fraction",
    "dataloadFraction": "distributedtraining_dataload_fraction",
    "computeFraction": "distributedtraining_compute_fraction",
    "allreduceBandwidthMBs": "distributedtraining_allreduce_bandwidth_mbs",
    "nfsReadBandwidthMBs": "distributedtraining_nfs_read_bandwidth_mbs",
}

_DUR = re.compile(r"(\d+(?:\.\d+)?)(h|m|s|ms)")


def go_duration_seconds(s):
    """'7m24s' → 444.0; returns None for non-durations like 'unknown (calibration run)'."""
    if not s or not _DUR.search(s) or not re.fullmatch(r"(\d+(?:\.\d+)?(h|m|s|ms))+", s.strip()):
        return None
    mult = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(v) * mult[u] for v, u in _DUR.findall(s))


def quantity(v):
    """Kubernetes resource.Quantity → float ('68080200502n' → 68.08)."""
    s = str(v).strip()
    suffixes = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "Ki": 2**10, "Mi": 2**20, "Gi": 2**30}
    for suf, f in sorted(suffixes.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * f
    return float(s)


def fnum(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def esc(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Run:
    def __init__(self, d: Path):
        self.dir = d
        self.cr = yaml.safe_load(open(d / "cr.yaml")) if (d / "cr.yaml").exists() else None
        hist = yaml.safe_load(open(d / "history.yaml")) if (d / "history.yaml").exists() else None
        self.history = None
        if hist:
            items = hist.get("items", [hist]) if hist.get("kind") == "List" else [hist]
            self.history = items[-1]["spec"] if items else None
        self.tp = json.load(open(d / "transfer_profile.json")) if (d / "transfer_profile.json").exists() else {}
        self.ar = json.load(open(d / "all_results.json")) if (d / "all_results.json").exists() else {}
        if self.cr is None:
            raise ValueError(f"{d}: cr.yaml missing — cannot determine namespace/dj_name")
        self.namespace = self.cr["metadata"].get("namespace", "default")
        self.name = self.cr["metadata"]["name"]
        self.status = self.cr.get("status", {}) or {}
        self.results = self.status.get("results", {}) or {}
        self.metrics = self.results.get("metrics", {}) or {}
        self.created = self.cr["metadata"].get("creationTimestamp", "")
        self.history_created = ""
        if hist:
            items = hist.get("items", [hist]) if hist.get("kind") == "List" else [hist]
            if items:
                self.history_created = items[-1].get("metadata", {}).get("creationTimestamp", "")

    @property
    def is_objective(self):
        return bool(self.cr["spec"].get("objective"))

    @property
    def config_hash(self):
        return self.history.get("configHash") if self.history else None

    @property
    def nodes(self):
        if self.history and self.history.get("nodes"):
            return int(self.history["nodes"])
        rt = self.status.get("resolvedTopology") or {}
        if rt.get("nodes"):
            return int(rt["nodes"])
        return int((self.cr["spec"].get("topology") or {}).get("nodes") or 0) or None

    @property
    def throughput(self):
        return quantity(self.history["throughput"]) if self.history and "throughput" in self.history else None

    def samples(self, alpha=None):
        """Yield (metric, labels dict, value)."""
        L = {"namespace": self.namespace, "dj_name": self.name}
        r, m = self.results, self.metrics

        for key, metric in (("trainingTime", "distributedtraining_training_seconds"),
                            ("provisioningTime", "distributedtraining_provisioning_seconds"),
                            ("collectionTime", "distributedtraining_collection_seconds")):
            v = go_duration_seconds(r.get(key, ""))
            if v is None and self.history:  # fall back to history
                hk = {"trainingTime": "trainingSeconds", "provisioningTime": "provisioningSeconds"}.get(key)
                v = fnum(self.history.get(hk)) if hk else None
            if v is not None:
                yield metric, L, v

        cost = fnum(r.get("estimatedCostUSD")) or (fnum(self.history.get("actualCostUSD")) if self.history else None)
        if cost is not None:
            yield "distributedtraining_cost_usd_actual", L, cost
        if self.nodes:
            yield "distributedtraining_nodes_provisioned", L, float(self.nodes)

        for k, v in m.items():
            f = fnum(v)
            if f is not None:
                yield "distributedtraining_training_metric", {**L, "metric": k}, f
                if k in FRACTION_GAUGES:
                    yield FRACTION_GAUGES[k], L, f
        # history carries the fractions too (for runs whose CR was deleted before archiving)
        if self.history and "allreduceFraction" not in m:
            for hk, metric in (("allReduceFraction", "distributedtraining_allreduce_fraction"),
                               ("dataloadFraction", "distributedtraining_dataload_fraction"),
                               ("computeFraction", "distributedtraining_compute_fraction"),
                               ("allReduceBandwidthMBs", "distributedtraining_allreduce_bandwidth_mbs"),
                               ("nfsReadBandwidthMBs", "distributedtraining_nfs_read_bandwidth_mbs")):
                f = fnum(self.history.get(hk))
                if f is not None:
                    yield metric, L, f
        if "totalSamples" not in m and self.history and self.history.get("totalWork"):
            yield "distributedtraining_training_metric", {**L, "metric": "totalSamples"}, fnum(self.history["totalWork"])

        rt = self.status.get("resolvedTopology") or {}
        if rt.get("nodes"):
            yield "distributedtraining_solver_nodes_selected", L, float(rt["nodes"])
        t = go_duration_seconds(rt.get("estimatedTime", ""))
        if t is not None:
            yield "distributedtraining_solver_estimated_time_seconds", L, t
        c = fnum(rt.get("estimatedCost"))
        if c is not None:
            yield "distributedtraining_solver_estimated_cost_usd", L, c
        obj = self.cr["spec"].get("objective") or {}
        t = go_duration_seconds(obj.get("targetTime", ""))
        if t is not None:
            yield "distributedtraining_objective_target_time_seconds", L, t
        c = fnum(obj.get("maxCost"))
        if c is not None:
            yield "distributedtraining_objective_max_cost_usd", L, c
        if alpha is not None:
            yield "distributedtraining_solver_alpha", L, alpha

        final = self.status.get("phase") or "Succeeded"
        for ph in PHASES + ([final] if final not in PHASES else []):
            yield "distributedtraining_phase_transitions_total", {**L, "phase": ph}, 1.0


def eta(n, alpha):
    return 1.0 / (1.0 + alpha * (n - 1))


def fit_alpha(runs):
    """Operator's estimateAlpha on a group of runs sharing a config hash."""
    by_n = {}
    for r in runs:
        if r.nodes and r.throughput:
            by_n.setdefault(r.nodes, []).append(r.throughput)
    ns = sorted(by_n)
    if len(ns) < 2:
        return None
    avg = {n: sum(v) / len(v) for n, v in by_n.items()}
    p_base = avg[ns[0]] / ns[0]
    if len(ns) == 2:
        n1, n2 = ns
        eta2 = (avg[n2] / n2) / (avg[n1] / n1)
        return max(0.0, (1 - eta2) / (eta2 * (n2 - 1))) if eta2 > 0 else 0.0
    best, best_sse = 0.0, float("inf")
    a = 0.0
    while a <= 1.0:
        sse = sum((p_base * n * eta(n, a) - avg[n]) ** 2 for n in ns)
        if sse < best_sse:
            best, best_sse = a, sse
        a += 1e-4
    return best


def exposition(samples):
    lines = []
    for metric, labels, value in samples:
        lab = ",".join(f'{k}="{esc(v)}"' for k, v in sorted(labels.items()))
        lines.append(f"{metric}{{{lab}}} {value!r}")
    return "\n".join(lines) + "\n"


JOB = "distributedtraining-replay"


def existing_groups(gateway):
    """instance → set of metric names currently held by the Pushgateway for our job."""
    with urllib.request.urlopen(f"{gateway.rstrip('/')}/api/v1/metrics", timeout=10) as resp:
        data = json.load(resp)["data"]
    return {g["labels"]["instance"] for g in data if g["labels"].get("job") == JOB and "instance" in g["labels"]}


def delete_group(gateway, instance, dry):
    url = f"{gateway.rstrip('/')}/metrics/job/{JOB}/instance/{instance}"
    if dry:
        print(f"--- would DELETE {url}")
        return
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 202):
            raise RuntimeError(f"DELETE {url}: HTTP {resp.status}")


def push(gateway, run, body, dry):
    url = f"{gateway.rstrip('/')}/metrics/job/{JOB}/instance/{run.name}"
    if dry:
        print(f"--- {url}\n{body}")
        return
    req = urllib.request.Request(url, data=body.encode(), method="PUT",
                                 headers={"Content-Type": "text/plain; version=0.0.4"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 202):
            raise RuntimeError(f"{url}: HTTP {resp.status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--gateway", default="http://localhost:9091")
    ap.add_argument("--only", nargs="*", help="RUN directory names to replay (default: all)")
    ap.add_argument("--no-prune", action="store_true", help="keep Pushgateway groups whose run directory is gone (default: prune on a full replay)")
    ap.add_argument("--dry-run", action="store_true", help="print the exposition text instead of pushing")
    args = ap.parse_args()

    root = Path(args.results)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "cr.yaml").exists())
    if args.only:
        dirs = [d for d in dirs if d.name in set(args.only)]
    if not dirs:
        sys.exit(f"no archived runs with cr.yaml under {root}")

    runs = []
    for d in dirs:
        try:
            runs.append(Run(d))
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"skip {d.name}: {e}", file=sys.stderr)

    groups = {}
    for r in runs:
        if r.config_hash:
            groups.setdefault(r.config_hash, []).append(r)

    def alpha_for(r):
        """α the solver saw for an objective run: fitted from the series' history
        entries that existed before this CR was created (RFC3339 strings sort)."""
        if not r.is_objective or not r.config_hash:
            return None
        prior = [x for x in groups[r.config_hash] if x.history_created and r.created and x.history_created < r.created]
        return fit_alpha(prior) if prior else fit_alpha(groups[r.config_hash])

    for r in runs:
        alpha = alpha_for(r)
        body = exposition(r.samples(alpha))
        push(args.gateway, r, body, args.dry_run)
        n = body.count("\n")
        a = f"alpha={alpha:.3f} (objective run)" if alpha is not None else "alpha=— (fixed topology)"
        print(f"{'would push' if args.dry_run else 'pushed'} {n:3d} samples  {r.name:34s} nodes={r.nodes} {a}  ({r.dir.name})")
    if not args.only and not args.no_prune:
        try:
            stale = existing_groups(args.gateway) - {r.name for r in runs}
        except Exception as e:  # noqa: BLE001
            stale = set(); print(f"prune skipped (cannot list gateway groups: {e})", file=sys.stderr)
        for inst in sorted(stale):
            delete_group(args.gateway, inst, args.dry_run)
            print(f"{'would remove' if args.dry_run else 'removed'} stale run {inst} (no results directory)")
    if not args.dry_run:
        print("\nPrometheus scrapes the gateway every 5 s; open Grafana → DistributedTraining Operator.")


if __name__ == "__main__":
    main()
