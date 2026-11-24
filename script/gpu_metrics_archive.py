#!/usr/bin/env python3
"""Archive the GPU metrics of one run from Prometheus into results/<RUN>/gpu_metrics.json.

    script/venv/bin/python script/gpu_metrics_archive.py --run e1-n1-pythia410m-batch64 \
        --prom http://localhost:9090            # Prometheus must be reachable (port-forward)

The training window comes from the archived cr.yaml (status.trainingStartTime to status.finishTime).
Pods are the PyTorchJob pods of the run: pj-<cr name>-master-0 and pj-<cr name>-worker-*.
Samples are taken every 15 s with query_range, so every value keeps its timestamp, which the
dashboard needs. Nothing is written if Prometheus has no data for the run.

Metric names are the NVIDIA DCGM exporter names. DCGM_FI_PROF_PIPE_TENSOR_ACTIVE is the usual name for
the "Tensor Core Utilization" panel of the NVIDIA dashboard; confirm it with the metric query in
EXPERIMENTS.md before relying on the file.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

METRICS = [
    "DCGM_FI_DEV_GPU_UTIL",             # busy time: fraction of time any kernel ran (overstates compute use)
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",  # tensor-core activity (profiling metric)
    "DCGM_FI_DEV_POWER_USAGE",          # watts
    "DCGM_FI_DEV_FB_USED",              # framebuffer memory, MiB
    "DCGM_FI_DEV_SM_CLOCK",             # MHz
    "DCGM_FI_DEV_GPU_TEMP",             # degrees C
]
STEP = 15


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def query_range(prom, metric, pod_regex, start, end):
    params = urllib.parse.urlencode({
        "query": f'{metric}{{exported_pod=~"{pod_regex}"}}',
        "start": f"{start:.0f}", "end": f"{end:.0f}", "step": str(STEP),
    })
    with urllib.request.urlopen(f"{prom}/api/v1/query_range?{params}", timeout=60) as r:
        body = json.load(r)
    if body.get("status") != "success":
        raise RuntimeError(f"{metric}: {body}")
    return body["data"]["result"]


def summarize(values):
    vals = [float(v) for _, v in values]
    if not vals:
        return None
    return {"mean": round(sum(vals) / len(vals), 3), "max": round(max(vals), 3), "samples": len(vals)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="archive folder name under results/")
    ap.add_argument("--prom", default="http://localhost:9090")
    ap.add_argument("--results", default=str(RESULTS))
    a = ap.parse_args()

    run_dir = Path(a.results) / a.run
    cr = yaml.safe_load((run_dir / "cr.yaml").read_text())
    status = cr.get("status") or {}
    start_s, end_s = status.get("trainingStartTime"), status.get("finishTime")
    if not (start_s and end_s):
        sys.exit("cr.yaml has no trainingStartTime/finishTime: cannot pick the window")
    start, end = parse_time(start_s), parse_time(end_s)
    job = cr["metadata"]["name"]
    pod_regex = f"pj-{job}-(master|worker)-.*"

    out = {"run": a.run, "cr": job, "window": {"start": start_s, "end": end_s},
           "pod_regex": pod_regex, "step_seconds": STEP, "series": [], "summary": {}}
    for metric in METRICS:
        try:
            result = query_range(a.prom, metric, pod_regex, start, end)
        except OSError as e:
            sys.exit(f"cannot reach Prometheus at {a.prom}: {e}")
        if not result:
            print(f"[warn] {metric}: no samples for {pod_regex} in the window")
        out["summary"][metric] = {}
        for s in result:
            pod = s["metric"].get("exported_pod", "?")
            out["series"].append({"metric": metric, "pod": pod, "labels": s["metric"],
                                  "values": [[float(t), v] for t, v in s["values"]]})
            out["summary"][metric][pod] = summarize(s["values"])

    if not out["series"]:
        sys.exit("no GPU metrics for this run: nothing written (is the run inside Prometheus retention?)")
    (run_dir / "gpu_metrics.json").write_text(json.dumps(out, indent=1))
    print(f"[ok] {run_dir / 'gpu_metrics.json'}: {len(out['series'])} series")


if __name__ == "__main__":
    main()
