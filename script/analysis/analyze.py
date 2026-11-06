#!/usr/bin/env python3
"""Thesis analysis: tables and figures from the archived runs in results/.

    script/venv/bin/python script/analysis/analyze.py            # all of it
    script/venv/bin/python script/analysis/analyze.py --netprobe-gbit 3.1 --out analysis_out
    script/venv/bin/python script/analysis/analyze.py --runlog    # only results/RUNLOG.md

Input  : results/<RUN>/{history.yaml, cr.yaml, transfer_profile.json, all_results.json}
         written by `make hs-archive-results`. Directories ending in "-warmup" are skipped.
         Optional results/<RUN>/notes.txt: free-text anomalies for that run (the only hand-written input).
Output : <out>/tables/*.csv, <out>/tables.md (every table, markdown), <out>/figures/*.{png,pdf}
         or, with --runlog, results/RUNLOG.md (one row per run, generated, do not edit by hand).

Runs are grouped into a *series* by config hash (same model, dataset, batch, accumulation,
epochs, machine type), which is the operator's own grouping. The series name comes from the
`experiment` label of its fixed-topology runs (e1, e2, lora). Anything that needs data that
does not exist yet (for example solver accuracy before any objective run) is skipped with a
note, so the script runs at every stage of the campaign.

Definitions (match PROJECT.md section 5 and METRICS.md):
  wall throughput  P = totalWork / trainingSeconds         (what the solver fits)
  inner throughput   = totalWork / train_runtime           (Trainer's inner loop)
  efficiency       eta(n) = (P(n)/n) / (P(nmin)/nmin),  alpha = (1 - eta) / (eta (n - 1))
                   (alpha is only reported when the series has a 1-node run)
  cost             nodes * $/node-hour * (provisioning + training) / 3600
  bottleneck       compute-bound if compute fraction >= 0.6, else the larger of
                   AllReduce / data-load fraction (same rule as the operator)
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import solver_whatif as sw  # noqa: E402  (same estimator as the operator)

PRICES = {"n3-L40x1": 1.00, "n3-A100x1": 1.35, "c4-highcpu-8": 0.35}
COMPUTE_BOUND = 0.6
# An AllReduce bandwidth above this cannot have crossed the Hyperstack network (25 Gbit/s); it means the
# timer measured kernel enqueue, not completion (found 2026-11-06: 370 GB/s on a ~0.4 GB/s link).
MAX_PLAUSIBLE_MBS = 3125.0

# ---- design tokens (dataviz reference palette, light surface; validated for 3 slots) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua - first three slots pass all-pairs CVD
MARKERS = ["o", "s", "^"]                  # secondary encoding: shape as well as colour


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def go_duration(s):
    if not s:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(h|m|s)", str(s))
    if not parts:
        return None
    return sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in parts)


def jload(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def yload(path):
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def read_note(run_dir):
    """Anomalies from results/<RUN>/notes.txt, one line of text (empty if the file is absent)."""
    try:
        lines = (run_dir / "notes.txt").read_text().splitlines()
    except OSError:
        return ""
    return "; ".join(s.strip() for s in lines if s.strip()).replace("|", "/")


def load_runs(results, include_warmups=False):
    runs = []
    for d in sorted(Path(results).iterdir()):
        if not d.is_dir() or not (d / "history.yaml").exists():
            continue
        if d.name.endswith("-warmup") and not include_warmups:
            continue
        spec = sw.history_spec(d / "history.yaml")
        cr = yload(d / "cr.yaml")
        tp, ar = jload(d / "transfer_profile.json"), jload(d / "all_results.json")
        status = (cr.get("status") or {})
        res = status.get("results") or {}
        resolved = status.get("resolvedTopology") or {}
        objective = (cr.get("spec") or {}).get("objective")
        labels = (cr.get("metadata") or {}).get("labels") or {}
        q = sw.quantity
        n = int(spec.get("nodes", 0))
        W = q(spec["totalWork"]) if "totalWork" in spec else None
        T = q(spec["trainingSeconds"]) if "trainingSeconds" in spec else None
        inner = ar.get("train_runtime")
        price = PRICES.get(spec.get("machineType", ""), None)
        prov = q(spec["provisioningSeconds"]) if "provisioningSeconds" in spec else None
        f = lambda k, src=spec: float(src[k]) if k in src and src[k] not in (None, "") else None
        r = {
            "run": d.name, "hash": spec.get("configHash", ""), "nodes": n,
            "experiment": labels.get("experiment", ""),
            "kind": "objective" if objective else ("repeat" if labels.get("experiment", "").endswith("-rep") else "fixed"),
            "machine": spec.get("machineType", ""),
            "work": W, "train_s": T, "prov_s": prov, "inner_s": inner,
            "P_wall": W / T if W and T else None,
            "P_inner": W / inner if W and inner else None,
            "f_compute": f("computeFraction"), "f_allreduce": f("allReduceFraction"),
            "f_dataload": f("dataloadFraction"),
            "ar_bw_mbs": f("allReduceBandwidthMBs"), "nfs_bw_mbs": f("nfsReadBandwidthMBs"),
            "grad_bytes": spec.get("gradSizeBytes"),
            "hook_coverage": tp.get("hook_coverage"), "nfs_read_bytes": tp.get("nfs_read_bytes"),
            "eval_loss": ar.get("eval_loss"), "perplexity": ar.get("perplexity"),
            "target_s": go_duration(((objective or {}).get("targetTime"))),
            "max_nodes": (objective or {}).get("maxNodes"),
            "pred_n": resolved.get("nodes"), "pred_s": go_duration(resolved.get("estimatedTime")),
            "pred_cost": float(resolved["estimatedCost"]) if resolved.get("estimatedCost") else None,
            "bottleneck_status": res.get("bottleneck"),
            "created": str((cr.get("metadata") or {}).get("creationTimestamp") or ""),
            "note": read_note(d),
            "entry": sw.history_name(d / "history.yaml"),
        }
        r["cost"] = n * price * (prov + T) / 3600.0 if price and prov is not None and T is not None else None
        r["ar_flag"] = ar_invalid_reason(n, tp, r["ar_bw_mbs"])
        r["ar_raw"] = (r["f_allreduce"], r["ar_bw_mbs"])
        r["fix_flag"] = ""
        if (n > 1 and not r["ar_flag"] and not tp.get("compute_excludes_allreduce")
                and r["f_compute"] is not None and r["f_allreduce"] is not None):
            # Profiles from images before v2.6.0 report the whole step window as compute, AllReduce included.
            r["f_compute"] = max(0.0, r["f_compute"] - r["f_allreduce"])
            r["fix_flag"] = "compute fraction corrected (profile predates v2.6.0); the operator's stored class is unreliable"
        if r["ar_flag"]:   # never feed a wrong timer into fractions, classes, alpha scatter or bandwidth plots
            r["f_allreduce"] = r["f_compute"] = r["ar_bw_mbs"] = None
        fr = (r["f_compute"], r["f_allreduce"], r["f_dataload"])
        r["bottleneck"] = classify(*fr) if None not in fr else None
        runs.append(r)
    return runs


def ar_invalid_reason(n, tp, bw_mbs):
    """Why this run's AllReduce fraction / bandwidth / class must not be used, or "" if it may be."""
    if n < 2:
        return ""
    if "allreduce_timing" not in tp:
        return "AllReduce timer predates the CUDA-event fix"
    if tp.get("allreduce_timing_check") == "suspect":
        return "AllReduce timer disagrees with the in-run probe"
    if bw_mbs is not None and bw_mbs > MAX_PLAUSIBLE_MBS:
        return f"AllReduce bandwidth {bw_mbs:.0f} MB/s exceeds the link"
    return ""


def classify(compute, allreduce, dataload):
    if compute >= COMPUTE_BOUND:
        return "compute-bound"
    return "communication-bound" if allreduce >= dataload else "data-bound"


def name_series(runs):
    """Series name per config hash, from the experiment labels of its runs."""
    by_hash = {}
    for r in runs:
        by_hash.setdefault(r["hash"], []).append(r)
    names = {}
    for h, rs in by_hash.items():
        lab = next((r["experiment"] for r in rs if r["kind"] == "fixed" and r["experiment"]), "")
        m = re.match(r"(e\d+|lora)", lab or rs[0]["run"].replace("distributedtraining-llm-gpu", "lora"))
        names[h] = m.group(1) if m else h[:6]
    for r in runs:
        r["series"] = names[r["hash"]]
    # A history entry belongs to exactly one run. Two directories holding the same entry mean one archive was
    # taken under the wrong RUN name (make hs-archive-results CR=<x> RUN=<y> with the wrong CR).
    first = {}
    for r in runs:
        r["dup_flag"] = ""
        if r.get("entry"):
            if r["entry"] in first:
                r["dup_flag"] = f"same history entry as {first[r['entry']]}: duplicate archive under the wrong RUN name"
                print(f"[warn] {r['run']} duplicates {first[r['entry']]} (history entry {r['entry']}); fix or delete one")
            else:
                first[r["entry"]] = r["run"]
    # Fixed work is the premise of the scaling comparison: flag series whose runs processed different totals.
    for sname in {r["series"] for r in runs}:
        rs = [r for r in runs if r["series"] == sname]
        works = sorted({round(r["work"]) for r in rs if r["work"]})
        for r in rs:
            r["work_flag"] = f"series mixes totalWork values {works}" if len(works) > 1 else ""
        if len(works) > 1:
            print(f"[warn] series {sname}: runs processed different amounts of work {works}: "
                  + ", ".join(f"{r['run']}={round(r['work'])}" for r in rs if r["work"]))
    return runs


def short(r):
    tag = {"objective": " (solver)", "repeat": " (repeat)"}.get(r["kind"], "")
    return f"{r['series']} n={r['nodes']}{tag}"


# --------------------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------------------
def add_scaling(runs):
    """eta and alpha per run, wall-clock and inner-loop, against the series' 1-node baseline."""
    for r in runs:
        r.update(eta_wall=None, alpha_wall=None, eta_inner=None, alpha_inner=None,
                 speedup_wall=None, speedup_inner=None)
    for s in {r["series"] for r in runs}:
        rs = [r for r in runs if r["series"] == s]
        base = [r for r in rs if r["nodes"] == 1]
        if not base:
            continue
        for tag, key in (("wall", "P_wall"), ("inner", "P_inner")):
            vals = [r[key] for r in base if r[key]]
            if not vals:
                continue
            p1 = sum(vals) / len(vals)
            for r in rs:
                if not r[key]:
                    continue
                r["speedup_" + tag] = r[key] / p1
                eta = r[key] / r["nodes"] / p1
                r["eta_" + tag] = eta
                if r["nodes"] > 1 and eta > 0:
                    r["alpha_" + tag] = max(0.0, (1 - eta) / (eta * (r["nodes"] - 1)))
    return runs


def rank(a):
    order = np.argsort(a)
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):               # average ties
        idx = np.where(a == v)[0]
        ranks[idx] = ranks[idx].mean()
    return ranks


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    pear = float(np.corrcoef(x, y)[0, 1])
    rx, ry = rank(x), rank(y)
    spear = float(np.corrcoef(rx, ry)[0, 1]) if np.ptp(rx) and np.ptp(ry) else float("nan")
    return {"n": len(x), "pearson": pear, "spearman": spear}


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def minutes(s):
    return "-" if s is None else f"{int(s) // 60}m{int(round(s)) % 60:02d}s"


class Tables:
    def __init__(self, out):
        self.dir = out / "tables"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.md = []

    def add(self, name, title, header, rows, note="", foot=""):
        if not rows:
            self.md.append(f"## {title}\n\n_No data yet: {note or 'nothing to tabulate'}._\n")
            print(f"[skip] {name}: no data")
            return
        with open(self.dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        line = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
        line += "".join("| " + " | ".join(str(c) for c in row) + " |\n" for row in rows)
        self.md.append(f"## {title}\n\n{line}\n" + (f"{foot}\n" if foot else ""))
        print(f"[ok]   {name}: {len(rows)} rows")

    def write(self, path):
        path.write_text("# Analysis tables\n\nGenerated by `script/analysis/analyze.py`.\n\n" + "\n".join(self.md))


def order(runs):
    return sorted(runs, key=lambda r: (r["series"], r["nodes"], r["kind"] != "fixed", r["run"]))


def build_tables(runs, results, T):
    rs = order(runs)

    T.add("runs", "All runs",
          ["run", "series", "kind", "n", "provisioning", "training", "inner loop", "samples/s (wall)",
           "compute", "allreduce", "dataload", "class", "eval loss", "ppl", "cost $"],
          [[r["run"], r["series"], r["kind"], r["nodes"], minutes(r["prov_s"]), minutes(r["train_s"]),
            minutes(r["inner_s"]), fmt(r["P_wall"], 1), fmt(r["f_compute"], 3), fmt(r["f_allreduce"], 3),
            fmt(r["f_dataload"], 3), r["bottleneck"] or "-", fmt(r["eval_loss"], 3), fmt(r["perplexity"], 2),
            fmt(r["cost"], 2)] for r in rs])

    T.add("scaling", "Scaling per run (against the series' 1-node run)",
          ["run", "series", "n", "speedup wall", "eta wall", "alpha wall", "speedup inner", "eta inner", "alpha inner"],
          [[r["run"], r["series"], r["nodes"], fmt(r["speedup_wall"]), fmt(r["eta_wall"]), fmt(r["alpha_wall"], 3),
            fmt(r["speedup_inner"]), fmt(r["eta_inner"]), fmt(r["alpha_inner"], 3)] for r in rs if r["eta_wall"]],
          note="needs a 1-node run in the series",
          foot="Wall alpha is what the operator fits (training clock from the Running condition). "
               "Inner alpha uses the Trainer's `train_runtime`; the gap is the serial remainder "
               "(model load, eval, save).")

    T.add("bottleneck", "Bottleneck class per run",
          ["run", "n", "compute", "allreduce", "dataload", "class (recomputed)", "class (operator status)", "agree"],
          [[r["run"], r["nodes"], fmt(r["f_compute"], 3), fmt(r["f_allreduce"], 3), fmt(r["f_dataload"], 3),
            r["bottleneck"] or "-", r["bottleneck_status"] or "-",
            "-" if not (r["bottleneck"] and r["bottleneck_status"]) else ("yes" if r["bottleneck"] == r["bottleneck_status"] else "NO")]
           for r in rs if r["bottleneck"]])

    T.add("instrumentation", "Instrumentation checks",
          ["run", "n", "hook_coverage (expect 1.0)", "grad bytes", "AllReduce MB/s", "NFS read MB/s", "NFS read bytes"],
          [[r["run"], r["nodes"], fmt(r["hook_coverage"], 3), fmt(r["grad_bytes"], 0), fmt(r["ar_bw_mbs"], 1),
            fmt(r["nfs_bw_mbs"], 2), fmt(r["nfs_read_bytes"], 0)] for r in rs])

    bad = [r for r in rs if r["ar_flag"]]
    if bad:
        print(f"[warn] {len(bad)} run(s) excluded from every AllReduce-based table and figure: "
              + ", ".join(r["run"] for r in bad))
    T.add("excluded_allreduce", "Runs whose AllReduce measurement was rejected",
          ["run", "n", "reason", "reported AllReduce fraction", "reported MB/s"],
          [[r["run"], r["nodes"], r["ar_flag"], fmt(r["ar_raw"][0], 4), fmt(r["ar_raw"][1], 1)] for r in bad],
          note="none: every multi-node run has a verified AllReduce timer",
          foot="Throughput, wall-clock alpha, cost and the 1-node runs are unaffected. These runs are left out of "
               "the time breakdown, the alpha scatter, the bandwidth figure and the bottleneck table because the "
               "AllReduce fraction (and with it the compute fraction and class) is wrong; rerun them with the "
               "fixed image.")

    # alpha vs AllReduce fraction
    pts = [r for r in rs if r["nodes"] > 1 and r["f_allreduce"] is not None]
    rows = []
    for tag in ("wall", "inner"):
        for scope, sel in (("all series", pts), *[(f"series {s}", [r for r in pts if r["series"] == s])
                                                  for s in sorted({r["series"] for r in pts})]):
            sel = [r for r in sel if r["alpha_" + tag] is not None]
            c = correlation([r["f_allreduce"] for r in sel], [r["alpha_" + tag] for r in sel])
            if c:
                rows.append([f"alpha {tag}", scope, c["n"], fmt(c["pearson"], 3), fmt(c["spearman"], 3)])
    T.add("alpha_correlation", "Alpha vs measured AllReduce fraction",
          ["alpha", "scope", "points", "Pearson r", "Spearman rho"], rows,
          note="needs at least three multi-node runs with a 1-node baseline",
          foot="With so few points quote the scatter and Spearman rho, not a significance claim.")

    # solver accuracy: live predictions
    obj = [r for r in rs if r["kind"] == "objective" and r["pred_s"] and r["train_s"]]
    acc = []
    for r in obj:
        err = (r["pred_s"] - r["train_s"]) / r["train_s"] * 100
        met = "-" if not r["target_s"] else ("yes" if r["train_s"] <= r["target_s"] else "NO")
        acc.append([r["run"], minutes(r["target_s"]), r["pred_n"], r["nodes"], minutes(r["pred_s"]),
                    minutes(r["train_s"]), f"{err:+.1f}%", f"{abs(err):.1f}%", met,
                    fmt(r["pred_cost"], 2), fmt(r["cost"], 2)])
    T.add("solver_live", "Solver: live predictions vs actual",
          ["run", "target", "predicted n", "actual n", "predicted training", "actual training", "signed error",
           "abs error", "target met", "predicted cost $", "actual cost $ (incl. provisioning)"], acc,
          note="needs objective runs whose archived cr.yaml carries status.resolvedTopology")
    if acc:
        apes = [abs((r["pred_s"] - r["train_s"]) / r["train_s"]) * 100 for r in obj]
        T.md[-1] += f"MAPE over {len(apes)} live predictions: {sum(apes) / len(apes):.1f}%.\n"

    # leave-one-node-count-out (same estimator as the operator)
    loo, entries = [], sw.load(results, None)
    for h in sorted({e["hash"] for e in entries}):
        es = [e for e in entries if e["hash"] == h]
        name = next((r["series"] for r in runs if r["hash"] == h), h[:6])
        ns = sorted({e["nodes"] for e in es})
        for n in ns:
            rest = [e for e in es if e["nodes"] != n]
            if len({e["nodes"] for e in rest}) < 2:
                continue
            m = sw.Model(rest)
            act = sum(e["T"] for e in es if e["nodes"] == n) / sum(1 for e in es if e["nodes"] == n)
            loo.append([name, n, str(m.ns), fmt(m.alpha, 3), minutes(m.t(n)), minutes(act),
                        f"{(m.t(n) - act) / act * 100:+.1f}%", abs(m.t(n) - act) / act * 100])
    if loo:
        for name in sorted({row[0] for row in loo}):
            sel = [row[7] for row in loo if row[0] == name]
            T.md.append(f"_Series {name}: leave-one-out MAPE {sum(sel) / len(sel):.1f}% over {len(sel)} held-out node counts._\n")
    T.add("solver_loo", "Solver: leave-one-node-count-out on archived history",
          ["series", "held-out n", "fit on", "alpha", "predicted", "measured", "signed error", "abs error %"],
          [row[:7] + [fmt(row[7], 1)] for row in loo],
          note="needs three or more distinct node counts in a series")

    # cost vs naive maximum
    cost_rows = []
    for s in sorted({r["series"] for r in rs}):
        fixed = [r for r in rs if r["series"] == s and r["kind"] == "fixed" and r["cost"]]
        if not fixed:
            continue
        for r in [x for x in rs if x["series"] == s and x["kind"] == "objective" and x["cost"]]:
            # naive = provisioning the maximum allowed nodes: the fixed run at objective.maxNodes if it exists,
            # otherwise the series' largest fixed run (and the row says so).
            at_max = [f for f in fixed if f["nodes"] == r["max_nodes"]]
            naive = at_max[0] if at_max else max(fixed, key=lambda f: f["nodes"])
            if r["nodes"] >= naive["nodes"]:
                # same configuration as the naive run: any cost difference is run-to-run noise, not a saving
                saving, pct = "-", "-"
                naive_label = f"{naive['run']}{'' if at_max else ' (largest measured, not maxNodes)'}; same n, no saving possible"
            else:
                saving = fmt(naive["cost"] - r["cost"], 2)
                pct = f"{(naive['cost'] - r['cost']) / naive['cost'] * 100:.0f}%"
                naive_label = naive["run"] + ("" if at_max else " (largest measured, not maxNodes)")
            cost_rows.append([r["run"], r["nodes"], fmt(r["cost"], 2), naive_label, naive["nodes"],
                              fmt(naive["cost"], 2), saving, pct, minutes(r["train_s"]), minutes(naive["train_s"])])
    T.add("cost_vs_naive", "Cost against the naive maximum-node run",
          ["objective run", "n", "cost $", "naive run", "naive n", "naive cost $", "saving $", "saving %",
           "training", "naive training"], cost_rows,
          note="needs objective runs plus the series' largest fixed run (e2-n10)",
          foot="Zero saving is correct when the target needs every node.")

    # variance: same series and node count
    var = []
    for s in sorted({r["series"] for r in rs}):
        for n in sorted({r["nodes"] for r in rs if r["series"] == s}):
            g = [r for r in rs if r["series"] == s and r["nodes"] == n and r["train_s"]]
            if len(g) < 2:
                continue
            t = [r["train_s"] for r in g]
            ar = [r["f_allreduce"] for r in g if r["f_allreduce"] is not None]
            var.append([s, n, len(g), ", ".join(r["run"] for r in g), minutes(sum(t) / len(t)),
                        f"{(max(t) - min(t)) / (sum(t) / len(t)) * 100:.1f}%",
                        "-" if len(ar) < 2 else f"{max(ar) - min(ar):.3f}"])
    T.add("variance", "Run-to-run variance (same series and node count)",
          ["series", "n", "runs", "which", "mean training", "spread (max-min)/mean", "allreduce fraction spread"], var,
          note="needs two runs of the same configuration (e2-n2 and e2-n2-rep)")

    # lifecycle
    life = []
    pv = [r for r in rs if r["prov_s"]]
    for label, sel in [("all runs", pv)] + [(f"n={n}", [r for r in pv if r["nodes"] == n]) for n in sorted({r["nodes"] for r in pv})]:
        v = sorted(r["prov_s"] for r in sel)
        if v:
            life.append([label, len(v), minutes(float(np.median(v))), minutes(v[0]), minutes(v[-1])])
    T.add("provisioning", "Provisioning time (pool request to GPU allocatable)",
          ["group", "runs", "median", "min", "max"], life,
          note="no runs with provisioning times",
          foot="Teardown and collection times are not in the archive; read them from the phase-transition "
               "metrics or the CR status timestamps.")


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 100,
    })


def save(fig, out, name):
    d = out / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[ok]   figure {name}")


FIXED_SLOT = {"e1": 0, "e2": 1, "lora": 2}   # colour follows the series, whatever else is in the chart


def series_style(series):
    free = [i for i in range(3) if i not in FIXED_SLOT.values()]
    style = {}
    for s in sorted(series):
        i = FIXED_SLOT.get(s, free.pop(0) if free else 2)
        style[s] = (SLOT[i], MARKERS[i])
    return style


def fig_breakdown(runs, out):
    import matplotlib.pyplot as plt
    rs = [r for r in order(runs) if None not in (r["f_compute"], r["f_allreduce"], r["f_dataload"])]
    if not rs:
        return print("[skip] fig_time_breakdown: no transfer fractions")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(rs) + 2), 4.2))
    x = np.arange(len(rs))
    bottom = np.zeros(len(rs))
    parts = [("compute", "f_compute", SLOT[0]), ("AllReduce", "f_allreduce", SLOT[1]), ("data loading", "f_dataload", SLOT[2])]
    for label, key, col in parts:
        v = np.array([r[key] for r in rs])
        ax.bar(x, v, 0.62, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=1.5, label=label)
        for xi, vi, bi in zip(x, v, bottom):
            if vi >= 0.07:
                ax.text(xi, bi + vi / 2, f"{vi * 100:.0f}%", ha="center", va="center", fontsize=8,
                        color="#ffffff" if col != SLOT[2] else INK)
        bottom += v
    ax.set_xticks(x, [short(r).replace(" (", "\n(") for r in rs], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of step time")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.text(0.0, 1.12, "Where each run's step time goes", fontsize=11, ha="left", transform=ax.transAxes)
    save(fig, out, "fig_time_breakdown")


def fig_scaling(runs, out):
    import matplotlib.pyplot as plt
    have = [r for r in runs if r["speedup_wall"] is not None]
    if not have:
        return print("[skip] fig_scaling: no series has a 1-node run")
    st = series_style({r["series"] for r in have})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    nmax = max(r["nodes"] for r in have)
    for ax, key, title in ((axes[0], "speedup_wall", "Wall clock (what the solver fits)"),
                           (axes[1], "speedup_inner", "Inner loop (Trainer train_runtime)")):
        ax.plot([1, nmax], [1, nmax], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
        ax.text(nmax, nmax, " ideal", color=MUTED, fontsize=8, va="center")
        for s, (col, mk) in st.items():
            pts = {}
            for r in have:
                if r["series"] == s and r[key] is not None:
                    pts.setdefault(r["nodes"], []).append(r[key])
            if not pts:
                continue
            xs = sorted(pts)
            ys = [sum(pts[n]) / len(pts[n]) for n in xs]
            ax.plot(xs, ys, color=col, marker=mk, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=s, zorder=3)
            ax.text(xs[-1], ys[-1], f"  {s}", color=INK2, fontsize=9, va="center")
        ax.set_xticks(sorted({r["nodes"] for r in have}))
        ax.set_xlabel("nodes")
        ax.set_title(title)
        ax.set_xlim(0.7, nmax + 0.9)
    axes[0].set_ylabel("speedup over 1 node")
    axes[0].legend(loc="upper left", fontsize=9, title="series", title_fontsize=8)
    save(fig, out, "fig_scaling")


def fig_alpha(runs, out):
    import matplotlib.pyplot as plt
    pts = [r for r in runs if r["nodes"] > 1 and r["f_allreduce"] is not None and r["alpha_wall"] is not None]
    if len(pts) < 2:
        return print("[skip] fig_alpha_vs_allreduce: need at least two multi-node runs with a 1-node baseline")
    st = series_style({r["series"] for r in pts})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=True)
    for ax, tag, title in ((axes[0], "wall", "alpha from wall clock"), (axes[1], "inner", "alpha from inner loop")):
        sel = [r for r in pts if r["alpha_" + tag] is not None]
        for s, (col, mk) in st.items():
            g = [r for r in sel if r["series"] == s]
            ax.scatter([r["f_allreduce"] for r in g], [r["alpha_" + tag] for r in g], s=70, color=col, marker=mk,
                       edgecolor=SURFACE, linewidth=2, label=s, zorder=3)
            for r in g:
                ax.annotate(f"n={r['nodes']}", (r["f_allreduce"], r["alpha_" + tag]), xytext=(7, -3),
                            textcoords="offset points", fontsize=8, color=INK2)
        c = correlation([r["f_allreduce"] for r in sel], [r["alpha_" + tag] for r in sel])
        if c:
            ax.text(0.03, 0.97, f"Spearman rho = {c['spearman']:.2f}   Pearson r = {c['pearson']:.2f}   points = {c['n']}",
                    transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        ax.set_xlabel("measured AllReduce fraction of step time")
        ax.set_title(title, pad=10)
        ymax = max([r["alpha_" + tag] for r in sel] + [0.1])
        xmax = max(r["f_allreduce"] for r in sel)
        ax.set_ylim(0, ymax * 1.45)
        ax.set_xlim(0, xmax * 1.2)
    axes[0].set_ylabel("alpha (scaling overhead)")
    axes[0].legend(loc="lower right", fontsize=9)
    save(fig, out, "fig_alpha_vs_allreduce")


def fig_bandwidth(runs, out, netprobe_gbit):
    import matplotlib.pyplot as plt
    rs = [r for r in order(runs) if r["nodes"] > 1 and r["ar_bw_mbs"] and r["f_allreduce"] and r["f_allreduce"] > 0.02]
    if not rs:
        return print("[skip] fig_allreduce_bandwidth: no multi-node run with a non-trivial AllReduce share")
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(rs) + 2), 3.8))
    x = np.arange(len(rs))
    ax.bar(x, [r["ar_bw_mbs"] for r in rs], 0.55, color=SLOT[0], edgecolor=SURFACE, linewidth=1.5)
    for xi, r in zip(x, rs):
        ax.text(xi, r["ar_bw_mbs"], f"{r['ar_bw_mbs']:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    if netprobe_gbit:
        mbs = netprobe_gbit * 1000 / 8
        ax.axhline(mbs, color=SLOT[1], linestyle=(0, (4, 3)), linewidth=1.5)
        ax.text(len(rs) - 0.5, mbs, f"iperf3 link rate {netprobe_gbit:g} Gbit/s = {mbs:.0f} MB/s", color=INK2,
                fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x, [short(r).replace(" (", "\n(") for r in rs], fontsize=8)
    top = max([r["ar_bw_mbs"] for r in rs] + ([netprobe_gbit * 125] if netprobe_gbit else []))
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("effective AllReduce bandwidth (MB/s)")
    ax.set_title("AllReduce throughput actually achieved")
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_allreduce_bandwidth")


def fig_solver(runs, results, out):
    import matplotlib.pyplot as plt
    obj = [r for r in order(runs) if r["kind"] == "objective" and r["pred_s"] and r["train_s"]]
    entries = sw.load(results, None)
    loo = []
    for h in sorted({e["hash"] for e in entries}):
        es = [e for e in entries if e["hash"] == h]
        for n in sorted({e["nodes"] for e in es}):
            rest = [e for e in es if e["nodes"] != n]
            if len({e["nodes"] for e in rest}) >= 2:
                m = sw.Model(rest)
                nm = next((r["series"] for r in runs if r["hash"] == h), h[:4])
                loo.append((f"{nm} n={n}", m.t(n), np.mean([e["T"] for e in es if e["nodes"] == n])))
    if not obj and not loo:
        return print("[skip] fig_solver_accuracy: no objective runs and not enough node counts for leave-one-out")
    panels = (1 if obj else 0) + (1 if loo else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels + 1, 3.8), squeeze=False)
    axes = list(axes[0])
    if obj:
        ax = axes.pop(0)
        x = np.arange(len(obj))
        ax.bar(x - 0.19, [r["pred_s"] / 60 for r in obj], 0.36, color=SLOT[0], edgecolor=SURFACE, linewidth=1.5, label="predicted")
        ax.bar(x + 0.19, [r["train_s"] / 60 for r in obj], 0.36, color=SLOT[1], edgecolor=SURFACE, linewidth=1.5, label="actual")
        for xi, r in zip(x, obj):
            if r["target_s"]:
                ax.plot([xi - 0.4, xi + 0.4], [r["target_s"] / 60] * 2, color=INK, linewidth=1.5, linestyle=(0, (3, 2)),
                        label="targetTime" if xi == 0 else None)
        ax.set_xticks(x, [f"{r['run']}\nsolver n={r['nodes']}" for r in obj], fontsize=7)
        ax.set_ylabel("training time (min)")
        ax.set_title("Live predictions")
        ax.set_ylim(0, max(max(r["pred_s"], r["train_s"], r["target_s"] or 0) for r in obj) / 60 * 1.3)
        ax.legend(fontsize=9, loc="upper right", ncol=3)
        ax.grid(axis="x", visible=False)
    if loo:
        ax = axes.pop(0)
        x = np.arange(len(loo))
        ax.bar(x - 0.19, [p / 60 for _, p, _ in loo], 0.36, color=SLOT[0], edgecolor=SURFACE, linewidth=1.5, label="predicted (n held out)")
        ax.bar(x + 0.19, [a / 60 for _, _, a in loo], 0.36, color=SLOT[1], edgecolor=SURFACE, linewidth=1.5, label="measured")
        ax.set_xticks(x, [lab.replace(" ", "\n") for lab, _, _ in loo], fontsize=8)
        ax.set_ylabel("training time (min)")
        ax.set_title("Leave-one-node-count-out")
        ax.set_ylim(0, max(max(p, a) for _, p, a in loo) / 60 * 1.3)
        ax.legend(fontsize=9, loc="upper right", ncol=2)
        ax.grid(axis="x", visible=False)
    save(fig, out, "fig_solver_accuracy")


def fig_cost(runs, out):
    import matplotlib.pyplot as plt
    series = [s for s in sorted({r["series"] for r in runs})
              if any(r["kind"] == "objective" for r in runs if r["series"] == s)]
    if not series:
        return print("[skip] fig_cost: no objective runs")
    for s in series:
        rs = [r for r in order(runs) if r["series"] == s and r["cost"]]
        naive = max((r for r in rs if r["kind"] == "fixed"), key=lambda r: r["nodes"], default=None)
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(rs) + 2), 3.8))
        x = np.arange(len(rs))
        cols = [SLOT[1] if r is naive else SLOT[0] if r["kind"] == "objective" else MUTED for r in rs]
        ax.bar(x, [r["cost"] for r in rs], 0.55, color=cols, edgecolor=SURFACE, linewidth=1.5)
        for xi, r in zip(x, rs):
            ax.text(xi, r["cost"], f"${r['cost']:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [short(r).replace(" (", "\n(") for r in rs], fontsize=8)
        ax.set_ylabel("actual cost, incl. provisioning (USD)")
        ax.set_title(f"Cost per run, series {s}")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=SLOT[1], label="naive maximum"), Patch(color=SLOT[0], label="solver-chosen"),
                           Patch(color=MUTED, label="other fixed runs")], fontsize=9, loc="upper left")
        ax.set_ylim(0, max(r["cost"] for r in rs) * 1.2)
        ax.grid(axis="x", visible=False)
        save(fig, out, f"fig_cost_{s}")


def fig_provisioning(runs, out):
    import matplotlib.pyplot as plt
    pv = [r for r in runs if r["prov_s"]]
    if not pv:
        return print("[skip] fig_provisioning: none")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rng = np.random.default_rng(0)
    ns = sorted({r["nodes"] for r in pv})
    for i, n in enumerate(ns):
        v = np.array([r["prov_s"] / 60 for r in pv if r["nodes"] == n])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=55, color=SLOT[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.3, np.median(v), f"median {np.median(v):.1f}", fontsize=8, color=INK2, va="center")
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("nodes in the job")
    ax.set_ylabel("provisioning time (min)")
    ax.set_title("Provisioning time (pool request to GPU allocatable)")
    ax.set_ylim(0, max(r["prov_s"] for r in pv) / 60 * 1.2)
    ax.set_xlim(-0.5, len(ns) - 0.1)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_provisioning")


# --------------------------------------------------------------------------------------
RUNLOG_HEADER = ["date", "run", "series", "kind", "n", "provisioning", "training", "inner loop",
                 "samples/s (wall)", "alpha (wall)", "bottleneck", "eval loss", "cost $", "anomalies"]


def write_runlog(runs, path):
    """results/RUNLOG.md: one row per archived run, oldest first. Generated from the archives;
    the only hand-written input is results/<RUN>/notes.txt (the anomalies column)."""
    rows = []
    for r in sorted(runs, key=lambda r: (r["created"], r["run"])):
        rows.append([
            r["created"][:10] or "-", r["run"], r["series"], r["kind"], r["nodes"],
            minutes(r["prov_s"]), minutes(r["train_s"]), minutes(r["inner_s"]),
            fmt(r["P_wall"], 1), fmt(r["alpha_wall"], 3),
            r["bottleneck"] or ("timer invalid" if r["ar_flag"] else "-"),
            fmt(r["eval_loss"], 3), fmt(r["cost"], 2),
            "; ".join(x for x in (r["note"], ("AUTO: " + r["ar_flag"]) if r["ar_flag"] else "",
                                  ("AUTO: " + r["work_flag"]) if r.get("work_flag") else "",
                                  ("AUTO: " + r["fix_flag"]) if r.get("fix_flag") else "",
                                  ("AUTO: " + r["dup_flag"]) if r.get("dup_flag") else "") if x),
        ])
    lines = [
        "# Run log",
        "",
        "Generated by `script/analysis/analyze.py --runlog` from `results/<RUN>/`. Do not edit by hand:",
        "put anomalies in `results/<RUN>/notes.txt` and regenerate.",
        "",
        "| " + " | ".join(RUNLOG_HEADER) + " |",
        "|" + "---|" * len(RUNLOG_HEADER),
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n")
    print(f"[ok]   {path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE.parent.parent / "results"))
    ap.add_argument("--out", default=str(HERE.parent.parent / "analysis_out"))
    ap.add_argument("--netprobe-gbit", type=float, help="measured iperf3 rate; drawn on the bandwidth figure")
    ap.add_argument("--include-warmups", action="store_true", help="include *-warmup runs (off by default)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--runlog", action="store_true",
                    help="write <results>/RUNLOG.md only (no tables or figures)")
    a = ap.parse_args()

    out = Path(a.out)
    runs = load_runs(a.results, a.include_warmups)
    if not runs:
        sys.exit(f"no run directories with history.yaml under {a.results}")
    runs = add_scaling(name_series(runs))
    print(f"{len(runs)} runs, series: {sorted({r['series'] for r in runs})}")
    if a.runlog:
        write_runlog(runs, Path(a.results) / "RUNLOG.md")
        return

    out.mkdir(parents=True, exist_ok=True)
    T = Tables(out)
    build_tables(runs, a.results, T)
    T.write(out / "tables.md")
    if not a.no_figures:
        style()
        fig_breakdown(runs, out)
        fig_scaling(runs, out)
        fig_alpha(runs, out)
        fig_bandwidth(runs, out, a.netprobe_gbit)
        fig_solver(runs, a.results, out)
        fig_cost(runs, out)
        fig_provisioning(runs, out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
