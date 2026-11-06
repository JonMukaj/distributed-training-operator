#!/usr/bin/env python3
"""Offline what-if for the operator's topology solver (no cluster needed).

Reads the archived history entries (results/<RUN>/history.yaml, written by
`make hs-archive-results`), fits the same model the operator fits
(P(n) = p * n * eta(n), eta = 1 / (1 + alpha (n-1))) and answers three questions:

  1. What does the solver predict, and which node count would it pick for a target?
       python3 solver_whatif.py --hash <prefix> --target 12m
  2. What would it have predicted from only some node counts (2-point vs 3-point)?
       python3 solver_whatif.py --hash <prefix> --subset 1,10 --target 17m
  3. How accurate is the model on held-out node counts (leave-one-n-out MAPE)?
       python3 solver_whatif.py --hash <prefix> --loo

Used to (a) choose the `targetTime` of the objective CRs from measured history and
(b) compute the solver-accuracy numbers of thesis chapter 5 without extra GPU time.
The estimator mirrors operator/internal/controller/solver.go (closed form for two
distinct node counts, golden-section search with a data-driven bracket for three or
more; solver never picks fewer than 2 nodes, like binarySearchMinN(2, ...)).

Warm-up runs (directories ending in "-warmup") are skipped: their history entry is
deleted on the cluster too, so they never reach the solver.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import yaml

MAX_ALPHA = 1000.0


def quantity(v):
    s = str(v).strip()
    suf = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
    if s and s[-1] in suf and re.fullmatch(r"[-+0-9.eE]+[numkMG]", s):
        return float(s[:-1]) * suf[s[-1]]
    return float(s)


def duration_seconds(s):
    mult = {"h": 3600, "m": 60, "s": 1}
    parts = re.findall(r"(\d+(?:\.\d+)?)([hms])", s)
    if not parts or "".join(a + b for a, b in parts) != s.strip():
        raise SystemExit(f"cannot parse duration {s!r} (use e.g. 90m, 1h30m, 600s)")
    return sum(float(v) * mult[u] for v, u in parts)


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 60}m{sec % 60:02d}s"


def eta(n, a):
    return 1.0 / (1.0 + a * (n - 1))


def golden(f, a, b, tol=1e-6):
    phi = 0.6180339887498949
    c, d = b - phi * (b - a), a + phi * (b - a)
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return (a + b) / 2


class Model:
    """Operator's estimateModel on a list of history entries."""

    def __init__(self, entries):
        by_n = {}
        for e in entries:
            by_n.setdefault(e["nodes"], []).append(e)
        self.ns = sorted(by_n)
        avg = lambda xs: sum(xs) / len(xs)
        self.avg_p = {n: avg([e["P"] for e in v]) for n, v in by_n.items()}
        n_min = self.ns[0]
        self.p_base = self.avg_p[n_min] / n_min
        self.w = avg([e["W"] for e in by_n[n_min]])
        self.t_prov = avg([e["prov"] for e in entries])
        self.alpha = self._alpha()

    def _alpha(self):
        ns, p = self.ns, self.avg_p
        if len(ns) < 2:
            return 0.0
        if len(ns) == 2:
            n1, n2 = ns
            e2 = (p[n2] / n2) / (p[n1] / n1)
            return max(0.0, (1 - e2) / (e2 * (n2 - 1))) if e2 > 0 else 0.0
        hi, (lo_n, hi_n) = 1.0, (ns[0], ns[-1])
        e_max = (p[hi_n] / hi_n) / (p[lo_n] / lo_n)
        if e_max > 0:
            hi = max(hi, 2 * (1 - e_max) / (e_max * (hi_n - lo_n)))
        hi = min(hi, MAX_ALPHA)
        sse = lambda a: sum((self.p_base * n * eta(n, a) - p[n]) ** 2 for n in ns)
        return golden(sse, 0.0, hi)

    def p(self, n):
        return self.p_base * n * eta(n, self.alpha)

    def t(self, n):
        return self.w / self.p(n) if self.p(n) > 0 else math.inf

    def cost(self, n, ch):
        return ch * n * (self.t(n) + self.t_prov) / 3600.0

    def pick(self, target, max_nodes):
        """binarySearchMinN(2, max_nodes, target)."""
        if self.t(max_nodes) > target:
            return max_nodes
        lo, hi = 2, max_nodes
        while lo < hi:
            mid = (lo + hi) // 2
            if self.t(mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        return lo


def history_spec(path):
    """The spec of the history entry that belongs to the archived run.

    `make archive-results` dumps every entry whose name starts with djh-<CR>-. A single entry is a plain
    object; a CR name that was run more than once gives a List. The newest entry is the archived run
    (the older ones are earlier attempts that should also have been deleted from the cluster)."""
    d = yaml.safe_load(path.read_text()) or {}
    items = d["items"] if isinstance(d.get("items"), list) else [d]
    items = [i for i in items if isinstance(i, dict) and "spec" in i]
    if not items:
        raise ValueError(f"{path}: no history entry")
    items.sort(key=lambda i: (i.get("metadata") or {}).get("creationTimestamp", ""))
    if len(items) > 1:
        print(f"[note] {path.parent.name}/history.yaml holds {len(items)} entries; using the newest "
              f"({items[-1]['metadata'].get('creationTimestamp', '?')}). Delete the older ones from the cluster "
              f"history so the solver does not see them.", file=sys.stderr)
    return items[-1]["spec"]


def history_name(path):
    """metadata.name of the newest history entry in an archive file (identifies the cluster run)."""
    d = yaml.safe_load(path.read_text()) or {}
    items = d["items"] if isinstance(d.get("items"), list) else [d]
    items = [i for i in items if isinstance(i, dict) and "spec" in i]
    items.sort(key=lambda i: (i.get("metadata") or {}).get("creationTimestamp", ""))
    return (items[-1].get("metadata") or {}).get("name", "") if items else ""


def load(results, only):
    entries = []
    for hf in sorted(Path(results).glob("*/history.yaml")):
        run = hf.parent.name
        if run.endswith("-warmup") or (only and run not in only):
            continue
        s = history_spec(hf)
        entries.append({
            "run": run, "hash": s["configHash"], "nodes": int(s["nodes"]),
            "P": quantity(s["throughput"]), "W": quantity(s["totalWork"]),
            "T": quantity(s["trainingSeconds"]), "prov": quantity(s["provisioningSeconds"]),
            "machine": s.get("machineType", ""),
        })
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--hash", help="config-hash prefix to analyse (default: list the hashes and exit)")
    ap.add_argument("--only", nargs="*", help="restrict to these RUN directories")
    ap.add_argument("--subset", help="comma-separated node counts to fit on, e.g. 1,10")
    ap.add_argument("--target", help="targetTime to evaluate, e.g. 12m")
    ap.add_argument("--max-nodes", type=int, default=10, help="maxNodes of the objective CRs (E-series plan: 10)")
    ap.add_argument("--cost", type=float, default=1.00, help="USD per node-hour (n3-L40x1 = 1.00)")
    ap.add_argument("--loo", action="store_true", help="leave-one-node-count-out prediction error")
    a = ap.parse_args()

    entries = load(a.results, a.only)
    hashes = {}
    for e in entries:
        hashes.setdefault(e["hash"], []).append(e)
    if not a.hash:
        for h, es in hashes.items():
            print(f"{h[:12]}  n={sorted({e['nodes'] for e in es})}  runs={[e['run'] for e in es]}")
        return
    match = [h for h in hashes if h.startswith(a.hash)]
    if len(match) != 1:
        sys.exit(f"--hash {a.hash!r} matches {len(match)} config hashes; known: {[h[:12] for h in hashes]}")
    es = hashes[match[0]]
    if a.subset:
        keep = {int(x) for x in a.subset.split(",")}
        es = [e for e in es if e["nodes"] in keep]
    if len({e["nodes"] for e in es}) < 2:
        sys.exit("need history at two or more distinct node counts")

    if a.loo:
        measured = {}
        for e in es:
            measured.setdefault(e["nodes"], []).append(e["T"])
        print(f"{'held-out n':>10} {'fit on':>10} {'alpha':>8} {'predicted':>10} {'measured':>10} {'APE':>7}")
        apes = []
        for n in sorted(measured):
            rest = [e for e in es if e["nodes"] != n]
            if len({e["nodes"] for e in rest}) < 2:
                continue
            m = Model(rest)
            act = sum(measured[n]) / len(measured[n])
            ape = abs(m.t(n) - act) / act * 100
            apes.append(ape)
            print(f"{n:>10} {str(m.ns):>10} {m.alpha:>8.3f} {fmt(m.t(n)):>10} {fmt(act):>10} {ape:>6.1f}%")
        if apes:
            print(f"MAPE over {len(apes)} held-out node counts: {sum(apes) / len(apes):.1f}%")
        else:
            print("leave-one-out needs history at three or more distinct node counts")
        return

    m = Model(es)
    print(f"fit on n={m.ns} ({len(es)} entries): alpha={m.alpha:.3f}  p/node={m.p_base:.2f} samples/s  "
          f"W={m.w:.0f}  avg provisioning={fmt(m.t_prov)}")
    print(f"{'n':>3} {'eta':>6} {'pred train':>11} {'pred cost':>10} {'measured':>10}")
    for n in range(1, a.max_nodes + 1):
        got = [e["T"] for e in es if e["nodes"] == n]
        print(f"{n:>3} {eta(n, m.alpha):>6.2f} {fmt(m.t(n)):>11} {'$%.2f' % m.cost(n, a.cost):>10} "
              f"{fmt(sum(got) / len(got)) if got else '-':>10}")
    if a.target:
        tgt = duration_seconds(a.target)
        n = m.pick(tgt, a.max_nodes)
        note = "" if m.t(n) <= tgt else "  (target unreachable: solver falls back to maxNodes)"
        print(f"targetTime {a.target}: solver picks n={n}, predicted {fmt(m.t(n))}, ${m.cost(n, a.cost):.2f}{note}")


if __name__ == "__main__":
    main()
