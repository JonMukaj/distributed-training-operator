#!/usr/bin/env python3
"""Recreate DistributedTrainingHistory entries on a (new) cluster from the archive.

    script/venv/bin/python script/restore_history.py --dry-run          # print what would be created
    script/venv/bin/python script/restore_history.py                    # create them (current kubeconfig)
    script/venv/bin/python script/restore_history.py --only e2-n1 e2-n10

Why: the solver reads history CRs of the same config hash. They live in the cluster, so deleting the
cluster deletes them; the archived copies (results/<RUN>/history.yaml, written by `make
hs-archive-results`) bring them back. The CRDs must be installed first (`make hs-deploy`).

What it does with each archive file:
  * takes the NEWEST entry (a CR name that was run twice archives both);
  * drops server-managed metadata (uid, resourceVersion, generateName, managedFields, ...) that makes
    `kubectl create` fail, and keeps the entry's name, so running this twice is harmless
    (AlreadyExists is reported and skipped);
  * skips a run whose entry is identical to another run's (a duplicate archive under the wrong name);
  * skips runs named *-warmup: a warm-up entry would be averaged into the 1-node baseline;
  * refuses a config-hash group whose runs processed different amounts of work (totalWork), because the
    solver takes W from the smallest node count: pass --allow-mixed-work to override;
  * leaves everything else to you: use --only / --exclude to pick runs, for example to leave out a run
    you consider invalid.

The solver only reads nodes, throughput, totalWork and the provisioning/training seconds, so a restored
entry works even if its transfer fractions came from an old timer. It never deletes anything.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solver_whatif as sw  # noqa: E402

DROP = ("uid", "resourceVersion", "generateName", "generation", "creationTimestamp",
        "managedFields", "selfLink", "ownerReferences", "deletionTimestamp")


def newest_item(path):
    d = yaml.safe_load(path.read_text()) or {}
    items = d["items"] if isinstance(d.get("items"), list) else [d]
    items = [i for i in items if isinstance(i, dict) and "spec" in i]
    if not items:
        return None
    items.sort(key=lambda i: (i.get("metadata") or {}).get("creationTimestamp", ""))
    return items[-1], len(items)


def clean(item):
    meta = {k: v for k, v in item["metadata"].items() if k not in DROP}
    return {"apiVersion": item.get("apiVersion", "training.distributedtraining.io/v1"),
            "kind": "DistributedTrainingHistory", "metadata": meta, "spec": item["spec"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--only", nargs="*", help="restore only these RUN directories")
    ap.add_argument("--exclude", nargs="*", default=[], help="skip these RUN directories")
    ap.add_argument("--allow-mixed-work", action="store_true",
                    help="restore config-hash groups whose runs processed different totalWork")
    ap.add_argument("--dry-run", action="store_true", help="print the cleaned YAML instead of creating it")
    a = ap.parse_args()

    picked = []   # (run, cleaned object)
    for hf in sorted(Path(a.results).glob("*/history.yaml")):
        run = hf.parent.name
        if run.endswith("-warmup"):
            print(f"[skip] {run}: warm-up run")
            continue
        if (a.only and run not in a.only) or run in a.exclude:
            continue
        got = newest_item(hf)
        if not got:
            print(f"[skip] {run}: no history entry in {hf.name}")
            continue
        item, n = got
        if n > 1:
            print(f"[note] {run}: archive holds {n} entries, restoring the newest only")
        picked.append((run, item))
    if not picked:
        sys.exit("nothing to restore")

    seen = {}
    unique = []
    for run, item in picked:
        name = item["metadata"]["name"]
        if name in seen:
            print(f"[skip] {run}: holds the same history entry ({name}) as {seen[name]}; a duplicate archive of one "
                  f"cluster run, probably archived under the wrong RUN name. Delete or fix one of them.")
            continue
        seen[name] = run
        unique.append((run, item))
    picked = unique

    groups = {}
    for run, item in picked:
        groups.setdefault(item["spec"].get("configHash", "?"), []).append((run, item))
    out = []
    for h, members in groups.items():
        work = {round(sw.quantity(i["spec"]["totalWork"])) for _, i in members if "totalWork" in i["spec"]}
        if len(work) > 1 and not a.allow_mixed_work:
            print(f"[skip] config hash {h[:12]}: runs processed different totalWork {sorted(work)} "
                  f"({', '.join(r for r, _ in members)}). Rerun the odd one out, use --exclude, or --allow-mixed-work.")
            continue
        out += members

    objs = [(run, clean(item)) for run, item in out]
    if a.dry_run:
        print(yaml.safe_dump_all([o for _, o in objs], sort_keys=False))
        print(f"[dry-run] {len(objs)} entr{'y' if len(objs) == 1 else 'ies'}: " + ", ".join(r for r, _ in objs))
        return
    created = existing = failed = 0
    for run, obj in objs:
        r = subprocess.run(["kubectl", "create", "-f", "-"], input=yaml.safe_dump(obj), text=True,
                           capture_output=True)
        name = obj["metadata"]["name"]
        if r.returncode == 0:
            created += 1
            print(f"[ok]   {run}: created {name}")
        elif "AlreadyExists" in r.stderr:
            existing += 1
            print(f"[skip] {run}: {name} already exists")
        else:
            failed += 1
            print(f"[FAIL] {run}: {r.stderr.strip()}")
    print(f"created {created}, already present {existing}, failed {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
