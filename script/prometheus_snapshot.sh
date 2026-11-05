#!/usr/bin/env bash
# Take a consistent snapshot of the cluster's Prometheus TSDB and copy it to this
# machine, so the whole metric history (with original timestamps) can be served
# by the local stack in example/local-grafana/ after the cluster is gone.
#
#   make prometheus-snapshot [SNAPSHOT_DIR=../results/prometheus-snapshots]
#
# How: Prometheus (kube-prometheus-stack, admin API enabled in our values) creates
# the snapshot via POST /api/v1/admin/tsdb/snapshot. Its image has no shell or tar,
# so an ephemeral busybox container sharing the pod's PID namespace reads the data
# directory through /proc/1/root and streams it out with tar.
#
# Restore: copy the snapshot's block directories (01XXXX…) into
# example/local-grafana/data/prometheus/ and run `make replay-up`.
set -euo pipefail

NS="${MONITORING_NS:-monitoring}"
DEST_ROOT="${SNAPSHOT_DIR:-$(dirname "$0")/../results/prometheus-snapshots}"
PORT="${LOCAL_PORT:-19090}"

POD=$(kubectl -n "$NS" get pod -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].metadata.name}')
[ -n "$POD" ] || { echo "no Prometheus pod in namespace $NS"; exit 1; }
echo "Prometheus pod: $NS/$POD"

# 1. Ask Prometheus for a snapshot (through a short-lived port-forward).
kubectl -n "$NS" port-forward "pod/$POD" "$PORT:9090" >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do curl -fs "http://localhost:$PORT/-/ready" >/dev/null 2>&1 && break; sleep 0.5; done
RESP=$(curl -fs -XPOST "http://localhost:$PORT/api/v1/admin/tsdb/snapshot")
NAME=$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["name"])')
echo "snapshot created in pod: /prometheus/snapshots/$NAME"

# 2. Attach an ephemeral container that can see the Prometheus filesystem.
DBG="snap-$(date +%s)"
kubectl -n "$NS" debug "pod/$POD" --image=busybox:1.36 --target=prometheus --container="$DBG" --quiet -- sleep 1800 >/dev/null
for _ in $(seq 1 60); do
  STATE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath="{.status.ephemeralContainerStatuses[?(@.name==\"$DBG\")].state.running.startedAt}" 2>/dev/null || true)
  [ -n "$STATE" ] && break; sleep 1
done
[ -n "$STATE" ] || { echo "ephemeral container $DBG did not start"; exit 1; }

# 3. Stream the snapshot out.
DEST="$DEST_ROOT/$NAME"
mkdir -p "$DEST"
kubectl -n "$NS" exec "$POD" -c "$DBG" -- tar cf - -C "/proc/1/root/prometheus/snapshots/$NAME" . | tar xf - -C "$DEST"
# the snapshot dir inside the pod is no longer needed
kubectl -n "$NS" exec "$POD" -c "$DBG" -- rm -rf "/proc/1/root/prometheus/snapshots/$NAME" || true

BLOCKS=$(find "$DEST" -maxdepth 1 -mindepth 1 -type d | wc -l)
SIZE=$(du -sh "$DEST" | cut -f1)
echo "saved $BLOCKS block(s), $SIZE → $DEST"
echo "restore: cp -r $DEST/* example/local-grafana/data/prometheus/ && make replay-up"
