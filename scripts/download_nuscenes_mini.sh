#!/usr/bin/env bash
# Download nuScenes v1.0-mini to /mnt/d/data/nuscenes (public metadata URL + blob URLs if reachable)
set -euo pipefail
OUT="${1:-/mnt/d/data/nuscenes}"
mkdir -p "$OUT"
cd "$OUT"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "GET $url -> $out"
  if wget -c --timeout=60 -O "$out" "$url"; then
    return 0
  fi
  rm -f "$out"
  return 1
}

# Metadata (official tutorial URL; usually works without login)
download "https://www.nuscenes.org/data/v1.0-mini.tgz" "v1.0-mini.tgz" || true

# Common mini sensor archives (may require signed URLs on some networks)
for name in \
  "https://www.nuscenes.org/data/nuScenes-mini-sample-data.tgz" \
  "https://www.nuscenes.org/data/nuScenes-mini-sample-data.tar" \
  "https://www.nuscenes.org/data/nuScenes-mini-map-expansion-v1.3.tgz" \
  "https://www.nuscenes.org/data/nuScenes-map-expansion-v1.3.zip"; do
  base="$(basename "$name")"
  download "$name" "$base" || true
done

echo "=== Extract ==="
if [[ -f v1.0-mini.tgz ]]; then
  tar -xzf v1.0-mini.tgz
fi
for f in nuScenes-mini-sample-data.tgz nuScenes-mini-map-expansion-v1.3.tgz; do
  if [[ -f "$f" ]]; then
    tar -xzf "$f"
  fi
done
if [[ -f nuScenes-map-expansion-v1.3.zip ]]; then
  unzip -o nuScenes-map-expansion-v1.3.zip || true
fi

echo "=== Layout ==="
ls -la "$OUT" | head -20
if [[ -d "$OUT/v1.0-mini" ]]; then
  echo "OK metadata"
else
  echo "MISSING v1.0-mini"
  exit 2
fi
if [[ -d "$OUT/samples" ]]; then
  echo "OK samples"
else
  echo "WARN: samples/ missing — CAM_FRONT images need nuScenes-mini-sample-data"
  exit 3
fi
