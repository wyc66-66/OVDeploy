#!/usr/bin/env bash
# Run one stratified B0-cache shard (Phase 1 only). Merge caches on host, then --metrics-only.
set -eo pipefail
cd /mnt/d/ccfa/submission-a

SHARD_ID="${1:?usage: $0 SHARD_ID NUM_SHARDS [DEVICE]}"
NUM_SHARDS="${2:?usage: $0 SHARD_ID NUM_SHARDS [DEVICE]}"
DEVICE="${3:-cuda:0}"
MAX_IMAGES="${4:-1000}"
REPORT="${5:-reports/REPORT_4b_gdino_base_stratified_1k.json}"

echo "=== stratified shard cache: id=$SHARD_ID / $NUM_SHARDS device=$DEVICE ==="
bash scripts/wsl_stratified_glip.sh "$MAX_IMAGES" "$DEVICE" "$REPORT" gdino_base \
  --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" --cache-only

echo "=== shard $SHARD_ID done; when all shards finish, run metrics-only on one machine ==="
echo "  bash scripts/wsl_stratified_glip.sh $MAX_IMAGES $DEVICE $REPORT gdino_base --metrics-only"
