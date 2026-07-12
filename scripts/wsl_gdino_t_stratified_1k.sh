#!/usr/bin/env bash
# GDINO-T stratified held-out 1k full run (supplementary; replaces n=100 report).
set -eo pipefail
cd /mnt/d/ccfa/submission-a
bash scripts/wsl_stratified_glip.sh 1000 cuda:0 \
  reports/REPORT_4b_gdino_stratified_1k.json glip
