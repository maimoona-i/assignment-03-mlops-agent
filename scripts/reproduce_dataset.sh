#!/usr/bin/env bash
# Demonstrate reproducible dataset checkout.   Usage: scripts/reproduce_dataset.sh <git-tag-or-commit>   e.g. v1
set -euo pipefail
REV="${1:?give a git tag/commit, e.g. v1}"
echo "md5 BEFORE: $(md5sum data/dataset.csv 2>/dev/null | cut -d' ' -f1 || echo none)"
git checkout "$REV" -- data/dataset.csv.dvc params.yaml
dvc checkout data/dataset.csv.dvc --force
echo "md5 AFTER ($REV): $(md5sum data/dataset.csv | cut -d' ' -f1)"
grep -E "md5|dataset_version" data/dataset.csv.dvc params.yaml
dvc status data/dataset.csv.dvc
echo "To go back to the latest version:  git checkout HEAD -- data/dataset.csv.dvc params.yaml && dvc checkout data/dataset.csv.dvc --force"
