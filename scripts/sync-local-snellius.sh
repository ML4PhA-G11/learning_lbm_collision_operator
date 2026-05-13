#!/usr/bin/env bash
# Sync this project from the local workstation to the Snellius login node.
#
# Excludes the local venv, caches, and generated artifacts so they get rebuilt
# remotely instead of shipping ~hundreds of MB over the network. .git is kept
# so you can branch/pull on Snellius — drop the `--include='.git/'` line if
# you would rather not ship history.
#
# Usage:
#   bash scripts/sync-local-snellius.sh              # dry run (preview only)
#   bash scripts/sync-local-snellius.sh --apply      # actually copy
#
# Override the destination with:
#   REMOTE_HOST=snellius REMOTE_DIR='~/path/on/remote' \
#       bash scripts/sync-local-snellius.sh --apply

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-snellius}"
REMOTE_DIR="${REMOTE_DIR:-~/work-my-projects/workspace-master.course.block05-ML4PhA/learning_lbm_collision_operator}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN_FLAG="-n"
if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN_FLAG=""
    echo "[sync] APPLYING changes to $REMOTE_HOST:$REMOTE_DIR"
else
    echo "[sync] DRY RUN — re-run with '--apply' to actually copy."
    echo "[sync] Target: $REMOTE_HOST:$REMOTE_DIR"
fi

# Ensure the remote parent directory exists (no-op if it already does).
PARENT_DIR="$(dirname "$REMOTE_DIR")"
ssh "$REMOTE_HOST" "mkdir -p $PARENT_DIR"

rsync -avhP $DRY_RUN_FLAG \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='artifacts-run-all-tensorflow/' \
    --exclude='example_dataset.npz' \
    --exclude='example_network.keras' \
    --exclude='weights.keras' \
    ./ \
    "$REMOTE_HOST:$REMOTE_DIR/"

echo "[sync] Done."
