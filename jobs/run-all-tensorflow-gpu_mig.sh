#!/usr/bin/env bash
# Wrapper that submits jobs/run-all-tensorflow.sh to gpu_mig with the
# correct *typed* MIG GPU request, so the job actually matches the
# Gres=gpu:a100_3g.20gb:8 that gpu_mig nodes advertise.
#
# Background:
#   jobs/run-all-tensorflow.sh has `#SBATCH --gpus=1` (untyped). On gpu_mig,
#   slurm does NOT auto-match an untyped GPU request to typed MIG slices —
#   the job stays PENDING with REASON=Resources even when slots are free.
#   This wrapper replaces that flag with --gpus=a100_3g.20gb:1 and shortens
#   --time to 45 min so the job is backfill-eligible.
#
# Usage (run from the project root on the Snellius login node):
#   bash jobs/run-all-tensorflow-gpu_mig.sh
#
# Any extra arguments are forwarded to sbatch (placed BEFORE the script),
# so you can override settings ad-hoc, e.g.:
#   bash jobs/run-all-tensorflow-gpu_mig.sh --time=00:30:00 --job-name=lbm-mig-fast

set -euo pipefail

# Refuse to run inside a Slurm allocation. This wrapper is a submitter,
# not a batch script — `sbatch jobs/run-all-tensorflow-gpu_mig.sh` would
# create a default (rome) job that then tries to nest another sbatch from
# the spool dir, which silently fails. Run with `bash` instead.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cat >&2 <<'EOF'
ERROR: jobs/run-all-tensorflow-gpu_mig.sh is a SUBMITTER wrapper.
       You appear to have invoked it via sbatch (SLURM_JOB_ID is set).
       This wrapper must be run with `bash`, not `sbatch`:

           bash jobs/run-all-tensorflow-gpu_mig.sh

       To override sbatch flags, pass them through to bash:

           bash jobs/run-all-tensorflow-gpu_mig.sh --time=00:30:00

EOF
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec sbatch \
    --partition=gpu_mig \
    --gpus=a100_3g.20gb:1 \
    --cpus-per-task=9 \
    --time=00:45:00 \
    "$@" \
    "${SCRIPT_DIR}/run-all-tensorflow.sh"
