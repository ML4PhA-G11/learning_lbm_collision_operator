#!/usr/bin/env bash
#SBATCH --job-name=lbm-pt
#SBATCH --partition=gpu_mig
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#
# Submit run-all-pytorch.py under Slurm. Default: gpu_mig with the typed
# MIG slice request (a100_3g.20gb) so the job is immediately schedulable
# on gpu_mig (untyped --gpus=1 sits as Reason=Resources there).
#
# Submit from the project root:
#   sbatch jobs/run-all-pytorch.sh
#
# Override the partition / GPU type from the command line, e.g.:
#   sbatch -p gpu_a100 --gpus=1 --cpus-per-task=18 jobs/run-all-pytorch.sh
#   sbatch -p gpu_h100 --gpus=1 --cpus-per-task=16 jobs/run-all-pytorch.sh
#   sbatch -p rome     --gpus=0 --cpus-per-task=16 jobs/run-all-pytorch.sh
# CLI flags override #SBATCH directives in the header.
#
# Laptops without Slurm: `uv run python run-all-pytorch.py` works directly.
# torch's PyPI wheel bundles its own CUDA runtime, so there is no GPU-only
# install step needed at job time.

set -euo pipefail

############################################
# 0. timing — capture the wall clock at job start
############################################
JOB_START_EPOCH=$(date +%s)
JOB_START_HUMAN=$(date -Is)
echo "[job] Started at ${JOB_START_HUMAN}"
echo "[job] SLURM_JOB_ID=${SLURM_JOB_ID:-<not-in-slurm>} on $(hostname -s)"
echo "[job] Partition: ${SLURM_JOB_PARTITION:-<n/a>}  GPUs: ${SLURM_JOB_GPUS:-<none>}"

############################################
# 1. project root + log dir
############################################
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p jobs/logs

echo "[job] Project root: ${PROJECT_ROOT}"

############################################
# 2. environment — same modules as the setup script
############################################
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    fi
fi

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

export PATH="$HOME/.local/bin:$PATH"  # for uv

echo "[job] Loaded modules:"
module list 2>&1 | sed 's/^/         /'

############################################
# 3. CPU/threading hints
############################################
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NCPU}"
export MKL_NUM_THREADS="${NCPU}"
export NUMBA_NUM_THREADS="${NCPU}"
# Torch respects torch.set_num_threads() too — we leave that as-is since
# OMP_NUM_THREADS already governs intra-op parallelism for the CPU path.
echo "[job] CPUs allocated: ${NCPU}"

############################################
# 4. GPU detection (informational; torch wheel already has CUDA bundled)
############################################
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    echo "[job] GPU(s) visible to the job:"
    nvidia-smi -L | sed 's/^/         /'
else
    echo "[job] No GPU detected — running on CPU."
fi

# Ensure torch is in the venv. Idempotent — only installs the first time
# this script runs on a fresh .venv.
if ! uv pip show torch >/dev/null 2>&1; then
    echo "[job] torch not in .venv — installing (PyPI wheel bundles CUDA) ..."
    uv pip install --quiet torch
fi

############################################
# 5. run training + simulation
############################################
echo "[job] Launching run-all-pytorch.py ..."
set +e
uv run python -u run-all-pytorch.py
RUN_RC=$?
set -e

############################################
# 6. timing — print wall clock at job end
############################################
JOB_END_EPOCH=$(date +%s)
JOB_END_HUMAN=$(date -Is)
ELAPSED_SECONDS=$(( JOB_END_EPOCH - JOB_START_EPOCH ))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $(( ELAPSED_SECONDS / 3600 )) \
    $(( (ELAPSED_SECONDS % 3600) / 60 )) \
    $(( ELAPSED_SECONDS % 60 )))

echo "[job] Finished at ${JOB_END_HUMAN}"
echo "[job] Exit code: ${RUN_RC}"
echo "[job] Total wall time: ${ELAPSED_SECONDS}s (${ELAPSED_HMS})"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sacct >/dev/null 2>&1; then
    echo "[job] sacct view (Elapsed / MaxRSS / State):"
    sacct -j "${SLURM_JOB_ID}" \
        --format=JobID,JobName,Partition,Elapsed,MaxRSS,State \
        2>&1 | sed 's/^/         /'
fi

exit "${RUN_RC}"
