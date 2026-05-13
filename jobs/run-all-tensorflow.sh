#!/usr/bin/env bash
#SBATCH --job-name=lbm-tf
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#
# Run run-all-tensorflow.py under Slurm. Default target: gpu_mig (cheapest
# GPU on Snellius — one MIG slice of an A100). The script auto-detects
# whether a GPU is actually present and, on GPU, makes sure the CUDA-bundled
# tensorflow build is installed into .venv. On CPU it does nothing extra,
# so the same .venv keeps working for laptop runs.
#
# Submit from the project root:
#   sbatch jobs/run-all-tensorflow.sh                          # gpu_mig (default)
#
# Pick a different node type by overriding the SBATCH directives from the
# command line — flags passed to sbatch win over #SBATCH lines in the file:
#   sbatch -p gpu_a100 --gpus=1 --cpus-per-task=18 jobs/run-all-tensorflow.sh
#   sbatch -p gpu_h100 --gpus=1 --cpus-per-task=16 jobs/run-all-tensorflow.sh
#   sbatch -p rome     --gpus=0 --cpus-per-task=16 jobs/run-all-tensorflow.sh
#   sbatch -p genoa    --gpus=0 --cpus-per-task=24 jobs/run-all-tensorflow.sh
# Anything with `--gpus=0` (or a non-gpu partition) skips the CUDA install.
#
# Running on a laptop without Slurm still works — just:
#   uv run python run-all-tensorflow.py
# The .py code itself is GPU-agnostic; only this wrapper touches CUDA.

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
# Slurm copies the batch script into a spool dir before running it, so
# $BASH_SOURCE[0] does not point at the project. Prefer $SLURM_SUBMIT_DIR
# (the directory `sbatch` was invoked from); fall back to script-relative
# for direct/local execution. SBATCH --output paths are also relative to
# $SLURM_SUBMIT_DIR, so always submit from the project root.
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
# 3. CPU/threading hints — applied always
############################################
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NCPU}"
export TF_NUM_INTRAOP_THREADS="${NCPU}"
export TF_NUM_INTEROP_THREADS=2
export NUMBA_NUM_THREADS="${NCPU}"
echo "[job] CPUs allocated: ${NCPU}"

############################################
# 4. GPU detection + CUDA-bundled tensorflow (idempotent)
############################################
# Tensorflow's default pip wheel has CUDA *stubs* but no runtime libraries;
# `tensorflow[and-cuda]` adds the nvidia-* pip packages that ship the actual
# CUDA + cuDNN libs alongside the wheel. We install that extra into the
# existing .venv with `uv pip install` (does NOT touch pyproject.toml or
# uv.lock), so laptop users running plain `uv sync` are unaffected.
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    HAS_GPU=1
    echo "[job] GPU(s) visible to the job:"
    nvidia-smi -L | sed 's/^/         /'
else
    echo "[job] No GPU detected — running on CPU."
fi

if [[ $HAS_GPU -eq 1 ]]; then
    # Sentinel: presence of nvidia-cudnn-cu12 in the venv means the cuda
    # extras are already installed; skip the (slow) install on re-runs.
    NEEDS_CUDA_INSTALL=1
    if uv pip show nvidia-cudnn-cu12 >/dev/null 2>&1; then
        NEEDS_CUDA_INSTALL=0
    fi

    if [[ $NEEDS_CUDA_INSTALL -eq 1 ]]; then
        echo "[job] Installing tensorflow[and-cuda] into .venv (one-time, ~2-3 GB) ..."
        # Pin the same TF version that's already in uv.lock so the bundled
        # cuda libs match. Reads the version straight out of the venv.
        TF_VERSION="$(uv pip show tensorflow 2>/dev/null | awk '/^Version:/ {print $2}')"
        if [[ -z "${TF_VERSION}" ]]; then
            echo "[job] ERROR: tensorflow not in .venv — run scripts/snellius-py-runtime-setup.sh first." >&2
            exit 1
        fi
        uv pip install --quiet "tensorflow[and-cuda]==${TF_VERSION}"
    else
        echo "[job] CUDA extras already installed in .venv — skipping."
    fi

    # Help TF find the bundled NVIDIA libs and ptxas at runtime. uv places
    # them under .venv/lib/python3.12/site-packages/nvidia/...
    NVIDIA_SITE="$(uv run python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')"
    # Each nvidia-* package exposes its libs under <pkg>/lib; collect them all.
    CUDA_LIBS=""
    for d in "${NVIDIA_SITE}"/*/lib; do
        [[ -d "$d" ]] && CUDA_LIBS="${CUDA_LIBS}${d}:"
    done
    export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"
    # ptxas ships with nvidia-cuda-nvcc-cu12; XLA needs to be told where.
    PTXAS_DIR="$(dirname "$(find "${NVIDIA_SITE}" -name ptxas -type f 2>/dev/null | head -1)")"
    if [[ -n "${PTXAS_DIR}" && -d "${PTXAS_DIR}" ]]; then
        export XLA_FLAGS="--xla_gpu_cuda_data_dir=${PTXAS_DIR%/bin}"
    fi
fi

############################################
# 5. run training + simulation
############################################
# `uv run` activates .venv automatically. -u flushes stdout so logs stream.
echo "[job] Launching run-all-tensorflow.py ..."
set +e
uv run python -u run-all-tensorflow.py
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
