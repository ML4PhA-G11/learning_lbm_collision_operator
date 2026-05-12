#!/usr/bin/env bash
#SBATCH --job-name=lbm-tf-das5
#SBATCH --partition=proq
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#
# Run run-all-tensorflow.py under Slurm on DAS-5. Default target: proq with
# one RTX2080Ti — the only GPU on DAS-5 with compute capability >= 7.0,
# which modern TensorFlow needs to actually accelerate on the GPU. Older
# GPUs (TitanX/Titan/GTX980 in defq) are CC 5.2 and may transparently fall
# back to CPU even when the job has a GPU allocated.
#
# Submit from the project root:
#   sbatch jobs/run-all-tensorflow-das5.sh                       # proq + RTX2080Ti
#
# Pick a different node type by overriding the SBATCH directives — flags
# passed to sbatch win over #SBATCH lines:
#   sbatch -p defq --gpus=1 --constraint=TitanX jobs/run-all-tensorflow-das5.sh
#   sbatch -p defq --gpus=0 --constraint=cpunode --cpus-per-task=16 \
#                                                jobs/run-all-tensorflow-das5.sh
# Anything with `--gpus=0` skips the CUDA install.

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
# 2. environment
############################################
# Unlike Snellius, DAS-5 has no Python Lmod module new enough for our
# pyproject.toml. We rely on a uv-managed Python (installed by
# scripts/das5-py-runtime-setup.sh) under ~/.local/share/uv. We only load
# CUDA modules when a GPU is present.
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    fi
fi

export PATH="$HOME/.local/bin:$PATH"  # for uv

if ! command -v uv >/dev/null 2>&1; then
    echo "[job] ERROR: 'uv' not on PATH. Run scripts/das5-py-runtime-setup.sh first." >&2
    exit 1
fi

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
# DAS-5 quirks vs. Snellius:
#   - nvidia-smi is NOT on PATH on compute nodes; the driver lives at
#     /cm/local/apps/cuda-driver/libs/<version>/{bin,lib64} and the Lmod
#     cuda*/toolkit modules do not expose it. We add it ourselves.
#   - SLURM_JOB_GPUS is empty on this Slurm config even when a GPU is
#     allocated. CUDA_VISIBLE_DEVICES is the reliable signal.
DRIVER_ROOT="$(ls -d /cm/local/apps/cuda-driver/libs/*/ 2>/dev/null | sort -V | tail -1 | sed 's:/$::' )"
if [[ -n "$DRIVER_ROOT" && -d "$DRIVER_ROOT/bin" ]]; then
    export PATH="$DRIVER_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$DRIVER_ROOT/lib64:${LD_LIBRARY_PATH:-}"
fi

HAS_GPU=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    HAS_GPU=1
    echo "[job] GPU allocated by Slurm: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>&1 | sed 's/^/         /'
    fi
else
    echo "[job] No GPU detected — running on CPU."
fi

if [[ $HAS_GPU -eq 1 ]]; then
    # Best-effort: load a recent CUDA toolkit + cuDNN so ptxas etc. exist
    # on PATH. tensorflow[and-cuda] still bundles its own libcudart/libcudnn,
    # but having the toolkit visible avoids XLA warnings.
    module purge >/dev/null 2>&1 || true
    module load cuda12.6/toolkit/12.6 >/dev/null 2>&1 || \
        module load cuda12.3/toolkit/12.3 >/dev/null 2>&1 || true
    module load cuDNN/cuda12.3/9.1.0.70 >/dev/null 2>&1 || \
        module load cuDNN/cuda12.1/9.1.0.70 >/dev/null 2>&1 || true
    echo "[job] Loaded CUDA modules:"
    module list 2>&1 | sed 's/^/         /'

    # Sentinel: presence of nvidia-cudnn-cu12 means the cuda extras are
    # already installed; skip the (slow) install on re-runs.
    NEEDS_CUDA_INSTALL=1
    if uv pip show nvidia-cudnn-cu12 >/dev/null 2>&1; then
        NEEDS_CUDA_INSTALL=0
    fi

    if [[ $NEEDS_CUDA_INSTALL -eq 1 ]]; then
        echo "[job] Installing tensorflow[and-cuda] into .venv (one-time, ~2-3 GB) ..."
        TF_VERSION="$(uv pip show tensorflow 2>/dev/null | awk '/^Version:/ {print $2}')"
        if [[ -z "${TF_VERSION}" ]]; then
            echo "[job] ERROR: tensorflow not in .venv — run scripts/das5-py-runtime-setup.sh first." >&2
            exit 1
        fi
        uv pip install --quiet "tensorflow[and-cuda]==${TF_VERSION}"
    else
        echo "[job] CUDA extras already installed in .venv — skipping."
    fi

    # Help TF find the bundled NVIDIA libs and ptxas at runtime.
    NVIDIA_SITE="$(uv run python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')"
    CUDA_LIBS=""
    for d in "${NVIDIA_SITE}"/*/lib; do
        [[ -d "$d" ]] && CUDA_LIBS="${CUDA_LIBS}${d}:"
    done
    export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"
    PTXAS_DIR="$(dirname "$(find "${NVIDIA_SITE}" -name ptxas -type f 2>/dev/null | head -1)")"
    if [[ -n "${PTXAS_DIR}" && -d "${PTXAS_DIR}" ]]; then
        export XLA_FLAGS="--xla_gpu_cuda_data_dir=${PTXAS_DIR%/bin}"
    fi
fi

############################################
# 5. run training + simulation
############################################
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
