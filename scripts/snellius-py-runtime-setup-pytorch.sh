#!/usr/bin/env bash
# Set up the PyTorch runtime needed by run-all-pytorch.py on Snellius.
#
# Run this ONCE on a Snellius login node, AFTER (or instead of) the
# TensorFlow setup script. It shares the same .venv: TF and PyTorch
# co-exist fine in one environment, so re-running this script on a
# venv already prepped for TF is safe and only adds torch.
#
# What it does:
#   1. Loads the same Snellius Python module the TF setup uses
#      (matches pyproject.toml's requires-python >=3.10,<3.14).
#   2. Installs `uv` to ~/.local/bin if missing.
#   3. Ensures the project's .venv exists (`uv sync`) — same as the TF
#      setup, idempotent if it already does.
#   4. Installs torch into the .venv with `uv pip install torch`. The
#      PyPI wheel for Linux already bundles CUDA 12 runtime libraries,
#      so no separate `[and-cuda]` extra is needed; CPU machines will
#      simply not see a GPU and torch falls back gracefully.
#   5. Smoke-tests by importing torch and checking GPU detection.
#
# Usage:
#   bash scripts/snellius-py-runtime-setup-pytorch.sh
#
# Re-running is safe: module load is idempotent, uv install is skipped
# if present, and `uv pip install torch` is a no-op when torch is
# already at the latest compatible version.

set -euo pipefail

############################################
# 0. sanity: are we on a Snellius login node?
############################################
HOSTNAME_SHORT="$(hostname -s)"
case "$HOSTNAME_SHORT" in
    int*|login*|tcn*|gcn*|hcn*|fcn*)
        echo "[setup-pt] Host: $HOSTNAME_SHORT — looks like a Snellius node."
        ;;
    *)
        echo "[setup-pt] Warning: host '$HOSTNAME_SHORT' does not match a"
        echo "           typical Snellius login pattern. Continue only if"
        echo "           you are sure this is the right machine."
        ;;
esac

if [[ "$HOSTNAME_SHORT" == tcn* || "$HOSTNAME_SHORT" == gcn* ]]; then
    echo "[setup-pt] You appear to be on a COMPUTE node. This setup script"
    echo "           is meant for the LOGIN node."
fi

############################################
# 1. project root (script lives in scripts/)
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[setup-pt] Project root: $PROJECT_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "[setup-pt] ERROR: pyproject.toml not found in $PROJECT_ROOT" >&2
    exit 1
fi

############################################
# 2. load Snellius modules (Lmod)
############################################
echo "[setup-pt] Loading modules from the 2024 Snellius stack ..."
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[setup-pt] ERROR: 'module' command not found and Lmod init not"
        echo "           in expected location. Are you on Snellius?" >&2
        exit 1
    fi
fi

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

echo "[setup-pt] Loaded modules:"
module list 2>&1 | sed 's/^/           /'

PYTHON_BIN="$(command -v python3)"
echo "[setup-pt] python3 -> $PYTHON_BIN ($(python3 --version 2>&1))"

############################################
# 3. install uv (user-local, no sudo)
############################################
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup-pt] 'uv' not found — installing to ~/.local/bin."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[setup-pt] uv already installed: $(uv --version)"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup-pt] ERROR: uv installation appears to have failed." >&2
    exit 1
fi

############################################
# 4. create .venv (uv.lock-locked deps) and add torch
############################################
echo "[setup-pt] Ensuring .venv is in sync with uv.lock ..."
uv sync --python "$PYTHON_BIN"

if uv pip show torch >/dev/null 2>&1; then
    echo "[setup-pt] torch already installed: $(uv pip show torch | awk '/^Version:/ {print $2}')"
else
    echo "[setup-pt] Installing torch into .venv (PyPI wheel bundles CUDA 12)..."
    uv pip install --quiet torch
fi

############################################
# 5. smoke test
############################################
echo "[setup-pt] Verifying that torch imports cleanly and detects GPUs (if any) ..."
uv run --python "$PYTHON_BIN" python - <<'PY'
import sys
print(f"python: {sys.version.split()[0]}")
import numpy, torch
print(f"numpy: {numpy.__version__}")
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device 0: {torch.cuda.get_device_name(0)}")
PY

echo
echo "[setup-pt] Done. Next steps:"
echo "  - Interactive use:           source .venv/bin/activate"
echo "  - One-off command:           uv run python run-all-pytorch.py"
echo "  - Submit a Slurm job (CPU/GPU agnostic):"
echo "                               sbatch jobs/run-all-pytorch.sh"
echo "  - Do NOT train on the login node."
