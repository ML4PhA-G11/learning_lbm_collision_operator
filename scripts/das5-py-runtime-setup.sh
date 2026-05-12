#!/usr/bin/env bash
# Set up the Python runtime needed by run-all-tensorflow.py on DAS-5.
#
# Run this ONCE on the DAS-5 login node (fs0.das5.cs.vu.nl). It does NOT
# train anything — training must be submitted through Slurm (sbatch), never
# on the login node. After this script finishes, the project's .venv is
# ready and `uv run python run-all-tensorflow.py` will work inside a job.
#
# What it does (DAS-5 differs from Snellius):
#   1. DAS-5 has no Lmod Python module new enough for pyproject.toml
#      (requires-python >=3.10,<3.14) — system python is 3.6.8 — so we
#      let uv download a standalone Python 3.12 instead.
#   2. Installs `uv` to ~/.local/bin if it is not already on PATH.
#   3. Runs `uv sync` in the project root to create .venv and install
#      every pinned dependency from uv.lock.
#   4. Smoke-tests the venv by importing tensorflow and keras.
#
# Usage:
#   bash scripts/das5-py-runtime-setup.sh
#
# Re-running is safe: uv install is skipped if present, and `uv sync` only
# changes the venv when uv.lock has changed.

set -euo pipefail

############################################
# 0. sanity: are we on the DAS-5 login node?
############################################
HOSTNAME_SHORT="$(hostname -s)"
case "$HOSTNAME_SHORT" in
    fs0|fs1)
        echo "[setup] Host: $HOSTNAME_SHORT — looks like a DAS-5 login node."
        ;;
    node*)
        echo "[setup] You appear to be on a COMPUTE node ($HOSTNAME_SHORT)."
        echo "        This setup script is meant for the LOGIN node. Submit"
        echo "        it via Slurm only if you know what you are doing."
        ;;
    *)
        echo "[setup] Warning: host '$HOSTNAME_SHORT' does not match a known"
        echo "        DAS-5 pattern (fs0, fs1, node*). Continue only if you"
        echo "        are sure this is the right machine."
        ;;
esac

############################################
# 1. project root (script lives in scripts/)
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[setup] Project root: $PROJECT_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "[setup] ERROR: pyproject.toml not found in $PROJECT_ROOT" >&2
    exit 1
fi

############################################
# 2. install uv (user-local, no sudo)
############################################
# DAS-5 does not ship a recent Python via Lmod, so we cannot follow the
# Snellius pattern of `module load Python/...`. Instead, uv downloads a
# self-contained Python 3.12 build under ~/.local/share/uv/python.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] 'uv' not found — installing to ~/.local/bin via the"
    echo "        official installer (no sudo, no system changes)."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[setup] uv already installed: $(uv --version)"
fi

# Re-check after install.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] ERROR: uv installation appears to have failed." >&2
    exit 1
fi

############################################
# 3. ensure a uv-managed Python 3.12 is present
############################################
# `uv python install` is idempotent: it is a no-op if 3.12 is already
# downloaded. We pin to 3.12 (matches Snellius for parity).
echo "[setup] Ensuring uv-managed CPython 3.12 is available ..."
uv python install 3.12

############################################
# 4. create .venv and install deps from uv.lock
############################################
echo "[setup] Running 'uv sync' (creates .venv, installs locked deps) ..."
uv sync --python 3.12

############################################
# 5. smoke test
############################################
echo "[setup] Verifying that tensorflow and keras import cleanly ..."
uv run python - <<'PY'
import sys
print(f"python: {sys.version.split()[0]}")
import numpy, tensorflow as tf, keras
print(f"numpy:      {numpy.__version__}")
print(f"tensorflow: {tf.__version__}")
print(f"keras:      {keras.__version__}")
PY

echo
echo "[setup] Done. Next steps:"
echo "  - To use the venv interactively:  source .venv/bin/activate"
echo "  - To run a one-off command:       uv run python <script>.py"
echo "  - Do NOT train on the login node. Submit a Slurm job (sbatch)."
