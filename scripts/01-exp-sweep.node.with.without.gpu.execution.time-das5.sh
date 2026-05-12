#!/usr/bin/env bash
# Sweep run-all-tensorflow.py across CPU + the two GPU options on DAS-5 and
# build a speed comparison table.
#
# What it does:
#   1. (Once) Pre-installs tensorflow[and-cuda] into .venv so each GPU job
#      does not race against the others trying to install it.
#   2. sbatch-submits 3 jobs with overrides on top of
#      jobs/run-all-tensorflow-das5.sh:
#        cpu_defq        defq  gpus=0  cpus=16  --constraint=cpunode
#        gpu_titanx      defq  gpus=1  cpus=16  --constraint=TitanX
#        gpu_rtx2080ti   proq  gpus=1  cpus=16
#   3. Waits for all three to finish (poll squeue every 30s).
#   4. Parses each .out log for the `[job] Total wall time: ...` line plus
#      sacct Elapsed/State and writes a markdown comparison to:
#        artifacts-run-all-tensorflow/node-execution-time-comparison-das5.md
#      Also keeps the job-id map at:
#        artifacts-run-all-tensorflow/sweep-jobs-das5.tsv
#
# Usage (run on the DAS-5 login node):
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time-das5.sh
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time-das5.sh --no-wait
#       just submits and exits; re-run with --summarize to rebuild the
#       markdown from sweep-jobs-das5.tsv.
#
# Notes on DAS-5 GPU choices:
#   - TitanX/Titan/GTX980 are compute capability 5.2 — current TensorFlow
#     wheels typically require 7.0+, so the gpu_titanx job will likely run
#     on the CPU even with a GPU allocated. Included for comparison.
#   - RTX2080Ti (proq) is compute capability 7.5 — the only DAS-5 GPU that
#     should actually be used by modern TF.

set -euo pipefail

############################################
# args
############################################
WAIT=1
SUBMIT=1
for arg in "$@"; do
    case "$arg" in
        --no-wait)     WAIT=0 ;;
        --summarize)   SUBMIT=0 ;;  # rebuild markdown only
        *)             echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

############################################
# locate project
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="$PROJECT_ROOT/artifacts-run-all-tensorflow"
mkdir -p "$OUT_DIR" jobs/logs
MAP_FILE="$OUT_DIR/sweep-jobs-das5.tsv"
SUMMARY="$OUT_DIR/node-execution-time-comparison-das5.md"

############################################
# environment for `uv` (login-node side)
############################################
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[sweep] ERROR: 'uv' not on PATH. Run scripts/das5-py-runtime-setup.sh first." >&2
    exit 1
fi

############################################
# 1. pre-install CUDA extras (idempotent, shared across jobs)
############################################
if [[ $SUBMIT -eq 1 ]]; then
    if ! uv pip show nvidia-cudnn-cu12 >/dev/null 2>&1; then
        TF_VERSION="$(uv pip show tensorflow 2>/dev/null | awk '/^Version:/ {print $2}')"
        if [[ -z "${TF_VERSION}" ]]; then
            echo "[sweep] ERROR: tensorflow not in .venv — run scripts/das5-py-runtime-setup.sh first." >&2
            exit 1
        fi
        echo "[sweep] Installing tensorflow[and-cuda]==${TF_VERSION} into .venv (~2-3 GB) ..."
        uv pip install --quiet "tensorflow[and-cuda]==${TF_VERSION}"
    else
        echo "[sweep] CUDA extras already in .venv — skip."
    fi
fi

############################################
# 2. submit (unless --summarize)
############################################
# label          partition  gpus  cpus  constraint
CONFIGS=(
    "cpu_defq       defq  0  16  cpunode"
    "gpu_titanx     defq  1  16  TitanX"
    "gpu_rtx2080ti  proq  1  16  -"
)

if [[ $SUBMIT -eq 1 ]]; then
    printf 'label\tpartition\tgpus\tcpus\tconstraint\tjobid\n' > "$MAP_FILE"
    declare -a JOB_IDS=()
    for cfg in "${CONFIGS[@]}"; do
        read -r LABEL PART GPUS CPUS CONSTR <<<"$cfg"
        NAME="lbm-tf-${LABEL}"
        SBATCH_EXTRA=()
        if [[ "$CONSTR" != "-" ]]; then
            SBATCH_EXTRA+=(--constraint="$CONSTR")
        fi
        JID=$(sbatch --parsable \
                --job-name="$NAME" \
                --partition="$PART" \
                --gpus="$GPUS" \
                --cpus-per-task="$CPUS" \
                --output="jobs/logs/${NAME}-%j.out" \
                --error="jobs/logs/${NAME}-%j.err" \
                "${SBATCH_EXTRA[@]}" \
                jobs/run-all-tensorflow-das5.sh)
        echo "[sweep] $LABEL on $PART  gpus=$GPUS cpus=$CPUS constraint=$CONSTR  -> jobid $JID"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$PART" "$GPUS" "$CPUS" "$CONSTR" "$JID" >> "$MAP_FILE"
        JOB_IDS+=("$JID")
    done
else
    [[ -f "$MAP_FILE" ]] || { echo "[sweep] $MAP_FILE missing, nothing to summarize." >&2; exit 1; }
    JOB_IDS=()
    while IFS=$'\t' read -r LABEL PART GPUS CPUS CONSTR JID; do
        [[ "$LABEL" == "label" ]] && continue
        JOB_IDS+=("$JID")
    done < "$MAP_FILE"
fi

if [[ $WAIT -eq 0 ]]; then
    echo "[sweep] --no-wait: skipping wait/summary; job IDs in $MAP_FILE"
    exit 0
fi

############################################
# 3. wait for jobs to leave the queue
############################################
echo "[sweep] Waiting for jobs to finish ..."
JIDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
while :; do
    IN_QUEUE=$(squeue -h -j "$JIDS_CSV" -o "%i %T %M" 2>/dev/null || true)
    [[ -z "$IN_QUEUE" ]] && break
    echo "[sweep] $(date +%H:%M:%S) — still active:"
    echo "$IN_QUEUE" | sed 's/^/         /'
    sleep 30
done
echo "[sweep] All jobs have left the queue."

############################################
# 4. parse logs + build markdown
############################################
{
    echo "# DAS-5 node/GPU sweep — \`run-all-tensorflow.py\`"
    echo
    echo "_Generated: $(date -Is)_"
    echo
    echo "Each row launches the same \`run-all-tensorflow.py\` via"
    echo "\`jobs/run-all-tensorflow-das5.sh\` with sbatch overrides for"
    echo "partition, GPU count, CPU count, and node-feature constraint."
    echo
    echo "## Execution times"
    echo
    echo "| Label | Partition | GPUs | CPUs | Constraint | Job ID | Wall time | Slurm Elapsed | State | Speedup vs CPU |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

declare -A WALL_SEC WALL_HMS ELAPSED STATE

while IFS=$'\t' read -r LABEL PART GPUS CPUS CONSTR JID; do
    [[ "$LABEL" == "label" ]] && continue
    OUT="jobs/logs/lbm-tf-${LABEL}-${JID}.out"
    sec="N/A"; hms="N/A"
    rc=""
    if [[ -f "$OUT" ]]; then
        line=$(grep -E '^\[job\] Total wall time:' "$OUT" | tail -1 || true)
        if [[ -n "$line" ]]; then
            sec=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+s$/){gsub(/s/,"",$i); print $i; exit}}' <<<"$line")
            hms=$(grep -oE '\([0-9:]+\)' <<<"$line" | tr -d '()')
        fi
        rc=$(grep -E '^\[job\] Exit code:' "$OUT" | tail -1 | awk '{print $NF}')
    fi
    WALL_SEC[$LABEL]="$sec"
    WALL_HMS[$LABEL]="$hms"

    # sacct is disabled on DAS-5; rely on the job-script's exit-code line.
    sac=$(sacct -j "$JID" -X --noheader -P --format=Elapsed,State 2>/dev/null | head -1 || true)
    ELAPSED[$LABEL]="$(cut -d'|' -f1 <<<"$sac")"
    sacct_state="$(cut -d'|' -f2 <<<"$sac")"
    if [[ -n "$sacct_state" ]]; then
        STATE[$LABEL]="$sacct_state"
    elif [[ -n "$rc" ]]; then
        if [[ "$rc" == "0" ]]; then
            STATE[$LABEL]="COMPLETED"
        else
            STATE[$LABEL]="FAILED (rc=$rc)"
        fi
    else
        STATE[$LABEL]="?"
    fi
done < "$MAP_FILE"

CPU_SEC="${WALL_SEC[cpu_defq]:-N/A}"

while IFS=$'\t' read -r LABEL PART GPUS CPUS CONSTR JID; do
    [[ "$LABEL" == "label" ]] && continue
    sec="${WALL_SEC[$LABEL]}"
    hms="${WALL_HMS[$LABEL]}"
    elapsed="${ELAPSED[$LABEL]:-?}"
    state="${STATE[$LABEL]:-?}"
    if [[ "$state" == FAILED* ]]; then
        speedup="—"
    elif [[ "$sec" != "N/A" && "$CPU_SEC" != "N/A" && "$sec" =~ ^[0-9]+$ && "$CPU_SEC" =~ ^[0-9]+$ ]]; then
        speedup=$(awk -v a="$CPU_SEC" -v b="$sec" 'BEGIN{ if(b==0) print "—"; else printf "%.2fx", a/b }')
    else
        speedup="—"
    fi
    wall_str="N/A"
    [[ "$sec" != "N/A" ]] && wall_str="${sec}s (${hms})"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$LABEL" "$PART" "$GPUS" "$CPUS" "$CONSTR" "$JID" \
        "$wall_str" "$elapsed" "$state" "$speedup" \
        >> "$SUMMARY"
done < "$MAP_FILE"

{
    echo
    echo "## Approximate resource cost (core-hours / gpu-hours)"
    echo
    echo "DAS-5 is a free academic cluster — there is no SBU/billing system"
    echo "like Snellius. The numbers below are the raw resources held, useful"
    echo "for fair-share comparisons and capacity planning."
    echo
    echo "| Label | Wall time (h) | CPU-hours | GPU-hours |"
    echo "|---|---|---|---|"
} >> "$SUMMARY"

while IFS=$'\t' read -r LABEL PART GPUS CPUS CONSTR JID; do
    [[ "$LABEL" == "label" ]] && continue
    sec="${WALL_SEC[$LABEL]}"
    if [[ "$sec" =~ ^[0-9]+$ ]]; then
        hours=$(awk -v s="$sec" 'BEGIN{printf "%.3f", s/3600.0}')
        cpuh=$(awk -v s="$sec" -v c="$CPUS" 'BEGIN{printf "%.3f", c*s/3600.0}')
        gpuh=$(awk -v s="$sec" -v g="$GPUS" 'BEGIN{printf "%.3f", g*s/3600.0}')
    else
        hours="—"; cpuh="—"; gpuh="—"
    fi
    printf '| %s | %s | %s | %s |\n' \
        "$LABEL" "$hours" "$cpuh" "$gpuh" \
        >> "$SUMMARY"
done < "$MAP_FILE"

{
    echo
    echo "## Notes"
    echo
    echo "- The model is tiny (17,702 params, batch_size=32). GPUs are"
    echo "  under-utilised at this batch size."
    echo "- \`gpu_titanx\` uses a Maxwell GPU (compute capability 5.2),"
    echo "  which current TensorFlow wheels do not target. The job may"
    echo "  silently fall back to CPU even with a GPU allocated — check"
    echo "  the log for 'Created device /job:.../gpu:0' to confirm."
    echo "- \`gpu_rtx2080ti\` on \`proq\` is the only DAS-5 GPU (Turing,"
    echo "  CC 7.5) that should actually accelerate modern TF."
    echo "- DAS-5 partitions:"
    echo "  - \`defq\` — default, 7-day limit, mix of CPU-only and TitanX"
    echo "    GPU nodes; use \`--constraint=cpunode\` or \`--constraint=TitanX\`"
    echo "    to disambiguate."
    echo "  - \`proq\` — 7-day limit, 4 nodes each with 1× RTX2080Ti."
} >> "$SUMMARY"

echo "[sweep] Summary written to $SUMMARY"
