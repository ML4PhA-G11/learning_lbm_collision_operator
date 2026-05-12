#!/usr/bin/env bash
# Sweep run-all-<framework>.py across CPU + 3 GPU partitions on Snellius
# and build a cost/speed comparison table.
#
# Default framework: tensorflow. Switch to pytorch with the first
# positional arg:
#
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh                  # tensorflow
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh pytorch          # pytorch
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh tensorflow --no-wait
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh pytorch --summarize
#
# What it does:
#   1. (tensorflow only) Pre-installs tensorflow[and-cuda] into .venv so
#      each GPU job does not race against the others on `uv pip install`.
#   2. sbatch-submits 4 jobs using the per-framework Slurm wrapper:
#        cpu_rome   rome     gpus=0                        cpus=16
#        gpu_mig    gpu_mig  --gpus=a100_3g.20gb:1         cpus=9   time=00:45:00
#        gpu_a100   gpu_a100 gpus=1                        cpus=18
#        gpu_h100   gpu_h100 gpus=1                        cpus=16
#   3. Waits for all four to finish (poll squeue every 30s).
#   4. Parses each .out log for `[job] Total wall time:` plus sacct
#      Elapsed/State and writes a per-framework markdown comparison at:
#        artifacts-run-all-<framework>/node-execution-time-comparison.md
#      with the job-id map at:
#        artifacts-run-all-<framework>/sweep-jobs.tsv
#
# Re-runs:
#   - `--summarize` rebuilds the markdown from the cached job IDs (cheap).
#   - `--no-wait` just submits and exits; re-run without flags to summarize.

set -euo pipefail

############################################
# args (first positional = framework; rest = flags)
############################################
FRAMEWORK="tensorflow"
WAIT=1
SUBMIT=1
for arg in "$@"; do
    case "$arg" in
        tensorflow|pytorch) FRAMEWORK="$arg" ;;
        --no-wait)          WAIT=0 ;;
        --summarize)        SUBMIT=0 ;;
        *)                  echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

case "$FRAMEWORK" in
    tensorflow)
        JOB_SH="jobs/run-all-tensorflow.sh"
        JOB_PREFIX="lbm-tf"
        ART_SUFFIX="tensorflow"
        ;;
    pytorch)
        JOB_SH="jobs/run-all-pytorch.sh"
        JOB_PREFIX="lbm-pt"
        ART_SUFFIX="pytorch"
        ;;
esac

############################################
# locate project
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="$PROJECT_ROOT/artifacts-run-all-${ART_SUFFIX}"
mkdir -p "$OUT_DIR" jobs/logs
MAP_FILE="$OUT_DIR/sweep-jobs.tsv"
SUMMARY="$OUT_DIR/node-execution-time-comparison.md"

echo "[sweep] Framework: ${FRAMEWORK}"
echo "[sweep] Job script: ${JOB_SH}"
echo "[sweep] Output dir: ${OUT_DIR}"

############################################
# environment for `uv` (login-node side)
############################################
if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge >/dev/null 2>&1 || true
module load 2024 >/dev/null 2>&1 || true
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"

############################################
# 1. one-time installs (per framework, idempotent)
############################################
if [[ $SUBMIT -eq 1 ]]; then
    case "$FRAMEWORK" in
        tensorflow)
            if ! uv pip show nvidia-cudnn-cu12 >/dev/null 2>&1; then
                TF_VERSION="$(uv pip show tensorflow 2>/dev/null | awk '/^Version:/ {print $2}')"
                if [[ -z "${TF_VERSION}" ]]; then
                    echo "[sweep] ERROR: tensorflow not in .venv — run scripts/snellius-py-runtime-setup.sh first." >&2
                    exit 1
                fi
                echo "[sweep] Installing tensorflow[and-cuda]==${TF_VERSION} into .venv (~2-3 GB) ..."
                uv pip install --quiet "tensorflow[and-cuda]==${TF_VERSION}"
            else
                echo "[sweep] tensorflow CUDA extras already in .venv — skip."
            fi
            ;;
        pytorch)
            if ! uv pip show torch >/dev/null 2>&1; then
                echo "[sweep] torch not in .venv — installing (PyPI wheel bundles CUDA) ..."
                uv pip install --quiet torch
            else
                echo "[sweep] torch already in .venv — skip."
            fi
            ;;
    esac
fi

############################################
# 2. submit (unless --summarize)
############################################
# label  partition  gpus  cpus
CONFIGS=(
    "cpu_rome rome     0  16"
    "gpu_mig  gpu_mig  1  9"
    "gpu_a100 gpu_a100 1  18"
    "gpu_h100 gpu_h100 1  16"
)

if [[ $SUBMIT -eq 1 ]]; then
    printf 'label\tpartition\tgpus\tcpus\tjobid\n' > "$MAP_FILE"
    declare -a JOB_IDS=()
    for cfg in "${CONFIGS[@]}"; do
        read -r LABEL PART GPUS CPUS <<<"$cfg"
        NAME="${JOB_PREFIX}-${LABEL}"

        # Per-partition tweaks. For gpu_mig we pin the exact MIG profile
        # Snellius exposes (a100_3g.20gb, 32 slices total across gcn[2-5])
        # so the scheduler matches us against any of them, and we request
        # 45 min instead of 1 h so the job is eligible for backfill into
        # gaps between longer jobs.
        TIME_ARG="--time=01:00:00"
        case "$PART" in
            gpu_mig)
                # Typed --gpus form cleanly overrides the default --gpus=1
                # in the TF wrapper; --gres would conflict with it
                # ("with and without type identification" sbatch error).
                GPU_FLAG=(--gpus="a100_3g.20gb:${GPUS}")
                TIME_ARG="--time=00:45:00"
                ;;
            *)
                GPU_FLAG=(--gpus="$GPUS")
                ;;
        esac

        JID=$(sbatch --parsable \
                --job-name="$NAME" \
                --partition="$PART" \
                "${GPU_FLAG[@]}" \
                --cpus-per-task="$CPUS" \
                "$TIME_ARG" \
                --output="jobs/logs/${NAME}-%j.out" \
                --error="jobs/logs/${NAME}-%j.err" \
                "$JOB_SH")
        echo "[sweep] $LABEL on $PART  gpus=$GPUS cpus=$CPUS  -> jobid $JID"
        printf '%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$PART" "$GPUS" "$CPUS" "$JID" >> "$MAP_FILE"
        JOB_IDS+=("$JID")
    done
else
    [[ -f "$MAP_FILE" ]] || { echo "[sweep] $MAP_FILE missing, nothing to summarize." >&2; exit 1; }
    JOB_IDS=()
    while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
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
    echo "# Snellius node/GPU sweep — \`run-all-${ART_SUFFIX}.py\`"
    echo
    echo "_Generated: $(date -Is)_"
    echo
    echo "Each row launches the same \`run-all-${ART_SUFFIX}.py\` via"
    echo "\`${JOB_SH}\` with sbatch overrides for partition, GPU"
    echo "count, and CPU count."
    echo
    echo "## Execution times"
    echo
    echo "| Label | Partition | GPUs | CPUs | Job ID | Wall time | Slurm Elapsed | State | Speedup vs CPU |"
    echo "|---|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

declare -A WALL_SEC WALL_HMS ELAPSED STATE

while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    OUT="jobs/logs/${JOB_PREFIX}-${LABEL}-${JID}.out"
    sec="N/A"; hms="N/A"
    if [[ -f "$OUT" ]]; then
        line=$(grep -E '^\[job\] Total wall time:' "$OUT" | tail -1 || true)
        if [[ -n "$line" ]]; then
            sec=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+s$/){gsub(/s/,"",$i); print $i; exit}}' <<<"$line")
            hms=$(grep -oE '\([0-9:]+\)' <<<"$line" | tr -d '()')
        fi
    fi
    WALL_SEC[$LABEL]="$sec"
    WALL_HMS[$LABEL]="$hms"

    sac=$(sacct -j "$JID" -X --noheader -P --format=Elapsed,State 2>/dev/null | head -1 || true)
    ELAPSED[$LABEL]="$(cut -d'|' -f1 <<<"$sac")"
    STATE[$LABEL]="$(cut -d'|' -f2 <<<"$sac")"
done < "$MAP_FILE"

CPU_SEC="${WALL_SEC[cpu_rome]:-N/A}"

while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    sec="${WALL_SEC[$LABEL]}"
    hms="${WALL_HMS[$LABEL]}"
    elapsed="${ELAPSED[$LABEL]:-?}"
    state="${STATE[$LABEL]:-?}"
    if [[ "$sec" != "N/A" && "$CPU_SEC" != "N/A" && "$sec" =~ ^[0-9]+$ && "$CPU_SEC" =~ ^[0-9]+$ ]]; then
        speedup=$(awk -v a="$CPU_SEC" -v b="$sec" 'BEGIN{ if(b==0) print "—"; else printf "%.2fx", a/b }')
    else
        speedup="—"
    fi
    wall_str="N/A"
    [[ "$sec" != "N/A" ]] && wall_str="${sec}s (${hms})"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$LABEL" "$PART" "$GPUS" "$CPUS" "$JID" \
        "$wall_str" "$elapsed" "$state" "$speedup" \
        >> "$SUMMARY"
done < "$MAP_FILE"

{
    echo
    echo "## Approximate cost (SBU = Service Billing Unit, Snellius)"
    echo
    echo "Rates are estimates from public Snellius documentation; the"
    echo "authoritative numbers live in \`accinfo\` / \`mybudget\`."
    echo
    echo "| Label | Approx. SBU rate | Wall time (h) | Approx. SBUs |"
    echo "|---|---|---|---|"
} >> "$SUMMARY"

# rough SBU rates (per allocated unit per hour):
#   rome:     1 SBU per core-hour                                 -> 16 cpu ≈ 16/h
#   gpu_mig:  Snellius slices each A100 into a100_3g.20gb (3/7 of
#             the GPU). 3/7 × 128 ≈ 55 SBU/h per slice.
#   gpu_a100: ~128 SBU per A100-hour
#   gpu_h100: ~192 SBU per H100-hour (newer; check `accinfo`)
declare -A RATE=(
    [cpu_rome]=16
    [gpu_mig]=55
    [gpu_a100]=128
    [gpu_h100]=192
)

while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    sec="${WALL_SEC[$LABEL]}"
    rate="${RATE[$LABEL]:-?}"
    if [[ "$sec" =~ ^[0-9]+$ ]]; then
        sbus=$(awk -v r="$rate" -v s="$sec" 'BEGIN{printf "%.1f", r*s/3600.0}')
        hours=$(awk -v s="$sec" 'BEGIN{printf "%.3f", s/3600.0}')
    else
        sbus="—"; hours="—"
    fi
    printf '| %s | %s | %s | %s |\n' \
        "$LABEL" "$rate" "$hours" "$sbus" \
        >> "$SUMMARY"
done < "$MAP_FILE"

{
    echo
    echo "## Notes"
    echo
    echo "- Framework: **${FRAMEWORK}**."
    echo "- The model is tiny (17,702 params, batch_size=32). GPUs are"
    echo "  under-utilised at this batch size; CPU partitions are often"
    echo "  the best cost/speed pick **without** code changes."
    echo "- To make GPUs pay off, raise \`batch_size\` (e.g. 1024-4096)"
    echo "  and verify loss still converges."
    echo "- \`gpu_mig\` is the cheapest GPU on Snellius — request the"
    echo "  typed profile (\`--gpus=a100_3g.20gb:1\`) plus a short"
    echo "  \`--time\` to be backfill-eligible, both already baked into"
    echo "  this sweep."
} >> "$SUMMARY"

echo "[sweep] Summary written to $SUMMARY"
