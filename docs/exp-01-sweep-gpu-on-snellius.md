# Experiment 01 — Snellius CPU vs GPU sweep for `run-all-tensorflow.py`

## Goal

Measure end-to-end wall time of `run-all-tensorflow.py` (training +
LBM simulation) on four Snellius partitions and decide which one
delivers the best cost/speed ratio for this specific workload.

## Setup

- Reproducer script: `scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh`
- Slurm wrapper:    `jobs/run-all-tensorflow.sh` (auto-installs
  `tensorflow[and-cuda]` into the existing `.venv` when a GPU is detected;
  CPU partitions leave the venv untouched).
- Workload: 100 k training samples, batch_size=32, 200 epochs (with
  early-stopping patience=50), then a 1000-step LBM simulation that calls
  `model.predict` on a 1024-row batch each step.
- Date: 2026-05-11 / -12 (Snellius account `scur0076`).

## Results

| Label    | Partition | GPUs | CPUs | Job ID    | Wall time          | Speedup vs CPU | Approx SBUs |
|----------|-----------|------|------|-----------|--------------------|----------------|-------------|
| cpu_rome | rome      | 0    | 16   | 22654734  | 31:23 (1883 s)     | 1.00×          | **8.4**     |
| gpu_mig  | gpu_mig   | 1    | 9    | 22655497  | **10:44 (644 s)**  | **2.92×**      | **9.8**     |
| gpu_a100 | gpu_a100  | 1    | 18   | 22654736  | 16:03 (963 s)      | 1.96×          | 34.2        |
| gpu_h100 | gpu_h100  | 1    | 16   | 22654737  | **09:21 (561 s)**  | **3.36×**      | 29.9        |

SBU estimates use public Snellius rates:
- `rome`: 1 SBU/core-h → 16 SBU/h for our allocation.
- `gpu_mig` (a100_3g.20gb): 3⁄7 of A100 → ≈55 SBU/h.
- `gpu_a100`: ≈128 SBU/h per A100.
- `gpu_h100`: ≈192 SBU/h per H100.

Authoritative numbers come from `accinfo` / `mybudget`.

Detailed numbers (incl. Slurm `Elapsed`) live in
`artifacts-run-all-tensorflow/node-execution-time-comparison.md`.

## Conclusion

**`gpu_mig` is the sweet spot.** Nearly identical SBU cost to the CPU
baseline (9.8 vs 8.4) but ~3× faster wall time. Cheapest GPU on Snellius
and the right amount of GPU for a 17 k-param batch=32 workload.

Detailed take by axis:

- **Cheapest absolute**: `cpu_rome` (8.4 SBU). Pick this if SBU budget is
  the only thing you care about and a 31-min wait is fine.
- **Best cost/speed**: `gpu_mig` (9.8 SBU, 3× faster than CPU). Nearly
  CPU-cheap and most of the way to H100 speed.
- **Fastest absolute**: `gpu_h100` (9:21). 3.6× the price of `gpu_mig`
  for a 13 % wall-time improvement — only worth it if those 80 seconds
  per run matter.
- **Avoid `gpu_a100`** for this workload: strictly dominated by
  `gpu_h100` (slower *and* more expensive).

### Why MIG beats the full A100 here

A `3g.20gb` MIG slice is only 3⁄7 of an A100. That it still finishes
faster than a full A100 run tells us:

1. The workload doesn't saturate a whole A100 anyway (17 k params,
   batch=32 — GPU is mostly idle waiting for the next launch).
2. The MIG slice is hardware-isolated, so it doesn't share memory
   bandwidth with neighbours; the full A100 may have been on a node
   where other jobs competed for memory bandwidth or PCIe traffic.

This is consistent with the general guidance: for small models, MIG
slices are often a better fit than full GPUs.

## Why GPUs do not give bigger speedups here

Even on H100, the model trains in ~9 min — only 3.4× faster than 16
CPU cores. The cause is the same as above: 17 k params + batch_size=32
makes per-step GPU compute dwarfed by Python and kernel-launch overhead.
The LBM simulation loop is worse still — 1000 sequential
`model.predict` calls on just 1024 rows each.

To make GPUs really pay off, raise `batch_size` (e.g. 1024–4096) in
`run-all-tensorflow.py` and verify the loss still converges. With a
saturated batch, `gpu_h100` should pull ahead by an order of magnitude.

## How we got `gpu_mig` to schedule

First attempt (in the original sweep) sat PENDING for 45 min and was
cancelled. Two sbatch tweaks fixed it:

1. **Pin the exact MIG profile**: `--gpus=a100_3g.20gb:1`. Without the
   profile, the scheduler waits for an untyped match; with it, the job
   becomes eligible for any of the 32 `3g.20gb` slices on gcn[2-5]. Note
   the *typed* `--gpus=` form is required — using `--gres=gpu:<type>:1`
   in addition to the inherited `--gpus=1` from the job script triggers
   sbatch's "with and without type identification" error.
2. **Shorter `--time=00:45:00`** instead of the 1-hour default. Short
   jobs become backfill candidates and slip into gaps between bigger
   jobs ahead in the queue.

With these, the second submission (22655497) ran almost immediately.
The sweep script already encodes both tweaks for the `gpu_mig` row.

## Practical recommendation

- Day-to-day iterations on this exact script → **`gpu_mig`** (best
  cost/speed; ~3× faster than CPU at ~the same SBU cost).
- Tightest possible SBU budget → **`rome`** (8.4 SBU per run, 31 min).
- Time-critical and budget-irrelevant → **`gpu_h100`** (9:21 wall time).
- **Skip `gpu_a100`** for this unchanged workload.

## Reproduce

```bash
# from the project root, on the Snellius login node
bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh
```
The script submits the four jobs (with the MIG profile + 45-min
backfill tweak baked in), polls until they leave the queue, and writes
both `artifacts-run-all-tensorflow/sweep-jobs.tsv` (job-id map) and
`artifacts-run-all-tensorflow/node-execution-time-comparison.md` (the
full table).
