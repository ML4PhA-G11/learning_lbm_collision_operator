# Snellius node/GPU sweep — `run-all-pytorch.py`

_Generated: 2026-05-13T07:17:10+02:00_

Each row launches the same `run-all-pytorch.py` via
`jobs/run-all-pytorch.sh` with sbatch overrides for partition, GPU
count, and CPU count.

## Execution times

| Label | Partition | GPUs | CPUs | Job ID | Wall time | Slurm Elapsed | State | Speedup vs CPU |
|---|---|---|---|---|---|---|---|---|
| cpu_rome | rome | 0 | 16 | 22692039 | 5104s (01:25:04) | 01:25:15 | COMPLETED | 1.00x |
| gpu_mig | gpu_mig | 1 | 9 | 22694418 | 10474s (02:54:34) | 02:54:43 | COMPLETED | 0.49x |
| gpu_a100 | gpu_a100 | 1 | 18 | 22692041 | 10485s (02:54:45) | 02:54:54 | COMPLETED | 0.49x |
| gpu_h100 | gpu_h100 | 1 | 16 | 22692042 | 6840s (01:54:00) | 01:54:08 | COMPLETED | 0.75x |

## Approximate cost (SBU = Service Billing Unit, Snellius)

Rates are estimates from public Snellius documentation; the
authoritative numbers live in `accinfo` / `mybudget`.

| Label | Approx. SBU rate | Wall time (h) | Approx. SBUs |
|---|---|---|---|
| cpu_rome | 16 | 1.418 | 22.7 |
| gpu_mig | 55 | 2.909 | 160.0 |
| gpu_a100 | 128 | 2.913 | 372.8 |
| gpu_h100 | 192 | 1.900 | 364.8 |

## Notes

- Framework: **pytorch**.
- The model is tiny (17,702 params, batch_size=32). GPUs are
  under-utilised at this batch size; CPU partitions are often
  the best cost/speed pick **without** code changes.
- To make GPUs pay off, raise `batch_size` (e.g. 1024-4096)
  and verify loss still converges.
- `gpu_mig` is the cheapest GPU on Snellius — request the
  typed profile (`--gpus=a100_3g.20gb:1`) plus a short
  `--time` to be backfill-eligible, both already baked into
  this sweep.
