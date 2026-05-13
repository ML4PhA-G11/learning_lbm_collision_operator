# Experiment 02 — PyTorch vs TensorFlow on Snellius

## Goal

Run the same numerical workload (training-set generation, NN training,
LBM simulation) under both TensorFlow and PyTorch on identical Snellius
hardware and compare wall time + cost per partition.

## Setup

- TF script:      `run-all-tensorflow.py`  (Keras 3, XLA, `tensorflow[and-cuda]` on GPU partitions)
- PyTorch script: `run-all-pytorch.py`     (eager mode, no `torch.compile`)
- Slurm wrappers: `jobs/run-all-tensorflow.sh`, `jobs/run-all-pytorch.sh`
- Reproducer:     `scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh tensorflow|pytorch`
- Workload (identical for both frameworks): 100 k training samples,
  `batch_size=32`, 200 epochs, EarlyStopping `patience=50`, then a
  1000-step LBM simulation that calls the model on a 1024-row batch
  each step.
- Hyper-parameters, layer count, layer widths, activations (softmax
  throughout), and the D4-symmetry pipeline (8 forward passes per batch)
  are 1:1 between the two scripts.

## Headline results

### Wall time per (framework × partition)

| Partition  | TensorFlow | PyTorch    | PyTorch slowdown vs TF |
|------------|------------|------------|------------------------|
| cpu_rome   | 31:23      | 1:25:04    | 2.71×                  |
| gpu_mig    | **10:44**  | 2:54:34    | 16.27×                 |
| gpu_a100   | 16:03      | 2:54:45    | 10.89×                 |
| gpu_h100   | **09:21**  | 1:54:00    | 12.18×                 |

### Speedup vs the same-framework CPU baseline

| Partition  | TensorFlow | PyTorch     |
|------------|------------|-------------|
| cpu_rome   | 1.00×      | 1.00×       |
| gpu_mig    | 2.92×      | **0.49×**   |
| gpu_a100   | 1.96×      | **0.49×**   |
| gpu_h100   | 3.36×      | **0.75×**   |

PyTorch on GPU is **slower than PyTorch on CPU**, on every GPU partition.

### Approximate SBU cost per run

| Partition  | TensorFlow | PyTorch     |
|------------|------------|-------------|
| cpu_rome   | **8.4**    | 22.7        |
| gpu_mig    | **9.8**    | 160.0       |
| gpu_a100   | 34.2       | 372.8       |
| gpu_h100   | 29.9       | 364.8       |

SBU rates: rome 16/h (1/core-h × 16 cores); gpu_mig ≈55/h (3⁄7 of A100);
gpu_a100 ≈128/h; gpu_h100 ≈192/h. Authoritative numbers via
`accinfo` / `mybudget`.

## What the numbers say

- **Cheapest run anywhere**: TF on `cpu_rome` (8.4 SBU, 31 min). Tightest
  budget winner.
- **Best cost/speed**: TF on `gpu_mig` (9.8 SBU, 10:44). ~3× faster than
  TF-CPU for ~16 % more SBU.
- **Fastest absolute**: TF on `gpu_h100` (9:21).
- **PyTorch wins nothing here.** Its best wall time (1:25 on CPU) is
  ~9× slower than the TF best (10:44 on gpu_mig); its best SBU cost
  (22.7 on CPU) is ~2.7× more expensive than TF-CPU.

So for this workload as-written, **TensorFlow + `gpu_mig`** is the
default to recommend, and PyTorch should not be used unless the port
gets the optimisations below.

## Why PyTorch is so much slower

PyTorch eager mode pays a fixed Python + CUDA-launch cost per kernel
call. On this workload that overhead dominates because:

1. **Tiny per-step compute.** Model is 17,702 parameters, batch_size=32.
   A forward pass is microseconds of GPU compute; the launch and Python
   bookkeeping is comparable or larger.
2. **8× forward passes per training step.** The D4-symmetry pipeline
   feeds the same input through `seq_model` eight times (one per orbit
   element). PyTorch launches all eight separately; TF/Keras traces and
   fuses much of that work via XLA (we saw `"Compiled cluster using XLA!"`
   in the TF logs).
3. **Per-batch host→device copies.** The PyTorch port leaves the dataset
   on the host and calls `.to(device)` on every 32-sample mini-batch
   (2,188 copies per epoch × 200 epochs). TF builds a `tf.data` pipeline
   that pre-stages tensors device-side.
4. **No JIT.** No `torch.compile`, no `torch.jit.trace`, no CUDA graphs.
   This is the cleanest possible PyTorch port — and the slowest.

The TF–PyTorch slowdown is much smaller on CPU (2.7×) than on GPU
(10–16×). That's consistent with the kernel-launch hypothesis: on CPU
there is no GPU launch and no host→device traffic, just Python overhead,
which `torch` and `tf` pay roughly equally.

## What would close the gap

In rough order of effort vs payoff:

1. **Pre-stage tensors on the device.** Build the train/val tensors once
   on `cuda` and skip `.to(device)` per batch. Cuts the 2,188 H2D copies
   per epoch.
2. **Larger batch size**, e.g. 1024–4096. Same kernel-launch budget but
   way more compute per launch. Verify the loss still converges.
3. **`torch.compile(model)`** (since PyTorch 2.0). Compiles the forward
   graph; expect 2–10× on small ops-bound models.
4. **CUDA Graphs** for the inference path used in the LBM simulation
   loop (`model.predict`-equivalent on 1024 rows × 1000 steps). Avoids
   per-step Python and launch overhead.
5. **Fold the D4 8× into one matmul.** Stack the 8 symmetry orbits along
   the batch dimension and run a single forward pass on `(8B, 9)`.
   Cuts 8× separate launches to 1.

With (1) + (3) + (5) the PyTorch GPU runs should land within ~2× of TF.
With (4) added, it should match or beat TF on `gpu_h100` for the
simulation phase.

## Practical recommendation

For *this* workload as-written:

- **Default**: TensorFlow on `gpu_mig` (best cost/speed) or `cpu_rome`
  (cheapest absolute). See `docs/exp-01-sweep-gpu-on-snellius.md`.
- **PyTorch**: only if you are committed to porting and optimising. The
  current eager-mode port is correct but pays a 10–16× GPU penalty.

For new workloads with larger models or batch sizes, the comparison can
flip — PyTorch's overhead is amortised better when each kernel call has
real work to do.

## Reproduce

```bash
# from the project root on the Snellius login node
bash scripts/snellius-py-runtime-setup.sh           # tensorflow venv prep
bash scripts/snellius-py-runtime-setup-pytorch.sh   # add torch (PyPI wheel bundles CUDA)

bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh tensorflow
bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh pytorch
```

Each sweep writes its own artifacts:

- `artifacts-run-all-tensorflow/node-execution-time-comparison.md`
- `artifacts-run-all-pytorch/node-execution-time-comparison.md`
- `artifacts-run-all-<framework>/sweep-jobs.tsv`

The PyTorch sweep uses `--time=03:00:00` per non-MIG job and
`--time=01:30:00` for `gpu_mig`, because the eager-mode port needs
those budgets to complete 200 epochs.
