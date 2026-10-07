# PowerSGD-DP

Code and results for

> **Communication-Efficient Distributed Learning with Differential Privacy
> via PowerSGD Compression**
> Sungyeon Hong, Independent Researcher
> Workshop on Recent Advancement in Agentic and Federated AI (RAAF-AI 2026),
> in conjunction with IEEE AGCS 2026, Paris, France, October 2026.

We combine PowerSGD low-rank gradient compression with per-sample differential
privacy (Opacus). The ordering matters: Opacus applies clipping and Gaussian
noise first, and PowerSGD compresses the already-noised gradient. Compression
is therefore post-processing and consumes no additional privacy budget.

```
per-sample clipping -> Gaussian noise (Opacus) -> PowerSGD compression -> update
```

---

## Results

All numbers below come from the runs in `results/`. Nothing is hardcoded.

### Accuracy — MNIST, 30 epochs, world size 2, five seeds

| Method | Accuracy | ε | Compression |
|---|---|---|---|
| SGD | 99.11 ± 0.02 % | ∞ | 1× |
| PowerSGD only | 98.78 ± 0.11 % | ∞ | 118.6× |
| DP-SGD only | 88.03 ± 0.86 % | 1.07 | 1× |
| **PowerSGD-DP** | **87.27 ± 0.47 %** | **1.07** | **118.6×** |

σ = 1.0, C = 1.0, δ = 1e-5, rank r = 1, seeds 0/1/2/7/42.
The accountant returns ε = 1.07 in all twenty runs, compressed or not.

### Communication — measured, per worker

| Configuration | Sent/step | Collectives/step | Aggregation | Total |
|---|---|---|---|---|
| SGD | 4.58 MiB | 8 | 159.8 s | 347 s |
| PowerSGD only | 39.5 KiB | 2 | 66.0 s | 256 s |
| DP-SGD only | 4.58 MiB | 8 | 139.2 s | 430 s |
| **PowerSGD-DP** | **39.5 KiB** | **2** | **68.5 s** | **363 s** |

118.6× less transmitted per step in every seed; training is 14.4–15.6 % shorter across seeds (table: seed 42).
The 70.7 s saved in aggregation accounts for essentially all of the 67 s
saved overall.

### Per-tensor accounting at rank 1

| Tensor | (m, n) | Params | Sent |
|---|---|---|---|
| conv1.weight | (32, 9) | 288 | 41 |
| conv1.bias | (32, 1) | 32 | 33 |
| conv2.weight | (64, 288) | 18,432 | 352 |
| conv2.bias | (64, 1) | 64 | 65 |
| fc1.weight | (128, 9216) | 1,179,648 | 9,344 |
| fc1.bias | (128, 1) | 128 | 129 |
| fc2.weight | (10, 128) | 1,280 | 138 |
| fc2.bias | (10, 1) | 10 | 11 |
| **Total** | | **1,199,882** | **10,113** |

1,199,882 / 10,113 = **118.6×**, or 0.84 % of the uncompressed volume.
Two collectives per step, one for the batched P factors and one for Q.
Reproduce with `python count_comm.py`.

### Main finding

DP does not change what compression costs on average; it changes how
predictable that cost is.

| | Compression cost | per seed |
|---|---|---|
| Without DP | 0.33 ± 0.12 pts | +0.25, +0.39, +0.30, +0.20, +0.51 |
| With DP | 0.76 ± 0.89 pts | +1.60, +1.71, +0.03, +0.72, −0.25 |

Five runs do not separate the means (0.43 ± 0.40) but they do separate the
variances: F(4,4) = 52.5, p < 0.01, a 7.2× increase in standard deviation.
A single-worker cross-check on different hardware reproduces this
independently (0.29 ± 0.13 vs 0.93 ± 1.09, F = 74.7, p < 0.001).

A single run would have supported any conclusion between "compression helps"
and "compression costs more than a point and a half".

---

## Layout

```
powersgd_dp.py        PowerSGD aggregator used by all experiments
                      (reference library code; DP is applied by Opacus)
count_comm.py         per-tensor transmitted-scalar accounting (Table II)
run_2gpu.py           two-worker 2x2 grid, five seeds (Tables I and III)
fill_grid.py          single-worker 2x2 grid, five seeds (cross-check)
ablate_2gpu.py        diagnostics and ablations (see below)

scripts/              earlier single-purpose runners whose results seed
                      cells of the single-worker grid (see fill_grid.py)
results/              raw JSON from the runs reported in the paper
```

---

## Reproducing

### Environment

```bash
pip install -r requirements.txt
```

Reported results use PyTorch 2.11.0, Opacus 1.6.0, powersgd 0.0.2
(commit `f07be92`), NCCL 2.28.9, CUDA 12.8.
Run all scripts from the repository root.

### Communication accounting (CPU, a few seconds)

```bash
python count_comm.py
```

Prints the per-tensor table and the 118.6× ratio. Cross-checks the static
count from tensor shapes against the elements actually passed to
`all_reduce`; the two agree exactly.

### Two-worker grid (2 GPUs, a few hours for 20 runs)

```bash
torchrun --nproc_per_node=2 run_2gpu.py
```

Writes `dist_<gpu>_w<world>_torch<ver>.json`. Resumable: finished cells are
skipped on restart. Records per-epoch accuracy, ε, transmitted bytes,
collective counts and timings, plus the full environment.

Per-device batch is `GLOBAL_BATCH / world_size`, so the logical batch stays
at 64 and q = 64/60000 regardless of world size. This keeps ε comparable
across world sizes and is the single most important setting here.

### Single-worker cross-check (1 GPU, ~5 h)

```bash
python fill_grid.py
```

The published grid was filled incrementally in one environment; a few cells
were imported from earlier runs in that same environment (`scripts/`).
The docstring of `fill_grid.py` lists them and explains how to run the whole
grid from scratch.

### Diagnostics

```bash
torchrun --nproc_per_node=2 ablate_2gpu.py --diag      # gradient consistency
torchrun --nproc_per_node=1 ablate_2gpu.py --diag      # baseline
torchrun --nproc_per_node=2 ablate_2gpu.py --ablate    # variants
torchrun --nproc_per_node=2 ablate_2gpu.py --sgd-comm  # SGD byte count
```

`--diag` compares the exact sum across workers against the low-rank
approximation on the same batch and reports the norm ratio and cosine
similarity. During development it located a scaling error in an earlier
version of the two-worker path: the norm ratio was exactly twice the
single-worker value while the cosine was unchanged, which isolated a
duplicated `world_size` multiplication. All reported results use the
corrected path (the `noscale` variant of `--ablate`).

---

## Configuration

Reported in full because these settings decide both convergence and
transmitted volume.

| Setting | Value |
|---|---|
| Rank | r = 1 |
| Power iterations | 2 per step |
| Warm start | P and Q retained across steps; Q seeded once |
| Compression start | first step, no warm-up phase |
| Error feedback | **not applied** — residual discarded each step |
| Min compression rate | no threshold; every tensor factorized |
| Bucketing | no DDP buckets; tensors batched by matrix shape, no padding |
| 1-D tensors (biases) | reshaped to (n, 1), same rank; sends n+1 in place of n |
| Aggregator | library's basic aggregator, not the PyTorch DDP hook |

The absence of error feedback explains the slow first epoch (11–30 % across
seeds, against 87–89 % for uncompressed DP-SGD). Enabling it is the first thing to
try for anyone building on this.

---

## Implementation notes

**Ordering.** In every private run Opacus clips per-sample gradients and adds
Gaussian noise *before* PowerSGD sees the gradient. On two workers the noise
is added once, on rank 0, and PowerSGD's power iteration is linear in the
gradient, so the factors it all-reduces are exactly those of the noised
aggregate.

**`scale_grad()` position.** `run_2gpu.py` calls Opacus' `scale_grad()` after
compression, while the paper's implementation paragraph (and `fill_grid.py`)
calls it before. The two give identical updates: power iteration
orthogonalizes its factors, so the compressed result scales linearly with its
input. Noise precedes compression in both.

**`powersgd_dp.py`.** This is the reference PowerSGD library code with a
`dp_config` attribute kept for the scripts, which always set
`enable_dp=False`. It does not clip or add noise, and rejects
`enable_dp=True`. Its `compression_rate` property is the library's analytic
estimate (121.7× here); the 118.6× reported in the paper is measured by
`count_comm.py`.

---

## Threat model

Workers are mutually trusted and the adversary observes the released model —
the standard central-DP setting. Opacus adds the Gaussian noise once, on rank
zero, and the aggregation that follows sums the per-worker contributions, so
the quantity the optimizer consumes is a correctly noised aggregate over the
logical batch.

**Limit.** A worker's local factor `P_i = G_i Q` is a function of its own
unnoised gradient, so an adversary observing inter-worker traffic is not
covered. Protecting the transcript would require each worker to noise its own
contribution, which changes the accounting.

---

## Limitations

- MNIST with a small CNN only.
- Five seeds establish that the penalty under DP is highly dispersed but are
  too few to characterize it precisely. The fixed learning rate is the most
  likely contributor.
- Fixed rank r = 1 and a single privacy budget. A rank sweep and a second σ
  would be needed to attribute the gap to the approximation rather than the
  noise. The mechanism we propose predicts dispersion shrinking with larger r
  and growing with σ.
- Two workers on one machine over PCIe. Transmitted volume is independent of
  the interconnect; the time saving is not.

---

## Citation

If you use this code, please cite the paper. GitHub's "Cite this repository"
button gives the same entry from `CITATION.cff`.

```bibtex
@inproceedings{hong2026powersgddp,
  title     = {Communication-Efficient Distributed Learning with Differential
               Privacy via PowerSGD Compression},
  author    = {Hong, Sungyeon},
  booktitle = {Proc. Workshop on Recent Advancement in Agentic and Federated
               AI (RAAF-AI), in conjunction with IEEE AGCS},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). `powersgd_dp.py` contains code from
[epfml/powersgd](https://github.com/epfml/powersgd),
Copyright (c) 2019 EPFL Machine Learning and Optimization Laboratory,
also under the MIT License.
