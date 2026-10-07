# Raw results

JSON files produced by the experiment scripts for the runs reported in the
paper.

```
dist_rtx3090_w2_torch2.11.json              two-worker grid, five seeds
                                            -> Tables I and III

grid_rtx4500adageneration_torch2.11.json    single-worker cross-check
                                            -> dispersion result, Section V

ablate_w2.json                              two-worker ablation, corrected
                                            path (`noscale`), seeds 42 and 0
```

Each file records the environment it was produced in (GPU, world size,
per-device and logical batch, torch / opacus / NCCL versions, and every
hyperparameter) alongside per-epoch accuracy, epsilon, transmitted bytes,
collective counts and timings.

The single-worker grid also contains a `_legacy_rtx3090` block: results of an
earlier round on a different machine whose library versions were not
recorded. It is kept for the record and excluded from every statistic.
