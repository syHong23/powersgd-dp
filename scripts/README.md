# scripts/

Earlier single-purpose runners on one GPU. They are kept because their
results seed cells of the single-worker cross-check grid (see the docstring
of `fill_grid.py`):

```
run_sgd_control.py        SGD, seed 42           -> KNOWN_SEED42 in fill_grid.py
run_powersgd_matched.py   PowerSGD only, seed 42 -> KNOWN_SEED42 in fill_grid.py
run_seeds.py              DP-SGD only and PowerSGD-DP, seeds 0 and 7
                          -> imported from seed_results.json
```

For new experiments use `fill_grid.py` (one worker) or `run_2gpu.py`
(two workers), which run all four configurations. Run these scripts from
the repository root so that `powersgd_dp.py` can be imported, e.g.
`python3 scripts/run_seeds.py`.
