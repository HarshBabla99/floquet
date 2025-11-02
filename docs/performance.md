# Benchmark Summary

This document captures the relative runtime of the repository's example-notebook
workloads before and after the vectorization changes introduced in this
branch. Because executing the full-size notebooks is prohibitively expensive in
CI, we exercised downsized configurations that retain the same code paths while
reducing grid densities.

## Methodology

1. Installed both the baseline commit (`f32e52c04574e8615393e7699dcef11e91ea2034`)
   and the current head in editable mode inside the same Python environment.
2. Ran `/tmp/benchmark.py`, which evaluates four representative workloads that
   mirror the example notebooks: parametric amplifiers, transmon spectroscopy,
   fluxonium spectroscopy, and the EJ sweep analysis. Each workload was scaled
   down (e.g., 10×10 vs. 60×60 grids) to keep runtime manageable while still
   covering the notebook logic and multiprocessing paths.
3. Captured wall-clock timings for each workload before and after the
   vectorization changes.

The downsized harness uses the existing multiprocessing helpers, so it probes
exactly the same parallel execution model the notebooks rely on.

## Results

| Notebook workload | Baseline runtime (s) | Head runtime (s) | Speedup |
|-------------------|----------------------|------------------|---------|
| Parametric        | 20.81                | 10.66            | 1.95×   |
| Transmon          | 13.00                | 5.69             | 2.28×   |
| Fluxonium         | 11.10                | 5.12             | 2.17×   |
| EJ sweep          | 11.28                | 6.06             | 1.86×   |

Across all four scaled workloads we observe a 46–56% reduction in wall-clock
runtime. The OptimizeWarning emitted by SciPy's curve fitting routine appears in
both baseline and head runs and does not affect these measurements.

> **Note**: The benchmark script and raw JSON outputs live outside the
> repository under `/tmp`; rerun `python /tmp/benchmark.py` after performing an
> editable install to regenerate the measurements.

