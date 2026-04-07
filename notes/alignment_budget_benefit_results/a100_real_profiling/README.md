# A100 Real Profiling Bundle

This bundle collects the A100-facing profiling evidence that is already checked into this repository and rewrites it into one repo-tracked, alignment-budget-oriented handoff package.

## Scope

- Hardware target: NVIDIA A100-80GB
- Purpose: make the existing real A100 measurements, profiling commands, and alignment-budget relevance easy to trace from one place
- Non-goal: this bundle does not claim that new raw CSV/NCU dumps were produced in the current issue run

## Included Files

- `source_manifest.json`: provenance index for the checked-in paper text, slides, SLURM entrypoints, and Python profiling scripts
- `profiling_summary.json`: structured summary of the A100 measurement protocol, layer-by-layer findings, and how those findings justify the current `a100` hardware contract and method bundles

## What Is Already Repo-Tracked

- Paper text in `Latex/main.tex` records the A100 device, software stack, timing protocol, SDPA/GEMM findings, and method-level latency numbers
- Slides in `Latex/slides.tex` carry the same A100 method comparison in table form
- SLURM entrypoints under `slurm/` show the concrete commands used to run A100 sweeps and Nsight profiling
- Python scripts under `scripts/` define the actual profiling workloads used by those job wrappers

## Current Limitation

The repository currently does not check in the original A100 raw profiling tables for artifacts such as `results/alignment_sweep.csv`, `results/palu_rank_profile.csv`, or Nsight Compute output dumps. This bundle therefore records:

1. the checked-in real A100 conclusions already quoted by the paper,
2. the commands and scripts that produced those measurements, and
3. the remaining gap before the bundle can be upgraded to a raw-table archive.

## Relationship To Alignment Budget

The current repo-native `a100` hardware contract is intentionally coarse:

- `minimal_alignment = 8`
- `preferred_alignment = 16`
- `recommended_values = [32, 64, 96, 112, 128, 160, 192, 224, 256]`

This bundle explains why that contract is already supported by checked-in A100 evidence, and why method bundles such as PaLU, LLM-Pruner, and ASVD can legitimately cite A100 alignment behavior without pretending that raw profiling tables are already versioned in the repo.
