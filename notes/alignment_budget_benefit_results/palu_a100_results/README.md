# PaLU A100 Actual Results Bundle

This bundle collects the repository's currently available PaLU-related A100 evidence into one repo-tracked handoff package under `notes/alignment_budget_benefit_results/`.

## Scope

- Hardware target: NVIDIA A100-80GB
- Model family: Llama-3-8B / PaLU compressed Llama-3-8B
- Goal: make the actual A100-facing PaLU result provenance easy to inspect from one place
- Non-goal: this bundle does not pretend the repository already contains a completed real A100 PaLU inference JSON for baseline, PaLU, and PaLU+Repair

## What Counts As "Actual" In This Bundle

This bundle separates three evidence classes:

1. actual A100 runtime logs already present in `slurm_logs/`
2. actual failure provenance showing why the historical PaLU run did not finish
3. actual repo-tracked checkpoint build metadata and checked-in paper/slide numbers that provide the surrounding A100 context

## Included Files

- `source_manifest.json`: provenance index for the real A100 logs, checkpoint build summary, and paper/slide references
- `actual_results_summary.json`: structured extraction of the partial baseline run, failed PaLU attempt, checkpoint metadata, and current missing-artifact boundary
- `latency_comparison.json`: comparison-shaped summary with explicit `baseline` / `unaligned` / `aligned_gac` latency fields; measured entries are filled from current evidence and missing entries are preserved as structured gaps

## Current Best Actual Evidence

- A partial baseline run exists in `slurm_logs/25035_C5_palu_env.out`
  - prefill `batch=1, seq_len=256`: `29.86 ms`, `8608 tok/s`
  - prefill `batch=1, seq_len=512`: `46.46 ms`, `11019 tok/s`
  - decode `batch=1, ctx=256, gen=32`: `666.00 ms`, `48.0 tok/s`
- The corresponding PaLU attempt failed before benchmarking because the historical run still used the old absolute PaLU path under `/home/xinj/rap/submodules/palu`
- A repo-local PaLU checkpoint build summary exists in `results/palu_checkpoints/.../build_summary.json`, confirming the compressed checkpoint metadata now tracked in this workspace

## Why The Bundle Still Matters

The repository previously had no single place that answered:

- which PaLU A100 measurements actually ran,
- which run failed and why,
- what compressed checkpoint metadata exists locally, and
- what is still missing before a complete baseline / PaLU / PaLU+Repair A100 result set can be claimed.

This bundle makes that boundary explicit.

## Comparison Shape

To match the ASVD / LLM-Pruner artifact style, `latency_comparison.json` always exposes:

- `prefill_latency_ms.baseline|unaligned|aligned_gac`
- `decode_latency_ms.baseline|unaligned|aligned_gac`
- `alignment_pct.baseline|unaligned|aligned_gac`

For the current checked-in PaLU evidence:

- `baseline` is partially measured from the real A100 smoke run in `slurm_logs/25035_C5_palu_env.out`
- `unaligned` is marked missing because the historical run failed before the PaLU benchmark started
- `aligned_gac` is marked missing because no completed repo-tracked PaLU+Repair A100 run has been checked in yet

## Current Gap

The repository still has no completed real A100 PaLU inference JSON under a repo-tracked results directory for:

- baseline
- palu
- palu_repair

So the correct reading today is:

- actual A100 partial baseline run: present
- actual A100 PaLU attempt provenance: present
- actual completed PaLU / PaLU+Repair A100 benchmark artifact set: still missing
