# Prefill Operator Attribution

This note is the next concrete slice after [`notes/operator_impact_map.md`](notes/operator_impact_map.md):
instead of asking "which operator matters most in general?", it narrows the current best-supported hypothesis into one repo-tracked artifact that can be profiled next.

## Why This Artifact Exists

The checked-in evidence is already asymmetric:

- ASVD and LLM-Pruner have strong `prefill` wins in `Latex/main.tex` and `notes/alignment_budget_benefit_results/a100_real_profiling/profiling_summary.json`
- the fresh fixed-length `decode` reruns do not show an aligned win for either method
- `notes/full_paper_plan.md` therefore lists "operator-attribution artifact for ASVD / LLM-Pruner `prefill`" as the first must-have missing experiment

So this issue does not reopen the whole operator-mapping problem.
It only prepares the highest-priority next artifact: a `prefill`-focused attribution slice for the two methods whose checked-in end-to-end evidence is already strongest.

## Current Working Claim

Based on the current repo evidence, the first operator family to isolate is large-`prefill` GEMM:

- ASVD most likely pays through GEMM `K`, because factorization introduces a middle rank `r` in the projection path
- LLM-Pruner most likely pays through MLP GEMM `N` and the paired downstream GEMM `K`, because pruning changes kept intermediate width

This is consistent with the conclusion already recorded in `notes/operator_impact_map.md`:
for the current checked-in results, GEMM `K/N` is better supported than SDPA as the main source of model-level slowdown.

## ASVD Attribution Slice

### Operator hypothesis

- Method path: low-rank factorization across projection weights
- Dominant stage: `prefill`
- Primary operator: GEMM
- Primary changed axis: GEMM `K`

`Latex/main.tex` already states that ASVD applies across all 32 layers and 7 projections, and that the unconstrained path can erase expected speedup even after parameter reduction.
For the next attribution artifact, the key question is not whether ASVD can be slow in aggregate, but whether the slowdown is dominated by the factorized middle-rank GEMM checkpoints.

### Real-shape checkpoints

The new experiment spec tracks the following ASVD rank checkpoints:

- `300`
- `512`
- `1024`
- `1536`
- `2048`
- `3072`
- `3184`
- `3192`

These values are not another toy `64..160` sweep.
They come from the paper-side ASVD rank distribution and the repo scripts that already discuss the upper real-rank window, especially `scripts/create_paper_figures.py` and `scripts/gemv_small_k_test_v2.py`.

## LLM-Pruner Attribution Slice

### Operator hypothesis

- Method path: structured MLP pruning
- Dominant stage: `prefill`
- Primary operators: GEMM on `gate_proj` / `up_proj` and the paired `down_proj`
- Primary changed axes: GEMM `N` for gate/up, paired GEMM `K` for down

`Latex/main.tex` already records that LLM-Pruner only prunes MLP widths and that the aligned variant removes a large `prefill` penalty while preserving quality.
That makes LLM-Pruner the best current target for a clean "MLP width -> GEMM axis -> prefill latency" attribution step.

### Real-shape checkpoints

The new experiment spec tracks representative kept-width checkpoints and their nearest aligned neighbors:

- `4872` vs `4880`
- `5632` vs `5640`
- `6096` vs `6104`

These checkpoints are anchored to the repo-native LLM-Pruner artifact and code path:

- `notes/alignment_budget_benefit_results/llmpruner_example_input.json`
- `scripts/llmpruner_gac_experiment.py`

The goal is to profile both sides of the same structural width story:

- `LLMPRUNER_prefill_gateup_n_real_shapes` isolates the MLP output-width (`N`) effect
- `LLMPRUNER_prefill_down_k_real_shapes` mirrors the same widths onto the paired downstream `K` effect

## Ready-to-Run Experiment Specs

The checked-in spec lives at [`experiments/operator_attribution_prefill.yaml`](experiments/operator_attribution_prefill.yaml) and currently exposes three experiments:

1. `ASVD_prefill_gemm_k_real_shapes`
2. `LLMPRUNER_prefill_gateup_n_real_shapes`
3. `LLMPRUNER_prefill_down_k_real_shapes`

All three can be launched with the existing runner:

```bash
python -m scripts.run_experiment run \
  --spec experiments/operator_attribution_prefill.yaml \
  --name ASVD_prefill_gemm_k_real_shapes \
  --results-root results
```

Swap `--name` to either LLM-Pruner entry for the other two experiments.

The supporting code change in `src/experiment_runner.py` adds explicit dense sweep value support:

- `K_values` for `gemm_k_dense`
- `N_values` for `gemm_n_dense`

That matters because the old runner logic implicitly assumed the dense sweep always lived near `64..160`.
For attribution runs on real ASVD ranks or LLM-Pruner kept widths, that would silently collapse the sweep to the wrong low-dimension window.

## How To Read The Next Result

When these specs are profiled, the next attribution artifact should answer three concrete questions:

1. Does ASVD's current `prefill` story stay strongest when only real-shape GEMM `K` checkpoints are isolated?
2. For LLM-Pruner, is the larger penalty concentrated on gate/up GEMM `N`, down-projection GEMM `K`, or both?
3. Do the aligned checkpoints consistently recover the same direction of win already seen in the paper-side `prefill` table?

If the answer is "yes", the repo will finally have an operator-facing explanation for the current strongest method-level result instead of only an end-to-end before/after table.

## Deliverable Boundary

This artifact is intentionally one step short of full operator-level timing.

What it does provide:

- a narrowed `prefill` operator hypothesis for ASVD and LLM-Pruner
- real-shape checkpoints instead of toy sweep ranges
- a repo-tracked experiment spec that can be run directly by the existing benchmark CLI

What it does not claim yet:

- that operator-level attribution has already been measured
- that PaLU or token-eviction attribution is solved
- that GEMM `K/N` is the only bottleneck for all compression methods
