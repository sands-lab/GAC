# PaLU Inference Operator Profile Bundle

This directory is the tracked landing zone for direct operator-profiler snapshots on the repo's runnable PaLU path:

- `baseline`: original Llama checkpoint
- `palu`: repo-local unaligned `rb1` PaLU checkpoint
- `aligned_gac`: repo-local aligned `rb1-gac-a100` PaLU checkpoint

The goal is to answer one question with a repo-native artifact instead of an ad hoc profiler dump:
when alignment changes real LLM inference latency, which operator family actually shrinks?

## Runtime Boundary

- Real collection must run on a Slurm GPU compute node.
- The current login / management node should not be treated as a CUDA profiling target.
- The deterministic issue test only validates the summary contract and bundle publisher using fixture runs.

## Collection Commands

Run the profiler once per variant:

```bash
python3 scripts/profile_llm_inference_operators.py \
  --variant baseline \
  --output-dir results/operator_profiles/palu_issue35/baseline \
  --prefill-seq-len 1024 \
  --decode-context-len 1024 \
  --export-trace
```

```bash
python3 scripts/profile_llm_inference_operators.py \
  --variant palu \
  --palu-pattern Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1 \
  --output-dir results/operator_profiles/palu_issue35/palu \
  --prefill-seq-len 1024 \
  --decode-context-len 1024 \
  --export-trace
```

```bash
python3 scripts/profile_llm_inference_operators.py \
  --variant aligned_gac \
  --palu-aligned-pattern Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100 \
  --output-dir results/operator_profiles/palu_issue35/aligned_gac \
  --prefill-seq-len 1024 \
  --decode-context-len 1024 \
  --export-trace
```

Each run directory is expected to contain:

- `config.json`
- `raw.json`
- `summary.json`
- `env.json`
- optional `traces/*.json`

## Publish Command

```bash
python3 scripts/publish_llm_inference_operator_profile_bundle.py \
  --baseline-run-dir results/operator_profiles/palu_issue35/baseline \
  --palu-run-dir results/operator_profiles/palu_issue35/palu \
  --aligned-run-dir results/operator_profiles/palu_issue35/aligned_gac \
  --output-dir notes/alignment_budget_benefit_results/palu_a100_results/operator_profile_issue35
```

## Published Artifacts

- `palu_inference_operator_profile_summary.json`: structured summary with per-stage total self CUDA time, operator-family shares, and baseline / unaligned / aligned comparisons.
- `source_manifest.json`: provenance index for the three source run directories and copied `raw/config/summary/env` files.

## Interpretation Contract

- `prefill` is profiled as a single forward pass over a fixed input tensor.
- `decode` is profiled as a single cached token step after a fixed-length context prefill.
- `gemm`, `sdpa`, `norm`, `elementwise`, `data_movement`, and `other` are coarse profiler families meant for stable comparison, not for claiming kernel-level root cause in isolation.
