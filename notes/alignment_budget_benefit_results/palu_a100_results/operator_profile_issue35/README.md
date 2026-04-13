# PaLU Inference Operator Profile Bundle

This directory is the tracked landing zone for direct operator-profiler snapshots on the repo's runnable PaLU path:

- `baseline`: original Llama checkpoint
- `palu`: repo-local unaligned `rb1` PaLU checkpoint
- `aligned_gac`: repo-local aligned `rb1-gac-a100` PaLU checkpoint

The goal is to answer one question with a repo-native artifact instead of an ad hoc profiler dump:
when alignment changes real LLM inference latency, which operator family actually shrinks?

## Result Snapshot

- The tracked real-A100 collection completed as Slurm job `26081` on `acclnode06`; see `submission_status.json` for provenance.
- The human-readable interpretation now lives in `analysis.md`.
- The deeper kernel-level explanation now lives in `gemm_root_cause_analysis.md`, with machine-readable bucket totals in `gemm_root_cause_summary.json`.
- The current result is directional rather than dramatic: `aligned_gac` recovers `1.28 ms / 0.71%` in `prefill` and `0.20 ms / 0.55%` in `decode` versus `palu`, and the recovered family is `gemm` in both stages.
- `aligned_gac` still remains about `3.1%` behind `baseline` in both `prefill` and `decode`, so the current alignment closes only part of the compressed-model gap.
- The root-cause follow-up shows why the cliff is muted in full inference: prefill is still dispatch-heavy, the `align1/align2` kernel tail is only a few milliseconds, decode is dominated by a small `gemv` tail, and the coarse prefill GEMM view also contains flash-attention leakage.

## Runtime Boundary

- Real collection must run on a Slurm GPU compute node.
- The current login / management node should not be treated as a CUDA profiling target.
- The deterministic issue test only validates the summary contract and bundle publisher using fixture runs.

## Tracked Slurm Entrypoint

The repo-tracked way to collect all three variants and publish the bundle in one job is:

```bash
sbatch slurm/run_llm_operator_profile.sbatch
```

Useful overrides:

```bash
sbatch slurm/run_llm_operator_profile.sbatch \
  --output-root results/operator_profiles/palu_issue35 \
  --bundle-output notes/alignment_budget_benefit_results/palu_a100_results/operator_profile_issue35 \
  --prefill-seq-len 1024 \
  --decode-context-len 1024
```

For local contract verification without a Slurm allocation or conda activation, use:

```bash
bash slurm/run_llm_operator_profile.sbatch --dry-run --skip-env-setup
```

The batch entrypoint temporarily switches from `set -u` to `set +u` around
`conda activate palu`, because the current `activate-binutils_linux-64.sh`
hook in that environment expects some shell variables to remain unset and
otherwise fails with `ADDR2LINE: unbound variable`.

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
- `analysis.md`: human-readable interpretation of the tracked summary, including stage tables and pairwise delta analysis.
- `gemm_root_cause_summary.json`: machine-readable kernel-level split for dispatch ops, alignment-sensitive kernels, align8 kernels, `gemv` tails, attention leakage, and other GEMM kernels.
- `gemm_root_cause_analysis.md`: human-readable root-cause explanation for why the 8-aligned cliff from simulation becomes a small end-to-end gain in real inference.
- `source_manifest.json`: provenance index for the three source run directories and copied `raw/config/summary/env` files.
- `submission_status.json`: latest tracked Slurm submission status, including the rerun job id, observed outputs, and where to find the final tracked bundle.

## Interpretation Contract

- `prefill` is profiled as a single forward pass over a fixed input tensor.
- `decode` is profiled as a single cached token step after a fixed-length context prefill.
- `gemm`, `sdpa`, `norm`, `elementwise`, `data_movement`, and `other` are coarse profiler families meant for stable comparison, not for claiming kernel-level root cause in isolation.
- When the question is specifically "why did 8-aligned repair not create a larger inference win?", use `gemm_root_cause_analysis.md` / `gemm_root_cause_summary.json` instead of the coarse family summary alone.
