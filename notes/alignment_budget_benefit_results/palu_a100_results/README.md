# PaLU A100 Results Bundle

This bundle collects the repo-tracked PaLU A100 latency comparison and its provenance under `notes/alignment_budget_benefit_results/`.
It preserves the older actual A100 partial baseline run provenance introduced by the previous issue, and now also carries a same-hardware fixed-length rerun that supersedes the old decode conclusion.

## Scope

- Hardware target: NVIDIA A100
- Model family: Llama-3-8B / PaLU compressed Llama-3-8B
- Goal: make the final `baseline / unaligned / aligned_gac` PaLU comparison easy to inspect from one place, with the fixed-length decode rerun treated as the current source of truth
- Historical provenance retained: actual A100 partial baseline run plus the failed pre-fix PaLU attempt
- Non-goal: this bundle does not preserve the older `palu_repair` runtime path as the aligned reference

## Included Files

- `source_manifest.json`: provenance index for the historical actual A100 smoke run, recovered baseline results, completed comparison run, and aligned checkpoint metadata
- `actual_results_summary.json`: issue-20 summary of the actual A100 partial baseline run, failed PaLU attempt, and remaining historical gap
- `latency_comparison.json`: normalized comparison summary with explicit `baseline` / `unaligned` / `aligned_gac` latency and alignment fields, now pointing at the fixed-length rerun
- `fixed_length_decode_comparison.json`: issue-23 summary that pairs the same-hardware prefill numbers with the fixed-length decode rerun and states whether the main gain is still decode
- `decode_root_cause_summary.json`: structured note explaining why the historical decode throughput looks too good on some shapes and how the benchmark contract was fixed
- `README.md`: human-readable summary of the completed PaLU A100 comparison

## Fixed-Length Rerun Result

- `baseline`
  - prefill average latency: `194.49 ms`
  - decode average latency: `2098.30 ms`
  - alignment: `100.0%`
- `unaligned PaLU`
  - prefill average latency: `200.37 ms`
  - decode average latency: `2618.11 ms`
  - alignment: `63.44%`
- `aligned GAC`
  - prefill average latency: `198.75 ms`
  - decode average latency: `2615.53 ms`
  - alignment: `100.0%`

## Main Takeaway

- GAC checkpoint alignment still restores PaLU to full `mod 8` alignment on A100
- Under the same-hardware fixed-length rerun, prefill changes only slightly: `200.37 ms -> 198.75 ms`
- Under the same-hardware fixed-length rerun, decode is effectively unchanged: `2618.11 ms -> 2615.53 ms`
- In throughput terms, prefill rises by about `1.0%` over unaligned PaLU, while decode rises by only about `0.1%`
- The earlier decode-dominant gain does not survive the fixed-length contract fix

## Comparison Shape

To match the ASVD / LLM-Pruner artifact style, `latency_comparison.json` always exposes:

- `prefill_latency_ms.baseline|unaligned|aligned_gac`
- `decode_latency_ms.baseline|unaligned|aligned_gac`
- `alignment_pct.baseline|unaligned|aligned_gac`

For the current checked-in fixed-length evidence:

- `baseline`, `unaligned`, and `aligned_gac` all come from the same `20260409_083500_palu_fixed_length_full_fast` rerun on `mcnode25`
- `unaligned` comes from the true non-rounded `rb1` PaLU checkpoint
- `aligned_gac` comes from the repo-native GAC-aligned checkpoint `...-rb1-gac-a100`
- The rerun keeps the original shape grid but lowers `warmup/measure/trials` to `2/5/1` so the fixed-length retest can finish on a shared 40GB A100 node

## Decode Measurement Caveat

- `decode_root_cause_summary.json` records a benchmark-contract issue in the historical decode numbers from `results/C5/20260408_120500_palu_unaligned_vs_gac/`
- The old decode path only set `max_new_tokens=gen`, so `generate()` could stop early while throughput still assumed the full requested length
- That is why several historical `gen=64` / `gen=128` pairs keep nearly the same latency but almost exactly double the reported tok/s
- Prefill does not have the same ambiguity because it is measured with a fixed forward pass over an explicit input tensor
- `scripts/run_c5_e2e_comparison.py` now enforces `min_new_tokens=gen` and records `actual_new_tokens`
- `fixed_length_decode_comparison.json` is the repo-tracked summary of the post-fix rerun and should be used for current conclusions; the 2026-04-08 run remains historical provenance only

## Provenance

- Historical actual A100 partial baseline run:
  - `slurm_logs/25035_C5_palu_env.out`
  - `slurm_logs/25035_C5_palu_env.err`
- Baseline-only recovered artifact:
  - `results/C5/20260408_102500_palu_latency_compare_retry1_baseline_only/results.json`
- Historical pre-fix completed comparison run:
  - `results/C5/20260408_120500_palu_unaligned_vs_gac/`
- Same-hardware fixed-length rerun:
  - `results/C5/20260409_083500_palu_fixed_length_full_fast/`
- Aligned checkpoint build metadata:
  - `results/palu_checkpoints/Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100/build_summary.json`
