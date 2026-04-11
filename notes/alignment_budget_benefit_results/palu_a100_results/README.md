# PaLU A100 Results Bundle

This bundle collects the repo-tracked PaLU A100 latency comparison and its provenance under `notes/alignment_budget_benefit_results/`.
It preserves the older actual A100 partial baseline evidence and now points the current fixed-length decode conclusion to the clean issue-30 rerun.

## Scope

- Hardware target: NVIDIA A100
- Model family: Llama-3-8B / PaLU compressed Llama-3-8B
- Goal: make the final `baseline / unaligned / aligned_gac` PaLU comparison easy to inspect from one place, with the issue-30 fixed-length rerun treated as the current source of truth
- Historical provenance retained: actual A100 partial baseline run, the pre-fix comparison run, and the earlier fast fixed-length rerun
- Non-goal: this bundle does not preserve the older `palu_repair` runtime path as the aligned reference

## Included Files

- `source_manifest.json`: provenance index for the historical runs, the issue-30 raw/published rerun artifacts, and checkpoint metadata
- `actual_results_summary.json`: issue-20 summary of the actual A100 partial baseline run, failed PaLU attempt, and remaining historical gap
- `latency_comparison.json`: normalized comparison summary with explicit `baseline` / `unaligned` / `aligned_gac` latency and alignment fields, now pointing at the issue-30 fixed-length rerun
- `fixed_length_decode_comparison.json`: current fixed-length summary that pairs same-run prefill numbers with decode numbers and records whether the main aligned gain is still decode
- `decode_root_cause_summary.json`: structured note explaining why the historical decode throughput looked too good on some shapes and how the benchmark contract was fixed
- `operator_split_issue32/`: structural operator-split bundle showing that issue-30 PaLU keeps the SDPA shape contract unchanged and only changes the attention-adjacent projection GEMM path
- `README.md`: human-readable summary of the current PaLU A100 comparison

## Fixed-Length Rerun Result

- `baseline`
  - prefill average latency: `208.06 ms`
  - decode average latency: `2227.65 ms`
  - alignment: `100.0%`
- `unaligned PaLU`
  - prefill average latency: `214.61 ms`
  - decode average latency: `2803.90 ms`
  - alignment: `63.44%`
- `aligned GAC`
  - prefill average latency: `210.62 ms`
  - decode average latency: `2678.13 ms`
  - alignment: `100.0%`

## Main Takeaway

- GAC checkpoint alignment still restores PaLU to full `mod 8` alignment on A100
- Under the issue-30 fixed-length rerun, prefill improves from `214.61 ms` to `210.62 ms`
- Under the issue-30 fixed-length rerun, decode improves from `2803.90 ms` to `2678.13 ms`
- In throughput terms, prefill rises by `2.18%` over unaligned PaLU and decode rises by `4.37%`
- The clean rerun still shows a decode-side gain, but it is moderate and now backed by rebuilt checkpoints plus the fixed-token contract
- Issue 32 further narrows the changed operator family: `HeadwiseLowRankModule` reconstructs back to the original attention width, so the checked-in gain should be attributed to the `k_proj` / `v_proj` low-rank projection path rather than an SDPA width cliff

## Comparison Shape

To match the ASVD / LLM-Pruner artifact style, `latency_comparison.json` always exposes:

- `prefill_latency_ms.baseline|unaligned|aligned_gac`
- `decode_latency_ms.baseline|unaligned|aligned_gac`
- `alignment_pct.baseline|unaligned|aligned_gac`

For the current checked-in fixed-length evidence:

- `baseline`, `unaligned`, and `aligned_gac` all come from `results/C5/issue30_palu_fixed_token/`
- tracked copies live under `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_issue30/`
- `unaligned` comes from the true non-rounded `rb1` PaLU checkpoint
- `aligned_gac` comes from the repo-native GAC-aligned checkpoint `...-rb1-gac-a100`
- the rerun keeps the original shape grid and uses `warmup/measure/trials = 2/5/1` on an A100 80GB node (`acclnode06`)

## Decode Measurement Caveat

- `decode_root_cause_summary.json` records a benchmark-contract issue in the historical decode numbers from `results/C5/20260408_120500_palu_unaligned_vs_gac/`
- the old decode path only set `max_new_tokens=gen`, so `generate()` could stop early while throughput still assumed the full requested length
- that is why several historical `gen=64` / `gen=128` pairs keep nearly the same latency but almost exactly double the reported tok/s
- prefill does not have the same ambiguity because it is measured with a fixed forward pass over an explicit input tensor
- `scripts/run_c5_e2e_comparison.py` now enforces `min_new_tokens=gen` and records `actual_new_tokens`
- `fixed_length_decode_comparison.json` and `latency_comparison.json` should be read from the issue-30 rerun; the 2026-04-08 and 2026-04-09 runs remain historical provenance only

## Operator Split

- `operator_split_issue32/operator_split_summary.json` is the current checked-in PaLU split artifact under the corrected fixed-length contract
- it uses the issue-30 fixed-length rerun, the `rb1` / `gac-a100` checkpoint configs, and `third_party/palu/palu/model/modules/svd_linear.py`
- the key finding is structural: both PaLU checkpoints keep `head_dim=128`, `num_attention_heads=32`, and `num_key_value_heads=8`
- because `HeadwiseLowRankModule` reconstructs back to the original output width before attention, SDPA shape does not change between unaligned and aligned PaLU
- the changed operator family is therefore the attention-adjacent projection GEMM path inside `k_proj` / `v_proj`

## Provenance

- Historical actual A100 partial baseline run:
  - `slurm_logs/25035_C5_palu_env.out`
  - `slurm_logs/25035_C5_palu_env.err`
- Baseline-only recovered artifact:
  - `results/C5/20260408_102500_palu_latency_compare_retry1_baseline_only/results.json`
- Historical pre-fix completed comparison run:
  - `results/C5/20260408_120500_palu_unaligned_vs_gac/`
- Issue-30 raw fixed-length rerun:
  - `results/C5/issue30_palu_fixed_token/`
- Issue-30 tracked fixed-length copies:
  - `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_issue30/results.json`
  - `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_issue30/comparison_summary.json`
- Historical fast fixed-length rerun retained for provenance only:
  - `results/C5/20260409_083500_palu_fixed_length_full_fast/`
- Aligned checkpoint build metadata:
  - `results/palu_checkpoints/Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100/build_summary.json`
