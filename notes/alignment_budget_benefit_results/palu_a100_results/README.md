# PaLU A100 Results Bundle

This bundle collects the repo-tracked PaLU A100 latency comparison and its provenance under `notes/alignment_budget_benefit_results/`.

## Scope

- Hardware target: NVIDIA A100
- Model family: Llama-3-8B / PaLU compressed Llama-3-8B
- Goal: make the final `baseline / unaligned / aligned_gac` PaLU comparison easy to inspect from one place
- Non-goal: this bundle does not preserve the older `palu_repair` runtime path as the aligned reference

## Included Files

- `source_manifest.json`: provenance index for the recovered baseline results, completed comparison run, and aligned checkpoint metadata
- `latency_comparison.json`: normalized comparison summary with explicit `baseline` / `unaligned` / `aligned_gac` latency and alignment fields
- `README.md`: human-readable summary of the completed PaLU A100 comparison

## Final Measured Result

- `baseline`
  - prefill average latency: `214.61 ms`
  - decode average latency: `1636.75 ms`
  - alignment: `100.0%`
- `unaligned PaLU`
  - prefill average latency: `217.16 ms`
  - decode average latency: `358.80 ms`
  - alignment: `63.44%`
- `aligned GAC`
  - prefill average latency: `214.93 ms`
  - decode average latency: `192.19 ms`
  - alignment: `100.0%`

## Main Takeaway

- GAC checkpoint alignment restores PaLU to full `mod 8` alignment on A100
- Prefill latency changes only slightly: `217.16 ms -> 214.93 ms`
- Decode latency improves substantially: `358.80 ms -> 192.19 ms`
- In throughput terms, decode rises from `637.5 tok/s` to `1409.8 tok/s` over unaligned PaLU

## Comparison Shape

To match the ASVD / LLM-Pruner artifact style, `latency_comparison.json` always exposes:

- `prefill_latency_ms.baseline|unaligned|aligned_gac`
- `decode_latency_ms.baseline|unaligned|aligned_gac`
- `alignment_pct.baseline|unaligned|aligned_gac`

For the current checked-in PaLU evidence:

- `baseline` is reused from the successful baseline portion of job `25269`
- `unaligned` comes from the true non-rounded `rb1` PaLU checkpoint
- `aligned_gac` comes from the repo-native GAC-aligned checkpoint `...-rb1-gac-a100`

## Provenance

- Baseline-only recovered artifact:
  - `results/C5/20260408_102500_palu_latency_compare_retry1_baseline_only/results.json`
- Completed comparison run:
  - `results/C5/20260408_120500_palu_unaligned_vs_gac/`
- Aligned checkpoint build metadata:
  - `results/palu_checkpoints/Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100/build_summary.json`
