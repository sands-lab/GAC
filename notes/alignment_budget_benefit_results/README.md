# Alignment Budget Benefit Results

This folder isolates the smallest repo-tracked artifact set that demonstrates the current alignment budget prototype already produces a useful benefit on the PaLU path, and now also carries non-PaLU method bundles for LLM-Pruner and ASVD.

## Source

- Issue: `12-validate-alignment-benefit`
- Adapter: `src/gcompress_bench/alignment_budget/adapters/palu.py`
- Validation contract:
  - `minimal_alignment = 8`
  - `preferred_alignment = 16`
  - `recommended_values = (96, 112, 128)`
  - `cliff_values = (120,)`

## Additional Method Bundle

- Issue: `15-expand-alignment-benefit`
- Adapter: `src/gcompress_bench/alignment_budget/adapters/llm_pruner.py`
- Contract: `get_hardware_contract("a100")`
- Evidence split:
  - `proxy_summary`: deterministic repo-native alignment search output on a checked-in example width config
  - `paper_*`: existing LLM-Pruner alignment / prefill / quality numbers structured from `Latex/main.tex` and `Latex/slides.tex`

## ASVD Method Bundle

- Issue: `17-asvd-alignment-budget`
- Adapter: `src/gcompress_bench/alignment_budget/adapters/asvd.py`
- Contract: `get_hardware_contract("a100")`
- Evidence split:
  - `proxy_summary`: deterministic repo-native alignment search output on a checked-in ASVD per-projection rank config
  - `paper_*`: existing ASVD alignment / prefill / quality numbers structured from `Latex/main.tex` and `Latex/slides.tex`

## Included Files

- `palu_example_input.json`: original PaLU `head_wise_ranks` example used for benefit validation.
- `palu_example_aligned.json`: aligned config emitted by the repo-native budget adapter.
- `palu_example_summary.json`: minimal benefit summary showing budget overhead and hardware penalty change.
- `llmpruner_example_input.json`: minimal LLM-Pruner per-layer kept-width example used to exercise the non-PaLU adapter.
- `llmpruner_example_aligned.json`: aligned width config emitted by the repo-native LLM-Pruner adapter.
- `llmpruner_example_summary.json`: combines deterministic proxy summary with structured paper-side alignment and prefill latency evidence.
- `asvd_example_input.json`: minimal ASVD per-projection rank example used to exercise the repo-native ASVD adapter.
- `asvd_example_aligned.json`: aligned rank config emitted by the repo-native ASVD adapter.
- `asvd_example_summary.json`: combines deterministic proxy summary with structured paper-side alignment and prefill latency evidence.
- `llmpruner_fixed_length_decode_summary.json`: A100 fixed-length decode rerun summary for `baseline / unaligned / aligned_gac`, normalized onto the shared comparison schema used by issue 24.
- `asvd_fixed_length_decode_summary.json`: A100 fixed-length decode rerun summary for `baseline / unaligned / aligned_gac`, generated from the same fixed-token measurement contract and an ASVD stable-rank compression pass.
- `a100_real_profiling/`: repo-tracked A100 profiling bundle with a provenance manifest and structured summary of the checked-in real-measurement evidence and run commands.
- `palu_a100_results/`: repo-tracked PaLU A100 evidence bundle combining the actual partial baseline run, the historical PaLU failure provenance, the local checkpoint-build metadata, and a comparison-shaped latency summary with explicit baseline / unaligned / aligned status fields.

## Key Takeaway

For this checked-in example, the aligned result keeps total budget overhead at `2.88%` while reducing the latency-proxy hardware penalty from `90.78125` to `0.0`. This is the smallest deterministic artifact bundle proving the prototype is already useful before adding real profiling or more pruning methods.

For LLM-Pruner, the checked-in repo-native example shows the same adapter/search flow can align non-PaLU structural widths, while the structured paper evidence records that the aligned GAC variant improved prefill latency from `137.7 ms` to `88.0 ms` and raised alignment from `83%` to `100%`.

For ASVD, the checked-in repo-native example keeps budget overhead at `1.32%` while cutting the proxy hardware penalty from `173.6875` to `2.0`. The structured paper evidence records the same qualitative story at model scale: alignment rises from `5%` to `100%`, and prefill latency improves from `100.5 ms` to `67.1 ms`.

The fixed-length decode reruns added for issue `24-asvd-llmpruner-fixed-length` do not reproduce a decode-side win for alignment on either non-PaLU method. Under the corrected `min_new_tokens=gen` contract, `llmpruner_fixed_length_decode_summary.json` records `aligned_gac` at `38.97 tok/s` versus `39.31 tok/s` for unaligned (`-0.85%`), while `asvd_fixed_length_decode_summary.json` records `aligned_gac` at `38.82 tok/s` versus `39.30 tok/s` for unaligned (`-1.24%`).

The `a100_real_profiling/` bundle complements these method bundles by consolidating the repository's checked-in A100 measurement protocol, layer-wise profiling conclusions, and profiling command provenance in one place. It also records the remaining gap that raw A100 CSV / NCU outputs are still not versioned in the repo.

The `palu_a100_results/` bundle complements that hardware-level bundle with method-specific provenance for PaLU itself: it now records the recovered baseline artifact from job `25269`, the true unaligned `rb1` PaLU checkpoint, the repo-native GAC-aligned checkpoint build metadata, and the completed A100 comparison showing `baseline / unaligned / aligned_gac` side by side.
