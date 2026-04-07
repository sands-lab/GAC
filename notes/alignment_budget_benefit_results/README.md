# Alignment Budget Benefit Results

This folder isolates the smallest repo-tracked artifact set that demonstrates the current alignment budget prototype already produces a useful benefit on the PaLU path, and now also carries a second non-PaLU method bundle for LLM-Pruner.

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

## Included Files

- `palu_example_input.json`: original PaLU `head_wise_ranks` example used for benefit validation.
- `palu_example_aligned.json`: aligned config emitted by the repo-native budget adapter.
- `palu_example_summary.json`: minimal benefit summary showing budget overhead and hardware penalty change.
- `llmpruner_example_input.json`: minimal LLM-Pruner per-layer kept-width example used to exercise the non-PaLU adapter.
- `llmpruner_example_aligned.json`: aligned width config emitted by the repo-native LLM-Pruner adapter.
- `llmpruner_example_summary.json`: combines deterministic proxy summary with structured paper-side alignment and prefill latency evidence.

## Key Takeaway

For this checked-in example, the aligned result keeps total budget overhead at `2.88%` while reducing the latency-proxy hardware penalty from `90.78125` to `0.0`. This is the smallest deterministic artifact bundle proving the prototype is already useful before adding real profiling or more pruning methods.

For LLM-Pruner, the checked-in repo-native example shows the same adapter/search flow can align non-PaLU structural widths, while the structured paper evidence records that the aligned GAC variant improved prefill latency from `137.7 ms` to `88.0 ms` and raised alignment from `83%` to `100%`.
