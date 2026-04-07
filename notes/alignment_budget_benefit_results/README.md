# Alignment Budget Benefit Results

This folder isolates the smallest repo-tracked artifact set that demonstrates the current alignment budget prototype already produces a useful benefit on the PaLU path.

## Source

- Issue: `12-validate-alignment-benefit`
- Adapter: `src/gcompress_bench/alignment_budget/adapters/palu.py`
- Validation contract:
  - `minimal_alignment = 8`
  - `preferred_alignment = 16`
  - `recommended_values = (96, 112, 128)`
  - `cliff_values = (120,)`

## Included Files

- `palu_example_input.json`: original PaLU `head_wise_ranks` example used for benefit validation.
- `palu_example_aligned.json`: aligned config emitted by the repo-native budget adapter.
- `palu_example_summary.json`: minimal benefit summary showing budget overhead and hardware penalty change.

## Key Takeaway

For this checked-in example, the aligned result keeps total budget overhead at `2.88%` while reducing the latency-proxy hardware penalty from `90.78125` to `0.0`. This is the smallest deterministic artifact bundle proving the prototype is already useful before adding real profiling or more pruning methods.
