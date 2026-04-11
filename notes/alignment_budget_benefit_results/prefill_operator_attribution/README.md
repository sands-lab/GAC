# Prefill Operator Attribution Bundle

This directory is the repo-tracked landing zone for issue `31-inference-operator-impact`.
Its scope is intentionally narrow: publish the real-shape `prefill` GEMM attribution runs for ASVD and LLM-Pruner into one reusable bundle, instead of mixing PaLU, token eviction, `prefill`, and `decode` into the same artifact.

## Why This Bundle Exists

- `notes/operator_impact_map.md` already narrows the strongest current hypothesis to ASVD / LLM-Pruner `prefill` GEMM `K/N`
- `notes/prefill_operator_attribution.md` already defines the runnable real-shape experiments in `experiments/operator_attribution_prefill.yaml`
- what was still missing was a repo-native publish path that turns those raw `run_experiment.py` outputs into a tracked summary and provenance manifest

This issue fills that gap with `scripts/publish_prefill_operator_attribution_bundle.py`.

## Input Contract

The bundle expects three runs derived from `experiments/operator_attribution_prefill.yaml`:

1. `ASVD_prefill_gemm_k_real_shapes`
2. `LLMPRUNER_prefill_gateup_n_real_shapes`
3. `LLMPRUNER_prefill_down_k_real_shapes`

Each run should come from the existing runner and therefore contain the usual `raw.json`, `config.json`, `summary.json`, and `env.json` files.

## Publish Command

```bash
python3 scripts/publish_prefill_operator_attribution_bundle.py \
  --spec experiments/operator_attribution_prefill.yaml \
  --asvd-run-dir results/G3/<asvd-run-id> \
  --llmpruner-gateup-run-dir results/G4/<llmpruner-gateup-run-id> \
  --llmpruner-down-run-dir results/G3/<llmpruner-down-run-id> \
  --output-dir notes/alignment_budget_benefit_results/prefill_operator_attribution
```

## Output Files

- `prefill_operator_attribution_summary.json`: structured ASVD checkpoint sweep summary plus LLM-Pruner paired `N/K` checkpoint comparisons
- `source_manifest.json`: provenance index pointing back to the published raw/config/summary/env files for each run
- `asvd/`, `llmpruner_gateup/`, `llmpruner_down/`: tracked copies of the source run payloads used by the summary

## Interpretation Boundary

This bundle does not claim that full operator-level timing is already solved for every method.
It only makes the next `prefill` operator attribution result reproducible and easy to cite:

- ASVD remains a real-shape GEMM `K` checkpoint sweep
- LLM-Pruner remains a paired MLP-width attribution story over `N` and downstream `K`
- PaLU and token-eviction artifacts stay out of scope for this directory

If the repository later adds measured outputs here, those JSON files become the tracked source of truth for this `prefill` slice.
