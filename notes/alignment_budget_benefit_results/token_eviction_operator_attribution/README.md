# Token Eviction Operator Attribution Bundle

This directory is the repo-tracked landing zone for issue `34-token-eviction-bundle`.
Its scope is intentionally narrow: publish the token-eviction-focused `prefill` GEMM `M`
and `decode` SDPA context-length runs into one reusable bundle, without claiming that the
repository already contains real measured token-eviction results.

## Why This Bundle Exists

- `notes/operator_impact_map.md` already narrows the remaining open question to token-eviction-style `M` / context sensitivity
- `notes/token_eviction_m_sweep.md` and `experiments/token_eviction_m_sweep.yaml` already define the runnable benchmark slices
- what was still missing was a repo-native publish path that turns those raw `run_experiment.py` outputs into a tracked summary and provenance manifest

This issue fills that gap with `scripts/publish_token_eviction_operator_attribution_bundle.py`.

## Input Contract

The bundle expects two runs derived from `experiments/token_eviction_m_sweep.yaml`:

1. `TOKENEVICTION_prefill_gemm_m_real_shapes`
2. `TOKENEVICTION_decode_sdpa_context_real_shapes`

Each run should come from the existing benchmark runner and therefore contain the usual
`raw.json`, `config.json`, `summary.json`, and `env.json` files.

## Publish Command

```bash
python3 scripts/publish_token_eviction_operator_attribution_bundle.py \
  --spec experiments/token_eviction_m_sweep.yaml \
  --prefill-run-dir results/TOKENEVICTION/<prefill-run-id> \
  --decode-run-dir results/TOKENEVICTION/<decode-run-id> \
  --output-dir notes/alignment_budget_benefit_results/token_eviction_operator_attribution
```

## Output Files

- `token_eviction_operator_attribution_summary.json`: structured summary for the token-eviction `prefill` GEMM `M` sweep and `decode` SDPA context-length sweep
- `source_manifest.json`: provenance index pointing back to the published raw/config/summary/env files for each run
- `prefill_gemm_m/`: tracked copies of the source run payloads used by the `prefill` summary
- `decode_sdpa_context/`: tracked copies of the source run payloads used by the `decode` summary

## Interpretation Boundary

This bundle does not claim that token-eviction attribution is already solved end to end.
It only makes the next measured artifact reproducible and easy to cite:

- `prefill` remains a sequence-driven GEMM `M` story
- `decode` remains a context-length SDPA story with fixed `head_dim`
- the generated summary is a publish contract and provenance wrapper, not a substitute for real measured runs

If the repository later adds measured outputs here, those JSON files become the tracked
source of truth for the token-eviction attribution slice.
