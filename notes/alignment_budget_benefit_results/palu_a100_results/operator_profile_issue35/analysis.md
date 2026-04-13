# PaLU Inference Operator Profile Analysis

This note is the human-readable companion to `palu_inference_operator_profile_summary.json`.
It compares `baseline`, `palu`, and `aligned_gac` using self CUDA time aggregated by operator family.
Positive `delta_ms` means the candidate reduced self CUDA time relative to the reference; negative `delta_ms` means regression.

## Executive Summary
- `palu` is slower than `baseline` in both stages: prefill adds 6.68 ms (3.85%) and decode adds 1.33 ms (3.68%). In both stages the dominant regression family is `gemm`.
- Aligned GAC recovers 1.28 ms (0.71%) in prefill and 0.20 ms (0.55%) in decode versus `palu`. The largest recovered family is `gemm` in both stages.
- `aligned_gac` still trails `baseline` by 5.40 ms (3.11%) in prefill and 1.12 ms (3.12%) in decode, so alignment closes only part of the compressed-model gap.

## Bundle Takeaways
- Prefill: Aligned GAC changes prefill total self CUDA time by 0.71% versus unaligned PaLU; largest recovered family: gemm.
- Decode: Aligned GAC changes decode total self CUDA time by 0.55% versus unaligned PaLU; largest recovered family: gemm.

## Prefill

| Variant | Total self CUDA time (ms) | GEMM | Elementwise | Data movement | SDPA | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 173.49 | 80.25% | 11.62% | 4.03% | 0.00% | 4.11% |
| `palu` | 180.17 | 80.32% | 11.27% | 4.43% | 0.00% | 3.98% |
| `aligned_gac` | 178.89 | 79.96% | 11.43% | 4.50% | 0.00% | 4.11% |

| Comparison | Total delta (ms) | Delta (%) | Key family delta |
| --- | ---: | ---: | --- |
| `palu` vs `baseline` | -6.68 | -3.85% | `gemm` -5.49 ms |
| `aligned_gac` vs `palu` | +1.28 | 0.71% | `gemm` +1.68 ms |
| `aligned_gac` vs `baseline` | -5.40 | -3.11% | `gemm` -3.81 ms |

GEMM stays between 79.96% and 80.32% of prefill self CUDA time across all three variants, so the aligned gain is mostly a GEMM recovery: `aligned_gac` pulls back 1.68 ms of GEMM time versus `palu`, but it still trails `baseline` by 5.40 ms overall.

## Decode

| Variant | Total self CUDA time (ms) | GEMM | Elementwise | Data movement | SDPA | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 36.08 | 61.98% | 9.26% | 13.50% | 5.42% | 9.85% |
| `palu` | 37.41 | 62.38% | 8.82% | 13.96% | 5.22% | 9.61% |
| `aligned_gac` | 37.20 | 62.38% | 8.77% | 14.03% | 5.26% | 9.55% |

| Comparison | Total delta (ms) | Delta (%) | Key family delta |
| --- | ---: | ---: | --- |
| `palu` vs `baseline` | -1.33 | -3.68% | `gemm` -0.98 ms |
| `aligned_gac` vs `palu` | +0.20 | 0.55% | `gemm` +0.13 ms |
| `aligned_gac` vs `baseline` | -1.12 | -3.12% | `gemm` -0.85 ms |

Decode remains projection-dominated: GEMM stays between 61.98% and 62.38%, while SDPA is only 5.22% to 5.42%. That means the small aligned gain is not coming from the attention kernel itself; `aligned_gac` recovers 0.13 ms of GEMM time versus `palu`, but still sits 1.12 ms behind `baseline`.

## Artifact Map

- `palu_inference_operator_profile_summary.json`: machine-readable per-stage totals, family shares, and pairwise comparisons.
- `source_manifest.json`: provenance for the source run directories and copied artifacts.
- `submission_status.json`: Slurm provenance and job-tracking state for the real A100 collection.
