# PaLU Operator Split Bundle

This bundle records the checked-in issue-32 operator split for PaLU under the issue-30 fixed-length contract.
It is a structural attribution artifact: it combines the fixed-length rerun with the checkpoint and module contracts to separate SDPA from the changed projection path.

## Main Conclusion

- `HeadwiseLowRankModule` keeps the SDPA shape contract unchanged: both PaLU checkpoints still use `head_dim=128`, `num_attention_heads=32`, and `num_key_value_heads=8`.
- The changed operator family is the attention-adjacent projection GEMM path inside `k_proj` / `v_proj`, not an SDPA width cliff.
- Runtime alignment on the loaded-model view rises from `63.4375%` to `100.0%`.
- Raw config rank alignment rises from `35.9375%` to `100.0%`.
- Under the issue-30 fixed-length rerun, aligned GAC changes prefill throughput by `+2.18%` and decode throughput by `+4.37%` versus unaligned.

## Included Files

- `operator_split_summary.json`: structured SDPA-vs-projection split artifact for issue 32.
- `source_manifest.json`: provenance index for the checkpoint config, issue-30 fixed-length evidence, and `HeadwiseLowRankModule` source contract.
- `README.md`: human-readable summary of the structural split and its caveat.

## Why SDPA Is Not The Changed Path

- `third_party/palu/palu/model/modules/svd_linear.py` defines `HeadwiseLowRankModule` with an internal `VT` projection and per-group `U[*]` reconstruction.
- The module returns `torch.cat(outputs, dim=-1)`, which restores the original `out_features` before attention consumes the tensor.
- The PaLU checkpoint configs therefore keep the same attention width contract as the baseline Llama model, even though the internal low-rank projection path changes.

## Caveat

- This bundle is not a per-kernel profiler trace. It is the checked-in structural attribution result available in the repo today.
- If a future PaLU variant changes the true attention output width rather than only the internal low-rank path, this conclusion must be revisited with a new artifact.

## Provenance

- Fixed-length run results: `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_issue30/results.json`
- Fixed-length run summary: `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_issue30/comparison_summary.json`
- Existing issue-30 bundle summary: `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_comparison.json`
- Unaligned checkpoint config: `results/palu_checkpoints/Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1/config.json`
- Aligned checkpoint config: `results/palu_checkpoints/Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100/config.json`
- Module source: `third_party/palu/palu/model/modules/svd_linear.py`

