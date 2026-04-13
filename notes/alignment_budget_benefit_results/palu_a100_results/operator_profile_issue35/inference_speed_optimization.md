# Inference Speed Optimization Opportunities For PaLU

This note turns the checked-in operator-profile summary and GEMM root-cause summary into a ranked next-step list.
It is an offline prioritization artifact, not a claim that a new optimization has already been measured on GPU.

## Summary
- The current aligned repair recovers 1.28 ms / 0.71% in `prefill` and 0.20 ms / 0.55% in `decode` versus unaligned PaLU.
- `aligned_gac` still trails `baseline` by 3.11% in `prefill` and 3.12% in `decode`.
- Prefill still dominates the next optimization pass: the selected GEMM view is 48.10% `aten::mm` dispatch, while the recoverable align1/align2 tail is only 3.43 ms. Decode `gemv` micro-tuning stays low leverage because the current tail is only 1.09 ms.

## High Priority

### Prefill dispatch reduction on the projection path
- Target: HeadwiseLowRankModule (`k_proj` / `v_proj` low-rank path)
- Why now: The current 8-aligned repair only recovers a small prefill win, while the selected GEMM view remains dominated by `aten::mm` dispatch rather than by the removable align1/align2 tail.
- Evidence: the selected prefill GEMM view is still 48.10% `aten::mm` dispatch, while the removable align1/align2 tail is only 3.43 ms. The measured aligned gain is 1.28 ms (0.71%), and `aligned_gac` still trails baseline by 3.11%.
- Prototype grouped or fused execution for the `HeadwiseLowRankModule` projection path so prefill pays fewer `aten::mm` launches.
- Audit whether `k_proj` / `v_proj` `VT` + `U[*]` can be materialized as fewer larger GEMMs during prefill instead of many small dispatches.
- Treat success as a dispatch-share reduction, not just an align1/align2-kernel elimination.
- Success signal: Prefill total self CUDA time improves materially beyond the current 0.71% versus unaligned PaLU, and the selected-view dispatch share falls from the current 48.10%.

## Medium Priority

### Profile-guided rank retuning beyond simple 8-alignment
- Target: Alignment-budget search for PaLU `k_proj` / `v_proj` ranks
- Why now: The current repair removes the measurable align1/align2 tail, but a large fraction of that gain is given back by regressions in `other_gemm_kernels`, which means nearest 8-alignment is too coarse as an optimization objective.
- Evidence: alignment-sensitive kernels recover 3.43 ms, but `other_gemm_kernels` gives back 2.37 ms. The coarse prefill GEMM view also still contains 5.87 ms of attention leakage, so candidate scoring must stay root-cause-aware.
- Extend the PaLU path from simple round-to-8 repair to contract-aware candidate search over profiled `recommended_values` and cliff-aware rank choices.
- Score candidates against both eliminated `alignment_sensitive_kernels` and regressions in `other_gemm_kernels` instead of minimizing alignment penalty alone.
- Reuse the repo-native `alignment_budget` machinery for a small DP or global re-allocation pass on the attention-adjacent projection ranks.
- Success signal: A follow-up retuning pass keeps the recovered align1/align2 benefit while reducing the current 2.37 ms give-back in `other_gemm_kernels`.

## Deprioritized

### Decode GEMV micro-tuning
- Why not now: Decode is not where the current PaLU gap is being created. The `gemv` tail is small and alignment barely changes it.
- Evidence: the decode `gemv` tail is only 1.09 ms, and alignment changes that bucket by -0.02 ms. The total decode gain stays at 0.20 ms (0.55%).
- Revisit trigger: Only revisit this path if future traces show a materially larger decode `gemv` tail than the current 1.09 ms, or if decode becomes the dominant residual gap after prefill work.

## Guardrails
- Treat this artifact as offline prioritization, not as proof of a new measured speedup.
- Use the issue-38 bucketed root-cause split when ranking GEMM optimizations; the coarse GEMM family still mixes in attention leakage.
- For PaLU, optimize the attention-adjacent projection path first; do not start from decode-only GEMV micro-tuning.
