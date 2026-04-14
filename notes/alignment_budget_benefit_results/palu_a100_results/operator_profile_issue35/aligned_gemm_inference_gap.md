# Aligned GEMM Inference Gap Analysis

This note bridges `gemm_root_cause_summary.json` and `prefill_dispatch_runtime_profile_summary.json`.
It explains why the aligned GEMM cliff observed in simulation still turns into a muted end-to-end inference win on the checked-in PaLU path.
It is a sidecar analysis of checked-in summaries, not a measured GPU speedup.

## Executive Summary
- Prefill: `aligned_gac` only closes part of the prefill gap because it removes 3.43 ms of `align1/align2` tail, but still gives back 2.37 ms in `other_gemm_kernels` while the selected view remains dispatch-heavy at 48.10%. The `grouped_bmm` runtime path recovers 0.98 ms inside the selected view, but adds new costs such as `aten::bmm`, `void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align1::Params)`, so total-stage time still moves by -1.26 ms.
- Decode: `decode` has no large aligned cliff left to cash in: `grouped_bmm` still recovers 0.30 ms inside the selected view, but the total-stage delta stays -0.04 ms once new costs such as `aten::bmm`, `void gemvNSP_kernel<__half, __half, __half, float, 1, 32, 4, 1024, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` are included.

## Prefill

| Comparison | Stage-total delta | Selected-view delta | Dispatch signal | Main give-back |
| --- | ---: | ---: | ---: | --- |
| `aligned_gac` vs `palu` | +1.28 ms | +1.65 ms | selected view still 48.10% dispatch | `alignment_sensitive_kernels` + `other_gemm_kernels` give-back 2.37 ms |
| `grouped_bmm` vs `palu` | -1.26 ms | +0.98 ms | dispatch share delta -0.00 pp | `dispatch_ops` plus new costs |

- `aligned_gac` root-cause signals:
  - alignment tail recovered: 3.43 ms
  - other GEMM give-back: 2.37 ms
  - attention leakage inside the coarse selected view: 5.87 ms

Top new grouped-runtime regressions in prefill:
- `aten::bmm` -1.61 ms: ref 0.00 ms, cand 1.62 ms
- `void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align1::Params)` -0.93 ms: ref 0.00 ms, cand 0.93 ms
- `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_nn` -0.28 ms: ref 0.00 ms, cand 0.28 ms
- `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align2>(cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align2::Params)` -0.26 ms: ref 0.00 ms, cand 0.26 ms
- `void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1::Params)` -0.11 ms: ref 0.00 ms, cand 0.11 ms

Top grouped-runtime recovered prefill events:
- `aten::mm` +2.08 ms: ref 71.75 ms, cand 69.67 ms
- `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn_align1::Params)` +1.17 ms: ref 1.17 ms, cand 0.00 ms
- `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn` +0.38 ms: ref 0.38 ms, cand 0.00 ms
- `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align2::Params)` +0.34 ms: ref 0.34 ms, cand 0.00 ms
- `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn` +0.12 ms: ref 20.93 ms, cand 20.82 ms

## Decode

| Comparison | Stage-total delta | Selected-view delta | Dispatch signal | Main give-back |
| --- | ---: | ---: | ---: | --- |
| `aligned_gac` vs `palu` | +0.20 ms | +0.12 ms | selected view still 42.83% dispatch | `other_gemm_kernels` + `other_gemm_kernels` give-back 0.00 ms |
| `grouped_bmm` vs `palu` | -0.04 ms | +0.30 ms | dispatch share delta +0.12 pp | `gemv_kernels` plus new costs |

Top new grouped-runtime regressions in decode:
- `aten::bmm` -0.39 ms: ref 0.00 ms, cand 0.39 ms
- `void gemvNSP_kernel<__half, __half, __half, float, 1, 32, 4, 1024, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` -0.23 ms: ref 0.00 ms, cand 0.23 ms
- `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` -0.14 ms: ref 0.00 ms, cand 0.14 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, false, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` -0.01 ms: ref 0.00 ms, cand 0.01 ms

Top grouped-runtime recovered decode events:
- `aten::mm` +0.55 ms: ref 11.68 ms, cand 11.13 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 8, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` +0.28 ms: ref 0.28 ms, cand 0.00 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` +0.19 ms: ref 0.61 ms, cand 0.42 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` +0.05 ms: ref 0.05 ms, cand 0.00 ms
- `void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 2, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>, float, float)` +0.02 ms: ref 0.02 ms, cand 0.00 ms

## Guardrails

- This analysis reuses checked-in summary artifacts; it does not claim a new measured GPU speedup.
- `aligned_gac` and `grouped_bmm` should be read as separate follow-up probes: one changes checkpoint ranks, the other changes the runtime path.
- The dominant story is still `dispatch` plus give-back in `other_gemm_kernels`; eliminating the old `align1/align2` tail alone is not enough.

## Artifact Map

- `aligned_gemm_inference_gap_summary.json`: machine-readable bridge between the aligned root-cause summary and the grouped-runtime sidecar summary.
- `aligned_gemm_inference_gap.md`: human-readable explanation of why selected-view GEMM gains still fail to turn into a large end-to-end win.
