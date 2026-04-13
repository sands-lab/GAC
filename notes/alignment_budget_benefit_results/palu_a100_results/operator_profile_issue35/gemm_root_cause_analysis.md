# GEMM Root-Cause Analysis For PaLU Inference

This note is the kernel-level companion to `analysis.md` and `palu_inference_operator_profile_summary.json`.
It re-reads the raw profiler events to explain why the 8-aligned GEMM cliff from simulation does not turn into a large end-to-end inference win.

## Executive Summary
- The coarse GEMM-like view is dispatch-heavy rather than purely kernel-heavy: `aten::mm` contributes 71.02 ms in prefill (48.10%) and 11.67 ms in decode (42.83%).
- Prefill alignment-sensitive kernels are a long tail: `align1/align2` kernels account for 3.43 ms in `palu`, and `aligned_gac` recovers 3.43 ms from that tail, but other already-aligned tensorcore kernels give some of the gain back.
- Decode has no large cliff to recover: the `gemv` tail is only 1.09 ms in `palu`, and `aligned_gac` changes that bucket by -0.02 ms.
- Prefill also contains attention leakage: the flash-attention operator/kernel pair contributes 5.87 ms to the coarse GEMM-like view even though it is attention work, so the older family-level GEMM total overstates the truly alignment-sensitive part.

## Prefill

`palu` prefill is dispatch-heavy rather than purely kernel-heavy: `aten::mm`-style dispatch contributes 71.02 ms (48.10%) of the selected view, while `align1/align2` alignment-sensitive kernels contribute only 3.43 ms. `aligned_gac` removes 3.43 ms from that alignment-sensitive tail, but some of the gain is offset by already-large tensorcore kernels. Prefill also carries 5.87 ms of attention leakage from flash-attention kernels.

| Variant | Selected view (ms) | Dispatch ops | align1/align2 tail | align8 kernels | gemv kernels | attention leakage | other GEMM kernels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 142.12 | 68.29 | 0.00 | 10.14 | 0.00 | 5.79 | 57.90 |
| `palu` | 147.65 | 71.02 | 3.43 | 10.21 | 0.00 | 5.87 | 57.12 |
| `aligned_gac` | 146.00 | 70.16 | 0.00 | 10.42 | 0.00 | 5.92 | 59.49 |

| Comparison | Total delta (ms) | Kernel-only delta (ms) | Largest recovered bucket |
| --- | ---: | ---: | --- |
| `palu` vs `baseline` | -5.53 | -2.80 | `other_gemm_kernels` |
| `aligned_gac` vs `palu` | +1.65 | +0.80 | `alignment_sensitive_kernels` |
| `aligned_gac` vs `baseline` | -3.88 | -2.00 | `alignment_sensitive_kernels` |

Top `palu` -> `aligned_gac` recovered prefill events:
- `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_128x128_64x3_tn_align2>(cutlass_80_tensorop_f16_s16816gemm_f16_128x128_64x3_tn_align2::Params)` (alignment_sensitive_kernels) +1.54 ms: ref 1.54 ms, cand 0.00 ms
- `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn_align1::Params)` (alignment_sensitive_kernels) +1.16 ms: ref 1.16 ms, cand 0.00 ms
- `aten::mm` (dispatch_ops) +0.86 ms: ref 71.01 ms, cand 70.16 ms
- `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align2::Params)` (alignment_sensitive_kernels) +0.34 ms: ref 0.34 ms, cand 0.00 ms
- `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_256x128_64x3_tn_align2>(cutlass_80_tensorop_f16_s16816gemm_f16_256x128_64x3_tn_align2::Params)` (alignment_sensitive_kernels) +0.31 ms: ref 0.31 ms, cand 0.00 ms

Top `palu` -> `aligned_gac` regressed prefill events:
- `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn` (other_gemm_kernels) -0.61 ms: ref 0.37 ms, cand 0.98 ms
- `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn` (other_gemm_kernels) -0.57 ms: ref 0.21 ms, cand 0.78 ms
- `sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel` (other_gemm_kernels) -0.50 ms: ref 1.09 ms, cand 1.60 ms
- `ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn` (other_gemm_kernels) -0.41 ms: ref 34.58 ms, cand 34.99 ms
- `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` (other_gemm_kernels) -0.27 ms: ref 0.03 ms, cand 0.30 ms

## Decode

`palu` decode has no large 8-aligned cliff to recover: the `gemv` tail is only 1.09 ms, while dispatch still contributes 11.67 ms (42.83%) of the selected view. `aligned_gac` changes the `gemv` bucket by -0.02 ms, so decode mostly reshuffles small GEMV variants instead of eliminating a dominant kernel.

| Variant | Selected view (ms) | Dispatch ops | align1/align2 tail | align8 kernels | gemv kernels | attention leakage | other GEMM kernels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 26.27 | 11.18 | 0.00 | 0.00 | 0.68 | 3.91 | 10.50 |
| `palu` | 27.24 | 11.67 | 0.00 | 0.00 | 1.09 | 3.91 | 10.58 |
| `aligned_gac` | 27.12 | 11.60 | 0.00 | 0.00 | 1.10 | 3.92 | 10.50 |

| Comparison | Total delta (ms) | Kernel-only delta (ms) | Largest recovered bucket |
| --- | ---: | ---: | --- |
| `palu` vs `baseline` | -0.97 | -0.48 | `attention_leakage` |
| `aligned_gac` vs `palu` | +0.12 | +0.05 | `other_gemm_kernels` |
| `aligned_gac` vs `baseline` | -0.85 | -0.43 | `alignment_sensitive_kernels` |

Top `palu` -> `aligned_gac` recovered decode events:
- `void gemv2T_kernel_val<int, int, __half, float, float, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)` (gemv_kernels) +0.12 ms: ref 0.12 ms, cand 0.00 ms
- `aten::mm` (dispatch_ops) +0.06 ms: ref 11.67 ms, cand 11.60 ms
- `void cublasLt::splitKreduce_kernel<32, 16, int, float, __half, float, false, __half, __half, __half, true, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __half const*, __half*, __half*, float const*, float const*, __half const*, float const*, __half*, void*, long, float*, int*, float*, float const*, float const*, float const*, float const*)` (other_gemm_kernels) +0.06 ms: ref 0.06 ms, cand 0.00 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) +0.02 ms: ref 0.62 ms, cand 0.59 ms
- `void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 2, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>, float, float)` (gemv_kernels) +0.02 ms: ref 0.02 ms, cand 0.00 ms

Top `palu` -> `aligned_gac` regressed decode events:
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) -0.09 ms: ref 0.05 ms, cand 0.14 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 8, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) -0.07 ms: ref 0.28 ms, cand 0.35 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 5, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) -0.01 ms: ref 0.01 ms, cand 0.02 ms
- `aten::_efficient_attention_forward` (attention_leakage) -0.01 ms: ref 1.95 ms, cand 1.96 ms
- `fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)` (attention_leakage) -0.01 ms: ref 1.95 ms, cand 1.96 ms

## Root Cause

- The dominant runtime is still `aten::mm` dispatch plus already-large tensorcore kernels, not the small `align1/align2` tail that changes most under alignment.
- The 8-aligned repair removes a few obviously suboptimal kernels, but their absolute contribution is too small to create a simulation-style cliff once the rest of inference is included.
- Decode is especially constrained: the `gemv` path is already a small tail, so changing alignment mostly redistributes work among nearby `gemv` kernels instead of deleting a dominant hotspot.
- The older coarse family summary is still directionally useful, but it is not enough for kernel root cause because it mixes dispatch ops, low-level kernels, and prefill attention leakage in the same broad GEMM view.

## Artifact Map

- `gemm_root_cause_summary.json`: machine-readable bucketed summary for dispatch, alignment-sensitive kernels, align8 kernels, GEMV tails, attention leakage, and other GEMM kernels.
- `gemm_root_cause_analysis.md`: human-readable explanation of why the 8-aligned cliff is muted in full inference.
- `analysis.md`: earlier coarse operator-family interpretation kept for stage-level context.
