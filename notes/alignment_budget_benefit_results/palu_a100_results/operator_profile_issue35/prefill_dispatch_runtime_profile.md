# PaLU Prefill Dispatch Runtime Profile

This note compares unaligned `palu` against the same checkpoint with the issue40 grouped reconstruction runtime path enabled as `palu_grouped_bmm`.
Positive delta means `grouped_bmm` reduced self CUDA time relative to `palu`; negative delta means regression.

## Executive Summary
- Prefill: `grouped_bmm` changes prefill total self CUDA time by -0.69% (-1.26 ms) and changes the selected GEMM-like view by 0.66% (+0.98 ms) versus `palu`; dispatch share of the selected view moves from 48.07% to 48.07% (-0.00 pp), with `dispatch_ops` as the largest recovered bucket.
- Decode: `grouped_bmm` changes decode total self CUDA time by -0.10% (-0.04 ms) and changes the selected GEMM-like view by 1.08% (+0.30 ms) versus `palu`; dispatch share of the selected view moves from 42.83% to 42.71% (+0.12 pp), with `gemv_kernels` as the largest recovered bucket.

## Prefill

| Variant | Total self CUDA time (ms) | Selected view (ms) | Dispatch share | Dispatch ops | align1/2/4 tail | align8 kernels | gemv kernels | attention leakage | other GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `palu` | 182.03 | 149.27 | 48.07% | 71.76 | 3.48 | 10.28 | 0.00 | 6.02 | 57.74 |
| `grouped_bmm` | 183.29 | 148.29 | 48.07% | 71.29 | 3.27 | 10.25 | 0.00 | 5.98 | 57.50 |

| Comparison | Total delta (ms) | Total delta (%) | Selected-view delta (ms) | Selected-view delta (%) | Dispatch share delta | Largest recovered bucket |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `grouped_bmm` vs `palu` | -1.26 | -0.69% | +0.98 | 0.66% | -0.00 pp | `dispatch_ops` |

`grouped_bmm` changes prefill total self CUDA time by -0.69% (-1.26 ms) and changes the selected GEMM-like view by 0.66% (+0.98 ms) versus `palu`; dispatch share of the selected view moves from 48.07% to 48.07% (-0.00 pp), with `dispatch_ops` as the largest recovered bucket.

Top `palu` -> `grouped_bmm` recovered prefill events:
- `aten::mm` (dispatch_ops) +2.08 ms: ref 71.75 ms, cand 69.67 ms
- `void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn_align1>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn_align1::Params)` (alignment_sensitive_kernels) +1.17 ms: ref 1.17 ms, cand 0.00 ms
- `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn` (other_gemm_kernels) +0.38 ms: ref 0.38 ms, cand 0.00 ms
- `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align2>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn_align2::Params)` (alignment_sensitive_kernels) +0.34 ms: ref 0.34 ms, cand 0.00 ms
- `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn` (other_gemm_kernels) +0.12 ms: ref 20.93 ms, cand 20.82 ms

Top `palu` -> `grouped_bmm` regressed prefill events:
- `aten::bmm` (dispatch_ops) -1.61 ms: ref 0.00 ms, cand 1.62 ms
- `void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align1::Params)` (alignment_sensitive_kernels) -0.93 ms: ref 0.00 ms, cand 0.93 ms
- `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_nn` (other_gemm_kernels) -0.28 ms: ref 0.00 ms, cand 0.28 ms
- `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align2>(cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align2::Params)` (alignment_sensitive_kernels) -0.26 ms: ref 0.00 ms, cand 0.26 ms
- `void cutlass::Kernel2<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1::Params)` (alignment_sensitive_kernels) -0.11 ms: ref 0.00 ms, cand 0.11 ms

## Decode

| Variant | Total self CUDA time (ms) | Selected view (ms) | Dispatch share | Dispatch ops | align1/2/4 tail | align8 kernels | gemv kernels | attention leakage | other GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `palu` | 37.44 | 27.27 | 42.83% | 11.68 | 0.00 | 0.00 | 1.08 | 3.91 | 10.59 |
| `grouped_bmm` | 37.47 | 26.97 | 42.71% | 11.52 | 0.00 | 0.00 | 0.79 | 3.93 | 10.73 |

| Comparison | Total delta (ms) | Total delta (%) | Selected-view delta (ms) | Selected-view delta (%) | Dispatch share delta | Largest recovered bucket |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `grouped_bmm` vs `palu` | -0.04 | -0.10% | +0.30 | 1.08% | +0.12 pp | `gemv_kernels` |

`grouped_bmm` changes decode total self CUDA time by -0.10% (-0.04 ms) and changes the selected GEMM-like view by 1.08% (+0.30 ms) versus `palu`; dispatch share of the selected view moves from 42.83% to 42.71% (+0.12 pp), with `gemv_kernels` as the largest recovered bucket.

Top `palu` -> `grouped_bmm` recovered decode events:
- `aten::mm` (dispatch_ops) +0.55 ms: ref 11.68 ms, cand 11.13 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 8, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) +0.28 ms: ref 0.28 ms, cand 0.00 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) +0.19 ms: ref 0.61 ms, cand 0.42 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) +0.05 ms: ref 0.05 ms, cand 0.00 ms
- `void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 2, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>, float, float)` (gemv_kernels) +0.02 ms: ref 0.02 ms, cand 0.00 ms

Top `palu` -> `grouped_bmm` regressed decode events:
- `aten::bmm` (dispatch_ops) -0.39 ms: ref 0.00 ms, cand 0.39 ms
- `void gemvNSP_kernel<__half, __half, __half, float, 1, 32, 4, 1024, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) -0.23 ms: ref 0.00 ms, cand 0.23 ms
- `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn` (other_gemm_kernels) -0.14 ms: ref 0.00 ms, cand 0.14 ms
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, false, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>)` (gemv_kernels) -0.01 ms: ref 0.00 ms, cand 0.01 ms
- `aten::_efficient_attention_forward` (attention_leakage) -0.01 ms: ref 1.96 ms, cand 1.97 ms

## Guardrails

- This sidecar compares runtime behavior for the same PaLU checkpoint; it does not change the checkpoint weights themselves.
- The grouped runtime path only targets `HeadwiseLowRankModule` reconstruction on the attention-adjacent projection path.
- The selected root-cause view is intentionally GEMM-focused, so dispatch-share changes should be read together with the total-stage delta.

## Artifact Map

- `prefill_dispatch_runtime_profile_summary.json`: machine-readable runtime comparison between `palu` and `palu_grouped_bmm`.
- `prefill_dispatch_runtime_profile.md`: human-readable explanation of the runtime grouped-reconstruction follow-up.
