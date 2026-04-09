# LLM Operator Impact Map

This note turns the current paper draft, repo-tracked profiling bundles, and recent fixed-length reruns into one question-oriented map:
when compression changes model dimensions, which operator and which inference stage should we expect to matter most?

We use `M/K/N` and `head_dim` as the operator-facing dimensions:

- `M`: sequence-length or token-count axis seen by GEMM/GEMV
- `K`: reduction / inner dimension of GEMM
- `N`: output width / channel dimension of GEMM
- `head_dim`: per-head attention width seen by SDPA / FlashAttention

## Compression-to-Operator Mapping

| Compression family | Repo methods | Primary changed dimension | Operator manifestation | Most likely affected stage |
| --- | --- | --- | --- | --- |
| Low-rank factorization | ASVD, SVD-style paths | rank `r` | changes GEMM `K` in the factorized `A @ B` path; can also change neighboring projection widths | mostly `prefill`, also `decode` |
| Head-wise low-rank factorization | PaLU | per-head rank / effective attention width | changes attention-side `head_dim` and surrounding projection GEMMs | `decode` risk is high in theory; `prefill` and `decode` both need measurement |
| Structured pruning | LLM-Pruner | kept hidden width / neuron count | shrinks MLP GEMM `N` and the paired downstream `K` | mostly `prefill` |
| Token Eviction / KV eviction | H2O, PyramidKV-style methods | active sequence length | changes GEMM/GEMV `M` and attention context length | `decode` first, then long-context `prefill` |

## Current Evidence Base

### 1. Framework-layer cliffs are real, but not every cliff dominates end-to-end runtime

`Latex/main.tex` shows that SDPA has a hard fast-path requirement at `head_dim % 8 == 0`, and the A100 profiling bundle records an example where `d=129` is about `90%` slower than `d=128`.
The same bundle also records template-tier cliffs around 32-dimension boundaries inside FlashAttention-2.

Interpretation:

- `head_dim` misalignment is a real cliff and should not be ignored for methods that directly perturb attention width.
- But a strong kernel cliff is not enough by itself to prove it is the dominant end-to-end bottleneck for a given method or stage.

### 2. The strongest current end-to-end evidence points to prefill GEMM `K/N`, not decode

The most stable repo-tracked method-level evidence today comes from A100 prefill results summarized in `notes/alignment_budget_benefit_results/a100_real_profiling/profiling_summary.json`:

- ASVD: unaligned prefill latency `100.5 ms` vs aligned `67.1 ms`
- LLM-Pruner: unaligned prefill latency `137.7 ms` vs aligned `88.0 ms`

Those two methods primarily perturb GEMM-facing dimensions:

- ASVD changes low-rank middle dimensions, so the dominant operator hypothesis is GEMM `K`
- LLM-Pruner changes MLP width, so the dominant operator hypothesis is GEMM `N` and the paired downstream `K`

The same profiling summary states that A100 GEMM pays up to roughly `30%` penalty on misaligned `K/N`, while hardware-level throughput can drop from `160-175 TFLOPS` to `50-110 TFLOPS` on misaligned dimensions.

Interpretation:

- The highest-confidence hotspot today is large prefill GEMM on compressed projection / MLP paths.
- For the current repo evidence, GEMM `K/N` is more strongly supported than SDPA as the main source of end-to-end slowdown.

## Inference Hotspots

### Prefill

Current priority ranking for `prefill`:

1. GEMM `K/N` on projection and MLP paths
2. GEMM `M` only when sequence length crosses heuristic cliff regions
3. SDPA `head_dim` for methods that directly perturb attention width

Why:

- `prefill` executes large dense matmuls over the full prompt, so GEMM penalties compound over long sequence length.
- `Latex/main.tex` explicitly states that the alignment penalty grows with sequence length because longer prompts push GEMMs deeper into the compute-bound regime.
- ASVD and LLM-Pruner already provide end-to-end evidence that fixing alignment restores large `prefill` gains.

### Decode

Current priority ranking for `decode`:

1. SDPA `head_dim` and template-tier cliffs for attention-width-changing methods
2. attention-adjacent projection GEMMs
3. GEMM `M` / context-length effects for token-eviction-style methods

Why:

- `decode` repeatedly pays for attention over the KV cache, so SDPA cliffs remain the first mechanism to audit for PaLU-like methods.
- However, the corrected fixed-length reruns weaken the claim that decode is the dominant source of alignment benefit in the current repo evidence.

Measured fixed-length `decode` evidence:

- PaLU fixed-length rerun: aligned is only about `0.1%` faster than unaligned in decode throughput, while prefill is about `1.0%` faster
- ASVD fixed-length rerun: aligned `38.37 tok/s` vs unaligned `38.84 tok/s` (`-1.21%`)
- LLM-Pruner fixed-length rerun: aligned `38.10 tok/s` vs unaligned `38.58 tok/s` (`-1.25%`)

Interpretation:

- `decode` is not currently the best-supported explanation for why aligned variants win end-to-end.
- For PaLU-like methods, SDPA still has the sharpest per-operator cliff, but the corrected repo-tracked end-to-end comparisons no longer show a large decode-side gain.

### Working Conclusion

If the question is "what should we investigate first to explain model-level inference impact?", the current answer is:

1. Start from `prefill` GEMM on compressed projection / MLP paths
2. Treat SDPA `head_dim` as a method-specific cliff analysis, especially for PaLU and future attention-width-changing methods
3. Treat token-eviction and other `M`-changing methods as a separate line, because their dominant risk is likely sequence-length-driven operator behavior rather than the same `K/N` story

## Method-by-Method Bottleneck Hypotheses

| Method family | Primary operator hypothesis | Why this is the current best hypothesis | Confidence |
| --- | --- | --- | --- |
| ASVD | GEMM `K` during factorized projections | strongest checked-in speedup is prefill; method changes the low-rank middle dimension directly | high |
| LLM-Pruner | MLP GEMM `N/K` | method only prunes MLP widths; checked-in prefill slowdown and recovery are large | high |
| PaLU | SDPA `head_dim` plus attention-adjacent GEMMs | method perturbs attention-side dimensions, but current fixed-length reruns do not show a large decode win | medium |
| Token eviction | attention context length and GEMM `M` | paper plan identifies `M` as the changed axis, but the repo lacks a checked-in end-to-end attribution artifact | low |

## Experiment Matrix

The next experiments should answer operator attribution directly instead of only comparing model variants end to end.

| Question | Method family | Operator to isolate | Dimension to sweep | Stage | Existing evidence | Missing artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Is the ASVD slowdown mostly projection GEMM? | ASVD | GEMM | `K` around real ranks | `prefill` | A100 prefill win is large; fixed-length `decode` win is absent | per-layer or per-op prefill attribution table |
| Does LLM-Pruner slow down mainly in MLP blocks? | LLM-Pruner | GEMM/GEMV | `N` and paired `K` | `prefill` | paper-side result already shows large prefill penalty | MLP-only operator attribution bundle |
| Does PaLU really pay mostly through SDPA cliffs after the contract fix? | PaLU | SDPA + projection GEMM | `head_dim` and nearby aligned values | `decode` then `prefill` | SDPA cliff is strong in microbenchmarks; end-to-end decode gain is weak after fix | split operator timing for SDPA vs projection |
| When token count changes, is the dominant effect on attention or GEMM? | token eviction / KV eviction | SDPA + GEMM | `M` / context length | `decode` and long-context `prefill` | only planning evidence exists today | token-eviction-specific operator sweep |

## Recommended Issue Order

To keep the next backlog slices concrete:

1. Build a repo-tracked ASVD/LLM-Pruner prefill operator-attribution artifact focused on GEMM `K/N`
2. Build a PaLU operator-split artifact that times SDPA separately from projection GEMMs under the fixed-length contract
3. Add a token-eviction-oriented `M`-sweep plan or artifact, because that line likely follows a different bottleneck story

## Deliverable Boundary

This note is intentionally a mapping and experiment-planning artifact, not a claim that the repo already contains full operator-level attribution.
Its job is to narrow the next question:

- not "are aligned dimensions good?"
- but "which operator and which stage actually create the observed model-level speed difference?"

Based on the currently checked-in evidence, the first operator to investigate is large-`prefill` GEMM on compressed `K/N` dimensions.
