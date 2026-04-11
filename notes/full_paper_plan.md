# GAC Full Paper Plan

This note upgrades the existing workshop-oriented `notes/paper_plan.md` into a full-paper writing plan grounded in the repository's current reliable evidence.
The rule for this plan is simple: only treat checked-in artifacts and paper-side numbers already present in the repo as established facts; everything else stays in the "Missing Experiments" section.

## Core Thesis

The full paper should argue three linked claims:

1. Dimension-changing compression changes operator-facing dimensions, not just parameter count.
2. The dominant inference penalty is method- and stage-specific, so operator attribution matters more than a single global "alignment is good" slogan.
3. A hardware-constrained dimension re-selection strategy can recover performance without changing the upstream compressor's objective.

Short version:

- ASVD and LLM-Pruner currently have the strongest checked-in evidence on `prefill`
- PaLU motivated the decode investigation, but the corrected fixed-length reruns show that large `decode` gains do not currently hold up
- token eviction remains an important future comparison because it changes a different axis than ASVD / PaLU / LLM-Pruner

## Chapter Plan

### 1. Introduction

What to say:

- Start from the paradox already visible in the repo and paper draft: smaller compressed models can still be slower.
- Position the problem as operator-facing dimension mismatch rather than only a model-size issue.
- Preview that `prefill` and `decode` should not be mixed into one undifferentiated latency story.

Evidence to cite:

- `Latex/main.tex`
- `notes/operator_impact_map.md`

### 2. Background and Problem Setup

What to say:

- Introduce the operator-facing dimensions `M/K/N/head_dim`.
- Explain how ASVD, PaLU, LLM-Pruner, and token eviction each perturb a different part of the inference graph.
- Define why this matters differently for `prefill` and `decode`.

Evidence to cite:

- `Latex/main.tex`
- `notes/operator_impact_map.md`
- `notes/paper_plan.md`

### 3. Full-Stack Analysis of Alignment Constraints

What to say:

- Reuse the three-layer structure from the workshop paper: framework, library, hardware.
- Keep SDPA backend cliffs, cuBLAS kernel tiers, and hardware tile/bandwidth effects, but tighten the narrative around which operators these mechanisms actually touch.
- Separate "microbenchmark cliff exists" from "this cliff dominates end-to-end inference".

Evidence to cite:

- `Latex/main.tex`
- `notes/alignment_budget_benefit_results/a100_real_profiling/profiling_summary.json`

### 4. Method-to-Operator Attribution

What to say:

- Add the operator map as a new bridge section that the workshop version does not fully spell out.
- Show that ASVD mainly stresses GEMM `K`, LLM-Pruner mainly stresses MLP GEMM `N/K`, PaLU can stress `head_dim` and attention-adjacent GEMMs, and token eviction changes sequence-driven `M`.
- Use this section to justify why later experiments must be method-specific instead of one-size-fits-all.

Evidence to cite:

- `notes/operator_impact_map.md`
- `notes/alignment_budget_benefit_results/README.md`

### 5. GAC Method

What to say:

- Keep the compressor-agnostic framing and the constrained budget-allocation story.
- Emphasize that GAC is not a padding trick and not a serving-side workaround.
- Clarify that the candidate set should be chosen from measured operator constraints, not abstract alignment folklore.

Evidence to cite:

- `Latex/main.tex`
- `src/gcompress_bench/alignment_budget/`

### 6. End-to-End Evaluation

What to say:

- Split evaluation by method and by stage.
- Present `prefill` as the strongest established result today.
- Present corrected fixed-length `decode` results as a negative but important finding: some earlier decode-side gains do not survive stricter measurement contracts.

Evidence to cite:

- `notes/alignment_budget_benefit_results/a100_real_profiling/profiling_summary.json`
- `notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_comparison.json`
- `notes/alignment_budget_benefit_results/asvd_fixed_length_decode_summary.json`
- `notes/alignment_budget_benefit_results/llmpruner_fixed_length_decode_summary.json`

### 7. Discussion

What to say:

- Explain what the current repo evidence already proves and what it still does not.
- Be explicit that operator attribution is still incomplete for PaLU and token eviction.
- Frame the negative `decode` reruns as a refinement of the paper's claim, not a failure.

### 8. Limitations and Future Work

What to say:

- No fully checked-in raw A100 CSV / NCU dumps yet
- No repo-tracked token eviction attribution artifact yet
- No full operator-level decomposition of PaLU `prefill` vs `decode`
- Limited H100 scope today

## Comparison Matrix

| Comparison | Why it matters | Current status | Best chapter |
| --- | --- | --- | --- |
| Baseline vs ASVD vs ASVD-GAC | strongest checked-in evidence that misaligned low-rank compression can erase expected speedup | ready from paper-side and repo summaries | Evaluation |
| Baseline vs LLM-Pruner vs LLM-Pruner-GAC | shows that even relatively coarse structured pruning still suffers from alignment penalties | ready from paper-side and repo summaries | Evaluation |
| Baseline vs PaLU vs PaLU-GAC | keeps the motivating attention-width case in scope, but should use corrected fixed-length language | partially ready; current decode story is weaker than before | Evaluation + Discussion |
| `prefill` vs `decode` within each method | prevents overgeneralizing one stage's behavior to the other | partially ready; strongest for negative decode finding | Method-to-Operator Attribution + Evaluation |
| A100 vs H100 | shows whether the operator story generalizes across GPU generations | partial; H100 contract exists but full evidence is still thinner | Discussion / Appendix |
| GAC vs naive round-to-8 | clarifies whether the main benefit comes from global budget reallocation or simple local rounding | not yet ready as a checked-in end-to-end comparison | Missing Experiments |
| GAC vs serving-side mitigation | distinguishes compression-time alignment from runtime padding / fallback handling | conceptual argument exists; measured comparison still missing | Discussion |
| Compression families vs token eviction | broadens the claim beyond `K/N/head_dim` changes | not yet ready | Missing Experiments |

## Missing Experiments

### Must-have for a stronger full paper

1. Operator-attribution artifact for ASVD / LLM-Pruner `prefill`
   - Goal: prove whether the current method-level speedups are really dominated by large GEMM `K/N`
   - Comparison: unaligned vs GAC-aligned, broken down by operator family or layer group

2. Optional PaLU per-kernel timing follow-up after the structural operator split
   - Current checked-in result: issue 32 already shows the corrected fixed-length contract leaves SDPA shape unchanged and that the changed operator family is the `k_proj` / `v_proj` low-rank projection path
   - Remaining comparison: if a stronger paper figure is still needed, isolate `VT` / `U[*]` projection kernels directly rather than reopening SDPA head-dim sweeps

3. Naive rounding baseline
   - Goal: show why GAC is better than "just round everything to 8"
   - Comparison: unaligned vs nearest-round vs GAC

4. Token eviction line
   - Goal: test whether changing sequence-driven `M` produces a different dominant bottleneck story
   - Comparison: token-eviction method vs GAC-style constrained variant or at least an operator sweep

### Nice-to-have

1. Stronger H100 appendix with measured operator tables
2. More downstream task coverage beyond the current lightweight quality checks
3. Explicit serving-system discussion with vLLM / TensorRT-LLM style mitigation comparisons

## Evidence Status by Method

| Method | What is reliable now | What is still missing |
| --- | --- | --- |
| ASVD | strong `prefill` story; issue-30 fixed-length `decode` rerun still shows no win | operator-level `prefill` attribution |
| LLM-Pruner | strong `prefill` story; issue-30 fixed-length `decode` rerun now shows a moderate aligned gain | replication, MLP-only operator attribution, and naive-round baseline |
| PaLU | strong motivation for `head_dim` cliffs; corrected `decode` story is now conservative | per-operator timing and stronger `prefill` attribution |
| token eviction | conceptual fit is strong because it changes `M` | almost all checked-in evaluation evidence is still missing |

## Writing Order

1. Freeze the thesis and chapter skeleton using this note.
2. Rework the workshop introduction and analysis sections into the full-paper structure.
3. Insert the new method-to-operator attribution section before rewriting evaluation.
4. Rewrite evaluation so that `prefill` and `decode` are always reported separately.
5. Fill missing-experiment placeholders only after their artifacts are checked in.
6. Write discussion and limitations last, using the exact current evidence boundary.

## Suggested Main Figures / Tables

1. One overview figure mapping compression family to changed operator dimension (`M/K/N/head_dim`)
2. Existing full-stack microbenchmark figure(s) from the workshop version
3. Main end-to-end table: baseline / unaligned / GAC for ASVD, LLM-Pruner, PaLU
4. `prefill` vs `decode` comparison table with corrected measurement contracts
5. One future-work / evidence-status table summarizing what is already proven and what still needs experiments

## Submission Path

- Short-term target: convert the current workshop narrative into a full-paper outline that is internally consistent with the corrected fixed-length `decode` findings
- Mid-term target: add the missing operator-attribution artifacts before locking the evaluation section
- Final target: make the paper's main claim precise enough that negative `decode` results strengthen the story instead of weakening it
