# Pruning Alignment Framework

## Problem Statement

Current compression methods in this repo optimize for parameter or importance budget first, but hardware alignment only appears as an afterthought. That mismatch already shows up in existing results:

- `report.md` shows that non-8-aligned dimensions can erase compression gains and even make models slower than baseline.
- `scripts/gac_rank_allocation.py` already contains a DP-style aligned re-allocation path for PaLU/SVD-style ranks.
- `report.md` also shows that `LLM-Pruner` with `round_to=8` can recover speed without measurable quality loss.

The missing piece is a unified framework that can sit between "method-specific budget output" and "hardware-aware executable shape", so pruning methods do not need to natively understand alignment rules.

The target question is:

How do we transform per-layer budgets produced by diverse pruning methods into hardware-friendly dimensions such as `8-aligned`, while preserving the original method's accuracy intent and staying inside a global resource budget?

## Design Goals

1. Support multiple method families, not just one implementation.
2. Keep the original method-specific importance signal intact instead of replacing it with a pure systems heuristic.
3. Make hardware constraints explicit and pluggable.
4. Support both local repair and global budget re-allocation.
5. Separate "what the method wants" from "what the hardware prefers".
6. Produce artifacts that can be consumed by downstream code paths such as `PaLU`, `ASVD`, and pruning pipelines.

## Non-Goals

1. This framework is not a new pruning algorithm by itself.
2. It does not try to cover training-time structured pruning or fine-tuning pipelines.
3. It does not assume every compression method should be converted into the same tensor format.

## Design Constraints From Existing Repo

### Hardware constraints

- Minimal safe target is `8-aligned` for vectorized loading and many SDPA/GEMM fast paths.
- Better targets may be `16-aligned` or selected predefined values such as `96`, `112`, `128`, depending on operator behavior.
- Some dimensions have cliff behavior; the adapter should avoid "legal but slow" values when profiling data says so.

### Method constraints

- `PaLU` and other low-rank methods emit per-layer or per-head ranks.
- `ASVD` emits rank-like middle dimensions with existing aligned vs unaligned variants in this repo.
- `LLM-Pruner` changes structural dimensions more directly and already demonstrates a simple `round_to=8` path.
- Some methods expose continuous budgets; others expose discrete kept channels or kept heads.

### Repo constraints

- Existing architecture does not allow `third_party` code to depend on `src/`.
- Therefore the framework should live in the main repo orchestration layer, and only export method-neutral outputs into downstream method-specific code.

## Proposed Architecture

The framework should be a four-stage pipeline:

1. `Method Adapter`
2. `Budget Normalizer`
3. `Alignment Search Engine`
4. `Materializer`

High-level flow:

```text
method output
  -> method adapter
  -> normalized budget items
  -> hardware constraints + profile table
  -> aligned candidate search
  -> global budget reconciliation
  -> method-specific materialization
  -> executable compressed model / config
```

### Stage 1: Method Adapter

Each compression method exports a common intermediate representation instead of directly emitting final executable dimensions.

Example normalized item:

```python
{
    "layer": "model.layers.12.self_attn.k_proj",
    "axis": "out_features",
    "budget_kind": "rank",
    "original_budget": 107,
    "importance": 0.83,
    "granularity": "per_group",
    "group_count": 2,
    "full_budget": 512,
    "shape_role": "attention_kv",
}
```

### Stage 2: Budget Normalizer

Different methods produce different raw budget semantics. The normalizer converts them into a shared format:

- low-rank methods: `rank`
- structured pruning: `kept_channels`
- head pruning: `kept_heads`
- token methods: `effective_seq_len`

This stage is also responsible for turning method-specific budget totals into a comparable global cost function:

- parameter budget
- FLOP budget
- KV-cache budget
- latency proxy budget

### Stage 3: Alignment Search Engine

This is the core of the framework. For each normalized item:

1. build candidate aligned dimensions near the original budget
2. remove candidates that violate method or shape constraints
3. estimate each candidate's cost and accuracy deviation
4. solve a global constrained optimization problem

The search engine should support three modes:

- local repair: nearest legal aligned value
- greedy global repair: adjust least sensitive items first
- DP / multi-choice knapsack: globally optimal discrete assignment under total budget

The DP path is already consistent with the logic demonstrated in `scripts/gac_rank_allocation.py`.

### Stage 4: Materializer

Convert aligned budgets back into method-specific artifacts:

- `PaLU`: `head_wise_ranks`
- `ASVD`: aligned rank config for target linear layers
- `LLM-Pruner`: rounded kept-channel config or mask shape targets
- future methods: adapter-owned output format

This stage should be the only place where method-specific schema leaks back in.

## Budget Adapter Interface

The framework should expose a small interface instead of a monolithic pipeline.

```python
class BudgetAdapter:
    method_name: str

    def export_items(self, model, method_state) -> list[dict]:
        ...

    def candidate_set(self, item: dict, hardware_contract: dict) -> list[int]:
        ...

    def materialize(self, aligned_items: list[dict], model, method_state):
        ...
```

Recommended companion objects:

```python
class HardwareContract:
    minimal_alignment: int
    preferred_alignment: int
    recommended_values: tuple[int, ...]
    cliff_table: dict


class BudgetObjective:
    total_budget: int
    budget_metric: str  # params, flops, kv_bytes, latency_proxy
```

The important point is that the adapter owns method semantics, while the framework owns alignment and optimization semantics.

## Alignment Search Strategy

### Candidate generation

Given an original budget `r`, generate a small candidate set:

- nearest lower aligned value
- nearest higher aligned value
- nearest preferred aligned value
- optionally nearest profiled fast-path value

Example for `r=107`:

- local candidates: `104`, `112`, `128`
- remove values that exceed method-specific max overhead or violate granularity

### Scoring

Each candidate should receive a composite score with two parts:

1. accuracy proxy
2. hardware benefit

A practical initial form:

```text
score(candidate) =
    alpha * importance_weight * abs(candidate - original_budget)
  + beta  * latency_penalty(candidate, operator_role)
```

For methods like `LLM-Pruner`, the deviation term should reflect channel removal or recovery. For methods like `PaLU` and `ASVD`, it naturally maps to rank deviation.

### Global reconciliation

After local scoring, solve for global feasibility:

- exact budget match if possible
- otherwise bounded-overhead match under a user-specified tolerance

This is where DP is strongest for discrete rank/channel candidates.

### Failure-safe fallback

If the search space is too large or profiling data is missing:

- fall back to nearest `8-aligned`
- preserve monotonic budget sanity
- emit a warning that the result is heuristic rather than globally optimized

## Method Adapters

### PaLU

- Input: per-layer `head_wise_ranks`
- Budget kind: per-group low-rank dimension
- Materialization target: `head_wise_ranks` config or SVD decomposition input
- Special note: the aligned rank must be compatible with `HeadwiseLowRankModule` group structure

### ASVD

- Input: per-layer sensitivity-derived rank
- Budget kind: low-rank middle dimension
- Materialization target: aligned ASVD rank config
- Opportunity: reuse the same DP machinery already used for SVD-style rank search

### LLM-Pruner

- Input: kept channels or structured pruning masks
- Budget kind: kept structural width
- Materialization target: rounded structural dimensions, such as `round_to=8`
- Existing evidence: `report.md` already shows that `LLM-Pruner` plus `8-aligned` rounding preserves quality while recovering latency

### Future Methods

The same interface should also cover:

- head pruning
- MoE expert pruning with aligned expert hidden sizes
- token eviction when the changed dimension affects hardware-visible sequence tiles

The adapter only needs to explain:

1. what the budget variable means
2. what aligned candidates are legal
3. how to write the aligned result back

## Data Model

A minimal normalized item schema should include:

```python
{
    "name": str,
    "method": str,
    "budget_kind": str,
    "operator_role": str,
    "original_budget": int,
    "candidate_budgets": list[int],
    "importance": float,
    "cost_per_unit": int,
    "granularity": str,
    "constraints": dict,
}
```

This is enough to drive both greedy and DP-based solvers.

## Implementation Sketch In This Repo

Recommended placement:

- `src/gcompress_bench/alignment_budget/contract.py`
- `src/gcompress_bench/alignment_budget/search.py`
- `src/gcompress_bench/alignment_budget/adapters/palu.py`
- `src/gcompress_bench/alignment_budget/adapters/asvd.py`
- `src/gcompress_bench/alignment_budget/adapters/llm_pruner.py`
- `scripts/` keeps experiment runners and offline analysis

Dependency direction:

- `src/` can depend on profiling tables and repo-native utilities
- `third_party/` remains isolated
- final materialization into `third_party` configs should happen through exported JSON/config artifacts, not reverse imports

## Evaluation Plan

### Functional evaluation

1. Verify every adapter preserves total budget within tolerance.
2. Verify all emitted dimensions are `8-aligned` or satisfy the chosen hardware contract.
3. Verify method-specific outputs can be consumed by downstream pipelines.

### Systems evaluation

1. Compare unaligned vs nearest-rounding vs global DP allocation.
2. Measure aligned ratio, average latency penalty, and actual prefill/decode speed.
3. Reuse existing profiling assets such as `results/alignment_sweep.csv`.

### Quality evaluation

1. Compare perplexity and task accuracy before and after alignment adaptation.
2. Focus on pairs already present in the repo:
   - `PaLU`
   - `ASVD`
   - `LLM-Pruner`

### Ablations

1. `8-aligned` vs `16-aligned`
2. local round vs global DP
3. no cliff filtering vs cliff-aware candidate filtering
4. parameter-budget objective vs latency-aware objective

## Suggested Milestones

1. Adapter-only prototype for `LLM-Pruner` using existing `round_to=8` behavior.
2. Generalized normalized-item pipeline for `PaLU` and `ASVD`.
3. Shared global solver using existing GAC DP formulation.
4. End-to-end experiment driver that emits both quality and latency comparisons.

## Open Questions

1. Should the framework optimize for parameter budget, FLOP budget, or an empirical latency proxy by default?
2. For pruning methods, is the best action always "round kept width upward", or should some layers round down and compensate elsewhere?
3. How should cliff regions be represented when profiling coverage is incomplete?
4. Can one shared solver handle `PaLU`, `ASVD`, and `LLM-Pruner` without method-specific hacks dominating the candidate generation stage?
5. For methods that already restore a hardware-friendly execution shape, when should the framework abstain instead of forcing alignment repair?
