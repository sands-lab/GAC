# H100 Shape Contract

## Scope

This note consolidates the H100-specific shape contract evidence already present in the repo, so the alignment budget code can consume one explicit Hopper profile instead of relying on scattered paper text and figures.

## Repo Evidence

- `Latex/main.tex` states that H100 still shows the same coarse fallback boundary when `d mod 8 != 0`, with a larger slowdown than A100 because the math-backend fallback wastes more of Hopper's peak throughput.
- `Latex/main.tex` and `report.md` both describe a second, stricter performance tier at `mod 16`, inherited from Tensor Core tile behavior.
- `report.md` also records Hopper-specific hypotheses around TMA and WGMMA, especially that `K % 64` may become a better deployment target on H100-class hardware, but it explicitly leaves that point in future-work status rather than marking it as fully validated.
- The checked-in H100 figure set in `Latex/figures_h100/` confirms that this repository already contains H100-facing analysis assets, even though the raw profiling CSV tables are not checked in.

## Current Contract

- `minimal_alignment`: `8`
- `preferred_alignment`: `16`
- `recommended_values`: `(64, 128, 192, 256)`
- `cliff_values`: all dimensions in the current `64..256` sweep window that violate `minimal_alignment`

## Rationale

- `minimal_alignment=8` is the strongest confirmed Hopper rule in the repo today. It is directly supported by the H100 appendix text that documents fallback when `d mod 8 != 0`.
- `preferred_alignment=16` remains the conservative Tensor Core preference. The repo has evidence that `mod 16` is the next performance tier, but not enough checked-in Hopper-only raw data to replace it with a stricter universal rule.
- `recommended_values=(64, 128, 192, 256)` reflects the only Hopper-safe fast-path family that is both hardware-plausible and explicitly consistent with the current notes: multiples of 64 are compatible with the WGMMA discussion in `report.md`, without claiming that every non-64 multiple is already disproven.
- `cliff_values` are intentionally conservative. They do not claim a fine-grained Hopper profiler map; they simply encode the currently validated severe fallback region so search code can avoid dimensions that the repo already treats as outside the fast path.

## Validation

- Validated by repo evidence:
  - `d mod 8 != 0` should be treated as outside the preferred Hopper fast path.
  - `mod 16` is still a useful preferred_alignment tier for search and ranking.
  - multiples of 64 are safe recommended_values for a conservative H100 deployment contract.
- Not yet validated by checked-in raw data:
  - exact Hopper-only cliff points inside the `mod 8 == 0` region
  - whether `K % 64` should replace `preferred_alignment=16` as a universal optimization target
  - whether different operator roles (`attention_kv`, GEMM, SDPA) need separate Hopper contracts

This validation section is intentionally split between "validated now" and "still pending" so the current repo-native contract stays explicit about what is measured versus inferred.

## Next Step

When raw H100 profiling tables are checked in, this note should be upgraded from a conservative contract summary to a measured contract table, and `cliff_values` should be narrowed from the coarse misaligned region to empirically observed Hopper-specific penalty points.
