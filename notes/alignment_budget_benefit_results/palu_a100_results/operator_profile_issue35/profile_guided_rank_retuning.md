# PaLU Profile-Guided Rank Retuning

This profile-guided rank retuning artifact re-allocates the existing aligned budget across PaLU `k_proj` / `v_proj` modules.
It is budget-preserving and not a measured GPU speedup.

## Why This Follow-Up Exists
- The issue-35 optimization summary says the current simple aligned baseline removes the measurable alignment-sensitive tail, but still gives back meaningful time in `other_gemm_kernels`.
- The retuner therefore keeps the simple aligned baseline budget fixed while searching for lower-penalty contract-aware rank choices near the existing `rb1` / `rb1-gac-a100` configs.
- The score is root-cause-aware: it uses both recovered alignment-sensitive kernels and the `other_gemm_kernels` regression weight instead of minimizing alignment penalty alone.

## Strategy Snapshot
- Base budget: `45880` with estimated hardware penalty `4945.0`.
- Simple aligned baseline: `45904` with estimated hardware penalty `2570.0` and `20` recommended-value groups.
- Profile-guided retuned: `45904` with estimated hardware penalty `2490.0` and `50` recommended-value groups.
- Estimated penalty delta versus the simple aligned baseline: `-80.0`.
- Changed modules: `35` modules / `70` groups.

## Weight Calibration
- `alignment_sensitive_recovery_ms`: `3.427298`
- `other_gemm_kernels` regression magnitude: `2.366672` ms
- `attention_leakage_ms`: `5.87135`
- Derived `regression_weight`: `0.690536`
- Objective: `maximize sqrt(base_rank / mean_base_rank) * retained_rank - regression_weight * estimated_hardware_penalty`

## Largest Rank Increases
- `model.layers.7.self_attn.k_proj`: `240` -> `256` per group, budget delta `32`, penalty `5.0` -> `0.0`.
- `model.layers.6.self_attn.k_proj`: `240` -> `256` per group, budget delta `32`, penalty `5.0` -> `0.0`.
- `model.layers.31.self_attn.k_proj`: `416` -> `432` per group, budget delta `32`, penalty `50.0` -> `55.0`.
- `model.layers.28.self_attn.v_proj`: `480` -> `496` per group, budget delta `32`, penalty `70.0` -> `75.0`.
- `model.layers.26.self_attn.v_proj`: `432` -> `448` per group, budget delta `32`, penalty `55.0` -> `60.0`.
- `model.layers.25.self_attn.v_proj`: `416` -> `432` per group, budget delta `32`, penalty `50.0` -> `55.0`.
- `model.layers.24.self_attn.v_proj`: `400` -> `416` per group, budget delta `32`, penalty `45.0` -> `50.0`.
- `model.layers.23.self_attn.v_proj`: `368` -> `384` per group, budget delta `32`, penalty `35.0` -> `40.0`.

## Largest Rank Decreases
- `model.layers.4.self_attn.k_proj`: `272` -> `256` per group, budget delta `-32`, penalty `5.0` -> `0.0`.
- `model.layers.3.self_attn.k_proj`: `272` -> `256` per group, budget delta `-32`, penalty `5.0` -> `0.0`.
- `model.layers.28.self_attn.k_proj`: `288` -> `272` per group, budget delta `-32`, penalty `10.0` -> `5.0`.
- `model.layers.25.self_attn.k_proj`: `208` -> `192` per group, budget delta `-32`, penalty `5.0` -> `0.0`.
- `model.layers.24.self_attn.k_proj`: `176` -> `160` per group, budget delta `-32`, penalty `5.0` -> `0.0`.
- `model.layers.23.self_attn.k_proj`: `176` -> `160` per group, budget delta `-32`, penalty `5.0` -> `0.0`.
- `model.layers.22.self_attn.k_proj`: `160` -> `144` per group, budget delta `-32`, penalty `0.0` -> `5.0`.
- `model.layers.20.self_attn.k_proj`: `176` -> `160` per group, budget delta `-32`, penalty `5.0` -> `0.0`.

## Caveats
- This is a budget-preserving offline retuning note, not a measured GPU speedup.
- The proxy objective only estimates whether the candidate should reduce the simple aligned baseline's hardware penalty; it does not replace profiler traces.
- Any claim about real end-to-end inference wins still requires rerunning the Slurm/A100 operator profile collection for the retuned config.
