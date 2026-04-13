# PaLU Prefill Dispatch Reduction Prototype

This note records a grouped reconstruction prototype for `HeadwiseLowRankModule` on the PaLU `k_proj` / `v_proj` path.
It is not a measured GPU speedup; it only verifies a code-path-level dispatch reduction contract.

## Checkpoint Dispatch Summary
- Affected modules: `64`
- Total head groups: `128`
- Current projection-path dispatch: `192` total calls (`64` `VT` calls + `128` per-group `U[*]` calls).
- Grouped reconstruction dispatch: `128` total calls (`64` `VT` calls + `64` grouped reconstruction calls).
- Projection-path reduction factor: `1.50x`
- `U[*]`-only reduction factor: `2.00x`
- Padding overhead: `0` extra latent rank slots (`0.00%`).

## Prototype Validation
- Prototype ranks: `[3, 5, 2, 4]`
- Input shape: `[2, 3, 12]`
- Legacy reconstruct dispatches: `4`
- Grouped reconstruct dispatches: `1`
- Observed per-group `U[*]` calls under grouped reconstruction: `0`
- Max abs diff between legacy and grouped reconstruction: `1.91e-06`
- Prototype padding overhead: `6` latent slots (`42.86%`).

## Highest Padding Modules
- `model.layers.9.self_attn.v_proj`: 2 groups, max rank 512, padding overhead 0.00% (0 extra slots).
- `model.layers.9.self_attn.k_proj`: 2 groups, max rank 254, padding overhead 0.00% (0 extra slots).
- `model.layers.8.self_attn.v_proj`: 2 groups, max rank 512, padding overhead 0.00% (0 extra slots).
- `model.layers.8.self_attn.k_proj`: 2 groups, max rank 232, padding overhead 0.00% (0 extra slots).
- `model.layers.7.self_attn.v_proj`: 2 groups, max rank 512, padding overhead 0.00% (0 extra slots).

## Caveats
- This grouped reconstruction prototype keeps the existing `VT` projection untouched; it only changes the `U[*]` reconstruction step.
- The artifact reports dispatch counts and padding overhead, not a measured GPU speedup.
- Any claim about real prefill latency still needs profiler traces on the Slurm/A100 path used by `operator_profile_issue35`.
