# Token Eviction M-Sweep

This note turns the open token-eviction line in [`notes/operator_impact_map.md`](notes/operator_impact_map.md)
into one runnable artifact family.
Instead of mixing token eviction into the ASVD / LLM-Pruner / PaLU stories, it locks the
next question to the one axis those methods do not cover well: sequence-driven `M`.

## Why This Artifact Exists

The current repo evidence is still asymmetric:

- ASVD and LLM-Pruner already have the strongest checked-in `prefill` evidence
- issue 32 narrowed PaLU's changed path to attention-adjacent projection GEMM rather than SDPA width
- token eviction is still mostly a conceptual line in `notes/operator_impact_map.md`, `notes/full_paper_plan.md`, and `notes/paper_plan.md`

That makes token eviction the next open operator question, but it is a different question.
It does not primarily ask about GEMM `K`, GEMM `N`, or attention `head_dim`.
It asks how sequence-driven `M` or context length changes runtime across `prefill` and `decode`.

This issue therefore does not claim measured attribution results yet.
It only publishes a runnable plan that the existing benchmark CLI can execute directly.

## Long-Context Prefill GEMM Slice

### Operator hypothesis

- Method path: token eviction / KV eviction changes the number of active tokens kept in the prompt or cache
- Dominant stage: long-context `prefill`
- Primary operator: GEMM
- Primary changed axis: GEMM `M`

`notes/operator_impact_map.md` already ranks token eviction as an `M`-axis story, and
`notes/paper_plan.md` explicitly maps token eviction to sequence length.
For `prefill`, the next runnable slice should therefore test large prompt matmuls with fixed
projection widths while sweeping only `M`.

### Real-shape checkpoints

The new spec tracks explicit long-context checkpoints:

- `255`, `256`, `257`
- `511`, `512`, `513`
- `1023`, `1024`, `1025`
- `2047`, `2048`, `2049`
- `4095`, `4096`, `4097`
- `8191`, `8192`, `8193`

These are intentionally not another toy sweep.
They sit around context-length boundaries that are plausible for token eviction, long prompts,
and sequence-tile transitions, and they preserve the exact checkpoints in the checked-in spec.

The supporting code change adds `gemm_m_dense` to `src/experiment_runner.py`, so this slice now
has the same repo-native status as the earlier `gemm_k_dense` and `gemm_n_dense` experiments.

## Decode Context-Length SDPA Slice

### Operator hypothesis

- Method path: token eviction changes the active attention context length during generation
- Dominant stage: `decode`
- Primary operator: SDPA
- Primary changed axis: context length (`seq_len`)

For decode, the first step is not to reopen the `head_dim` alignment story.
That question already belongs to PaLU-like methods.
The token-eviction slice should instead hold `head_dim` fixed at an aligned value and ask how
context length alone changes the attention path.

### Real-shape checkpoints

The decode-side SDPA slice fixes:

- `batch = 1`
- `n_heads = 32`
- `head_dim = 128`

and sweeps these explicit context length checkpoints:

- `255`, `256`, `257`
- `511`, `512`, `513`
- `1023`, `1024`, `1025`
- `2047`, `2048`, `2049`

Keeping `head_dim = 128` is intentional.
It prevents the context length sweep from being contaminated by the separate `head_dim`
alignment cliffs already studied elsewhere in the repo.

## Ready-to-Run Experiment Specs

The checked-in spec lives at [`experiments/token_eviction_m_sweep.yaml`](experiments/token_eviction_m_sweep.yaml)
and currently exposes two experiments:

1. `TOKENEVICTION_prefill_gemm_m_real_shapes`
2. `TOKENEVICTION_decode_sdpa_context_real_shapes`

The prefill GEMM slice can be launched with:

```bash
python3 scripts/run_experiment.py run \
  --spec experiments/token_eviction_m_sweep.yaml \
  --name TOKENEVICTION_prefill_gemm_m_real_shapes \
  --results-root results
```

The decode SDPA slice can be launched with:

```bash
python3 scripts/run_experiment.py run \
  --spec experiments/token_eviction_m_sweep.yaml \
  --name TOKENEVICTION_decode_sdpa_context_real_shapes \
  --results-root results
```

Together, these two slices make the token eviction story concrete:

- `gemm_m_dense` covers long-context `prefill` GEMM `M`
- `sdpa_dense` with fixed `head_dim` covers decode-side context length sensitivity

## Deliverable Boundary

This artifact provides:

- a runnable token eviction plan anchored to `notes/operator_impact_map.md`
- explicit real-shape `M` checkpoints instead of a toy dense sweep
- one `prefill` slice and one `decode` slice, so stage-specific behavior stays separated

This artifact does not claim:

- that the repo already contains measured token-eviction latency results
- that GEMV- or KV-cache-specific decode kernels are fully attributed yet
- that a GAC-style constrained token-eviction variant has already been built or evaluated

The immediate goal is narrower:
make the next token eviction profiling run deterministic, repo-native, and easy to cite.
