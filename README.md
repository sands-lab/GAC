# GAC: GPU-Aligned Compression

**When Smaller Is Slower: Dimensional Collapse in Compressed LLMs**

## Overview

GAC (GPU-Aligned Compression) is a research project studying *dimensional collapse* in compressed LLMs - the phenomenon where post-training compression produces irregular tensor dimensions that cause significant GPU performance degradation despite reducing FLOPs.

### Key Findings

- **88% latency increase** when head_dim=107 vs head_dim=96
- **96.9%** of PaLU-compressed dimensions are misaligned
- **Root causes identified**:
  - Tensor Core tile alignment (58% impact, K%16)
  - Vectorized load degradation (50% impact, K%8)
  - SDPA bandwidth efficiency (40% impact)
  - L2 cache sector waste (5.8%, negligible)
- **Solution**: Dimension repair with 3.72% memory overhead recovers 25-28% performance

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run SDPA benchmark
python -m scripts.run_benchmarks --experiment sdpa_backend

# Run GEMM benchmark
python -m scripts.run_benchmarks --experiment gemm_projection --dtype float16

# Run automated research system
python auto_research/orchestrator.py --max-iterations 10
```

## Baseline Model Selection

The repo-native LLM benchmark entrypoints now accept an explicit baseline model id instead of assuming a fixed Llama checkpoint.

```bash
# Evaluate a Mistral baseline
python -m src.gcompress_bench.llm_eval \
  --variant baseline \
  --suite ppl \
  --baseline-model-id mistralai/Mistral-7B-Instruct-v0.2 \
  --out results/llm_eval

# Run inference benchmarking for a Llama baseline
python -m src.gcompress_bench.llm_run \
  --variant baseline \
  --suite infer_sweep \
  --baseline-model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --out results/llm_run
```

## Project Structure

```
GAC/
├── auto_research/           # Automated research system
│   ├── orchestrator.py     # Main orchestrator
│   ├── agents/             # Agent prompts
│   └── state/              # Research state tracking
├── src/                    # Core benchmarks
│   ├── benchmark_gemm.py   # GEMM benchmarks
│   ├── benchmark_sdpa.py   # SDPA benchmarks
│   └── gcompress_bench/    # LLM benchmarking
├── Latex/                  # Paper (SIGPLAN format)
├── results/                # Experiment results
├── scripts/                # Analysis scripts
├── slurm/                  # Slurm job scripts
└── report.md              # Chinese report
```

## Slurm Usage

All GPU tasks must run through Slurm:

```bash
# Interactive (quick tests < 3 min)
srun --gres=gpu:a100:1 --constraint=gpu_a100 --pty bash

# Batch jobs
sbatch slurm/run_bench.sbatch
```

## Citation

```bibtex
@inproceedings{gac2026,
  title={When Smaller Is Slower: Dimensional Collapse in Compressed LLMs},
  author={Anonymous},
  booktitle={EuroMLSys},
  year={2026}
}
```

## License

MIT
