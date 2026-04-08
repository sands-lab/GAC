#!/bin/bash
#SBATCH --job-name=C5_e2e
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_C5_e2e_comparison.out
#SBATCH --error=slurm_logs/%j_C5_e2e_comparison.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

# C5 End-to-End LLM Inference Comparison
# Compares: Baseline vs PaLU vs aligned variant
# Requires 80GB GPU for full LLM experiments

set -euo pipefail

source /home/lyut/miniforge3/etc/profile.d/conda.sh
conda activate palu
cd $SLURM_SUBMIT_DIR

# Set up Python path
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR/third_party/palu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# Create output directories
mkdir -p results/C5
mkdir -p slurm_logs

# Parse arguments
SMOKE=""
REPAIR_STRATEGY="minimal"
SKIP_BASELINE=""
RUN_ID=""
PALU_PATTERN="Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4*"
ALIGNED_SOURCE="repair"
PALU_ALIGNED_PATTERN=""
BASELINE_RESULTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE="--smoke"; shift;;
    --repair-strategy) REPAIR_STRATEGY="$2"; shift 2;;
    --skip-baseline) SKIP_BASELINE="--skip-baseline"; shift;;
    --run-id) RUN_ID="$2"; shift 2;;
    --palu-pattern) PALU_PATTERN="$2"; shift 2;;
    --aligned-source) ALIGNED_SOURCE="$2"; shift 2;;
    --palu-aligned-pattern) PALU_ALIGNED_PATTERN="$2"; shift 2;;
    --baseline-results) BASELINE_RESULTS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=============================================="
echo "C5 End-to-End LLM Inference Comparison"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo "Repair Strategy: $REPAIR_STRATEGY"
echo "Smoke Test: ${SMOKE:-no}"
echo "PaLU Pattern: $PALU_PATTERN"
echo "Aligned Source: $ALIGNED_SOURCE"
echo "Aligned Pattern: ${PALU_ALIGNED_PATTERN:-<none>}"
echo "Baseline Results: ${BASELINE_RESULTS:-<none>}"
echo "=============================================="

# Run the comparison experiment
if [[ -n "$RUN_ID" ]]; then
  RUN_ID_ARG="--run-id $RUN_ID"
else
  RUN_ID_ARG=""
fi
if [[ -n "$PALU_ALIGNED_PATTERN" ]]; then
  PALU_ALIGNED_ARG="--palu-aligned-pattern $PALU_ALIGNED_PATTERN"
else
  PALU_ALIGNED_ARG=""
fi
if [[ -n "$BASELINE_RESULTS" ]]; then
  BASELINE_RESULTS_ARG="--baseline-results $BASELINE_RESULTS"
else
  BASELINE_RESULTS_ARG=""
fi

python scripts/run_c5_e2e_comparison.py \
    --out results/C5 \
    --device cuda:0 \
    --dtype float16 \
    --repair-strategy "$REPAIR_STRATEGY" \
    --palu-pattern "$PALU_PATTERN" \
    --aligned-source "$ALIGNED_SOURCE" \
    $PALU_ALIGNED_ARG \
    $BASELINE_RESULTS_ARG \
    $SMOKE \
    $SKIP_BASELINE \
    $RUN_ID_ARG

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "Results saved to: results/C5/"
echo "=============================================="
