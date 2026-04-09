#!/bin/bash
#SBATCH --job-name=unified_eval
#SBATCH --output=slurm_logs/unified_eval_%j.out
#SBATCH --error=slurm_logs/unified_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

METHOD=""
MODEL_ID="meta-llama/Meta-Llama-3-8B"
OUTPUT_ROOT="results/fixed_length_decode"
PROMPT_LEN=128
GEN_TOKENS=64
DECODE_WARMUP=2
DECODE_MEASURE=5
METHOD_RESULTS_DIR=""
CONDA_ENV_NAME=""
SUMMARY_PATH=""
ASVD_SENSITIVITY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --prompt-len) PROMPT_LEN="$2"; shift 2;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2;;
    --decode-warmup) DECODE_WARMUP="$2"; shift 2;;
    --decode-measure) DECODE_MEASURE="$2"; shift 2;;
    --results-dir) METHOD_RESULTS_DIR="$2"; shift 2;;
    --conda-env) CONDA_ENV_NAME="$2"; shift 2;;
    --summary-path) SUMMARY_PATH="$2"; shift 2;;
    --asvd-sensitivity) ASVD_SENSITIVITY="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$METHOD" ]]; then
  echo "Missing required --method {asvd|llmpruner}" >&2
  exit 1
fi

case "$METHOD" in
  asvd)
    DEFAULT_ENV="sparsegpt"
    VARIANTS=(baseline unaligned aligned)
    DEFAULT_RESULTS_DIR="results/asvd_simple"
    DEFAULT_SUMMARY_PATH="notes/alignment_budget_benefit_results/asvd_fixed_length_decode_summary.json"
    DEFAULT_ASVD_SENSITIVITY="stable_rank"
    ;;
  llmpruner)
    DEFAULT_ENV="prunelayers"
    VARIANTS=(baseline pruned pruned_r8)
    DEFAULT_RESULTS_DIR="results/llmpruner_llama3_v2"
    DEFAULT_SUMMARY_PATH="notes/alignment_budget_benefit_results/llmpruner_fixed_length_decode_summary.json"
    DEFAULT_ASVD_SENSITIVITY="ppl"
    ;;
  *)
    echo "Unsupported method: $METHOD" >&2
    exit 1
    ;;
esac

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-$DEFAULT_ENV}"
METHOD_RESULTS_DIR="${METHOD_RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
SUMMARY_PATH="${SUMMARY_PATH:-$DEFAULT_SUMMARY_PATH}"
ASVD_SENSITIVITY="${ASVD_SENSITIVITY:-$DEFAULT_ASVD_SENSITIVITY}"
RAW_OUTPUT_DIR="$OUTPUT_ROOT/$METHOD"

echo "============================================="
echo "Unified Fixed-Length Decode Evaluation"
echo "Job ID: ${SLURM_JOB_ID:-<local>}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo: $REPO_ROOT"
echo "Method: $METHOD"
echo "Conda Env: $CONDA_ENV_NAME"
if [[ "$METHOD" == "asvd" ]]; then
  echo "ASVD Sensitivity: $ASVD_SENSITIVITY"
fi
echo "Start: $(date)"
echo "============================================="

set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
set -u

cd "$REPO_ROOT"
mkdir -p "$RAW_OUTPUT_DIR" slurm_logs
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/third_party/LLM-Pruner:$REPO_ROOT/third_party/ASVD4LLM:$REPO_ROOT/scripts/asvd_simple:${PYTHONPATH:-}"

for variant in "${VARIANTS[@]}"; do
  python scripts/unified_eval.py \
    --method "$METHOD" \
    --variant "$variant" \
    --model_id "$MODEL_ID" \
    --results_dir "$METHOD_RESULTS_DIR" \
    --output "$RAW_OUTPUT_DIR" \
    --prompt_len "$PROMPT_LEN" \
    --gen_tokens "$GEN_TOKENS" \
    --decode-warmup "$DECODE_WARMUP" \
    --decode-measure "$DECODE_MEASURE" \
    --asvd-sensitivity "$ASVD_SENSITIVITY" \
    --skip_accuracy
done

python scripts/collect_fixed_length_decode_summary.py \
  --method "$METHOD" \
  --results-dir "$RAW_OUTPUT_DIR" \
  --output "$SUMMARY_PATH"

echo ""
echo "============================================="
echo "Experiment Complete: $(date)"
echo "Raw results: $RAW_OUTPUT_DIR"
echo "Summary: $SUMMARY_PATH"
echo "============================================="
