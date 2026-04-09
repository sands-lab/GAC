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
ISSUE_ID=""
PUBLISH_ROOT=""

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
    --issue-id) ISSUE_ID="$2"; shift 2;;
    --publish-root) PUBLISH_ROOT="$2"; shift 2;;
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
PUBLISH_DIR=""
if [[ -n "$PUBLISH_ROOT" ]]; then
  PUBLISH_DIR="$PUBLISH_ROOT/$METHOD"
fi
START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

METADATA_PATH="$RAW_OUTPUT_DIR/run_metadata.json"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
METADATA_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export ISSUE_ID METHOD MODEL_ID CONDA_ENV_NAME METHOD_RESULTS_DIR
export DECODE_WARMUP DECODE_MEASURE PROMPT_LEN GEN_TOKENS
export START_TIME_UTC METADATA_TIME_UTC GIT_COMMIT ASVD_SENSITIVITY
python3 - "$METADATA_PATH" "$REPO_ROOT" "$RAW_OUTPUT_DIR" "$SUMMARY_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2]).resolve()
raw_output_dir = Path(sys.argv[3]).resolve()
summary_path = Path(sys.argv[4]).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


payload = {
    "issue_id": os.environ.get("ISSUE_ID") or None,
    "submitted_via": "slurm/run_unified_eval.sbatch",
    "job_id": os.environ.get("SLURM_JOB_ID", "<local>"),
    "git_commit": os.environ["GIT_COMMIT"],
    "method": os.environ["METHOD"],
    "model_id": os.environ["MODEL_ID"],
    "conda_env": os.environ["CONDA_ENV_NAME"],
    "raw_results_root": display_path(raw_output_dir),
    "summary_path": display_path(summary_path),
    "results_dir": os.environ["METHOD_RESULTS_DIR"],
    "decode_warmup": int(os.environ["DECODE_WARMUP"]),
    "decode_measure": int(os.environ["DECODE_MEASURE"]),
    "prompt_len": int(os.environ["PROMPT_LEN"]),
    "gen_tokens": int(os.environ["GEN_TOKENS"]),
    "started_at_utc": os.environ["START_TIME_UTC"],
    "metadata_written_at_utc": os.environ["METADATA_TIME_UTC"],
}

asvd_sensitivity = os.environ.get("ASVD_SENSITIVITY")
if os.environ["METHOD"] == "asvd" and asvd_sensitivity:
    payload["asvd_sensitivity"] = asvd_sensitivity

metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

COLLECT_ARGS=(
  --method "$METHOD"
  --results-dir "$RAW_OUTPUT_DIR"
  --output "$SUMMARY_PATH"
)
if [[ -n "$PUBLISH_DIR" ]]; then
  COLLECT_ARGS+=(--publish-dir "$PUBLISH_DIR")
fi

python scripts/collect_fixed_length_decode_summary.py "${COLLECT_ARGS[@]}"

echo ""
echo "============================================="
echo "Experiment Complete: $(date)"
echo "Raw results: $RAW_OUTPUT_DIR"
echo "Summary: $SUMMARY_PATH"
echo "============================================="
