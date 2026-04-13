#!/bin/bash
#SBATCH --job-name=llm_operator_profile
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j_llm_operator_profile.out
#SBATCH --error=slurm_logs/%j_llm_operator_profile.err
#SBATCH --qos=spot

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"

DRY_RUN=0
SKIP_ENV_SETUP=0
OUTPUT_ROOT="results/operator_profiles/palu_issue35"
BUNDLE_OUTPUT="notes/alignment_budget_benefit_results/palu_a100_results/operator_profile_issue35"
DEVICE="cuda:0"
DTYPE="float16"
BASELINE_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
PALU_PATTERN="Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1"
PALU_ALIGNED_PATTERN="Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100"
STAGE="both"
PREFILL_BATCH=1
PREFILL_SEQ_LEN=1024
DECODE_BATCH=1
DECODE_CONTEXT_LEN=1024
WARMUP=3
EXPORT_TRACE=1
CONDA_ENV_NAME="${CONDA_ENV_NAME:-palu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: bash slurm/run_llm_operator_profile.sbatch [options]

Options:
  --dry-run                  Print the three profiling commands plus publish command without executing them.
  --skip-env-setup           Skip conda activation. Intended for local dry-run validation.
  --output-root PATH         Root directory for baseline / palu / aligned_gac run directories.
  --bundle-output PATH       Output directory for the published repo-tracked bundle.
  --device DEVICE            Profiling device passed to scripts/profile_llm_inference_operators.py.
  --dtype DTYPE              float16 or bfloat16.
  --baseline-model-id ID     Baseline model id for the profiling script.
  --palu-pattern PATTERN     Unaligned PaLU checkpoint pattern.
  --palu-aligned-pattern P   Aligned PaLU checkpoint pattern.
  --stage STAGE              prefill, decode, or both.
  --prefill-batch N          Prefill batch size.
  --prefill-seq-len N        Prefill sequence length.
  --decode-batch N           Decode batch size.
  --decode-context-len N     Decode context length.
  --warmup N                 Warmup iterations for each stage.
  --export-trace             Export Chrome traces for each stage run.
  --no-export-trace          Disable trace export.
  --help                     Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-env-setup) SKIP_ENV_SETUP=1; shift ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --bundle-output) BUNDLE_OUTPUT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --baseline-model-id) BASELINE_MODEL_ID="$2"; shift 2 ;;
    --palu-pattern) PALU_PATTERN="$2"; shift 2 ;;
    --palu-aligned-pattern) PALU_ALIGNED_PATTERN="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --prefill-batch) PREFILL_BATCH="$2"; shift 2 ;;
    --prefill-seq-len) PREFILL_SEQ_LEN="$2"; shift 2 ;;
    --decode-batch) DECODE_BATCH="$2"; shift 2 ;;
    --decode-context-len) DECODE_CONTEXT_LEN="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --export-trace) EXPORT_TRACE=1; shift ;;
    --no-export-trace) EXPORT_TRACE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" ]]; then
    printf '%s\n' "$CONDA_SH"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base
    conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    printf '%s\n' "$conda_base/etc/profile.d/conda.sh"
    return 0
  fi
  printf '%s\n' "$HOME/miniforge3/etc/profile.d/conda.sh"
}

setup_runtime_env() {
  if [[ "$SKIP_ENV_SETUP" == "1" ]]; then
    return 0
  fi

  local conda_sh
  conda_sh="$(resolve_conda_sh)"
  if [[ ! -f "$conda_sh" ]]; then
    echo "Missing conda activation script: $conda_sh" >&2
    exit 1
  fi

  # The activate-binutils hook in the current palu env expects unset shell vars.
  set +u
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$CONDA_ENV_NAME"
  set -u
}

print_command() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_command() {
  print_command "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

cd "$SUBMIT_DIR"
mkdir -p slurm_logs "$OUTPUT_ROOT"

setup_runtime_env

export PYTHONPATH="$SUBMIT_DIR:$SUBMIT_DIR/src:$SUBMIT_DIR/third_party/palu:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

COMMON_ARGS=(
  --device "$DEVICE"
  --dtype "$DTYPE"
  --baseline-model-id "$BASELINE_MODEL_ID"
  --stage "$STAGE"
  --prefill-batch "$PREFILL_BATCH"
  --prefill-seq-len "$PREFILL_SEQ_LEN"
  --decode-batch "$DECODE_BATCH"
  --decode-context-len "$DECODE_CONTEXT_LEN"
  --warmup "$WARMUP"
)
if [[ "$EXPORT_TRACE" == "1" ]]; then
  COMMON_ARGS+=(--export-trace)
fi

BASELINE_CMD=(
  "$PYTHON_BIN" scripts/profile_llm_inference_operators.py
  --variant baseline
  --output-dir "$OUTPUT_ROOT/baseline"
  "${COMMON_ARGS[@]}"
)
PALU_CMD=(
  "$PYTHON_BIN" scripts/profile_llm_inference_operators.py
  --variant palu
  --output-dir "$OUTPUT_ROOT/palu"
  --palu-pattern "$PALU_PATTERN"
  "${COMMON_ARGS[@]}"
)
ALIGNED_CMD=(
  "$PYTHON_BIN" scripts/profile_llm_inference_operators.py
  --variant aligned_gac
  --output-dir "$OUTPUT_ROOT/aligned_gac"
  --palu-aligned-pattern "$PALU_ALIGNED_PATTERN"
  "${COMMON_ARGS[@]}"
)
PUBLISH_CMD=(
  "$PYTHON_BIN" scripts/publish_llm_inference_operator_profile_bundle.py
  --baseline-run-dir "$OUTPUT_ROOT/baseline"
  --palu-run-dir "$OUTPUT_ROOT/palu"
  --aligned-run-dir "$OUTPUT_ROOT/aligned_gac"
  --output-dir "$BUNDLE_OUTPUT"
)

echo "=============================================="
echo "LLM Inference Operator Profile Bundle"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local-dry-run}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Date: $(date)"
echo "Dry Run: $DRY_RUN"
echo "Skip Env Setup: $SKIP_ENV_SETUP"
echo "Output Root: $OUTPUT_ROOT"
echo "Bundle Output: $BUNDLE_OUTPUT"
echo "Stage: $STAGE"
echo "Prefill Seq Len: $PREFILL_SEQ_LEN"
echo "Decode Context Len: $DECODE_CONTEXT_LEN"
echo "Warmup: $WARMUP"
if [[ "$DRY_RUN" != "1" ]]; then
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "=============================================="

run_command "${BASELINE_CMD[@]}"
run_command "${PALU_CMD[@]}"
run_command "${ALIGNED_CMD[@]}"
run_command "${PUBLISH_CMD[@]}"

echo ""
echo "=============================================="
echo "LLM operator-profile bundle complete at $(date)"
echo "Output root: $OUTPUT_ROOT"
echo "Bundle output: $BUNDLE_OUTPUT"
echo "=============================================="
