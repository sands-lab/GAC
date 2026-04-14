#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$ROOT/third_party/LLM-Pruner"
PATCH_PATH="$ROOT/third_party_patches/LLM-Pruner/quac_eval.patch"

if [[ ! -d "$SUBMODULE_DIR/.git" && ! -f "$SUBMODULE_DIR/.git" ]]; then
  echo "FAIL [apply_llmpruner_quac_patch]: missing LLM-Pruner git worktree at '$SUBMODULE_DIR'." >&2
  exit 1
fi

if [[ ! -f "$PATCH_PATH" ]]; then
  echo "FAIL [apply_llmpruner_quac_patch]: missing patch file '$PATCH_PATH'." >&2
  exit 1
fi

if git -C "$SUBMODULE_DIR" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
  echo "LLM-Pruner QuAC patch already applied."
  exit 0
fi

if ! git -C "$SUBMODULE_DIR" apply --check "$PATCH_PATH" >/dev/null 2>&1; then
  echo "FAIL [apply_llmpruner_quac_patch]: patch cannot be applied cleanly." >&2
  echo "Targets: $SUBMODULE_DIR/lm-evaluation-harness/lm_eval/tasks/quac.py and __init__.py" >&2
  exit 1
fi

git -C "$SUBMODULE_DIR" apply "$PATCH_PATH"
echo "Applied LLM-Pruner QuAC patch."
