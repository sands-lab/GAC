#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$ROOT/third_party/SVD-LLM"
PATCH_PATH="$ROOT/third_party_patches/SVD-LLM/svd_llama_kvcache.patch"

if [[ ! -d "$SUBMODULE_DIR/.git" && ! -f "$SUBMODULE_DIR/.git" ]]; then
  echo "FAIL [apply_svd_llama_kvcache_patch]: missing SVD-LLM git worktree at '$SUBMODULE_DIR'." >&2
  exit 1
fi

if [[ ! -f "$PATCH_PATH" ]]; then
  echo "FAIL [apply_svd_llama_kvcache_patch]: missing patch file '$PATCH_PATH'." >&2
  exit 1
fi

if git -C "$SUBMODULE_DIR" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
  echo "SVD-LLM KV-cache patch already applied."
  exit 0
fi

if ! git -C "$SUBMODULE_DIR" apply --check "$PATCH_PATH" >/dev/null 2>&1; then
  echo "FAIL [apply_svd_llama_kvcache_patch]: patch cannot be applied cleanly." >&2
  echo "Target: $SUBMODULE_DIR/component/svd_llama_kvcache.py" >&2
  exit 1
fi

git -C "$SUBMODULE_DIR" apply "$PATCH_PATH"
echo "Applied SVD-LLM KV-cache patch."
