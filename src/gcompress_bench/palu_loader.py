"""
PaLU helper: locate compressed checkpoint directory and load model/tokenizer.
Uses the checkpoint's registered config/model type to resolve the right PaLU architecture.
"""
import glob
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PALU_VENDOR_DIR = REPO_ROOT / "third_party" / "palu"
if str(PALU_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(PALU_VENDOR_DIR))

import palu.model  # noqa: F401 - ensure PaLU AutoConfig/AutoModel registrations are loaded


DEFAULT_BASELINE_IDS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "palullama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "palumistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "paluqwen2": "Qwen/Qwen2-7B-Instruct",
}


def default_palu_base_dir() -> Path:
    """Return the repo-local PaLU vendor root."""
    return PALU_VENDOR_DIR


def find_palu_dir(
    base: str | Path = default_palu_base_dir(),
    pattern: str = "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4*",
) -> Path:
    candidates = sorted(glob.glob(str(Path(base) / pattern)))
    if not candidates:
        raise FileNotFoundError(f"No PaLU ratio directory matching {pattern} under {base}")
    return Path(candidates[0])


def _resolve_baseline_id(config, palu_dir: Path, baseline_id: str | None = None) -> str:
    if baseline_id:
        return baseline_id

    model_type = getattr(config, "model_type", None)
    if model_type in DEFAULT_BASELINE_IDS:
        return DEFAULT_BASELINE_IDS[model_type]

    for attr in ("base_model_name_or_path", "_name_or_path"):
        candidate = getattr(config, attr, None)
        if candidate and Path(str(candidate)) != palu_dir:
            return str(candidate)

    raise ValueError(
        f"Unsupported PaLU checkpoint model_type={model_type!r}; pass baseline_id explicitly."
    )


def load_palu_model(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    base: str | Path = default_palu_base_dir(),
    pattern: str = "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4*",
    baseline_id: str | None = None,
) -> Tuple[torch.nn.Module, AutoTokenizer, Path]:
    palu_dir = find_palu_dir(base=base, pattern=pattern)

    config = AutoConfig.from_pretrained(palu_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        palu_dir,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )

    # Use the matching baseline tokenizer to avoid tokenizer.json format issues in the PaLU directory.
    baseline_id = _resolve_baseline_id(config, palu_dir, baseline_id=baseline_id)
    tokenizer = AutoTokenizer.from_pretrained(baseline_id, use_fast=True)
    return model, tokenizer, palu_dir
