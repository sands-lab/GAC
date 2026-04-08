#!/usr/bin/env python3
"""
Build a PaLU checkpoint from a baseline model using the vendored PaLU package.

This wraps the upstream Palu flow in a repo-native entrypoint so the output
checkpoint directory can later be consumed by src.gcompress_bench.palu_loader.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.gcompress_bench.palu_loader import default_palu_base_dir  # noqa: E402

PALU_VENDOR_DIR = default_palu_base_dir()
if str(PALU_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(PALU_VENDOR_DIR))

from palu.model import HeadwiseLowRankModule  # noqa: E402
from palu.rank_search import rank_search  # noqa: E402
from palu.decomposition import compress_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_output_dir(args: argparse.Namespace) -> Path:
    model_name = args.model_id.rstrip("/").split("/")[-1]
    dirname = (
        f"{model_name}_ratio-{args.param_ratio_target}"
        f"_gs-{args.head_group_size}"
        f"-{args.search_method}-{args.decompose_method}"
    )
    return Path(args.output_root) / dirname


def dump_palu_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    save_path: Path,
    baseline_model_id: str,
) -> dict:
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    config = model.config.to_dict()
    config["head_wise_ranks"] = {}
    for name, module in model.named_modules():
        if isinstance(module, HeadwiseLowRankModule):
            config["head_wise_ranks"][name] = module.ranks

    model_type = getattr(model.config, "model_type", None)
    if model_type in {"llama", "palullama"}:
        config["model_type"] = "palullama"
        config["architectures"] = ["PaluLlamaForCausalLM"]
    elif model_type in {"mistral", "palumistral"}:
        config["model_type"] = "palumistral"
        config["architectures"] = ["PaluMistralForCausalLM"]
    elif model_type in {"qwen2", "paluqwen2"}:
        config["model_type"] = "paluqwen2"
        config["architectures"] = ["PaluQwen2ForCausalLM"]
    else:
        raise ValueError(f"Unsupported model_type for PaLU checkpoint export: {model_type!r}")

    config["original_model_name_or_path"] = baseline_model_id
    (save_path / "config.json").write_text(json.dumps(config, indent=2))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PaLU checkpoint")
    parser.add_argument("--model-id", required=True, help="Baseline model id or local model path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--calib-dataset", default="wikitext2", choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--calib-seqlen", type=int, default=1024)
    parser.add_argument("--n-fisher-calib-samples", type=int, default=32)
    parser.add_argument("--n-whiten-calib-samples", type=int, default=256)
    parser.add_argument("--head-group-size", type=int, default=4)
    parser.add_argument("--param-ratio-target", type=float, required=True)
    parser.add_argument(
        "--search-method",
        default="fisher_uniform",
        choices=["fisher", "fisher_uniform", "uniform"],
    )
    parser.add_argument(
        "--decompose-method",
        default="whiten",
        choices=["whiten", "svd"],
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dump-huggingface-model", action="store_true")
    parser.add_argument("--output-root", default=str(default_palu_base_dir()))
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--dimension-repair-strategy", default=None)
    parser.add_argument("--dimension-repair-max-overhead-pct", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if args.device.startswith("cuda") else None,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)
    compress_model(model, tokenizer, args, args.device, search_results)

    output_dir = build_output_dir(args)
    summary = {
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "param_ratio_target": args.param_ratio_target,
        "head_group_size": args.head_group_size,
        "search_method": args.search_method,
        "decompose_method": args.decompose_method,
        "rank_sum": rank_sum,
        "total_rank": total_rank,
        "effective_ratio": (rank_sum / total_rank) if total_rank else None,
    }

    if args.dump_huggingface_model:
        dump_palu_checkpoint(model, tokenizer, output_dir, baseline_model_id=args.model_id)
        (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"PaLU checkpoint saved to: {output_dir}")
    else:
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
