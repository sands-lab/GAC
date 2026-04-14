#!/usr/bin/env python3
"""Collect a repo-native operator profile for real LLM inference stages."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.environment import collect_environment
from src.gcompress_bench.operator_profile import classify_operator_family, summarize_profile_run
from src.gcompress_bench.palu_loader import load_palu_model


DEFAULT_BASELINE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_PALU_PATTERN = "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1"
DEFAULT_PALU_ALIGNED_PATTERN = (
    "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten-rb1-gac-a100"
)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def normalize_input_shapes(shapes: Any) -> List[Any]:
    if not shapes:
        return []

    normalized: List[Any] = []
    for shape in shapes:
        if isinstance(shape, (list, tuple)):
            normalized.append(
                [
                    int(value) if isinstance(value, (int, float)) else value
                    for value in shape
                ]
            )
        else:
            normalized.append(shape)
    return normalized


def profiler_event_to_dict(event: Any) -> Dict[str, Any]:
    name = str(getattr(event, "key", ""))
    self_cuda_time_us = float(
        getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        or 0.0
    )
    cuda_time_us = float(
        getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
        or 0.0
    )
    self_cpu_time_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
    cpu_time_us = float(getattr(event, "cpu_time_total", 0.0) or 0.0)

    return {
        "name": name,
        "operator_family": classify_operator_family(name),
        "count": int(getattr(event, "count", 0) or 0),
        "input_shapes": normalize_input_shapes(getattr(event, "input_shapes", [])),
        "self_cuda_time_us": self_cuda_time_us,
        "cuda_time_us": cuda_time_us,
        "self_cpu_time_us": self_cpu_time_us,
        "cpu_time_us": cpu_time_us,
    }


def gen_input(tokenizer: Any, batch: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
    repeated = torch.cat([prompt_ids] * batch, dim=0)
    if repeated.shape[1] < seq_len:
        pad_id = tokenizer.eos_token_id or 0
        padding = torch.full(
            (batch, seq_len - repeated.shape[1]),
            pad_id,
            device=device,
            dtype=repeated.dtype,
        )
        repeated = torch.cat([repeated, padding], dim=1)
    else:
        repeated = repeated[:, :seq_len]
    attention_mask = torch.ones_like(repeated, device=device)
    return repeated, attention_mask


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_variant_model(args: argparse.Namespace, torch_dtype: torch.dtype):
    if args.variant == "baseline":
        tokenizer = AutoTokenizer.from_pretrained(args.baseline_model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.baseline_model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        source = args.baseline_model_id
    elif args.variant == "palu":
        model, tokenizer, palu_dir = load_palu_model(
            device=args.device,
            torch_dtype=torch_dtype,
            pattern=args.palu_pattern,
            baseline_id=args.baseline_model_id,
        )
        source = str(palu_dir)
    elif args.variant == "palu_grouped_bmm":
        model, tokenizer, palu_dir = load_palu_model(
            device=args.device,
            torch_dtype=torch_dtype,
            pattern=args.palu_pattern,
            baseline_id=args.baseline_model_id,
            reconstruct_strategy="grouped_bmm",
        )
        source = f"{palu_dir}::grouped_bmm"
    elif args.variant == "aligned_gac":
        model, tokenizer, palu_dir = load_palu_model(
            device=args.device,
            torch_dtype=torch_dtype,
            pattern=args.palu_aligned_pattern,
            baseline_id=args.baseline_model_id,
        )
        source = str(palu_dir)
    else:
        raise ValueError(f"Unsupported variant: {args.variant}")

    ensure_tokenizer_padding(tokenizer)
    if not any(parameter.is_cuda for parameter in model.parameters()):
        model.to(args.device)
    model.eval()
    return model, tokenizer, source


def profile_callable(
    stage_name: str,
    fn,
    warmup: int,
    device: str,
    trace_dir: Path | None = None,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn()
        torch.cuda.synchronize(device)

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_dir / f"{stage_name}_trace.json"))

    return {
        "events": [profiler_event_to_dict(event) for event in prof.key_averages()],
    }


def build_prefill_stage(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch: int,
    seq_len: int,
    warmup: int,
    trace_dir: Path | None,
) -> Dict[str, Any]:
    input_ids, attention_mask = gen_input(tokenizer, batch, seq_len, device)

    def run_prefill():
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    payload = profile_callable("prefill", run_prefill, warmup, device, trace_dir=trace_dir)
    payload["profile"] = {
        "batch": batch,
        "seq_len": seq_len,
        "mode": "forward",
    }
    return payload


def build_decode_stage(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch: int,
    context_len: int,
    warmup: int,
    trace_dir: Path | None,
) -> Dict[str, Any]:
    context_ids, context_attention_mask = gen_input(tokenizer, batch, context_len, device)

    with torch.inference_mode():
        cached = model(
            input_ids=context_ids,
            attention_mask=context_attention_mask,
            use_cache=True,
        )

    next_token_ids = context_ids[:, -1:]
    decode_attention_mask = torch.ones(
        (batch, context_len + 1),
        device=device,
        dtype=context_attention_mask.dtype,
    )

    def run_decode():
        with torch.inference_mode():
            return model(
                input_ids=next_token_ids,
                attention_mask=decode_attention_mask,
                past_key_values=cached.past_key_values,
                use_cache=True,
            )

    payload = profile_callable("decode", run_decode, warmup, device, trace_dir=trace_dir)
    payload["profile"] = {
        "batch": batch,
        "context_len": context_len,
        "decode_tokens": 1,
        "mode": "single_token_with_cache",
    }
    return payload


def build_config(args: argparse.Namespace, source: str) -> Dict[str, Any]:
    return {
        "variant": args.variant,
        "baseline_model_id": args.baseline_model_id,
        "dtype": args.dtype,
        "device": args.device,
        "stages": args.stage,
        "prefill_batch": args.prefill_batch,
        "prefill_seq_len": args.prefill_seq_len,
        "decode_batch": args.decode_batch,
        "decode_context_len": args.decode_context_len,
        "warmup": args.warmup,
        "model_source": source,
        "palu_pattern": args.palu_pattern if args.variant in {"palu", "palu_grouped_bmm"} else None,
        "palu_aligned_pattern": (
            args.palu_aligned_pattern if args.variant == "aligned_gac" else None
        ),
        "reconstruct_strategy": (
            "grouped_bmm" if args.variant == "palu_grouped_bmm"
            else "per_group" if args.variant in {"palu", "aligned_gac"}
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile real LLM inference operators")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["baseline", "palu", "palu_grouped_bmm", "aligned_gac"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--baseline-model-id", default=DEFAULT_BASELINE_MODEL_ID)
    parser.add_argument("--palu-pattern", default=DEFAULT_PALU_PATTERN)
    parser.add_argument("--palu-aligned-pattern", default=DEFAULT_PALU_ALIGNED_PATTERN)
    parser.add_argument("--stage", default="both", choices=["prefill", "decode", "both"])
    parser.add_argument("--prefill-batch", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=1024)
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--decode-context-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--export-trace", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real inference operator profiling")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces" if args.export_trace else None
    torch_dtype = get_torch_dtype(args.dtype)

    model, tokenizer, source = load_variant_model(args, torch_dtype)
    config = build_config(args, source)

    raw_payload: Dict[str, Any] = {
        "variant": args.variant,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_source": source,
        "parameter_count": count_parameters(model),
        "stages": {},
    }

    if args.stage in {"prefill", "both"}:
        raw_payload["stages"]["prefill"] = build_prefill_stage(
            model,
            tokenizer,
            args.device,
            args.prefill_batch,
            args.prefill_seq_len,
            args.warmup,
            trace_dir=trace_dir,
        )

    if args.stage in {"decode", "both"}:
        raw_payload["stages"]["decode"] = build_decode_stage(
            model,
            tokenizer,
            args.device,
            args.decode_batch,
            args.decode_context_len,
            args.warmup,
            trace_dir=trace_dir,
        )

    summary_payload = summarize_profile_run(raw_payload)
    env_payload = collect_environment()

    save_json(output_dir / "config.json", config)
    save_json(output_dir / "raw.json", raw_payload)
    save_json(output_dir / "summary.json", summary_payload)
    save_json(output_dir / "env.json", env_payload)

    run_summary = {
        "variant": args.variant,
        "output_dir": str(output_dir),
        "stages": list(raw_payload["stages"].keys()),
    }
    save_json(output_dir / "run_summary.json", run_summary)

    print(f"Wrote LLM inference operator profile to {output_dir}")


if __name__ == "__main__":
    main()
