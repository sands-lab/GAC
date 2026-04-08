"""
Inference benchmarking for Llama-3-8B baseline and PaLU variant.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import measure_kernel, compute_stats, memory_stats, reset_memory
from environment import collect_environment
from .palu_loader import load_palu_model

DEFAULT_BASELINE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_model(variant: str, device: str, dtype_str: str = "float16", baseline_model_id: str = DEFAULT_BASELINE_MODEL_ID):
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    if variant == "baseline":
        tokenizer = AutoTokenizer.from_pretrained(baseline_model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            baseline_model_id,
            dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        palu_dir = None
    elif variant == "palu":
        model, tokenizer, palu_dir = load_palu_model(device=device, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model, tokenizer, palu_dir


def gen_input(tokenizer, batch: int, seq_len: int, device: str):
    prompt_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
    repeat = torch.cat([prompt_ids] * batch, dim=0)
    if repeat.shape[1] < seq_len:
        pad_id = tokenizer.eos_token_id or 0
        pad = torch.full((batch, seq_len - repeat.shape[1]), pad_id, device=device, dtype=repeat.dtype)
        repeat = torch.cat([repeat, pad], dim=1)
    else:
        repeat = repeat[:, :seq_len]
    attn = torch.ones_like(repeat, device=device)
    return repeat, attn


def benchmark_prefill(model, tokenizer, device, batches, seq_lens, warmup, measure, trials):
    prefill_results = []
    for b in batches:
        for s in seq_lens:
            input_ids, attention_mask = gen_input(tokenizer, b, s, device)
            reset_memory()
            def fn():
                with torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=attention_mask)
            try:
                res = measure_kernel(fn, warmup=warmup, measure=measure, trials=trials, device=device)
                mem = memory_stats()
                tokens = b * s
                throughput = [tokens / (t / 1000.0) for t in res["times_ms"]]
                prefill_results.append({
                    "batch": b,
                    "seq_len": s,
                    "timing": res["stats"],
                    "throughput_toks_per_s": compute_stats(throughput),
                    "memory": mem,
                })
            except RuntimeError as e:
                # OOM or other CUDA error: record and skip this (and larger seq for same batch)
                prefill_results.append({
                    "batch": b,
                    "seq_len": s,
                    "error": str(e),
                })
                # For this batch size, larger seq_len will be worse; break inner loop
                del input_ids, attention_mask
                torch.cuda.empty_cache()
                break
            del input_ids, attention_mask
            torch.cuda.empty_cache()
    return prefill_results


def benchmark_decode(model, tokenizer, device, batches, ctx_lens, gen_lens, warmup, measure, trials):
    decode_results = []
    for b in batches:
        for ctx in ctx_lens:
            for gen in gen_lens:
                input_ids, attention_mask = gen_input(tokenizer, b, ctx, device)
                gen_kwargs = dict(max_new_tokens=gen, do_sample=False)
                reset_memory()
                def fn():
                    with torch.inference_mode():
                        model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
                try:
                    res = measure_kernel(fn, warmup=warmup, measure=measure, trials=trials, device=device)
                    mem = memory_stats()
                    tokens = b * gen
                    throughput = [tokens / (t / 1000.0) for t in res["times_ms"]]
                    decode_results.append({
                        "batch": b,
                        "context_len": ctx,
                        "gen_len": gen,
                        "timing": res["stats"],
                        "throughput_toks_per_s": compute_stats(throughput),
                        "memory": mem,
                    })
                except RuntimeError as e:
                    decode_results.append({
                        "batch": b,
                        "context_len": ctx,
                        "gen_len": gen,
                        "error": str(e),
                    })
                    del input_ids, attention_mask
                    torch.cuda.empty_cache()
                    # For this (b, ctx), larger gen will be worse; break inner gen loop
                    break
                del input_ids, attention_mask
                torch.cuda.empty_cache()
    return decode_results


def save_results(run_dir: Path, config: dict, raw: dict, summary: dict, run_summary: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    env = collect_environment()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    (run_dir / "raw.json").write_text(json.dumps(raw, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "run_summary.md").write_text(run_summary)


def build_summary(prefill, decode):
    return {
        "prefill_count": len(prefill),
        "decode_count": len(decode),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM inference benchmark")
    parser.add_argument("--variant", choices=["baseline", "palu"], required=True)
    parser.add_argument("--suite", choices=["infer_sweep"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--baseline-model-id", default=DEFAULT_BASELINE_MODEL_ID)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-prefill-len", type=int, default=None, help="Cap prefill seq len for smoke tests")
    parser.add_argument("--max-decode-ctx", type=int, default=None, help="Cap decode context len for smoke tests")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.variant}_{args.suite}"
    run_dir = args.out / run_id

    model, tokenizer, palu_dir = load_model(args.variant, args.device, args.dtype, args.baseline_model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    config = {
        "variant": args.variant,
        "suite": args.suite,
        "device": args.device,
        "dtype": args.dtype,
        "baseline_model_id": args.baseline_model_id,
        "palu_dir": str(palu_dir) if palu_dir else None,
    }

    prefill_batches = [1, 4, 8]
    prefill_seqs = [256, 512, 1024, 2048, 4096]
    if args.max_prefill_len:
        prefill_seqs = [s for s in prefill_seqs if s <= args.max_prefill_len]
    decode_batches = [1, 4]
    decode_ctx = [512, 1024, 2048]
    if args.max_decode_ctx:
        decode_ctx = [c for c in decode_ctx if c <= args.max_decode_ctx]
    decode_gen = [64, 128]

    prefill = benchmark_prefill(model, tokenizer, args.device, prefill_batches, prefill_seqs, warmup=10, measure=30, trials=3)
    decode = benchmark_decode(model, tokenizer, args.device, decode_batches, decode_ctx, decode_gen, warmup=5, measure=30, trials=3)

    raw = {
        "prefill": prefill,
        "decode": decode,
    }
    summary = build_summary(prefill, decode)
    run_summary = f"# Run summary\\n\\nVariant: {args.variant}\\nSuite: {args.suite}\\nRun ID: {run_id}\\n"

    save_results(run_dir, config, raw, summary, run_summary)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
