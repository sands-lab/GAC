"""
Unified evaluation script for ASVD and LLM-Pruner models.
Measures: Accuracy (PIQA, HellaSwag) and Decode Latency.

Usage:
    python scripts/unified_eval.py --method asvd --variant aligned
    python scripts/unified_eval.py --method llmpruner --variant pruned_r8
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add local ASVD wrappers first, then third_party dependencies.
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "LLM-Pruner"))
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "ASVD4LLM"))
sys.path.insert(0, str(SCRIPT_DIR / "scripts" / "asvd_simple"))


def resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def load_causal_lm(model_ref, *, trust_remote_code=True):
    """Load a causal LM with or without accelerate/device_map support."""
    kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": trust_remote_code,
    }
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["device_map"] = "auto"
        return AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def measure_decode_latency(model, tokenizer, prompt_len=128, gen_tokens=64,
                           n_warmup=3, n_measure=10):
    """Measure decode latency under a fixed-length contract."""
    device = next(model.parameters()).device
    pad_token_id = resolve_pad_token_id(tokenizer)

    # Create input prompt
    input_ids = torch.randint(1, 1000, (1, prompt_len), device=device)
    gen_kwargs = dict(
        max_new_tokens=gen_tokens,
        min_new_tokens=gen_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(input_ids, **gen_kwargs)

    torch.cuda.synchronize()

    # Measure
    latencies = []
    actual_new_tokens = set()
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(input_ids, **gen_kwargs)

        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
        actual_new_tokens.add(int(outputs.shape[1] - input_ids.shape[1]))

    if len(actual_new_tokens) != 1:
        raise RuntimeError(
            f"Fixed-length decode contract violated: observed actual_new_tokens={sorted(actual_new_tokens)}"
        )

    actual_new_tokens_value = next(iter(actual_new_tokens))
    total_mean = np.mean(latencies)
    total_std = np.std(latencies)
    per_token = total_mean / actual_new_tokens_value
    tokens_per_sec = actual_new_tokens_value / (total_mean / 1000)

    return {
        "prompt_len": prompt_len,
        "requested_new_tokens": gen_tokens,
        "actual_new_tokens": actual_new_tokens_value,
        "decode_length_mode": "fixed_new_tokens",
        "generation_guard": "max_new_tokens=gen, min_new_tokens=gen",
        "warmup_count": n_warmup,
        "measure_count": n_measure,
        "total_mean_ms": total_mean,
        "total_std_ms": total_std,
        "per_token_ms": per_token,
        "tokens_per_sec": tokens_per_sec,
        "count": n_measure,
    }


def evaluate_accuracy(model, tokenizer, tasks=["piqa", "hellaswag"], limit=200):
    """Evaluate on lm-eval tasks."""
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=limit,
            batch_size=1,
        )

        accuracy = {}
        for task in tasks:
            if task in results["results"]:
                # Try different accuracy key names
                task_results = results["results"][task]
                for key in ["acc,none", "acc", "accuracy"]:
                    if key in task_results:
                        accuracy[task] = task_results[key]
                        break
        return accuracy
    except Exception as e:
        print(f"lm-eval failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _jsonify_sensitivity(sensitivity):
    return {
        layer: {str(k): float(v) for k, v in ratios.items()}
        for layer, ratios in sensitivity.items()
    }


def _dejsonify_sensitivity(raw_sensitivity):
    return {
        layer: {float(k): float(v) for k, v in ratios.items()}
        for layer, ratios in raw_sensitivity.items()
    }


def load_asvd_model(model_id, variant, results_dir, sensitivity_metric="ppl"):
    """Load ASVD compressed model."""
    from datautils import get_calib_data
    from act_aware_utils import calib_input_distribution
    from sensitivity_simple import calib_sensitivity_ppl, calib_sensitivity_stable_rank
    from binary_search_simple import binary_search_truncation_rank

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if variant == "baseline":
        model = load_causal_lm(model_id)
        return model, tokenizer

    # Load compressed model
    rank_align = 8 if variant == "aligned" else 1

    model = load_causal_lm(model_id)

    # Load calibration data
    calib_loader = get_calib_data("wikitext2", tokenizer, model_id, 32, seed=42)

    # Calibrate activation distribution
    calib_input_distribution(model, calib_loader, "abs_mean", use_cache=True)

    # Load or compute sensitivity
    sensitivity_file = Path(results_dir) / f"sensitivity_{sensitivity_metric}.json"
    if sensitivity_file.exists():
        with open(sensitivity_file) as f:
            sensitivity = _dejsonify_sensitivity(json.load(f))
    else:
        print(f"Computing ASVD {sensitivity_metric} sensitivity...")

        class Args:
            param_ratio_target = 0.85
            n_calib_samples = 32
            calib_dataset = "wikitext2"
            scaling_method = "abs_mean"
            alpha = 0.5
            compress_kv_cache = False

        if sensitivity_metric == "ppl":
            sensitivity = calib_sensitivity_ppl(model, calib_loader, Args(), use_cache=True)
        elif sensitivity_metric == "stable_rank":
            sensitivity = calib_sensitivity_stable_rank(model, calib_loader, Args(), use_cache=True)
        else:
            raise ValueError(f"Unsupported ASVD sensitivity metric: {sensitivity_metric}")

        sensitivity_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sensitivity_file, 'w') as f:
            json.dump(_jsonify_sensitivity(sensitivity), f, indent=2)

    # Apply compression
    class CompressionArgs:
        param_ratio_target = 0.85
        ppl_target = -1
        n_calib_samples = 32
        alpha = 0.5
        act_aware = True
        sigma_fuse = "UV"
        compress_kv_cache = False

    CompressionArgs.rank_align = rank_align
    binary_search_truncation_rank(model, sensitivity, calib_loader, CompressionArgs())

    return model, tokenizer


def load_llmpruner_model(model_id, variant, results_dir):
    """Load LLM-Pruner compressed model."""
    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from LLMPruner.datasets.example_samples import get_examples
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if variant == "baseline":
        model = load_causal_lm(model_id)
        return model, tokenizer

    # Load pruned model checkpoint
    checkpoint_dir = Path(results_dir) / f"{variant}_checkpoint"

    if checkpoint_dir.exists():
        print(f"Loading checkpoint from {checkpoint_dir}")
        model = load_causal_lm(checkpoint_dir)
        return model, tokenizer

    model = load_causal_lm(model_id)
    device = next(model.parameters()).device
    forward_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(device)

    round_to = 8 if variant == "pruned_r8" else None
    imp = llama_pruner.TaylorImportance(group_reduction='sum', taylor='param_first')
    kwargs = {
        "importance": imp,
        "global_pruning": True,
        "iterative_steps": 1,
        "ch_sparsity": 0.15,
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim
            for layer in model.model.layers
        },
        "customized_pruners": {LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
        "root_module_types": None,
        "root_instances": (
            [model.model.layers[i].self_attn.k_proj for i in range(3, 31)] +
            [model.model.layers[i].mlp.gate_proj for i in range(3, 31)]
        ),
    }
    if round_to is not None:
        kwargs["round_to"] = round_to

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        **kwargs,
        output_transform=lambda out: out.logits if hasattr(out, 'logits') else out[0],
    )
    model.zero_grad()

    print(f"Checkpoint not found at {checkpoint_dir}; building {variant} on the fly.")
    example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(device)
    loss = model(example_prompts, labels=example_prompts).loss
    loss.backward()
    pruner.step()

    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.head_dim
        layer.self_attn.num_key_value_heads = (
            layer.self_attn.k_proj.weight.shape[0] // layer.self_attn.head_dim
        )

    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["asvd", "llmpruner"])
    parser.add_argument("--variant", type=str, required=True,
                        help="baseline, aligned, unaligned (ASVD) or baseline, pruned, pruned_r8 (LLM-Pruner)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/unified_eval")
    parser.add_argument("--tasks", type=str, nargs="+", default=["piqa", "hellaswag"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--gen_tokens", type=int, default=64)
    parser.add_argument("--decode-warmup", type=int, default=3)
    parser.add_argument("--decode-measure", type=int, default=10)
    parser.add_argument("--asvd-sensitivity", type=str, default="ppl", choices=["ppl", "stable_rank"])
    parser.add_argument("--skip_accuracy", action="store_true")
    parser.add_argument("--skip_decode", action="store_true")
    args = parser.parse_args()

    if args.results_dir is None:
        if args.method == "asvd":
            args.results_dir = "results/asvd_simple"
        else:
            args.results_dir = "results/llmpruner_llama3_v2"

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"Unified Evaluation: {args.method.upper()} - {args.variant}")
    print(f"Model: {args.model_id}")
    print("=" * 60)

    # Load model
    print("\n[Step 1] Loading model...")
    if args.method == "asvd":
        model, tokenizer = load_asvd_model(
            args.model_id,
            args.variant,
            args.results_dir,
            sensitivity_metric=args.asvd_sensitivity,
        )
    else:
        model, tokenizer = load_llmpruner_model(args.model_id, args.variant, args.results_dir)

    results = {
        "method": args.method,
        "variant": args.variant,
        "model_id": args.model_id,
        "decode_contract": {
            "prompt_len": args.prompt_len,
            "requested_new_tokens": args.gen_tokens,
            "decode_warmup": args.decode_warmup,
            "decode_measure": args.decode_measure,
        },
    }
    if args.method == "asvd":
        results["construction"] = {
            "sensitivity_metric": args.asvd_sensitivity,
            "param_ratio_target": 0.85,
            "rank_align": 8 if args.variant == "aligned" else (1 if args.variant == "unaligned" else None),
        }

    # Measure accuracy
    if not args.skip_accuracy:
        print(f"\n[Step 2] Evaluating accuracy on {args.tasks}...")
        accuracy = evaluate_accuracy(model, tokenizer, args.tasks, args.limit)
        results["accuracy"] = accuracy
        print(f"  Accuracy: {accuracy}")

    # Measure decode latency
    if not args.skip_decode:
        print(f"\n[Step 3] Measuring decode latency...")
        decode_latency = measure_decode_latency(
            model,
            tokenizer,
            args.prompt_len,
            args.gen_tokens,
            n_warmup=args.decode_warmup,
            n_measure=args.decode_measure,
        )
        results["decode_latency"] = decode_latency
        print(f"  Decode: {decode_latency['per_token_ms']:.2f} ms/token, "
              f"{decode_latency['tokens_per_sec']:.1f} tok/s")

    # Save results
    output_file = Path(args.output) / f"{args.method}_{args.variant}_eval.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
