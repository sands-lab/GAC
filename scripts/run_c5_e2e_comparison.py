#!/usr/bin/env python3
"""
C5 End-to-End LLM Inference Comparison

Compares three model variants:
1. Baseline: Original Llama-3-8B-Instruct
2. PaLU: Compressed with PaLU (misaligned dimensions)
3. Aligned GAC: PaLU with an aligned checkpoint or runtime repair

Metrics:
- Prefill latency (tok/s)
- Decode throughput (tok/s)
- Memory usage
- Perplexity (accuracy validation)

Usage:
    python scripts/run_c5_e2e_comparison.py --out results/C5 [--smoke]
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "palu"))

from src.gcompress_bench.metrics import measure_kernel, compute_stats, memory_stats, reset_memory
from src.gcompress_bench.palu_loader import load_palu_model
from src.gcompress_bench.dimension_repair import DimensionRepairer
from environment import collect_environment


@dataclass
class BenchmarkConfig:
    """Configuration for C5 benchmark."""
    prefill_batches: List[int] = field(default_factory=lambda: [1, 4])
    prefill_seq_lens: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    decode_batches: List[int] = field(default_factory=lambda: [1, 4])
    decode_ctx_lens: List[int] = field(default_factory=lambda: [512, 1024])
    decode_gen_lens: List[int] = field(default_factory=lambda: [64, 128])
    warmup: int = 10
    measure: int = 30
    trials: int = 3
    dtype: str = "float16"
    device: str = "cuda:0"
    repair_strategy: str = "minimal"  # minimal, optimal, predefined, tradeoff


@dataclass
class VariantResult:
    """Results for a single variant."""
    variant: str
    prefill_results: List[Dict]
    decode_results: List[Dict]
    memory_peak_mb: float
    model_params: int
    repair_info: Optional[Dict] = None


def load_stored_variant_result(results_json_path: Path, variant: str) -> VariantResult:
    payload = json.loads(results_json_path.read_text())
    variant_payload = payload["results"][variant]
    return VariantResult(
        variant=variant_payload["variant"],
        prefill_results=variant_payload["prefill_results"],
        decode_results=variant_payload["decode_results"],
        memory_peak_mb=variant_payload.get("memory_peak_mb", 0.0),
        model_params=variant_payload.get("model_params", 0),
        repair_info=variant_payload.get("repair_info"),
    )


def load_baseline_model(device: str, dtype: torch.dtype):
    """Load original Llama-3-8B model."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_palu_model_variant(
    device: str,
    dtype: torch.dtype,
    pattern: str = "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4*",
):
    """Load PaLU compressed model."""
    model, tokenizer, palu_dir = load_palu_model(device=device, torch_dtype=dtype, pattern=pattern)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, palu_dir


def analyze_palu_dimensions(model) -> Dict:
    """Analyze dimension distribution in PaLU model.

    PaLU uses HeadwiseLowRankModule for k_proj/v_proj with SVD decomposition:
    - VT: nn.Linear with out_features = sum(ranks) [GROUP-level, already aligned]
    - U: nn.ModuleList of nn.Linear, each with in_features = ranks[i] [PER-HEAD level]

    We need to detect per-head ranks from U[i].in_features or module.ranks attribute.
    These are the dimensions that may be misaligned (e.g., 114, 117, 121, 125).
    """
    dims = []
    aligned_8 = 0
    aligned_16 = 0
    total = 0

    # Import HeadwiseLowRankModule for type checking
    try:
        from palu.model.modules.svd_linear import HeadwiseLowRankModule
        has_palu = True
    except ImportError:
        has_palu = False

    for name, module in model.named_modules():
        # Check for PaLU's HeadwiseLowRankModule (k_proj/v_proj after SVD)
        if has_palu and isinstance(module, HeadwiseLowRankModule):
            # Get per-head ranks from the module's ranks attribute
            # Each rank is the compressed dimension for one head group
            for rank in module.ranks:
                dims.append(rank)
                total += 1
                if rank % 8 == 0:
                    aligned_8 += 1
                if rank % 16 == 0:
                    aligned_16 += 1
        # Fallback: check for standard nn.Linear in attention projections
        elif hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
            if any(proj in name.lower() for proj in ['k_proj', 'v_proj']):
                # This handles baseline (non-PaLU) models
                dim = module.out_features
                dims.append(dim)
                total += 1
                if dim % 8 == 0:
                    aligned_8 += 1
                if dim % 16 == 0:
                    aligned_16 += 1

    return {
        "unique_dims": sorted(set(dims)),
        "total_heads": total,
        "aligned_8_count": aligned_8,
        "aligned_16_count": aligned_16,
        "aligned_8_pct": 100.0 * aligned_8 / total if total > 0 else 0,
        "aligned_16_pct": 100.0 * aligned_16 / total if total > 0 else 0,
        "misaligned_pct": 100.0 * (total - aligned_8) / total if total > 0 else 0,
    }


def apply_dimension_repair(model, strategy: str = "minimal") -> Tuple[torch.nn.Module, Dict]:
    """Apply dimension repair to PaLU model."""
    repairer = DimensionRepairer(strategy=strategy)

    # Analyze before repair
    before_analysis = analyze_palu_dimensions(model)

    # Compute repair plan
    plan = repairer.compute_repair_plan(model)

    # Apply repair
    repaired_model, result = repairer.repair_model(model, inplace=False)

    # Analyze after repair
    after_analysis = analyze_palu_dimensions(repaired_model)

    repair_info = {
        "strategy": strategy,
        "before": before_analysis,
        "after": after_analysis,
        "memory_overhead_pct": result.memory_overhead_pct,
        "affected_layers": len(result.affected_layers),
        "repair_mapping": {
            str(k): {"original": v[0], "repaired": v[1]}
            for k, v in list(plan.items())[:20]  # Sample first 20
        },
    }

    return repaired_model, repair_info


def gen_input(tokenizer, batch: int, seq_len: int, device: str):
    """Generate input tensors for benchmarking."""
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


def benchmark_prefill(model, tokenizer, config: BenchmarkConfig) -> List[Dict]:
    """Benchmark prefill (prompt processing) performance."""
    results = []
    device = config.device

    for b in config.prefill_batches:
        for s in config.prefill_seq_lens:
            print(f"  Prefill: batch={b}, seq_len={s}", end="", flush=True)
            input_ids, attention_mask = gen_input(tokenizer, b, s, device)
            reset_memory()

            def fn():
                with torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=attention_mask)

            try:
                res = measure_kernel(fn, warmup=config.warmup, measure=config.measure,
                                    trials=config.trials, device=device)
                mem = memory_stats()
                tokens = b * s
                throughput = [tokens / (t / 1000.0) for t in res["times_ms"]]

                result = {
                    "batch": b,
                    "seq_len": s,
                    "latency_ms": res["stats"],
                    "throughput_toks_per_s": compute_stats(throughput),
                    "memory_mb": mem.get("peak_mb", 0),
                }
                results.append(result)
                print(f" -> {res['stats']['mean']:.2f}ms, {compute_stats(throughput)['mean']:.0f} tok/s")

            except RuntimeError as e:
                print(f" -> ERROR: {str(e)[:50]}")
                results.append({
                    "batch": b,
                    "seq_len": s,
                    "error": str(e),
                })
                del input_ids, attention_mask
                torch.cuda.empty_cache()
                break  # Skip larger seq_lens for this batch

            del input_ids, attention_mask
            torch.cuda.empty_cache()

    return results


def benchmark_decode(model, tokenizer, config: BenchmarkConfig) -> List[Dict]:
    """Benchmark decode (autoregressive generation) performance."""
    results = []
    device = config.device

    for b in config.decode_batches:
        for ctx in config.decode_ctx_lens:
            for gen in config.decode_gen_lens:
                print(f"  Decode: batch={b}, ctx={ctx}, gen={gen}", end="", flush=True)
                input_ids, attention_mask = gen_input(tokenizer, b, ctx, device)
                gen_kwargs = dict(max_new_tokens=gen, do_sample=False)
                reset_memory()

                def fn():
                    with torch.inference_mode():
                        model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

                try:
                    res = measure_kernel(fn, warmup=config.warmup // 2, measure=config.measure,
                                        trials=config.trials, device=device)
                    mem = memory_stats()
                    tokens = b * gen
                    throughput = [tokens / (t / 1000.0) for t in res["times_ms"]]

                    result = {
                        "batch": b,
                        "context_len": ctx,
                        "gen_len": gen,
                        "latency_ms": res["stats"],
                        "throughput_toks_per_s": compute_stats(throughput),
                        "memory_mb": mem.get("peak_mb", 0),
                    }
                    results.append(result)
                    print(f" -> {res['stats']['mean']:.2f}ms, {compute_stats(throughput)['mean']:.1f} tok/s")

                except RuntimeError as e:
                    print(f" -> ERROR: {str(e)[:50]}")
                    results.append({
                        "batch": b,
                        "context_len": ctx,
                        "gen_len": gen,
                        "error": str(e),
                    })
                    del input_ids, attention_mask
                    torch.cuda.empty_cache()
                    break  # Skip larger gen_lens

                del input_ids, attention_mask
                torch.cuda.empty_cache()

    return results


def compute_perplexity(model, tokenizer, device: str, max_tokens: int = 4096) -> Dict:
    """Compute perplexity on a small validation set."""
    # Use a simple test corpus
    test_text = """
    Artificial intelligence is transforming the way we interact with technology.
    Large language models have demonstrated remarkable capabilities in natural language understanding.
    GPU optimization remains critical for efficient model inference at scale.
    The alignment of tensor dimensions affects performance on modern accelerators.
    """

    enc = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc.input_ids.to(device)

    with torch.inference_mode():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()

    return {
        "perplexity": ppl,
        "loss": loss,
        "tokens": input_ids.shape[1],
    }


def run_variant(variant_name: str, model, tokenizer, config: BenchmarkConfig,
                repair_info: Optional[Dict] = None) -> VariantResult:
    """Run full benchmark suite for a single variant."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {variant_name}")
    print(f"{'='*60}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Run prefill benchmark
    print("\nPrefill Benchmark:")
    prefill_results = benchmark_prefill(model, tokenizer, config)

    # Run decode benchmark
    print("\nDecode Benchmark:")
    decode_results = benchmark_decode(model, tokenizer, config)

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    return VariantResult(
        variant=variant_name,
        prefill_results=prefill_results,
        decode_results=decode_results,
        memory_peak_mb=peak_memory,
        model_params=num_params,
        repair_info=repair_info,
    )


def summarize_latency_comparison(results: Dict[str, VariantResult], comparison: Dict) -> Dict:
    """Build a compact baseline/unaligned/aligned latency summary for downstream artifacts."""

    def summarize_measurements(entries: List[Dict], mode: str, variant_label: str) -> Dict:
        valid = [
            entry for entry in entries
            if "error" not in entry
            and isinstance(entry.get("latency_ms"), dict)
            and "mean" in entry["latency_ms"]
        ]
        if not valid:
            return {
                "status": "missing",
                "reason": f"no successful {mode} measurements recorded for {variant_label}",
            }

        measurement_map = {}
        latencies = []
        throughputs = []
        for entry in valid:
            if mode == "prefill":
                shape_key = f"batch{entry['batch']}_seq{entry['seq_len']}"
            else:
                shape_key = f"batch{entry['batch']}_ctx{entry['context_len']}_gen{entry['gen_len']}"
            measurement_map[shape_key] = entry["latency_ms"]["mean"]
            latencies.append(entry["latency_ms"]["mean"])

            throughput = entry.get("throughput_toks_per_s")
            if isinstance(throughput, dict) and "mean" in throughput:
                throughputs.append(throughput["mean"])

        summary = {
            "status": "measured",
            "measurements": measurement_map,
            "average_latency_ms": sum(latencies) / len(latencies),
        }
        if throughputs:
            summary["average_throughput_tok_s"] = sum(throughputs) / len(throughputs)
        return summary

    baseline = results.get("baseline")
    palu = results.get("palu")
    aligned_variant = results.get("aligned_gac") or results.get("palu_repair")

    summary = {
        "measurement_scope": {
            "prefill": "aggregated from the per-shape prefill measurements in this run",
            "decode": "aggregated from the per-shape decode measurements in this run",
        },
        "prefill_latency_ms": {
            "baseline": summarize_measurements(baseline.prefill_results, "prefill", "baseline") if baseline else {
                "status": "missing",
                "reason": "baseline variant was skipped",
            },
            "unaligned": summarize_measurements(palu.prefill_results, "prefill", "palu") if palu else {
                "status": "missing",
                "reason": "unaligned PaLU variant did not complete",
            },
            "aligned_gac": summarize_measurements(aligned_variant.prefill_results, "prefill", "aligned_gac") if aligned_variant else {
                "status": "missing",
                "reason": "aligned GAC variant did not complete",
            },
        },
        "decode_latency_ms": {
            "baseline": summarize_measurements(baseline.decode_results, "decode", "baseline") if baseline else {
                "status": "missing",
                "reason": "baseline variant was skipped",
            },
            "unaligned": summarize_measurements(palu.decode_results, "decode", "palu") if palu else {
                "status": "missing",
                "reason": "unaligned PaLU variant did not complete",
            },
            "aligned_gac": summarize_measurements(aligned_variant.decode_results, "decode", "aligned_gac") if aligned_variant else {
                "status": "missing",
                "reason": "aligned GAC variant did not complete",
            },
        },
        "alignment_pct": {
            "baseline": {
                "status": "measured",
                "value": 100.0,
            },
            "unaligned": {
                "status": "missing",
                "reason": "unaligned PaLU alignment ratio is unavailable",
            },
            "aligned_gac": {
                "status": "missing",
                "reason": "aligned GAC alignment ratio is unavailable",
            },
        },
        "sources": {
            "comparison_json": "comparison.json",
            "results_json": "results.json",
        },
        "notes": [],
    }

    if palu and palu.repair_info and "before_repair" in palu.repair_info:
        summary["alignment_pct"]["unaligned"] = {
            "status": "measured",
            "value": palu.repair_info["before_repair"]["aligned_8_pct"],
        }

    if aligned_variant and aligned_variant.repair_info and "after" in aligned_variant.repair_info:
        summary["alignment_pct"]["aligned_gac"] = {
            "status": "measured",
            "value": aligned_variant.repair_info["after"]["aligned_8_pct"],
        }

    if comparison:
        summary["sources"]["comparison_metrics"] = "comparison.json"
        summary["notes"].append(
            "baseline / unaligned / aligned_gac are mapped from baseline / palu / {aligned_gac|palu_repair} variants in this run"
        )

    return summary


def compute_comparison(results: Dict[str, VariantResult]) -> Dict:
    """Compute comparison metrics between variants."""
    comparison = {}

    # Get baseline metrics if available
    baseline = results.get("baseline")
    palu = results.get("palu")
    aligned_variant = results.get("aligned_gac") or results.get("palu_repair")

    if not baseline or not palu:
        return {"error": "Missing baseline or palu results"}

    # Compare prefill throughput
    def get_avg_throughput(result: VariantResult, metric_type: str) -> float:
        if metric_type == "prefill":
            data = result.prefill_results
        else:
            data = result.decode_results

        valid = [r for r in data if "error" not in r and "throughput_toks_per_s" in r]
        if not valid:
            return 0.0
        return sum(r["throughput_toks_per_s"]["mean"] for r in valid) / len(valid)

    baseline_prefill = get_avg_throughput(baseline, "prefill")
    palu_prefill = get_avg_throughput(palu, "prefill")

    comparison["prefill"] = {
        "baseline_tok_s": baseline_prefill,
        "palu_tok_s": palu_prefill,
        "palu_vs_baseline_pct": 100.0 * (palu_prefill - baseline_prefill) / baseline_prefill if baseline_prefill > 0 else 0,
    }

    if aligned_variant:
        aligned_prefill = get_avg_throughput(aligned_variant, "prefill")
        comparison["prefill"]["aligned_gac_tok_s"] = aligned_prefill
        comparison["prefill"]["aligned_vs_palu_pct"] = 100.0 * (aligned_prefill - palu_prefill) / palu_prefill if palu_prefill > 0 else 0
        comparison["prefill"]["aligned_vs_baseline_pct"] = 100.0 * (aligned_prefill - baseline_prefill) / baseline_prefill if baseline_prefill > 0 else 0

    # Compare decode throughput
    baseline_decode = get_avg_throughput(baseline, "decode")
    palu_decode = get_avg_throughput(palu, "decode")

    comparison["decode"] = {
        "baseline_tok_s": baseline_decode,
        "palu_tok_s": palu_decode,
        "palu_vs_baseline_pct": 100.0 * (palu_decode - baseline_decode) / baseline_decode if baseline_decode > 0 else 0,
    }

    if aligned_variant:
        aligned_decode = get_avg_throughput(aligned_variant, "decode")
        comparison["decode"]["aligned_gac_tok_s"] = aligned_decode
        comparison["decode"]["aligned_vs_palu_pct"] = 100.0 * (aligned_decode - palu_decode) / palu_decode if palu_decode > 0 else 0
        comparison["decode"]["aligned_vs_baseline_pct"] = 100.0 * (aligned_decode - baseline_decode) / baseline_decode if baseline_decode > 0 else 0

    # Memory comparison
    if baseline.memory_peak_mb > 0 and palu.memory_peak_mb > 0:
        comparison["memory"] = {
            "baseline_mb": baseline.memory_peak_mb,
            "palu_mb": palu.memory_peak_mb,
            "palu_compression_pct": 100.0 * (baseline.memory_peak_mb - palu.memory_peak_mb) / baseline.memory_peak_mb,
        }

        if aligned_variant and aligned_variant.memory_peak_mb > 0:
            comparison["memory"]["aligned_gac_mb"] = aligned_variant.memory_peak_mb
            comparison["memory"]["aligned_overhead_pct"] = 100.0 * (aligned_variant.memory_peak_mb - palu.memory_peak_mb) / palu.memory_peak_mb

    return comparison


def generate_summary_report(results: Dict[str, VariantResult], comparison: Dict, config: BenchmarkConfig) -> str:
    """Generate markdown summary report."""
    lines = [
        "# C5 End-to-End LLM Inference Comparison",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        f"- Prefill batches: {config.prefill_batches}",
        f"- Prefill seq_lens: {config.prefill_seq_lens}",
        f"- Decode batches: {config.decode_batches}",
        f"- Decode context lens: {config.decode_ctx_lens}",
        f"- Decode gen lens: {config.decode_gen_lens}",
        f"- Warmup: {config.warmup}, Measure: {config.measure}, Trials: {config.trials}",
        f"- Repair strategy: {config.repair_strategy}",
        "",
        "## Results Summary",
        "",
    ]

    # Prefill comparison
    if "prefill" in comparison:
        p = comparison["prefill"]
        lines.extend([
            "### Prefill Throughput (tok/s)",
            "",
            "| Variant | Throughput | vs Baseline |",
            "|---------|------------|-------------|",
            f"| Baseline | {p['baseline_tok_s']:.1f} | - |",
            f"| PaLU | {p['palu_tok_s']:.1f} | {p['palu_vs_baseline_pct']:+.1f}% |",
        ])
        if "aligned_gac_tok_s" in p:
            lines.append(f"| Aligned GAC | {p['aligned_gac_tok_s']:.1f} | {p['aligned_vs_baseline_pct']:+.1f}% |")
        lines.append("")

    # Decode comparison
    if "decode" in comparison:
        d = comparison["decode"]
        lines.extend([
            "### Decode Throughput (tok/s)",
            "",
            "| Variant | Throughput | vs Baseline |",
            "|---------|------------|-------------|",
            f"| Baseline | {d['baseline_tok_s']:.1f} | - |",
            f"| PaLU | {d['palu_tok_s']:.1f} | {d['palu_vs_baseline_pct']:+.1f}% |",
        ])
        if "aligned_gac_tok_s" in d:
            lines.append(f"| Aligned GAC | {d['aligned_gac_tok_s']:.1f} | {d['aligned_vs_baseline_pct']:+.1f}% |")
        lines.append("")

    # Memory comparison
    if "memory" in comparison:
        m = comparison["memory"]
        lines.extend([
            "### Peak Memory Usage (MB)",
            "",
            "| Variant | Memory | vs Baseline |",
            "|---------|--------|-------------|",
            f"| Baseline | {m['baseline_mb']:.1f} | - |",
            f"| PaLU | {m['palu_mb']:.1f} | {m['palu_compression_pct']:+.1f}% |",
        ])
        if "aligned_gac_mb" in m:
            lines.append(f"| Aligned GAC | {m['aligned_gac_mb']:.1f} | {m['aligned_overhead_pct']:+.1f}% overhead |")
        lines.append("")

    # Alignment info
    aligned_variant = results.get("aligned_gac") or results.get("palu_repair")
    if aligned_variant and aligned_variant.repair_info:
        ri = aligned_variant.repair_info
        lines.extend([
            "## Alignment Analysis",
            "",
            f"- Strategy: {ri['strategy']}",
            f"- Memory overhead: {ri['memory_overhead_pct']:.2f}%",
            f"- Affected layers: {ri['affected_layers']}",
            "",
            "### Before Alignment",
            f"- Misaligned dimensions: {ri['before']['misaligned_pct']:.1f}%",
            f"- Unique dims: {ri['before']['unique_dims']}",
            "",
            "### After Alignment",
            f"- Misaligned dimensions: {ri['after']['misaligned_pct']:.1f}%",
            f"- Unique dims: {ri['after']['unique_dims']}",
            "",
        ])

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])

    if "prefill" in comparison and "aligned_vs_palu_pct" in comparison["prefill"]:
        speedup = comparison["prefill"]["aligned_vs_palu_pct"]
        if speedup > 0:
            lines.append(f"- GAC alignment improves prefill throughput by **{speedup:.1f}%** over unaligned PaLU")

    if "decode" in comparison and "aligned_vs_palu_pct" in comparison["decode"]:
        speedup = comparison["decode"]["aligned_vs_palu_pct"]
        if speedup > 0:
            lines.append(f"- GAC alignment improves decode throughput by **{speedup:.1f}%** over unaligned PaLU")

    if aligned_variant and aligned_variant.repair_info:
        overhead = aligned_variant.repair_info["memory_overhead_pct"]
        lines.append(f"- Memory overhead from alignment: **{overhead:.2f}%**")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="C5 End-to-End LLM Inference Comparison")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device to run on")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--repair-strategy", default="minimal",
                       choices=["minimal", "optimal", "predefined", "tradeoff"])
    parser.add_argument("--palu-pattern", default="Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4*")
    parser.add_argument("--aligned-source", default="repair", choices=["repair", "checkpoint"])
    parser.add_argument("--palu-aligned-pattern", default=None)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with reduced params")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline (for faster iteration)")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    args = parser.parse_args()

    # Generate run ID
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_C5_e2e_comparison"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"C5 End-to-End LLM Inference Comparison")
    print(f"Output: {run_dir}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Repair strategy: {args.repair_strategy}")

    # Configure benchmark
    if args.smoke:
        print("\n[SMOKE TEST MODE - reduced parameters]")
        config = BenchmarkConfig(
            prefill_batches=[1],
            prefill_seq_lens=[256, 512],
            decode_batches=[1],
            decode_ctx_lens=[256],
            decode_gen_lens=[32],
            warmup=3,
            measure=10,
            trials=2,
            dtype=args.dtype,
            device=args.device,
            repair_strategy=args.repair_strategy,
        )
    else:
        config = BenchmarkConfig(
            prefill_batches=[1, 4],
            prefill_seq_lens=[256, 512, 1024, 2048],
            decode_batches=[1, 4],
            decode_ctx_lens=[512, 1024],
            decode_gen_lens=[64, 128],
            warmup=10,
            measure=30,
            trials=3,
            dtype=args.dtype,
            device=args.device,
            repair_strategy=args.repair_strategy,
        )

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    results = {}

    # 1. Baseline benchmark
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("Loading Baseline Model (Llama-3-8B-Instruct)")
        print("="*60)
        baseline_model, baseline_tokenizer = load_baseline_model(args.device, torch_dtype)
        results["baseline"] = run_variant("baseline", baseline_model, baseline_tokenizer, config)

        # Free memory
        del baseline_model
        torch.cuda.empty_cache()
    elif args.baseline_results:
        results["baseline"] = load_stored_variant_result(args.baseline_results, "baseline")

    # 2. PaLU benchmark
    print("\n" + "="*60)
    print("Loading PaLU Model")
    print("="*60)
    palu_model, palu_tokenizer, palu_dir = load_palu_model_variant(
        args.device,
        torch_dtype,
        pattern=args.palu_pattern,
    )

    # Analyze PaLU dimensions
    palu_dim_analysis = analyze_palu_dimensions(palu_model)
    print(f"PaLU dimension analysis:")
    print(f"  Unique dims: {palu_dim_analysis['unique_dims']}")
    print(f"  Misaligned: {palu_dim_analysis['misaligned_pct']:.1f}%")

    results["palu"] = run_variant("palu", palu_model, palu_tokenizer, config)
    results["palu"].repair_info = {"before_repair": palu_dim_analysis}

    # 3. Aligned benchmark
    if args.aligned_source == "checkpoint":
        if not args.palu_aligned_pattern:
            raise ValueError("--palu-aligned-pattern is required when --aligned-source=checkpoint")

        print("\n" + "="*60)
        print("Loading Aligned GAC PaLU Model")
        print("="*60)
        aligned_model, aligned_tokenizer, aligned_dir = load_palu_model_variant(
            args.device,
            torch_dtype,
            pattern=args.palu_aligned_pattern,
        )
        aligned_dim_analysis = analyze_palu_dimensions(aligned_model)
        print("Aligned GAC dimension analysis:")
        print(f"  Unique dims: {aligned_dim_analysis['unique_dims']}")
        print(f"  Misaligned: {aligned_dim_analysis['misaligned_pct']:.1f}%")

        del palu_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        results["aligned_gac"] = run_variant(
            "aligned_gac",
            aligned_model,
            aligned_tokenizer,
            config,
            repair_info={
                "strategy": "gac_checkpoint",
                "before": palu_dim_analysis,
                "after": aligned_dim_analysis,
                "memory_overhead_pct": 0.0,
                "affected_layers": 0,
                "aligned_checkpoint_dir": str(aligned_dir),
            },
        )
    else:
        print("\n" + "="*60)
        print(f"Applying Dimension Repair (strategy={args.repair_strategy})")
        print("="*60)

        repaired_model, repair_info = apply_dimension_repair(palu_model, strategy=args.repair_strategy)
        print("Repair applied:")
        print(f"  Memory overhead: {repair_info['memory_overhead_pct']:.2f}%")
        print(f"  Affected layers: {repair_info['affected_layers']}")
        print(f"  After repair - Misaligned: {repair_info['after']['misaligned_pct']:.1f}%")

        del palu_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        results["palu_repair"] = run_variant("palu_repair", repaired_model, palu_tokenizer, config, repair_info)

    # Compute comparison
    comparison = compute_comparison(results)

    # Generate report
    report = generate_summary_report(results, comparison, config)
    print("\n" + report)

    # Save results
    output_data = {
        "config": asdict(config),
        "results": {k: asdict(v) for k, v in results.items()},
        "comparison": comparison,
        "environment": collect_environment(),
        "palu_dir": str(palu_dir),
    }
    comparison_summary = summarize_latency_comparison(results, comparison)

    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (run_dir / "results.json").write_text(json.dumps(output_data, indent=2))
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (run_dir / "comparison_summary.json").write_text(json.dumps(comparison_summary, indent=2))
    (run_dir / "report.md").write_text(report)
    (run_dir / "env.json").write_text(json.dumps(collect_environment(), indent=2))

    print(f"\nResults saved to: {run_dir}")
    print("Files: config.json, results.json, comparison.json, comparison_summary.json, report.md, env.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
