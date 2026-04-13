#!/usr/bin/env python3
"""Generate a kernel-level GEMM root-cause analysis for PaLU inference profiles."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.gcompress_bench.operator_profile_root_cause import (
    BUCKET_ORDER,
    compare_root_cause_stages,
    summarize_root_cause_stage,
)


VARIANT_ORDER = ("baseline", "palu", "aligned_gac")
COMPARISON_ORDER = (
    ("palu_vs_baseline", "baseline", "palu"),
    ("aligned_gac_vs_palu", "palu", "aligned_gac"),
    ("aligned_gac_vs_baseline", "baseline", "aligned_gac"),
)
BUCKET_COLUMN_ORDER = (
    "dispatch_ops",
    "alignment_sensitive_kernels",
    "align8_tensorcore_kernels",
    "gemv_kernels",
    "attention_leakage",
    "other_gemm_kernels",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def display_path(path: Path) -> str:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def format_ms(value: float) -> str:
    return f"{value:.2f}"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def format_signed_ms(value: float) -> str:
    return f"{value:+.2f}"


def build_stage_summary(
    stage_name: str,
    raw_variants: Dict[str, Dict[str, Any]],
    config_variants: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    variants: Dict[str, Any] = {}
    for variant_name in VARIANT_ORDER:
        raw_stage = raw_variants[variant_name]["stages"][stage_name]
        variants[variant_name] = summarize_root_cause_stage(raw_stage.get("events", []))
        variants[variant_name]["profile"] = raw_stage.get("profile", {})
        variants[variant_name]["config"] = {
            "variant": config_variants[variant_name].get("variant"),
            "device": config_variants[variant_name].get("device"),
            "dtype": config_variants[variant_name].get("dtype"),
        }

    comparisons: Dict[str, Any] = {}
    for comparison_name, reference_name, candidate_name in COMPARISON_ORDER:
        comparisons[comparison_name] = compare_root_cause_stages(
            variants[reference_name],
            variants[candidate_name],
        )

    palu_summary = variants["palu"]
    aligned_vs_palu = comparisons["aligned_gac_vs_palu"]
    prefill_attention_ms = palu_summary["bucket_totals"]["attention_leakage"]["self_cuda_time_ms"]
    low_alignment_ms = palu_summary["bucket_totals"]["alignment_sensitive_kernels"]["self_cuda_time_ms"]
    gemv_ms = palu_summary["bucket_totals"]["gemv_kernels"]["self_cuda_time_ms"]

    if stage_name == "prefill":
        takeaway = (
            "`palu` prefill is dispatch-heavy rather than purely kernel-heavy: "
            f"`aten::mm`-style dispatch contributes {format_ms(palu_summary['dispatch_total_ms'])} ms "
            f"({format_pct(palu_summary['dispatch_share_of_selected_pct'])}) of the selected view, "
            f"while `align1/align2` alignment-sensitive kernels contribute only {format_ms(low_alignment_ms)} ms. "
            f"`aligned_gac` removes {format_ms(aligned_vs_palu['bucket_deltas']['alignment_sensitive_kernels']['delta_ms'])} ms "
            "from that alignment-sensitive tail, but some of the gain is offset by already-large tensorcore kernels. "
            f"Prefill also carries {format_ms(prefill_attention_ms)} ms of attention leakage from flash-attention kernels."
        )
    else:
        takeaway = (
            "`palu` decode has no large 8-aligned cliff to recover: the `gemv` tail is only "
            f"{format_ms(gemv_ms)} ms, while dispatch still contributes "
            f"{format_ms(palu_summary['dispatch_total_ms'])} ms "
            f"({format_pct(palu_summary['dispatch_share_of_selected_pct'])}) of the selected view. "
            f"`aligned_gac` changes the `gemv` bucket by "
            f"{format_signed_ms(aligned_vs_palu['bucket_deltas']['gemv_kernels']['delta_ms'])} ms, "
            "so decode mostly reshuffles small GEMV variants instead of eliminating a dominant kernel."
        )

    return {
        "variants": variants,
        "comparisons": comparisons,
        "takeaway": takeaway,
    }


def render_bucket_table(stage_payload: Dict[str, Any]) -> str:
    lines = [
        "| Variant | Selected view (ms) | Dispatch ops | align1/align2 tail | align8 kernels | gemv kernels | attention leakage | other GEMM kernels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_name in VARIANT_ORDER:
        variant = stage_payload["variants"][variant_name]
        buckets = variant["bucket_totals"]
        lines.append(
            "| `{variant}` | {selected} | {dispatch} | {align_tail} | {align8} | {gemv} | {attention} | {other} |".format(
                variant=variant_name,
                selected=format_ms(variant["selected_total_ms"]),
                dispatch=format_ms(buckets["dispatch_ops"]["self_cuda_time_ms"]),
                align_tail=format_ms(buckets["alignment_sensitive_kernels"]["self_cuda_time_ms"]),
                align8=format_ms(buckets["align8_tensorcore_kernels"]["self_cuda_time_ms"]),
                gemv=format_ms(buckets["gemv_kernels"]["self_cuda_time_ms"]),
                attention=format_ms(buckets["attention_leakage"]["self_cuda_time_ms"]),
                other=format_ms(buckets["other_gemm_kernels"]["self_cuda_time_ms"]),
            )
        )
    return "\n".join(lines)


def render_comparison_table(stage_payload: Dict[str, Any]) -> str:
    lines = [
        "| Comparison | Total delta (ms) | Kernel-only delta (ms) | Largest recovered bucket |",
        "| --- | ---: | ---: | --- |",
    ]
    for comparison_name, reference_name, candidate_name in COMPARISON_ORDER:
        comparison = stage_payload["comparisons"][comparison_name]
        lines.append(
            "| `{candidate}` vs `{reference}` | {total_delta} | {kernel_delta} | `{bucket}` |".format(
                candidate=candidate_name,
                reference=reference_name,
                total_delta=format_signed_ms(comparison["selected_total_delta_ms"]),
                kernel_delta=format_signed_ms(comparison["kernel_total_delta_ms"]),
                bucket=comparison["largest_recovered_bucket"],
            )
        )
    return "\n".join(lines)


def render_event_list(title: str, items: list[Dict[str, Any]]) -> list[str]:
    lines = [title]
    if not items:
        lines.append("- None")
        return lines
    for item in items:
        lines.append(
            "- `{name}` ({bucket}) {delta} ms: ref {ref} ms, cand {cand} ms".format(
                name=item["name"],
                bucket=item["bucket"],
                delta=format_signed_ms(item["delta_ms"]),
                ref=format_ms(item["reference_ms"]),
                cand=format_ms(item["candidate_ms"]),
            )
        )
    return lines


def build_analysis_markdown(summary_payload: Dict[str, Any]) -> str:
    prefill = summary_payload["stages"]["prefill"]
    decode = summary_payload["stages"]["decode"]
    prefill_palu = prefill["variants"]["palu"]
    decode_palu = decode["variants"]["palu"]
    prefill_recovery = prefill["comparisons"]["aligned_gac_vs_palu"]
    decode_recovery = decode["comparisons"]["aligned_gac_vs_palu"]

    lines = [
        "# GEMM Root-Cause Analysis For PaLU Inference",
        "",
        "This note is the kernel-level companion to `analysis.md` and `palu_inference_operator_profile_summary.json`.",
        "It re-reads the raw profiler events to explain why the 8-aligned GEMM cliff from simulation does not turn into a large end-to-end inference win.",
        "",
        "## Executive Summary",
        (
            "- The coarse GEMM-like view is dispatch-heavy rather than purely kernel-heavy: "
            f"`aten::mm` contributes {format_ms(prefill_palu['dispatch_total_ms'])} ms in prefill "
            f"({format_pct(prefill_palu['dispatch_share_of_selected_pct'])}) and "
            f"{format_ms(decode_palu['dispatch_total_ms'])} ms in decode "
            f"({format_pct(decode_palu['dispatch_share_of_selected_pct'])})."
        ),
        (
            "- Prefill alignment-sensitive kernels are a long tail: `align1/align2` kernels account for "
            f"{format_ms(prefill_palu['bucket_totals']['alignment_sensitive_kernels']['self_cuda_time_ms'])} ms in `palu`, "
            f"and `aligned_gac` recovers {format_ms(prefill_recovery['bucket_deltas']['alignment_sensitive_kernels']['delta_ms'])} ms "
            "from that tail, but other already-aligned tensorcore kernels give some of the gain back."
        ),
        (
            "- Decode has no large cliff to recover: the `gemv` tail is only "
            f"{format_ms(decode_palu['bucket_totals']['gemv_kernels']['self_cuda_time_ms'])} ms in `palu`, "
            f"and `aligned_gac` changes that bucket by {format_signed_ms(decode_recovery['bucket_deltas']['gemv_kernels']['delta_ms'])} ms."
        ),
        (
            "- Prefill also contains attention leakage: the flash-attention operator/kernel pair contributes "
            f"{format_ms(prefill_palu['bucket_totals']['attention_leakage']['self_cuda_time_ms'])} ms "
            "to the coarse GEMM-like view even though it is attention work, so the older family-level GEMM total overstates the truly alignment-sensitive part."
        ),
        "",
        "## Prefill",
        "",
        prefill["takeaway"],
        "",
        render_bucket_table(prefill),
        "",
        render_comparison_table(prefill),
        "",
        *render_event_list(
            "Top `palu` -> `aligned_gac` recovered prefill events:",
            prefill_recovery["top_recovered_events"],
        ),
        "",
        *render_event_list(
            "Top `palu` -> `aligned_gac` regressed prefill events:",
            prefill_recovery["top_regressed_events"],
        ),
        "",
        "## Decode",
        "",
        decode["takeaway"],
        "",
        render_bucket_table(decode),
        "",
        render_comparison_table(decode),
        "",
        *render_event_list(
            "Top `palu` -> `aligned_gac` recovered decode events:",
            decode_recovery["top_recovered_events"],
        ),
        "",
        *render_event_list(
            "Top `palu` -> `aligned_gac` regressed decode events:",
            decode_recovery["top_regressed_events"],
        ),
        "",
        "## Root Cause",
        "",
        "- The dominant runtime is still `aten::mm` dispatch plus already-large tensorcore kernels, not the small `align1/align2` tail that changes most under alignment.",
        "- The 8-aligned repair removes a few obviously suboptimal kernels, but their absolute contribution is too small to create a simulation-style cliff once the rest of inference is included.",
        "- Decode is especially constrained: the `gemv` path is already a small tail, so changing alignment mostly redistributes work among nearby `gemv` kernels instead of deleting a dominant hotspot.",
        "- The older coarse family summary is still directionally useful, but it is not enough for kernel root cause because it mixes dispatch ops, low-level kernels, and prefill attention leakage in the same broad GEMM view.",
        "",
        "## Artifact Map",
        "",
        "- `gemm_root_cause_summary.json`: machine-readable bucketed summary for dispatch, alignment-sensitive kernels, align8 kernels, GEMV tails, attention leakage, and other GEMM kernels.",
        "- `gemm_root_cause_analysis.md`: human-readable explanation of why the 8-aligned cliff is muted in full inference.",
        "- `analysis.md`: earlier coarse operator-family interpretation kept for stage-level context.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GEMM root cause for LLM inference operator profiles")
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--palu-run-dir", type=Path, required=True)
    parser.add_argument("--aligned-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = {
        "baseline": args.baseline_run_dir,
        "palu": args.palu_run_dir,
        "aligned_gac": args.aligned_run_dir,
    }
    raw_variants = {variant: load_json(path / "raw.json") for variant, path in run_dirs.items()}
    config_variants = {variant: load_json(path / "config.json") for variant, path in run_dirs.items()}

    stage_names = sorted(
        set.intersection(*(set(payload["stages"]) for payload in raw_variants.values()))
    )
    stages = {
        stage_name: build_stage_summary(stage_name, raw_variants, config_variants)
        for stage_name in stage_names
    }

    summary_payload = {
        "focus": "palu_inference_gemm_root_cause",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dirs": {variant: display_path(path) for variant, path in run_dirs.items()},
        "bucket_order": list(BUCKET_ORDER),
        "stages": stages,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "gemm_root_cause_summary.json", summary_payload)
    (output_dir / "gemm_root_cause_analysis.md").write_text(
        build_analysis_markdown(summary_payload)
    )


if __name__ == "__main__":
    main()
