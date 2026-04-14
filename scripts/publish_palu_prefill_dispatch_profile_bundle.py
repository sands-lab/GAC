#!/usr/bin/env python3
"""Publish a grouped-runtime dispatch profile sidecar for PaLU inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.gcompress_bench.operator_profile_root_cause import (  # noqa: E402
    BUCKET_ORDER,
    compare_root_cause_stages,
    summarize_root_cause_stage,
)


VARIANT_ORDER = ("palu", "grouped_bmm")
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


def copy_run_files(run_dir: Path, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    published_files: Dict[str, str] = {}
    for name in ("raw", "config", "summary", "env"):
        source = run_dir / f"{name}.json"
        if not source.exists():
            raise FileNotFoundError(f"Missing required run artifact: {source}")
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        published_files[name] = display_path(destination)
    return published_files


def _round_pct(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return round((float(numerator) / float(denominator)) * 100.0, 4)


def format_ms(value: float) -> str:
    return f"{value:.2f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def format_signed_ms(value: float) -> str:
    return f"{value:+.2f}"


def format_signed_pct_points(value: float) -> str:
    return f"{value:+.2f} pp"


def build_variant_summary(raw_payload: Dict[str, Any], config_payload: Dict[str, Any]) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}
    for stage_name, stage_payload in raw_payload.get("stages", {}).items():
        stage_summary = summarize_root_cause_stage(stage_payload.get("events", []))
        stage_summary["profile"] = stage_payload.get("profile", {})
        stages[stage_name] = stage_summary
    return {
        "config": {
            "variant": config_payload.get("variant"),
            "device": config_payload.get("device"),
            "dtype": config_payload.get("dtype"),
            "model_source": config_payload.get("model_source"),
            "stages": config_payload.get("stages"),
            "prefill_seq_len": config_payload.get("prefill_seq_len"),
            "decode_context_len": config_payload.get("decode_context_len"),
            "reconstruct_strategy": config_payload.get("reconstruct_strategy"),
        },
        "stages": stages,
    }


def build_comparisons(variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {"grouped_vs_palu": {}}
    reference = variants["palu"]["stages"]
    candidate = variants["grouped_bmm"]["stages"]
    for stage_name in sorted(set(reference) | set(candidate)):
        if stage_name not in reference or stage_name not in candidate:
            continue
        reference_stage = reference[stage_name]
        candidate_stage = candidate[stage_name]
        comparison = compare_root_cause_stages(reference_stage, candidate_stage)

        stage_total_delta = round(
            float(reference_stage.get("all_event_total_ms", 0.0))
            - float(candidate_stage.get("all_event_total_ms", 0.0)),
            6,
        )
        comparison["stage_total_delta_ms"] = stage_total_delta
        comparison["stage_total_improvement_pct"] = _round_pct(
            stage_total_delta,
            float(reference_stage.get("all_event_total_ms", 0.0)),
        )
        comparison["reference_dispatch_share_of_selected_pct"] = float(
            reference_stage.get("dispatch_share_of_selected_pct", 0.0)
        )
        comparison["candidate_dispatch_share_of_selected_pct"] = float(
            candidate_stage.get("dispatch_share_of_selected_pct", 0.0)
        )
        comparison["dispatch_share_delta_pct_points"] = round(
            comparison["reference_dispatch_share_of_selected_pct"]
            - comparison["candidate_dispatch_share_of_selected_pct"],
            4,
        )
        comparisons["grouped_vs_palu"][stage_name] = comparison
    return comparisons


def build_takeaways(summary_payload: Dict[str, Any]) -> Dict[str, str]:
    takeaways: Dict[str, str] = {}
    grouped_vs_palu = summary_payload["comparisons"]["grouped_vs_palu"]
    for stage_name, payload in grouped_vs_palu.items():
        takeaways[stage_name] = (
            "`grouped_bmm` changes {stage} total self CUDA time by {total_pct} ({total_ms} ms) and "
            "changes the selected GEMM-like view by {selected_pct} ({selected_ms} ms) versus `palu`; "
            "dispatch share of the selected view moves from {ref_dispatch} to {cand_dispatch} ({dispatch_delta}), "
            "with `{bucket}` as the largest recovered bucket."
        ).format(
            stage=stage_name,
            total_pct=format_pct(payload.get("stage_total_improvement_pct")),
            total_ms=format_signed_ms(float(payload.get("stage_total_delta_ms", 0.0))),
            selected_pct=format_pct(payload.get("selected_total_improvement_pct")),
            selected_ms=format_signed_ms(float(payload.get("selected_total_delta_ms", 0.0))),
            ref_dispatch=format_pct(payload.get("reference_dispatch_share_of_selected_pct")),
            cand_dispatch=format_pct(payload.get("candidate_dispatch_share_of_selected_pct")),
            dispatch_delta=format_signed_pct_points(float(payload.get("dispatch_share_delta_pct_points", 0.0))),
            bucket=payload.get("largest_recovered_bucket", "n/a"),
        )
    return takeaways


def render_stage_totals_table(summary_payload: Dict[str, Any], stage_name: str) -> str:
    lines = [
        "| Variant | Total self CUDA time (ms) | Selected view (ms) | Dispatch share | Dispatch ops | align1/2/4 tail | align8 kernels | gemv kernels | attention leakage | other GEMM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_name in VARIANT_ORDER:
        stage_payload = summary_payload["variants"][variant_name]["stages"][stage_name]
        bucket_totals = stage_payload["bucket_totals"]
        lines.append(
            "| `{variant}` | {total} | {selected} | {dispatch_share} | {dispatch_ops} | {align_tail} | {align8} | {gemv} | {attention} | {other} |".format(
                variant=variant_name,
                total=format_ms(float(stage_payload["all_event_total_ms"])),
                selected=format_ms(float(stage_payload["selected_total_ms"])),
                dispatch_share=format_pct(float(stage_payload["dispatch_share_of_selected_pct"])),
                dispatch_ops=format_ms(float(bucket_totals["dispatch_ops"]["self_cuda_time_ms"])),
                align_tail=format_ms(float(bucket_totals["alignment_sensitive_kernels"]["self_cuda_time_ms"])),
                align8=format_ms(float(bucket_totals["align8_tensorcore_kernels"]["self_cuda_time_ms"])),
                gemv=format_ms(float(bucket_totals["gemv_kernels"]["self_cuda_time_ms"])),
                attention=format_ms(float(bucket_totals["attention_leakage"]["self_cuda_time_ms"])),
                other=format_ms(float(bucket_totals["other_gemm_kernels"]["self_cuda_time_ms"])),
            )
        )
    return "\n".join(lines)


def render_comparison_table(summary_payload: Dict[str, Any], stage_name: str) -> str:
    comparison = summary_payload["comparisons"]["grouped_vs_palu"][stage_name]
    lines = [
        "| Comparison | Total delta (ms) | Total delta (%) | Selected-view delta (ms) | Selected-view delta (%) | Dispatch share delta | Largest recovered bucket |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        "| `grouped_bmm` vs `palu` | {total_ms} | {total_pct} | {selected_ms} | {selected_pct} | {dispatch_delta} | `{bucket}` |".format(
            total_ms=format_signed_ms(float(comparison["stage_total_delta_ms"])),
            total_pct=format_pct(comparison.get("stage_total_improvement_pct")),
            selected_ms=format_signed_ms(float(comparison["selected_total_delta_ms"])),
            selected_pct=format_pct(comparison.get("selected_total_improvement_pct")),
            dispatch_delta=format_signed_pct_points(float(comparison["dispatch_share_delta_pct_points"])),
            bucket=comparison.get("largest_recovered_bucket", "n/a"),
        ),
    ]
    return "\n".join(lines)


def render_event_list(title: str, items: Iterable[Dict[str, Any]]) -> list[str]:
    lines = [title]
    materialized = list(items)
    if not materialized:
        lines.append("- None")
        return lines
    for item in materialized:
        lines.append(
            "- `{name}` ({bucket}) {delta} ms: ref {ref} ms, cand {cand} ms".format(
                name=item["name"],
                bucket=item["bucket"],
                delta=format_signed_ms(float(item["delta_ms"])),
                ref=format_ms(float(item["reference_ms"])),
                cand=format_ms(float(item["candidate_ms"])),
            )
        )
    return lines


def build_analysis_markdown(summary_payload: Dict[str, Any]) -> str:
    lines = [
        "# PaLU Prefill Dispatch Runtime Profile",
        "",
        "This note compares unaligned `palu` against the same checkpoint with the issue40 grouped reconstruction runtime path enabled as `palu_grouped_bmm`.",
        "Positive delta means `grouped_bmm` reduced self CUDA time relative to `palu`; negative delta means regression.",
        "",
        "## Executive Summary",
    ]
    for stage_name in ("prefill", "decode"):
        takeaway = summary_payload.get("takeaways", {}).get(stage_name)
        if takeaway:
            lines.append(f"- {stage_name.title()}: {takeaway}")

    for stage_name in ("prefill", "decode"):
        comparison = summary_payload["comparisons"]["grouped_vs_palu"].get(stage_name)
        if comparison is None:
            continue
        lines.extend(
            [
                "",
                f"## {stage_name.title()}",
                "",
                render_stage_totals_table(summary_payload, stage_name),
                "",
                render_comparison_table(summary_payload, stage_name),
                "",
                summary_payload["takeaways"][stage_name],
                "",
                *render_event_list(
                    f"Top `palu` -> `grouped_bmm` recovered {stage_name} events:",
                    comparison.get("top_recovered_events", []),
                ),
                "",
                *render_event_list(
                    f"Top `palu` -> `grouped_bmm` regressed {stage_name} events:",
                    comparison.get("top_regressed_events", []),
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This sidecar compares runtime behavior for the same PaLU checkpoint; it does not change the checkpoint weights themselves.",
            "- The grouped runtime path only targets `HeadwiseLowRankModule` reconstruction on the attention-adjacent projection path.",
            "- The selected root-cause view is intentionally GEMM-focused, so dispatch-share changes should be read together with the total-stage delta.",
            "",
            "## Artifact Map",
            "",
            "- `prefill_dispatch_runtime_profile_summary.json`: machine-readable runtime comparison between `palu` and `palu_grouped_bmm`.",
            "- `prefill_dispatch_runtime_profile.md`: human-readable explanation of the runtime grouped-reconstruction follow-up.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary_payload(
    palu_run_dir: Path,
    grouped_run_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    run_dirs = {
        "palu": palu_run_dir,
        "grouped_bmm": grouped_run_dir,
    }
    raw_variants = {variant: load_json(path / "raw.json") for variant, path in run_dirs.items()}
    config_variants = {variant: load_json(path / "config.json") for variant, path in run_dirs.items()}
    variants = {
        variant: build_variant_summary(raw_variants[variant], config_variants[variant])
        for variant in VARIANT_ORDER
    }
    publish_dirs = {
        "palu": output_dir / "dispatch_runtime_palu",
        "grouped_bmm": output_dir / "dispatch_runtime_grouped_bmm",
    }
    manifest = {
        variant: copy_run_files(path, publish_dirs[variant])
        for variant, path in run_dirs.items()
    }
    summary_payload = {
        "focus": "palu_prefill_dispatch_runtime_profile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dirs": {variant: display_path(path) for variant, path in run_dirs.items()},
        "source_manifest": manifest,
        "bucket_order": list(BUCKET_ORDER),
        "variants": variants,
    }
    summary_payload["comparisons"] = build_comparisons(summary_payload["variants"])
    summary_payload["takeaways"] = build_takeaways(summary_payload)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palu-run-dir", type=Path, required=True)
    parser.add_argument("--grouped-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = build_summary_payload(
        palu_run_dir=args.palu_run_dir,
        grouped_run_dir=args.grouped_run_dir,
        output_dir=args.output_dir,
    )
    report = build_analysis_markdown(summary_payload)

    save_json(args.output_dir / "prefill_dispatch_runtime_profile_summary.json", summary_payload)
    (args.output_dir / "prefill_dispatch_runtime_profile.md").write_text(report)

    print(
        "Wrote PaLU prefill dispatch runtime summary to "
        f"{args.output_dir / 'prefill_dispatch_runtime_profile_summary.json'}"
    )
    print(
        "Wrote PaLU prefill dispatch runtime report to "
        f"{args.output_dir / 'prefill_dispatch_runtime_profile.md'}"
    )


if __name__ == "__main__":
    main()
