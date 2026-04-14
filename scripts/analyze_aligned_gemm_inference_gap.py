#!/usr/bin/env python3
"""Explain why aligned GEMM wins stay muted in real PaLU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def format_ms(value: float) -> str:
    return f"{value:.2f} ms"


def format_signed_ms(value: float) -> str:
    return f"{value:+.2f} ms"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def format_signed_pct_points(value: float) -> str:
    return f"{value:+.2f} pp"


def _round_pct(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _stage_total_delta(reference: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    return round(
        float(reference.get("all_event_total_ms", 0.0))
        - float(candidate.get("all_event_total_ms", 0.0)),
        6,
    )


def _new_regression_events(items: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [
        item
        for item in items
        if float(item.get("reference_ms", 0.0)) <= 0.01 and float(item.get("candidate_ms", 0.0)) > 0.0
    ]


def build_aligned_section(stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    variants = stage_payload["variants"]
    comparison = stage_payload["comparisons"]["aligned_gac_vs_palu"]
    baseline = variants["baseline"]
    palu = variants["palu"]
    aligned = variants["aligned_gac"]

    remaining_gap_vs_baseline_ms = round(
        float(aligned.get("all_event_total_ms", 0.0)) - float(baseline.get("all_event_total_ms", 0.0)),
        6,
    )
    remaining_gap_vs_baseline_pct = _round_pct(
        remaining_gap_vs_baseline_ms,
        float(baseline.get("all_event_total_ms", 0.0)),
    )
    stage_total_recovered_ms = _stage_total_delta(palu, aligned)
    other_gemm_delta = float(
        comparison["bucket_deltas"]["other_gemm_kernels"]["delta_ms"]
    )

    return {
        "stage_total_recovered_ms": stage_total_recovered_ms,
        "selected_view_recovered_ms": float(comparison["selected_total_delta_ms"]),
        "remaining_gap_vs_baseline_ms": remaining_gap_vs_baseline_ms,
        "remaining_gap_vs_baseline_pct": remaining_gap_vs_baseline_pct,
        "dispatch_share_of_selected_pct": float(palu["dispatch_share_of_selected_pct"]),
        "dispatch_ops_recovered_ms": float(comparison["bucket_deltas"]["dispatch_ops"]["delta_ms"]),
        "alignment_tail_recovered_ms": float(
            comparison["bucket_deltas"]["alignment_sensitive_kernels"]["delta_ms"]
        ),
        "other_gemm_give_back_ms": max(-other_gemm_delta, 0.0),
        "attention_leakage_ms": float(
            palu["bucket_totals"]["attention_leakage"]["self_cuda_time_ms"]
        ),
        "largest_recovered_bucket": comparison["largest_recovered_bucket"],
        "top_recovered_events": comparison["top_recovered_events"],
        "top_regressed_events": comparison["top_regressed_events"],
    }


def build_grouped_section(stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    comparison = stage_payload
    stage_total_delta_ms = float(comparison["stage_total_delta_ms"])
    return {
        "stage_total_delta_ms": stage_total_delta_ms,
        "stage_total_regression_ms": max(-stage_total_delta_ms, 0.0),
        "selected_view_recovered_ms": float(comparison["selected_total_delta_ms"]),
        "dispatch_bucket_recovered_ms": float(
            comparison["bucket_deltas"]["dispatch_ops"]["delta_ms"]
        ),
        "dispatch_share_delta_pct_points": float(comparison["dispatch_share_delta_pct_points"]),
        "largest_recovered_bucket": comparison["largest_recovered_bucket"],
        "top_recovered_events": comparison["top_recovered_events"],
        "top_regressed_events": comparison["top_regressed_events"],
        "new_regression_events": _new_regression_events(comparison["top_regressed_events"]),
    }


def _event_label(event: Dict[str, Any]) -> str:
    return f"`{event['name']}`"


def build_stage_takeaway(
    stage_name: str,
    aligned_section: Dict[str, Any],
    grouped_section: Dict[str, Any],
) -> str:
    new_regressions = grouped_section.get("new_regression_events", [])
    highlighted = ", ".join(_event_label(event) for event in new_regressions[:2]) or "new grouped-runtime kernels"
    dispatch_share = format_pct(float(aligned_section["dispatch_share_of_selected_pct"]))
    if stage_name == "prefill":
        return (
            "`aligned_gac` only closes part of the prefill gap because it removes "
            f"{format_ms(float(aligned_section['alignment_tail_recovered_ms']))} of `align1/align2` tail, "
            f"but still gives back {format_ms(float(aligned_section['other_gemm_give_back_ms']))} in "
            "`other_gemm_kernels` while the selected view remains dispatch-heavy at "
            f"{dispatch_share}. The `grouped_bmm` runtime path recovers "
            f"{format_ms(float(grouped_section['selected_view_recovered_ms']))} inside the selected view, "
            f"but adds new costs such as {highlighted}, so total-stage time still moves by "
            f"{format_signed_ms(float(grouped_section['stage_total_delta_ms']))}."
        )
    return (
        "`decode` has no large aligned cliff left to cash in: "
        f"`grouped_bmm` still recovers {format_ms(float(grouped_section['selected_view_recovered_ms']))} "
        "inside the selected view, but the total-stage delta stays "
        f"{format_signed_ms(float(grouped_section['stage_total_delta_ms']))} once new costs such as "
        f"{highlighted} are included."
    )


def build_summary(root_cause_summary: Dict[str, Any], runtime_summary: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "focus": "aligned_gemm_inference_gap",
        "bucket_order": root_cause_summary.get("bucket_order", []),
        "root_cause_focus": root_cause_summary.get("focus"),
        "runtime_focus": runtime_summary.get("focus"),
        "root_cause_sources": {
            "source_run_dirs": root_cause_summary.get("source_run_dirs", {}),
        },
        "runtime_sources": {
            "source_run_dirs": runtime_summary.get("source_run_dirs", {}),
            "source_manifest": runtime_summary.get("source_manifest", {}),
        },
        "stages": {},
    }

    runtime_comparisons = runtime_summary.get("comparisons", {}).get("grouped_vs_palu", {})
    for stage_name in ("prefill", "decode"):
        root_stage = root_cause_summary["stages"][stage_name]
        runtime_stage = runtime_comparisons[stage_name]
        aligned_section = build_aligned_section(root_stage)
        grouped_section = build_grouped_section(runtime_stage)
        summary["stages"][stage_name] = {
            "aligned_gac_vs_palu": aligned_section,
            "grouped_bmm_vs_palu": grouped_section,
            "takeaway": build_stage_takeaway(stage_name, aligned_section, grouped_section),
        }

    return summary


def render_event_list(title: str, items: Iterable[Dict[str, Any]]) -> list[str]:
    lines = [title]
    materialized = list(items)
    if not materialized:
        lines.append("- None")
        return lines
    for item in materialized:
        lines.append(
            "- `{name}` {delta}: ref {ref}, cand {cand}".format(
                name=item["name"],
                delta=format_signed_ms(float(item["delta_ms"])),
                ref=format_ms(float(item["reference_ms"])),
                cand=format_ms(float(item["candidate_ms"])),
            )
        )
    return lines


def render_stage_table(stage_payload: Dict[str, Any]) -> str:
    aligned = stage_payload["aligned_gac_vs_palu"]
    grouped = stage_payload["grouped_bmm_vs_palu"]
    lines = [
        "| Comparison | Stage-total delta | Selected-view delta | Dispatch signal | Main give-back |",
        "| --- | ---: | ---: | ---: | --- |",
        "| `aligned_gac` vs `palu` | {stage_delta} | {selected_delta} | selected view still {dispatch_share} dispatch | `{bucket}` + `other_gemm_kernels` give-back {give_back} |".format(
            stage_delta=format_signed_ms(float(aligned["stage_total_recovered_ms"])),
            selected_delta=format_signed_ms(float(aligned["selected_view_recovered_ms"])),
            dispatch_share=format_pct(float(aligned["dispatch_share_of_selected_pct"])),
            bucket=aligned["largest_recovered_bucket"],
            give_back=format_ms(float(aligned["other_gemm_give_back_ms"])),
        ),
        "| `grouped_bmm` vs `palu` | {stage_delta} | {selected_delta} | dispatch share delta {dispatch_delta} | `{bucket}` plus new costs |".format(
            stage_delta=format_signed_ms(float(grouped["stage_total_delta_ms"])),
            selected_delta=format_signed_ms(float(grouped["selected_view_recovered_ms"])),
            dispatch_delta=format_signed_pct_points(float(grouped["dispatch_share_delta_pct_points"])),
            bucket=grouped["largest_recovered_bucket"],
        ),
    ]
    return "\n".join(lines)


def build_markdown(summary: Dict[str, Any]) -> str:
    prefill = summary["stages"]["prefill"]
    decode = summary["stages"]["decode"]
    lines = [
        "# Aligned GEMM Inference Gap Analysis",
        "",
        "This note bridges `gemm_root_cause_summary.json` and `prefill_dispatch_runtime_profile_summary.json`.",
        "It explains why the aligned GEMM cliff observed in simulation still turns into a muted end-to-end inference win on the checked-in PaLU path.",
        "It is a sidecar analysis of checked-in summaries, not a measured GPU speedup.",
        "",
        "## Executive Summary",
        f"- Prefill: {prefill['takeaway']}",
        f"- Decode: {decode['takeaway']}",
        "",
        "## Prefill",
        "",
        render_stage_table(prefill),
        "",
        "- `aligned_gac` root-cause signals:",
        "  - alignment tail recovered: {alignment_tail}".format(
            alignment_tail=format_ms(float(prefill["aligned_gac_vs_palu"]["alignment_tail_recovered_ms"]))
        ),
        "  - other GEMM give-back: {give_back}".format(
            give_back=format_ms(float(prefill["aligned_gac_vs_palu"]["other_gemm_give_back_ms"]))
        ),
        "  - attention leakage inside the coarse selected view: {attention}".format(
            attention=format_ms(float(prefill["aligned_gac_vs_palu"]["attention_leakage_ms"]))
        ),
        "",
        *render_event_list(
            "Top new grouped-runtime regressions in prefill:",
            prefill["grouped_bmm_vs_palu"]["new_regression_events"],
        ),
        "",
        *render_event_list(
            "Top grouped-runtime recovered prefill events:",
            prefill["grouped_bmm_vs_palu"]["top_recovered_events"],
        ),
        "",
        "## Decode",
        "",
        render_stage_table(decode),
        "",
        *render_event_list(
            "Top new grouped-runtime regressions in decode:",
            decode["grouped_bmm_vs_palu"]["new_regression_events"],
        ),
        "",
        *render_event_list(
            "Top grouped-runtime recovered decode events:",
            decode["grouped_bmm_vs_palu"]["top_recovered_events"],
        ),
        "",
        "## Guardrails",
        "",
        "- This analysis reuses checked-in summary artifacts; it does not claim a new measured GPU speedup.",
        "- `aligned_gac` and `grouped_bmm` should be read as separate follow-up probes: one changes checkpoint ranks, the other changes the runtime path.",
        "- The dominant story is still `dispatch` plus give-back in `other_gemm_kernels`; eliminating the old `align1/align2` tail alone is not enough.",
        "",
        "## Artifact Map",
        "",
        "- `aligned_gemm_inference_gap_summary.json`: machine-readable bridge between the aligned root-cause summary and the grouped-runtime sidecar summary.",
        "- `aligned_gemm_inference_gap.md`: human-readable explanation of why selected-view GEMM gains still fail to turn into a large end-to-end win.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine root-cause and grouped-runtime summaries into an aligned-GEMM gap analysis."
    )
    parser.add_argument(
        "--root-cause-summary",
        type=Path,
        required=True,
        help="Path to gemm_root_cause_summary.json.",
    )
    parser.add_argument(
        "--runtime-profile-summary",
        type=Path,
        required=True,
        help="Path to prefill_dispatch_runtime_profile_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for aligned_gemm_inference_gap_summary.json and .md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_cause_summary = load_json(args.root_cause_summary)
    runtime_summary = load_json(args.runtime_profile_summary)
    summary = build_summary(root_cause_summary, runtime_summary)
    report = build_markdown(summary)

    summary_path = args.output_dir / "aligned_gemm_inference_gap_summary.json"
    report_path = args.output_dir / "aligned_gemm_inference_gap.md"
    save_json(summary_path, summary)
    save_text(report_path, report)

    print(f"Wrote aligned GEMM inference gap summary to {summary_path}")
    print(f"Wrote aligned GEMM inference gap report to {report_path}")


if __name__ == "__main__":
    main()
