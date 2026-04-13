"""Inference optimization recommendation helpers for tracked profiler artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _format_ms(value: float) -> str:
    return f"{float(value):.2f} ms"


def _format_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _residual_gap_pct(candidate_total_ms: float, baseline_total_ms: float) -> float:
    if baseline_total_ms == 0:
        return 0.0
    return _round(((candidate_total_ms - baseline_total_ms) / baseline_total_ms) * 100.0, 4)


def _require_stage(payload: Mapping[str, Any], stage_name: str) -> Mapping[str, Any]:
    stages = payload.get("stages", {})
    if stage_name not in stages:
        raise KeyError(f"missing stage {stage_name!r}")
    return stages[stage_name]


def _collect_stage_metrics(
    operator_profile_summary: Mapping[str, Any],
    gemm_root_cause_summary: Mapping[str, Any],
    stage_name: str,
) -> Dict[str, Any]:
    profile_stage = operator_profile_summary["comparisons"]["aligned_gac_vs_palu"][stage_name]
    baseline_stage = operator_profile_summary["variants"]["baseline"]["stages"][stage_name]
    aligned_stage = operator_profile_summary["variants"]["aligned_gac"]["stages"][stage_name]
    root_stage = _require_stage(gemm_root_cause_summary, stage_name)
    root_variant = root_stage["variants"]["palu"]
    root_comparison = root_stage["comparisons"]["aligned_gac_vs_palu"]

    baseline_total_ms = float(baseline_stage["total_self_cuda_time_ms"])
    aligned_total_ms = float(aligned_stage["total_self_cuda_time_ms"])

    return {
        "aligned_vs_palu_delta_ms": _round(profile_stage["delta_ms"]),
        "aligned_vs_palu_improvement_pct": _round(profile_stage["total_self_cuda_time_improvement_pct"], 4),
        "largest_recovered_family": profile_stage["largest_recovered_family"],
        "aligned_vs_baseline_residual_gap_ms": _round(aligned_total_ms - baseline_total_ms),
        "aligned_vs_baseline_residual_gap_pct": _residual_gap_pct(aligned_total_ms, baseline_total_ms),
        "dispatch_share_of_selected_pct": _round(root_variant["dispatch_share_of_selected_pct"], 4),
        "alignment_sensitive_tail_ms": _round(
            root_variant["bucket_totals"]["alignment_sensitive_kernels"]["self_cuda_time_ms"]
        ),
        "attention_leakage_ms": _round(
            root_variant["bucket_totals"]["attention_leakage"]["self_cuda_time_ms"]
        ),
        "gemv_tail_ms": _round(root_variant["bucket_totals"]["gemv_kernels"]["self_cuda_time_ms"]),
        "alignment_sensitive_recovery_ms": _round(
            root_comparison["bucket_deltas"]["alignment_sensitive_kernels"]["delta_ms"]
        ),
        "other_gemm_regression_ms": _round(
            root_comparison["bucket_deltas"]["other_gemm_kernels"]["delta_ms"]
        ),
        "gemv_delta_ms": _round(root_comparison["bucket_deltas"]["gemv_kernels"]["delta_ms"]),
    }


def build_inference_speed_optimization_summary(
    operator_profile_summary: Mapping[str, Any],
    gemm_root_cause_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prioritize next optimization steps from tracked profile and root-cause artifacts."""

    prefill = _collect_stage_metrics(
        operator_profile_summary=operator_profile_summary,
        gemm_root_cause_summary=gemm_root_cause_summary,
        stage_name="prefill",
    )
    decode = _collect_stage_metrics(
        operator_profile_summary=operator_profile_summary,
        gemm_root_cause_summary=gemm_root_cause_summary,
        stage_name="decode",
    )

    opportunities = [
        {
            "id": "prefill_dispatch_reduction",
            "priority": "high",
            "stage": "prefill",
            "title": "Prefill dispatch reduction on the projection path",
            "target_module": "HeadwiseLowRankModule (`k_proj` / `v_proj` low-rank path)",
            "why_now": (
                "The current 8-aligned repair only recovers a small prefill win, while the selected GEMM view remains "
                "dominated by `aten::mm` dispatch rather than by the removable align1/align2 tail."
            ),
            "evidence": {
                "largest_recovered_family": prefill["largest_recovered_family"],
                "aligned_vs_palu_delta_ms": prefill["aligned_vs_palu_delta_ms"],
                "aligned_vs_palu_improvement_pct": prefill["aligned_vs_palu_improvement_pct"],
                "aligned_vs_baseline_residual_gap_pct": prefill["aligned_vs_baseline_residual_gap_pct"],
                "dispatch_share_of_selected_pct": prefill["dispatch_share_of_selected_pct"],
                "alignment_sensitive_tail_ms": prefill["alignment_sensitive_tail_ms"],
                "dominant_dispatch_op": "aten::mm",
            },
            "candidate_actions": [
                "Prototype grouped or fused execution for the `HeadwiseLowRankModule` projection path so prefill pays fewer `aten::mm` launches.",
                "Audit whether `k_proj` / `v_proj` `VT` + `U[*]` can be materialized as fewer larger GEMMs during prefill instead of many small dispatches.",
                "Treat success as a dispatch-share reduction, not just an align1/align2-kernel elimination."
            ],
            "success_signal": (
                "Prefill total self CUDA time improves materially beyond the current "
                f"{_format_pct(prefill['aligned_vs_palu_improvement_pct'])} versus unaligned PaLU, "
                "and the selected-view dispatch share falls from the current "
                f"{_format_pct(prefill['dispatch_share_of_selected_pct'])}."
            ),
        },
        {
            "id": "prefill_profile_guided_rank_retuning",
            "priority": "medium",
            "stage": "prefill",
            "title": "Profile-guided rank retuning beyond simple 8-alignment",
            "target_module": "Alignment-budget search for PaLU `k_proj` / `v_proj` ranks",
            "why_now": (
                "The current repair removes the measurable align1/align2 tail, but a large fraction of that gain is given back "
                "by regressions in `other_gemm_kernels`, which means nearest 8-alignment is too coarse as an optimization objective."
            ),
            "evidence": {
                "alignment_sensitive_recovery_ms": prefill["alignment_sensitive_recovery_ms"],
                "other_gemm_regression_ms": prefill["other_gemm_regression_ms"],
                "attention_leakage_ms": prefill["attention_leakage_ms"],
                "dominant_regression_bucket": "other_gemm_kernels",
                "current_summary_caveat": "Do not score candidates from the coarse GEMM family alone because that view still contains attention leakage.",
            },
            "candidate_actions": [
                "Extend the PaLU path from simple round-to-8 repair to contract-aware candidate search over profiled `recommended_values` and cliff-aware rank choices.",
                "Score candidates against both eliminated `alignment_sensitive_kernels` and regressions in `other_gemm_kernels` instead of minimizing alignment penalty alone.",
                "Reuse the repo-native `alignment_budget` machinery for a small DP or global re-allocation pass on the attention-adjacent projection ranks."
            ],
            "success_signal": (
                "A follow-up retuning pass keeps the recovered align1/align2 benefit while reducing the current "
                f"{_format_ms(abs(prefill['other_gemm_regression_ms']))} give-back in `other_gemm_kernels`."
            ),
        },
    ]

    deprioritized_paths = [
        {
            "id": "decode_gemv_micro_tuning",
            "stage": "decode",
            "title": "Decode GEMV micro-tuning",
            "why_not_now": (
                "Decode is not where the current PaLU gap is being created. The `gemv` tail is small and alignment barely changes it."
            ),
            "evidence": {
                "aligned_vs_palu_delta_ms": decode["aligned_vs_palu_delta_ms"],
                "aligned_vs_palu_improvement_pct": decode["aligned_vs_palu_improvement_pct"],
                "aligned_vs_baseline_residual_gap_pct": decode["aligned_vs_baseline_residual_gap_pct"],
                "gemv_tail_ms": decode["gemv_tail_ms"],
                "gemv_delta_ms": decode["gemv_delta_ms"],
            },
            "revisit_when": (
                "Only revisit this path if future traces show a materially larger decode `gemv` tail than the current "
                f"{_format_ms(decode['gemv_tail_ms'])}, or if decode becomes the dominant residual gap after prefill work."
            ),
        }
    ]

    guardrails = [
        "Treat this artifact as offline prioritization, not as proof of a new measured speedup.",
        "Use the issue-38 bucketed root-cause split when ranking GEMM optimizations; the coarse GEMM family still mixes in attention leakage.",
        "For PaLU, optimize the attention-adjacent projection path first; do not start from decode-only GEMV micro-tuning.",
    ]

    return {
        "focus": "palu_inference_speed_optimization",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_focus": {
            "operator_profile_summary": operator_profile_summary.get("focus"),
            "gemm_root_cause_summary": gemm_root_cause_summary.get("focus"),
        },
        "high_level_summary": {
            "primary_stage": "prefill",
            "current_aligned_gain_vs_palu": {
                "prefill_ms": prefill["aligned_vs_palu_delta_ms"],
                "prefill_pct": prefill["aligned_vs_palu_improvement_pct"],
                "decode_ms": decode["aligned_vs_palu_delta_ms"],
                "decode_pct": decode["aligned_vs_palu_improvement_pct"],
            },
            "residual_gap_vs_baseline_pct": {
                "prefill": prefill["aligned_vs_baseline_residual_gap_pct"],
                "decode": decode["aligned_vs_baseline_residual_gap_pct"],
            },
            "current_bottleneck_summary": (
                "Prefill still dominates the next optimization pass: the selected GEMM view is "
                f"{_format_pct(prefill['dispatch_share_of_selected_pct'])} `aten::mm` dispatch, while the recoverable "
                f"align1/align2 tail is only {_format_ms(prefill['alignment_sensitive_tail_ms'])}. "
                "Decode `gemv` micro-tuning stays low leverage because the current tail is only "
                f"{_format_ms(decode['gemv_tail_ms'])}."
            ),
        },
        "stage_metrics": {
            "prefill": prefill,
            "decode": decode,
        },
        "opportunities": opportunities,
        "deprioritized_paths": deprioritized_paths,
        "guardrails": guardrails,
    }


def _render_action_lines(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_inference_speed_optimization_markdown(summary_payload: Mapping[str, Any]) -> str:
    """Render a human-readable markdown report from the optimization summary payload."""

    high_level = summary_payload["high_level_summary"]
    opportunities = list(summary_payload.get("opportunities", []))
    deprioritized = list(summary_payload.get("deprioritized_paths", []))
    guardrails = list(summary_payload.get("guardrails", []))

    def find_opportunity(opportunity_id: str) -> Mapping[str, Any]:
        for item in opportunities:
            if item.get("id") == opportunity_id:
                return item
        raise KeyError(f"missing opportunity {opportunity_id!r}")

    dispatch = find_opportunity("prefill_dispatch_reduction")
    rank_retuning = find_opportunity("prefill_profile_guided_rank_retuning")
    decode_deprioritized = deprioritized[0]

    lines = [
        "# Inference Speed Optimization Opportunities For PaLU",
        "",
        "This note turns the checked-in operator-profile summary and GEMM root-cause summary into a ranked next-step list.",
        "It is an offline prioritization artifact, not a claim that a new optimization has already been measured on GPU.",
        "",
        "## Summary",
        (
            "- The current aligned repair recovers "
            f"{_format_ms(high_level['current_aligned_gain_vs_palu']['prefill_ms'])} / "
            f"{_format_pct(high_level['current_aligned_gain_vs_palu']['prefill_pct'])} in `prefill` and "
            f"{_format_ms(high_level['current_aligned_gain_vs_palu']['decode_ms'])} / "
            f"{_format_pct(high_level['current_aligned_gain_vs_palu']['decode_pct'])} in `decode` versus unaligned PaLU."
        ),
        (
            "- `aligned_gac` still trails `baseline` by "
            f"{_format_pct(high_level['residual_gap_vs_baseline_pct']['prefill'])} in `prefill` and "
            f"{_format_pct(high_level['residual_gap_vs_baseline_pct']['decode'])} in `decode`."
        ),
        f"- {high_level['current_bottleneck_summary']}",
        "",
        "## High Priority",
        "",
        "### Prefill dispatch reduction on the projection path",
        f"- Target: {dispatch['target_module']}",
        f"- Why now: {dispatch['why_now']}",
        (
            "- Evidence: the selected prefill GEMM view is still "
            f"{_format_pct(dispatch['evidence']['dispatch_share_of_selected_pct'])} `aten::mm` dispatch, "
            f"while the removable align1/align2 tail is only {_format_ms(dispatch['evidence']['alignment_sensitive_tail_ms'])}. "
            f"The measured aligned gain is {_format_ms(dispatch['evidence']['aligned_vs_palu_delta_ms'])} "
            f"({_format_pct(dispatch['evidence']['aligned_vs_palu_improvement_pct'])}), "
            f"and `aligned_gac` still trails baseline by {_format_pct(dispatch['evidence']['aligned_vs_baseline_residual_gap_pct'])}."
        ),
        *(_render_action_lines(dispatch["candidate_actions"])),
        f"- Success signal: {dispatch['success_signal']}",
        "",
        "## Medium Priority",
        "",
        "### Profile-guided rank retuning beyond simple 8-alignment",
        f"- Target: {rank_retuning['target_module']}",
        f"- Why now: {rank_retuning['why_now']}",
        (
            "- Evidence: alignment-sensitive kernels recover "
            f"{_format_ms(rank_retuning['evidence']['alignment_sensitive_recovery_ms'])}, but "
            f"`other_gemm_kernels` gives back {_format_ms(abs(rank_retuning['evidence']['other_gemm_regression_ms']))}. "
            f"The coarse prefill GEMM view also still contains {_format_ms(rank_retuning['evidence']['attention_leakage_ms'])} "
            "of attention leakage, so candidate scoring must stay root-cause-aware."
        ),
        *(_render_action_lines(rank_retuning["candidate_actions"])),
        f"- Success signal: {rank_retuning['success_signal']}",
        "",
        "## Deprioritized",
        "",
        "### Decode GEMV micro-tuning",
        f"- Why not now: {decode_deprioritized['why_not_now']}",
        (
            "- Evidence: the decode `gemv` tail is only "
            f"{_format_ms(decode_deprioritized['evidence']['gemv_tail_ms'])}, and alignment changes that bucket by "
            f"{_format_ms(decode_deprioritized['evidence']['gemv_delta_ms'])}. "
            f"The total decode gain stays at {_format_ms(decode_deprioritized['evidence']['aligned_vs_palu_delta_ms'])} "
            f"({_format_pct(decode_deprioritized['evidence']['aligned_vs_palu_improvement_pct'])})."
        ),
        f"- Revisit trigger: {decode_deprioritized['revisit_when']}",
        "",
        "## Guardrails",
        *(_render_action_lines(guardrails)),
    ]
    return "\n".join(lines) + "\n"
