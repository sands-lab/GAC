"""Kernel-level GEMM root-cause helpers for LLM operator profiles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from .operator_profile import normalize_profiler_event


BUCKET_ORDER = (
    "dispatch_ops",
    "alignment_sensitive_kernels",
    "align8_tensorcore_kernels",
    "gemv_kernels",
    "attention_leakage",
    "other_gemm_kernels",
)

BUCKET_LABELS = {
    "dispatch_ops": "Dispatch ops",
    "alignment_sensitive_kernels": "Alignment-sensitive kernels",
    "align8_tensorcore_kernels": "Align8 tensorcore kernels",
    "gemv_kernels": "GEMV kernels",
    "attention_leakage": "Attention leakage",
    "other_gemm_kernels": "Other GEMM kernels",
}


def _round_ms(value_us: float) -> float:
    return round(value_us / 1000.0, 6)


def _round_pct(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def is_attention_like(name: str) -> bool:
    lowered = name.lower()
    markers = (
        "attention",
        "flash_fwd_kernel",
        "_flash_attention",
        "_efficient_attention",
        "scaled_dot_product",
        "fmha",
    )
    return any(marker in lowered for marker in markers)


def is_dispatch_op(name: str) -> bool:
    lowered = name.lower()
    if not lowered.startswith("aten::"):
        return False
    markers = ("mm", "matmul", "linear", "addmm", "bmm")
    return any(marker in lowered for marker in markers)


def is_gemv_kernel(name: str) -> bool:
    lowered = name.lower()
    return "gemv" in lowered or "gemvx" in lowered


def is_alignment_sensitive_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("align1", "align2", "align4"))


def is_align8_tensorcore_kernel(name: str) -> bool:
    return "align8" in name.lower()


def is_gemm_like_kernel(name: str) -> bool:
    lowered = name.lower()
    markers = (
        "gemm",
        "cublaslt",
        "cutlass",
        "xmma",
        "wmma",
        "splitkreduce_kernel",
        "ampere_fp16",
        "ampere_sgemm",
        "sm80_xmma",
    )
    return any(marker in lowered for marker in markers)


def classify_gemm_root_cause_bucket(name: str) -> str | None:
    if is_attention_like(name):
        return "attention_leakage"
    if is_dispatch_op(name):
        return "dispatch_ops"
    if is_gemv_kernel(name):
        return "gemv_kernels"
    if is_alignment_sensitive_kernel(name):
        return "alignment_sensitive_kernels"
    if is_align8_tensorcore_kernel(name):
        return "align8_tensorcore_kernels"
    if is_gemm_like_kernel(name):
        return "other_gemm_kernels"
    return None


def summarize_root_cause_stage(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized = [normalize_profiler_event(event) for event in events]
    all_event_total_us = sum(event["self_cuda_time_us"] for event in normalized)

    bucket_totals_us = {bucket: 0.0 for bucket in BUCKET_ORDER}
    bucket_counts = {bucket: 0 for bucket in BUCKET_ORDER}
    bucket_events_us: Dict[str, Dict[str, float]] = {bucket: {} for bucket in BUCKET_ORDER}
    event_totals_us: Dict[str, float] = {}

    for event in normalized:
        bucket = classify_gemm_root_cause_bucket(event["name"])
        if bucket is None:
            continue
        event_us = float(event["self_cuda_time_us"])
        bucket_totals_us[bucket] += event_us
        bucket_counts[bucket] += int(event.get("count", 0) or 0)
        bucket_events_us[bucket][event["name"]] = bucket_events_us[bucket].get(event["name"], 0.0) + event_us
        event_totals_us[event["name"]] = event_totals_us.get(event["name"], 0.0) + event_us

    selected_total_us = sum(bucket_totals_us.values())
    dispatch_total_us = bucket_totals_us["dispatch_ops"]
    kernel_total_us = selected_total_us - dispatch_total_us

    def top_events_for_bucket(bucket: str) -> list[Dict[str, Any]]:
        return [
            {
                "name": name,
                "self_cuda_time_ms": _round_ms(total_us),
            }
            for name, total_us in sorted(
                bucket_events_us[bucket].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]

    bucket_totals: Dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        bucket_us = bucket_totals_us[bucket]
        bucket_totals[bucket] = {
            "label": BUCKET_LABELS[bucket],
            "self_cuda_time_ms": _round_ms(bucket_us),
            "share_of_selected_pct": _round_pct(bucket_us, selected_total_us),
            "share_of_stage_pct": _round_pct(bucket_us, all_event_total_us),
            "count": bucket_counts[bucket],
            "top_events": top_events_for_bucket(bucket),
        }

    top_selected_events = [
        {
            "name": name,
            "bucket": classify_gemm_root_cause_bucket(name),
            "self_cuda_time_ms": _round_ms(total_us),
            "share_of_selected_pct": _round_pct(total_us, selected_total_us),
        }
        for name, total_us in sorted(
            event_totals_us.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
    ]

    return {
        "all_event_total_ms": _round_ms(all_event_total_us),
        "selected_total_ms": _round_ms(selected_total_us),
        "dispatch_total_ms": _round_ms(dispatch_total_us),
        "kernel_total_ms": _round_ms(kernel_total_us),
        "dispatch_share_of_selected_pct": _round_pct(dispatch_total_us, selected_total_us),
        "bucket_totals": bucket_totals,
        "top_selected_events": top_selected_events,
        "event_totals_ms": {
            name: _round_ms(total_us) for name, total_us in event_totals_us.items()
        },
    }


def compare_root_cause_stages(
    reference_stage: Mapping[str, Any], candidate_stage: Mapping[str, Any]
) -> Dict[str, Any]:
    reference_total = float(reference_stage.get("selected_total_ms", 0.0))
    candidate_total = float(candidate_stage.get("selected_total_ms", 0.0))
    total_delta = round(reference_total - candidate_total, 6)
    improvement_pct = _round_pct(total_delta, reference_total)

    reference_kernel_total = float(reference_stage.get("kernel_total_ms", 0.0))
    candidate_kernel_total = float(candidate_stage.get("kernel_total_ms", 0.0))
    kernel_delta = round(reference_kernel_total - candidate_kernel_total, 6)

    bucket_deltas: Dict[str, Any] = {}
    largest_bucket = None
    largest_bucket_delta = None
    for bucket in BUCKET_ORDER:
        reference_ms = float(
            reference_stage.get("bucket_totals", {}).get(bucket, {}).get("self_cuda_time_ms", 0.0)
        )
        candidate_ms = float(
            candidate_stage.get("bucket_totals", {}).get(bucket, {}).get("self_cuda_time_ms", 0.0)
        )
        delta_ms = round(reference_ms - candidate_ms, 6)
        bucket_deltas[bucket] = {
            "label": BUCKET_LABELS[bucket],
            "reference_ms": reference_ms,
            "candidate_ms": candidate_ms,
            "delta_ms": delta_ms,
        }
        if largest_bucket_delta is None or delta_ms > largest_bucket_delta:
            largest_bucket = bucket
            largest_bucket_delta = delta_ms

    reference_events = reference_stage.get("event_totals_ms", {})
    candidate_events = candidate_stage.get("event_totals_ms", {})
    event_deltas = []
    for name in sorted(set(reference_events) | set(candidate_events)):
        reference_ms = float(reference_events.get(name, 0.0))
        candidate_ms = float(candidate_events.get(name, 0.0))
        delta_ms = round(reference_ms - candidate_ms, 6)
        event_deltas.append(
            {
                "name": name,
                "bucket": classify_gemm_root_cause_bucket(name),
                "reference_ms": reference_ms,
                "candidate_ms": candidate_ms,
                "delta_ms": delta_ms,
            }
        )

    top_recovered_events = [
        item
        for item in sorted(event_deltas, key=lambda item: item["delta_ms"], reverse=True)
        if item["delta_ms"] > 0
    ][:5]
    top_regressed_events = [
        item
        for item in sorted(event_deltas, key=lambda item: item["delta_ms"])
        if item["delta_ms"] < 0
    ][:5]

    return {
        "selected_total_delta_ms": total_delta,
        "selected_total_improvement_pct": improvement_pct,
        "kernel_total_delta_ms": kernel_delta,
        "largest_recovered_bucket": largest_bucket,
        "bucket_deltas": bucket_deltas,
        "top_recovered_events": top_recovered_events,
        "top_regressed_events": top_regressed_events,
    }
