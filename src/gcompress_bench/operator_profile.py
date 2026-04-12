"""Utilities for summarizing LLM operator profiler events."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def classify_operator_family(name: str) -> str:
    """Map a profiler event name onto a small operator-family vocabulary."""
    lowered = name.lower()

    sdpa_markers = (
        "scaled_dot_product_attention",
        "flash_attn",
        "flashattention",
        "fmha",
        "sdpa",
    )
    if any(marker in lowered for marker in sdpa_markers):
        return "sdpa"

    gemm_markers = (
        "aten::mm",
        "aten::bmm",
        "aten::addmm",
        "aten::matmul",
        "aten::_scaled_mm",
        "aten::linear",
        "gemm",
        "gemv",
        "cublas",
        "cutlass",
    )
    if any(marker in lowered for marker in gemm_markers):
        return "gemm"

    norm_markers = (
        "layer_norm",
        "native_layer_norm",
        "rms_norm",
        "group_norm",
    )
    if any(marker in lowered for marker in norm_markers):
        return "norm"

    data_movement_markers = (
        "copy",
        "clone",
        "contiguous",
        "index",
        "gather",
        "scatter",
        "slice",
        "transpose",
        "permute",
        "view",
        "reshape",
        "cat",
    )
    if any(marker in lowered for marker in data_movement_markers):
        return "data_movement"

    elementwise_markers = (
        "softmax",
        "gelu",
        "silu",
        "relu",
        "mul",
        "add",
        "sub",
        "div",
        "pow",
        "where",
        "clamp",
    )
    if any(marker in lowered for marker in elementwise_markers):
        return "elementwise"

    return "other"


def _event_name(event: Mapping[str, Any]) -> str:
    value = event.get("name") or event.get("key") or ""
    return str(value)


def _event_float(event: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in event and event[key] is not None:
            return float(event[key])
    return 0.0


def _event_int(event: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in event and event[key] is not None:
            return int(event[key])
    return 0


def _round_ms(value_us: float) -> float:
    return round(value_us / 1000.0, 6)


def normalize_profiler_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize profiler event dictionaries into a stable repo-native shape."""
    name = _event_name(event)
    return {
        "name": name,
        "operator_family": event.get("operator_family") or classify_operator_family(name),
        "count": _event_int(event, "count"),
        "input_shapes": event.get("input_shapes") or [],
        "self_cuda_time_us": _event_float(
            event,
            "self_cuda_time_us",
            "self_device_time_us",
            "self_cuda_time_total",
            "self_device_time_total",
        ),
        "cuda_time_us": _event_float(
            event,
            "cuda_time_us",
            "device_time_us",
            "cuda_time_total",
            "device_time_total",
        ),
        "self_cpu_time_us": _event_float(event, "self_cpu_time_us", "self_cpu_time_total"),
        "cpu_time_us": _event_float(event, "cpu_time_us", "cpu_time_total"),
    }


def summarize_profile_events(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate normalized profiler events into operator-family summaries."""
    normalized = [normalize_profiler_event(event) for event in events]
    normalized.sort(key=lambda item: item["self_cuda_time_us"], reverse=True)

    total_self_cuda_us = sum(item["self_cuda_time_us"] for item in normalized)
    total_self_cpu_us = sum(item["self_cpu_time_us"] for item in normalized)

    family_buckets: Dict[str, Dict[str, Any]] = {}
    top_ops: Dict[str, Dict[str, Any]] = {}

    for event in normalized:
        family = event["operator_family"]
        family_bucket = family_buckets.setdefault(
            family,
            {
                "self_cuda_time_us": 0.0,
                "self_cpu_time_us": 0.0,
                "count": 0,
                "_ops": {},
            },
        )
        family_bucket["self_cuda_time_us"] += event["self_cuda_time_us"]
        family_bucket["self_cpu_time_us"] += event["self_cpu_time_us"]
        family_bucket["count"] += event["count"]

        op_bucket = family_bucket["_ops"].setdefault(
            event["name"],
            {
                "name": event["name"],
                "count": 0,
                "self_cuda_time_us": 0.0,
                "self_cpu_time_us": 0.0,
            },
        )
        op_bucket["count"] += event["count"]
        op_bucket["self_cuda_time_us"] += event["self_cuda_time_us"]
        op_bucket["self_cpu_time_us"] += event["self_cpu_time_us"]

        global_bucket = top_ops.setdefault(
            event["name"],
            {
                "name": event["name"],
                "operator_family": family,
                "count": 0,
                "self_cuda_time_us": 0.0,
            },
        )
        global_bucket["count"] += event["count"]
        global_bucket["self_cuda_time_us"] += event["self_cuda_time_us"]

    sorted_families = sorted(
        family_buckets.items(),
        key=lambda item: item[1]["self_cuda_time_us"],
        reverse=True,
    )

    operator_families: Dict[str, Dict[str, Any]] = {}
    for family_name, bucket in sorted_families:
        family_cuda_us = bucket["self_cuda_time_us"]
        family_ops = sorted(
            bucket["_ops"].values(),
            key=lambda item: item["self_cuda_time_us"],
            reverse=True,
        )
        operator_families[family_name] = {
            "self_cuda_time_ms": _round_ms(family_cuda_us),
            "self_cpu_time_ms": _round_ms(bucket["self_cpu_time_us"]),
            "share_pct": round(
                (family_cuda_us / total_self_cuda_us) * 100.0, 4
            )
            if total_self_cuda_us
            else 0.0,
            "count": bucket["count"],
            "top_ops": [
                {
                    "name": item["name"],
                    "count": item["count"],
                    "self_cuda_time_ms": _round_ms(item["self_cuda_time_us"]),
                    "self_cpu_time_ms": _round_ms(item["self_cpu_time_us"]),
                }
                for item in family_ops[:3]
            ],
        }

    sorted_top_ops = sorted(
        top_ops.values(),
        key=lambda item: item["self_cuda_time_us"],
        reverse=True,
    )

    return {
        "event_count": len(normalized),
        "total_self_cuda_time_ms": _round_ms(total_self_cuda_us),
        "total_self_cpu_time_ms": _round_ms(total_self_cpu_us),
        "operator_families": operator_families,
        "top_cuda_ops": [
            {
                "name": item["name"],
                "operator_family": item["operator_family"],
                "count": item["count"],
                "self_cuda_time_ms": _round_ms(item["self_cuda_time_us"]),
                "share_pct": round(
                    (item["self_cuda_time_us"] / total_self_cuda_us) * 100.0, 4
                )
                if total_self_cuda_us
                else 0.0,
            }
            for item in sorted_top_ops[:5]
        ],
    }


def summarize_profile_run(raw_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize every stage found in a raw profiler payload."""
    stages = raw_payload.get("stages", {})
    summary_stages: Dict[str, Any] = {}
    for stage_name, stage_payload in stages.items():
        stage_summary = summarize_profile_events(stage_payload.get("events", []))
        stage_summary["profile"] = stage_payload.get("profile", {})
        summary_stages[stage_name] = stage_summary
    return {
        "variant": raw_payload.get("variant"),
        "stages": summary_stages,
    }


def compare_stage_summaries(
    reference_stage: Mapping[str, Any], candidate_stage: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare candidate against reference for one stage."""
    reference_total = float(reference_stage.get("total_self_cuda_time_ms", 0.0))
    candidate_total = float(candidate_stage.get("total_self_cuda_time_ms", 0.0))
    total_delta = reference_total - candidate_total
    improvement_pct = None
    if reference_total:
        improvement_pct = round((total_delta / reference_total) * 100.0, 4)

    family_deltas: Dict[str, Any] = {}
    largest_recovered_family = None
    largest_delta = None

    reference_families = reference_stage.get("operator_families", {})
    candidate_families = candidate_stage.get("operator_families", {})
    for family_name in sorted(set(reference_families) | set(candidate_families)):
        ref_ms = float(reference_families.get(family_name, {}).get("self_cuda_time_ms", 0.0))
        cand_ms = float(candidate_families.get(family_name, {}).get("self_cuda_time_ms", 0.0))
        delta_ms = round(ref_ms - cand_ms, 6)
        family_improvement_pct = None
        if ref_ms:
            family_improvement_pct = round((delta_ms / ref_ms) * 100.0, 4)
        share_delta_pct_points = round(
            float(reference_families.get(family_name, {}).get("share_pct", 0.0))
            - float(candidate_families.get(family_name, {}).get("share_pct", 0.0)),
            4,
        )

        family_deltas[family_name] = {
            "reference_self_cuda_time_ms": ref_ms,
            "candidate_self_cuda_time_ms": cand_ms,
            "delta_ms": delta_ms,
            "improvement_pct": family_improvement_pct,
            "share_delta_pct_points": share_delta_pct_points,
        }

        if largest_delta is None or delta_ms > largest_delta:
            largest_delta = delta_ms
            largest_recovered_family = family_name

    return {
        "reference_total_self_cuda_time_ms": reference_total,
        "candidate_total_self_cuda_time_ms": candidate_total,
        "delta_ms": round(total_delta, 6),
        "total_self_cuda_time_improvement_pct": improvement_pct,
        "largest_recovered_family": largest_recovered_family,
        "family_deltas": family_deltas,
    }
