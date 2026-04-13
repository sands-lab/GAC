"""Helpers for PaLU grouped-reconstruction dispatch analysis."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return _round((float(numerator) / float(denominator)) * 100.0, 4)


def _factor(before: float, after: float) -> float:
    if after == 0:
        return 0.0
    return _round(float(before) / float(after), 4)


def summarize_checkpoint_dispatch(config_payload: Mapping[str, Any]) -> Dict[str, Any]:
    head_wise_ranks = config_payload.get("head_wise_ranks")
    if not isinstance(head_wise_ranks, dict) or not head_wise_ranks:
        raise ValueError("Checkpoint config is missing non-empty head_wise_ranks.")

    module_summaries = []
    totals = {
        "module_count": 0,
        "group_count": 0,
        "vt_total_calls": 0,
        "u_dispatch_before": 0,
        "u_dispatch_after": 0,
        "per_group_total_calls": 0,
        "grouped_total_calls": 0,
        "rank_slots_before": 0,
        "rank_slots_after_padding": 0,
    }
    by_operator: Dict[str, Dict[str, float]] = {}

    for name, values in sorted(head_wise_ranks.items()):
        if ".k_proj" in name:
            operator = "k_proj"
        elif ".v_proj" in name:
            operator = "v_proj"
        else:
            continue

        ranks = [int(value) for value in values]
        if not ranks:
            continue

        group_count = len(ranks)
        max_rank = max(ranks)
        rank_slots_before = sum(ranks)
        rank_slots_after_padding = max_rank * group_count
        padding_rank_slots = rank_slots_after_padding - rank_slots_before
        per_group_total_calls = 1 + group_count
        grouped_total_calls = 2

        module_summary = {
            "name": name,
            "operator": operator,
            "group_count": group_count,
            "min_rank": min(ranks),
            "max_rank": max_rank,
            "rank_slots_before": rank_slots_before,
            "rank_slots_after_padding": rank_slots_after_padding,
            "padding_rank_slots": padding_rank_slots,
            "padding_overhead_pct": _pct(padding_rank_slots, rank_slots_before),
            "per_group_total_calls": per_group_total_calls,
            "grouped_total_calls": grouped_total_calls,
            "u_dispatch_before": group_count,
            "u_dispatch_after": 1,
        }
        module_summaries.append(module_summary)

        totals["module_count"] += 1
        totals["group_count"] += group_count
        totals["vt_total_calls"] += 1
        totals["u_dispatch_before"] += group_count
        totals["u_dispatch_after"] += 1
        totals["per_group_total_calls"] += per_group_total_calls
        totals["grouped_total_calls"] += grouped_total_calls
        totals["rank_slots_before"] += rank_slots_before
        totals["rank_slots_after_padding"] += rank_slots_after_padding

        bucket = by_operator.setdefault(
            operator,
            {
                "module_count": 0,
                "group_count": 0,
                "per_group_total_calls": 0,
                "grouped_total_calls": 0,
                "rank_slots_before": 0,
                "rank_slots_after_padding": 0,
            },
        )
        bucket["module_count"] += 1
        bucket["group_count"] += group_count
        bucket["per_group_total_calls"] += per_group_total_calls
        bucket["grouped_total_calls"] += grouped_total_calls
        bucket["rank_slots_before"] += rank_slots_before
        bucket["rank_slots_after_padding"] += rank_slots_after_padding

    if not module_summaries:
        raise ValueError("Checkpoint config does not contain PaLU k_proj/v_proj head_wise_ranks.")

    for operator, payload in by_operator.items():
        padding_rank_slots = payload["rank_slots_after_padding"] - payload["rank_slots_before"]
        payload["reduction_factor"] = _factor(
            payload["per_group_total_calls"],
            payload["grouped_total_calls"],
        )
        payload["padding_rank_slots"] = padding_rank_slots
        payload["padding_overhead_pct"] = _pct(padding_rank_slots, payload["rank_slots_before"])

    padding_rank_slots = totals["rank_slots_after_padding"] - totals["rank_slots_before"]
    top_modules = sorted(
        module_summaries,
        key=lambda item: (item["padding_overhead_pct"], item["padding_rank_slots"], item["name"]),
        reverse=True,
    )[:5]

    return {
        "module_count": totals["module_count"],
        "group_count": totals["group_count"],
        "avg_groups_per_module": _round(totals["group_count"] / totals["module_count"], 4),
        "vt_total_calls": totals["vt_total_calls"],
        "u_dispatch_before": totals["u_dispatch_before"],
        "u_dispatch_after": totals["u_dispatch_after"],
        "u_reduction_factor": _factor(totals["u_dispatch_before"], totals["u_dispatch_after"]),
        "per_group_total_calls": totals["per_group_total_calls"],
        "grouped_total_calls": totals["grouped_total_calls"],
        "reduction_factor": _factor(totals["per_group_total_calls"], totals["grouped_total_calls"]),
        "rank_slots_before": totals["rank_slots_before"],
        "rank_slots_after_padding": totals["rank_slots_after_padding"],
        "padding_rank_slots": padding_rank_slots,
        "padding_overhead_pct": _pct(padding_rank_slots, totals["rank_slots_before"]),
        "by_operator": by_operator,
        "top_modules_by_padding_overhead": top_modules,
    }


def validate_grouped_reconstruction(module: torch.nn.Module, hidden_states: torch.Tensor) -> Dict[str, Any]:
    if not hasattr(module, "set_reconstruct_strategy"):
        raise TypeError("Prototype module does not expose set_reconstruct_strategy().")
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must be a rank-3 tensor.")

    module.eval()
    grouped_u_linear_calls_observed = 0

    with torch.no_grad():
        module.set_reconstruct_strategy("per_group")
        legacy_output = module(hidden_states)

        originals = []
        try:
            for linear in module.U:
                original_forward = linear.forward
                originals.append((linear, original_forward))

                def counted_forward(x, _original=original_forward):
                    nonlocal grouped_u_linear_calls_observed
                    grouped_u_linear_calls_observed += 1
                    return _original(x)

                linear.forward = counted_forward

            module.set_reconstruct_strategy("grouped_bmm")
            grouped_output = module(hidden_states)
        finally:
            for linear, original_forward in originals:
                linear.forward = original_forward
            module.set_reconstruct_strategy("per_group")

    ranks = [int(rank) for rank in module.ranks]
    padding_rank_slots = max(ranks) * len(ranks) - sum(ranks)
    max_abs_diff = float((legacy_output - grouped_output).abs().max().item())

    return {
        "ranks": ranks,
        "hidden_states_shape": list(hidden_states.shape),
        "output_shape": list(legacy_output.shape),
        "legacy_reconstruct_calls": len(ranks),
        "grouped_reconstruct_calls": 1,
        "grouped_u_linear_calls_observed": grouped_u_linear_calls_observed,
        "padding_rank_slots": padding_rank_slots,
        "padding_overhead_pct": _pct(padding_rank_slots, sum(ranks)),
        "max_abs_diff": _round(max_abs_diff, 8),
    }
