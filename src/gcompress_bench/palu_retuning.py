"""Helpers for profile-guided PaLU rank retuning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping

from .alignment_budget.contract import HardwareContract
from .alignment_budget.search import estimate_hardware_penalty, generate_aligned_candidates


@dataclass(frozen=True)
class RetuningModule:
    """One PaLU projection whose per-group ranks move together."""

    name: str
    operator: str
    operator_role: str
    group_count: int
    base_rank: int
    aligned_rank: int
    importance: float


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return _round((float(numerator) / float(denominator)) * 100.0, 4)


def _module_operator(name: str) -> str:
    if ".k_proj" in name:
        return "k_proj"
    if ".v_proj" in name:
        return "v_proj"
    return "other"


def _normalize_head_wise_ranks(head_wise_ranks: Mapping[str, Iterable[int]]) -> Dict[str, List[int]]:
    normalized: Dict[str, List[int]] = {}
    for name, values in sorted(head_wise_ranks.items()):
        normalized[name] = [int(value) for value in values]
    return normalized


def load_head_wise_ranks(config_payload: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Extract and normalize PaLU `head_wise_ranks` from a checkpoint config payload."""
    head_wise_ranks = config_payload.get("head_wise_ranks")
    if not isinstance(head_wise_ranks, Mapping) or not head_wise_ranks:
        raise ValueError("checkpoint config is missing non-empty head_wise_ranks")
    return _normalize_head_wise_ranks(head_wise_ranks)


def build_profile_guided_weight_calibration(
    optimization_summary_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive the retuning penalty weight from the issue-35 optimization summary."""
    opportunities = optimization_summary_payload.get("opportunities")
    if not isinstance(opportunities, list):
        raise ValueError("optimization summary is missing opportunities")

    for opportunity in opportunities:
        if opportunity.get("id") != "prefill_profile_guided_rank_retuning":
            continue

        evidence = opportunity.get("evidence", {})
        alignment_sensitive_recovery_ms = float(
            evidence.get("alignment_sensitive_recovery_ms", 0.0)
        )
        raw_other_gemm_regression_ms = float(evidence.get("other_gemm_regression_ms", 0.0))
        other_gemm_regression_ms = abs(raw_other_gemm_regression_ms)
        attention_leakage_ms = float(evidence.get("attention_leakage_ms", 0.0))
        regression_weight = 1.0
        if alignment_sensitive_recovery_ms > 0.0:
            regression_weight = other_gemm_regression_ms / alignment_sensitive_recovery_ms

        leakage_ratio = 0.0
        if alignment_sensitive_recovery_ms > 0.0:
            leakage_ratio = attention_leakage_ms / alignment_sensitive_recovery_ms

        return {
            "target_opportunity": opportunity["id"],
            "alignment_sensitive_recovery_ms": _round(alignment_sensitive_recovery_ms),
            "other_gemm_regression_ms": _round(other_gemm_regression_ms),
            "raw_other_gemm_regression_ms": _round(raw_other_gemm_regression_ms),
            "attention_leakage_ms": _round(attention_leakage_ms),
            "regression_weight": _round(regression_weight),
            "attention_leakage_ratio": _round(leakage_ratio),
            "objective": (
                "maximize sqrt(base_rank / mean_base_rank) * retained_rank "
                "- regression_weight * estimated_hardware_penalty"
            ),
            "scoring_caveat": evidence.get("current_summary_caveat"),
        }

    raise ValueError(
        "optimization summary does not contain the prefill_profile_guided_rank_retuning opportunity"
    )


def _build_retuning_modules(
    base_head_wise_ranks: Mapping[str, Iterable[int]],
    aligned_head_wise_ranks: Mapping[str, Iterable[int]],
) -> List[RetuningModule]:
    base = _normalize_head_wise_ranks(base_head_wise_ranks)
    aligned = _normalize_head_wise_ranks(aligned_head_wise_ranks)
    if set(base) != set(aligned):
        raise ValueError("base and aligned configs do not cover the same PaLU modules")

    raw_records: List[tuple[str, str, str, int, int, int]] = []
    for name in sorted(base):
        base_values = base[name]
        aligned_values = aligned[name]
        if len(base_values) != len(aligned_values):
            raise ValueError(f"group count mismatch for module {name!r}")
        if len(set(base_values)) != 1 or len(set(aligned_values)) != 1:
            raise ValueError(
                "profile-guided retuning expects uniform per-group ranks inside each module; "
                f"module {name!r} is not uniform"
            )
        operator = _module_operator(name)
        operator_role = "attention_kv" if operator in {"k_proj", "v_proj"} else "linear"
        raw_records.append(
            (
                name,
                operator,
                operator_role,
                len(base_values),
                int(base_values[0]),
                int(aligned_values[0]),
            )
        )

    if not raw_records:
        raise ValueError("no PaLU modules were found for retuning")

    mean_base_rank = sum(record[4] for record in raw_records) / float(len(raw_records))
    modules: List[RetuningModule] = []
    for name, operator, operator_role, group_count, base_rank, aligned_rank in raw_records:
        importance = math.sqrt(base_rank / mean_base_rank)
        modules.append(
            RetuningModule(
                name=name,
                operator=operator,
                operator_role=operator_role,
                group_count=group_count,
                base_rank=base_rank,
                aligned_rank=aligned_rank,
                importance=importance,
            )
        )
    return modules


def _module_candidates(
    module: RetuningModule,
    contract: HardwareContract,
    full_rank: int,
    search_radius: int,
) -> List[int]:
    candidates = set(
        generate_aligned_candidates(
            module.base_rank,
            contract=contract,
            search_radius=search_radius,
        )
    )
    candidates.update(
        generate_aligned_candidates(
            module.aligned_rank,
            contract=contract,
            search_radius=search_radius,
        )
    )

    lower_bound = max(
        contract.minimal_alignment,
        min(module.base_rank, module.aligned_rank) - search_radius,
    )
    upper_bound = min(
        full_rank,
        max(module.base_rank, module.aligned_rank) + search_radius,
    )
    for value in contract.recommended_values:
        if lower_bound <= value <= upper_bound:
            candidates.add(int(value))

    candidates.add(module.aligned_rank)
    return sorted(
        candidate
        for candidate in candidates
        if contract.minimal_alignment <= candidate <= full_rank
        and candidate % contract.minimal_alignment == 0
    )


def summarize_rank_strategy(
    head_wise_ranks: Mapping[str, Iterable[int]],
    contract: HardwareContract,
) -> Dict[str, Any]:
    """Summarize one PaLU rank configuration using the shared hardware proxy."""
    normalized = _normalize_head_wise_ranks(head_wise_ranks)
    total_budget = 0
    total_groups = 0
    minimal_alignment_count = 0
    preferred_alignment_count = 0
    recommended_value_count = 0
    estimated_hardware_penalty_total = 0.0
    unique_ranks = set()
    min_rank = None
    max_rank = None

    for name, values in normalized.items():
        operator = _module_operator(name)
        operator_role = "attention_kv" if operator in {"k_proj", "v_proj"} else "linear"
        for value in values:
            total_budget += value
            total_groups += 1
            unique_ranks.add(value)
            if value % contract.minimal_alignment == 0:
                minimal_alignment_count += 1
            if value % contract.preferred_alignment == 0:
                preferred_alignment_count += 1
            if value in contract.recommended_values:
                recommended_value_count += 1
            estimated_hardware_penalty_total += estimate_hardware_penalty(
                value,
                contract=contract,
                operator_role=operator_role,
            )
            min_rank = value if min_rank is None else min(min_rank, value)
            max_rank = value if max_rank is None else max(max_rank, value)

    return {
        "module_count": len(normalized),
        "group_count": total_groups,
        "total_budget": int(total_budget),
        "average_rank": _round(total_budget / float(total_groups)) if total_groups else 0.0,
        "min_rank": int(min_rank or 0),
        "max_rank": int(max_rank or 0),
        "unique_rank_count": len(unique_ranks),
        "minimal_alignment_count": int(minimal_alignment_count),
        "minimal_alignment_pct": _pct(minimal_alignment_count, total_groups),
        "preferred_alignment_count": int(preferred_alignment_count),
        "preferred_alignment_pct": _pct(preferred_alignment_count, total_groups),
        "recommended_value_count": int(recommended_value_count),
        "recommended_value_pct": _pct(recommended_value_count, total_groups),
        "estimated_hardware_penalty": _round(estimated_hardware_penalty_total),
    }


def _materialize_retuned_config(
    modules: List[RetuningModule],
    selected_ranks: List[int],
) -> Dict[str, List[int]]:
    materialized: Dict[str, List[int]] = {}
    for module, selected_rank in zip(modules, selected_ranks):
        materialized[module.name] = [int(selected_rank)] * module.group_count
    return materialized


def _movement_summary(
    modules: List[RetuningModule],
    retuned_head_wise_ranks: Mapping[str, Iterable[int]],
    contract: HardwareContract,
    simple_aligned_summary: Mapping[str, Any],
    retuned_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized_retuned = _normalize_head_wise_ranks(retuned_head_wise_ranks)
    changes: List[Dict[str, Any]] = []
    for module in modules:
        retuned_rank = int(normalized_retuned[module.name][0])
        if retuned_rank == module.aligned_rank:
            continue
        simple_penalty = estimate_hardware_penalty(
            module.aligned_rank,
            contract=contract,
            operator_role=module.operator_role,
        ) * module.group_count
        retuned_penalty = estimate_hardware_penalty(
            retuned_rank,
            contract=contract,
            operator_role=module.operator_role,
        ) * module.group_count
        changes.append(
            {
                "name": module.name,
                "operator": module.operator,
                "group_count": module.group_count,
                "base_rank": module.base_rank,
                "simple_aligned_rank": module.aligned_rank,
                "retuned_rank": retuned_rank,
                "delta_per_group": retuned_rank - module.aligned_rank,
                "budget_delta": (retuned_rank - module.aligned_rank) * module.group_count,
                "simple_aligned_penalty": _round(simple_penalty),
                "retuned_penalty": _round(retuned_penalty),
                "penalty_delta": _round(retuned_penalty - simple_penalty),
            }
        )

    increases = sorted(
        (change for change in changes if change["delta_per_group"] > 0),
        key=lambda change: (
            change["delta_per_group"],
            change["budget_delta"],
            change["name"],
        ),
        reverse=True,
    )
    decreases = sorted(
        (change for change in changes if change["delta_per_group"] < 0),
        key=lambda change: (
            abs(change["delta_per_group"]),
            abs(change["budget_delta"]),
            change["name"],
        ),
        reverse=True,
    )

    total_budget_reallocated = sum(abs(change["budget_delta"]) for change in changes) // 2
    return {
        "changed_module_count": len(changes),
        "changed_group_count": sum(change["group_count"] for change in changes),
        "total_budget_reallocated": int(total_budget_reallocated),
        "budget_delta_vs_simple_aligned": int(
            retuned_summary["total_budget"] - simple_aligned_summary["total_budget"]
        ),
        "estimated_hardware_penalty_delta_vs_simple_aligned": _round(
            retuned_summary["estimated_hardware_penalty"]
            - simple_aligned_summary["estimated_hardware_penalty"]
        ),
        "recommended_value_delta_vs_simple_aligned": int(
            retuned_summary["recommended_value_count"]
            - simple_aligned_summary["recommended_value_count"]
        ),
        "top_rank_increases": increases[:8],
        "top_rank_decreases": decreases[:8],
        "module_changes": changes,
    }


def retune_palu_config(
    base_head_wise_ranks: Mapping[str, Iterable[int]],
    aligned_head_wise_ranks: Mapping[str, Iterable[int]],
    optimization_summary_payload: Mapping[str, Any],
    contract: HardwareContract,
    search_radius: int = 16,
) -> Dict[str, Any]:
    """Retune PaLU ranks under the aligned budget using a profile-guided DP."""
    modules = _build_retuning_modules(
        base_head_wise_ranks=base_head_wise_ranks,
        aligned_head_wise_ranks=aligned_head_wise_ranks,
    )
    weight_calibration = build_profile_guided_weight_calibration(
        optimization_summary_payload
    )

    full_rank = max(
        max(module.base_rank, module.aligned_rank) for module in modules
    )
    if contract.recommended_values:
        full_rank = max(full_rank, max(contract.recommended_values))

    group_gcd = 0
    for module in modules:
        group_gcd = math.gcd(group_gcd, module.group_count)
    budget_unit = contract.minimal_alignment * max(group_gcd, 1)

    target_budget = sum(module.aligned_rank * module.group_count for module in modules)
    if target_budget % budget_unit != 0:
        raise ValueError(
            f"aligned target budget {target_budget} is not divisible by budget unit {budget_unit}"
        )
    target_units = target_budget // budget_unit

    candidate_lists = [
        _module_candidates(
            module,
            contract=contract,
            full_rank=full_rank,
            search_radius=search_radius,
        )
        for module in modules
    ]

    negative_infinity = float("-inf")
    dp = [negative_infinity] * (target_units + 1)
    dp[0] = 0.0
    choice: List[List[int | None]] = [
        [None] * (target_units + 1) for _ in range(len(modules))
    ]

    for index, (module, candidates) in enumerate(zip(modules, candidate_lists)):
        next_dp = [negative_infinity] * (target_units + 1)
        for candidate in candidates:
            candidate_cost = (candidate * module.group_count) // budget_unit
            candidate_penalty = estimate_hardware_penalty(
                candidate,
                contract=contract,
                operator_role=module.operator_role,
            ) * module.group_count
            candidate_value = (
                module.importance * candidate * module.group_count
                - weight_calibration["regression_weight"] * candidate_penalty
            )
            for budget_index in range(candidate_cost, target_units + 1):
                previous = dp[budget_index - candidate_cost]
                if previous == negative_infinity:
                    continue
                score = previous + candidate_value
                if score > next_dp[budget_index]:
                    next_dp[budget_index] = score
                    choice[index][budget_index] = candidate
        dp = next_dp

    if dp[target_units] == negative_infinity:
        raise ValueError("unable to find an exact budget-preserving retuning solution")

    selected_ranks: List[int] = []
    remaining_units = target_units
    for index in range(len(modules) - 1, -1, -1):
        candidate = choice[index][remaining_units]
        if candidate is None:
            raise RuntimeError("retuning DP backtracking failed")
        selected_ranks.append(candidate)
        remaining_units -= (candidate * modules[index].group_count) // budget_unit
    selected_ranks.reverse()

    retuned_head_wise_ranks = _materialize_retuned_config(modules, selected_ranks)
    base_summary = summarize_rank_strategy(base_head_wise_ranks, contract=contract)
    simple_aligned_summary = summarize_rank_strategy(aligned_head_wise_ranks, contract=contract)
    retuned_summary = summarize_rank_strategy(retuned_head_wise_ranks, contract=contract)
    movement_summary = _movement_summary(
        modules=modules,
        retuned_head_wise_ranks=retuned_head_wise_ranks,
        contract=contract,
        simple_aligned_summary=simple_aligned_summary,
        retuned_summary=retuned_summary,
    )

    summary = {
        "focus": "palu_profile_guided_rank_retuning",
        "source_focus": {
            "optimization_summary": optimization_summary_payload.get("focus"),
            "target_opportunity": weight_calibration["target_opportunity"],
        },
        "hardware_contract": {
            "minimal_alignment": contract.minimal_alignment,
            "preferred_alignment": contract.preferred_alignment,
            "recommended_values": list(contract.recommended_values),
        },
        "search_parameters": {
            "search_radius": int(search_radius),
            "budget_unit": int(budget_unit),
            "importance_mode": "sqrt_base_rank_ratio",
        },
        "weight_calibration": weight_calibration,
        "strategy_summaries": {
            "base": base_summary,
            "simple_aligned": simple_aligned_summary,
            "profile_guided_retuned": retuned_summary,
        },
        "movement_summary": movement_summary,
        "guardrails": [
            "This is an offline, budget-preserving retuning artifact, not a measured GPU speedup.",
            "Candidate scoring uses the issue-35 root-cause split so regressions in other_gemm_kernels matter during selection.",
            "The current score is a hardware proxy only; real latency still requires rerunning the Slurm/A100 operator profiler.",
        ],
    }

    retuned_config_payload = {
        "focus": "palu_profile_guided_rank_retuning_config",
        "hardware_contract": "a100",
        "search_radius": int(search_radius),
        "head_wise_ranks": retuned_head_wise_ranks,
    }
    return {
        "retuned_head_wise_ranks": retuned_head_wise_ranks,
        "retuned_config_payload": retuned_config_payload,
        "summary": summary,
    }

