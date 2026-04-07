from typing import Dict, List

from .contract import HardwareContract, NormalizedBudgetItem


def generate_aligned_candidates(
    original_budget: int,
    contract: HardwareContract,
    search_radius: int = 16,
) -> List[int]:
    """Generate nearby aligned candidates under a simple hardware contract."""
    lower = max(contract.minimal_alignment, original_budget - search_radius)
    upper = original_budget + search_radius
    candidates = set()

    for value in range(lower, upper + 1):
        if value % contract.minimal_alignment == 0 and value not in contract.cliff_values:
            candidates.add(value)

    for value in contract.recommended_values:
        if lower <= value <= upper and value not in contract.cliff_values:
            candidates.add(value)

    if not candidates:
        rounded = ((original_budget + contract.minimal_alignment - 1) // contract.minimal_alignment) * contract.minimal_alignment
        candidates.add(rounded)

    return sorted(candidates)


def estimate_hardware_penalty(
    budget: int,
    contract: HardwareContract,
    operator_role: str = "linear",
) -> float:
    """Estimate a simple latency proxy where lower is better."""
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    penalty = 0.0

    if budget in contract.cliff_values:
        penalty += 100.0

    if budget % contract.minimal_alignment != 0:
        penalty += 20.0 + float(budget % contract.minimal_alignment)
    elif budget % contract.preferred_alignment != 0:
        penalty += 3.0

    if contract.recommended_values:
        nearest_recommended_gap = min(abs(budget - value) for value in contract.recommended_values)
        penalty += nearest_recommended_gap / float(contract.minimal_alignment)

    if operator_role == "attention_kv":
        penalty *= 1.25

    return penalty


def choose_aligned_candidate(
    item: NormalizedBudgetItem,
    contract: HardwareContract,
    max_overhead_pct: float = 20.0,
    search_radius: int = 16,
) -> int:
    """Pick the best aligned candidate under a bounded-overhead budget."""
    max_budget = int(item.original_budget * (1.0 + max_overhead_pct / 100.0))
    candidates = generate_aligned_candidates(
        item.original_budget,
        contract=contract,
        search_radius=search_radius,
    )

    feasible_candidates = [candidate for candidate in candidates if candidate <= max_budget]
    if not feasible_candidates:
        raise ValueError(
            "no feasible aligned candidates for "
            f"{item.name} within {max_overhead_pct}% overhead"
        )

    def candidate_score(candidate: int) -> tuple[float, int]:
        deviation_cost = item.importance * abs(candidate - item.original_budget) * item.cost_per_unit
        hardware_cost = estimate_hardware_penalty(
            candidate,
            contract=contract,
            operator_role=item.operator_role,
        )
        return (deviation_cost + hardware_cost, candidate)

    return min(feasible_candidates, key=candidate_score)


def summarize_alignment(
    original_items: List[NormalizedBudgetItem],
    aligned_items: List[NormalizedBudgetItem],
    contract: HardwareContract,
) -> Dict[str, float]:
    """Summarize total budget and latency-proxy deltas for one alignment run."""
    total_original_budget = sum(item.original_budget for item in original_items)
    total_aligned_budget = sum(item.original_budget for item in aligned_items)
    original_hardware_penalty = sum(
        estimate_hardware_penalty(
            item.original_budget,
            contract=contract,
            operator_role=item.operator_role,
        )
        for item in original_items
    )
    aligned_hardware_penalty = sum(
        estimate_hardware_penalty(
            item.original_budget,
            contract=contract,
            operator_role=item.operator_role,
        )
        for item in aligned_items
    )
    overhead_pct = 0.0
    if total_original_budget > 0:
        overhead_pct = 100.0 * (total_aligned_budget - total_original_budget) / float(total_original_budget)

    return {
        "total_original_budget": float(total_original_budget),
        "total_aligned_budget": float(total_aligned_budget),
        "budget_overhead_pct": overhead_pct,
        "original_hardware_penalty": float(original_hardware_penalty),
        "aligned_hardware_penalty": float(aligned_hardware_penalty),
    }
