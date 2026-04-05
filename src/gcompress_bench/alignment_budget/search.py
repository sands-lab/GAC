from typing import List

from .contract import HardwareContract


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
