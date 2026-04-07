"""Alignment-aware budget allocation prototype."""

from .contract import HardwareContract, NormalizedBudgetItem, get_hardware_contract
from .search import (
    choose_aligned_candidate,
    estimate_hardware_penalty,
    generate_aligned_candidates,
    summarize_alignment,
)

__all__ = [
    "HardwareContract",
    "NormalizedBudgetItem",
    "get_hardware_contract",
    "choose_aligned_candidate",
    "estimate_hardware_penalty",
    "generate_aligned_candidates",
    "summarize_alignment",
]
