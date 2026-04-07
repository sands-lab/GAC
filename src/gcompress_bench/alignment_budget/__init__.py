"""Alignment-aware budget allocation prototype."""

from .contract import HardwareContract, NormalizedBudgetItem
from .search import (
    choose_aligned_candidate,
    estimate_hardware_penalty,
    generate_aligned_candidates,
    summarize_alignment,
)

__all__ = [
    "HardwareContract",
    "NormalizedBudgetItem",
    "choose_aligned_candidate",
    "estimate_hardware_penalty",
    "generate_aligned_candidates",
    "summarize_alignment",
]
