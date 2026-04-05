"""Alignment-aware budget allocation prototype."""

from .contract import HardwareContract, NormalizedBudgetItem
from .search import generate_aligned_candidates

__all__ = [
    "HardwareContract",
    "NormalizedBudgetItem",
    "generate_aligned_candidates",
]
