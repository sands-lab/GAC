from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class HardwareContract:
    """Minimal hardware-facing alignment rules for budget search."""

    minimal_alignment: int = 8
    preferred_alignment: int = 16
    recommended_values: Tuple[int, ...] = (32, 64, 96, 112, 128, 160, 192, 224, 256)
    cliff_values: Tuple[int, ...] = ()


@dataclass(frozen=True)
class NormalizedBudgetItem:
    """Method-neutral representation of one alignable budget decision."""

    name: str
    method: str
    budget_kind: str
    operator_role: str
    original_budget: int
    importance: float = 1.0
    cost_per_unit: int = 1
    granularity: str = "scalar"
    constraints: Dict[str, Any] = field(default_factory=dict)
