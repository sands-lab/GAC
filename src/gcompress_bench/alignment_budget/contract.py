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


def _misaligned_values(start: int, stop: int, alignment: int) -> Tuple[int, ...]:
    """Enumerate coarse cliff values inside a profiled range."""
    return tuple(value for value in range(start, stop + 1) if value % alignment != 0)


HARDWARE_CONTRACTS: Dict[str, HardwareContract] = {
    "default": HardwareContract(),
    "a100": HardwareContract(),
    "h100": HardwareContract(
        minimal_alignment=8,
        preferred_alignment=16,
        recommended_values=(64, 128, 192, 256),
        # Current Hopper evidence in this repo is coarse-grained: the paper and
        # figures confirm severe fallback outside mod-8 fast paths, but do not
        # yet provide a checked-in raw table for finer Hopper-only cliffs.
        cliff_values=_misaligned_values(64, 256, 8),
    ),
}


def get_hardware_contract(name: str | None = None) -> HardwareContract:
    """Return a named hardware contract with conservative defaults."""
    if name is None:
        name = "default"

    normalized_name = name.strip().lower()
    try:
        return HARDWARE_CONTRACTS[normalized_name]
    except KeyError as exc:
        available = ", ".join(sorted(HARDWARE_CONTRACTS))
        raise ValueError(
            f"unknown hardware contract {name!r}; available contracts: {available}"
        ) from exc
