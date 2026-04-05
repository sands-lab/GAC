from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_RECOMMENDED_VALUES = (32, 64, 96, 112, 128, 160, 192, 224, 256)


def repair_rank(
    rank: int,
    strategy: Optional[str] = None,
    max_overhead_pct: float = 20.0,
    recommended_values: Sequence[int] = DEFAULT_RECOMMENDED_VALUES,
) -> int:
    if strategy in (None, "", "none"):
        return rank

    if rank in recommended_values or rank % 16 == 0:
        return rank

    if strategy == "minimal":
        return ((rank + 7) // 8) * 8

    if strategy == "optimal":
        return ((rank + 15) // 16) * 16

    if strategy == "predefined":
        for value in recommended_values:
            if value >= rank:
                return value
        return ((rank + 15) // 16) * 16

    if strategy == "tradeoff":
        candidates = [
            ((rank + 7) // 8) * 8,
            ((rank + 15) // 16) * 16,
        ]
        for value in recommended_values:
            if value >= rank:
                candidates.append(value)
                break
        for candidate in sorted(set(candidates)):
            overhead = 100.0 * (candidate - rank) / rank if rank > 0 else 0.0
            if overhead <= max_overhead_pct:
                return candidate
        return ((rank + 7) // 8) * 8

    raise ValueError(f"Unsupported dimension repair strategy: {strategy}")


def repair_ranks(
    ranks: Iterable[int],
    strategy: Optional[str] = None,
    max_overhead_pct: float = 20.0,
    recommended_values: Sequence[int] = DEFAULT_RECOMMENDED_VALUES,
) -> List[int]:
    return [
        repair_rank(
            rank,
            strategy=strategy,
            max_overhead_pct=max_overhead_pct,
            recommended_values=recommended_values,
        )
        for rank in ranks
    ]


def repair_selection_result(
    selection_result: Dict[str, Sequence[int]],
    strategy: Optional[str] = None,
    max_overhead_pct: float = 20.0,
    recommended_values: Sequence[int] = DEFAULT_RECOMMENDED_VALUES,
) -> Dict[str, List[int]]:
    return {
        layer_name: repair_ranks(
            ranks,
            strategy=strategy,
            max_overhead_pct=max_overhead_pct,
            recommended_values=recommended_values,
        )
        for layer_name, ranks in selection_result.items()
    }
