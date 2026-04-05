from collections import defaultdict
from typing import Dict, Iterable, List

from ..contract import NormalizedBudgetItem


class PaluBudgetAdapter:
    """Adapter for PaLU `head_wise_ranks` config dictionaries."""

    method_name = "palu"

    def export_items(self, head_wise_ranks: Dict[str, Iterable[int]]) -> List[NormalizedBudgetItem]:
        items: List[NormalizedBudgetItem] = []
        for name, ranks in sorted(head_wise_ranks.items()):
            operator_role = "attention_kv" if any(token in name for token in ("k_proj", "v_proj")) else "linear"
            for index, rank in enumerate(ranks):
                items.append(
                    NormalizedBudgetItem(
                        name=f"{name}#group{index}",
                        method=self.method_name,
                        budget_kind="rank",
                        operator_role=operator_role,
                        original_budget=int(rank),
                        granularity="per_group",
                        constraints={
                            "config_key": name,
                            "group_index": index,
                            "group_count": len(list(ranks)) if not isinstance(ranks, list) else len(ranks),
                        },
                    )
                )
        return items

    def materialize(self, items: List[NormalizedBudgetItem]) -> Dict[str, List[int]]:
        grouped: Dict[str, Dict[int, int]] = defaultdict(dict)
        for item in items:
            config_key = item.constraints["config_key"]
            group_index = int(item.constraints["group_index"])
            grouped[config_key][group_index] = int(item.original_budget)

        materialized: Dict[str, List[int]] = {}
        for key, values in grouped.items():
            materialized[key] = [value for _, value in sorted(values.items())]
        return materialized
