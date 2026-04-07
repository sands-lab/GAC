from collections import defaultdict
from typing import Dict, Iterable, List

from ..contract import NormalizedBudgetItem
from ..search import choose_aligned_candidate, summarize_alignment


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

    def align_items(
        self,
        items: List[NormalizedBudgetItem],
        contract,
        max_overhead_pct: float = 20.0,
        search_radius: int = 16,
    ) -> List[NormalizedBudgetItem]:
        aligned_items: List[NormalizedBudgetItem] = []
        for item in items:
            aligned_budget = choose_aligned_candidate(
                item,
                contract=contract,
                max_overhead_pct=max_overhead_pct,
                search_radius=search_radius,
            )
            aligned_items.append(
                NormalizedBudgetItem(
                    name=item.name,
                    method=item.method,
                    budget_kind=item.budget_kind,
                    operator_role=item.operator_role,
                    original_budget=aligned_budget,
                    importance=item.importance,
                    cost_per_unit=item.cost_per_unit,
                    granularity=item.granularity,
                    constraints=dict(item.constraints),
                )
            )
        return aligned_items

    def align_config(
        self,
        head_wise_ranks: Dict[str, Iterable[int]],
        contract,
        max_overhead_pct: float = 20.0,
        search_radius: int = 16,
    ) -> Dict[str, object]:
        original_items = self.export_items(head_wise_ranks)
        aligned_items = self.align_items(
            original_items,
            contract=contract,
            max_overhead_pct=max_overhead_pct,
            search_radius=search_radius,
        )
        return {
            "original_items": original_items,
            "aligned_items": aligned_items,
            "aligned_config": self.materialize(aligned_items),
            "summary": summarize_alignment(
                original_items,
                aligned_items,
                contract=contract,
            ),
        }
