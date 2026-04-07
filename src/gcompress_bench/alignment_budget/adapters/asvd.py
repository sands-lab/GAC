from typing import Dict, List

from ..contract import NormalizedBudgetItem
from ..search import choose_aligned_candidate, summarize_alignment


class ASVDBudgetAdapter:
    """Adapter for ASVD-style per-projection rank allocations."""

    method_name = "asvd"

    def export_items(self, rank_config: Dict[str, int]) -> List[NormalizedBudgetItem]:
        items: List[NormalizedBudgetItem] = []
        for name, rank in sorted(rank_config.items()):
            operator_role = "attention_kv" if any(token in name for token in ("k_proj", "v_proj")) else "linear"
            items.append(
                NormalizedBudgetItem(
                    name=name,
                    method=self.method_name,
                    budget_kind="rank",
                    operator_role=operator_role,
                    original_budget=int(rank),
                    granularity="per_layer",
                    constraints={
                        "config_key": name,
                    },
                )
            )
        return items

    def materialize(self, items: List[NormalizedBudgetItem]) -> Dict[str, int]:
        return {
            str(item.constraints["config_key"]): int(item.original_budget)
            for item in items
        }

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
        rank_config: Dict[str, int],
        contract,
        max_overhead_pct: float = 20.0,
        search_radius: int = 16,
    ) -> Dict[str, object]:
        original_items = self.export_items(rank_config)
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
