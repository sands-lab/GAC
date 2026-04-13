#!/usr/bin/env python3
"""Generate a budget-preserving profile-guided rank-retuning artifact for PaLU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.gcompress_bench.alignment_budget import get_hardware_contract  # noqa: E402
from src.gcompress_bench.palu_retuning import (  # noqa: E402
    load_head_wise_ranks,
    retune_palu_config,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def display_path(path: Path) -> str:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _iter_changes(
    changes: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return list(changes)


def render_markdown(summary_payload: Mapping[str, Any]) -> str:
    strategies = summary_payload["strategy_summaries"]
    base = strategies["base"]
    simple_aligned = strategies["simple_aligned"]
    retuned = strategies["profile_guided_retuned"]
    movement = summary_payload["movement_summary"]
    weights = summary_payload["weight_calibration"]
    increases = _iter_changes(movement["top_rank_increases"])
    decreases = _iter_changes(movement["top_rank_decreases"])

    lines = [
        "# PaLU Profile-Guided Rank Retuning",
        "",
        "This profile-guided rank retuning artifact re-allocates the existing aligned budget across PaLU `k_proj` / `v_proj` modules.",
        "It is budget-preserving and not a measured GPU speedup.",
        "",
        "## Why This Follow-Up Exists",
        "- The issue-35 optimization summary says the current simple aligned baseline removes the measurable alignment-sensitive tail, but still gives back meaningful time in `other_gemm_kernels`.",
        "- The retuner therefore keeps the simple aligned baseline budget fixed while searching for lower-penalty contract-aware rank choices near the existing `rb1` / `rb1-gac-a100` configs.",
        "- The score is root-cause-aware: it uses both recovered alignment-sensitive kernels and the `other_gemm_kernels` regression weight instead of minimizing alignment penalty alone.",
        "",
        "## Strategy Snapshot",
        f"- Base budget: `{base['total_budget']}` with estimated hardware penalty `{base['estimated_hardware_penalty']}`.",
        f"- Simple aligned baseline: `{simple_aligned['total_budget']}` with estimated hardware penalty `{simple_aligned['estimated_hardware_penalty']}` and `{simple_aligned['recommended_value_count']}` recommended-value groups.",
        f"- Profile-guided retuned: `{retuned['total_budget']}` with estimated hardware penalty `{retuned['estimated_hardware_penalty']}` and `{retuned['recommended_value_count']}` recommended-value groups.",
        f"- Estimated penalty delta versus the simple aligned baseline: `{movement['estimated_hardware_penalty_delta_vs_simple_aligned']}`.",
        f"- Changed modules: `{movement['changed_module_count']}` modules / `{movement['changed_group_count']}` groups.",
        "",
        "## Weight Calibration",
        f"- `alignment_sensitive_recovery_ms`: `{weights['alignment_sensitive_recovery_ms']}`",
        f"- `other_gemm_kernels` regression magnitude: `{weights['other_gemm_regression_ms']}` ms",
        f"- `attention_leakage_ms`: `{weights['attention_leakage_ms']}`",
        f"- Derived `regression_weight`: `{weights['regression_weight']}`",
        f"- Objective: `{weights['objective']}`",
        "",
        "## Largest Rank Increases",
    ]

    if increases:
        for change in increases:
            lines.append(
                "- `{name}`: `{simple_aligned_rank}` -> `{retuned_rank}` per group, "
                "budget delta `{budget_delta}`, penalty `{simple_aligned_penalty}` -> `{retuned_penalty}`.".format(
                    **change
                )
            )
    else:
        lines.append("- No upward reallocations were needed.")

    lines.extend(
        [
            "",
            "## Largest Rank Decreases",
        ]
    )
    if decreases:
        for change in decreases:
            lines.append(
                "- `{name}`: `{simple_aligned_rank}` -> `{retuned_rank}` per group, "
                "budget delta `{budget_delta}`, penalty `{simple_aligned_penalty}` -> `{retuned_penalty}`.".format(
                    **change
                )
            )
    else:
        lines.append("- No downward reallocations were needed.")

    lines.extend(
        [
            "",
            "## Caveats",
            "- This is a budget-preserving offline retuning note, not a measured GPU speedup.",
            "- The proxy objective only estimates whether the candidate should reduce the simple aligned baseline's hardware penalty; it does not replace profiler traces.",
            "- Any claim about real end-to-end inference wins still requires rerunning the Slurm/A100 operator profile collection for the retuned config.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_analysis_payload(
    base_config_path: Path,
    aligned_config_path: Path,
    optimization_summary_path: Path,
    hardware_contract_name: str,
    search_radius: int,
) -> Dict[str, Any]:
    base_config_payload = load_json(base_config_path)
    aligned_config_payload = load_json(aligned_config_path)
    optimization_summary_payload = load_json(optimization_summary_path)
    contract = get_hardware_contract(hardware_contract_name)

    analysis = retune_palu_config(
        base_head_wise_ranks=load_head_wise_ranks(base_config_payload),
        aligned_head_wise_ranks=load_head_wise_ranks(aligned_config_payload),
        optimization_summary_payload=optimization_summary_payload,
        contract=contract,
        search_radius=search_radius,
    )
    analysis["retuned_config_payload"]["hardware_contract"] = hardware_contract_name
    summary = dict(analysis["summary"])
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["source_paths"] = {
        "base_config": display_path(base_config_path),
        "aligned_config": display_path(aligned_config_path),
        "optimization_summary": display_path(optimization_summary_path),
    }
    return {
        "summary": summary,
        "retuned_config_payload": analysis["retuned_config_payload"],
        "retuned_head_wise_ranks": analysis["retuned_head_wise_ranks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--aligned-config", type=Path, required=True)
    parser.add_argument("--optimization-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-contract", default="a100")
    parser.add_argument("--search-radius", type=int, default=16)
    args = parser.parse_args()

    analysis = build_analysis_payload(
        base_config_path=args.base_config,
        aligned_config_path=args.aligned_config,
        optimization_summary_path=args.optimization_summary,
        hardware_contract_name=args.hardware_contract,
        search_radius=args.search_radius,
    )
    report = render_markdown(analysis["summary"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "profile_guided_rank_retuning_summary.json"
    report_path = args.output_dir / "profile_guided_rank_retuning.md"
    config_path = args.output_dir / "profile_guided_rank_retuning_config.json"

    summary_path.write_text(json.dumps(analysis["summary"], indent=2) + "\n")
    report_path.write_text(report)
    config_path.write_text(json.dumps(analysis["retuned_config_payload"], indent=2) + "\n")

    print(f"Wrote profile-guided rank retuning summary to {summary_path}")
    print(f"Wrote profile-guided rank retuning report to {report_path}")
    print(f"Wrote profile-guided rank retuning config to {config_path}")


if __name__ == "__main__":
    main()
