#!/usr/bin/env python3
"""Analyze a grouped-reconstruction prototype for PaLU prefill dispatch reduction."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.gcompress_bench.palu_dispatch import (  # noqa: E402
    summarize_checkpoint_dispatch,
    validate_grouped_reconstruction,
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


def load_headwise_low_rank_module_class():
    palu_root = ROOT / "third_party" / "palu" / "palu"
    package_paths = {
        "palu": palu_root,
        "palu.model": palu_root / "model",
        "palu.model.modules": palu_root / "model" / "modules",
    }

    if "fast_hadamard_transform" not in sys.modules:
        stub = types.ModuleType("fast_hadamard_transform")

        def _missing_fast_hadamard_transform(*args, **kwargs):
            raise RuntimeError("fast_hadamard_transform is unavailable in this environment.")

        stub.hadamard_transform = _missing_fast_hadamard_transform
        sys.modules["fast_hadamard_transform"] = stub

    for name, path in package_paths.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    return importlib.import_module("palu.model.modules.svd_linear").HeadwiseLowRankModule


def build_hidden_states(spec: Dict[str, Any]) -> torch.Tensor:
    seed = int(spec.get("seed", 0))
    batch_size = int(spec["batch_size"])
    seq_len = int(spec["seq_len"])
    in_features = int(spec["in_features"])
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)


def initialize_module(module: HeadwiseLowRankModule, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, dtype=param.dtype, device=param.device))


def build_summary(
    checkpoint_config_path: Path,
    prototype_spec_path: Path,
) -> Dict[str, Any]:
    checkpoint_config = load_json(checkpoint_config_path)
    prototype_spec = load_json(prototype_spec_path)

    checkpoint_dispatch = summarize_checkpoint_dispatch(checkpoint_config)
    headwise_module_cls = load_headwise_low_rank_module_class()
    prototype_module = headwise_module_cls(
        ranks=[int(rank) for rank in prototype_spec["ranks"]],
        in_features=int(prototype_spec["in_features"]),
        out_features=int(prototype_spec["out_features"]),
        bias=bool(prototype_spec.get("bias", False)),
    )
    initialize_module(prototype_module, int(prototype_spec.get("seed", 0)))
    hidden_states = build_hidden_states(prototype_spec)
    prototype_validation = validate_grouped_reconstruction(prototype_module, hidden_states)
    prototype_validation["seed"] = int(prototype_spec.get("seed", 0))

    return {
        "focus": "palu_prefill_dispatch_reduction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_config": display_path(checkpoint_config_path),
        "prototype_spec": display_path(prototype_spec_path),
        "checkpoint_dispatch": checkpoint_dispatch,
        "prototype_validation": prototype_validation,
        "summary": (
            "Grouping HeadwiseLowRankModule reconstruction reduces projection-path dispatches by collapsing "
            "per-group U[*] linears into one grouped reconstruction call, while preserving outputs on a deterministic prototype."
        ),
        "guardrails": [
            "This is a code-path prototype for grouped reconstruction, not a measured GPU speedup.",
            "The checkpoint summary reports dispatch-count and padding-overhead tradeoffs only; real prefill gains still require Slurm/A100 profiling.",
            "The current prototype reduces U[*] reconstruction dispatches but does not fuse the VT projection itself.",
        ],
    }


def render_markdown(summary_payload: Dict[str, Any]) -> str:
    checkpoint = summary_payload["checkpoint_dispatch"]
    prototype = summary_payload["prototype_validation"]
    top_modules = checkpoint["top_modules_by_padding_overhead"]

    lines = [
        "# PaLU Prefill Dispatch Reduction Prototype",
        "",
        "This note records a grouped reconstruction prototype for `HeadwiseLowRankModule` on the PaLU `k_proj` / `v_proj` path.",
        "It is not a measured GPU speedup; it only verifies a code-path-level dispatch reduction contract.",
        "",
        "## Checkpoint Dispatch Summary",
        f"- Affected modules: `{checkpoint['module_count']}`",
        f"- Total head groups: `{checkpoint['group_count']}`",
        f"- Current projection-path dispatch: `{checkpoint['per_group_total_calls']}` total calls (`{checkpoint['vt_total_calls']}` `VT` calls + `{checkpoint['u_dispatch_before']}` per-group `U[*]` calls).",
        f"- Grouped reconstruction dispatch: `{checkpoint['grouped_total_calls']}` total calls (`{checkpoint['vt_total_calls']}` `VT` calls + `{checkpoint['u_dispatch_after']}` grouped reconstruction calls).",
        f"- Projection-path reduction factor: `{checkpoint['reduction_factor']:.2f}x`",
        f"- `U[*]`-only reduction factor: `{checkpoint['u_reduction_factor']:.2f}x`",
        f"- Padding overhead: `{checkpoint['padding_rank_slots']}` extra latent rank slots (`{checkpoint['padding_overhead_pct']:.2f}%`).",
        "",
        "## Prototype Validation",
        f"- Prototype ranks: `{prototype['ranks']}`",
        f"- Input shape: `{prototype['hidden_states_shape']}`",
        f"- Legacy reconstruct dispatches: `{prototype['legacy_reconstruct_calls']}`",
        f"- Grouped reconstruct dispatches: `{prototype['grouped_reconstruct_calls']}`",
        f"- Observed per-group `U[*]` calls under grouped reconstruction: `{prototype['grouped_u_linear_calls_observed']}`",
        f"- Max abs diff between legacy and grouped reconstruction: `{prototype['max_abs_diff']}`",
        f"- Prototype padding overhead: `{prototype['padding_rank_slots']}` latent slots (`{prototype['padding_overhead_pct']:.2f}%`).",
        "",
        "## Highest Padding Modules",
    ]

    for item in top_modules:
        lines.append(
            "- `{name}`: {groups} groups, max rank {max_rank}, padding overhead {padding_pct:.2f}% ({padding} extra slots).".format(
                name=item["name"],
                groups=item["group_count"],
                max_rank=item["max_rank"],
                padding_pct=item["padding_overhead_pct"],
                padding=item["padding_rank_slots"],
            )
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "- This grouped reconstruction prototype keeps the existing `VT` projection untouched; it only changes the `U[*]` reconstruction step.",
            "- The artifact reports dispatch counts and padding overhead, not a measured GPU speedup.",
            "- Any claim about real prefill latency still needs profiler traces on the Slurm/A100 path used by `operator_profile_issue35`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--prototype-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(
        checkpoint_config_path=args.checkpoint_config,
        prototype_spec_path=args.prototype_spec,
    )
    report = render_markdown(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "prefill_dispatch_reduction_summary.json"
    report_path = args.output_dir / "prefill_dispatch_reduction.md"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(report)

    print(f"Wrote prefill dispatch reduction summary to {summary_path}")
    print(f"Wrote prefill dispatch reduction report to {report_path}")


if __name__ == "__main__":
    main()
