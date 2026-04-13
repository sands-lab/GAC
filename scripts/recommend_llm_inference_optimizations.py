#!/usr/bin/env python3
"""Generate ranked inference optimization opportunities from tracked profiler artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.gcompress_bench.inference_optimization import (  # noqa: E402
    build_inference_speed_optimization_summary,
    render_inference_speed_optimization_markdown,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend next inference optimizations from operator-profile and GEMM root-cause artifacts.",
    )
    parser.add_argument(
        "--operator-profile-summary",
        type=Path,
        required=True,
        help="Path to palu_inference_operator_profile_summary.json",
    )
    parser.add_argument(
        "--gemm-root-cause-summary",
        type=Path,
        required=True,
        help="Path to gemm_root_cause_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for inference_speed_optimization_summary.json and inference_speed_optimization.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operator_profile_summary = load_json(args.operator_profile_summary)
    gemm_root_cause_summary = load_json(args.gemm_root_cause_summary)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = build_inference_speed_optimization_summary(
        operator_profile_summary=operator_profile_summary,
        gemm_root_cause_summary=gemm_root_cause_summary,
    )
    markdown = render_inference_speed_optimization_markdown(summary_payload)

    summary_path = output_dir / "inference_speed_optimization_summary.json"
    report_path = output_dir / "inference_speed_optimization.md"
    save_json(summary_path, summary_payload)
    report_path.write_text(markdown)

    print(f"Wrote inference speed optimization summary to {summary_path}")
    print(f"Wrote inference speed optimization report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
