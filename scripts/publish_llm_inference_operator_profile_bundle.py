#!/usr/bin/env python3
"""Publish baseline / PaLU / aligned-GAC operator profiles into a tracked bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.gcompress_bench.operator_profile import compare_stage_summaries, summarize_profile_run


RUN_LAYOUT = {
    "baseline": {"label": "baseline"},
    "palu": {"label": "unaligned_palu"},
    "aligned_gac": {"label": "aligned_gac_palu"},
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def display_path(path: Path) -> str:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def copy_run_files(run_dir: Path, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    published_files: Dict[str, str] = {}
    for name in ("raw", "config", "summary", "env"):
        source = run_dir / f"{name}.json"
        if not source.exists():
            raise FileNotFoundError(f"Missing required run artifact: {source}")
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        published_files[name] = display_path(destination)
    return published_files


def build_variant_summary(raw_payload: Dict[str, Any], config_payload: Dict[str, Any]) -> Dict[str, Any]:
    summarized = summarize_profile_run(raw_payload)
    return {
        "config": {
            "variant": config_payload.get("variant"),
            "device": config_payload.get("device"),
            "dtype": config_payload.get("dtype"),
            "model_source": config_payload.get("model_source"),
            "stages": config_payload.get("stages"),
            "prefill_seq_len": config_payload.get("prefill_seq_len"),
            "decode_context_len": config_payload.get("decode_context_len"),
        },
        "stages": summarized.get("stages", {}),
    }


def build_comparisons(variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    pairs = {
        "palu_vs_baseline": ("baseline", "palu"),
        "aligned_gac_vs_palu": ("palu", "aligned_gac"),
        "aligned_gac_vs_baseline": ("baseline", "aligned_gac"),
    }

    for name, (reference_key, candidate_key) in pairs.items():
        reference = variants[reference_key]["stages"]
        candidate = variants[candidate_key]["stages"]
        stage_comparisons: Dict[str, Any] = {}
        for stage_name in sorted(set(reference) | set(candidate)):
            if stage_name not in reference or stage_name not in candidate:
                continue
            stage_comparisons[stage_name] = compare_stage_summaries(
                reference[stage_name],
                candidate[stage_name],
            )
        comparisons[name] = stage_comparisons
    return comparisons


def build_takeaways(comparisons: Dict[str, Any]) -> Dict[str, str]:
    takeaways: Dict[str, str] = {}
    aligned_vs_palu = comparisons.get("aligned_gac_vs_palu", {})
    for stage_name, payload in aligned_vs_palu.items():
        improvement_pct = payload.get("total_self_cuda_time_improvement_pct")
        family = payload.get("largest_recovered_family")
        if improvement_pct is None:
            continue
        takeaways[stage_name] = (
            f"Aligned GAC changes {stage_name} total self CUDA time by "
            f"{improvement_pct:.2f}% versus unaligned PaLU; largest recovered family: {family}."
        )
    return takeaways


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish LLM inference operator profile bundle")
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--palu-run-dir", type=Path, required=True)
    parser.add_argument("--aligned-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = {
        "baseline": args.baseline_run_dir,
        "palu": args.palu_run_dir,
        "aligned_gac": args.aligned_run_dir,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: Dict[str, Any] = {}
    manifest_runs: Dict[str, Any] = {}
    for variant_name, run_dir in run_dirs.items():
        raw_payload = load_json(run_dir / "raw.json")
        config_payload = load_json(run_dir / "config.json")
        variants[variant_name] = build_variant_summary(raw_payload, config_payload)

        publish_subdir = output_dir / variant_name
        manifest_runs[variant_name] = {
            "source_run_dir": display_path(run_dir),
            "published_files": copy_run_files(run_dir, publish_subdir),
        }

    comparisons = build_comparisons(variants)
    summary_payload = {
        "focus": "palu_inference_operator_profile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variants": variants,
        "comparisons": comparisons,
        "takeaways": build_takeaways(comparisons),
    }

    manifest_payload = {
        "focus": "palu_inference_operator_profile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_script": "scripts/publish_llm_inference_operator_profile_bundle.py",
        "published_runs": manifest_runs,
    }

    (output_dir / "palu_inference_operator_profile_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    print(f"Wrote LLM inference operator profile bundle to {output_dir}")


if __name__ == "__main__":
    main()
