#!/usr/bin/env python3
"""Publish PaLU fixed-length rerun outputs into the repo-tracked bundle."""

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict


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


def publish_file(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def build_hardware(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    config = results_payload.get("config", {})
    environment = results_payload.get("environment", {})
    hardware = {
        "gpu": "unknown",
        "device": config.get("device", "cuda:0"),
        "dtype": config.get("dtype", "float16"),
    }
    cuda_info = environment.get("cuda")
    if isinstance(cuda_info, dict):
        gpu_name = cuda_info.get("device_name") or cuda_info.get("name")
        if isinstance(gpu_name, str) and gpu_name.strip():
            hardware["gpu"] = gpu_name
    return hardware


def rewrite_sources(entry: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    payload = copy.deepcopy(entry)
    if payload.get("status") == "measured":
        payload["source"] = source_path
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="PaLU run directory under results/C5")
    parser.add_argument("--publish-dir", required=True, help="Tracked artifact directory for copied raw files")
    parser.add_argument("--bundle-dir", required=True, help="PaLU bundle directory under notes/")
    parser.add_argument("--decode-root-cause", required=True, help="Existing decode root-cause summary path")
    parser.add_argument("--aligned-build-summary", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    publish_dir = Path(args.publish_dir)
    bundle_dir = Path(args.bundle_dir)
    decode_root_cause = Path(args.decode_root_cause)
    aligned_build_summary = Path(args.aligned_build_summary) if args.aligned_build_summary else None

    comparison_summary_path = run_dir / "comparison_summary.json"
    results_path = run_dir / "results.json"
    if not comparison_summary_path.exists():
        raise FileNotFoundError(f"Missing comparison summary: {comparison_summary_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing run results: {results_path}")
    if not decode_root_cause.exists():
        raise FileNotFoundError(f"Missing decode root-cause summary: {decode_root_cause}")

    comparison_summary = load_json(comparison_summary_path)
    results_payload = load_json(results_path)
    hardware = build_hardware(results_payload)

    tracked_comparison_summary = publish_file(comparison_summary_path, publish_dir)
    tracked_results = publish_file(results_path, publish_dir)

    latency_summary = copy.deepcopy(comparison_summary)
    latency_summary["hardware"] = hardware
    latency_summary["measurement_scope"] = {
        "prefill": f"aggregated from the same-hardware fixed-length rerun in {run_dir.name}",
        "decode": (
            f"aggregated from the same-hardware fixed-length rerun in {run_dir.name} "
            "with min_new_tokens=gen"
        ),
    }
    for section_name in ("prefill_latency_ms", "decode_latency_ms"):
        section = latency_summary.get(section_name, {})
        for variant_name, payload in list(section.items()):
            section[variant_name] = rewrite_sources(payload, display_path(tracked_comparison_summary))

    latency_sources = {
        "fixed_length_run_summary": display_path(tracked_comparison_summary),
        "fixed_length_run_results": display_path(tracked_results),
    }
    if aligned_build_summary is not None and aligned_build_summary.exists():
        latency_sources["aligned_checkpoint_build_summary"] = display_path(aligned_build_summary)
    latency_summary["sources"] = latency_sources
    latency_summary["notes"] = [
        f"This file tracks the issue-30 fixed-length rerun stored under {display_path(publish_dir)}.",
        "All decode entries in this summary were measured with max_new_tokens=gen, min_new_tokens=gen, and actual_new_tokens recorded.",
        "The aligned_gac figures come from the checkpoint-level GAC path rebuilt in the clean issue-30 environment.",
    ]

    prefill_section = {}
    decode_section = {}
    for variant_name in ("baseline", "unaligned", "aligned_gac"):
        prefill_payload = comparison_summary["prefill_latency_ms"][variant_name]
        decode_payload = comparison_summary["decode_latency_ms"][variant_name]

        prefill_entry = {"status": prefill_payload["status"]}
        if prefill_payload["status"] == "measured":
            prefill_entry.update(
                {
                    "average_latency_ms": prefill_payload["average_latency_ms"],
                    "average_throughput_tok_s": prefill_payload.get("average_throughput_tok_s"),
                    "source": display_path(tracked_comparison_summary),
                }
            )
        else:
            prefill_entry["reason"] = prefill_payload.get("reason", "measurement unavailable")
        prefill_section[variant_name] = prefill_entry

        decode_entry = {"status": decode_payload["status"]}
        if decode_payload["status"] == "measured":
            decode_entry.update(
                {
                    "average_latency_ms": decode_payload["average_latency_ms"],
                    "average_throughput_tok_s": decode_payload.get("average_throughput_tok_s"),
                    "actual_new_tokens": decode_payload.get("actual_new_tokens", {}),
                    "source": display_path(tracked_comparison_summary),
                }
            )
        else:
            decode_entry["reason"] = decode_payload.get("reason", "measurement unavailable")
        decode_section[variant_name] = decode_entry

    unaligned_prefill = prefill_section["unaligned"].get("average_throughput_tok_s", 0.0) or 0.0
    aligned_prefill = prefill_section["aligned_gac"].get("average_throughput_tok_s", 0.0) or 0.0
    unaligned_decode = decode_section["unaligned"].get("average_throughput_tok_s", 0.0) or 0.0
    aligned_decode = decode_section["aligned_gac"].get("average_throughput_tok_s", 0.0) or 0.0

    prefill_delta_pct = (
        ((aligned_prefill / unaligned_prefill) - 1.0) * 100.0 if unaligned_prefill else 0.0
    )
    decode_delta_pct = (
        ((aligned_decode / unaligned_decode) - 1.0) * 100.0 if unaligned_decode else 0.0
    )
    primary_gain_is_decode = abs(decode_delta_pct) > abs(prefill_delta_pct)

    fixed_length_summary = {
        "measurement_contract": {
            "run_id": run_dir.name,
            "hardware": hardware,
            "decode_length_mode": "fixed_new_tokens",
            "generation_guard": "max_new_tokens=gen, min_new_tokens=gen",
            "benchmark_overrides": {
                "warmup": results_payload.get("config", {}).get("warmup"),
                "measure": results_payload.get("config", {}).get("measure"),
                "trials": results_payload.get("config", {}).get("trials"),
            },
            "notes": (
                "This rerun was produced after deleting old local experiment outputs and "
                "rebuilding the PaLU checkpoints in the issue-30 clean environment."
            ),
        },
        "prefill_comparison": prefill_section,
        "fixed_length_decode_comparison": decode_section,
        "conclusion": {
            "primary_gain_is_decode": primary_gain_is_decode,
            "summary": (
                f"Under the issue-30 fixed-length rerun, aligned_gac changes prefill throughput by "
                f"{prefill_delta_pct:+.2f}% versus unaligned and decode throughput by "
                f"{decode_delta_pct:+.2f}%."
            ),
        },
        "sources": [
            display_path(tracked_comparison_summary),
            display_path(tracked_results),
            display_path(decode_root_cause),
        ],
    }

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "latency_comparison.json").write_text(json.dumps(latency_summary, indent=2) + "\n")
    (bundle_dir / "fixed_length_decode_comparison.json").write_text(
        json.dumps(fixed_length_summary, indent=2) + "\n"
    )
    print(f"Wrote PaLU latency summary to {bundle_dir / 'latency_comparison.json'}")
    print(f"Wrote PaLU fixed-length summary to {bundle_dir / 'fixed_length_decode_comparison.json'}")


if __name__ == "__main__":
    main()
