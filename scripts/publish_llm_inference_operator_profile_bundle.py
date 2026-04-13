#!/usr/bin/env python3
"""Publish baseline / PaLU / aligned-GAC operator profiles into a tracked bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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

VARIANT_ORDER = ("baseline", "palu", "aligned_gac")
FAMILY_ORDER = (
    ("gemm", "GEMM"),
    ("elementwise", "Elementwise"),
    ("data_movement", "Data movement"),
    ("sdpa", "SDPA"),
    ("other", "Other"),
)
COMPARISON_ORDER = (
    ("palu_vs_baseline", "baseline", "palu"),
    ("aligned_gac_vs_palu", "palu", "aligned_gac"),
    ("aligned_gac_vs_baseline", "baseline", "aligned_gac"),
)


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


def format_ms(value: float) -> str:
    return f"{value:.2f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def format_signed_ms(value: float) -> str:
    return f"{value:+.2f}"


def family_share(stage_payload: Dict[str, Any], family_name: str) -> float:
    family_payload = stage_payload.get("operator_families", {}).get(family_name, {})
    return float(family_payload.get("share_pct", 0.0))


def largest_family_delta(
    comparison_payload: Dict[str, Any], prefer_positive: bool
) -> Tuple[str, float]:
    family_deltas = comparison_payload.get("family_deltas", {})
    items = [
        (family_name, float(family_payload.get("delta_ms", 0.0)))
        for family_name, family_payload in family_deltas.items()
    ]
    if not items:
        return "n/a", 0.0
    key_fn = max if prefer_positive else min
    return key_fn(items, key=lambda item: item[1])


def render_stage_totals_table(summary_payload: Dict[str, Any], stage_name: str) -> str:
    lines = [
        "| Variant | Total self CUDA time (ms) | GEMM | Elementwise | Data movement | SDPA | Other |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_name in VARIANT_ORDER:
        stage_payload = summary_payload["variants"][variant_name]["stages"][stage_name]
        lines.append(
            "| `{variant}` | {total} | {gemm} | {elementwise} | {data_movement} | {sdpa} | {other} |".format(
                variant=variant_name,
                total=format_ms(stage_payload["total_self_cuda_time_ms"]),
                gemm=format_pct(family_share(stage_payload, "gemm")),
                elementwise=format_pct(family_share(stage_payload, "elementwise")),
                data_movement=format_pct(family_share(stage_payload, "data_movement")),
                sdpa=format_pct(family_share(stage_payload, "sdpa")),
                other=format_pct(family_share(stage_payload, "other")),
            )
        )
    return "\n".join(lines)


def render_comparison_table(summary_payload: Dict[str, Any], stage_name: str) -> str:
    lines = [
        "| Comparison | Total delta (ms) | Delta (%) | Key family delta |",
        "| --- | ---: | ---: | --- |",
    ]
    for comparison_name, reference_name, candidate_name in COMPARISON_ORDER:
        comparison_payload = summary_payload["comparisons"][comparison_name][stage_name]
        delta_ms = float(comparison_payload["delta_ms"])
        improve_positive = delta_ms >= 0
        family_name, family_delta = largest_family_delta(
            comparison_payload, prefer_positive=improve_positive
        )
        lines.append(
            "| `{candidate}` vs `{reference}` | {delta_ms} | {delta_pct} | `{family}` {family_delta} ms |".format(
                candidate=candidate_name,
                reference=reference_name,
                delta_ms=format_signed_ms(delta_ms),
                delta_pct=format_pct(comparison_payload.get("total_self_cuda_time_improvement_pct")),
                family=family_name,
                family_delta=format_signed_ms(family_delta),
            )
        )
    return "\n".join(lines)


def render_stage_interpretation(summary_payload: Dict[str, Any], stage_name: str) -> str:
    gemm_shares = [
        family_share(summary_payload["variants"][variant]["stages"][stage_name], "gemm")
        for variant in VARIANT_ORDER
    ]
    sdpa_shares = [
        family_share(summary_payload["variants"][variant]["stages"][stage_name], "sdpa")
        for variant in VARIANT_ORDER
    ]
    aligned_vs_palu = summary_payload["comparisons"]["aligned_gac_vs_palu"][stage_name]
    aligned_vs_baseline = summary_payload["comparisons"]["aligned_gac_vs_baseline"][stage_name]
    gemm_delta = float(
        aligned_vs_palu["family_deltas"]["gemm"]["delta_ms"]
    )
    baseline_gap = abs(float(aligned_vs_baseline["delta_ms"]))

    if stage_name == "prefill":
        return (
            "GEMM stays between {gemm_min} and {gemm_max} of prefill self CUDA time across all "
            "three variants, so the aligned gain is mostly a GEMM recovery: `aligned_gac` pulls back "
            "{gemm_delta} ms of GEMM time versus `palu`, but it still trails `baseline` by {baseline_gap} ms overall.".format(
                gemm_min=format_pct(min(gemm_shares)),
                gemm_max=format_pct(max(gemm_shares)),
                gemm_delta=format_ms(gemm_delta),
                baseline_gap=format_ms(baseline_gap),
            )
        )

    return (
        "Decode remains projection-dominated: GEMM stays between {gemm_min} and {gemm_max}, while "
        "SDPA is only {sdpa_min} to {sdpa_max}. That means the small aligned gain is not coming from "
        "the attention kernel itself; `aligned_gac` recovers {gemm_delta} ms of GEMM time versus `palu`, "
        "but still sits {baseline_gap} ms behind `baseline`.".format(
            gemm_min=format_pct(min(gemm_shares)),
            gemm_max=format_pct(max(gemm_shares)),
            sdpa_min=format_pct(min(sdpa_shares)),
            sdpa_max=format_pct(max(sdpa_shares)),
            gemm_delta=format_ms(gemm_delta),
            baseline_gap=format_ms(baseline_gap),
        )
    )


def summary_bullets(summary_payload: Dict[str, Any]) -> Iterable[str]:
    palu_vs_baseline = summary_payload["comparisons"]["palu_vs_baseline"]
    aligned_vs_palu = summary_payload["comparisons"]["aligned_gac_vs_palu"]
    aligned_vs_baseline = summary_payload["comparisons"]["aligned_gac_vs_baseline"]

    prefill_regression = palu_vs_baseline["prefill"]
    decode_regression = palu_vs_baseline["decode"]
    prefill_recovery = aligned_vs_palu["prefill"]
    decode_recovery = aligned_vs_palu["decode"]
    prefill_gap = aligned_vs_baseline["prefill"]
    decode_gap = aligned_vs_baseline["decode"]

    return (
        (
            "`palu` is slower than `baseline` in both stages: prefill adds "
            f"{format_ms(abs(float(prefill_regression['delta_ms'])))} ms "
            f"({format_pct(abs(float(prefill_regression['total_self_cuda_time_improvement_pct'])) )}) "
            "and decode adds "
            f"{format_ms(abs(float(decode_regression['delta_ms'])))} ms "
            f"({format_pct(abs(float(decode_regression['total_self_cuda_time_improvement_pct'])) )}). "
            "In both stages the dominant regression family is `gemm`."
        ),
        (
            "Aligned GAC recovers "
            f"{format_ms(float(prefill_recovery['delta_ms']))} ms "
            f"({format_pct(prefill_recovery['total_self_cuda_time_improvement_pct'])}) in prefill and "
            f"{format_ms(float(decode_recovery['delta_ms']))} ms "
            f"({format_pct(decode_recovery['total_self_cuda_time_improvement_pct'])}) in decode versus "
            "`palu`. The largest recovered family is `gemm` in both stages."
        ),
        (
            "`aligned_gac` still trails `baseline` by "
            f"{format_ms(abs(float(prefill_gap['delta_ms'])))} ms "
            f"({format_pct(abs(float(prefill_gap['total_self_cuda_time_improvement_pct'])) )}) in prefill and "
            f"{format_ms(abs(float(decode_gap['delta_ms'])))} ms "
            f"({format_pct(abs(float(decode_gap['total_self_cuda_time_improvement_pct'])) )}) in decode, "
            "so alignment closes only part of the compressed-model gap."
        ),
    )


def build_analysis_markdown(summary_payload: Dict[str, Any]) -> str:
    prefill_takeaway = summary_payload.get("takeaways", {}).get("prefill", "")
    decode_takeaway = summary_payload.get("takeaways", {}).get("decode", "")
    lines = [
        "# PaLU Inference Operator Profile Analysis",
        "",
        "This note is the human-readable companion to `palu_inference_operator_profile_summary.json`.",
        "It compares `baseline`, `palu`, and `aligned_gac` using self CUDA time aggregated by operator family.",
        "Positive `delta_ms` means the candidate reduced self CUDA time relative to the reference; negative `delta_ms` means regression.",
        "",
        "## Executive Summary",
    ]
    for bullet in summary_bullets(summary_payload):
        lines.append(f"- {bullet}")

    if prefill_takeaway or decode_takeaway:
        lines.extend(["", "## Bundle Takeaways"])
        if prefill_takeaway:
            lines.append(f"- Prefill: {prefill_takeaway}")
        if decode_takeaway:
            lines.append(f"- Decode: {decode_takeaway}")

    for stage_name in ("prefill", "decode"):
        lines.extend(
            [
                "",
                f"## {stage_name.title()}",
                "",
                render_stage_totals_table(summary_payload, stage_name),
                "",
                render_comparison_table(summary_payload, stage_name),
                "",
                render_stage_interpretation(summary_payload, stage_name),
            ]
        )

    lines.extend(
        [
            "",
            "## Artifact Map",
            "",
            "- `palu_inference_operator_profile_summary.json`: machine-readable per-stage totals, family shares, and pairwise comparisons.",
            "- `source_manifest.json`: provenance for the source run directories and copied artifacts.",
            "- `submission_status.json`: Slurm provenance and job-tracking state for the real A100 collection.",
        ]
    )
    return "\n".join(lines) + "\n"


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
        json.dumps(summary_payload, indent=2) + "\n"
    )
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest_payload, indent=2) + "\n")
    (output_dir / "analysis.md").write_text(build_analysis_markdown(summary_payload))

    print(f"Wrote LLM inference operator profile bundle to {output_dir}")


if __name__ == "__main__":
    main()
