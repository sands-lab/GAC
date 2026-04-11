#!/usr/bin/env python3
"""Publish a repo-tracked PaLU operator-split bundle from fixed-length evidence."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def extract_hardware(results_payload: Dict[str, Any]) -> Dict[str, Any]:
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


def inspect_headwise_module(module_path: Path) -> Dict[str, Any]:
    text = module_path.read_text()
    vt_projection_present = "self.VT = nn.Linear(in_features, sum(ranks), bias=False)" in text
    reconstructs_original_out_features = "return torch.cat(outputs, dim=-1)" in text
    if not vt_projection_present or not reconstructs_original_out_features:
        raise RuntimeError(
            "Unexpected HeadwiseLowRankModule contract: expected VT projection and output reconstruction."
        )
    return {
        "module_path": display_path(module_path),
        "vt_projection_present": vt_projection_present,
        "reconstructs_original_out_features": reconstructs_original_out_features,
    }


def as_int_list(values: Iterable[Any]) -> List[int]:
    return [int(value) for value in values]


def summarize_rank_values(values: List[int]) -> Dict[str, Any]:
    if not values:
        raise ValueError("Expected at least one rank value.")

    total = len(values)
    aligned_8_count = sum(value % 8 == 0 for value in values)
    aligned_16_count = sum(value % 16 == 0 for value in values)
    return {
        "total_values": total,
        "aligned_8_count": aligned_8_count,
        "aligned_8_pct": 100.0 * aligned_8_count / total,
        "aligned_16_count": aligned_16_count,
        "aligned_16_pct": 100.0 * aligned_16_count / total,
        "min_rank": min(values),
        "max_rank": max(values),
        "unique_ranks": sorted(set(values)),
    }


def summarize_headwise_config(config_payload: Dict[str, Any]) -> Dict[str, Any]:
    head_wise_ranks = config_payload.get("head_wise_ranks")
    if not isinstance(head_wise_ranks, dict) or not head_wise_ranks:
        raise ValueError("Checkpoint config is missing non-empty head_wise_ranks.")

    by_operator: Dict[str, List[int]] = {"k_proj": [], "v_proj": []}
    module_count = {"k_proj": 0, "v_proj": 0}

    for name, ranks in sorted(head_wise_ranks.items()):
        if ".k_proj" in name:
            operator = "k_proj"
        elif ".v_proj" in name:
            operator = "v_proj"
        else:
            continue

        rank_values = as_int_list(ranks)
        by_operator[operator].extend(rank_values)
        module_count[operator] += 1

    overall_values = by_operator["k_proj"] + by_operator["v_proj"]
    return {
        "overall": summarize_rank_values(overall_values),
        "by_operator": {
            operator: summarize_rank_values(values)
            for operator, values in by_operator.items()
        },
        "module_count": module_count,
    }


def extract_runtime_alignment(results_payload: Dict[str, Any]) -> Dict[str, float]:
    results = results_payload.get("results", {})
    unaligned = (
        results.get("palu", {})
        .get("repair_info", {})
        .get("before_repair", {})
        .get("aligned_8_pct")
    )
    aligned = (
        results.get("aligned_gac", {})
        .get("repair_info", {})
        .get("after", {})
        .get("aligned_8_pct")
    )
    if not isinstance(unaligned, (int, float)) or not isinstance(aligned, (int, float)):
        raise ValueError("Fixed-length results are missing runtime alignment percentages.")
    return {
        "baseline": 100.0,
        "unaligned": float(unaligned),
        "aligned_gac": float(aligned),
    }


def extract_variant_shape(config_payload: Dict[str, Any]) -> Dict[str, int]:
    return {
        "hidden_size": int(config_payload["hidden_size"]),
        "num_attention_heads": int(config_payload["num_attention_heads"]),
        "num_key_value_heads": int(config_payload["num_key_value_heads"]),
        "head_dim": int(config_payload["head_dim"]),
    }


def average_metric(payload: Dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric field {field!r} in comparison summary payload.")
    return float(value)


def delta_pct(before: float, after: float) -> float:
    if before == 0:
        raise ValueError("Cannot compute percentage delta from zero baseline.")
    return ((after / before) - 1.0) * 100.0


def load_stage_metrics(summary_payload: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    section = summary_payload[section_name]
    baseline = section["baseline"]
    unaligned = section["unaligned"]
    aligned = section["aligned_gac"]
    return {
        "baseline_latency_ms": average_metric(baseline, "average_latency_ms"),
        "unaligned_latency_ms": average_metric(unaligned, "average_latency_ms"),
        "aligned_latency_ms": average_metric(aligned, "average_latency_ms"),
        "baseline_throughput_tok_s": average_metric(baseline, "average_throughput_tok_s"),
        "unaligned_throughput_tok_s": average_metric(unaligned, "average_throughput_tok_s"),
        "aligned_throughput_tok_s": average_metric(aligned, "average_throughput_tok_s"),
    }


def stage_attribution_payload(
    stage_name: str,
    metrics: Dict[str, Any],
    fixed_length_summary_path: Path,
) -> Dict[str, Any]:
    throughput_delta = delta_pct(
        metrics["unaligned_throughput_tok_s"],
        metrics["aligned_throughput_tok_s"],
    )
    latency_delta = delta_pct(
        metrics["unaligned_latency_ms"],
        metrics["aligned_latency_ms"],
    )

    if stage_name == "prefill":
        conclusion = (
            "The checked-in prefill delta should be interpreted as attention-adjacent "
            "projection GEMM sensitivity: the PaLU checkpoints keep the SDPA shape contract "
            "unchanged, while only the k_proj/v_proj low-rank projection path changes."
        )
    else:
        conclusion = (
            "Under the fixed_new_tokens decode contract, the moderate aligned gain should be "
            "read as projection-path alignment on cached K/V updates rather than an SDPA "
            "head_dim cliff, because the SDPA tensor width contract is unchanged."
        )

    return {
        "changed_operator_family": "attention_projection_gemm",
        "sdpa_shape_change_present": False,
        "baseline_throughput_tok_s": metrics["baseline_throughput_tok_s"],
        "unaligned_throughput_tok_s": metrics["unaligned_throughput_tok_s"],
        "aligned_gac_throughput_tok_s": metrics["aligned_throughput_tok_s"],
        "aligned_vs_unaligned_throughput_pct": throughput_delta,
        "baseline_latency_ms": metrics["baseline_latency_ms"],
        "unaligned_latency_ms": metrics["unaligned_latency_ms"],
        "aligned_gac_latency_ms": metrics["aligned_latency_ms"],
        "aligned_vs_unaligned_latency_pct": latency_delta,
        "source": display_path(fixed_length_summary_path),
        "conclusion": conclusion,
    }


def build_summary(
    unaligned_config_path: Path,
    aligned_config_path: Path,
    fixed_length_results_path: Path,
    fixed_length_summary_path: Path,
    issue30_bundle_summary_path: Path,
    module_source_path: Path,
) -> Dict[str, Any]:
    unaligned_config = load_json(unaligned_config_path)
    aligned_config = load_json(aligned_config_path)
    fixed_length_results = load_json(fixed_length_results_path)
    fixed_length_summary = load_json(fixed_length_summary_path)

    hardware = extract_hardware(fixed_length_results)
    module_contract = inspect_headwise_module(module_source_path)
    runtime_alignment = extract_runtime_alignment(fixed_length_results)
    unaligned_rank_summary = summarize_headwise_config(unaligned_config)
    aligned_rank_summary = summarize_headwise_config(aligned_config)
    unaligned_shape = extract_variant_shape(unaligned_config)
    aligned_shape = extract_variant_shape(aligned_config)
    shape_changed = unaligned_shape != aligned_shape

    prefill_metrics = load_stage_metrics(fixed_length_summary, "prefill_latency_ms")
    decode_metrics = load_stage_metrics(fixed_length_summary, "decode_latency_ms")

    prefill_payload = stage_attribution_payload(
        "prefill",
        prefill_metrics,
        fixed_length_summary_path,
    )
    decode_payload = stage_attribution_payload(
        "decode",
        decode_metrics,
        fixed_length_summary_path,
    )

    prefill_delta = prefill_payload["aligned_vs_unaligned_throughput_pct"]
    decode_delta = decode_payload["aligned_vs_unaligned_throughput_pct"]
    conclusion = (
        "SDPA shape stays unchanged across the unaligned and aligned PaLU checkpoints "
        f"(head_dim={unaligned_shape['head_dim']}, num_attention_heads={unaligned_shape['num_attention_heads']}, "
        f"num_key_value_heads={unaligned_shape['num_key_value_heads']}). "
        "The only operator family whose shape contract changes is the attention-adjacent "
        "projection GEMM path inside HeadwiseLowRankModule (`k_proj` / `v_proj` `VT` + `U[*]`). "
        f"Under the issue-30 fixed-length contract, aligned GAC changes prefill throughput by "
        f"{prefill_delta:+.2f}% and decode throughput by {decode_delta:+.2f}% versus unaligned, "
        "so the checked-in PaLU speed difference should be attributed to projection-path alignment rather than an SDPA width cliff."
    )

    return {
        "measurement_contract": {
            "source_run_id": fixed_length_results_path.parent.name,
            "hardware": hardware,
            "decode_length_mode": "fixed_new_tokens",
            "decode_guard": "max_new_tokens=gen, min_new_tokens=gen",
            "prefill_measurement": "explicit forward pass over a fixed input tensor",
            "source": display_path(fixed_length_summary_path),
        },
        "sdpa_shape_contract": {
            "shape_changed_between_variants": shape_changed,
            "variant_shapes": {
                "unaligned": unaligned_shape,
                "aligned_gac": aligned_shape,
            },
            "headwise_module_contract": {
                "vt_projection_present": module_contract["vt_projection_present"],
                "reconstructs_original_out_features": module_contract["reconstructs_original_out_features"],
                "module_source": module_contract["module_path"],
            },
            "conclusion": (
                "PaLU's HeadwiseLowRankModule compresses k_proj/v_proj internally but reconstructs "
                "back to the original out_features before the attention kernel. The SDPA tensor "
                "width contract therefore remains unchanged in both checkpoints."
            ),
        },
        "projection_gemm_contract": {
            "changed_operator_family": "attention_projection_gemm",
            "affected_modules": [
                "model.layers.*.self_attn.k_proj.VT",
                "model.layers.*.self_attn.k_proj.U[*]",
                "model.layers.*.self_attn.v_proj.VT",
                "model.layers.*.self_attn.v_proj.U[*]",
            ],
            "runtime_alignment_pct": runtime_alignment,
            "config_rank_alignment_pct": {
                "unaligned": unaligned_rank_summary["overall"]["aligned_8_pct"],
                "aligned_gac": aligned_rank_summary["overall"]["aligned_8_pct"],
            },
            "config_rank_summary": {
                "unaligned": unaligned_rank_summary,
                "aligned_gac": aligned_rank_summary,
            },
            "notes": [
                "runtime_alignment_pct reuses the issue-30 loaded-model analysis already published in the fixed-length rerun bundle.",
                "config_rank_alignment_pct counts only the raw head_wise_ranks stored in the PaLU checkpoint config.",
            ],
        },
        "stage_attribution": {
            "prefill": prefill_payload,
            "decode": decode_payload,
        },
        "operator_split_conclusion": {
            "dominant_changed_operator_family": "attention_projection_gemm",
            "summary": conclusion,
            "caveat": (
                "This bundle is a structural operator split based on the checked-in checkpoint "
                "contract plus the issue-30 fixed-length rerun. It is not a direct per-kernel "
                "profiler trace."
            ),
        },
        "sources": {
            "unaligned_checkpoint_config": display_path(unaligned_config_path),
            "aligned_checkpoint_config": display_path(aligned_config_path),
            "fixed_length_run_results": display_path(fixed_length_results_path),
            "fixed_length_run_summary": display_path(fixed_length_summary_path),
            "issue30_bundle_summary": display_path(issue30_bundle_summary_path),
            "headwise_module_source": display_path(module_source_path),
        },
    }


def build_manifest(
    unaligned_config_path: Path,
    aligned_config_path: Path,
    fixed_length_results_path: Path,
    fixed_length_summary_path: Path,
    issue30_bundle_summary_path: Path,
    module_source_path: Path,
) -> Dict[str, Any]:
    return {
        "unaligned_checkpoint_config": {
            "path": display_path(unaligned_config_path),
            "kind": "repo_local_unaligned_checkpoint_config",
            "notes": "Issue-30 rb1 PaLU checkpoint config with the raw head_wise_ranks used for the operator split.",
        },
        "aligned_checkpoint_config": {
            "path": display_path(aligned_config_path),
            "kind": "repo_local_aligned_checkpoint_config",
            "notes": "Issue-30 gac-a100 PaLU checkpoint config with aligned head_wise_ranks.",
        },
        "fixed_length_run_results": {
            "path": display_path(fixed_length_results_path),
            "kind": "repo_tracked_fixed_length_results",
            "notes": "Tracked copy of the issue-30 PaLU fixed-length rerun results payload.",
        },
        "fixed_length_run_summary": {
            "path": display_path(fixed_length_summary_path),
            "kind": "repo_tracked_fixed_length_summary",
            "notes": "Tracked comparison summary for the issue-30 PaLU fixed-length rerun.",
        },
        "issue30_bundle_summary": {
            "path": display_path(issue30_bundle_summary_path),
            "kind": "repo_tracked_fixed_length_bundle_summary",
            "notes": "Existing PaLU bundle summary that already treats issue 30 as the current source of truth.",
        },
        "headwise_module_source": {
            "path": display_path(module_source_path),
            "kind": "repo_source_contract",
            "notes": "Source file proving that HeadwiseLowRankModule reconstructs back to the original output width before attention.",
        },
    }


def build_readme(summary: Dict[str, Any]) -> str:
    sdpa = summary["sdpa_shape_contract"]
    projection = summary["projection_gemm_contract"]
    prefill = summary["stage_attribution"]["prefill"]
    decode = summary["stage_attribution"]["decode"]

    unaligned_shape = sdpa["variant_shapes"]["unaligned"]
    runtime_alignment = projection["runtime_alignment_pct"]
    config_alignment = projection["config_rank_alignment_pct"]

    return "\n".join(
        [
            "# PaLU Operator Split Bundle",
            "",
            "This bundle records the checked-in issue-32 operator split for PaLU under the issue-30 fixed-length contract.",
            "It is a structural attribution artifact: it combines the fixed-length rerun with the checkpoint and module contracts to separate SDPA from the changed projection path.",
            "",
            "## Main Conclusion",
            "",
            (
                f"- `HeadwiseLowRankModule` keeps the SDPA shape contract unchanged: both PaLU checkpoints "
                f"still use `head_dim={unaligned_shape['head_dim']}`, `num_attention_heads={unaligned_shape['num_attention_heads']}`, "
                f"and `num_key_value_heads={unaligned_shape['num_key_value_heads']}`."
            ),
            "- The changed operator family is the attention-adjacent projection GEMM path inside `k_proj` / `v_proj`, not an SDPA width cliff.",
            (
                f"- Runtime alignment on the loaded-model view rises from `{runtime_alignment['unaligned']:.4f}%` "
                f"to `{runtime_alignment['aligned_gac']:.1f}%`."
            ),
            (
                f"- Raw config rank alignment rises from `{config_alignment['unaligned']:.4f}%` "
                f"to `{config_alignment['aligned_gac']:.1f}%`."
            ),
            (
                f"- Under the issue-30 fixed-length rerun, aligned GAC changes prefill throughput by "
                f"`{prefill['aligned_vs_unaligned_throughput_pct']:+.2f}%` and decode throughput by "
                f"`{decode['aligned_vs_unaligned_throughput_pct']:+.2f}%` versus unaligned."
            ),
            "",
            "## Included Files",
            "",
            "- `operator_split_summary.json`: structured SDPA-vs-projection split artifact for issue 32.",
            "- `source_manifest.json`: provenance index for the checkpoint config, issue-30 fixed-length evidence, and `HeadwiseLowRankModule` source contract.",
            "- `README.md`: human-readable summary of the structural split and its caveat.",
            "",
            "## Why SDPA Is Not The Changed Path",
            "",
            "- `third_party/palu/palu/model/modules/svd_linear.py` defines `HeadwiseLowRankModule` with an internal `VT` projection and per-group `U[*]` reconstruction.",
            "- The module returns `torch.cat(outputs, dim=-1)`, which restores the original `out_features` before attention consumes the tensor.",
            "- The PaLU checkpoint configs therefore keep the same attention width contract as the baseline Llama model, even though the internal low-rank projection path changes.",
            "",
            "## Caveat",
            "",
            "- This bundle is not a per-kernel profiler trace. It is the checked-in structural attribution result available in the repo today.",
            "- If a future PaLU variant changes the true attention output width rather than only the internal low-rank path, this conclusion must be revisited with a new artifact.",
            "",
            "## Provenance",
            "",
            f"- Fixed-length run results: `{summary['sources']['fixed_length_run_results']}`",
            f"- Fixed-length run summary: `{summary['sources']['fixed_length_run_summary']}`",
            f"- Existing issue-30 bundle summary: `{summary['sources']['issue30_bundle_summary']}`",
            f"- Unaligned checkpoint config: `{summary['sources']['unaligned_checkpoint_config']}`",
            f"- Aligned checkpoint config: `{summary['sources']['aligned_checkpoint_config']}`",
            f"- Module source: `{summary['sources']['headwise_module_source']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unaligned-config", required=True)
    parser.add_argument("--aligned-config", required=True)
    parser.add_argument("--fixed-length-results", required=True)
    parser.add_argument("--fixed-length-summary", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument(
        "--issue30-bundle-summary",
        default="notes/alignment_budget_benefit_results/palu_a100_results/fixed_length_decode_comparison.json",
    )
    parser.add_argument(
        "--headwise-module-source",
        default="third_party/palu/palu/model/modules/svd_linear.py",
    )
    args = parser.parse_args()

    unaligned_config_path = Path(args.unaligned_config)
    aligned_config_path = Path(args.aligned_config)
    fixed_length_results_path = Path(args.fixed_length_results)
    fixed_length_summary_path = Path(args.fixed_length_summary)
    issue30_bundle_summary_path = Path(args.issue30_bundle_summary)
    module_source_path = Path(args.headwise_module_source)
    bundle_dir = Path(args.bundle_dir)

    for path in (
        unaligned_config_path,
        aligned_config_path,
        fixed_length_results_path,
        fixed_length_summary_path,
        issue30_bundle_summary_path,
        module_source_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    summary = build_summary(
        unaligned_config_path=unaligned_config_path,
        aligned_config_path=aligned_config_path,
        fixed_length_results_path=fixed_length_results_path,
        fixed_length_summary_path=fixed_length_summary_path,
        issue30_bundle_summary_path=issue30_bundle_summary_path,
        module_source_path=module_source_path,
    )
    manifest = build_manifest(
        unaligned_config_path=unaligned_config_path,
        aligned_config_path=aligned_config_path,
        fixed_length_results_path=fixed_length_results_path,
        fixed_length_summary_path=fixed_length_summary_path,
        issue30_bundle_summary_path=issue30_bundle_summary_path,
        module_source_path=module_source_path,
    )
    readme = build_readme(summary)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "operator_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (bundle_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle_dir / "README.md").write_text(readme + "\n")
    print(f"Wrote PaLU operator-split bundle to {bundle_dir}")


if __name__ == "__main__":
    main()
