#!/usr/bin/env python3
"""Collect fixed-length decode results into a normalized repo-tracked summary."""

import argparse
import json
from pathlib import Path


METHOD_SPECS = {
    "asvd": {
        "summary_method": "asvd",
        "variants": {
            "baseline": ("baseline", "asvd_baseline_eval.json"),
            "unaligned": ("unaligned", "asvd_unaligned_eval.json"),
            "aligned_gac": ("aligned", "asvd_aligned_eval.json"),
        },
    },
    "llmpruner": {
        "summary_method": "llm_pruner",
        "variants": {
            "baseline": ("baseline", "llmpruner_baseline_eval.json"),
            "unaligned": ("pruned", "llmpruner_pruned_eval.json"),
            "aligned_gac": ("pruned_r8", "llmpruner_pruned_r8_eval.json"),
        },
    },
}


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def display_path(path: Path) -> str:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHOD_SPECS))
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = METHOD_SPECS[args.method]
    loaded = {}
    for normalized_name, (raw_variant, filename) in spec["variants"].items():
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw result for {normalized_name}: {path}")
        loaded[normalized_name] = {
            "raw_variant": raw_variant,
            "path": path,
            "payload": load_json(path),
        }

    baseline_decode = loaded["baseline"]["payload"]["decode_latency"]
    measurement_contract = {
        "prompt_len": baseline_decode["prompt_len"],
        "requested_new_tokens": baseline_decode["requested_new_tokens"],
        "decode_length_mode": baseline_decode["decode_length_mode"],
        "generation_guard": baseline_decode["generation_guard"],
        "decode_warmup": baseline_decode["warmup_count"],
        "decode_measure": baseline_decode["measure_count"],
    }

    decode_comparison = {}
    normalized_variants = {}
    for normalized_name, entry in loaded.items():
        payload = entry["payload"]["decode_latency"]
        normalized_variants[normalized_name] = {"raw_variant": entry["raw_variant"]}
        decode_comparison[normalized_name] = {
            "status": "measured",
            "source": display_path(entry["path"]),
            "total_mean_ms": payload["total_mean_ms"],
            "total_std_ms": payload["total_std_ms"],
            "per_token_ms": payload["per_token_ms"],
            "tokens_per_sec": payload["tokens_per_sec"],
            "actual_new_tokens": payload["actual_new_tokens"],
        }

    unaligned_tps = decode_comparison["unaligned"]["tokens_per_sec"]
    aligned_tps = decode_comparison["aligned_gac"]["tokens_per_sec"]
    speedup_pct = ((aligned_tps / unaligned_tps) - 1.0) * 100.0
    aligned_beats = aligned_tps > unaligned_tps

    summary = {
        "method": spec["summary_method"],
        "normalized_variants": normalized_variants,
        "measurement_contract": measurement_contract,
        "decode_comparison": decode_comparison,
        "conclusion": {
            "aligned_beats_unaligned": aligned_beats,
            "speedup_vs_unaligned_pct": speedup_pct,
            "summary": (
                f"Under the fixed-length decode contract, aligned_gac reaches "
                f"{aligned_tps:.2f} tok/s versus {unaligned_tps:.2f} tok/s for unaligned "
                f"({speedup_pct:+.2f}%)."
            ),
        },
        "sources": [
            display_path(entry["path"])
            for entry in loaded.values()
        ],
    }

    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote fixed-length summary to {output_path}")


if __name__ == "__main__":
    main()
