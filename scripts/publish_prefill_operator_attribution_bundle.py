#!/usr/bin/env python3
"""Publish real-shape prefill operator-attribution runs into a tracked bundle."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


RUN_LAYOUT: Dict[str, Dict[str, str]] = {
    "asvd": {
        "spec_name": "ASVD_prefill_gemm_k_real_shapes",
        "axis": "K",
        "mode": "checkpoint_sweep",
        "publish_subdir": "asvd",
    },
    "llmpruner_gateup": {
        "spec_name": "LLMPRUNER_prefill_gateup_n_real_shapes",
        "axis": "N",
        "mode": "paired_by_m",
        "publish_subdir": "llmpruner_gateup",
    },
    "llmpruner_down": {
        "spec_name": "LLMPRUNER_prefill_down_k_real_shapes",
        "axis": "K",
        "mode": "paired",
        "publish_subdir": "llmpruner_down",
    },
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


def load_spec(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        payload = yaml.safe_load(handle)
    experiments = payload.get("experiments", {})
    missing = [
        layout["spec_name"]
        for layout in RUN_LAYOUT.values()
        if layout["spec_name"] not in experiments
    ]
    if missing:
        raise ValueError(
            "Missing expected experiment definitions in spec: "
            + ", ".join(sorted(missing))
        )
    return experiments


def copy_run_files(run_dir: Path, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    published_files: Dict[str, str] = {}
    for name in ("raw", "config", "summary", "env"):
        source = run_dir / "{name}.json".format(name=name)
        if not source.exists():
            if name == "raw":
                raise FileNotFoundError("Missing raw run payload: {path}".format(path=source))
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        published_files[name] = display_path(destination)
    return published_files


def normalize_measurement(entry: Dict[str, Any], axis: str) -> Dict[str, Any]:
    shape = entry.get("shape", {})
    timing = entry.get("timing", {})
    derived = entry.get("derived", {})
    value = int(shape[axis])
    normalized = {
        "value": value,
        "aligned_to_8": value % 8 == 0,
        "shape": shape,
        "latency_mean_ms": timing.get("mean"),
        "latency_std_ms": timing.get("std"),
        "tflops_mean": derived.get("tflops_mean"),
        "bandwidth_gbs_mean": derived.get("bandwidth_gbs_mean"),
    }
    return normalized


def pair_adjacent_values(values: List[int]) -> List[Tuple[int, int]]:
    if len(values) % 2 != 0:
        raise ValueError("Expected an even number of paired checkpoint values")
    pairs: List[Tuple[int, int]] = []
    for index in range(0, len(values), 2):
        pairs.append((int(values[index]), int(values[index + 1])))
    return pairs


def build_pair_entry(
    measurements_by_value: Dict[int, Dict[str, Any]],
    unaligned_value: int,
    aligned_value: int,
) -> Dict[str, Any]:
    if unaligned_value not in measurements_by_value:
        raise ValueError(
            "Missing unaligned checkpoint measurement for value {value}".format(
                value=unaligned_value
            )
        )
    if aligned_value not in measurements_by_value:
        raise ValueError(
            "Missing aligned checkpoint measurement for value {value}".format(
                value=aligned_value
            )
        )

    unaligned = measurements_by_value[unaligned_value]
    aligned = measurements_by_value[aligned_value]
    unaligned_latency = unaligned["latency_mean_ms"]
    aligned_latency = aligned["latency_mean_ms"]
    unaligned_tflops = unaligned["tflops_mean"]
    aligned_tflops = aligned["tflops_mean"]

    latency_improvement_pct = None
    if unaligned_latency:
        latency_improvement_pct = (
            (float(unaligned_latency) - float(aligned_latency)) / float(unaligned_latency)
        ) * 100.0

    tflops_gain_pct = None
    if unaligned_tflops:
        tflops_gain_pct = (
            (float(aligned_tflops) / float(unaligned_tflops)) - 1.0
        ) * 100.0

    return {
        "unaligned": unaligned,
        "aligned": aligned,
        "latency_improvement_pct": latency_improvement_pct,
        "tflops_gain_pct": tflops_gain_pct,
        "aligned_beats_unaligned": (
            aligned_latency is not None
            and unaligned_latency is not None
            and float(aligned_latency) < float(unaligned_latency)
        ),
    }


def summarize_checkpoint_sweep(
    raw_payload: Dict[str, Any], spec_payload: Dict[str, Any], axis: str
) -> Dict[str, Any]:
    measurements_by_dtype: Dict[str, Dict[str, Any]] = {}
    successful = [
        entry for entry in raw_payload.get("measurements", []) if "error" not in entry
    ]
    for entry in successful:
        dtype = entry["dtype"]
        bucket = measurements_by_dtype.setdefault(dtype, {"checkpoints": []})
        bucket["checkpoints"].append(normalize_measurement(entry, axis))

    for bucket in measurements_by_dtype.values():
        bucket["checkpoints"].sort(key=lambda item: item["value"])
        if bucket["checkpoints"]:
            bucket["fastest_checkpoint"] = min(
                bucket["checkpoints"], key=lambda item: float(item["latency_mean_ms"])
            )
            bucket["slowest_checkpoint"] = max(
                bucket["checkpoints"], key=lambda item: float(item["latency_mean_ms"])
            )

    shape_context = dict(spec_payload.get("shape", {}))
    shape_context.pop(axis, None)
    return {
        "shape_context": shape_context,
        "measurements_by_dtype": measurements_by_dtype,
    }


def summarize_paired_by_m(
    raw_payload: Dict[str, Any], spec_payload: Dict[str, Any], axis: str
) -> Dict[str, Any]:
    measurements_by_dtype: Dict[str, Dict[str, Any]] = {}
    successful = [
        entry for entry in raw_payload.get("measurements", []) if "error" not in entry
    ]
    pair_values = pair_adjacent_values(spec_payload["{axis}_values".format(axis=axis)])

    for entry in successful:
        dtype = entry["dtype"]
        shape = entry["shape"]
        m_value = str(int(shape["M"]))
        bucket = measurements_by_dtype.setdefault(
            dtype,
            {
                "shape_context": {"K": spec_payload["K"]},
                "by_M": {},
            },
        )
        m_bucket = bucket["by_M"].setdefault(m_value, {"_entries": {}})
        normalized = normalize_measurement(entry, axis)
        m_bucket["_entries"][normalized["value"]] = normalized

    for bucket in measurements_by_dtype.values():
        for m_value, m_bucket in bucket["by_M"].items():
            entry_map = m_bucket.pop("_entries")
            m_bucket["pairs"] = [
                build_pair_entry(entry_map, unaligned, aligned)
                for unaligned, aligned in pair_values
            ]

    return {"measurements_by_dtype": measurements_by_dtype}


def summarize_paired(
    raw_payload: Dict[str, Any], spec_payload: Dict[str, Any], axis: str
) -> Dict[str, Any]:
    measurements_by_dtype: Dict[str, Dict[str, Any]] = {}
    successful = [
        entry for entry in raw_payload.get("measurements", []) if "error" not in entry
    ]
    pair_values = pair_adjacent_values(spec_payload["{axis}_values".format(axis=axis)])

    for entry in successful:
        dtype = entry["dtype"]
        bucket = measurements_by_dtype.setdefault(
            dtype,
            {
                "shape_context": {
                    "M": spec_payload["shape"]["M"],
                    "N": spec_payload["shape"]["N"],
                },
                "_entries": {},
            },
        )
        normalized = normalize_measurement(entry, axis)
        bucket["_entries"][normalized["value"]] = normalized

    for bucket in measurements_by_dtype.values():
        entry_map = bucket.pop("_entries")
        bucket["pairs"] = [
            build_pair_entry(entry_map, unaligned, aligned)
            for unaligned, aligned in pair_values
        ]

    return {"measurements_by_dtype": measurements_by_dtype}


def summarize_run(
    raw_payload: Dict[str, Any],
    spec_payload: Dict[str, Any],
    axis: str,
    mode: str,
) -> Dict[str, Any]:
    provenance = spec_payload.get("provenance", {})
    summary = {
        "method": provenance.get("method"),
        "stage": provenance.get("stage"),
        "operator_family": provenance.get("operator_family"),
        "changed_axis": provenance.get("changed_axis"),
        "sources": provenance.get("sources", []),
        "rationale": provenance.get("rationale"),
        "raw_experiment": raw_payload.get("experiment"),
        "raw_metadata": raw_payload.get("metadata", {}),
    }

    if mode == "checkpoint_sweep":
        summary.update(summarize_checkpoint_sweep(raw_payload, spec_payload, axis))
    elif mode == "paired_by_m":
        summary.update(summarize_paired_by_m(raw_payload, spec_payload, axis))
    elif mode == "paired":
        summary.update(summarize_paired(raw_payload, spec_payload, axis))
    else:
        raise ValueError("Unsupported summary mode: {mode}".format(mode=mode))
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--asvd-run-dir", required=True)
    parser.add_argument("--llmpruner-gateup-run-dir", required=True)
    parser.add_argument("--llmpruner-down-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "asvd": Path(args.asvd_run_dir),
        "llmpruner_gateup": Path(args.llmpruner_gateup_run_dir),
        "llmpruner_down": Path(args.llmpruner_down_run_dir),
    }

    specs = load_spec(spec_path)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    summary_payload: Dict[str, Any] = {
        "focus": "prefill_operator_attribution",
        "generated_at": generated_at,
        "spec_path": display_path(spec_path),
        "experiments": {},
    }
    manifest_payload: Dict[str, Any] = {
        "focus": "prefill_operator_attribution",
        "generated_at": generated_at,
        "bundle_script": "scripts/publish_prefill_operator_attribution_bundle.py",
        "spec_path": display_path(spec_path),
        "published_runs": {},
    }

    for run_key, layout in RUN_LAYOUT.items():
        run_dir = run_dirs[run_key]
        raw_payload = load_json(run_dir / "raw.json")
        spec_name = layout["spec_name"]
        spec_payload = specs[spec_name]
        published_files = copy_run_files(
            run_dir, output_dir / layout["publish_subdir"]
        )

        summary_payload["experiments"][spec_name] = summarize_run(
            raw_payload=raw_payload,
            spec_payload=spec_payload,
            axis=layout["axis"],
            mode=layout["mode"],
        )
        manifest_payload["published_runs"][run_key] = {
            "spec_name": spec_name,
            "source_run_dir": display_path(run_dir),
            "published_files": published_files,
        }

    write_json(
        output_dir / "prefill_operator_attribution_summary.json", summary_payload
    )
    write_json(output_dir / "source_manifest.json", manifest_payload)
    print(
        "Wrote prefill operator attribution bundle to {path}".format(
            path=display_path(output_dir)
        )
    )


if __name__ == "__main__":
    main()
