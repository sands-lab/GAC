#!/usr/bin/env python3
"""Publish token-eviction operator-attribution runs into a tracked bundle."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


RUN_LAYOUT: Dict[str, Dict[str, str]] = {
    "prefill": {
        "spec_name": "TOKENEVICTION_prefill_gemm_m_real_shapes",
        "axis_key": "M",
        "publish_subdir": "prefill_gemm_m",
    },
    "decode": {
        "spec_name": "TOKENEVICTION_decode_sdpa_context_real_shapes",
        "axis_key": "seq_len",
        "publish_subdir": "decode_sdpa_context",
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


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
        source = run_dir / f"{name}.json"
        if not source.exists():
            raise FileNotFoundError(
                "Missing required run artifact: {path}".format(path=source)
            )
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        published_files[name] = display_path(destination)
    return published_files


def pct_delta(current: Any, baseline: Any) -> Any:
    if current is None or baseline in (None, 0):
        return None
    return ((float(current) - float(baseline)) / float(baseline)) * 100.0


def normalize_measurement(entry: Dict[str, Any], axis_key: str) -> Dict[str, Any]:
    shape = entry.get("shape", {})
    timing = entry.get("timing", {})
    derived = entry.get("derived", {})
    value = int(shape[axis_key])
    normalized: Dict[str, Any] = {
        "value": value,
        "multiple_of_8": value % 8 == 0,
        "shape": shape,
        "latency_mean_ms": timing.get("mean"),
        "latency_std_ms": timing.get("std"),
    }
    if derived:
        normalized["tflops_mean"] = derived.get("tflops_mean")
        normalized["bandwidth_gbs_mean"] = derived.get("bandwidth_gbs_mean")
    return normalized


def build_shape_context(checkpoints: List[Dict[str, Any]], axis_key: str) -> Dict[str, Any]:
    if not checkpoints:
        return {}
    shape_context: Dict[str, Any] = {}
    shape_keys = sorted(checkpoints[0]["shape"].keys())
    for key in shape_keys:
        if key == axis_key:
            continue
        values = sorted({item["shape"][key] for item in checkpoints})
        shape_context[key] = values[0] if len(values) == 1 else values
    return shape_context


def build_boundary_triplets(
    measurements_by_value: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    triplets: List[Dict[str, Any]] = []
    for value in sorted(measurements_by_value):
        before = measurements_by_value.get(value - 1)
        at_boundary = measurements_by_value.get(value)
        after = measurements_by_value.get(value + 1)
        if before is None or at_boundary is None or after is None:
            continue

        triplet_items = [before, at_boundary, after]
        fastest = min(triplet_items, key=lambda item: float(item["latency_mean_ms"]))

        triplets.append(
            {
                "boundary_value": value,
                "before": before,
                "at_boundary": at_boundary,
                "after": after,
                "latency_delta_pct_boundary_vs_before": pct_delta(
                    at_boundary["latency_mean_ms"], before["latency_mean_ms"]
                ),
                "latency_delta_pct_after_vs_boundary": pct_delta(
                    after["latency_mean_ms"], at_boundary["latency_mean_ms"]
                ),
                "fastest_value_in_triplet": fastest["value"],
            }
        )
    return triplets


def summarize_measurements(
    raw_payload: Dict[str, Any], axis_key: str
) -> Dict[str, Dict[str, Any]]:
    measurements_by_dtype: Dict[str, Dict[str, Any]] = {}
    successful = [
        entry for entry in raw_payload.get("measurements", []) if "error" not in entry
    ]

    for entry in successful:
        dtype = entry["dtype"]
        bucket = measurements_by_dtype.setdefault(dtype, {"checkpoints": []})
        bucket["checkpoints"].append(normalize_measurement(entry, axis_key))

    for bucket in measurements_by_dtype.values():
        bucket["checkpoints"].sort(key=lambda item: item["value"])
        measurements_by_value = {
            item["value"]: item for item in bucket["checkpoints"]
        }
        bucket["boundary_triplets"] = build_boundary_triplets(measurements_by_value)
        if bucket["checkpoints"]:
            bucket["fastest_checkpoint"] = min(
                bucket["checkpoints"], key=lambda item: float(item["latency_mean_ms"])
            )
            bucket["slowest_checkpoint"] = max(
                bucket["checkpoints"], key=lambda item: float(item["latency_mean_ms"])
            )

    return measurements_by_dtype


def summarize_run(
    raw_payload: Dict[str, Any], spec_payload: Dict[str, Any], axis_key: str
) -> Dict[str, Any]:
    provenance = spec_payload.get("provenance", {})
    successful = [
        entry for entry in raw_payload.get("measurements", []) if "error" not in entry
    ]
    normalized = [normalize_measurement(entry, axis_key) for entry in successful]

    return {
        "method": provenance.get("method"),
        "stage": provenance.get("stage"),
        "operator_family": provenance.get("operator_family"),
        "changed_axis": provenance.get("changed_axis"),
        "sources": provenance.get("sources", []),
        "rationale": provenance.get("rationale"),
        "raw_experiment": raw_payload.get("experiment"),
        "raw_metadata": raw_payload.get("metadata", {}),
        "shape_context": build_shape_context(normalized, axis_key),
        "measurements_by_dtype": summarize_measurements(raw_payload, axis_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--prefill-run-dir", required=True)
    parser.add_argument("--decode-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "prefill": Path(args.prefill_run_dir),
        "decode": Path(args.decode_run_dir),
    }

    specs = load_spec(spec_path)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    summary_payload: Dict[str, Any] = {
        "focus": "token_eviction_operator_attribution",
        "generated_at": generated_at,
        "spec_path": display_path(spec_path),
        "experiments": {},
    }
    manifest_payload: Dict[str, Any] = {
        "focus": "token_eviction_operator_attribution",
        "generated_at": generated_at,
        "bundle_script": "scripts/publish_token_eviction_operator_attribution_bundle.py",
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
            axis_key=layout["axis_key"],
        )
        manifest_payload["published_runs"][run_key] = {
            "spec_name": spec_name,
            "source_run_dir": display_path(run_dir),
            "published_files": published_files,
        }

    write_json(
        output_dir / "token_eviction_operator_attribution_summary.json",
        summary_payload,
    )
    write_json(output_dir / "source_manifest.json", manifest_payload)
    print(
        "Wrote token eviction operator attribution bundle to {path}".format(
            path=display_path(output_dir)
        )
    )


if __name__ == "__main__":
    main()
