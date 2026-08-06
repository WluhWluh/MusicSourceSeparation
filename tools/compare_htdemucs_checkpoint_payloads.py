#!/usr/bin/env python3
"""Prove legacy and safetensors HTDemucs checkpoints are numerically identical."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any


EXPECTED_LEGACY_SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
EXPECTED_SAFETENSORS_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
EXPECTED_METADATA_SHA256 = "72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def normalize(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "_type": "fraction",
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def tensor_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    import torch

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    common = sorted(reference_keys & candidate_keys)
    unequal_tensors = []
    shape_mismatches = []
    dtype_mismatches = []
    unequal_values = 0
    maximum_absolute_error = 0.0
    total_values = 0
    dtype_counts: dict[str, int] = {}
    for key in common:
        left = reference[key].detach().cpu()
        right = candidate[key].detach().cpu()
        total_values += int(left.numel())
        dtype_counts[str(left.dtype)] = dtype_counts.get(str(left.dtype), 0) + 1
        if left.shape != right.shape:
            shape_mismatches.append(key)
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(key)
            continue
        if not torch.equal(left, right):
            unequal_tensors.append(key)
            unequal_values += int(torch.count_nonzero(left != right).item())
            if left.numel():
                maximum_absolute_error = max(
                    maximum_absolute_error,
                    float((left.float() - right.float()).abs().max().item()),
                )
    return {
        "referenceTensorCount": len(reference_keys),
        "candidateTensorCount": len(candidate_keys),
        "commonTensorCount": len(common),
        "totalValueCount": total_values,
        "referenceOnlyKeys": sorted(reference_keys - candidate_keys),
        "candidateOnlyKeys": sorted(candidate_keys - reference_keys),
        "shapeMismatchKeys": shape_mismatches,
        "dtypeMismatchKeys": dtype_mismatches,
        "unequalTensorKeys": unequal_tensors,
        "unequalValueCount": unequal_values,
        "maximumAbsoluteError": maximum_absolute_error,
        "referenceDtypeTensorCounts": dtype_counts,
        "bitwiseEqual": (
            reference_keys == candidate_keys
            and not shape_mismatches
            and not dtype_mismatches
            and not unequal_tensors
        ),
    }


def checked_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Artifact SHA mismatch for {path}: {actual_sha256}")
    return {
        "path": str(path),
        "byteSize": path.stat().st_size,
        "sha256": actual_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demucs-root", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--safetensors", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demucs_root = args.demucs_root.resolve()
    legacy_path = args.legacy.resolve()
    safetensors_path = args.safetensors.resolve()
    metadata_path = args.metadata.resolve()
    sys.path.insert(0, str(demucs_root))

    import torch
    from safetensors.torch import load_file
    from demucs.hf import load_safetensors_model
    from demucs.htdemucs import HTDemucs
    from demucs.states import load_model

    artifacts = {
        "legacy": checked_artifact(legacy_path, EXPECTED_LEGACY_SHA256),
        "safetensors": checked_artifact(
            safetensors_path, EXPECTED_SAFETENSORS_SHA256
        ),
        "metadata": checked_artifact(metadata_path, EXPECTED_METADATA_SHA256),
    }

    with torch.serialization.safe_globals([HTDemucs, Fraction]):
        legacy_package = torch.load(
            legacy_path, map_location="cpu", weights_only=True
        )
    safe_state = load_file(safetensors_path, device="cpu")
    payload_comparison = tensor_comparison(legacy_package["state"], safe_state)

    legacy_model = load_model(legacy_package).cpu().float().eval()
    safe_model = load_safetensors_model(safetensors_path).cpu().float().eval()
    model_comparison = tensor_comparison(
        legacy_model.state_dict(), safe_model.state_dict()
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    legacy_config = {
        "klass": normalize(legacy_package["klass"]),
        "args": normalize(legacy_package["args"]),
        "kwargs": normalize(legacy_package["kwargs"]),
    }
    safe_config = {
        "klass": metadata["klass"],
        "args": json.loads(metadata["args"]),
        "kwargs": json.loads(metadata["kwargs"]),
    }
    config_fields = {
        key: legacy_config[key] == safe_config[key]
        for key in ("klass", "args", "kwargs")
    }
    training_args_equal = normalize(legacy_package.get("training_args")) == metadata.get(
        "training_args"
    )
    metrics_equal = normalize(legacy_package.get("metrics")) == metadata.get("metrics")

    report = {
        "schemaVersion": 1,
        "status": "complete",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safeLoad": (
                "torch.load(weights_only=True) with only HTDemucs and Fraction allowlisted"
            ),
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": artifacts,
        "serializedState": payload_comparison,
        "instantiatedFloat32ModelState": model_comparison,
        "configuration": {
            "fieldEquality": config_fields,
            "trainingArgsEqual": training_args_equal,
            "metricsEqual": metrics_equal,
            "sourceOrder": safe_config["kwargs"]["sources"],
            "segment": safe_config["kwargs"]["segment"],
            "kwargCount": len(safe_config["kwargs"]),
        },
        "conclusion": {
            "checkpointPayloadsNumericallyIdentical": payload_comparison[
                "bitwiseEqual"
            ],
            "instantiatedModelsNumericallyIdentical": model_comparison[
                "bitwiseEqual"
            ],
            "configurationIdentical": all(config_fields.values()),
            "legacyVsSafetensorsCanExplainOnnxResidual": False,
        },
    }
    if not all(
        (
            payload_comparison["bitwiseEqual"],
            model_comparison["bitwiseEqual"],
            all(config_fields.values()),
            training_args_equal,
            metrics_equal,
        )
    ):
        raise ValueError("Checkpoint equivalence audit failed")
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": "complete",
        "output": str(args.output.resolve()),
        "tensorCount": payload_comparison["referenceTensorCount"],
        "valueCount": payload_comparison["totalValueCount"],
        "bitwiseEqual": payload_comparison["bitwiseEqual"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
