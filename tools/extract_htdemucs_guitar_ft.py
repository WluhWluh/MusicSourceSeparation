#!/usr/bin/env python3
"""Safely extract and audit the published HTDemucs guitar fine-tune.

The source checkpoint is a training ``.pt`` payload.  This tool never loads
serialized Python objects: it uses ``weights_only=True``, extracts only the
tensor state, and writes a self-contained safetensors artifact with the
official HTDemucs-6s architecture metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any


CHECKPOINT_SHA256 = "4fde369e41582ba5c2759b6ab926a44af467c64d4566bf914374ab267b19260e"
CHECKPOINT_BYTES = 329_654_071
HF_REPOSITORY = "adityalakhani/htdemucs-6s-guitar-ft"
HF_REVISION = "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad"
OFFICIAL_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
OFFICIAL_WEIGHT_BYTES = 54_885_744
STEM_ORDER = ("drums", "bass", "other", "vocals", "guitar", "piano")
EXPECTED_TENSOR_COUNT = 525
EXPECTED_VALUE_COUNT = 27_414_996


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(
            f"Artifact identity mismatch for {path}: "
            f"bytes={actual_bytes} sha256={actual_sha256}"
        )
    return {
        "path": str(path.resolve()),
        "fileName": path.name,
        "byteSize": actual_bytes,
        "sha256": actual_sha256,
    }


def metric_state(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    import torch

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    common = reference_keys & candidate_keys
    shape_mismatches = sorted(
        key for key in common if tuple(reference[key].shape) != tuple(candidate[key].shape)
    )
    dtype_mismatches = sorted(
        key for key in common if reference[key].dtype != candidate[key].dtype
    )
    changed = []
    changed_values = 0
    maximum_absolute_error = 0.0
    squared_error = 0.0
    compared_values = 0
    for key in sorted(common - set(shape_mismatches) - set(dtype_mismatches)):
        left = reference[key].detach().cpu()
        right = candidate[key].detach().cpu()
        compared_values += int(left.numel())
        delta = (left.float() - right.float()).abs()
        if not torch.equal(left, right):
            changed.append(key)
            changed_values += int(torch.count_nonzero(delta).item())
            if delta.numel():
                maximum_absolute_error = max(maximum_absolute_error, float(delta.max().item()))
            squared_error += float((left.double() - right.double()).square().sum().item())
    return {
        "referenceTensorCount": len(reference_keys),
        "candidateTensorCount": len(candidate_keys),
        "referenceOnlyKeys": sorted(reference_keys - candidate_keys),
        "candidateOnlyKeys": sorted(candidate_keys - reference_keys),
        "shapeMismatchKeys": shape_mismatches,
        "dtypeMismatchKeys": dtype_mismatches,
        "changedTensorCount": len(changed),
        "changedTensorKeys": changed,
        "changedValueCount": changed_values,
        "comparedValueCount": compared_values,
        "maximumAbsoluteError": maximum_absolute_error,
        "rootMeanSquareDelta": (
            (squared_error / compared_values) ** 0.5 if compared_values else 0.0
        ),
        "bitwiseEqual": not (
            reference_keys - candidate_keys
            or candidate_keys - reference_keys
            or shape_mismatches
            or dtype_mismatches
            or changed
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    official_weights = args.official_weights.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()

    checkpoint_identity = checked_file(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256)
    official_identity = checked_file(
        official_weights, OFFICIAL_WEIGHT_BYTES, OFFICIAL_WEIGHT_SHA256
    )

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    demucs_root = Path(__file__).resolve().parent.parent / ".tmp" / "demucs-adefossez"
    sys.path.insert(0, str(demucs_root))
    from demucs.hf import load_safetensors_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint does not contain model_state_dict")
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("model_state_dict is not a non-empty tensor dictionary")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("model_state_dict must contain only string -> tensor entries")
    if len(state) != EXPECTED_TENSOR_COUNT or sum(value.numel() for value in state.values()) != EXPECTED_VALUE_COUNT:
        raise ValueError("Unexpected guitar fine-tune tensor count or value count")

    official_model = load_safetensors_model(official_weights).cpu().float().eval()
    if tuple(official_model.sources) != STEM_ORDER:
        raise ValueError(f"Unexpected official stem order: {tuple(official_model.sources)}")
    if official_model.samplerate != 44_100 or str(official_model.segment) != "39/5":
        raise ValueError("Unexpected official HTDemucs-6s workload")
    reference_state = official_model.state_dict()
    comparison = metric_state(reference_state, state)
    if (
        comparison["referenceOnlyKeys"]
        or comparison["candidateOnlyKeys"]
        or comparison["shapeMismatchKeys"]
        or comparison["dtypeMismatchKeys"]
    ):
        raise ValueError(f"guitar fine-tune architecture mismatch: {comparison}")
    official_model.load_state_dict(state, strict=True)

    with safe_open(str(official_weights), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    required_metadata = {"klass", "args", "kwargs"}
    if not required_metadata.issubset(metadata):
        raise ValueError("Official safetensors metadata is missing architecture fields")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_name(output.name + ".partial")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        str(partial_output),
        metadata=metadata,
    )
    partial_output.replace(output)
    output_identity = {
        "path": str(output),
        "fileName": output.name,
        "byteSize": output.stat().st_size,
        "sha256": sha256_file(output),
        "format": "safetensors",
        "dtype": "float32",
    }

    manifest = {
        "schemaVersion": 1,
        "status": "complete",
        "candidateId": "htdemucs_6s_guitar_ft_host_fp32_v1_0_0",
        "diagnosticOnly": True,
        "researchOnly": True,
        "source": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "checkpoint": checkpoint_identity,
            "checkpointPayload": {
                "topLevelKeys": sorted(payload),
                "epoch": payload.get("epoch"),
                "valLoss": payload.get("val_loss"),
                "valSdrDb": payload.get("val_sdr_db"),
                "loader": "torch.load(weights_only=True, map_location=cpu)",
            },
        },
        "officialArchitectureReference": official_identity,
        "artifact": output_identity,
        "modelSemantics": {
            "family": "HTDemucs-6s",
            "stemOrder": list(STEM_ORDER),
            "sampleRate": 44_100,
            "segmentSeconds": 7.8,
            "stateTensorCount": len(state),
            "stateValueCount": sum(value.numel() for value in state.values()),
        },
        "architectureAudit": comparison,
        "licenseEvidence": {
            "modelCard": "Apache-2.0 (publisher declaration; repository has no LICENSE file)",
            "trainingDataset": "MoisesDB v0.1; research/non-commercial terms require separate review",
            "modelCardMetricsVerified": False,
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    partial_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_manifest.replace(manifest_path)
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "manifest": str(manifest_path),
        "outputSha256": output_identity["sha256"],
        "outputBytes": output_identity["byteSize"],
        "architectureBitwiseEqual": comparison["bitwiseEqual"],
        "changedTensorCount": comparison["changedTensorCount"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
