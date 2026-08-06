#!/usr/bin/env python3
"""Export the audited HTDemucs-6s guitar fine-tune with the canonical ABI.

This is a deliberately separate entry point. It reuses the frozen neural-core
conversion implementation, but substitutes only the pinned guitar-ft weight
identity and model ID. The official exporter and official generated candidate
are never modified by this experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import export_htdemucs_litert_candidate as base


MODEL_ID = "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0"
PROFILE_NAME = "canonical_7p8s"
WEIGHT_BYTES = 109_716_096
WEIGHT_SHA256 = "e83f1e6ae8aaef5a177beaf6c6ff6bc3864c244dca123af7398eb1016a24fc6a"
CHECKPOINT_BYTES = 329_654_071
CHECKPOINT_SHA256 = "4fde369e41582ba5c2759b6ab926a44af467c64d4566bf914374ab267b19260e"
CHECKPOINT_REVISION = "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad"
CHECKPOINT_REPOSITORY = "adityalakhani/htdemucs-6s-guitar-ft"
BASE_WEIGHT = Path("models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors")
BASE_WEIGHT_BYTES = 54_885_744
BASE_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"


def identity(path: Path, *, format: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    return {
        "fileName": path.name,
        "byteSize": path.stat().st_size,
        "sha256": base.sha256(path),
        "format": format,
    }


def verify_identity(path: Path, byte_size: int, sha256: str, label: str) -> None:
    try:
        base.verify_file(path, byte_size, sha256)
    except RuntimeError as error:
        raise RuntimeError(f"{label} identity check failed: {error}") from error


def parse_variant_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fine-tune-checkpoint", type=Path, required=True)
    parser.add_argument("--base-architecture-weight", type=Path, default=BASE_WEIGHT)
    parser.add_argument("--report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    return args, ["--report", str(args.report), *remaining]


def configure_base() -> None:
    base.PROFILE_SPECS = {
        PROFILE_NAME: {
            "modelId": MODEL_ID,
            "sampleCount": 343_980,
        },
    }
    base.EXPECTED_WEIGHT_BYTES = WEIGHT_BYTES
    base.EXPECTED_WEIGHT_SHA256 = WEIGHT_SHA256


def annotate_report(report_path: Path, args: argparse.Namespace) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = report.setdefault("provenance", {})
    fine_tune_weight = provenance.pop("canonicalWeight")
    fine_tune_weight["role"] = "guitar-domain-fine-tuned-full-model-weight"
    provenance["fineTuneWeight"] = fine_tune_weight
    provenance["fineTuneCheckpoint"] = {
        **identity(args.fine_tune_checkpoint, format="pytorch-training-checkpoint"),
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "loadPolicy": "weights_only=True extraction; model_state_dict only",
    }
    provenance["baseArchitectureWeight"] = {
        **identity(args.base_architecture_weight, format="safetensors"),
        "role": "official-architecture-and-source-order-reference-only",
    }
    provenance["variantExporter"] = {
        **identity(Path(__file__), format="python-source"),
        "baseExporter": provenance.get("exportScript"),
    }
    report["variant"] = {
        "candidateFamily": "htdemucs-6s-guitar-ft",
        "weightMutation": "third-party-full-model-guitar-domain-fine-tune",
        "diagnosticOnly": True,
        "researchOnly": True,
        "licenseDisposition": "research-only-pending-independent-training-data-review",
        "parityReference": "same guitar-ft Torch model",
        "notComparedForAdmissionAgainst": "official HTDemucs-6s output",
    }
    report["scope"] = "diagnostic-research-only-host-candidate"
    report["qualityGate"]["comparisonReference"] = "guitar-ft Torch FP32"
    report["conversionRecipe"]["modelVariant"] = "guitar-ft"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def main() -> int:
    variant_args, base_args = parse_variant_args()
    verify_identity(
        variant_args.fine_tune_checkpoint,
        CHECKPOINT_BYTES,
        CHECKPOINT_SHA256,
        "Fine-tune checkpoint",
    )
    verify_identity(
        variant_args.base_architecture_weight,
        BASE_WEIGHT_BYTES,
        BASE_WEIGHT_SHA256,
        "Official architecture weight",
    )
    configure_base()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *base_args]
        status = base.main()
    finally:
        sys.argv = original_argv
    if variant_args.report.is_file():
        annotate_report(variant_args.report, variant_args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
