#!/usr/bin/env python3
"""Run the audited HTDemucs-6s guitar fine-tune with the canonical host contract.

Windowing, normalization, PCM encoding, and streaming OLA are imported from
the official reference runner. Only model loading and provenance are variant-
specific; the fine-tune is never treated as an official parity reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import run_htdemucs_canonical_torch as base


CHECKPOINT_SHA256 = "4fde369e41582ba5c2759b6ab926a44af467c64d4566bf914374ab267b19260e"
CHECKPOINT_BYTES = 329_654_071
HF_REPOSITORY = "adityalakhani/htdemucs-6s-guitar-ft"
HF_REVISION = "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad"
CANDIDATE_ID = "htdemucs_6s_guitar_ft_host_fp32_v1_0_0"
EXPECTED_ARTIFACT_SHA256 = "e83f1e6ae8aaef5a177beaf6c6ff6bc3864c244dca123af7398eb1016a24fc6a"
EXPECTED_ARTIFACT_BYTES = 109_716_096
EXPECTED_MANIFEST_SHA256 = "b584473224a5ec8991eecda6f2c2fadf43eee27abce0a9f782e0442a9cc66930"
STEM_ORDER = base.STEM_ORDER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_candidate(weight: Path, manifest_path: Path) -> dict[str, Any]:
    if not weight.is_file() or weight.stat().st_size != EXPECTED_ARTIFACT_BYTES:
        raise ValueError("guitar-ft safetensors size mismatch")
    weight_sha = sha256_file(weight)
    if weight_sha != EXPECTED_ARTIFACT_SHA256:
        raise ValueError(f"guitar-ft safetensors SHA mismatch: {weight_sha}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"guitar-ft manifest SHA mismatch: {manifest_sha}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("status") != "complete" or manifest.get("candidateId") != CANDIDATE_ID:
        raise ValueError("guitar-ft manifest is not the pinned complete candidate")
    if manifest.get("artifact", {}).get("sha256") != weight_sha:
        raise ValueError("guitar-ft manifest artifact SHA does not match weight")
    checkpoint = manifest.get("source", {}).get("checkpoint", {})
    if checkpoint.get("sha256") != CHECKPOINT_SHA256 or checkpoint.get("byteSize") != CHECKPOINT_BYTES:
        raise ValueError("guitar-ft manifest checkpoint identity mismatch")
    audit = manifest.get("architectureAudit", {})
    if any(audit.get(key) for key in (
        "referenceOnlyKeys", "candidateOnlyKeys", "shapeMismatchKeys", "dtypeMismatchKeys"
    )):
        raise ValueError("guitar-ft manifest architecture audit is not exact")
    return {
        "manifest": manifest,
        "manifestPath": str(manifest_path),
        "manifestSha256": manifest_sha,
        "weightPath": str(weight),
        "weightSha256": weight_sha,
    }


def load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    weight = args.weights.resolve()
    manifest_path = args.manifest.resolve()
    identity = verify_candidate(weight, manifest_path)
    demucs_root = args.demucs_root.resolve()
    revision = base.git_revision(demucs_root)
    if revision != base.DEMUCS_REVISION:
        raise ValueError(f"Demucs checkout is {revision}, expected {base.DEMUCS_REVISION}")

    sys.path.insert(0, str(demucs_root))
    import torch

    from demucs.hf import load_safetensors_model
    from export_htdemucs_litert_candidate import install_deterministic_pos_embedding

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = load_safetensors_model(weight).cpu().float().eval()
    if tuple(model.sources) != STEM_ORDER:
        raise ValueError(f"Unexpected guitar-ft stem order: {tuple(model.sources)}")
    if model.samplerate != base.SAMPLE_RATE or model.segment != base.Fraction(39, 5):
        raise ValueError("Unexpected guitar-ft workload contract")
    if not model.use_train_segment:
        raise ValueError("guitar-ft model must retain fixed training segment")
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("guitar-ft model contains non-finite parameters")
    install_deterministic_pos_embedding(model)
    return torch, model, {
        "variantId": CANDIDATE_ID,
        "diagnosticOnly": True,
        "researchOnly": True,
        "fineTuneCheckpoint": identity["manifest"]["source"]["checkpoint"],
        "safetensorsArtifact": identity["manifest"]["artifact"],
        "candidateManifest": {
            "path": identity["manifestPath"],
            "sha256": identity["manifestSha256"],
            "candidateId": CANDIDATE_ID,
        },
        "baseArchitecture": {
            "family": "HTDemucs-6s",
            "officialWeight": identity["manifest"]["officialArchitectureReference"],
            "demucsRevision": revision,
            "stemOrder": list(STEM_ORDER),
            "sampleRate": base.SAMPLE_RATE,
            "segmentSeconds": 7.8,
        },
        "deterministicPositionEmbedding": "sin-shift-zero",
        "weightMutation": "third-party-full-model-guitar-domain-fine-tune",
    }


def validate_fixture(torch: Any, model: Any, fixture_root: Path) -> dict[str, Any]:
    input_path = fixture_root / "waveform_input.f32le.raw"
    golden_path = fixture_root / "combined_golden.f32le.raw"
    waveform = np.fromfile(input_path, dtype="<f4").reshape(base.CHANNELS, base.WINDOW_SAMPLES)
    golden = np.fromfile(golden_path, dtype="<f4").reshape(
        1, base.STEM_COUNT, base.CHANNELS, base.WINDOW_SAMPLES
    )
    result = base.timed(lambda: base.infer(torch, model, waveform))
    aggregate = base.metric(golden, result.value)
    per_stem = {
        stem: base.metric(golden[:, index], result.value[:, index])
        for index, stem in enumerate(STEM_ORDER)
    }
    return {
        "inputSha256": base.sha256_file(input_path),
        "officialGoldenSha256": base.sha256_file(golden_path),
        "timing": result.timing.evidence(),
        "aggregate": aggregate,
        "perStem": per_stem,
        "interpretation": (
            "This is an official-weight fixture used as a delta diagnostic. "
            "It is not a parity gate for the fine-tuned model."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--duration-seconds")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demucs-root", type=Path, default=base.DEMUCS_ROOT)
    parser.add_argument("--canonical-manifest", type=Path, default=base.CANONICAL_MANIFEST)
    parser.add_argument("--fixture-root", type=Path, default=base.FIXTURE_ROOT)
    parser.add_argument("--skip-fixture-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    # The reference runner owns the exact OLA/window implementation. Its model
    # and fixture hooks are replaced before entering the run.
    base.load_model = load_model
    base.validate_fixture = validate_fixture
    report = base.run(args)
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    old_tool = report.get("tool")
    report["runner"] = "run_htdemucs_guitar_ft_torch.py"
    report["scope"] = "diagnostic-guitar-ft-safetensors-torch-fp32-reference"
    report["tool"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "baseRunner": old_tool,
    }
    report["variant"] = {
        "candidateId": CANDIDATE_ID,
        "diagnosticOnly": True,
        "researchOnly": True,
        "qualityInterpretation": (
            "Compared with the official model for regression and listening only; "
            "no ground-truth SDR is inferred from this corpus."
        ),
    }
    base.write_json_atomic(report_path, report)
    if args.report is not None:
        base.write_json_atomic(args.report.resolve(), report)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report["status"],
        "outputDir": str(args.output_dir.resolve()),
        "report": str((args.output_dir / "report.json").resolve()),
        "windowCount": report["contract"]["windowCount"],
        "separationWallMs": report["separationWallMs"],
        "separationRealtimeFactor": report["separationRealtimeFactor"],
        "totalWallMs": report["totalWallMs"],
        "totalRealtimeFactor": report["totalRealtimeFactor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
