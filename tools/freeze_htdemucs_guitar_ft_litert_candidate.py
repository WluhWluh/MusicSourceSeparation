#!/usr/bin/env python3
"""Freeze and verify the research-only guitar-ft LiteRT host candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import freeze_htdemucs_litert_candidate as base


MODEL_ID = "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0"
SOURCE_ROOT = (
    "models/demucs/candidates/htdemucs-6s-guitar-ft/"
    "163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad"
)
SOURCE_PATHS = {
    "fineTuneWeight": f"{SOURCE_ROOT}/htdemucs_6s_guitar_ft_fp32.safetensors",
    "fineTuneCheckpoint": f"{SOURCE_ROOT}/guitar_htdemucs_6s.pt",
    "baseArchitectureWeight": "models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors",
    "metadata": "models/demucs/official-hf/htdemucs_6s/5c90dfd2.json",
    "bagManifest": f"{SOURCE_ROOT}/htdemucs_6s.yaml",
    "boundarySource": ".tmp/demucs-lite-reference/demucs-for-onnx/demucs/htdemucs.py",
    "referenceExporter": ".tmp/demucs-lite-reference/scripts/convert-pth-to-onnx-chunked.py",
    "exportScript": "tools/export_htdemucs_litert_candidate.py",
    "variantExporter": "tools/export_htdemucs_guitar_ft_litert_candidate.py",
    "requirementsLock": "requirements-demucs-litert-export.txt",
}
OFFICIAL_HOST_MANIFEST = Path(
    "models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/"
    "candidate-manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("models/demucs/generated") / MODEL_ID,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-narrow-host-failure", action="store_true")
    return parser.parse_args()


def configure_base() -> None:
    base.MODEL_ID = MODEL_ID
    base.SOURCE_PATHS = SOURCE_PATHS


def build_manifest(repo_root: Path, candidate_root: Path) -> dict:
    configure_base()
    manifest = base.build_manifest(repo_root, candidate_root)
    report = json.loads(
        (candidate_root / "export-report.json").read_text(encoding="utf-8")
    )
    variant = report.get("variant")
    base.require(isinstance(variant, dict), "Export report lacks guitar-ft variant metadata")
    base.require(variant.get("diagnosticOnly") is True, "diagnosticOnly must be true")
    base.require(variant.get("researchOnly") is True, "researchOnly must be true")
    base.require(
        report.get("qualityGate", {}).get("comparisonReference")
        == "guitar-ft Torch FP32",
        "Host parity must use the same guitar-ft Torch model",
    )
    manifest["scope"] = "diagnostic-research-only-host-candidate"
    manifest["variant"] = variant
    manifest["license"] = {
        "distributionStatus": "research-only",
        "productUseApproved": False,
        "reason": "MoisesDB training-data terms require independent review",
    }
    manifest["hostValidation"]["parityReference"] = "guitar-ft Torch FP32"
    return manifest


def verified_provenance(
    repo_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, local_path in SOURCE_PATHS.items():
        declaration = report["provenance"][key]
        path = base.resolve_declared_source(repo_root, local_path, declaration, key)
        source_identity = {
            **base.identity(repo_root, path, format=declaration.get("format")),
            **{
                name: declaration[name]
                for name in (
                    "role",
                    "repository",
                    "revision",
                    "loadPolicy",
                )
                if name in declaration
            },
        }
        source_identity["localPath"] = local_path
        result[key] = source_identity
    return result


def build_diagnostic_manifest(repo_root: Path, candidate_root: Path) -> dict[str, Any]:
    report_path = candidate_root / "export-report.json"
    inspection_path = candidate_root / "flatbuffer-inspection.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))

    base.require(report["modelId"] == MODEL_ID, "Export report model ID mismatch")
    base.require(report["profileName"] == base.PROFILE_NAME, "Profile mismatch")
    base.require(report["status"] == "failed", "Expected a strict host failure")
    base.require(
        report.get("failure", {}).get("message")
        == "LiteRT output did not pass the host quality gate",
        "Unexpected export failure",
    )
    base.require(
        report["qualityGate"]["comparisonReference"] == "guitar-ft Torch FP32",
        "Parity reference mismatch",
    )
    base.require(
        all(value["accepted"] for value in report["singleWindowGateEvaluation"].values()),
        "Single-window parity did not pass",
    )
    ola = report["canonicalOlaValidation"]
    base.require(ola["torchCoreOlaVsOfficialApplyModel"]["aggregate"]["bitwiseEqual"],
        "Torch neural-core OLA does not match the same guitar-ft model")
    base.require(all(value["accepted"] for value in ola["windowParity"]),
        "A neural-core OLA window failed the aggregate gate")
    base.require(ola["liteRtVsTorchOla"]["full"]["gateEvaluation"]["accepted"],
        "Full OLA parity must pass before bounded diagnostics")
    base.require(ola["liteRtVsTorchOla"]["overlap"]["gateEvaluation"]["accepted"],
        "Overlap parity must pass before bounded diagnostics")
    base.require(not ola["liteRtVsTorchOla"]["eof"]["gateEvaluation"]["accepted"],
        "Diagnostic manifest is only for the frozen EOF gate failure")
    eof = ola["liteRtVsTorchOla"]["eof"]["metrics"]
    base.require(eof["aggregate"]["finite"], "EOF output is non-finite")
    base.require(eof["aggregate"]["maxAbsoluteError"] < 1e-3,
        "EOF error exceeds the absolute waveform ceiling")

    artifact_path = candidate_root / report["liteRtArtifact"]["fileName"]
    base.verify_declared_file(artifact_path, report["liteRtArtifact"], "LiteRT artifact")
    base.require(inspection["artifact"]["sha256"] == base.sha256(artifact_path),
        "Inspection artifact SHA mismatch")
    base.require(inspection["artifact"]["fileIdentifier"] == "TFL3",
        "FlatBuffer identifier mismatch")
    base.require(inspection["schemaVersion"] == 3, "FlatBuffer schema mismatch")
    base.require(inspection["subgraphCount"] == 1, "Expected one subgraph")
    base.require(inspection["customOperatorCount"] == 0, "Custom operators are forbidden")
    graph = inspection["subgraphs"][0]
    inputs = [
        base.bind_tensor(value, expected)
        for value, expected in zip(graph["inputs"], base.EXPECTED_INPUTS, strict=True)
    ]
    outputs = [
        base.bind_tensor(value, expected)
        for value, expected in zip(graph["outputs"], base.EXPECTED_OUTPUTS, strict=True)
    ]

    fixtures = []
    for key in (
        "waveformInput",
        "spectrumInput",
        "frequencyGolden",
        "waveformGolden",
        "frequencyWaveformGolden",
        "combinedGolden",
    ):
        fixtures.append(base.fixture_identity(
            repo_root,
            candidate_root,
            report["fixtures"][key],
            key,
        ))
    for key, role in (("mixInput", "olaMixInput"), ("combinedGolden", "olaCombinedGolden")):
        fixtures.append(base.fixture_identity(
            repo_root,
            candidate_root,
            ola["fixtures"][key],
            role,
        ))

    official_manifest_path = repo_root / OFFICIAL_HOST_MANIFEST
    official = json.loads(official_manifest_path.read_text(encoding="utf-8"))
    base.require(official["status"] == "host-pipeline-passed",
        "Official contract lineage is not frozen as passed")
    base.require(official["modelSemantics"]["stemOrder"] == base.SOURCE_ORDER,
        "Official stem order changed")
    variant = report["variant"]
    return {
        "manifestSchemaVersion": 1,
        "manifestKind": "generated-litert-diagnostic-candidate",
        "candidateId": f"{MODEL_ID}@diagnostic-host-1",
        "modelId": MODEL_ID,
        "status": "host-parity-narrow-failure",
        "scope": "bounded-device-diagnostics-only-not-product-admitted",
        "artifact": base.identity(repo_root, artifact_path, format="tflite-flatbuffer"),
        "variant": variant,
        "license": {
            "distributionStatus": "research-only",
            "productUseApproved": False,
            "reason": "MoisesDB training-data terms require independent review",
        },
        "provenance": {
            **verified_provenance(repo_root, report),
            "loaderRevision": report["provenance"]["loaderRevision"],
            "loaderCheckout": report["provenance"]["loaderCheckout"],
            "demucsLiteReferenceRevision": report["provenance"][
                "demucsLiteReferenceRevision"
            ],
            "demucsLiteReferenceCheckout": report["provenance"][
                "demucsLiteReferenceCheckout"
            ],
        },
        "conversion": {
            "versions": report["versions"],
            "recipe": report["conversionRecipe"],
            "exportReport": base.identity(repo_root, report_path, format="json"),
        },
        "flatBuffer": {
            "fileIdentifier": inspection["artifact"]["fileIdentifier"],
            "schemaVersion": inspection["schemaVersion"],
            "description": inspection["description"],
            "subgraphCount": inspection["subgraphCount"],
            "operatorCodeCount": inspection["operatorCodeCount"],
            "operatorCount": inspection["operatorCount"],
            "tensorCount": inspection["tensorCount"],
            "bufferCount": inspection["bufferCount"],
            "customOperatorCount": inspection["customOperatorCount"],
            "minimumRuntimeVersion": inspection["metadata"].get("min_runtime_version"),
            "inputs": inputs,
            "outputs": outputs,
            "signatures": inspection["signatures"],
            "operatorHistogram": graph["operatorHistogram"],
            "inspectionReport": base.identity(repo_root, inspection_path, format="json"),
        },
        "contractLineage": {
            "abiAndHostDsp": "identical-to-official-canonical-7p8s-contract",
            "officialHostManifest": base.identity(
                repo_root,
                official_manifest_path,
                format="json",
            ),
        },
        "modelSemantics": official["modelSemantics"],
        "hostDspContract": official["hostDspContract"],
        "hostValidation": {
            "parityReference": "guitar-ft Torch FP32",
            "qualityGate": report["qualityGate"],
            "torchCoreReconstruction": report["torchCoreReconstruction"],
            "singleWindowGateEvaluation": report["singleWindowGateEvaluation"],
            "singleWindowMetrics": report["liteRtVsTorch"],
            "canonicalOla": ola,
            "reproducedWithReuseLiteRt": True,
        },
        "admission": {
            "acceptedForProduct": False,
            "acceptedForGeneralDeviceTesting": False,
            "boundedDiagnosticTestingAllowed": True,
            "blockingGate": "canonical OLA EOF minimum 80 dB",
            "observed": {
                "fullOlaSignalToNoiseDb": ola["liteRtVsTorchOla"]["full"][
                    "metrics"
                ]["aggregate"]["signalToNoiseDb"],
                "eofAggregateSignalToNoiseDb": eof["aggregate"]["signalToNoiseDb"],
                "eofGuitarSignalToNoiseDb": eof["perStem"]["guitar"][
                    "signalToNoiseDb"
                ],
                "eofMaximumAbsoluteError": eof["aggregate"]["maxAbsoluteError"],
            },
        },
        "fixtures": fixtures,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    candidate_root = (
        args.candidate_root
        if args.candidate_root.is_absolute()
        else repo_root / args.candidate_root
    ).resolve()
    output = args.output or candidate_root / (
        "diagnostic-manifest.json"
        if args.allow_narrow_host_failure
        else "candidate-manifest.json"
    )
    if not output.is_absolute():
        output = repo_root / output
    output = output.resolve()
    manifest = (
        build_diagnostic_manifest(repo_root, candidate_root)
        if args.allow_narrow_host_failure
        else build_manifest(repo_root, candidate_root)
    )
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    if args.check:
        base.require(output.is_file(), f"Missing frozen manifest: {output}")
        base.require(output.read_bytes() == payload, "Frozen manifest is stale")
        status = "verified"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        status = "written"
    print(json.dumps({
        "status": status,
        "path": base.relative_path(repo_root, output),
        "byteSize": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
