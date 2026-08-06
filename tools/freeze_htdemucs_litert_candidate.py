#!/usr/bin/env python3
"""Freeze and verify the canonical generated HTDemucs LiteRT candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0"
PROFILE_NAME = "canonical_7p8s"
SAMPLE_COUNT = 343_980
SPECTRUM_FRAME_COUNT = 336
SOURCE_ORDER = ["drums", "bass", "other", "vocals", "guitar", "piano"]
FEATURE_ORDER = ["L.real", "L.imag", "R.real", "R.imag"]
EXPECTED_INPUTS = [
    ("serving_default_args_0", [1, 2, SAMPLE_COUNT], ["batch", "channel", "sample"]),
    (
        "serving_default_args_1",
        [1, 4, 2048, SPECTRUM_FRAME_COUNT],
        ["batch", "feature", "frequency", "frame"],
    ),
]
EXPECTED_OUTPUTS = [
    (
        "serving_default_output_0_output",
        [1, 6, 4, 2048, SPECTRUM_FRAME_COUNT],
        ["batch", "stem", "feature", "frequency", "frame"],
    ),
    (
        "serving_default_output_1_output",
        [1, 6, 2, SAMPLE_COUNT],
        ["batch", "stem", "channel", "sample"],
    ),
]
SOURCE_PATHS = {
    "canonicalWeight": "models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors",
    "metadata": "models/demucs/official-hf/htdemucs_6s/5c90dfd2.json",
    "bagManifest": "models/demucs/official-hf/htdemucs_6s/htdemucs_6s.yaml",
    "boundarySource": ".tmp/demucs-lite-reference/demucs-for-onnx/demucs/htdemucs.py",
    "referenceExporter": ".tmp/demucs-lite-reference/scripts/convert-pth-to-onnx-chunked.py",
    "exportScript": "tools/export_htdemucs_litert_candidate.py",
    "requirementsLock": "requirements-demucs-litert-export.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("models/demucs/generated") / MODEL_ID,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Path is outside the repository: {path}") from error


def identity(repo_root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    require(path.is_file(), f"Missing file: {path}")
    return {
        "localPath": relative_path(repo_root, path),
        "fileName": path.name,
        "byteSize": path.stat().st_size,
        "sha256": sha256(path),
        **extra,
    }


def verify_declared_file(path: Path, declaration: dict[str, Any], label: str) -> None:
    require(path.is_file(), f"Missing {label}: {path}")
    require(path.name == declaration["fileName"], f"{label} file name mismatch")
    if "byteSize" in declaration:
        require(path.stat().st_size == declaration["byteSize"], f"{label} byte size mismatch")
    require(sha256(path) == declaration["sha256"], f"{label} SHA-256 mismatch")


def bind_tensor(tensor: dict[str, Any], expected: tuple[str, list[int], list[str]]) -> dict[str, Any]:
    name, shape, axes = expected
    require(tensor["name"] == name, f"Tensor name mismatch: {tensor['name']} != {name}")
    require(tensor["shape"] == shape, f"Tensor shape mismatch for {name}")
    require(tensor["dtype"] == "float32", f"Tensor dtype mismatch for {name}")
    require(not tensor["dynamic"], f"Dynamic tensor is not allowed: {name}")
    quantization = tensor["quantization"]
    require(
        quantization["scaleCount"] == 0 and quantization["zeroPointCount"] == 0,
        f"Quantized tensor is not allowed: {name}",
    )
    return {
        "index": tensor["index"],
        "tensorIndex": tensor["tensorIndex"],
        "name": name,
        "dtype": tensor["dtype"],
        "shape": shape,
        "axes": axes,
    }


def fixture_identity(
    repo_root: Path,
    candidate_root: Path,
    declaration: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    path = candidate_root / "fixtures" / declaration["fileName"]
    verify_declared_file(path, declaration, role)
    return identity(
        repo_root,
        path,
        role=role,
        dtype=declaration["dtype"],
        shape=declaration["shape"],
    )


def build_manifest(repo_root: Path, candidate_root: Path) -> dict[str, Any]:
    export_report_path = candidate_root / "export-report.json"
    inspection_path = candidate_root / "flatbuffer-inspection.json"
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))

    require(export_report["modelId"] == MODEL_ID, "Export report model ID mismatch")
    require(export_report["profileName"] == PROFILE_NAME, "Export report profile mismatch")
    require(export_report["status"] == "passed", "Export report did not pass")
    require(export_report["completedStage"] == "complete", "Export report is incomplete")
    quality_gate = export_report["qualityGate"]
    require(quality_gate["hostPipelineGatePassed"], "Host pipeline gate did not pass")
    require(quality_gate["acceptedForDeviceTesting"], "Candidate was not accepted for device testing")

    profile = export_report["profile"]
    require(profile["name"] == PROFILE_NAME, "Profile name mismatch")
    require(profile["sampleRate"] == 44_100, "Sample rate mismatch")
    require(profile["sampleCount"] == SAMPLE_COUNT, "Sample count mismatch")
    require(profile["segment"] == {"numerator": 39, "denominator": 5}, "Segment mismatch")
    require(profile["sourceOrder"] == SOURCE_ORDER, "Source order mismatch")

    artifact_path = candidate_root / export_report["liteRtArtifact"]["fileName"]
    verify_declared_file(artifact_path, export_report["liteRtArtifact"], "LiteRT artifact")
    require(inspection["artifact"]["byteSize"] == artifact_path.stat().st_size, "Inspection byte size mismatch")
    require(inspection["artifact"]["sha256"] == sha256(artifact_path), "Inspection artifact SHA mismatch")
    require(inspection["artifact"]["fileIdentifier"] == "TFL3", "FlatBuffer identifier mismatch")
    require(inspection["schemaVersion"] == 3, "FlatBuffer schema mismatch")
    require(inspection["subgraphCount"] == 1, "Expected one FlatBuffer subgraph")
    require(inspection["customOperatorCount"] == 0, "Custom operators are not allowed")

    graph = inspection["subgraphs"][0]
    require(len(graph["inputs"]) == len(EXPECTED_INPUTS), "Input count mismatch")
    require(len(graph["outputs"]) == len(EXPECTED_OUTPUTS), "Output count mismatch")
    inputs = [bind_tensor(value, expected) for value, expected in zip(graph["inputs"], EXPECTED_INPUTS, strict=True)]
    outputs = [bind_tensor(value, expected) for value, expected in zip(graph["outputs"], EXPECTED_OUTPUTS, strict=True)]
    signature = inspection["signatures"]
    require(len(signature) == 1, "Expected one signature")
    signature = signature[0]
    require(signature["key"] == "serving_default", "Signature key mismatch")
    require(
        [value["tensorIndex"] for value in signature["inputs"]]
        == [value["tensorIndex"] for value in inputs],
        "Signature input binding mismatch",
    )
    require(
        [value["tensorIndex"] for value in signature["outputs"]]
        == [value["tensorIndex"] for value in outputs],
        "Signature output binding mismatch",
    )
    require(sum(graph["operatorHistogram"].values()) == inspection["operatorCount"], "Operator histogram mismatch")

    provenance: dict[str, Any] = {}
    for key, local_path in SOURCE_PATHS.items():
        path = repo_root / local_path
        declared = export_report["provenance"][key]
        verify_declared_file(path, declared, key)
        provenance[key] = identity(
            repo_root,
            path,
            format=declared.get("format"),
        )

    fixtures = []
    for key in (
        "waveformInput",
        "spectrumInput",
        "frequencyGolden",
        "waveformGolden",
        "frequencyWaveformGolden",
        "combinedGolden",
    ):
        fixtures.append(
            fixture_identity(
                repo_root,
                candidate_root,
                export_report["fixtures"][key],
                key,
            )
        )
    ola_validation = export_report["canonicalOlaValidation"]
    require(ola_validation["status"] == "passed", "Canonical OLA validation failed")
    require(ola_validation["paddingVsOfficialTensorChunkBitwiseEqual"], "Tail padding oracle mismatch")
    require(
        ola_validation["torchCoreOlaVsOfficialApplyModel"]["aggregate"]["bitwiseEqual"],
        "Torch OLA is not bitwise equal to official apply_model",
    )
    for key, role in (("mixInput", "olaMixInput"), ("combinedGolden", "olaCombinedGolden")):
        fixtures.append(
            fixture_identity(
                repo_root,
                candidate_root,
                ola_validation["fixtures"][key],
                role,
            )
        )

    metadata = inspection["metadata"]
    return {
        "manifestSchemaVersion": 1,
        "manifestKind": "generated-litert-host-candidate",
        "candidateId": f"{MODEL_ID}@host-1",
        "modelId": MODEL_ID,
        "status": "host-pipeline-passed",
        "scope": "host-only-not-a-device-result",
        "artifact": identity(repo_root, artifact_path, format="tflite-flatbuffer"),
        "provenance": {
            **provenance,
            "loaderRevision": export_report["provenance"]["loaderRevision"],
            "loaderCheckout": export_report["provenance"]["loaderCheckout"],
            "demucsLiteReferenceRevision": export_report["provenance"]["demucsLiteReferenceRevision"],
            "demucsLiteReferenceCheckout": export_report["provenance"]["demucsLiteReferenceCheckout"],
        },
        "conversion": {
            "versions": export_report["versions"],
            "recipe": export_report["conversionRecipe"],
            "exportReport": identity(repo_root, export_report_path, format="json"),
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
            "minimumRuntimeVersion": metadata.get("min_runtime_version"),
            "keepStablehloConstant": metadata.get("keep_stablehlo_constant"),
            "inputs": inputs,
            "outputs": outputs,
            "signature": signature,
            "operatorHistogram": graph["operatorHistogram"],
            "inspectionReport": identity(repo_root, inspection_path, format="json"),
        },
        "modelSemantics": {
            "sampleRate": 44_100,
            "channelCount": 2,
            "windowSamples": SAMPLE_COUNT,
            "spectrumFrameCount": SPECTRUM_FRAME_COUNT,
            "featureOrder": FEATURE_ORDER,
            "stemOrder": SOURCE_ORDER,
        },
        "hostDspContract": {
            "globalNormalization": {
                "reference": "mean-across-stereo-channels",
                "standardDeviationCorrection": 1,
                "epsilon": 1e-8,
            },
            "stft": {
                "nFft": 4096,
                "hopLength": 1024,
                "window": "periodic-hann",
                "normalized": True,
                "center": True,
                "padMode": "reflect",
                "outerPadLeft": 1536,
                "outerPadRight": 1620,
                "dropNyquistBin": True,
                "frameCropLeft": 2,
                "frameCropRight": 2,
            },
            "istft": {
                "restoreZeroNyquistBin": True,
                "framePadLeft": 2,
                "framePadRight": 2,
                "normalized": True,
                "center": True,
                "reconstructionLength": 347_136,
                "cropStart": 1536,
                "cropEnd": 345_516,
            },
            "branchCombination": "frequency-istft-plus-time-waveform",
            "ola": ola_validation["profile"],
            "tailWindowPlans": ola_validation["windowPlans"],
        },
        "hostValidation": {
            "qualityGate": quality_gate,
            "torchCoreReconstruction": export_report["torchCoreReconstruction"],
            "singleWindowGateEvaluation": export_report["singleWindowGateEvaluation"],
            "singleWindowMetrics": export_report["liteRtVsTorch"],
            "canonicalOla": {
                "status": ola_validation["status"],
                "uniformTensorGatePassed": ola_validation["uniformTensorGatePassed"],
                "qualityGatePolicy": ola_validation["qualityGatePolicy"],
                "paddingVsOfficialTensorChunkBitwiseEqual": ola_validation[
                    "paddingVsOfficialTensorChunkBitwiseEqual"
                ],
                "torchCoreOlaVsOfficialApplyModel": ola_validation[
                    "torchCoreOlaVsOfficialApplyModel"
                ],
                "liteRtVsTorchOla": ola_validation["liteRtVsTorchOla"],
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
    output = args.output or candidate_root / "candidate-manifest.json"
    if not output.is_absolute():
        output = repo_root / output
    output = output.resolve()
    manifest = build_manifest(repo_root, candidate_root)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    if args.check:
        require(output.is_file(), f"Missing frozen manifest: {output}")
        require(output.read_bytes() == payload, "Frozen manifest is stale")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(
        json.dumps(
            {
                "status": "verified" if args.check else "written",
                "path": relative_path(repo_root, output),
                "byteSize": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
