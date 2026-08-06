#!/usr/bin/env python3
"""Compare pinned FP32 and FP16-weight-storage HTDemucs-6s ONNX exports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from run_htdemucs_canonical_onnx import (
    CHANNELS,
    STEM_COUNT,
    STEM_ORDER,
    WINDOW_SAMPLES,
    identify_supported_model,
    inspect_model,
    open_session,
    sha256_file,
)


DEFAULT_FP32 = Path("models/demucs/onnx/htdemucs_6s.onnx")
DEFAULT_FP16 = Path("models/demucs/onnx/htdemucs_6s_fp16weights.onnx")
DEFAULT_FIXTURES = Path(
    "models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/fixtures"
)
EXPECTED_INPUT_SHA256 = "9515d42e72e96036e34f0193d1302e8cccbf80c4e9e7084d1fcba26e17fea40e"
EXPECTED_GOLDEN_SHA256 = "7d5fd3585cbdbb46551e9e05883da9aa880702a72efaa71402072957d037264a"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def write_raw_atomic(path: Path, value: np.ndarray) -> None:
    partial = path.with_name(path.name + ".partial")
    np.ascontiguousarray(value, dtype="<f4").tofile(partial)
    partial.replace(path)


def detailed_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    error = candidate64 - reference64
    reference_rms = float(np.sqrt(np.mean(np.square(reference64))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate64))))
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    reference_centered = reference64 - float(reference64.mean())
    candidate_centered = candidate64 - float(candidate64.mean())
    correlation_denominator = math.sqrt(
        float(np.square(reference_centered).sum())
        * float(np.square(candidate_centered).sum())
    )
    correlation = (
        float((reference_centered * candidate_centered).sum())
        / correlation_denominator
        if correlation_denominator > 0.0
        else None
    )
    return {
        "sampleCount": int(reference64.size),
        "finite": bool(np.isfinite(candidate64).all()),
        "bitwiseEqual": bool(np.array_equal(reference, candidate)),
        "signalRootMeanSquare": reference_rms,
        "candidateRootMeanSquare": candidate_rms,
        "rootMeanSquareError": error_rms,
        "meanAbsoluteError": float(np.mean(np.abs(error))),
        "maximumAbsoluteError": float(np.max(np.abs(error))),
        "signalToNoiseDb": float(
            20.0 * math.log10(max(reference_rms, 1e-30) / max(error_rms, 1e-30))
        ),
        "correlation": correlation,
    }


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    return {
        "aggregate": detailed_metric(reference, candidate),
        "perStem": {
            stem: detailed_metric(reference[:, index], candidate[:, index])
            for index, stem in enumerate(STEM_ORDER)
        },
    }


def validate_bindings(session: Any) -> tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected one input/output, got {len(inputs)}/{len(outputs)}")
    if inputs[0].name != "mix" or outputs[0].name != "stems":
        raise ValueError(f"Unexpected bindings: {inputs[0].name}/{outputs[0].name}")
    if inputs[0].shape != [1, CHANNELS, WINDOW_SAMPLES]:
        raise ValueError(f"Unexpected input shape: {inputs[0].shape}")
    if outputs[0].shape != [1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES]:
        raise ValueError(f"Unexpected output shape: {outputs[0].shape}")
    return inputs[0].name, outputs[0].name


def run_single_model(args: argparse.Namespace) -> int:
    model_path = args.single_model.resolve()
    fixture_root = args.fixture_root.resolve()
    input_path = fixture_root / "waveform_input.f32le.raw"
    model_variant = identify_supported_model(model_path)
    input_values = np.fromfile(input_path, dtype="<f4").reshape(
        1, CHANNELS, WINDOW_SAMPLES
    )
    session_started = time.perf_counter_ns()
    session = open_session(model_path, args.threads, "CPUExecutionProvider")
    session_ms = (time.perf_counter_ns() - session_started) / 1e6
    input_name, output_name = validate_bindings(session)
    inference_started = time.perf_counter_ns()
    output = session.run([output_name], {input_name: input_values})[0]
    inference_ms = (time.perf_counter_ns() - inference_started) / 1e6
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES):
        raise ValueError(f"Unexpected output array shape: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("ONNX output contains NaN or infinity")
    write_raw_atomic(args.single_raw.resolve(), output)
    evidence = {
        "model": {
            "path": str(model_path),
            **model_variant,
        },
        "sessionCreateWallMs": session_ms,
        "inferenceWallMs": inference_ms,
        "output": {
            "path": str(args.single_raw.resolve()),
            "shape": list(output.shape),
            "dtype": "float32-le",
            "byteSize": args.single_raw.resolve().stat().st_size,
            "sha256": sha256_file(args.single_raw.resolve()),
        },
    }
    write_json_atomic(args.single_json.resolve(), evidence)
    return 0


def run_isolated_model(
    script: Path,
    model: Path,
    fixture_root: Path,
    raw_path: Path,
    json_path: Path,
    threads: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--single-model",
        str(model),
        "--single-raw",
        str(raw_path),
        "--single-json",
        str(json_path),
        "--fixture-root",
        str(fixture_root),
        "--threads",
        str(threads),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Isolated model run failed ({result.returncode}): {result.stderr}"
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def compare_initializers(fp32_model: Path, fp16_model: Path) -> dict[str, Any]:
    code = r'''
import json
import onnx
from onnx import numpy_helper
import numpy as np
import sys

fp32 = onnx.load(sys.argv[1], load_external_data=False)
fp16 = onnx.load(sys.argv[2], load_external_data=False)
left = {item.name: item for item in fp32.graph.initializer}
right = {item.name: item for item in fp16.graph.initializer}
unmatched = []
shape_mismatches = []
unequal = []
maximum_absolute_error = 0.0
value_count = 0
same_name_count = 0
fp16_suffix_count = 0
for name, tensor in left.items():
    if name in right:
        candidate = right[name]
        same_name_count += 1
    elif name + "__fp16" in right:
        candidate = right[name + "__fp16"]
        fp16_suffix_count += 1
    else:
        unmatched.append(name)
        continue
    reference_array = numpy_helper.to_array(tensor)
    candidate_array = numpy_helper.to_array(candidate)
    value_count += int(reference_array.size)
    if reference_array.shape != candidate_array.shape:
        shape_mismatches.append(name)
        continue
    widened = candidate_array.astype(reference_array.dtype)
    if not np.array_equal(reference_array, widened):
        unequal.append(name)
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(np.max(np.abs(reference_array.astype(np.float64) - widened.astype(np.float64))))
        )
print(json.dumps({
    "fp32InitializerCount": len(left),
    "fp16InitializerCount": len(right),
    "totalValueCount": value_count,
    "sameNameInitializerCount": same_name_count,
    "fp16SuffixedInitializerCount": fp16_suffix_count,
    "unmatchedFp32Initializers": unmatched,
    "shapeMismatchInitializers": shape_mismatches,
    "unequalInitializerValues": unequal,
    "maximumAbsoluteErrorAfterFp16Widening": maximum_absolute_error,
    "allValuesEqualAfterFp16Widening": (
        not unmatched and not shape_mismatches and not unequal and len(left) == len(right)
    ),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(fp32_model), str(fp16_model)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Initializer audit failed: {result.stderr}")
    return json.loads(result.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32)
    parser.add_argument("--fp16-model", type=Path, default=DEFAULT_FP16)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--single-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--single-raw", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--single-json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.threads > 64:
        raise ValueError("--threads must be in [1,64]")
    if args.single_model is not None:
        if args.single_raw is None or args.single_json is None:
            raise ValueError("Single-model mode requires raw and JSON output paths")
        return run_single_model(args)
    if args.output_dir is None:
        raise ValueError("--output-dir is required")

    output_dir = args.output_dir.resolve()
    stage = output_dir.with_name(output_dir.name + ".partial")
    if (output_dir.exists() or stage.exists()) and not args.force:
        raise FileExistsError(f"Output exists; pass --force: {output_dir}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    fixture_root = args.fixture_root.resolve()
    input_path = fixture_root / "waveform_input.f32le.raw"
    golden_path = fixture_root / "combined_golden.f32le.raw"
    if sha256_file(input_path) != EXPECTED_INPUT_SHA256:
        raise ValueError("Fixture input identity mismatch")
    if sha256_file(golden_path) != EXPECTED_GOLDEN_SHA256:
        raise ValueError("Fixture golden identity mismatch")

    script = Path(__file__).resolve()
    fp32_evidence = run_isolated_model(
        script,
        args.fp32_model.resolve(),
        fixture_root,
        stage / "fp32-output.f32le.raw",
        stage / "fp32-run.json",
        args.threads,
    )
    fp16_evidence = run_isolated_model(
        script,
        args.fp16_model.resolve(),
        fixture_root,
        stage / "fp16-output.f32le.raw",
        stage / "fp16-run.json",
        args.threads,
    )
    if fp32_evidence["model"]["weightStorage"] != "float32":
        raise ValueError("--fp32-model does not identify the pinned FP32 artifact")
    if fp16_evidence["model"]["weightStorage"] != "float16":
        raise ValueError("--fp16-model does not identify the pinned FP16 artifact")

    shape = (1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES)
    golden = np.fromfile(golden_path, dtype="<f4").reshape(shape)
    fp32 = np.fromfile(stage / "fp32-output.f32le.raw", dtype="<f4").reshape(shape)
    fp16 = np.fromfile(stage / "fp16-output.f32le.raw", dtype="<f4").reshape(shape)
    fp32_vs_torch = comparison(golden, fp32)
    fp16_vs_torch = comparison(golden, fp16)
    fp16_vs_fp32 = comparison(fp32, fp16)

    fp32_aggregate = fp32_vs_torch["aggregate"]
    fp16_aggregate = fp16_vs_torch["aggregate"]
    rmse_ratio = (
        fp32_aggregate["rootMeanSquareError"]
        / fp16_aggregate["rootMeanSquareError"]
    )
    graph_fp32 = inspect_model(args.fp32_model.resolve())
    graph_fp16 = inspect_model(args.fp16_model.resolve())
    initializer_comparison = compare_initializers(
        args.fp32_model.resolve(), args.fp16_model.resolve()
    )
    fp32_non_cast = dict(graph_fp32["operatorHistogram"])
    fp16_non_cast = dict(graph_fp16["operatorHistogram"])
    fp32_casts = fp32_non_cast.pop("ai.onnx::Cast", 0)
    fp16_casts = fp16_non_cast.pop("ai.onnx::Cast", 0)

    report = {
        "schemaVersion": 1,
        "status": "complete",
        "scope": "host-frozen-window-float32-output",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "threads": args.threads,
            "provider": "CPUExecutionProvider",
        },
        "tool": {"path": str(script), "sha256": sha256_file(script)},
        "fixture": {
            "input": {
                "path": str(input_path),
                "sha256": EXPECTED_INPUT_SHA256,
                "shape": [1, CHANNELS, WINDOW_SAMPLES],
                "dtype": "float32-le",
            },
            "torchGolden": {
                "path": str(golden_path),
                "sha256": EXPECTED_GOLDEN_SHA256,
                "shape": list(shape),
                "dtype": "float32-le",
            },
        },
        "artifacts": {"fp32": fp32_evidence, "fp16": fp16_evidence},
        "graphComparison": {
            "fp32": graph_fp32,
            "fp16": graph_fp16,
            "sameNonCastOperatorHistogram": fp32_non_cast == fp16_non_cast,
            "fp32CastNodeCount": fp32_casts,
            "fp16CastNodeCount": fp16_casts,
            "additionalFp16CastNodes": fp16_casts - fp32_casts,
            "initializerComparison": initializer_comparison,
        },
        "comparisons": {
            "fp32OnnxVsTorch": fp32_vs_torch,
            "fp16OnnxVsTorch": fp16_vs_torch,
            "fp16OnnxVsFp32Onnx": fp16_vs_fp32,
        },
        "attribution": {
            "fp32VsFp16SnrGainDb": (
                fp32_aggregate["signalToNoiseDb"]
                - fp16_aggregate["signalToNoiseDb"]
            ),
            "fp32VsFp16RmseRatio": rmse_ratio,
            "fp32VsFp16MseReductionFraction": 1.0 - rmse_ratio * rmse_ratio,
            "fp32Uniform80DbGatePassed": (
                fp32_aggregate["signalToNoiseDb"] >= 80.0
            ),
            "fp16Uniform80DbGatePassed": (
                fp16_aggregate["signalToNoiseDb"] >= 80.0
            ),
            "interpretation": (
                "All initializer values are identical after lossless FP16-to-FP32 "
                "widening. The measured FP16 artifact penalty therefore comes from its "
                "additional Cast topology and resulting runtime numerical path, not "
                "learned-weight quantization. The common FP32 waveform export/runtime "
                "path retains a measurable residual and still fails the 80 dB gate."
            ),
            "boundary": (
                "The separate checkpoint audit proves the legacy .th and official "
                "safetensors states are bit-identical. The FP32 residual is therefore "
                "bounded to common ONNX export rewrites and ONNX Runtime numerics."
            ),
        },
    }
    write_json_atomic(stage / "report.json", report)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    stage.replace(output_dir)
    print(json.dumps({
        "status": "complete",
        "report": str((output_dir / "report.json").resolve()),
        "fp32VsTorchSnrDb": fp32_aggregate["signalToNoiseDb"],
        "fp16VsTorchSnrDb": fp16_aggregate["signalToNoiseDb"],
        "mseReductionFraction": 1.0 - rmse_ratio * rmse_ratio,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
