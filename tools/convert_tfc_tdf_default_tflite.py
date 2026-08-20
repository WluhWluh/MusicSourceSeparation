#!/usr/bin/env python3
"""Convert the frozen TFC-TDF ONNX core to FP32 TFLite and validate it."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from tfc_tdf_default_model import (
    DEFAULT_CONFIG,
    TfcTdfNchwWrapper,
    load_default_checkpoint,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "models" / "tfc-tdf" / "default-compact"
DEFAULT_ONNX = DEFAULT_OUTPUT_DIR / "tfc_tdf_default_vocals_core_fp32.onnx"
TFLITE_FILE_NAME = "tfc_tdf_default_vocals_core_fp32.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_CONFIG.num_frames)
    parser.add_argument("--artifact-name")
    parser.add_argument("--seed", type=int, default=9662)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(candidate).all():
        raise ValueError("Candidate contains non-finite values")
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    error = candidate64 - reference64
    signal_power = float(np.sum(reference64 * reference64))
    error_power = float(np.sum(error * error))
    candidate_power = float(np.sum(candidate64 * candidate64))
    snr = float("inf") if error_power == 0.0 else float(
        10.0 * np.log10(signal_power / error_power)
    )
    cosine = float(
        np.sum(reference64 * candidate64)
        / np.sqrt(signal_power * candidate_power)
    )
    return {
        "maxAbsError": float(np.max(np.abs(error))),
        "meanAbsError": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "signalRms": float(np.sqrt(np.mean(reference64 * reference64))),
        "snrDb": snr,
        "cosineSimilarity": cosine,
    }


def append_nhwc_output(source: Path, target: Path, num_frames: int) -> None:
    """Keep ONNX input NCHW but make its conversion output explicitly NHWC."""
    model = onnx.load(source, load_external_data=True)
    if len(model.graph.output) != 1:
        raise ValueError("Expected exactly one ONNX output")
    original_output = model.graph.output[0]
    original_name = original_output.name
    original_shape = [
        int(dimension.dim_value)
        for dimension in original_output.type.tensor_type.shape.dim
    ]
    if original_shape != [1, 4, 1025, num_frames]:
        raise ValueError(f"Unexpected ONNX output shape: {original_shape}")

    internal_name = f"{original_name}_nchw_internal"
    producers = [node for node in model.graph.node if original_name in node.output]
    if len(producers) != 1:
        raise ValueError(f"Expected one producer for ONNX output {original_name}")
    producer = producers[0]
    producer.output[list(producer.output).index(original_name)] = internal_name
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == original_name:
                node.input[index] = internal_name

    model.graph.output.pop()
    model.graph.node.append(
        helper.make_node(
            "Transpose",
            [internal_name],
            [original_name],
            perm=[0, 2, 3, 1],
            name="OutputToNhwc",
        )
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            original_name,
            TensorProto.FLOAT,
            [1, 1025, num_frames, 4],
        )
    )
    onnx.checker.check_model(model, full_check=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)


def run_onnx2tf(source: Path, work_dir: Path) -> tuple[list[str], Path, list[Path]]:
    converted_dir = work_dir / "onnx2tf"
    converted_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(source),
        "-o",
        str(converted_dir),
        "-tb",
        "flatbuffer_direct",
        "-coion",
        "-roc",
        "-n",
        "-v",
        "error",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = work_dir / "onnx2tf.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"onnx2tf failed with exit code {completed.returncode}; see {log_path}"
        )
    float32 = sorted(converted_dir.glob("*_float32.tflite"))
    if len(float32) != 1:
        raise RuntimeError(f"Expected one FP32 TFLite artifact, found {float32}")
    reports = sorted(converted_dir.glob("*_report.json"))
    return command, float32[0], reports


def load_interpreter(path: Path, threads: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
    return Interpreter(model_path=str(path), num_threads=threads)


def inspect_flatbuffer(path: Path, threads: int) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload[4:8] != b"TFL3":
        raise ValueError(f"Unexpected FlatBuffer file identifier: {payload[4:8]!r}")
    interpreter = load_interpreter(path, threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("Expected one inspected TFLite input and output")

    def tensor_detail(detail: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(detail["name"]),
            "tensorIndex": int(detail["index"]),
            "shape": [int(value) for value in detail["shape"]],
            "shapeSignature": [int(value) for value in detail["shape_signature"]],
            "dtype": np.dtype(detail["dtype"]).name,
            "quantizationScaleCount": int(
                len(detail["quantization_parameters"]["scales"])
            ),
            "quantizationZeroPointCount": int(
                len(detail["quantization_parameters"]["zero_points"])
            ),
        }

    operations = interpreter._get_ops_details()  # pylint: disable=protected-access
    histogram = Counter(str(operation["op_name"]) for operation in operations)
    custom_operators = sorted(
        name for name in histogram if name == "CUSTOM" or name.startswith("CUSTOM:")
    )
    return {
        "artifact": {
            "file": path.name,
            "bytes": len(payload),
            "sha256": sha256_file(path),
            "fileIdentifier": "TFL3",
        },
        "subgraphCount": 1,
        "operatorCount": len(operations),
        "tensorCount": len(interpreter.get_tensor_details()),
        "customOperatorCount": sum(histogram[name] for name in custom_operators),
        "inputs": [tensor_detail(input_details[0])],
        "outputs": [tensor_detail(output_details[0])],
        "operatorHistogram": dict(histogram.most_common()),
    }


def validate_flatbuffer(
    checkpoint: Path,
    onnx_path: Path,
    tflite_path: Path,
    samples: int,
    seed: int,
    threads: int,
    num_frames: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    model, _ = load_default_checkpoint(checkpoint)
    wrapper = TfcTdfNchwWrapper(model).eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    interpreter = load_interpreter(tflite_path, threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("Expected one TFLite input and output")
    input_detail = input_details[0]
    output_detail = output_details[0]
    expected_nhwc = (1, 1025, num_frames, 4)
    if tuple(int(value) for value in input_detail["shape"]) != expected_nhwc:
        raise ValueError(f"Unexpected TFLite input shape: {input_detail['shape']}")
    if tuple(int(value) for value in output_detail["shape"]) != expected_nhwc:
        raise ValueError(f"Unexpected TFLite output shape: {output_detail['shape']}")
    if input_detail["dtype"] != np.float32 or output_detail["dtype"] != np.float32:
        raise ValueError("TFLite boundary must remain float32")

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    shape_nchw = (1, 4, 1025, num_frames)
    with torch.inference_mode():
        for index in range(samples):
            input_nchw = rng.normal(0.0, 0.1, size=shape_nchw).astype(np.float32)
            pytorch_output = wrapper(torch.from_numpy(input_nchw)).numpy()
            onnx_output = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: input_nchw},
            )[0]
            input_nhwc = np.ascontiguousarray(
                np.transpose(input_nchw, (0, 2, 3, 1))
            )
            interpreter.set_tensor(int(input_detail["index"]), input_nhwc)
            interpreter.invoke()
            tflite_output_nhwc = interpreter.get_tensor(int(output_detail["index"]))
            tflite_output = np.transpose(tflite_output_nhwc, (0, 3, 1, 2))
            results.append(
                {
                    "sample": index,
                    "pytorchVsOnnx": metrics(pytorch_output, onnx_output),
                    "pytorchVsTflite": metrics(pytorch_output, tflite_output),
                    "onnxVsTflite": metrics(onnx_output, tflite_output),
                }
            )

    comparisons = ("pytorchVsOnnx", "pytorchVsTflite", "onnxVsTflite")
    aggregate = {
        comparison: {
            "minimumSnrDb": min(item[comparison]["snrDb"] for item in results),
            "maximumAbsoluteError": max(
                item[comparison]["maxAbsError"] for item in results
            ),
            "maximumRmse": max(item[comparison]["rmse"] for item in results),
            "minimumCosineSimilarity": min(
                item[comparison]["cosineSimilarity"] for item in results
            ),
        }
        for comparison in comparisons
    }
    gate = {"minimumSnrDb": 95.0, "maximumAbsoluteError": 1e-5}
    passed = all(
        aggregate[comparison]["minimumSnrDb"] >= gate["minimumSnrDb"]
        and aggregate[comparison]["maximumAbsoluteError"]
        <= gate["maximumAbsoluteError"]
        for comparison in comparisons
    )
    if not passed:
        raise ValueError(f"Three-way tensor parity failed: {aggregate}")
    return {
        "seed": seed,
        "sampleCount": samples,
        "gate": gate,
        "samples": results,
        "aggregate": aggregate,
        "passed": passed,
    }


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.num_frames < 8 or args.num_frames % 8 != 0:
        raise ValueError("num-frames must be at least 8 and divisible by 8")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "tflite-conversion-work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    conversion_source = work_dir / "tfc_tdf_default_vocals_core_fp32_nhwc.onnx"
    append_nhwc_output(args.onnx.resolve(), conversion_source, args.num_frames)
    command, generated_tflite, generated_reports = run_onnx2tf(
        conversion_source,
        work_dir,
    )
    artifact_name = args.artifact_name or (
        TFLITE_FILE_NAME
        if args.num_frames == DEFAULT_CONFIG.num_frames
        else f"tfc_tdf_default_vocals_core_fp32_f{args.num_frames}.tflite"
    )
    tflite_path = output_dir / artifact_name
    shutil.copy2(generated_tflite, tflite_path)

    report_identities: list[dict[str, Any]] = []
    for source_report in generated_reports:
        target_report = output_dir / source_report.name
        shutil.copy2(source_report, target_report)
        report_identities.append(
            {
                "file": target_report.name,
                "bytes": target_report.stat().st_size,
                "sha256": sha256_file(target_report),
            }
        )
    log_target = output_dir / "onnx2tf.log"
    shutil.copy2(work_dir / "onnx2tf.log", log_target)

    flatbuffer = inspect_flatbuffer(tflite_path, args.threads)
    expected_shape = [1, 1025, args.num_frames, 4]
    if flatbuffer["customOperatorCount"] != 0:
        raise ValueError("TFLite artifact contains custom operators")
    if flatbuffer["inputs"][0]["shape"] != expected_shape:
        raise ValueError(f"Unexpected inspected input: {flatbuffer['inputs'][0]}")
    if flatbuffer["outputs"][0]["shape"] != expected_shape:
        raise ValueError(f"Unexpected inspected output: {flatbuffer['outputs'][0]}")
    if flatbuffer["inputs"][0]["dtype"] != "float32":
        raise ValueError("TFLite input is not float32")
    if flatbuffer["outputs"][0]["dtype"] != "float32":
        raise ValueError("TFLite output is not float32")

    validation = validate_flatbuffer(
        args.checkpoint.resolve(),
        args.onnx.resolve(),
        tflite_path,
        args.samples,
        args.seed,
        args.threads,
        args.num_frames,
    )
    report = {
        "schemaVersion": 1,
        "candidateId": (
            "tfc_tdf_default_vocals_core_fp32@tflite-1"
            if args.num_frames == DEFAULT_CONFIG.num_frames
            else f"tfc_tdf_default_vocals_core_fp32_f{args.num_frames}@tflite-1"
        ),
        "status": "tflite-three-way-tensor-parity-passed",
        "sourceOnnx": {
            "file": args.onnx.name,
            "bytes": args.onnx.stat().st_size,
            "sha256": sha256_file(args.onnx),
            "publicOutputLayout": "NCHW",
        },
        "conversionSource": {
            "role": "onnx-with-explicit-nhwc-output-adapter",
            "bytes": conversion_source.stat().st_size,
            "sha256": sha256_file(conversion_source),
            "inputShape": [1, 4, 1025, args.num_frames],
            "outputShape": expected_shape,
        },
        "artifact": {
            "file": tflite_path.name,
            "bytes": tflite_path.stat().st_size,
            "sha256": sha256_file(tflite_path),
            "inputShape": expected_shape,
            "outputShape": expected_shape,
            "dtype": "float32",
            "featureOrder": [
                "left.real",
                "right.real",
                "left.imag",
                "right.imag",
            ],
            "staticNumFrames": args.num_frames,
            "sourceCheckpointNumFrames": DEFAULT_CONFIG.num_frames,
            "checkpointCompatibleShapeOverride": (
                args.num_frames != DEFAULT_CONFIG.num_frames
            ),
        },
        "flatbufferInspection": flatbuffer,
        "validation": validation,
        "conversion": {
            "command": command,
            "backend": "flatbuffer_direct",
            "generatedReports": report_identities,
            "log": {
                "file": log_target.name,
                "bytes": log_target.stat().st_size,
                "sha256": sha256_file(log_target),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "onnx2tf": package_version("onnx2tf"),
            "tensorflow": package_version("tensorflow"),
            "aiEdgeLiteRt": package_version("ai-edge-litert"),
            "converter": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    report_path = output_dir / "tflite-conversion-report.json"
    json_write(report_path, report)
    if not args.keep_work:
        shutil.rmtree(work_dir)

    print(
        json.dumps(
            {
                "artifact": report["artifact"],
                "tensorParity": validation["aggregate"],
            },
            indent=2,
        )
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
