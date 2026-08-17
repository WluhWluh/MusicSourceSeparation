#!/usr/bin/env python3
"""Export and validate the public default TFC-TDF neural core as ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tfc_tdf_default_model import (
    DEFAULT_CONFIG,
    TfcTdfNchwWrapper,
    load_default_checkpoint,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "models" / "tfc-tdf" / "default-compact"
ONNX_FILE_NAME = "tfc_tdf_default_vocals_core_fp32.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=9662)
    parser.add_argument("--threads", type=int, default=8)
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


def onnx_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    shape: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append("?")
    return shape


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected exactly one ONNX input and output")
    expected_shape = [1, 4, 1025, 128]
    if onnx_shape(inputs[0]) != expected_shape:
        raise ValueError(f"Unexpected ONNX input shape: {onnx_shape(inputs[0])}")
    if onnx_shape(outputs[0]) != expected_shape:
        raise ValueError(f"Unexpected ONNX output shape: {onnx_shape(outputs[0])}")
    non_float_initializers = [
        item.name
        for item in model.graph.initializer
        if item.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64)
    ]
    if non_float_initializers:
        raise ValueError(f"Unexpected initializer dtypes: {non_float_initializers}")
    operator_histogram: dict[str, int] = {}
    for node in model.graph.node:
        operator_histogram[node.op_type] = operator_histogram.get(node.op_type, 0) + 1
    forbidden_dsp_operators = sorted(
        operator
        for operator in operator_histogram
        if operator.lower() in {"dft", "stft", "hannwindow"}
    )
    if forbidden_dsp_operators:
        raise ValueError(f"Neural core contains DSP operators: {forbidden_dsp_operators}")
    return {
        "irVersion": int(model.ir_version),
        "opsets": {
            domain or "ai.onnx": int(opset.version)
            for opset in model.opset_import
            for domain in [opset.domain]
        },
        "input": {"name": inputs[0].name, "shape": expected_shape, "dtype": "float32"},
        "output": {"name": outputs[0].name, "shape": expected_shape, "dtype": "float32"},
        "nodeCount": len(model.graph.node),
        "initializerCount": len(model.graph.initializer),
        "operatorHistogram": dict(sorted(operator_histogram.items())),
        "externalData": any(bool(item.external_data) for item in model.graph.initializer),
        "containsStftOrIstft": False,
    }


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_metadata = load_default_checkpoint(args.checkpoint)
    wrapper = TfcTdfNchwWrapper(model).eval()

    input_shape = (
        1,
        DEFAULT_CONFIG.input_channels,
        DEFAULT_CONFIG.frequency_bins,
        DEFAULT_CONFIG.num_frames,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_tensor = torch.randn(input_shape, generator=generator) * 0.1
    with torch.inference_mode():
        torch_output = wrapper(input_tensor).numpy()

    onnx_path = output_dir / ONNX_FILE_NAME
    torch.onnx.export(
        wrapper,
        (input_tensor,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )
    inspection = inspect_onnx(onnx_path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: input_tensor.numpy()},
    )[0]
    parity = metrics(torch_output, onnx_output)
    if parity["snrDb"] < 100.0 or parity["maxAbsError"] > 1e-5:
        raise ValueError(f"PyTorch/ONNX parity failed: {parity}")

    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = {
        "input": fixture_dir / "synthetic_input_nchw_f32.bin",
        "pytorchOutput": fixture_dir / "synthetic_pytorch_output_nchw_f32.bin",
        "onnxOutput": fixture_dir / "synthetic_onnx_output_nchw_f32.bin",
    }
    input_tensor.numpy().astype("<f4").tofile(fixture_paths["input"])
    torch_output.astype("<f4").tofile(fixture_paths["pytorchOutput"])
    onnx_output.astype("<f4").tofile(fixture_paths["onnxOutput"])

    artifact = {
        "file": onnx_path.name,
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256_file(onnx_path),
    }
    fixture_identities = {
        key: {
            "file": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in fixture_paths.items()
    }
    report = {
        "schemaVersion": 1,
        "candidateId": "tfc_tdf_default_vocals_core_fp32@onnx-1",
        "status": "onnx-export-passed",
        "checkpoint": checkpoint_metadata["checkpoint"],
        "checkpointState": checkpoint_metadata["state"],
        "checkpointHyperparameters": checkpoint_metadata["hyperparameters"],
        "neuralCore": {
            "boundary": "external-stft-istft",
            "inputLayout": "NCHW",
            "inputFeatureOrder": [
                "left.real",
                "left.imag",
                "right.real",
                "right.imag",
            ],
            "inputShape": list(input_shape),
            "outputShape": list(input_shape),
            "dtype": "float32",
            "targetStem": "vocals",
            "residualStem": "instrumental",
        },
        "artifact": artifact,
        "inspection": inspection,
        "validation": {
            "seed": args.seed,
            "pytorchVsOnnx": parity,
            "gate": {"minimumSnrDb": 100.0, "maximumAbsoluteError": 1e-5},
            "passed": True,
        },
        "fixtures": fixture_identities,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "exporter": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__)),
            },
            "modelDefinition": {
                "file": "tfc_tdf_default_model.py",
                "sha256": sha256_file(Path(__file__).with_name("tfc_tdf_default_model.py")),
            },
        },
    }
    report_path = output_dir / "onnx-export-report.json"
    json_write(report_path, report)
    print(json.dumps({"artifact": artifact, "parity": parity}, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
