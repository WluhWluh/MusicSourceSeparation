#!/usr/bin/env python3
"""Prepare a digest-pinned real-audio TFC-TDF Android tensor fixture."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from tfc_tdf_default_model import DEFAULT_CONFIG, sha256_file
from validate_tfc_tdf_default_audio import (
    load_audio,
    load_interpreter,
    metrics,
    sha256_array,
    stft_centered,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "tfc-tdf" / "default-compact"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "android-fixture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_MODEL_DIR / "tfc_tdf_default_vocals_core_fp32.onnx",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=DEFAULT_MODEL_DIR / "tfc_tdf_default_vocals_core_fp32.tflite",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def tensor_stats(value: np.ndarray) -> dict[str, float | int]:
    value64 = value.astype(np.float64)
    return {
        "elements": int(value.size),
        "minimum": float(np.min(value64)),
        "maximum": float(np.max(value64)),
        "mean": float(np.mean(value64)),
        "meanAbsolute": float(np.mean(np.abs(value64))),
        "rms": float(np.sqrt(np.mean(value64 * value64))),
    }


def write_tensor(path: Path, value: np.ndarray) -> dict[str, Any]:
    little_endian = np.ascontiguousarray(value.astype("<f4", copy=False))
    little_endian.tofile(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": "float32-le",
        "layout": "NCHW",
        "shape": list(value.shape),
        "stats": tensor_stats(value),
    }


def json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundled_model = output_dir / args.tflite.name
    if args.tflite.resolve() != bundled_model:
        shutil.copy2(args.tflite.resolve(), bundled_model)
    audio, sample_rate = load_audio(args.audio.resolve())
    start_sample = int(round(args.start_seconds * sample_rate))
    segment = audio[start_sample : start_sample + DEFAULT_CONFIG.useful_samples]
    if len(segment) != DEFAULT_CONFIG.useful_samples:
        raise ValueError("The source is too short for one complete useful window")
    wave = np.zeros((DEFAULT_CONFIG.model_input_samples, 2), dtype=np.float32)
    wave[
        DEFAULT_CONFIG.trim_samples : DEFAULT_CONFIG.trim_samples + len(segment)
    ] = segment
    input_nchw = stft_centered(wave)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.onnx.resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: input_nchw},
    )[0]

    interpreter = load_interpreter(args.tflite.resolve(), args.threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_nhwc = np.ascontiguousarray(np.transpose(input_nchw, (0, 2, 3, 1)))
    interpreter.set_tensor(int(input_detail["index"]), input_nhwc)
    interpreter.invoke()
    tflite_output = np.ascontiguousarray(
        np.transpose(interpreter.get_tensor(int(output_detail["index"])), (0, 3, 1, 2))
    )
    parity = metrics(onnx_output, tflite_output)
    if parity["snrDb"] < 90.0 or parity["maxAbsError"] > 1e-4:
        raise ValueError(f"Host fixture parity failed: {parity}")

    input_identity = write_tensor(output_dir / "input-nchw-f32.bin", input_nchw)
    onnx_identity = write_tensor(output_dir / "onnx-output-nchw-f32.bin", onnx_output)
    tflite_identity = write_tensor(
        output_dir / "tflite-output-nchw-f32.bin",
        tflite_output,
    )
    manifest = {
        "schemaVersion": 1,
        "fixtureId": "tfc_tdf_default_real_window_0@1",
        "candidateId": "tfc_tdf_default_vocals_core_fp32@tflite-1",
        "featureOrder": [
            "left.real",
            "right.real",
            "left.imag",
            "right.imag",
        ],
        "model": {
            "file": bundled_model.name,
            "bytes": bundled_model.stat().st_size,
            "sha256": sha256_file(bundled_model),
            "inputName": str(input_detail["name"]),
            "outputName": str(output_detail["name"]),
            "runtimeShapeNhwc": [1, 1025, 128, 4],
        },
        "sourceOnnx": {
            "file": args.onnx.name,
            "bytes": args.onnx.stat().st_size,
            "sha256": sha256_file(args.onnx),
        },
        "sourceAudio": {
            "file": args.audio.name,
            "bytes": args.audio.stat().st_size,
            "sha256": sha256_file(args.audio),
            "decodedPcmFloat32Sha256": sha256_array(audio),
            "sampleRate": sample_rate,
        },
        "window": {
            "startSample": start_sample,
            "inputSamples": DEFAULT_CONFIG.model_input_samples,
            "trimSamplesPerSide": DEFAULT_CONFIG.trim_samples,
            "usefulSamples": DEFAULT_CONFIG.useful_samples,
        },
        "input": input_identity,
        "goldens": {
            "onnx": onnx_identity,
            "hostTflite": tflite_identity,
        },
        "hostParity": {
            "onnxVsTflite": parity,
            "gate": {"minimumSnrDb": 90.0, "maximumAbsoluteError": 1e-4},
            "passed": True,
        },
        "environment": {
            "python": platform.python_version(),
            "generator": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    manifest_path = output_dir / "android-fixture-manifest.json"
    json_write(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifestSha256": sha256_file(manifest_path),
                "hostParity": parity,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
