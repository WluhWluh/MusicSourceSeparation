#!/usr/bin/env python3
"""Generate an NCHW float32 ONNX output for Android benchmark comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=existing_file, required=True)
    parser.add_argument("--input", type=existing_file, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    shape = (1, 4, args.height, args.width)
    expected_elements = int(np.prod(shape))
    model_input = np.fromfile(args.input, dtype="<f4")
    if model_input.size != expected_elements:
        raise ValueError(
            f"Expected {expected_elements} float32 values for {shape}, got {model_input.size}."
        )
    model_input = model_input.reshape(shape)

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: model_input})[0]
    if output.shape != shape:
        raise ValueError(f"Expected output shape {shape}, got {output.shape}.")
    output = np.asarray(output, dtype="<f4", order="C")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "model": str(args.model),
                "input": str(args.input),
                "output": str(args.output),
                "shape": list(output.shape),
                "bytes": args.output.stat().st_size,
                "sha256": digest,
                "nonFiniteCount": int(np.size(output) - np.count_nonzero(np.isfinite(output))),
                "absMean": float(np.mean(np.abs(output))),
                "rms": float(np.sqrt(np.mean(np.square(output, dtype=np.float64)))),
                "min": float(np.min(output)),
                "max": float(np.max(output)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
