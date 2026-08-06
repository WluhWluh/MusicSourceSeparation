#!/usr/bin/env python3
"""Run a deterministic CPU smoke test for HTDemucs neural-core ONNX graphs."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import onnxruntime as ort


WAVEFORM_SHAPE = (1, 2, 343_980)
SPECTRUM_SHAPE = (1, 4, 2_048, 336)


def parse_model(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("model must use LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--waveform", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "byteSize": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_raw(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected_values = int(np.prod(shape, dtype=np.int64))
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected_values:
        raise ValueError(f"Unexpected value count for {path}: {values.size}")
    return values.reshape(shape)


def array_report(value: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(value)
    finite_count = int(np.count_nonzero(finite))
    if finite_count != value.size:
        minimum = maximum = mean = rms = None
    else:
        value64 = value.astype(np.float64, copy=False)
        minimum = float(value64.min())
        maximum = float(value64.max())
        mean = float(value64.mean())
        rms = float(np.sqrt(np.mean(np.square(value64))))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "valueCount": int(value.size),
        "finiteValueCount": finite_count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "rootMeanSquare": rms,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def compare_outputs(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shape mismatch: {reference.shape} != {candidate.shape}")
    left = reference.astype(np.float64, copy=False)
    right = candidate.astype(np.float64, copy=False)
    delta = left - right
    error_energy = float(np.dot(delta.reshape(-1), delta.reshape(-1)))
    reference_energy = float(np.dot(left.reshape(-1), left.reshape(-1)))
    return {
        "comparedValueCount": int(left.size),
        "changedValueCount": int(np.count_nonzero(delta)),
        "maximumAbsoluteError": float(np.max(np.abs(delta))),
        "rootMeanSquareDelta": float(np.sqrt(np.mean(np.square(delta)))),
        "referenceToDeltaSnrDb": (
            float(10.0 * np.log10(reference_energy / error_energy))
            if error_energy
            else None
        ),
    }


def session_options(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.warmup < 0 or args.runs <= 0:
        raise ValueError("Invalid threads, warmup, or runs")

    waveform_path = args.waveform.resolve()
    spectrum_path = args.spectrum.resolve()
    waveform = load_raw(waveform_path, WAVEFORM_SHAPE)
    spectrum = load_raw(spectrum_path, SPECTRUM_SHAPE)
    feeds = {"input": waveform, "x": spectrum}

    models: OrderedDict[str, Any] = OrderedDict()
    retained_outputs: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
    for label, unresolved_path in args.model:
        path = unresolved_path.resolve()
        prepare_start = time.perf_counter()
        session = ort.InferenceSession(
            str(path),
            sess_options=session_options(args.threads),
            providers=["CPUExecutionProvider"],
        )
        prepare_ms = (time.perf_counter() - prepare_start) * 1_000.0
        input_names = [value.name for value in session.get_inputs()]
        output_names = [value.name for value in session.get_outputs()]
        if input_names != ["input", "x"] or output_names != ["output", "add_67"]:
            raise ValueError(
                f"Unexpected ONNX ABI for {label}: inputs={input_names} outputs={output_names}"
            )
        for _ in range(args.warmup):
            session.run(output_names, feeds)
        elapsed_ms = []
        outputs = None
        for _ in range(args.runs):
            start = time.perf_counter()
            outputs = session.run(output_names, feeds)
            elapsed_ms.append((time.perf_counter() - start) * 1_000.0)
        if outputs is None:
            raise AssertionError("No inference output")
        named_outputs = {
            name: np.asarray(value) for name, value in zip(output_names, outputs)
        }
        retained_outputs[label] = named_outputs
        models[label] = {
            "identity": identity(path),
            "providers": session.get_providers(),
            "prepareMs": prepare_ms,
            "warmupCount": args.warmup,
            "timedRunCount": args.runs,
            "inferenceMs": elapsed_ms,
            "inferenceMeanMs": statistics.fmean(elapsed_ms),
            "inferenceMedianMs": statistics.median(elapsed_ms),
            "outputs": {
                name: array_report(value) for name, value in named_outputs.items()
            },
        }

    labels = list(retained_outputs)
    comparisons = []
    reference_label = labels[0]
    for candidate_label in labels[1:]:
        comparisons.append(
            {
                "reference": reference_label,
                "candidate": candidate_label,
                "outputs": {
                    name: compare_outputs(
                        retained_outputs[reference_label][name],
                        retained_outputs[candidate_label][name],
                    )
                    for name in retained_outputs[reference_label]
                },
            }
        )

    report = {
        "schemaVersion": 1,
        "scope": "deterministic fixed-window ONNX CPU smoke; not a quality evaluation",
        "runtime": {
            "onnxRuntime": ort.__version__,
            "numpy": np.__version__,
            "threads": args.threads,
        },
        "inputs": {
            "waveform": {**identity(waveform_path), "shape": list(WAVEFORM_SHAPE)},
            "spectrum": {**identity(spectrum_path), "shape": list(SPECTRUM_SHAPE)},
        },
        "models": models,
        "comparisons": comparisons,
        "tool": identity(Path(__file__).resolve()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
