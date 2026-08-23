#!/usr/bin/env python3
"""Render the original 891 checkpoint under the static F24/4-hop contract.

This renderer uses the already validated compact-24 ONNX export of the
original ``vocals_epoch=891.ckpt``.  It writes full-song vocals and
instrumental FLAC files for the frozen twelve-song private listening set using
continuous-context overlap-save assembly.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

import render_inst3_objective_listening as listening
import run_inst3_scale10_density as density
import run_inst3_distill_pilot as pilot
from tfc_tdf_short_window import ShortWindowContract, render_with_backend
from tfc_tdf_short_window import sha256_array, sha256_file
from tfc_tdf_short_window import write_flac


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
DEFAULT_ONNX = (
    ROOT
    / "models"
    / "tfc-tdf"
    / "compact-24"
    / "tfc_tdf_default_vocals_core_fp32_f24.onnx"
)
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "tfc-tdf" / "source" / "vocals_epoch=891.ckpt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-f24-original"
EXPECTED_CHECKPOINT_SHA256 = "101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d"
EXPECTED_ONNX_SHA256 = "ab71803120d709c3498635dcada1aeed925dc113531bd764a3552615e24bb7b1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_backend(path: Path, threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path.resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_detail = session.get_inputs()[0]
    output_detail = session.get_outputs()[0]
    expected = [1, 4, 1025, 24]
    if [int(value) for value in input_detail.shape] != expected:
        raise ValueError(f"Unexpected F24 input shape: {input_detail.shape}")
    if [int(value) for value in output_detail.shape] != expected:
        raise ValueError(f"Unexpected F24 output shape: {output_detail.shape}")

    def run(value: np.ndarray) -> np.ndarray:
        return session.run(
            [output_detail.name],
            {input_detail.name: np.ascontiguousarray(value, dtype=np.float32)},
        )[0]

    return run, session


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    onnx_path = args.onnx.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not onnx_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Original checkpoint or F24 ONNX is missing")
    checkpoint_sha = sha256_file(checkpoint_path)
    onnx_sha = sha256_file(onnx_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected original checkpoint hash: {checkpoint_sha}")
    if onnx_sha != EXPECTED_ONNX_SHA256:
        raise ValueError(f"Unexpected F24 ONNX hash: {onnx_sha}")

    songs = density.validate_private_songs(args.samples_root.resolve())
    contract = ShortWindowContract(
        num_frames=24,
        left_context_hops=4,
        right_context_hops=4,
    )
    backend, session = make_backend(onnx_path, args.threads)
    output_root = args.output_root.resolve()
    listening_root = output_root / "listening-12" / "original-vocals-epoch-891-f24-4hop"
    listening_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "local-inst3-tfc-original-f24-listening@1",
        "status": "running",
        "model": {
            "checkpoint": {
                "file": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_sha,
            },
            "onnx": {
                "file": str(onnx_path),
                "bytes": onnx_path.stat().st_size,
                "sha256": onnx_sha,
                "inputShape": [1, 4, 1025, 24],
                "outputShape": [1, 4, 1025, 24],
            },
        },
        "contract": contract.as_dict(),
        "assembly": "continuous-context-overlap-save",
        "semantic": "original-vocals-residual; instrumental=mixture-vocals",
        "runtime": {
            "providers": session.get_providers(),
            "threads": args.threads,
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "onnxruntime": package_version("onnxruntime"),
        },
        "songs": {},
    }
    try:
        for index, song in enumerate(songs, start=1):
            source, sample_rate = listening.load_audio(Path(song["file"]))
            if sample_rate != contract.sample_rate:
                raise ValueError(f"Unexpected sample rate for {song['name']}: {sample_rate}")
            started = time.perf_counter()
            rendered = render_with_backend(
                source,
                contract,
                backend,
                assembly_mode="continuous",
            )
            vocals = np.ascontiguousarray(rendered.audio, dtype=np.float32)
            instrumental = np.ascontiguousarray(source - vocals, dtype=np.float32)
            if not np.isfinite(vocals).all() or not np.isfinite(instrumental).all():
                raise ValueError(f"Non-finite output for {song['name']}")
            vocals_path = listening_root / f"{song['name']}-vocals.flac"
            instrumental_path = listening_root / f"{song['name']}-instrumental.flac"
            skipped = (
                vocals_path.is_file()
                and instrumental_path.is_file()
                and not args.force
            )
            if not skipped:
                write_flac(vocals_path, vocals, sample_rate)
                write_flac(instrumental_path, instrumental, sample_rate)
            report["songs"][song["name"]] = {
                "source": {
                    **song,
                    "sampleRate": sample_rate,
                    "frames": int(source.shape[0]),
                    "decodedFloat32Sha256": sha256_array(source),
                },
                "render": {
                    "windowCount": rendered.window_count,
                    "boundaries": len(rendered.boundaries),
                    "elapsedSeconds": rendered.elapsed_seconds,
                    "wallSeconds": time.perf_counter() - started,
                },
                "outputs": {
                    "vocals": {
                        "file": str(vocals_path.resolve()),
                        "bytes": vocals_path.stat().st_size,
                        "sha256": sha256_file(vocals_path),
                        "frames": int(vocals.shape[0]),
                        "skippedExisting": skipped,
                    },
                    "instrumental": {
                        "file": str(instrumental_path.resolve()),
                        "bytes": instrumental_path.stat().st_size,
                        "sha256": sha256_file(instrumental_path),
                        "frames": int(instrumental.shape[0]),
                        "skippedExisting": skipped,
                    },
                },
            }
            json_write(output_root / "render-report.json", report)
            print(f"rendered {index}/{len(songs)}: {song['name']}", flush=True)
            del source, vocals, instrumental, rendered

    finally:
        del session
    report["status"] = "completed"
    report["outputCount"] = sum(
        len(value["outputs"]) for value in report["songs"].values()
    )
    json_write(output_root / "render-report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str((output_root / "render-report.json").resolve()),
                "listeningRoot": str(listening_root.resolve()),
                "songCount": len(report["songs"]),
                "outputCount": report["outputCount"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
