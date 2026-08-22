#!/usr/bin/env python3
"""Render the four-song aggressive-target pilot on the frozen private set.

This is a full-song listening renderer for the original
``alpha-1.00-step-2048`` checkpoint.  It deliberately keeps the checkpoint's
original residual-vocals semantic: the neural output is subtracted from the
mixture spectrum and the resulting instrumental is written to PCM16 FLAC.

The source songs, checkpoint, and generated audio are local non-commercial
research artifacts.  This tool does not publish or copy any of them into a
release directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import render_inst3_objective_listening as listening
import run_inst3_aggressive_target as aggressive
import run_inst3_distill_pilot as pilot
import run_inst3_scale10_density as density


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_AGGRESSIVE_ROOT = ROOT / "data" / "musdb18-inst3-aggressive-target"
DEFAULT_CHECKPOINT_PAYLOAD = (
    DEFAULT_AGGRESSIVE_ROOT / "runs" / "alpha-1.00-step-2048.pt"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_AGGRESSIVE_ROOT / "listening-12" / "alpha-1.00-step-2048"
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-payload", type=Path, default=DEFAULT_CHECKPOINT_PAYLOAD
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload is not a mapping: {path}")
    if payload.get("format") != "local-inst3-aggressive-target":
        raise ValueError(f"Unexpected checkpoint format: {payload.get('format')!r}")
    if float(payload.get("alpha", -1.0)) != 1.0:
        raise ValueError(f"Expected alpha=1.0, got {payload.get('alpha')!r}")
    if int(payload.get("steps", -1)) != 2048:
        raise ValueError(f"Expected steps=2048, got {payload.get('steps')!r}")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint is missing state_dict")
    return payload


def render_or_load(
    *,
    source: np.ndarray,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    skipped = output_path.is_file() and not force
    timing: dict[str, Any] = {}
    if not skipped:
        audio, timing = listening.render_student(source, model, device)
        item = listening.write_flac(
            output_path, audio, listening.DEFAULT_CONFIG.sample_rate
        )
        del audio
    decoded, sample_rate = listening.load_audio(output_path)
    if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"Unexpected output sample rate: {output_path}")
    if decoded.shape != source.shape:
        raise ValueError(
            f"Output shape mismatch for {output_path}: {decoded.shape} != {source.shape}"
        )
    if not np.isfinite(decoded).all():
        raise ValueError(f"Output contains non-finite PCM: {output_path}")
    return {
        "file": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "sha256": pilot.sha256_file(output_path),
        "frames": int(decoded.shape[0]),
        "sampleRate": int(sample_rate),
        "channels": int(decoded.shape[1]),
        "semantic": "residual-vocals-to-instrumental",
        "skippedExisting": skipped,
        "windowCount": timing.get("windowCount"),
        "elapsedSeconds": time.perf_counter() - started,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    checkpoint = args.checkpoint.resolve()
    payload_path = args.checkpoint_payload.resolve()
    payload = load_payload(payload_path)
    songs = density.validate_private_songs(args.samples_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "render-report.json"
    report: dict[str, Any] = {
        "schema": "local-inst3-aggressive-target-listening@1",
        "status": "running",
        "semantic": "residual-vocals-to-instrumental",
        "format": "PCM16 FLAC",
        "songCount": len(songs),
        "outputCountExpected": len(songs),
        "studentConfig": dict(listening.DEFAULT_CONFIG.__dict__),
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "threads": args.threads,
        },
        "initialCheckpoint": {
            "file": str(checkpoint),
            "sha256": pilot.sha256_file(checkpoint),
        },
        "aggressiveCheckpoint": {
            "file": str(payload_path),
            "sha256": pilot.sha256_file(payload_path),
            "format": payload["format"],
            "alpha": payload["alpha"],
            "steps": payload["steps"],
            "learningRate": payload.get("learningRate"),
            "targetSemantic": payload.get("targetSemantic"),
        },
        "songs": {},
    }

    model = aggressive.load_checkpoint_model(checkpoint, payload_path, device)
    try:
        for index, song in enumerate(songs, start=1):
            name = song["name"]
            source_path = Path(song["file"])
            source, sample_rate = listening.load_audio(source_path)
            if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
                raise ValueError(f"Unexpected source sample rate: {source_path}")
            report["songs"][name] = {
                "source": {
                    **song,
                    "sampleRate": int(sample_rate),
                    "frames": int(source.shape[0]),
                    "durationSeconds": source.shape[0] / sample_rate,
                    "decodedFloat32Sha256": pilot.sha256_array(source),
                }
            }
            output_path = output_root / f"{name}.flac"
            metadata = render_or_load(
                source=source,
                output_path=output_path,
                model=model,
                device=device,
                force=args.force,
            )
            report["songs"][name]["output"] = metadata
            json_write(report_path, report)
            print(
                f"{index}/{len(songs)} {name}: {metadata['frames']} frames "
                f"({metadata['elapsedSeconds']:.2f}s)",
                flush=True,
            )
            del source
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["status"] = "completed"
    report["outputCount"] = len(report["songs"])
    json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "outputCount": report["outputCount"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
