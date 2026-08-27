#!/usr/bin/env python3
"""Render an FMA S/R continuation checkpoint on the frozen private set.

The checkpoint keeps the residual-vocals output semantic.  This renderer uses
the same continuous 128-frame overlap-save assembly as the training/evaluation
runner and writes both the residual and derived instrumental tracks as local
PCM16 FLAC research artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_fma_sr_event_only_continuation as continuation
import run_inst3_scale10_density as density
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "data"
    / "modern-song-fma-sr-event-only-continuation"
    / "extensions"
    / "from-step-5120"
    / "runs"
    / "step-5648.pt"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "modern-song-fma-sr-event-only-continuation"
    / "extensions"
    / "from-step-5120"
    / "listening-12"
    / "S-R-event-only-pass-13"
)
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
SAMPLE_RATE = 44_100
UNIFORM_FULL_TARGET_CHECKPOINT_FORMAT = (
    "local-inst3-fma-uniform-full-target-continuation-checkpoint@1"
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-songs", type=int)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def load_model(
    checkpoint_path: Path,
    architecture_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    supported_formats = {
        continuation.CHECKPOINT_FORMAT,
        "local-inst3-fma-s-leakage-survey-continuation-checkpoint@1",
        UNIFORM_FULL_TARGET_CHECKPOINT_FORMAT,
    }
    if payload.get("format") not in supported_formats:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    state_dict = payload.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing stateDict in {checkpoint_path}")
    model, architecture = pilot.make_model(architecture_checkpoint.resolve(), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "file": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "format": payload.get("format"),
        "variant": payload.get("variant"),
        "step": int(payload.get("step", -1)),
        "continuationPass": int(payload.get("continuationPass", -1)),
        "baseStep": int(payload.get("baseStep", -1)),
        "selectionSha256": payload.get("selectionSha256"),
        "scheduleSha256": payload.get("scheduleSha256"),
        "architecture": architecture["checkpoint"],
    }


def write_candidate(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected stereo audio, got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio before write: {path}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped_samples = int(np.count_nonzero(np.abs(audio) > 1.0))
    item = listening.write_flac(path, audio, SAMPLE_RATE)
    decoded, rate = listening.load_audio(path)
    if rate != SAMPLE_RATE or decoded.shape != audio.shape:
        raise ValueError(f"Decoded output mismatch: {path}")
    if not np.isfinite(decoded).all():
        raise ValueError(f"Non-finite decoded audio: {path}")
    item.update(
        {
            "peakBeforePcm16Clip": peak,
            "samplesAboveFullScaleBeforeWrite": clipped_samples,
            "decodedPeak": float(np.max(np.abs(decoded))) if decoded.size else 0.0,
        }
    )
    return item


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("threads and inference-batch-size must be positive")
    if args.max_songs is not None and args.max_songs <= 0:
        raise ValueError("max-songs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    songs = density.validate_private_songs(args.samples_root.resolve())
    if args.max_songs is not None:
        songs = songs[: args.max_songs]
    if len(songs) != 12 and args.max_songs is None:
        raise ValueError(f"Expected the complete 12-song private set, got {len(songs)}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "render-report.json"
    report: dict[str, Any] = {
        "schema": "local-inst3-fma-sr-continuation-private-listening@1",
        "status": "running",
        "semantic": "residual-vocals-to-instrumental",
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "format": "PCM16 FLAC",
        "checkpoint": {},
        "runtime": {
            "device": str(device),
            "threads": args.threads,
            "inferenceBatchSize": args.inference_batch_size,
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "songs": {},
    }
    model, checkpoint_meta = load_model(checkpoint_path, args.architecture_checkpoint, device)
    if checkpoint_meta["format"] == UNIFORM_FULL_TARGET_CHECKPOINT_FORMAT:
        report["schema"] = "local-inst3-fma-uniform-full-target-continuation-private-listening@1"
    report["checkpoint"] = checkpoint_meta
    started = time.perf_counter()
    try:
        for index, song in enumerate(songs, start=1):
            name = song["name"]
            source_path = Path(song["file"])
            source, sample_rate = listening.load_audio(source_path)
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f"Unexpected sample rate: {source_path}")
            print(f"render {index}/{len(songs)}: {name}", flush=True)
            candidate_report: dict[str, Any] = {
                "source": {
                    **song,
                    "sampleRate": sample_rate,
                    "frames": int(source.shape[0]),
                    "durationSeconds": source.shape[0] / SAMPLE_RATE,
                }
            }
            residual, timing = continuous.render_student(
                model,
                source,
                CONTRACT,
                device,
                args.inference_batch_size,
            )
            instrumental = np.ascontiguousarray(source - residual, dtype=np.float32)
            if residual.shape != source.shape or instrumental.shape != source.shape:
                raise ValueError(f"Shape mismatch for {name}")
            candidate_report["render"] = timing
            candidate_report["residual"] = write_candidate(
                output_root / f"{name}-residual.flac", residual
            )
            candidate_report["instrumental"] = write_candidate(
                output_root / f"{name}-instrumental.flac", instrumental
            )
            report["songs"][name] = candidate_report
            report["completedSongs"] = index
            report["elapsedSeconds"] = time.perf_counter() - started
            json_write(report_path, report)
            del source, residual, instrumental
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "completed"
    report["songCount"] = len(songs)
    report["elapsedSeconds"] = time.perf_counter() - started
    json_write(report_path, report)
    print(
        json.dumps(
            {"status": report["status"], "songs": report["songCount"], "report": str(report_path)},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
