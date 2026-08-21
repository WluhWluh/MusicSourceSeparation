#!/usr/bin/env python3
"""Render the corrected V-R hard-window checkpoints on the private songs.

The renderer uses the same twelve full-song private listening sources as the
scale-10 experiments.  Both checkpoints retain the original residual-vocals
semantic: the neural output is the removed vocal-like residual and the final
instrumental is the mixture spectrum minus that output.  Results are local
PCM16 FLAC research artifacts and are not product or release assets.
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
import run_inst3_distill_pilot as pilot
import run_inst3_scale10_density as density
import run_inst3_vr_hard_sampling as hard_sampling


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-sampling"
DEFAULT_OUTPUT_ROOT = DEFAULT_EXPERIMENT_ROOT / "listening-12"
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
DEFAULT_VARIANTS = ("V-R-U", "V-R-H25")
DEFAULT_STEP = 8000


def parse_variants(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or any(value not in hard_sampling.VARIANTS for value in values):
        raise ValueError(f"Expected variants from {hard_sampling.VARIANTS}, got {raw!r}")
    if len(set(values)) != len(values):
        raise ValueError("Variants must be unique")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
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


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def load_variant_model(
    *,
    checkpoint: Path,
    checkpoint_path: Path,
    variant: str,
    step: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-vr-hard-sampling-checkpoint@1":
        raise ValueError(f"Unexpected V-R checkpoint format: {checkpoint_path}")
    if payload.get("variant") != variant or int(payload.get("step", -1)) != step:
        raise ValueError(
            f"Checkpoint identity mismatch: expected {variant}@{step}, "
            f"got {payload.get('variant')}@{payload.get('step')}"
        )
    state_dict = payload.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing stateDict in {checkpoint_path}")
    model, initialization = pilot.make_model(checkpoint.resolve(), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "variant": variant,
        "step": step,
        "update": int(payload["step"]),
        "file": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "runContractId": payload.get("runContractId"),
        "checkpointSource": payload.get("checkpointSource"),
        "initialization": initialization,
    }


def render_student(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    useful = listening.DEFAULT_CONFIG.useful_samples
    output = np.empty_like(audio)
    window_count = (audio.shape[0] + useful - 1) // useful
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(window_count):
            start = index * useful
            length = min(useful, audio.shape[0] - start)
            input_spec = listening.student_window_input(audio, start, length)
            input_tensor = torch.from_numpy(input_spec).to(device)
            predicted_residual = model(input_tensor)
            predicted_instrumental = input_tensor - predicted_residual
            reconstructed = listening.student_istft_centered(
                predicted_instrumental.detach().cpu().numpy()
            )
            begin = listening.DEFAULT_CONFIG.trim_samples
            output[start : start + length] = reconstructed[begin : begin + length]
    if not np.isfinite(output).all():
        raise ValueError("Student output contains non-finite values")
    return output, {
        "windowCount": window_count,
        "windowUsefulSamples": useful,
        "modelInputSamples": listening.DEFAULT_CONFIG.model_input_samples,
        "trimSamplesPerSide": listening.DEFAULT_CONFIG.trim_samples,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def render_or_load(
    *,
    output_path: Path,
    source: Any,
    model: torch.nn.Module,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    skipped = output_path.is_file() and not force
    if not skipped:
        audio, timing = render_student(source, model, device)
        item = listening.write_flac(output_path, audio, listening.DEFAULT_CONFIG.sample_rate)
        item["renderElapsedSeconds"] = timing["elapsedSeconds"]
        item["windowCount"] = timing["windowCount"]
        del audio
    decoded, sample_rate = listening.load_audio(output_path)
    if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"Unexpected sample rate: {output_path}")
    if not torch.isfinite(torch.from_numpy(decoded)).all():
        raise ValueError(f"Non-finite decoded output: {output_path}")
    return {
        "file": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "frames": int(decoded.shape[0]),
        "sampleRate": sample_rate,
        "channels": int(decoded.shape[1]),
        "semantic": "residual-vocals-to-instrumental",
        "skippedExisting": skipped,
        "elapsedSeconds": time.perf_counter() - started,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    variants = parse_variants(args.variants)
    if args.step <= 0 or args.threads <= 0:
        raise ValueError("step and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    songs = density.validate_private_songs(args.samples_root.resolve())
    experiment_root = args.experiment_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "render-report.json"
    report: dict[str, Any] = {
        "schema": "local-inst3-vr-hard-sampling-listening@1",
        "status": "running",
        "experimentRoot": str(experiment_root),
        "outputRoot": str(output_root),
        "step": args.step,
        "variants": list(variants),
        "semantic": "residual-vocals-to-instrumental",
        "format": "PCM16 FLAC",
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
            "sha256": sha256_file(checkpoint),
        },
        "songs": {},
        "models": {},
    }

    decoded_sources: dict[str, Any] = {}
    for song in songs:
        source_path = Path(song["file"])
        source, sample_rate = listening.load_audio(source_path)
        if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
            raise ValueError(f"Unexpected source sample rate: {source_path}")
        decoded_sources[song["name"]] = source
        report["songs"][song["name"]] = {
            "source": {
                **song,
                "sampleRate": sample_rate,
                "frames": int(source.shape[0]),
                "durationSeconds": source.shape[0] / sample_rate,
                "decodedFloat32Sha256": pilot.sha256_array(source),
            },
            "candidates": {},
        }

    for variant in variants:
        checkpoint_path = experiment_root / "runs" / variant / f"step-{args.step}.pt"
        model, metadata = load_variant_model(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            variant=variant,
            step=args.step,
            device=device,
        )
        report["models"][variant] = metadata
        try:
            for index, song in enumerate(songs, start=1):
                name = song["name"]
                output_path = output_root / variant / f"{name}.flac"
                candidate_metadata = render_or_load(
                    output_path=output_path,
                    source=decoded_sources[name],
                    model=model,
                    device=device,
                    force=args.force,
                )
                report["songs"][name]["candidates"][variant] = candidate_metadata
                json_write(report_path, report)
                print(
                    f"{variant} {index}/{len(songs)} {name}: "
                    f"{candidate_metadata['frames']} frames",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report["status"] = "completed"
    report["outputCount"] = sum(
        len(song["candidates"]) for song in report["songs"].values()
    )
    json_write(report_path, report)
    print(
        json.dumps(
            {"status": report["status"], "report": str(report_path), "outputCount": report["outputCount"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
