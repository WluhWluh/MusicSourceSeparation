#!/usr/bin/env python3
"""Render the direct-instrumental W-D continuation for private listening.

This renderer keeps the frozen twelve-song listening set used by the other
Inst 3 experiments.  The W-D checkpoints are interpreted as direct
instrumental outputs; they must not go through the residual-vocal path used by
the original TFC-TDF checkpoint.

All outputs are local, non-commercial research artifacts under the ignored
``data`` tree.  They are not product-qualified or suitable for release.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_initialization_output_matrix as matrix
import run_inst3_scale10_density as density


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_PARENT_ROOT = ROOT / "data" / "musdb18-inst3-wd-continuation"
DEFAULT_MATRIX_ROOT = ROOT / "data" / "musdb18-inst3-init-output-matrix"
DEFAULT_OUTPUT_ROOT = DEFAULT_PARENT_ROOT / "listening-seed-891"
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
DEFAULT_TEACHER_ROOT = (
    ROOT / "data" / "musdb18-inst3-scale10-density" / "listening-full"
)
DEFAULT_MANIFEST = (
    ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
)
DEFAULT_CONTRACT = listening.DEFAULT_CONTRACT
DEFAULT_TEACHER = listening.DEFAULT_TEACHER
DEFAULT_TEACHER_TFLITE = listening.DEFAULT_TEACHER_TFLITE
DEFAULT_SEED = 891
DEFAULT_PASSES = (100, 125, 150, 200)


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--continuation-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--parent-experiment-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--passes",
        default=",".join(str(value) for value in DEFAULT_PASSES),
        help="W-D pass milestones to render",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--no-reuse-teacher", action="store_true")
    parser.add_argument("--require-teacher-cuda", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    passes = parse_csv_ints(args.passes)
    if tuple(sorted(set(passes))) != passes or any(value <= 0 for value in passes):
        raise ValueError("Pass milestones must be sorted, unique, and positive")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    if args.threads <= 0:
        raise ValueError("Threads must be positive")
    return passes


def checkpoint_path(continuation_root: Path, seed: int, pass_count: int) -> Path:
    updates = pass_count * 80
    return continuation_root / "runs" / f"seed-{seed}" / "W-D" / f"pass-{updates}.pt"


def source_checkpoint_path(
    continuation_root: Path,
    parent_experiment_root: Path,
    seed: int,
    pass_count: int,
) -> Path:
    if pass_count == 100:
        return (
            parent_experiment_root
            / "runs"
            / f"seed-{seed}"
            / "W-D"
            / "pass-8000.pt"
        )
    return checkpoint_path(continuation_root, seed, pass_count)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def load_checkpoint_payload(path: Path, seed: int, pass_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowed_formats = {
        "local-inst3-init-output-matrix-checkpoint@1",
        "local-inst3-wd-continuation-checkpoint@1",
    }
    if payload.get("format") not in allowed_formats:
        raise ValueError(f"Unexpected W-D checkpoint format: {path}")
    contract = payload.get("runContract", {}).get("contract", {})
    variant = contract.get("variant")
    variant_key = variant.get("key") if isinstance(variant, dict) else variant
    if variant_key != "W-D" or contract.get("seed") != seed:
        raise ValueError(f"Checkpoint is not W-D seed {seed}: {path}")
    if pass_count == 100:
        if (
            payload.get("format") != "local-inst3-init-output-matrix-checkpoint@1"
            or contract.get("passes") != 100
        ):
            raise ValueError(f"Checkpoint is not the frozen 100-pass W-D parent: {path}")
    elif (
        payload.get("format") != "local-inst3-wd-continuation-checkpoint@1"
        or contract.get("passes") != 200
        or contract.get("startPasses") != 100
    ):
        raise ValueError(f"Checkpoint does not belong to the 100-to-200 continuation: {path}")
    if int(payload.get("update", -1)) != pass_count * 80:
        raise ValueError(
            f"Checkpoint update mismatch for pass {pass_count}: {payload.get('update')}"
        )
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint has no state_dict: {path}")
    return payload


def freeze_for_inference(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_initial_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model, _ = pilot.make_model(checkpoint.resolve(), device)
    return freeze_for_inference(model, device)


def load_wd_model(
    checkpoint: Path,
    payload_path: Path,
    seed: int,
    pass_count: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint_payload(payload_path, seed, pass_count)
    model, initialization = matrix.make_matrix_model(
        matrix.VARIANT_SPECS["W-D"], checkpoint.resolve(), seed, device
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    freeze_for_inference(model, device)
    return model, {
        "pass": pass_count,
        "update": int(payload["update"]),
        "checkpoint": str(payload_path.resolve()),
        "checkpointSha256": sha256_file(payload_path),
        "runContractId": payload["runContract"]["id"],
        "initialization": initialization,
    }


def select_instrumental_spectrum(
    model_output: torch.Tensor,
    mixture_spectrum: torch.Tensor,
    semantic: str,
) -> torch.Tensor:
    if model_output.shape != mixture_spectrum.shape:
        raise ValueError(
            f"Model/mixture spectrum mismatch: {model_output.shape} != {mixture_spectrum.shape}"
        )
    if semantic == "direct-instrumental":
        return model_output
    if semantic == "residual-vocals":
        return mixture_spectrum - model_output
    raise ValueError(f"Unsupported student semantic: {semantic}")


def render_student(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    semantic: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    useful = listening.DEFAULT_CONFIG.useful_samples
    output = np.empty_like(audio)
    window_count = math.ceil(audio.shape[0] / useful)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(window_count):
            start = index * useful
            length = min(useful, audio.shape[0] - start)
            input_spec = listening.student_window_input(audio, start, length)
            input_tensor = torch.from_numpy(input_spec).to(device)
            model_output = model(input_tensor)
            predicted_instrumental = select_instrumental_spectrum(
                model_output, input_tensor, semantic
            )
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
        "semantic": semantic,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def write_or_load(
    output_path: Path,
    render: Callable[[], tuple[np.ndarray, dict[str, Any]]],
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    timing: dict[str, Any] = {}
    if force or not output_path.is_file():
        audio, timing = render()
        item = listening.write_flac(
            output_path, audio, listening.DEFAULT_CONFIG.sample_rate
        )
    else:
        item = {}
    decoded, sample_rate = listening.load_audio(output_path)
    if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"Unexpected sample rate: {output_path}")
    item.update(
        {
            "file": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "frames": int(decoded.shape[0]),
            "sampleRate": sample_rate,
            "channels": int(decoded.shape[1]),
            "skippedExisting": not force and not timing,
            **timing,
        }
    )
    return decoded, item


def copy_or_load_teacher(
    *,
    output_path: Path,
    cached_path: Path,
    source: np.ndarray,
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    reused = False
    if (force or not output_path.is_file()) and cached_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, output_path)
        reused = True
    decoded, sample_rate = listening.load_audio(output_path)
    if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"Unexpected teacher sample rate: {output_path}")
    if decoded.shape != source.shape:
        raise ValueError(
            f"Teacher frame/channel mismatch: {decoded.shape} != {source.shape}"
        )
    return decoded, {
        "file": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "frames": int(decoded.shape[0]),
        "sampleRate": sample_rate,
        "channels": int(decoded.shape[1]),
        "reusedFrom": str(cached_path.resolve()) if reused else None,
        "skippedExisting": not force and not reused,
        "semantic": "direct-instrumental",
    }


def source_metadata(source_path: Path, source: np.ndarray, sample_rate: int) -> dict[str, Any]:
    return {
        "file": str(source_path.resolve()),
        "bytes": source_path.stat().st_size,
        "sha256": sha256_file(source_path),
        "sampleRate": sample_rate,
        "frames": int(source.shape[0]),
        "channels": int(source.shape[1]),
        "durationSeconds": source.shape[0] / sample_rate,
        "decodedFloat32Sha256": pilot.sha256_array(source),
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    passes = validate_args(args)
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
    frozen_songs = density.validate_private_songs(args.samples_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "render-report.json"

    checkpoint_metadata: dict[str, Any] = {}
    checkpoint_payloads: dict[int, dict[str, Any]] = {}
    for pass_count in passes:
        path = source_checkpoint_path(
            args.continuation_root.resolve(),
            args.parent_experiment_root.resolve(),
            args.seed,
            pass_count,
        )
        checkpoint_payloads[pass_count] = load_checkpoint_payload(path, args.seed, pass_count)
        checkpoint_metadata[pass_count] = {
            "pass": pass_count,
            "update": pass_count * 80,
            "file": str(path.resolve()),
            "sha256": sha256_file(path),
            "runContractId": checkpoint_payloads[pass_count]["runContract"]["id"],
        }

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": "inst3-wd-listening-seed-891@1",
        "status": "running",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "teacher-derived local audio; not redistributed",
            "derivedWeights": "local W-D checkpoints; do not publish",
        },
        "spec": {
            "sampleRate": listening.DEFAULT_CONFIG.sample_rate,
            "channels": 2,
            "format": "FLAC",
            "subtype": "PCM_16",
            "studentConfig": listening.DEFAULT_CONFIG.__dict__,
            "studentWindowSemantic": {
                "initial-vocals": "residual-vocals converted to instrumental",
                "wd-pass": "direct-instrumental",
            },
        },
        "seed": args.seed,
        "passes": list(passes),
        "checkpoint": {
            "initial": str(checkpoint),
            "initialSha256": sha256_file(checkpoint),
            "wd": checkpoint_metadata,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "threads": args.threads,
        },
        "songs": {},
        "models": {},
        "teacher": {},
    }

    # Validate and decode each source once before rendering any model.
    decoded_sources: dict[str, tuple[Path, np.ndarray, int]] = {}
    for frozen in frozen_songs:
        name = frozen["name"]
        path = Path(frozen["file"])
        source, sample_rate = listening.load_audio(path)
        if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
            raise ValueError(f"Unexpected source sample rate: {path}")
        decoded_sources[name] = (path, source, sample_rate)
        report["songs"].setdefault(name, {})["source"] = source_metadata(
            path, source, sample_rate
        )

    teacher_session = None
    if not args.skip_teacher:
        teacher_contract = pilot.verify_teacher_contract(
            args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
        )
        report["teacherContract"] = teacher_contract
        if not args.no_reuse_teacher:
            cached_missing = [
                frozen["name"]
                for frozen in frozen_songs
                if not (args.teacher_root.resolve() / frozen["name"] / "teacher-inst3.flac").is_file()
            ]
        else:
            cached_missing = [frozen["name"] for frozen in frozen_songs]
        if cached_missing:
            teacher_session, providers = pilot.make_teacher_session(
                args.teacher.resolve(), args.threads, args.require_teacher_cuda
            )
            report["teacherProviders"] = providers
        else:
            report["teacherProviders"] = ["reused-local-cache"]

        for frozen in frozen_songs:
            name = frozen["name"]
            source_path, source, _ = decoded_sources[name]
            song_root = output_root / name
            target = song_root / "teacher-inst3.flac"
            cached = args.teacher_root.resolve() / name / "teacher-inst3.flac"
            if cached.is_file() and not args.no_reuse_teacher:
                teacher_audio, metadata = copy_or_load_teacher(
                    output_path=target,
                    cached_path=cached,
                    source=source,
                    force=args.force,
                )
            else:
                if teacher_session is None:
                    raise AssertionError("Teacher session was not created")
                teacher_audio, timing = listening.render_teacher(source, teacher_session)
                item = listening.write_flac(
                    target, teacher_audio, listening.DEFAULT_CONFIG.sample_rate
                )
                metadata = {
                    **item,
                    "frames": int(teacher_audio.shape[0]),
                    "sampleRate": listening.DEFAULT_CONFIG.sample_rate,
                    "channels": int(teacher_audio.shape[1]),
                    "semantic": "direct-instrumental",
                    **timing,
                }
            report["songs"][name]["teacher"] = metadata
            report["teacher"][name] = metadata
        if teacher_session is not None:
            del teacher_session
            teacher_session = None

    model_specs: list[tuple[str, str, Callable[[], tuple[torch.nn.Module, dict[str, Any]]]]] = [
        (
            "initial-vocals",
            "residual-vocals",
            lambda: (load_initial_model(checkpoint, device), {"semantic": "residual-vocals"}),
        )
    ]
    for pass_count in passes:
        path = source_checkpoint_path(
            args.continuation_root.resolve(),
            args.parent_experiment_root.resolve(),
            args.seed,
            pass_count,
        )
        model_specs.append(
            (
                f"wd-pass-{pass_count}",
                "direct-instrumental",
                lambda path=path, pass_count=pass_count: load_wd_model(
                    checkpoint, path, args.seed, pass_count, device
                ),
            )
        )

    for name, semantic, load_model in model_specs:
        print(f"rendering {name} ({semantic})", flush=True)
        model, model_metadata = load_model()
        report["models"][name] = {"semantic": semantic, **model_metadata}
        try:
            for index, frozen in enumerate(frozen_songs, start=1):
                song_name = frozen["name"]
                source_path, source, _ = decoded_sources[song_name]
                song_root = output_root / song_name
                output_path = song_root / f"{name}.flac"
                candidate_audio, metadata = write_or_load(
                    output_path,
                    lambda source=source, model=model, semantic=semantic: render_student(
                        source, model, device, semantic
                    ),
                    args.force,
                )
                metadata["semantic"] = semantic
                if "teacher" in report["songs"][song_name]:
                    teacher_path = output_root / song_name / "teacher-inst3.flac"
                    teacher_audio, _ = listening.load_audio(teacher_path)
                    metadata["diagnosticsVsTeacher"] = density.local_teacher_error_metrics(
                        source, teacher_audio, candidate_audio, listening.DEFAULT_CONFIG.sample_rate
                    )
                report["songs"][song_name].setdefault("candidates", {})[name] = metadata
                report_path.parent.mkdir(parents=True, exist_ok=True)
                json_write(report_path, report)
                print(
                    f"  {index}/{len(frozen_songs)} {song_name}: {metadata['frames']} frames",
                    flush=True,
                )
                del candidate_audio
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report["status"] = "completed"
    report["outputRoot"] = str(output_root)
    report["outputCount"] = sum(
        len(song.get("candidates", {})) + (1 if "teacher" in song else 0)
        for song in report["songs"].values()
    )
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "outputCount": report["outputCount"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
