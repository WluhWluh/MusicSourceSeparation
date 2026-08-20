#!/usr/bin/env python3
"""Render the objective-alignment models on arbitrary local listening tracks.

The MUSDB18 objective-alignment runner deliberately accepts only its frozen
song manifest.  This companion tool keeps that training boundary intact while
rendering the same listening set on unrelated local songs:

* 15.0 through 45.0 seconds
* 44.1 kHz stereo input and output
* the original 128-frame TFC-TDF contract
* five-hop trim on both sides of every student window
* PCM16 FLAC, without loudness normalization

The outputs are local, non-commercial research artifacts.  They must not be
uploaded with model weights or added to a release.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import run_inst3_distill_pilot as pilot
from mdx_reference import MdxParams, build_windows, istft_centered, stft_centered
from tfc_tdf_default_model import (
    DEFAULT_CONFIG,
    TfcTdfNchwWrapper,
    load_default_checkpoint,
    sha256_file,
)
from validate_tfc_tdf_default_audio import (
    istft_centered as student_istft_centered,
    stft_centered as student_stft_centered,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "inst3-objective-alignment-listening-extra"
DEFAULT_CHECKPOINT = ROOT / "models" / "tfc-tdf" / "source" / "vocals_epoch=891.ckpt"
DEFAULT_TEACHER = (
    ROOT
    / "models"
    / "uvr-mdx-candidates"
    / "trvlvr-all-public-uvr-models"
    / "UVR-MDX-NET-Inst_3.onnx"
)
DEFAULT_CONTRACT = Path(
    r"C:\Users\User\Documents\BSSModels\bss-tflite\contracts\v2\uvr_mdxnet_inst_3.json"
)
DEFAULT_TEACHER_TFLITE = Path(
    r"C:\Users\User\Documents\BSSModels\bss-tflite\artifacts\all-candidates-fp32\UVR-MDX-NET-Inst_3_static_float32.tflite"
)

DEFAULT_SONGS = {
    "yoru-ni-kakeru": ROOT / "data" / "samples" / "yoru_ni_kakeru.flac",
    "coldplay-tove-lo": ROOT / "data" / "samples" / "fun.mp3",
    "coast-town": ROOT / "data" / "samples" / "coast_town.mp3",
}

VARIANT_FILES = {
    "initial": None,
    "S0-ground-truth": "S0-ground-truth.pt",
    "S0-anchor": "S0-anchor.pt",
    "S0-vocal-projection": "S0-vocal-projection.pt",
    "S1-anchor-inst3-0.01": "S1-anchor-inst3-0.01.pt",
    "S1-anchor-inst3-0.03": "S1-anchor-inst3-0.03.pt",
    "S1-anchor-inst3-0.1": "S1-anchor-inst3-0.1.pt",
}

TEACHER_PARAMS = MdxParams(
    sample_rate=44_100,
    n_fft=7_680,
    hop_length=1_024,
    dim_f=3_072,
    dim_t_power=8,
    trim=3_840,
)
TEACHER_OUTPUT_SCALE = 1.028


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--song",
        action="append",
        metavar="NAME=PATH",
        help="Override or add a song input; may be supplied more than once",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--start-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-students", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_FILES),
        help="Comma-separated student variants (initial plus trained variants)",
    )
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Fail if the Inst 3 ONNX session cannot use CUDA",
    )
    return parser.parse_args(argv)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "song"


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_audio(path: Path, sample_rate: int = 44_100) -> tuple[np.ndarray, int]:
    """Decode through libsndfile, with an ffmpeg fallback for MP3 builds."""
    try:
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except RuntimeError:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        audio = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)
        source_rate = sample_rate
    if audio.shape[1] == 1:
        audio = np.repeat(audio, repeats=2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if source_rate != sample_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, sample_rate)
        audio = resample_poly(
            audio,
            sample_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frames": int(audio.shape[0]),
        "sampleRate": sample_rate,
        "channels": int(audio.shape[1]),
    }


def slice_listening_segment(
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    start = int(round(start_seconds * sample_rate))
    length = int(round(duration_seconds * sample_rate))
    end = start + length
    if start < 0 or end > audio.shape[0]:
        raise ValueError(
            f"Input is too short for {start_seconds:.3f}-{start_seconds + duration_seconds:.3f}s: "
            f"{audio.shape[0] / sample_rate:.3f}s"
        )
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def load_student_model(
    variant: str,
    checkpoint: Path,
    runs_root: Path,
    device: torch.device,
) -> TfcTdfNchwWrapper:
    core, _ = load_default_checkpoint(checkpoint)
    model = TfcTdfNchwWrapper(core)
    variant_file = VARIANT_FILES[variant]
    if variant_file is not None:
        path = runs_root / variant_file
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Missing state_dict in {path}")
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def student_window_input(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    window = np.zeros((DEFAULT_CONFIG.model_input_samples, 2), dtype=np.float32)
    segment = audio[start : start + length]
    begin = DEFAULT_CONFIG.trim_samples
    window[begin : begin + len(segment)] = segment
    return np.ascontiguousarray(student_stft_centered(window), dtype=np.float32)


def render_student(
    audio: np.ndarray,
    model: TfcTdfNchwWrapper,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    useful = DEFAULT_CONFIG.useful_samples
    output = np.empty_like(audio)
    window_count = math.ceil(audio.shape[0] / useful)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(window_count):
            start = index * useful
            length = min(useful, audio.shape[0] - start)
            input_spec = student_window_input(audio, start, length)
            input_tensor = torch.from_numpy(input_spec).to(device)
            predicted_instrumental = input_tensor - model(input_tensor)
            reconstructed = student_istft_centered(
                predicted_instrumental.detach().cpu().numpy()
            )
            begin = DEFAULT_CONFIG.trim_samples
            output[start : start + length] = reconstructed[begin : begin + length]
    if not np.isfinite(output).all():
        raise ValueError("Student output contains non-finite values")
    return output, {
        "windowCount": window_count,
        "windowUsefulSamples": useful,
        "modelInputSamples": DEFAULT_CONFIG.model_input_samples,
        "trimSamplesPerSide": DEFAULT_CONFIG.trim_samples,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def render_teacher(
    audio: np.ndarray,
    session: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    windows, pad = build_windows(audio.T, TEACHER_PARAMS)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pieces: list[np.ndarray] = []
    started = time.perf_counter()
    for window in windows:
        model_input = stft_centered(window[None, ...], TEACHER_PARAMS)
        model_output = session.run(
            [output_name],
            {input_name: np.ascontiguousarray(model_input, dtype=np.float32)},
        )[0]
        reconstructed = istft_centered(model_output.astype(np.float32), TEACHER_PARAMS)[0]
        pieces.append(
            np.ascontiguousarray(
                reconstructed[:, TEACHER_PARAMS.trim : -TEACHER_PARAMS.trim].T,
                dtype=np.float32,
            )
        )
    output = np.concatenate(pieces, axis=0)
    if pad:
        output = output[:-pad]
    output = np.ascontiguousarray(
        output[: audio.shape[0]] * np.float32(TEACHER_OUTPUT_SCALE), dtype=np.float32
    )
    if output.shape != audio.shape:
        raise ValueError(f"Teacher output shape mismatch: {output.shape} != {audio.shape}")
    return output, {
        "windowCount": int(windows.shape[0]),
        "paddingSamples": int(pad),
        "elapsedSeconds": time.perf_counter() - started,
        "providers": list(session.get_providers()),
        "modelOutputStem": "instrumental",
        "modelOutputScale": TEACHER_OUTPUT_SCALE,
    }


def parse_song_specs(values: list[str] | None) -> dict[str, Path]:
    songs = dict(DEFAULT_SONGS)
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"Expected non-empty NAME=PATH, got {value!r}")
        songs[safe_name(name.strip())] = Path(raw_path.strip()).expanduser()
    return songs


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    songs = parse_song_specs(args.song)
    for name, path in songs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANT_FILES]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if not variants and args.skip_students:
        raise ValueError("No work requested")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = ROOT / "data" / "musdb18-inst3-objective-alignment" / "runs"
    decoded: dict[str, np.ndarray] = {}
    source_metadata: dict[str, Any] = {}
    for name, path in songs.items():
        source, sample_rate = load_audio(path)
        segment = slice_listening_segment(
            source, sample_rate, args.start_seconds, args.duration_seconds
        )
        decoded[name] = segment
        source_metadata[name] = {
            "file": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "sourceSampleRate": sample_rate,
            "sourceFrames": int(source.shape[0]),
            "listeningStartSeconds": args.start_seconds,
            "listeningDurationSeconds": args.duration_seconds,
            "listeningFrames": int(segment.shape[0]),
            "listeningRawFloat32Sha256": pilot.sha256_array(segment),
        }

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experimentId": "inst3-objective-alignment-listening-extra@1",
        "status": "running",
        "spec": {
            "sampleRate": DEFAULT_CONFIG.sample_rate,
            "channels": 2,
            "format": "FLAC",
            "subtype": "PCM_16",
            "loudnessNormalization": False,
            "startSeconds": args.start_seconds,
            "durationSeconds": args.duration_seconds,
            "studentConfig": DEFAULT_CONFIG.__dict__,
            "teacherConfig": TEACHER_PARAMS.__dict__,
        },
        "sources": source_metadata,
        "students": {},
        "teacher": {},
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cudaAvailable": torch.cuda.is_available(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpointSha256": sha256_file(args.checkpoint),
        },
    }

    if not args.skip_students:
        for variant in variants:
            model = load_student_model(variant, args.checkpoint.resolve(), runs_root, device)
            try:
                for name, segment in decoded.items():
                    output_path = output_root / f"{name}-{safe_name(variant)}.flac"
                    if output_path.is_file() and not args.force:
                        continue
                    audio, timing = render_student(segment, model, device)
                    item = write_flac(output_path, audio, DEFAULT_CONFIG.sample_rate)
                    item.update({"variant": variant, **timing})
                    report["students"].setdefault(variant, {})[name] = item
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if not args.skip_teacher:
        if not args.teacher.is_file():
            raise FileNotFoundError(args.teacher)
        teacher_contract = pilot.verify_teacher_contract(
            args.contract.resolve(),
            args.teacher.resolve(),
            args.teacher_tflite.resolve(),
        )
        report["teacherContract"] = teacher_contract
        session, providers = pilot.make_teacher_session(
            args.teacher.resolve(), args.threads, args.require_teacher_cuda
        )
        try:
            for name, segment in decoded.items():
                output_path = output_root / f"{name}-teacher-inst3.flac"
                if output_path.is_file() and not args.force:
                    continue
                audio, timing = render_teacher(segment, session)
                item = write_flac(output_path, audio, DEFAULT_CONFIG.sample_rate)
                item.update({"teacher": "UVR-MDX-NET-Inst_3", **timing})
                report["teacher"][name] = item
        finally:
            del session

    report["status"] = "completed"
    report["outputRoot"] = str(output_root)
    json_write(output_root / "render-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
