#!/usr/bin/env python3
"""Run a local, non-commercial UVR Inst 3 -> TFC-TDF distillation pilot.

This is deliberately a small closed-loop experiment.  It decodes four selected
MUSDB18 songs, renders the Inst 3 teacher with the frozen MDX contract, then
compares two short warm-start training runs:

    S0: target = the MUSDB18 vocal stem
    S1: target = mixture - (Inst 3 instrumental output * 1.028)

The script is local-only.  It does not publish audio, checkpoints, or derived
weights.  All generated files default to the ignored ``.tmp`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import onnxruntime as ort

from mdx_reference import MdxParams, build_windows, istft_centered, stft_centered
from tfc_tdf_default_model import (
    DEFAULT_CONFIG,
    EXPECTED_CHECKPOINT_SHA256,
    TfcTdfNchwWrapper,
    TfcTdfNeuralCore,
    load_default_checkpoint,
    sha256_file,
)
from validate_tfc_tdf_default_audio import (
    istft_centered as student_istft_centered,
    stft_centered as student_stft_centered,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = ROOT / ".tmp" / "musdb18-inst3-pilot"
DEFAULT_ARCHIVE = Path(r"C:\Users\User\Downloads\musdb18.zip")
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

EXPECTED_TEACHER_SHA256 = (
    "2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb"
)
EXPECTED_TEACHER_BYTES = 66_759_214
TEACHER_PARAMS = MdxParams(
    sample_rate=44_100,
    n_fft=7_680,
    hop_length=1_024,
    dim_f=3_072,
    dim_t_power=8,
    trim=3_840,
)
TEACHER_OUTPUT_SCALE = 1.028

STREAM_NAMES = ("mixture", "drums", "bass", "other", "vocals")
SONG_SPECS = (
    ("train", "A Classic Education - NightOwl.stem.mp4"),
    ("train", "Actions - Devil's Words.stem.mp4"),
    ("calibration", "Aimee Norwich - Child.stem.mp4"),
    ("test", "AM Contra - Heart Peripheral.stem.mp4"),
)


@dataclass
class SongBundle:
    role: str
    source_path: Path
    slug: str
    mixture: np.ndarray
    vocals: np.ndarray
    instrumental: np.ndarray
    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    sample_rate: int
    teacher_instrumental: np.ndarray | None = None
    teacher_vocals: np.ndarray | None = None
    teacher_metadata: dict[str, Any] | None = None


@dataclass
class WindowRecord:
    song: SongBundle
    start: int
    length: int
    rms: float
    input_spec: np.ndarray | None = None
    true_spec: np.ndarray | None = None
    teacher_spec: np.ndarray | None = None

    def materialize(self) -> None:
        if self.input_spec is not None:
            return
        self.input_spec = student_window_spec(
            self.song.mixture, self.start, self.length
        )
        self.true_spec = student_window_spec(
            self.song.vocals, self.start, self.length
        )
        if self.song.teacher_vocals is not None:
            self.teacher_spec = student_window_spec(
                self.song.teacher_vocals, self.start, self.length
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--start-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=18.0)
    parser.add_argument("--train-windows-per-song", type=int, default=2)
    parser.add_argument("--eval-windows-per-song", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--force-decode", action="store_true")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Fail instead of silently falling back when CUDA EP is unavailable",
    )
    return parser.parse_args()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_array(value: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(value.astype("<f4", copy=False))
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value.removesuffix("-stem-mp4")


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {joined}\n{result.stderr[-4000:]}"
        )


def inspect_archive(archive: Path, expected_members: Iterable[str]) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        names = {info.filename for info in infos}
        missing = sorted(set(expected_members) - names)
        if missing:
            raise ValueError(f"MUSDB18 archive is missing entries: {missing}")
        return {
            "file": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "entryCount": len(infos),
            "uncompressedBytes": sum(info.file_size for info in infos),
            "expectedEntriesPresent": True,
        }


def ensure_raw_song(
    archive: Path,
    raw_root: Path,
    role: str,
    filename: str,
) -> Path:
    archive_role = "train" if role == "calibration" else role
    existing_train_path = raw_root / "train" / filename
    if role == "calibration" and existing_train_path.is_file():
        return existing_train_path
    destination = raw_root / role / filename
    if destination.is_file():
        return destination
    member = f"{archive_role}/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        try:
            info = handle.getinfo(member)
        except KeyError as error:
            raise FileNotFoundError(f"No {member} in {archive}") from error
        if Path(info.filename).name != filename or not info.filename.startswith(
            f"{archive_role}/"
        ):
            raise ValueError(f"Unsafe archive member: {info.filename}")
        with handle.open(info) as source, destination.open("wb") as target:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
    return destination


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if sample_rate != TEACHER_PARAMS.sample_rate:
        raise ValueError(f"Decoded {path} at {sample_rate} Hz, expected 44100 Hz")
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def decode_song(
    source: Path,
    role: str,
    decoded_root: Path,
    start_seconds: float,
    duration_seconds: float,
    force: bool,
) -> SongBundle:
    slug = slugify(source.name)
    segment_key = f"start-{start_seconds:.3f}-duration-{duration_seconds:.3f}"
    output_dir = decoded_root / segment_key
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{slug}-{name}.wav" for name in STREAM_NAMES]
    if force or not all(path.is_file() for path in paths):
        for stream_index, path in enumerate(paths):
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                f"0:{stream_index}",
                "-ss",
                f"{start_seconds:.6f}",
                "-t",
                f"{duration_seconds:.6f}",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_f32le",
                str(path),
            ]
            run_checked(command)

    decoded: dict[str, np.ndarray] = {}
    sample_rates: set[int] = set()
    for name, path in zip(STREAM_NAMES, paths):
        value, sample_rate = load_audio(path)
        decoded[name] = value
        sample_rates.add(sample_rate)
    if len(sample_rates) != 1:
        raise ValueError(f"Inconsistent sample rates for {source}: {sample_rates}")
    length = min(value.shape[0] for value in decoded.values())
    if length < DEFAULT_CONFIG.useful_samples:
        raise ValueError(f"Decoded segment is too short for one student window: {source}")
    decoded = {name: value[:length] for name, value in decoded.items()}
    instrumental = (
        decoded["drums"] + decoded["bass"] + decoded["other"]
    ).astype(np.float32)
    cache_path = output_dir / f"{slug}.npz"
    if force or not cache_path.is_file():
        np.savez_compressed(
            cache_path,
            mixture=decoded["mixture"],
            drums=decoded["drums"],
            bass=decoded["bass"],
            other=decoded["other"],
            vocals=decoded["vocals"],
            instrumental=instrumental,
        )
    return SongBundle(
        role=role,
        source_path=source,
        slug=slug,
        mixture=decoded["mixture"],
        vocals=decoded["vocals"],
        instrumental=instrumental,
        drums=decoded["drums"],
        bass=decoded["bass"],
        other=decoded["other"],
        sample_rate=next(iter(sample_rates)),
    )


def verify_teacher_contract(
    contract_path: Path,
    teacher_path: Path,
    teacher_tflite_path: Path,
) -> dict[str, Any]:
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source = contract["source"]
    dsp = contract["dsp"]
    tensor = contract["tensorContract"]
    actual_sha = sha256_file(teacher_path)
    if actual_sha.lower() != EXPECTED_TEACHER_SHA256:
        raise ValueError(f"Inst 3 source SHA-256 mismatch: {actual_sha}")
    if actual_sha.lower() != source["sha256"].lower():
        raise ValueError("Inst 3 source does not match contract source identity")
    if teacher_path.stat().st_size != EXPECTED_TEACHER_BYTES:
        raise ValueError(f"Unexpected Inst 3 source size: {teacher_path.stat().st_size}")
    expected_dsp = {
        "sampleRate": TEACHER_PARAMS.sample_rate,
        "nFft": TEACHER_PARAMS.n_fft,
        "hopLength": TEACHER_PARAMS.hop_length,
        "dimF": TEACHER_PARAMS.dim_f,
        "dimTPower": TEACHER_PARAMS.dim_t_power,
        "modelTimeFrames": TEACHER_PARAMS.dim_t,
        "modelOutputScale": TEACHER_OUTPUT_SCALE,
        "window": "periodic-hann",
    }
    for key, expected in expected_dsp.items():
        if dsp.get(key) != expected:
            raise ValueError(f"Inst 3 DSP contract mismatch for {key}: {dsp.get(key)}")
    expected_shape = [1, 3072, 256, 4]
    if tensor["input"]["shape"] != expected_shape or tensor["output"]["shape"] != expected_shape:
        raise ValueError("Inst 3 tensor contract shape mismatch")
    tflite_identity = {
        "file": str(teacher_tflite_path.resolve()),
        "present": teacher_tflite_path.is_file(),
    }
    if teacher_tflite_path.is_file():
        tflite_identity.update(
            {
                "bytes": teacher_tflite_path.stat().st_size,
                "sha256": sha256_file(teacher_tflite_path),
                "contractSha256": contract["artifact"]["sha256"],
            }
        )
        if tflite_identity["sha256"].lower() != contract["artifact"]["sha256"].lower():
            raise ValueError("Inst 3 TFLite artifact does not match contract")
    return {
        "contractId": contract["contractId"],
        "contractFile": str(contract_path.resolve()),
        "contractSha256": sha256_file(contract_path),
        "teacherSource": {
            "file": str(teacher_path.resolve()),
            "bytes": teacher_path.stat().st_size,
            "sha256": actual_sha,
            "attribution": source.get("attribution", []),
        },
        "teacherTflite": tflite_identity,
        "dsp": expected_dsp,
        "tensor": {
            "inputShape": expected_shape,
            "outputShape": expected_shape,
            "inputLayout": tensor["input"]["layout"],
            "outputLayout": tensor["output"]["layout"],
        },
        "passed": True,
    }


_CUDA_DLL_HANDLES: list[Any] = []
_LAST_CUDA_DLL_SETUP: dict[str, Any] = {}


def prepare_cuda_dlls() -> dict[str, Any]:
    """Make the PyTorch CUDA 12/cuDNN 9 runtime visible to ONNX Runtime."""
    global _LAST_CUDA_DLL_SETUP
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    result: dict[str, Any] = {
        "torchLib": str(torch_lib),
        "exists": torch_lib.is_dir(),
        "preloaded": False,
    }
    if not torch_lib.is_dir():
        _LAST_CUDA_DLL_SETUP = result
        return result
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload(directory=str(torch_lib))
        result["preloaded"] = True
    _LAST_CUDA_DLL_SETUP = result
    return result


def make_teacher_session(
    teacher_path: Path,
    threads: int,
    require_cuda: bool,
) -> tuple[ort.InferenceSession, list[str]]:
    cuda_dlls = prepare_cuda_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    available = ort.get_available_providers()
    if require_cuda and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu and "
            "make the PyTorch CUDA DLL directory visible before starting Python. "
            f"Available providers: {available}"
        )
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        str(teacher_path),
        sess_options=options,
        providers=providers,
    )
    input_detail = session.get_inputs()[0]
    output_detail = session.get_outputs()[0]
    expected_tail = [4, TEACHER_PARAMS.dim_f, TEACHER_PARAMS.dim_t]
    if list(input_detail.shape[1:]) != expected_tail:
        raise ValueError(f"Unexpected Inst 3 input shape: {input_detail.shape}")
    if list(output_detail.shape[1:]) != expected_tail:
        raise ValueError(f"Unexpected Inst 3 output shape: {output_detail.shape}")
    session_providers = list(session.get_providers())
    if require_cuda and "CUDAExecutionProvider" not in session_providers:
        raise RuntimeError(
            "Inst 3 session fell back without CUDAExecutionProvider. "
            f"Session providers: {session_providers}; DLL setup: {cuda_dlls}"
        )
    return session, session_providers


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        TEACHER_PARAMS.sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rawFloat32Sha256": sha256_array(audio),
    }


def render_teacher(
    song: SongBundle,
    session: ort.InferenceSession,
    providers: list[str],
    teacher_root: Path,
    force: bool,
) -> dict[str, Any]:
    output_npz = teacher_root / f"{song.slug}.npz"
    if output_npz.is_file() and not force:
        cached = np.load(output_npz)
        song.teacher_instrumental = np.ascontiguousarray(cached["instrumental"], dtype=np.float32)
        song.teacher_vocals = np.ascontiguousarray(cached["vocals"], dtype=np.float32)
        metadata_path = teacher_root / f"{song.slug}.json"
        return json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {
            "song": song.slug,
            "cached": True,
        }

    windows, pad = build_windows(song.mixture.T, TEACHER_PARAMS)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pieces: list[np.ndarray] = []
    window_timings: list[dict[str, float | int]] = []
    started_all = time.perf_counter()
    for index, window in enumerate(windows):
        dsp_started = time.perf_counter()
        model_input = stft_centered(window[None, ...], TEACHER_PARAMS)
        dsp_seconds = time.perf_counter() - dsp_started
        inference_started = time.perf_counter()
        model_output = session.run(
            [output_name],
            {input_name: np.ascontiguousarray(model_input, dtype=np.float32)},
        )[0]
        inference_seconds = time.perf_counter() - inference_started
        if tuple(model_output.shape) != (1, 4, TEACHER_PARAMS.dim_f, TEACHER_PARAMS.dim_t):
            raise ValueError(f"Unexpected Inst 3 output shape: {model_output.shape}")
        if not np.isfinite(model_output).all():
            raise ValueError(f"Inst 3 output contains non-finite values in window {index}")
        reconstructed = istft_centered(model_output.astype(np.float32), TEACHER_PARAMS)[0]
        useful = reconstructed[:, TEACHER_PARAMS.trim : -TEACHER_PARAMS.trim].T
        pieces.append(np.ascontiguousarray(useful, dtype=np.float32))
        window_timings.append(
            {
                "index": index,
                "stftSeconds": dsp_seconds,
                "inferenceSeconds": inference_seconds,
            }
        )
    instrumental = np.concatenate(pieces, axis=0)
    if pad:
        instrumental = instrumental[:-pad]
    instrumental = np.ascontiguousarray(
        instrumental[: song.mixture.shape[0]] * np.float32(TEACHER_OUTPUT_SCALE),
        dtype=np.float32,
    )
    if instrumental.shape != song.mixture.shape:
        raise ValueError(
            f"Teacher output length mismatch: {instrumental.shape} vs {song.mixture.shape}"
        )
    vocals = np.ascontiguousarray(song.mixture - instrumental, dtype=np.float32)
    reconstruction_error = float(np.max(np.abs(instrumental + vocals - song.mixture)))
    if reconstruction_error > 2e-6:
        raise ValueError(f"Teacher residual rule failed: {reconstruction_error}")
    song.teacher_instrumental = instrumental
    song.teacher_vocals = vocals
    teacher_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, instrumental=instrumental, vocals=vocals)
    artifacts = {
        "npz": {
            "file": str(output_npz),
            "bytes": output_npz.stat().st_size,
            "sha256": sha256_file(output_npz),
        },
        "instrumentalFlac": write_flac(
            teacher_root / f"{song.slug}-instrumental.flac", instrumental
        ),
        "vocalsFlac": write_flac(teacher_root / f"{song.slug}-vocals.flac", vocals),
    }
    metadata = {
        "song": song.slug,
        "source": str(song.source_path.resolve()),
        "sampleRate": song.sample_rate,
        "samples": int(song.mixture.shape[0]),
        "durationSeconds": song.mixture.shape[0] / song.sample_rate,
        "modelOutputStem": "instrumental",
        "residualStem": "vocals",
        "modelOutputScale": TEACHER_OUTPUT_SCALE,
        "windowCount": len(windows),
        "paddingSamples": int(pad),
        "providers": providers,
        "timing": {
            "totalSeconds": time.perf_counter() - started_all,
            "windows": window_timings,
        },
        "residualReconstructionMaxAbsError": reconstruction_error,
        "instrumentalRawFloat32Sha256": sha256_array(instrumental),
        "vocalsRawFloat32Sha256": sha256_array(vocals),
        "artifacts": artifacts,
        "passed": True,
    }
    json_write(teacher_root / f"{song.slug}.json", metadata)
    return metadata


def student_window_spec(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    window = np.zeros((DEFAULT_CONFIG.model_input_samples, 2), dtype=np.float32)
    segment = audio[start : start + length]
    window[DEFAULT_CONFIG.trim_samples : DEFAULT_CONFIG.trim_samples + len(segment)] = segment
    return np.ascontiguousarray(student_stft_centered(window), dtype=np.float32)[0]


def select_windows(song: SongBundle, count: int) -> list[WindowRecord]:
    if count <= 0:
        raise ValueError("window count must be positive")
    candidates: list[WindowRecord] = []
    useful = DEFAULT_CONFIG.useful_samples
    for start in range(0, song.mixture.shape[0], useful):
        length = min(useful, song.mixture.shape[0] - start)
        if length < useful // 3:
            continue
        segment = song.mixture[start : start + length]
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        candidates.append(WindowRecord(song, start, length, rms))
    candidates.sort(key=lambda item: (-item.rms, item.start))
    selected = candidates[: min(count, len(candidates))]
    selected.sort(key=lambda item: item.start)
    return selected


def make_model(
    checkpoint: Path, device: torch.device
) -> tuple[TfcTdfNchwWrapper, dict[str, Any]]:
    core, metadata = load_default_checkpoint(checkpoint)
    model = TfcTdfNchwWrapper(core)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, metadata


def train_variant(
    name: str,
    checkpoint: Path,
    records: list[WindowRecord],
    target_key: str,
    device: torch.device,
    output_path: Path,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No training records for {name}")
    model, _ = make_model(checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(steps):
        record = records[step % len(records)]
        record.materialize()
        target = record.true_spec if target_key == "true" else record.teacher_spec
        if target is None or record.input_spec is None:
            raise ValueError(f"Missing target for {name} at {record.song.slug}:{record.start}")
        input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
        target_tensor = torch.from_numpy(target[None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_tensor)
        loss = F.l1_loss(output, target_tensor)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu().item()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "local-inst3-distill-pilot",
            "variant": name,
            "target": target_key,
            "state_dict": model.state_dict(),
            "steps": steps,
            "learningRate": learning_rate,
        },
        output_path,
    )
    return {
        "variant": name,
        "target": target_key,
        "steps": steps,
        "learningRate": learning_rate,
        "history": history,
        "initialLoss": history[0]["loss"],
        "finalLoss": history[-1]["loss"],
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
        "checkpoint": {
            "file": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }, model


def energy_snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference64 = reference.astype(np.float64)
    error64 = (candidate - reference).astype(np.float64)
    signal = float(np.sum(reference64 * reference64))
    error = float(np.sum(error64 * error64))
    return 10.0 * math.log10(max(signal, 1e-20) / max(error, 1e-20))


def rms_dbfs(value: np.ndarray) -> float:
    power = float(np.mean(value.astype(np.float64) ** 2))
    return 10.0 * math.log10(max(power, 1e-20))


def vocal_projection_db(error: np.ndarray, vocals: np.ndarray) -> float:
    error_flat = error.astype(np.float64).reshape(-1)
    vocal_flat = vocals.astype(np.float64).reshape(-1)
    denominator = float(np.dot(vocal_flat, vocal_flat))
    if denominator <= 1e-20:
        return float("-inf")
    coefficient = float(np.dot(error_flat, vocal_flat) / denominator)
    projected = coefficient * vocal_flat
    return 10.0 * math.log10(
        max(float(np.dot(projected, projected)), 1e-20) / max(denominator, 1e-20)
    )


def evaluate_model(
    name: str,
    model: TfcTdfNchwWrapper,
    songs: list[SongBundle],
    records_by_song: dict[str, list[WindowRecord]],
    device: torch.device,
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    aggregate: dict[str, list[np.ndarray]] = {
        "mixture": [],
        "trueVocals": [],
        "trueInstrumental": [],
        "teacherVocals": [],
        "predictedVocals": [],
    }
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for song in songs:
            predictions: list[np.ndarray] = []
            truth_vocals: list[np.ndarray] = []
            truth_instrumental: list[np.ndarray] = []
            mixtures: list[np.ndarray] = []
            teacher_vocals: list[np.ndarray] = []
            for record in records_by_song[song.slug]:
                record.materialize()
                if record.input_spec is None or record.teacher_spec is None:
                    raise ValueError("Evaluation record is missing materialized data")
                output = model(torch.from_numpy(record.input_spec[None]).to(device))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                output_np = output.detach().cpu().numpy()
                reconstructed = student_istft_centered(output_np)
                segment = reconstructed[
                    DEFAULT_CONFIG.trim_samples : DEFAULT_CONFIG.trim_samples + record.length
                ]
                predictions.append(np.ascontiguousarray(segment, dtype=np.float32))
                mixtures.append(song.mixture[record.start : record.start + record.length])
                truth_vocals.append(song.vocals[record.start : record.start + record.length])
                truth_instrumental.append(
                    song.instrumental[record.start : record.start + record.length]
                )
                teacher_vocals.append(
                    song.teacher_vocals[record.start : record.start + record.length]
                )
            mix = np.concatenate(mixtures, axis=0)
            true_vocal = np.concatenate(truth_vocals, axis=0)
            true_instrumental = np.concatenate(truth_instrumental, axis=0)
            teacher_vocal = np.concatenate(teacher_vocals, axis=0)
            predicted_vocal = np.concatenate(predictions, axis=0)
            predicted_instrumental = mix - predicted_vocal
            error = predicted_instrumental - true_instrumental
            metrics = {
                "samples": int(mix.shape[0]),
                "vocalSdrDb": energy_snr_db(true_vocal, predicted_vocal),
                "teacherVocalSdrDb": energy_snr_db(teacher_vocal, predicted_vocal),
                "accompanimentSdrDb": energy_snr_db(
                    true_instrumental, predicted_instrumental
                ),
                "accompanimentErrorRmsDbfs": rms_dbfs(error),
                "accompanimentVocalProjectionDb": vocal_projection_db(error, true_vocal),
                "reconstructionMaxAbsError": float(
                    np.max(np.abs(predicted_vocal + predicted_instrumental - mix))
                ),
            }
            per_song[song.slug] = {
                "role": song.role,
                "windowStarts": [record.start for record in records_by_song[song.slug]],
                "windowRms": [record.rms for record in records_by_song[song.slug]],
                "metrics": metrics,
            }
            aggregate["mixture"].append(mix)
            aggregate["trueVocals"].append(true_vocal)
            aggregate["trueInstrumental"].append(true_instrumental)
            aggregate["teacherVocals"].append(teacher_vocal)
            aggregate["predictedVocals"].append(predicted_vocal)
    mix = np.concatenate(aggregate["mixture"], axis=0)
    true_vocal = np.concatenate(aggregate["trueVocals"], axis=0)
    true_instrumental = np.concatenate(aggregate["trueInstrumental"], axis=0)
    teacher_vocal = np.concatenate(aggregate["teacherVocals"], axis=0)
    predicted_vocal = np.concatenate(aggregate["predictedVocals"], axis=0)
    predicted_instrumental = mix - predicted_vocal
    error = predicted_instrumental - true_instrumental
    return {
        "variant": name,
        "perSong": per_song,
        "aggregate": {
            "samples": int(mix.shape[0]),
            "vocalSdrDb": energy_snr_db(true_vocal, predicted_vocal),
            "teacherVocalSdrDb": energy_snr_db(teacher_vocal, predicted_vocal),
            "accompanimentSdrDb": energy_snr_db(true_instrumental, predicted_instrumental),
            "accompanimentErrorRmsDbfs": rms_dbfs(error),
            "accompanimentVocalProjectionDb": vocal_projection_db(error, true_vocal),
            "reconstructionMaxAbsError": float(
                np.max(np.abs(predicted_vocal + predicted_instrumental - mix))
            ),
        },
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def evaluate_teacher(
    songs: list[SongBundle], records_by_song: dict[str, list[WindowRecord]]
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    for song in songs:
        if song.teacher_vocals is None or song.teacher_instrumental is None:
            raise ValueError("Teacher output missing")
        mix_parts: list[np.ndarray] = []
        true_vocal_parts: list[np.ndarray] = []
        true_inst_parts: list[np.ndarray] = []
        teacher_vocal_parts: list[np.ndarray] = []
        for record in records_by_song[song.slug]:
            begin, end = record.start, record.start + record.length
            mix_parts.append(song.mixture[begin:end])
            true_vocal_parts.append(song.vocals[begin:end])
            true_inst_parts.append(song.instrumental[begin:end])
            teacher_vocal_parts.append(song.teacher_vocals[begin:end])
        mix = np.concatenate(mix_parts, axis=0)
        true_vocal = np.concatenate(true_vocal_parts, axis=0)
        true_inst = np.concatenate(true_inst_parts, axis=0)
        teacher_vocal = np.concatenate(teacher_vocal_parts, axis=0)
        teacher_inst = mix - teacher_vocal
        error = teacher_inst - true_inst
        per_song[song.slug] = {
            "role": song.role,
            "metrics": {
                "vocalSdrDb": energy_snr_db(true_vocal, teacher_vocal),
                "accompanimentSdrDb": energy_snr_db(true_inst, teacher_inst),
                "accompanimentErrorRmsDbfs": rms_dbfs(error),
                "accompanimentVocalProjectionDb": vocal_projection_db(error, true_vocal),
                "reconstructionMaxAbsError": float(
                    np.max(np.abs(teacher_vocal + teacher_inst - mix))
                ),
            },
        }
    return {"perSong": per_song}


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0 or args.start_seconds < 0:
        raise ValueError("start and duration must be non-negative, duration must be positive")
    if args.train_windows_per_song <= 0 or args.eval_windows_per_song <= 0:
        raise ValueError("window counts must be positive")
    if args.steps <= 0 or args.learning_rate <= 0 or args.threads <= 0:
        raise ValueError("steps, learning rate, and threads must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pilot_root = args.pilot_root.resolve()
    raw_root = pilot_root / "raw"
    decoded_root = pilot_root / "decoded"
    teacher_root = pilot_root / "teacher"
    runs_root = pilot_root / "runs"
    reports_root = pilot_root / "reports"
    for path in (raw_root, decoded_root, teacher_root, runs_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)

    expected_members = [
        f"{'train' if role == 'calibration' else role}/{filename}"
        for role, filename in SONG_SPECS
    ]
    archive_info = inspect_archive(args.archive.resolve(), expected_members)
    source_paths: list[tuple[str, Path]] = []
    for role, filename in SONG_SPECS:
        source_paths.append(
            (role, ensure_raw_song(args.archive.resolve(), raw_root, role, filename))
        )

    contract_info = verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    session, teacher_providers = make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    bundles: list[SongBundle] = []
    for role, source in source_paths:
        bundles.append(
            decode_song(
                source,
                role,
                decoded_root,
                args.start_seconds,
                args.duration_seconds,
                args.force_decode,
            )
        )
    songs_by_slug = {song.slug: song for song in bundles}

    teacher_reports: dict[str, Any] = {}
    for song in bundles:
        teacher_reports[song.slug] = render_teacher(
            song,
            session,
            teacher_providers,
            teacher_root,
            args.force_teacher,
        )

    train_songs = [song for song in bundles if song.role == "train"]
    eval_songs = [song for song in bundles if song.role in {"calibration", "test"}]
    train_records: list[WindowRecord] = []
    eval_records: dict[str, list[WindowRecord]] = {}
    for song in train_songs:
        train_records.extend(select_windows(song, args.train_windows_per_song))
    for song in eval_songs:
        eval_records[song.slug] = select_windows(song, args.eval_windows_per_song)
    for record in train_records:
        record.materialize()
    for records in eval_records.values():
        for record in records:
            record.materialize()

    baseline_model, checkpoint_metadata = make_model(args.checkpoint.resolve(), device)
    baseline_eval = evaluate_model(
        "initial-checkpoint", baseline_model, eval_songs, eval_records, device
    )
    teacher_eval = evaluate_teacher(eval_songs, eval_records)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s0_report, s0_model = train_variant(
        "S0-supervised-vocals",
        args.checkpoint.resolve(),
        train_records,
        "true",
        device,
        runs_root / "s0-supervised-vocals.pt",
        args.steps,
        args.learning_rate,
    )
    s1_report, s1_model = train_variant(
        "S1-inst3-residual-distill",
        args.checkpoint.resolve(),
        train_records,
        "teacher",
        device,
        runs_root / "s1-inst3-residual-distill.pt",
        args.steps,
        args.learning_rate,
    )
    s0_eval = evaluate_model("S0-supervised-vocals", s0_model, eval_songs, eval_records, device)
    s1_eval = evaluate_model("S1-inst3-residual-distill", s1_model, eval_songs, eval_records, device)
    del s0_model, s1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    song_metadata = {
        song.slug: {
            "role": song.role,
            "source": str(song.source_path.resolve()),
            "sampleRate": song.sample_rate,
            "samples": int(song.mixture.shape[0]),
            "durationSeconds": song.mixture.shape[0] / song.sample_rate,
            "decodedPcmFloat32Sha256": {
                name: sha256_array(value)
                for name, value in {
                    "mixture": song.mixture,
                    "drums": song.drums,
                    "bass": song.bass,
                    "other": song.other,
                    "vocals": song.vocals,
                    "instrumental": song.instrumental,
                }.items()
            },
            "mixtureMinusStemSumRmsDbfs": rms_dbfs(
                song.mixture - (song.instrumental + song.vocals)
            ),
            "trainWindowStarts": [record.start for record in train_records if record.song.slug == song.slug],
            "evalWindowStarts": [record.start for record in eval_records.get(song.slug, [])],
        }
        for song in bundles
    }
    report = {
        "schemaVersion": 1,
        "status": "pilot-completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local pilot artifacts only; do not publish",
        },
        "split": {
            "train": [song.slug for song in train_songs],
            "calibration": [song.slug for song in bundles if song.role == "calibration"],
            "test": [song.slug for song in bundles if song.role == "test"],
            "startSeconds": args.start_seconds,
            "durationSeconds": args.duration_seconds,
        },
        "archive": archive_info,
        "contract": contract_info,
        "teacher": {
            "providers": teacher_providers,
            "onnxruntimeVersion": ort.__version__,
            "ortAvailableProviders": ort.get_available_providers(),
            "cudaRequired": args.require_teacher_cuda,
            "cudaDllSetup": _LAST_CUDA_DLL_SETUP,
            "parameters": {
                "sampleRate": TEACHER_PARAMS.sample_rate,
                "nFft": TEACHER_PARAMS.n_fft,
                "hopLength": TEACHER_PARAMS.hop_length,
                "dimF": TEACHER_PARAMS.dim_f,
                "modelTimeFrames": TEACHER_PARAMS.dim_t,
                "trim": TEACHER_PARAMS.trim,
                "outputScale": TEACHER_OUTPUT_SCALE,
            },
            "songs": teacher_reports,
        },
        "student": {
            "checkpoint": checkpoint_metadata["checkpoint"],
            "expectedCheckpointSha256": EXPECTED_CHECKPOINT_SHA256,
            "device": str(device),
            "cuda": {
                "available": torch.cuda.is_available(),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torchVersion": torch.__version__,
                "cudaVersion": torch.version.cuda,
            },
            "window": {
                "inputSamples": DEFAULT_CONFIG.model_input_samples,
                "trimSamplesPerSide": DEFAULT_CONFIG.trim_samples,
                "usefulSamples": DEFAULT_CONFIG.useful_samples,
                "featureOrder": [
                    "left.real",
                    "right.real",
                    "left.imag",
                    "right.imag",
                ],
            },
        },
        "songs": song_metadata,
        "training": {
            "trainWindowCount": len(train_records),
            "steps": args.steps,
            "learningRate": args.learning_rate,
            "loss": "mean-absolute-error in complex packed spectrum",
            "S0": s0_report,
            "S1": s1_report,
        },
        "evaluation": {
            "teacher": teacher_eval,
            "initialCheckpoint": baseline_eval,
            "S0": s0_eval,
            "S1": s1_eval,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    report_path = reports_root / "inst3-distill-pilot-report.json"
    json_write(report_path, report)
    summary = {
        "status": report["status"],
        "device": str(device),
        "teacherProviders": teacher_providers,
        "trainWindows": len(train_records),
        "teacher": teacher_eval,
        "initial": baseline_eval["aggregate"],
        "S0": s0_eval["aggregate"],
        "S1": s1_eval["aggregate"],
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
