#!/usr/bin/env python3
"""Run a local, non-commercial UVR Inst 3 -> TFC-TDF distillation pilot.

This is deliberately a small closed-loop experiment. It reads four songs from
the frozen MUSDB18 Stage 0.5 split, renders the Inst 3 teacher with the frozen
MDX contract, then compares two 128-frame instrumental-output students:

    S0: ground-truth instrumental loss
    S1: ground-truth instrumental loss + 0.10 * Inst 3 soft-target loss

Both students expose ``mixture - vocal_estimator(mixture)`` as their direct
instrumental output. This preserves the useful initialization of the existing
vocals checkpoint while freezing a distinct instrumental student contract.

The script is local-only. It does not publish audio, checkpoints, or derived
weights. All generated files default to the ignored ``data`` tree.
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
from dataclasses import dataclass
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
    load_default_checkpoint,
    sha256_file,
)
from validate_tfc_tdf_default_audio import (
    istft_centered as student_istft_centered,
    stft_centered as student_stft_centered,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = ROOT / "data" / "musdb18-inst3-student-pilot"
DEFAULT_ARCHIVE = Path(r"C:\Users\User\Downloads\musdb18.zip")
DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "musdb18-inst3-oracle"
    / "musdb18-inst3-oracle-manifest.json"
)
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
PILOT_MEMBERS = (
    "train/The So So Glos - Emergency.stem.mp4",
    "train/Dark Ride - Burning Bridges.stem.mp4",
    "train/Lushlife - Toynbee Suite.stem.mp4",
    "train/Triviul - Dorothy.stem.mp4",
)


@dataclass
class SongBundle:
    role: str
    member: str
    source_sha256: str
    source_path: Path
    slug: str
    mixture_encoded: np.ndarray
    mixture_gt: np.ndarray
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
    vocal_rms: float
    input_spec: np.ndarray | None = None
    true_instrumental_spec: np.ndarray | None = None
    teacher_instrumental_spec: np.ndarray | None = None

    def materialize(self) -> None:
        if self.input_spec is not None:
            return
        self.input_spec = student_window_spec(
            self.song.mixture_gt, self.start, self.length
        )
        self.true_instrumental_spec = student_window_spec(
            self.song.instrumental, self.start, self.length
        )
        if self.song.teacher_instrumental is not None:
            self.teacher_instrumental_spec = student_window_spec(
                self.song.teacher_instrumental, self.start, self.length
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--start-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--train-windows-per-song", type=int, default=4)
    parser.add_argument("--eval-windows-per-song", type=int, default=5)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--teacher-weight", type=float, default=0.10)
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


def load_frozen_pilot_entries(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifestId") != "musdb18-inst3-oracle-split@1":
        raise ValueError(f"Unexpected frozen manifest: {manifest.get('manifestId')}")
    by_member = {entry["member"]: entry for entry in manifest.get("entries", [])}
    missing = [member for member in PILOT_MEMBERS if member not in by_member]
    if missing:
        raise ValueError(f"Pilot members are missing from the frozen manifest: {missing}")
    selected = [by_member[member] for member in PILOT_MEMBERS]
    roles = [entry["role"] for entry in selected]
    if roles.count("train") != 2 or roles.count("calibration") != 1:
        raise ValueError(f"Pilot must contain 2 train and 1 calibration songs, got {roles}")
    if roles.count("internal-test") != 1 or "final-test" in roles:
        raise ValueError(f"Pilot selection must contain one internal-test and no final-test: {roles}")
    return manifest, selected


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
    entry: dict[str, Any],
) -> Path:
    member = entry["member"]
    role = entry["role"]
    filename = entry["fileName"]
    destination = raw_root / role / filename
    if destination.is_file() and sha256_file(destination).lower() == entry["sourceSha256"].lower():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        try:
            info = handle.getinfo(member)
        except KeyError as error:
            raise FileNotFoundError(f"No {member} in {archive}") from error
        if Path(info.filename).name != filename or info.filename != member:
            raise ValueError(f"Unsafe archive member: {info.filename}")
        with handle.open(info) as source, destination.open("wb") as target:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
    if sha256_file(destination).lower() != entry["sourceSha256"].lower():
        raise ValueError(f"Extracted source hash mismatch for {member}")
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
    entry: dict[str, Any],
    decoded_root: Path,
    start_seconds: float,
    duration_seconds: float,
    force: bool,
) -> SongBundle:
    role = entry["role"]
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
    mixture_gt = (
        decoded["vocals"] + decoded["drums"] + decoded["bass"] + decoded["other"]
    ).astype(np.float32)
    cache_path = output_dir / f"{slug}.npz"
    if force or not cache_path.is_file():
        np.savez_compressed(
            cache_path,
            mixtureEncoded=decoded["mixture"],
            mixtureGt=mixture_gt,
            drums=decoded["drums"],
            bass=decoded["bass"],
            other=decoded["other"],
            vocals=decoded["vocals"],
            instrumental=instrumental,
        )
    return SongBundle(
        role=role,
        member=entry["member"],
        source_sha256=entry["sourceSha256"],
        source_path=source,
        slug=slug,
        mixture_encoded=decoded["mixture"],
        mixture_gt=mixture_gt,
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

    windows, pad = build_windows(song.mixture_gt.T, TEACHER_PARAMS)
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
        instrumental[: song.mixture_gt.shape[0]] * np.float32(TEACHER_OUTPUT_SCALE),
        dtype=np.float32,
    )
    if instrumental.shape != song.mixture_gt.shape:
        raise ValueError(
            f"Teacher output length mismatch: {instrumental.shape} vs {song.mixture_gt.shape}"
        )
    vocals = np.ascontiguousarray(song.mixture_gt - instrumental, dtype=np.float32)
    reconstruction_error = float(np.max(np.abs(instrumental + vocals - song.mixture_gt)))
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
        "samples": int(song.mixture_gt.shape[0]),
        "durationSeconds": song.mixture_gt.shape[0] / song.sample_rate,
        "inputVariant": "mixture-gt",
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
    for start in range(0, song.mixture_gt.shape[0], useful):
        length = min(useful, song.mixture_gt.shape[0] - start)
        if length < useful // 3:
            continue
        segment = song.mixture_gt[start : start + length]
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        vocal_segment = song.vocals[start : start + length]
        vocal_rms = float(np.sqrt(np.mean(vocal_segment.astype(np.float64) ** 2)))
        candidates.append(WindowRecord(song, start, length, rms, vocal_rms))
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
    loss_mode: str,
    device: torch.device,
    output_path: Path,
    steps: int,
    learning_rate: float,
    teacher_weight: float,
) -> tuple[dict[str, Any], TfcTdfNchwWrapper]:
    if not records:
        raise ValueError(f"No training records for {name}")
    if loss_mode not in {"ground-truth", "ground-truth-plus-teacher"}:
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    model, _ = make_model(checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(steps):
        record = records[step % len(records)]
        record.materialize()
        if (
            record.input_spec is None
            or record.true_instrumental_spec is None
            or record.teacher_instrumental_spec is None
        ):
            raise ValueError(f"Missing target for {name} at {record.song.slug}:{record.start}")
        input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
        true_tensor = torch.from_numpy(record.true_instrumental_spec[None]).to(device)
        teacher_tensor = torch.from_numpy(record.teacher_instrumental_spec[None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_instrumental = input_tensor - model(input_tensor)
        ground_truth_loss = F.l1_loss(predicted_instrumental, true_tensor)
        teacher_loss = F.l1_loss(predicted_instrumental, teacher_tensor)
        loss = ground_truth_loss
        if loss_mode == "ground-truth-plus-teacher":
            loss = loss + teacher_weight * teacher_loss
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu().item()),
                "groundTruthLoss": float(ground_truth_loss.detach().cpu().item()),
                "teacherLoss": float(teacher_loss.detach().cpu().item()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "local-inst3-distill-pilot",
            "variant": name,
            "outputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
            "lossMode": loss_mode,
            "teacherWeight": teacher_weight,
            "state_dict": model.state_dict(),
            "steps": steps,
            "learningRate": learning_rate,
        },
        output_path,
    )
    return {
        "variant": name,
        "outputSemantic": "instrumental",
        "neuralCoreSemantic": "vocals-residual",
        "lossMode": loss_mode,
        "teacherWeight": teacher_weight,
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


def low_vocal_mask(vocals: np.ndarray) -> np.ndarray:
    frame_size = DEFAULT_CONFIG.hop_length
    frame_count = vocals.shape[0] // frame_size
    if frame_count == 0:
        return np.ones(vocals.shape[0], dtype=bool)
    usable = vocals[: frame_count * frame_size]
    rms = np.sqrt(np.mean(usable.reshape(frame_count, frame_size, -1) ** 2, axis=(1, 2)))
    threshold = float(np.percentile(rms, 20.0))
    mask = np.repeat(rms <= threshold, frame_size)
    result = np.zeros(vocals.shape[0], dtype=bool)
    result[: mask.shape[0]] = mask
    if not np.any(result):
        result[:] = True
    return result


def separation_metrics(
    mixture: np.ndarray,
    true_vocals: np.ndarray,
    true_instrumental: np.ndarray,
    predicted_instrumental: np.ndarray,
    teacher_instrumental: np.ndarray,
) -> dict[str, float | int]:
    predicted_vocals = mixture - predicted_instrumental
    error = predicted_instrumental - true_instrumental
    teacher_error = teacher_instrumental - true_instrumental
    low_mask = low_vocal_mask(true_vocals)
    low_error = error[low_mask]
    low_true = true_instrumental[low_mask]
    return {
        "samples": int(mixture.shape[0]),
        "instrumentalSdrDb": energy_snr_db(true_instrumental, predicted_instrumental),
        "teacherInstrumentalSdrDb": energy_snr_db(true_instrumental, teacher_instrumental),
        "vocalSdrDb": energy_snr_db(true_vocals, predicted_vocals),
        "teacherVocalSdrDb": energy_snr_db(true_vocals, mixture - teacher_instrumental),
        "accompanimentErrorRmsDbfs": rms_dbfs(error),
        "accompanimentVocalProjectionDb": vocal_projection_db(error, true_vocals),
        "lowVocalInstrumentalSdrDb": energy_snr_db(low_true, predicted_instrumental[low_mask]),
        "lowVocalInstrumentalErrorRmsDbfs": rms_dbfs(low_error),
        "teacherAccompanimentErrorRmsDbfs": rms_dbfs(teacher_error),
        "reconstructionMaxAbsError": float(
            np.max(np.abs(predicted_vocals + predicted_instrumental - mixture))
        ),
    }


def evaluate_model(
    name: str,
    model: TfcTdfNchwWrapper,
    songs: list[SongBundle],
    records_by_song: dict[str, list[WindowRecord]],
    device: torch.device,
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    aggregate: list[dict[str, np.ndarray]] = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for song in songs:
            predictions: list[np.ndarray] = []
            mixtures: list[np.ndarray] = []
            true_vocals: list[np.ndarray] = []
            true_instrumentals: list[np.ndarray] = []
            teacher_instrumentals: list[np.ndarray] = []
            for record in records_by_song[song.slug]:
                record.materialize()
                if (
                    record.input_spec is None
                    or record.teacher_instrumental_spec is None
                ):
                    raise ValueError("Evaluation record is missing materialized data")
                input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
                predicted_spec = input_tensor - model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reconstructed = student_istft_centered(
                    predicted_spec.detach().cpu().numpy()
                )
                segment = reconstructed[
                    DEFAULT_CONFIG.trim_samples : DEFAULT_CONFIG.trim_samples + record.length
                ]
                predictions.append(np.ascontiguousarray(segment, dtype=np.float32))
                begin, end = record.start, record.start + record.length
                mixtures.append(song.mixture_gt[begin:end])
                true_vocals.append(song.vocals[begin:end])
                true_instrumentals.append(song.instrumental[begin:end])
                teacher_instrumentals.append(song.teacher_instrumental[begin:end])
            mix = np.concatenate(mixtures, axis=0)
            vocals = np.concatenate(true_vocals, axis=0)
            instrumental = np.concatenate(true_instrumentals, axis=0)
            teacher = np.concatenate(teacher_instrumentals, axis=0)
            predicted = np.concatenate(predictions, axis=0)
            per_song[song.slug] = {
                "role": song.role,
                "windowStarts": [record.start for record in records_by_song[song.slug]],
                "windowRms": [record.rms for record in records_by_song[song.slug]],
                "windowVocalRms": [record.vocal_rms for record in records_by_song[song.slug]],
                "metrics": separation_metrics(mix, vocals, instrumental, predicted, teacher),
            }
            aggregate.append({
                "mixture": mix,
                "vocals": vocals,
                "instrumental": instrumental,
                "teacher": teacher,
                "predicted": predicted,
            })
    merged = {
        key: np.concatenate([item[key] for item in aggregate], axis=0)
        for key in aggregate[0]
    }
    metrics = separation_metrics(
        merged["mixture"],
        merged["vocals"],
        merged["instrumental"],
        merged["predicted"],
        merged["teacher"],
    )
    return {
        "variant": name,
        "outputSemantic": "instrumental",
        "perSong": per_song,
        "aggregate": metrics,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def evaluate_teacher(
    songs: list[SongBundle], records_by_song: dict[str, list[WindowRecord]]
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    aggregate: list[dict[str, np.ndarray]] = []
    for song in songs:
        if song.teacher_instrumental is None:
            raise ValueError("Teacher output missing")
        mixtures: list[np.ndarray] = []
        vocals: list[np.ndarray] = []
        instrumentals: list[np.ndarray] = []
        teachers: list[np.ndarray] = []
        for record in records_by_song[song.slug]:
            begin, end = record.start, record.start + record.length
            mixtures.append(song.mixture_gt[begin:end])
            vocals.append(song.vocals[begin:end])
            instrumentals.append(song.instrumental[begin:end])
            teachers.append(song.teacher_instrumental[begin:end])
        mix = np.concatenate(mixtures, axis=0)
        true_vocal = np.concatenate(vocals, axis=0)
        true_inst = np.concatenate(instrumentals, axis=0)
        teacher_inst = np.concatenate(teachers, axis=0)
        per_song[song.slug] = {
            "role": song.role,
            "metrics": separation_metrics(
                mix, true_vocal, true_inst, teacher_inst, teacher_inst
            ),
        }
        aggregate.append({
            "mixture": mix,
            "vocals": true_vocal,
            "instrumental": true_inst,
            "teacher": teacher_inst,
        })
    merged = {key: np.concatenate([item[key] for item in aggregate], axis=0) for key in aggregate[0]}
    return {
        "outputSemantic": "instrumental",
        "perSong": per_song,
        "aggregate": separation_metrics(
            merged["mixture"],
            merged["vocals"],
            merged["instrumental"],
            merged["teacher"],
            merged["teacher"],
        ),
    }


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0 or args.start_seconds < 0:
        raise ValueError("start and duration must be non-negative, duration must be positive")
    if args.train_windows_per_song <= 0 or args.eval_windows_per_song <= 0:
        raise ValueError("window counts must be positive")
    if args.steps <= 0 or args.learning_rate <= 0 or args.threads <= 0:
        raise ValueError("steps, learning rate, and threads must be positive")
    if args.teacher_weight < 0:
        raise ValueError("teacher weight must be non-negative")

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

    manifest_path = args.manifest.resolve()
    frozen_manifest, selected_entries = load_frozen_pilot_entries(manifest_path)
    expected_members = [entry["member"] for entry in selected_entries]
    archive_path = args.archive.resolve()
    archive_info = inspect_archive(archive_path, expected_members)
    source_paths: list[tuple[dict[str, Any], Path]] = []
    for entry in selected_entries:
        source_paths.append(
            (entry, ensure_raw_song(archive_path, raw_root, entry))
        )

    contract_info = verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    session, teacher_providers = make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    bundles: list[SongBundle] = []
    for entry, source in source_paths:
        bundles.append(
            decode_song(
                source,
                entry,
                decoded_root,
                args.start_seconds,
                args.duration_seconds,
                args.force_decode,
            )
        )

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
    eval_songs = [
        song for song in bundles if song.role in {"calibration", "internal-test"}
    ]
    if len(train_songs) != 2 or len(eval_songs) != 2:
        raise ValueError(
            "Pilot must contain exactly two train songs and calibration/internal-test "
            f"evaluation songs; got train={len(train_songs)}, eval={len(eval_songs)}"
        )
    if any(song.role == "final-test" for song in bundles):
        raise ValueError("The final-test split must never be used by this pilot")
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
        "S0-ground-truth",
        args.checkpoint.resolve(),
        train_records,
        "ground-truth",
        device,
        runs_root / "s0-ground-truth.pt",
        args.steps,
        args.learning_rate,
        0.0,
    )
    s1_report, s1_model = train_variant(
        "S1-ground-truth-plus-inst3",
        args.checkpoint.resolve(),
        train_records,
        "ground-truth-plus-teacher",
        device,
        runs_root / "s1-ground-truth-plus-inst3.pt",
        args.steps,
        args.learning_rate,
        args.teacher_weight,
    )
    s0_eval = evaluate_model(
        "S0-ground-truth", s0_model, eval_songs, eval_records, device
    )
    s1_eval = evaluate_model(
        "S1-ground-truth-plus-inst3", s1_model, eval_songs, eval_records, device
    )
    del s0_model, s1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    song_metadata = {
        song.slug: {
            "role": song.role,
            "member": song.member,
            "sourceSha256": song.source_sha256,
            "source": str(song.source_path.resolve()),
            "sampleRate": song.sample_rate,
            "samples": int(song.mixture_gt.shape[0]),
            "durationSeconds": song.mixture_gt.shape[0] / song.sample_rate,
            "decodedPcmFloat32Sha256": {
                name: sha256_array(value)
                for name, value in {
                    "mixtureEncoded": song.mixture_encoded,
                    "mixtureGt": song.mixture_gt,
                    "drums": song.drums,
                    "bass": song.bass,
                    "other": song.other,
                    "vocals": song.vocals,
                    "instrumental": song.instrumental,
                }.items()
            },
            "mixtureEncodedMinusMixtureGtRmsDbfs": rms_dbfs(
                song.mixture_encoded - song.mixture_gt
            ),
            "mixtureGtMinusStemSumRmsDbfs": rms_dbfs(
                song.mixture_gt - (song.instrumental + song.vocals)
            ),
            "trainWindowStarts": [record.start for record in train_records if record.song.slug == song.slug],
            "evalWindowStarts": [record.start for record in eval_records.get(song.slug, [])],
        }
        for song in bundles
    }
    report = {
        "schemaVersion": 2,
        "experimentId": "inst3-distill-pilot@1",
        "status": "pilot-completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local pilot artifacts only; do not publish",
        },
        "manifest": {
            "file": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifestId": frozen_manifest["manifestId"],
            "selectedEntries": [
                {
                    "member": entry["member"],
                    "fileName": entry["fileName"],
                    "role": entry["role"],
                    "sourceSha256": entry["sourceSha256"],
                }
                for entry in selected_entries
            ],
            "finalTestUsed": False,
        },
        "split": {
            "train": [song.slug for song in train_songs],
            "calibration": [song.slug for song in bundles if song.role == "calibration"],
            "internalTest": [
                song.slug for song in bundles if song.role == "internal-test"
            ],
            "startSeconds": args.start_seconds,
            "durationSeconds": args.duration_seconds,
        },
        "dataContract": {
            "inputSemantic": "mixture-gt",
            "studentOutputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
            "mixtureGtDefinition": "vocals + drums + bass + other",
            "encodedMixtureRetainedForCodecComparison": True,
        },
        "archive": archive_info,
        "contract": contract_info,
        "teacher": {
            "contractId": contract_info["contractId"],
            "sourceIdentity": contract_info["teacherSource"],
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
            "inputSemantic": "mixture-gt",
            "outputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
            "trainWindowCount": len(train_records),
            "steps": args.steps,
            "learningRate": args.learning_rate,
            "teacherWeight": args.teacher_weight,
            "loss": "mean-absolute-error in complex packed instrumental spectrum",
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
        "evalSongs": [song.slug for song in eval_songs],
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
