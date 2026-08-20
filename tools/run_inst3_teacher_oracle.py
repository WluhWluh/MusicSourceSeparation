#!/usr/bin/env python3
"""Build a frozen MUSDB18 split and run the local Inst 3 teacher oracle.

This is a local, non-commercial research runner.  It deliberately stops before
student training: its job is to make the data split, mixture definitions,
teacher cache identity, and whole-song quality baseline reproducible.

The official MUSDB18 train split is deterministically divided into 80 train,
10 calibration, and 10 internal-test songs.  The official test split remains
untouched as ``final-test``.  For selected songs the runner decodes all five
audio streams, constructs an exact stem-sum mixture, preserves the encoded
mixture stream, and renders UVR-MDX-NET Inst 3 for both inputs.
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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

from mdx_reference import MdxParams, build_windows, istft_centered, stft_centered
from run_inst3_distill_pilot import (
    DEFAULT_ARCHIVE,
    DEFAULT_CONTRACT,
    DEFAULT_TEACHER,
    DEFAULT_TEACHER_TFLITE,
    EXPECTED_TEACHER_SHA256,
    STREAM_NAMES,
    TEACHER_OUTPUT_SCALE,
    TEACHER_PARAMS,
    ensure_raw_song,
    make_teacher_session,
    prepare_cuda_dlls,
    sha256_array,
    sha256_file,
    verify_teacher_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / ".tmp" / "musdb18-inst3-oracle"
SPLIT_SEED = 20260819
MANIFEST_SCHEMA_VERSION = 1
ORACLE_SCHEMA_VERSION = 1


@dataclass
class DecodedSong:
    entry: dict[str, Any]
    source_path: Path
    slug: str
    sample_rate: int
    mixture_encoded: np.ndarray
    mixture_gt: np.ndarray
    vocals: np.ndarray
    instrumental: np.ndarray
    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--internal-test-count", type=int, default=10)
    parser.add_argument("--parity-count", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-decode", action="store_true")
    parser.add_argument(
        "--keep-decoded-wav",
        action="store_true",
        help="Keep intermediate float32 WAV files after a decoded NPZ is written",
    )
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Require CUDA for the primary teacher session",
    )
    return parser.parse_args()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value.removesuffix("-stem-mp4")


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\n{}".format(
                result.returncode,
                " ".join(command),
                result.stderr[-4000:],
            )
        )
    return result


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def hash_zip_member(handle: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with handle.open(info) as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def rank_member(member: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{member}".encode("utf-8")).hexdigest()


def build_manifest(
    archive: Path,
    manifest_path: Path,
    seed: int,
    calibration_count: int,
    internal_test_count: int,
    force: bool,
) -> dict[str, Any]:
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("archive", {}).get("bytes") == archive.stat().st_size
            and manifest.get("split", {}).get("seed") == seed
            and manifest.get("split", {}).get("calibrationCount") == calibration_count
            and manifest.get("split", {}).get("internalTestCount") == internal_test_count
        ):
            return manifest

    train_entries: list[dict[str, Any]] = []
    test_entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as handle:
        infos = [
            info
            for info in handle.infolist()
            if info.filename.lower().endswith(".stem.mp4")
        ]
        if len(infos) != 150:
            raise ValueError(f"Expected 150 MUSDB18 stem files, found {len(infos)}")
        for index, info in enumerate(sorted(infos, key=lambda item: item.filename)):
            parts = info.filename.split("/", 1)
            if len(parts) != 2 or parts[0] not in {"train", "test"}:
                raise ValueError(f"Unexpected MUSDB18 member: {info.filename}")
            entry = {
                "member": info.filename,
                "archiveSplit": parts[0],
                "fileName": parts[1],
                "bytes": info.file_size,
                "compressedBytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "sourceSha256": hash_zip_member(handle, info),
            }
            if parts[0] == "train":
                train_entries.append(entry)
            else:
                test_entries.append(entry)
            if (index + 1) % 10 == 0:
                print(f"hashed archive members: {index + 1}/{len(infos)}", flush=True)

    train_entries = sorted(train_entries, key=lambda item: item["member"])
    ranked_train = sorted(
        train_entries,
        key=lambda item: rank_member(item["member"], seed),
    )
    if len(ranked_train) != 100 or len(test_entries) != 50:
        raise ValueError(
            f"Unexpected split sizes: train={len(ranked_train)}, test={len(test_entries)}"
        )
    if calibration_count < 1 or internal_test_count < 1:
        raise ValueError("Calibration and internal-test counts must be positive")
    if calibration_count + internal_test_count >= len(ranked_train):
        raise ValueError("Internal holdout counts leave no train songs")

    role_by_member: dict[str, str] = {}
    for item in ranked_train[: len(ranked_train) - calibration_count - internal_test_count]:
        role_by_member[item["member"]] = "train"
    calibration_start = len(ranked_train) - calibration_count - internal_test_count
    for item in ranked_train[calibration_start : calibration_start + calibration_count]:
        role_by_member[item["member"]] = "calibration"
    for item in ranked_train[calibration_start + calibration_count :]:
        role_by_member[item["member"]] = "internal-test"
    for item in test_entries:
        role_by_member[item["member"]] = "final-test"

    entries: list[dict[str, Any]] = []
    for item in sorted(train_entries + test_entries, key=lambda value: value["member"]):
        enriched = dict(item)
        enriched["role"] = role_by_member[item["member"]]
        enriched["rank"] = (
            rank_member(item["member"], seed)
            if item["archiveSplit"] == "train"
            else None
        )
        entries.append(enriched)

    manifest = {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "manifestId": "musdb18-inst3-oracle-split@1",
        "archive": {
            "file": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "entryCount": 153,
            "stemEntryCount": len(entries),
            "uncompressedStemBytes": sum(item["bytes"] for item in entries),
        },
        "dataset": {
            "name": "MUSDB18",
            "zenodoRecord": "1117372",
            "licenseDisposition": "educational/non-commercial; local artifacts only",
            "mixtureStream": "audio stream 0",
            "stemStreams": {
                "mixture": 0,
                "drums": 1,
                "bass": 2,
                "other": 3,
                "vocals": 4,
            },
        },
        "split": {
            "algorithm": "sort SHA256(seed + NUL + archive member)",
            "seed": seed,
            "trainCount": 80,
            "calibrationCount": calibration_count,
            "internalTestCount": internal_test_count,
            "finalTestCount": len(test_entries),
            "calibrationSongCount": calibration_count,
            "internalTestSongCount": internal_test_count,
        },
        "entries": entries,
        "generator": {
            "file": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "gitRevision": git_revision(),
        },
    }
    json_write(manifest_path, manifest)
    return manifest


def extract_member(
    archive: Path,
    entry: dict[str, Any],
    raw_root: Path,
    force: bool,
) -> Path:
    destination = raw_root / entry["archiveSplit"] / entry["fileName"]
    if destination.is_file() and not force:
        if sha256_file(destination).lower() == entry["sourceSha256"].lower():
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        info = handle.getinfo(entry["member"])
        with handle.open(info) as source, destination.open("wb") as target:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
    actual = sha256_file(destination)
    if actual.lower() != entry["sourceSha256"].lower():
        raise ValueError(f"Extracted source hash mismatch: {destination}")
    return destination


def probe_audio(path: Path) -> dict[str, Any]:
    result = run_checked(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def decode_song(
    archive: Path,
    entry: dict[str, Any],
    raw_root: Path,
    decoded_root: Path,
    force_extract: bool,
    force_decode: bool,
    keep_decoded_wav: bool,
) -> DecodedSong:
    source_path = extract_member(archive, entry, raw_root, force_extract)
    slug = slugify(entry["fileName"])
    output_dir = decoded_root / entry["role"] / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "decoded.npz"
    metadata_path = output_dir / "decoded.json"
    if cache_path.is_file() and metadata_path.is_file() and not force_decode:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sourceSha256") == entry["sourceSha256"]:
            values = np.load(cache_path)
            arrays = {
                name: np.ascontiguousarray(values[name], dtype=np.float32)
                for name in (*STREAM_NAMES, "mixtureGt", "instrumentalGt")
            }
            return DecodedSong(
                entry=entry,
                source_path=source_path,
                slug=slug,
                sample_rate=int(metadata["sampleRate"]),
                mixture_encoded=arrays["mixture"],
                mixture_gt=arrays["mixtureGt"],
                vocals=arrays["vocals"],
                instrumental=arrays["instrumentalGt"],
                drums=arrays["drums"],
                bass=arrays["bass"],
                other=arrays["other"],
                metadata=metadata,
            )

    probe = probe_audio(source_path)
    streams = probe.get("streams", [])
    if len(streams) != 5:
        raise ValueError(f"Expected 5 audio streams in {source_path}, found {len(streams)}")
    decoded_paths: list[Path] = []
    for index, name in enumerate(STREAM_NAMES):
        output_path = output_dir / f"stream-{index}-{name}.wav"
        decoded_paths.append(output_path)
        if output_path.is_file() and not force_decode:
            continue
        run_checked(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-map",
                f"0:a:{index}",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "2",
                "-ar",
                str(TEACHER_PARAMS.sample_rate),
                "-c:a",
                "pcm_f32le",
                str(output_path),
            ]
        )

    decoded: dict[str, np.ndarray] = {}
    sample_rates: set[int] = set()
    for name, path in zip(STREAM_NAMES, decoded_paths):
        value, rate = sf.read(path, dtype="float32", always_2d=True)
        if value.shape[1] == 1:
            value = np.repeat(value, 2, axis=1)
        elif value.shape[1] > 2:
            value = value[:, :2]
        if rate != TEACHER_PARAMS.sample_rate:
            raise ValueError(f"Unexpected decoded rate for {path}: {rate}")
        decoded[name] = np.ascontiguousarray(value, dtype=np.float32)
        sample_rates.add(rate)
    if len(sample_rates) != 1:
        raise ValueError(f"Inconsistent sample rates for {source_path}: {sample_rates}")
    length = min(value.shape[0] for value in decoded.values())
    decoded = {name: value[:length] for name, value in decoded.items()}
    instrumental = (
        decoded["drums"] + decoded["bass"] + decoded["other"]
    ).astype(np.float32)
    mixture_gt = (instrumental + decoded["vocals"]).astype(np.float32)
    arrays = {
        **decoded,
        "mixtureGt": mixture_gt,
        "instrumentalGt": instrumental,
    }
    np.savez_compressed(cache_path, **arrays)
    metadata = {
        "sourceSha256": entry["sourceSha256"],
        "sourceFile": str(source_path.resolve()),
        "sampleRate": next(iter(sample_rates)),
        "samples": int(length),
        "durationSeconds": length / next(iter(sample_rates)),
        "streamProbe": streams,
        "decodedPcmFloat32Sha256": {
            name: sha256_array(value) for name, value in arrays.items()
        },
        "mixtureGtVsEncodedRmsDbfs": rms_dbfs(mixture_gt - decoded["mixture"]),
        "mixtureGtVsEncodedMaxAbs": float(
            np.max(np.abs(mixture_gt - decoded["mixture"]))
        ),
        "cache": {
            "file": str(cache_path),
            "bytes": cache_path.stat().st_size,
            "sha256": sha256_file(cache_path),
        },
    }
    if not keep_decoded_wav:
        for path in decoded_paths:
            path.unlink(missing_ok=True)
        metadata["decodedWavRetained"] = False
    else:
        metadata["decodedWavRetained"] = True
    json_write(metadata_path, metadata)
    return DecodedSong(
        entry=entry,
        source_path=source_path,
        slug=slug,
        sample_rate=next(iter(sample_rates)),
        mixture_encoded=decoded["mixture"],
        mixture_gt=mixture_gt,
        vocals=decoded["vocals"],
        instrumental=instrumental,
        drums=decoded["drums"],
        bass=decoded["bass"],
        other=decoded["other"],
        metadata=metadata,
    )


def rms_dbfs(value: np.ndarray) -> float:
    power = float(np.mean(value.astype(np.float64) ** 2))
    return 10.0 * math.log10(max(power, 1e-30))


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference_flat = reference.astype(np.float64).reshape(-1)
    estimate_flat = estimate.astype(np.float64).reshape(-1)
    reference_flat -= np.mean(reference_flat)
    estimate_flat -= np.mean(estimate_flat)
    denominator = float(np.dot(reference_flat, reference_flat))
    if denominator <= 1e-30:
        return float("nan")
    scale = float(np.dot(estimate_flat, reference_flat) / denominator)
    target = scale * reference_flat
    noise = estimate_flat - target
    return 10.0 * math.log10(
        max(float(np.dot(target, target)), 1e-30)
        / max(float(np.dot(noise, noise)), 1e-30)
    )


def projection_ratio_db(signal: np.ndarray, basis: np.ndarray) -> float:
    signal_flat = signal.astype(np.float64).reshape(-1)
    basis_flat = basis.astype(np.float64).reshape(-1)
    basis_power = float(np.dot(basis_flat, basis_flat))
    signal_power = float(np.dot(signal_flat, signal_flat))
    if basis_power <= 1e-30 or signal_power <= 1e-30:
        return float("-inf")
    coefficient = float(np.dot(signal_flat, basis_flat) / basis_power)
    projected_power = coefficient * coefficient * basis_power
    return 10.0 * math.log10(max(projected_power, 1e-30) / signal_power)


def array_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Array shape mismatch: {reference.shape} != {candidate.shape}")
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    ref_power = float(np.sum(ref * ref))
    error_power = float(np.sum(error * error))
    cand_power = float(np.sum(cand * cand))
    return {
        "maxAbsError": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "snrDb": 10.0 * math.log10(max(ref_power, 1e-30) / max(error_power, 1e-30)),
        "cosineSimilarity": float(
            np.sum(ref * cand)
            / max(math.sqrt(ref_power * cand_power), 1e-30)
        ),
    }


def percentile_frame_mask(value: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    frame_size = TEACHER_PARAMS.sample_rate // 2
    frame_count = value.shape[0] // frame_size
    if frame_count <= 0:
        return np.ones(value.shape[0], dtype=bool), 0.0
    usable = value[: frame_count * frame_size].reshape(frame_count, frame_size, -1)
    rms_values = np.sqrt(np.mean(usable.astype(np.float64) ** 2, axis=(1, 2)))
    threshold = float(np.percentile(rms_values, percentile))
    mask_frames = rms_values <= threshold
    mask = np.zeros(value.shape[0], dtype=bool)
    for index, selected in enumerate(mask_frames):
        if selected:
            mask[index * frame_size : (index + 1) * frame_size] = True
    return mask, threshold


def boundary_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    generation_size: int,
) -> dict[str, Any]:
    boundaries = list(range(generation_size, estimate.shape[0], generation_size))
    if not boundaries:
        return {
            "count": 0,
            "meanJumpDbfs": None,
            "referenceMeanJumpDbfs": None,
            "meanWindowErrorRmsDbfs": None,
            "maxWindowErrorRmsDbfs": None,
        }
    jumps: list[float] = []
    reference_jumps: list[float] = []
    errors: list[float] = []
    for boundary in boundaries:
        estimate_jump = estimate[boundary] - estimate[boundary - 1]
        reference_jump = reference[boundary] - reference[boundary - 1]
        jumps.append(rms_dbfs(estimate_jump[None, :]))
        reference_jumps.append(rms_dbfs(reference_jump[None, :]))
        left = max(0, boundary - 2048)
        right = min(estimate.shape[0], boundary + 2048)
        errors.append(rms_dbfs(estimate[left:right] - reference[left:right]))
    return {
        "count": len(boundaries),
        "meanJumpDbfs": float(np.mean(jumps)),
        "maxJumpDbfs": float(np.max(jumps)),
        "referenceMeanJumpDbfs": float(np.mean(reference_jumps)),
        "meanWindowErrorRmsDbfs": float(np.mean(errors)),
        "maxWindowErrorRmsDbfs": float(np.max(errors)),
        "boundariesSamples": boundaries,
    }


def quality_metrics(
    song: DecodedSong,
    input_audio: np.ndarray,
    instrumental_estimate: np.ndarray,
    vocal_estimate: np.ndarray,
    variant: str,
) -> dict[str, Any]:
    instrumental_reference = song.instrumental
    vocals_reference = song.vocals
    if input_audio.shape != song.mixture_gt.shape:
        raise ValueError("Teacher input and decoded song lengths differ")
    low_mask, low_threshold = percentile_frame_mask(vocals_reference, 20.0)
    high_mask, high_threshold = percentile_frame_mask(vocals_reference, 80.0)
    inst_error = instrumental_estimate - instrumental_reference
    reconstruction = instrumental_estimate + vocal_estimate
    result = {
        "variant": variant,
        "samples": int(input_audio.shape[0]),
        "durationSeconds": input_audio.shape[0] / song.sample_rate,
        "inputPcmFloat32Sha256": sha256_array(input_audio),
        "groundTruth": {
            "instrumentalRawFloat32Sha256": sha256_array(instrumental_reference),
            "vocalsRawFloat32Sha256": sha256_array(vocals_reference),
        },
        "instrumental": {
            "siSdrDb": si_sdr_db(instrumental_reference, instrumental_estimate),
            "rmsDbfs": rms_dbfs(instrumental_estimate),
            "referenceRmsDbfs": rms_dbfs(instrumental_reference),
            "peak": float(np.max(np.abs(instrumental_estimate))),
            "vocalLeakageProjectionDb": projection_ratio_db(
                instrumental_estimate, vocals_reference
            ),
            "errorRmsDbfs": rms_dbfs(inst_error),
            "lowVocalFrames": int(np.count_nonzero(low_mask)),
            "lowVocalThresholdRms": low_threshold,
            "lowVocalErrorRmsDbfs": rms_dbfs(inst_error[low_mask]),
            "lowVocalSiSdrDb": si_sdr_db(
                instrumental_reference[low_mask],
                instrumental_estimate[low_mask],
            ),
            "highVocalFrames": int(np.count_nonzero(high_mask)),
            "highVocalThresholdRms": high_threshold,
            "highVocalLeakageProjectionDb": projection_ratio_db(
                instrumental_estimate[high_mask], vocals_reference[high_mask]
            ),
        },
        "vocals": {
            "siSdrDb": si_sdr_db(vocals_reference, vocal_estimate),
            "rmsDbfs": rms_dbfs(vocal_estimate),
            "referenceRmsDbfs": rms_dbfs(vocals_reference),
            "peak": float(np.max(np.abs(vocal_estimate))),
            "energyDeltaDb": rms_dbfs(vocal_estimate) - rms_dbfs(vocals_reference),
        },
        "reconstruction": {
            "maxAbsError": float(np.max(np.abs(reconstruction - input_audio))),
            "rmsErrorDbfs": rms_dbfs(reconstruction - input_audio),
        },
        "inputComparison": {
            "rmsDbfs": rms_dbfs(input_audio),
            "peak": float(np.max(np.abs(input_audio))),
            "versusMixtureGtRmsDbfs": rms_dbfs(input_audio - song.mixture_gt),
            "versusMixtureGtMaxAbs": float(np.max(np.abs(input_audio - song.mixture_gt))),
        },
        "boundaries": boundary_metrics(
            instrumental_estimate,
            instrumental_reference,
            TEACHER_PARAMS.generation_size,
        ),
    }
    return result


def make_cpu_session(teacher_path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(teacher_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected CPU parity providers: {session.get_providers()}")
    return session


def render_teacher_audio(
    session: ort.InferenceSession,
    audio: np.ndarray,
    *,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected [samples, 2] audio, got {audio.shape}")
    windows, pad = build_windows(audio.T, TEACHER_PARAMS)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pieces: list[np.ndarray] = []
    window_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        try:
            dsp_started = time.perf_counter()
            model_input = stft_centered(window[None, ...], TEACHER_PARAMS)
            dsp_seconds = time.perf_counter() - dsp_started
            inference_started = time.perf_counter()
            model_output = session.run(
                [output_name],
                {input_name: np.ascontiguousarray(model_input, dtype=np.float32)},
            )[0]
            inference_seconds = time.perf_counter() - inference_started
            expected_shape = (1, 4, TEACHER_PARAMS.dim_f, TEACHER_PARAMS.dim_t)
            if tuple(model_output.shape) != expected_shape:
                raise ValueError(f"Unexpected output shape: {model_output.shape}")
            if not np.isfinite(model_output).all():
                raise ValueError("Teacher output contains non-finite values")
            reconstructed = istft_centered(model_output.astype(np.float32), TEACHER_PARAMS)[0]
            useful = reconstructed[:, TEACHER_PARAMS.trim : -TEACHER_PARAMS.trim].T
            pieces.append(np.ascontiguousarray(useful, dtype=np.float32))
            window_reports.append(
                {
                    "index": index,
                    "stftSeconds": dsp_seconds,
                    "inferenceSeconds": inference_seconds,
                    "outputPeak": float(np.max(np.abs(useful))),
                    "passed": True,
                }
            )
            if (index + 1) % 10 == 0 or index + 1 == len(windows):
                print(
                    f"{progress_label}: windows {index + 1}/{len(windows)}",
                    flush=True,
                )
        except Exception as error:  # noqa: BLE001 - preserve failed-window evidence
            failure = {"index": index, "error": repr(error)}
            failures.append(failure)
            window_reports.append({**failure, "passed": False})
            raise RuntimeError(f"{progress_label} failed at window {index}: {error}") from error
    instrumental = np.concatenate(pieces, axis=0)
    if pad:
        instrumental = instrumental[:-pad]
    instrumental = np.ascontiguousarray(
        instrumental[: audio.shape[0]] * np.float32(TEACHER_OUTPUT_SCALE),
        dtype=np.float32,
    )
    if instrumental.shape != audio.shape:
        raise ValueError(f"Teacher output shape mismatch: {instrumental.shape} != {audio.shape}")
    vocals = np.ascontiguousarray(audio - instrumental, dtype=np.float32)
    report = {
        "windowCount": len(windows),
        "paddingSamples": int(pad),
        "generationSize": TEACHER_PARAMS.generation_size,
        "failures": failures,
        "windows": window_reports,
        "timing": {
            "totalSeconds": time.perf_counter() - started,
            "inferenceSeconds": sum(
                float(item.get("inferenceSeconds", 0.0)) for item in window_reports
            ),
            "stftSeconds": sum(
                float(item.get("stftSeconds", 0.0)) for item in window_reports
            ),
        },
    }
    return instrumental, vocals, report


def cache_paths(
    output_root: Path,
    song: DecodedSong,
    variant: str,
    cache_key: str,
) -> tuple[Path, Path]:
    directory = output_root / "teacher" / song.entry["role"] / song.slug
    prefix = f"{variant}-{cache_key[:16]}"
    return directory / f"{prefix}.npz", directory / f"{prefix}.json"


def render_or_load_cached(
    output_root: Path,
    song: DecodedSong,
    variant: str,
    input_audio: np.ndarray,
    session: ort.InferenceSession,
    contract_info: dict[str, Any],
    runner_revision: str,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dsp_settings = {
        "sampleRate": TEACHER_PARAMS.sample_rate,
        "nFft": TEACHER_PARAMS.n_fft,
        "hopLength": TEACHER_PARAMS.hop_length,
        "dimF": TEACHER_PARAMS.dim_f,
        "dimTPower": TEACHER_PARAMS.dim_t_power,
        "modelTimeFrames": TEACHER_PARAMS.dim_t,
        "trim": TEACHER_PARAMS.trim,
        "window": "periodic-hann",
        "modelOutputScale": TEACHER_OUTPUT_SCALE,
    }
    key_payload = {
        "songSha256": song.entry["sourceSha256"],
        "inputVariant": variant,
        "inputPcmFloat32Sha256": sha256_array(input_audio),
        "teacherSourceSha256": EXPECTED_TEACHER_SHA256,
        "contractId": contract_info["contractId"],
        "runnerRevision": runner_revision,
        "dspSettings": dsp_settings,
    }
    cache_key = canonical_sha256(key_payload)
    npz_path, metadata_path = cache_paths(output_root, song, variant, cache_key)
    if npz_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cacheKey") == cache_key:
            values = np.load(npz_path)
            return (
                np.ascontiguousarray(values["instrumental"], dtype=np.float32),
                np.ascontiguousarray(values["vocals"], dtype=np.float32),
                metadata,
            )

    instrumental, vocals, render_report = render_teacher_audio(
        session,
        input_audio,
        progress_label=f"{song.entry['role']}/{song.slug}/{variant}",
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, instrumental=instrumental, vocals=vocals)
    metadata = {
        "schemaVersion": ORACLE_SCHEMA_VERSION,
        "cacheKey": cache_key,
        "cacheKeyPayload": key_payload,
        "song": {
            "member": song.entry["member"],
            "role": song.entry["role"],
            "sourceSha256": song.entry["sourceSha256"],
            "slug": song.slug,
        },
        "variant": variant,
        "contract": contract_info,
        "dspSettings": dsp_settings,
        "providers": session.get_providers(),
        "input": {
            "samples": int(input_audio.shape[0]),
            "pcmFloat32Sha256": sha256_array(input_audio),
        },
        "output": {
            "instrumentalRawFloat32Sha256": sha256_array(instrumental),
            "vocalsRawFloat32Sha256": sha256_array(vocals),
            "npz": {
                "file": str(npz_path),
                "bytes": npz_path.stat().st_size,
                "sha256": sha256_file(npz_path),
            },
        },
        "render": render_report,
    }
    json_write(metadata_path, metadata)
    return instrumental, vocals, metadata


def run_cpu_gpu_parity(
    output_root: Path,
    songs: list[DecodedSong],
    teacher_path: Path,
    gpu_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    threads: int,
) -> dict[str, Any]:
    cpu_session = make_cpu_session(teacher_path, threads)
    results: dict[str, Any] = {
        "providers": cpu_session.get_providers(),
        "songs": {},
    }
    for song in songs:
        gpu_instrumental, gpu_vocals = gpu_outputs[song.slug]
        cpu_instrumental, cpu_vocals, cpu_render = render_teacher_audio(
            cpu_session,
            song.mixture_gt,
            progress_label=f"cpu-parity/{song.slug}",
        )
        results["songs"][song.slug] = {
            "source": song.entry["member"],
            "instrumental": array_metrics(gpu_instrumental, cpu_instrumental),
            "vocals": array_metrics(gpu_vocals, cpu_vocals),
            "cpuRender": cpu_render,
            "passedFinite": bool(
                np.isfinite(cpu_instrumental).all()
                and np.isfinite(cpu_vocals).all()
            ),
        }
    all_inst = [item["instrumental"] for item in results["songs"].values()]
    all_vocals = [item["vocals"] for item in results["songs"].values()]
    results["aggregate"] = {
        "instrumentalMaxAbsError": max(item["maxAbsError"] for item in all_inst),
        "instrumentalMinSnrDb": min(item["snrDb"] for item in all_inst),
        "vocalsMaxAbsError": max(item["maxAbsError"] for item in all_vocals),
        "vocalsMinSnrDb": min(item["snrDb"] for item in all_vocals),
    }
    return results


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.calibration_count <= 0 or args.internal_test_count <= 0:
        raise ValueError("threads and split counts must be positive")
    if args.parity_count < 0:
        raise ValueError("parity-count cannot be negative")

    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_root / "musdb18-inst3-oracle-manifest.json"
    )
    manifest = build_manifest(
        archive,
        manifest_path,
        args.seed,
        args.calibration_count,
        args.internal_test_count,
        args.force_manifest,
    )
    print(
        "split manifest: train={} calibration={} internal-test={} final-test={}".format(
            manifest["split"]["trainCount"],
            manifest["split"]["calibrationCount"],
            manifest["split"]["internalTestCount"],
            manifest["split"]["finalTestCount"],
        ),
        flush=True,
    )

    contract_info = verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    runner_revision = sha256_file(Path(__file__).resolve())
    raw_root = output_root / "raw"
    decoded_root = output_root / "decoded"
    selected_entries = [
        item
        for item in manifest["entries"]
        if item["role"] in {"calibration", "internal-test"}
    ]
    selected_entries.sort(key=lambda item: (item["role"], item["member"]))
    if len(selected_entries) != args.calibration_count + args.internal_test_count:
        raise ValueError(f"Unexpected selected entry count: {len(selected_entries)}")

    songs: list[DecodedSong] = []
    for index, entry in enumerate(selected_entries, start=1):
        print(f"decode {index}/{len(selected_entries)}: {entry['member']}", flush=True)
        songs.append(
            decode_song(
                archive,
                entry,
                raw_root,
                decoded_root,
                args.force_extract,
                args.force_decode,
                args.keep_decoded_wav,
            )
        )

    if args.require_teacher_cuda:
        prepare_cuda_dlls()
    gpu_session, gpu_providers = make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    print(f"primary teacher providers: {gpu_providers}", flush=True)
    gpu_outputs_for_parity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    song_reports: dict[str, Any] = {}
    for index, song in enumerate(songs, start=1):
        print(f"teacher {index}/{len(songs)}: {song.entry['member']}", flush=True)
        variant_reports: dict[str, Any] = {}
        for variant, input_audio in (
            ("mixture-gt", song.mixture_gt),
            ("mixture-encoded", song.mixture_encoded),
        ):
            instrumental, vocals, metadata = render_or_load_cached(
                output_root,
                song,
                variant,
                input_audio,
                gpu_session,
                contract_info,
                runner_revision,
                args.force_teacher,
            )
            variant_reports[variant] = {
                "cacheKey": metadata["cacheKey"],
                "providers": metadata.get("providers", gpu_providers),
                "render": metadata.get("render", {}),
                "quality": quality_metrics(
                    song,
                    input_audio,
                    instrumental,
                    vocals,
                    variant,
                ),
            }
            if variant == "mixture-gt":
                gpu_outputs_for_parity[song.slug] = (instrumental, vocals)
        song_reports[song.slug] = {
            "member": song.entry["member"],
            "role": song.entry["role"],
            "sourceSha256": song.entry["sourceSha256"],
            "decoded": song.metadata,
            "variants": variant_reports,
        }

    parity_songs = songs[: min(args.parity_count, len(songs))]
    parity_report = (
        run_cpu_gpu_parity(
            output_root,
            parity_songs,
            args.teacher.resolve(),
            gpu_outputs_for_parity,
            args.threads,
        )
        if parity_songs
        else {"skipped": True, "songs": {}}
    )
    report = {
        "schemaVersion": ORACLE_SCHEMA_VERSION,
        "status": "teacher-oracle-completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedOutputs": "ignored local caches only; do not publish",
        },
        "archive": manifest["archive"],
        "manifest": {
            "file": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "id": manifest["manifestId"],
            "split": manifest["split"],
        },
        "contract": contract_info,
        "primaryTeacher": {
            "providers": gpu_providers,
            "ortVersion": ort.__version__,
            "availableProviders": ort.get_available_providers(),
            "cudaRequired": args.require_teacher_cuda,
            "cudaDllSetup": prepare_cuda_dlls(),
            "parameters": {
                "sampleRate": TEACHER_PARAMS.sample_rate,
                "nFft": TEACHER_PARAMS.n_fft,
                "hopLength": TEACHER_PARAMS.hop_length,
                "dimF": TEACHER_PARAMS.dim_f,
                "dimTPower": TEACHER_PARAMS.dim_t_power,
                "modelTimeFrames": TEACHER_PARAMS.dim_t,
                "trim": TEACHER_PARAMS.trim,
                "modelOutputScale": TEACHER_OUTPUT_SCALE,
            },
        },
        "cpuGpuParity": parity_report,
        "songs": song_reports,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": runner_revision,
                "gitRevision": git_revision(),
            },
        },
    }
    report_path = output_root / "reports" / "inst3-teacher-oracle-report.json"
    json_write(report_path, report)
    print(f"oracle report: {report_path}", flush=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selectedSongs": len(songs),
                "providers": gpu_providers,
                "parity": parity_report.get("aggregate"),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
