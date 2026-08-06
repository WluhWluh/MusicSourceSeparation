#!/usr/bin/env python3
"""Attribute guitar-ft cross-stem differences across Torch and S25 LiteRT outputs.

This is a no-ground-truth diagnostic. It uses one fixed drum-transient mask derived
from official Torch drums and one vocal-active mask derived from official Torch
vocals, then applies those masks to every backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import wave
from typing import Any

import numpy as np


SAMPLE_RATE = 44_100
CHANNELS = 2
SAMPLE_WIDTH = 2
STEMS = ("vocals", "drums", "piano")
VOCAL_DRUM_PARITY_STEMS = ("vocals", "drums")
PIANO_VOCAL_PARITY_STEMS = ("vocals", "piano")
BACKEND_ORDER = ("officialTorch", "guitarFtTorch", "s25GuitarFtLiteRt")
FFT_SIZE = 2_048
HOP_SIZE = 512
BAND_LOW_HZ = 2_000.0
BAND_HIGH_HZ = 8_000.0
PIANO_VOCAL_BAND_LOW_HZ = 300.0
PIANO_VOCAL_BAND_HIGH_HZ = 8_000.0
TRANSIENT_PERCENTILE = 85.0
TRANSIENT_DILATION_FRAMES = 1
VOCAL_ACTIVE_FRACTION = 0.40
VOCAL_INACTIVE_FRACTION = 0.40
VOCAL_ACTIVITY_SMOOTHING_FRAMES = 5
FFT_BATCH_FRAMES = 256
EPSILON = 1e-30
DEFAULT_DURATION_SECONDS = 30.0
EXPECTED_S25_MODEL_ID = (
    "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0"
)
EXPECTED_GUITAR_FT_VARIANT_ID = "htdemucs_6s_guitar_ft_host_fp32_v1_0_0"
EXPECTED_OFFICIAL_WEIGHT_SHA256 = (
    "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
)

# This screen operates on PCM16 files, not converter float tensors. A low-signal
# stem therefore needs an LSB-aware criterion instead of the float fixture gate.
MINIMUM_SNR_DB = 80.0
MAXIMUM_ABSOLUTE_ERROR = 1e-3
LOW_SIGNAL_REFERENCE_RMS = 1e-3
LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR_LSB = 2
LOW_SIGNAL_MINIMUM_CORRELATION = 0.9999


@dataclass(frozen=True)
class BackendInput:
    name: str
    stem_root: Path
    report_path: Path


@dataclass
class BackendAudio:
    input: BackendInput
    report: dict[str, Any]
    report_evidence: dict[str, Any]
    stems: dict[str, np.ndarray]
    files: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", required=True, type=Path)
    parser.add_argument("--official-report", required=True, type=Path)
    parser.add_argument("--guitar-ft-torch-dir", required=True, type=Path)
    parser.add_argument("--guitar-ft-torch-report", required=True, type=Path)
    parser.add_argument("--s25-dir", required=True, type=Path)
    parser.add_argument("--s25-report", required=True, type=Path)
    parser.add_argument("--track", required=True)
    parser.add_argument("--corpus-role", required=True)
    parser.add_argument(
        "--expected-duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS
    )
    parser.add_argument("--expected-s25-model-id", default=EXPECTED_S25_MODEL_ID)
    parser.add_argument(
        "--expected-guitar-ft-variant-id", default=EXPECTED_GUITAR_FT_VARIANT_ID
    )
    parser.add_argument(
        "--expected-official-weight-sha256",
        default=EXPECTED_OFFICIAL_WEIGHT_SHA256,
    )
    parser.add_argument(
        "--allow-model-identity-mismatch",
        action="store_true",
        help="Only for mechanical fixtures; real experiment reports must keep strict identity checks.",
    )
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def source_identity(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("source")
    if not isinstance(source, dict):
        raise ValueError("Report does not contain a source object")
    frames = source.get("selectedFrames")
    pcm_sha = source.get("selectedPcmSha256")
    if not isinstance(frames, int) or frames <= 0:
        raise ValueError(f"Invalid selectedFrames in report: {frames!r}")
    if not isinstance(pcm_sha, str) or len(pcm_sha) != 64:
        raise ValueError(f"Invalid selectedPcmSha256 in report: {pcm_sha!r}")
    return {
        "selectedFrames": frames,
        "durationSeconds": source.get("durationSeconds", frames / SAMPLE_RATE),
        "selectedPcmSha256": pcm_sha,
        "selectedStartFrame": source.get("selectedStartFrame"),
        "sampleRate": source.get("sampleRate", SAMPLE_RATE),
        "channelCount": source.get("channelCount", CHANNELS),
    }


def model_identity(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    variant = report.get("variant") if isinstance(report.get("variant"), dict) else {}
    provenance = (
        report.get("provenance") if isinstance(report.get("provenance"), dict) else {}
    )
    candidate_weight = provenance.get("safetensorsArtifact")
    official_weight = provenance.get("weight")
    if isinstance(candidate_weight, dict):
        weight = candidate_weight
    elif isinstance(official_weight, dict):
        weight = official_weight
    else:
        weight = {}
    canonical = (
        provenance.get("canonicalManifest")
        if isinstance(provenance.get("canonicalManifest"), dict)
        else {}
    )
    return {
        "backend": report.get("backend"),
        "modelId": model.get("modelId") or canonical.get("modelId"),
        "modelSha256": model.get("sha256"),
        "variantId": (
            variant.get("candidateId")
            or variant.get("variantId")
            or canonical.get("candidateId")
        ),
        "weightSha256": weight.get("sha256"),
        "weightMutation": provenance.get("weightMutation"),
        "diagnosticOnly": variant.get("diagnosticOnly", provenance.get("diagnosticOnly")),
        "researchOnly": variant.get("researchOnly", provenance.get("researchOnly")),
    }


def validate_model_identities(
    args: argparse.Namespace, backends: dict[str, BackendAudio]
) -> dict[str, Any]:
    actual = {
        name: backend.report_evidence["model"] for name, backend in backends.items()
    }
    checks = {
        "s25ModelId": {
            "expected": args.expected_s25_model_id,
            "actual": actual["s25GuitarFtLiteRt"]["modelId"],
        },
        "guitarFtTorchVariantId": {
            "expected": args.expected_guitar_ft_variant_id,
            "actual": actual["guitarFtTorch"]["variantId"],
        },
        "officialTorchWeightSha256": {
            "expected": args.expected_official_weight_sha256,
            "actual": actual["officialTorch"]["weightSha256"],
        },
    }
    for value in checks.values():
        value["passed"] = value["actual"] == value["expected"]
    passed = all(value["passed"] for value in checks.values())
    if not passed and not args.allow_model_identity_mismatch:
        raise ValueError(f"Model identity check failed: {checks}")
    return {
        "passed": passed,
        "mismatchAllowedForMechanicalFixture": bool(
            not passed and args.allow_model_identity_mismatch
        ),
        "checks": checks,
    }


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with wave.open(str(path), "rb") as reader:
        contract = {
            "sampleRate": reader.getframerate(),
            "channelCount": reader.getnchannels(),
            "sampleWidthBytes": reader.getsampwidth(),
            "frameCount": reader.getnframes(),
            "compressionType": reader.getcomptype(),
        }
        expected = (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, "NONE")
        actual = (
            contract["sampleRate"],
            contract["channelCount"],
            contract["sampleWidthBytes"],
            contract["compressionType"],
        )
        if actual != expected:
            raise ValueError(f"Unexpected WAV contract for {path}: {actual}")
        payload = reader.readframes(contract["frameCount"])
    expected_bytes = contract["frameCount"] * CHANNELS * SAMPLE_WIDTH
    if len(payload) != expected_bytes:
        raise ValueError(f"Short PCM read for {path}: {len(payload)} != {expected_bytes}")
    audio = np.frombuffer(payload, dtype="<i2").reshape(-1, CHANNELS).copy()
    return audio, {
        "path": str(path.resolve()),
        "byteSize": path.stat().st_size,
        "sha256": sha256_file(path),
        "contract": contract,
    }


def load_backend(value: BackendInput) -> BackendAudio:
    if not value.report_path.is_file():
        raise FileNotFoundError(value.report_path)
    report = json.loads(value.report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"Incomplete report for {value.name}: {report.get('status')!r}")
    identity = source_identity(report)
    stems: dict[str, np.ndarray] = {}
    files: dict[str, dict[str, Any]] = {}
    for stem in STEMS:
        stems[stem], files[stem] = read_pcm16(value.stem_root / f"{stem}.wav")
        if files[stem]["contract"]["frameCount"] != identity["selectedFrames"]:
            raise ValueError(
                f"{value.name} {stem} frame count does not match its source report"
            )
    return BackendAudio(
        input=value,
        report=report,
        report_evidence={
            "path": str(value.report_path.resolve()),
            "byteSize": value.report_path.stat().st_size,
            "sha256": sha256_file(value.report_path),
            "status": report["status"],
            "source": identity,
            "model": model_identity(report),
        },
        stems=stems,
        files=files,
    )


def rms(value: np.ndarray) -> float:
    flat = value.astype(np.float64, copy=False).reshape(-1)
    return math.sqrt(float(np.dot(flat, flat)) / max(flat.size, 1))


def db10_ratio(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10(max(numerator, EPSILON) / max(denominator, EPSILON))


def db20_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, EPSILON) / max(denominator, EPSILON))


def correlation(left: np.ndarray, right: np.ndarray, *, centered: bool = True) -> float | None:
    left64 = left.astype(np.float64, copy=False).reshape(-1)
    right64 = right.astype(np.float64, copy=False).reshape(-1)
    if left64.size != right64.size:
        raise ValueError("Correlation inputs have different sizes")
    if centered:
        left64 = left64 - float(left64.mean())
        right64 = right64 - float(right64.mean())
    denominator = math.sqrt(float(np.dot(left64, left64) * np.dot(right64, right64)))
    if denominator <= EPSILON:
        return None
    return float(np.dot(left64, right64) / denominator)


def waveform_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Waveform shape mismatch: {reference.shape} != {candidate.shape}")
    reference_i32 = reference.astype(np.int32)
    candidate_i32 = candidate.astype(np.int32)
    error_lsb = candidate_i32 - reference_i32
    reference_float = reference_i32.astype(np.float64) / 32768.0
    candidate_float = candidate_i32.astype(np.float64) / 32768.0
    error_float = error_lsb.astype(np.float64) / 32768.0
    reference_rms = rms(reference_float)
    candidate_rms = rms(candidate_float)
    error_rms = rms(error_float)
    maximum_lsb = int(np.abs(error_lsb).max(initial=0))
    evidence = {
        "sampleCount": int(reference.size),
        "differentSampleCount": int(np.count_nonzero(error_lsb)),
        "differentSampleFraction": float(np.count_nonzero(error_lsb) / reference.size),
        "referenceRms": reference_rms,
        "candidateRms": candidate_rms,
        "candidateVsReferenceLevelDb": db20_ratio(candidate_rms, reference_rms),
        "rootMeanSquareError": error_rms,
        "rootMeanSquareErrorLsb": error_rms * 32768.0,
        "maximumAbsoluteError": maximum_lsb / 32768.0,
        "maximumAbsoluteErrorLsb": maximum_lsb,
        "snrDb": db20_ratio(reference_rms, error_rms),
        "correlation": correlation(reference_float, candidate_float),
    }
    low_signal = reference_rms <= LOW_SIGNAL_REFERENCE_RMS
    if low_signal:
        passed = bool(
            maximum_lsb <= LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR_LSB
            and evidence["correlation"] is not None
            and evidence["correlation"] >= LOW_SIGNAL_MINIMUM_CORRELATION
        )
        basis = "pcm16-low-signal-absolute-and-correlation"
    else:
        passed = bool(
            evidence["snrDb"] >= MINIMUM_SNR_DB
            and evidence["maximumAbsoluteError"] <= MAXIMUM_ABSOLUTE_ERROR
        )
        basis = "snr-and-absolute"
    evidence["pcm16ParityScreen"] = {
        "passed": passed,
        "basis": basis,
        "lowSignal": low_signal,
    }
    return evidence


def relationship(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_float = left.astype(np.float64, copy=False) / 32768.0
    right_float = right.astype(np.float64, copy=False) / 32768.0
    cosine = correlation(left_float, right_float, centered=False)
    return {
        "normalizedWaveformCorrelation": correlation(left_float, right_float),
        "waveformCosineSimilarity": cosine,
        "squaredCosineProjectionFraction": None if cosine is None else cosine * cosine,
    }


def float_relationship(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    cosine = correlation(left, right, centered=False)
    return {
        "normalizedWaveformCorrelation": correlation(left, right),
        "waveformCosineSimilarity": cosine,
        "squaredCosineProjectionFraction": None if cosine is None else cosine * cosine,
    }


def spectral_features(
    audio_pcm16: np.ndarray,
    *,
    include_flux: bool,
    band_low_hz: float = BAND_LOW_HZ,
    band_high_hz: float = BAND_HIGH_HZ,
    include_band_magnitude: bool = False,
) -> dict[str, np.ndarray]:
    audio = audio_pcm16.astype(np.float32) / 32768.0
    if audio.shape[0] < FFT_SIZE:
        raise ValueError(f"Audio is shorter than FFT size: {audio.shape[0]} < {FFT_SIZE}")
    framed = np.lib.stride_tricks.sliding_window_view(audio, FFT_SIZE, axis=0)[::HOP_SIZE]
    window = np.hanning(FFT_SIZE).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    if not 0.0 <= band_low_hz < band_high_hz <= SAMPLE_RATE / 2.0:
        raise ValueError(f"Invalid frequency band: {band_low_hz}..{band_high_hz}")
    band = (frequencies >= band_low_hz) & (frequencies <= band_high_hz)
    band_power = np.empty(framed.shape[0], dtype=np.float64)
    band_magnitude = (
        np.empty((framed.shape[0], int(band.sum())), dtype=np.float64)
        if include_band_magnitude
        else None
    )
    flux = np.zeros(framed.shape[0], dtype=np.float64) if include_flux else None
    previous_log_magnitude: np.ndarray | None = None
    for start in range(0, framed.shape[0], FFT_BATCH_FRAMES):
        end = min(framed.shape[0], start + FFT_BATCH_FRAMES)
        batch = np.asarray(framed[start:end] * window[None, None, :], dtype=np.float32)
        spectrum = np.fft.rfft(batch, axis=-1)
        power = np.square(spectrum.real) + np.square(spectrum.imag)
        band_power[start:end] = power[:, :, band].sum(axis=(1, 2), dtype=np.float64)
        if band_magnitude is not None:
            band_magnitude[start:end] = np.sqrt(
                power[:, :, band].sum(axis=1, dtype=np.float64)
            )
        if flux is not None:
            magnitude = np.sqrt(power.sum(axis=1, dtype=np.float64))
            log_magnitude = np.log1p(magnitude)
            if previous_log_magnitude is None:
                differences = np.diff(log_magnitude, axis=0, prepend=log_magnitude[:1])
            else:
                differences = np.diff(
                    log_magnitude,
                    axis=0,
                    prepend=previous_log_magnitude[None, :],
                )
            positive = np.maximum(differences, 0.0)
            flux[start:end] = np.sqrt(np.mean(np.square(positive), axis=1))
            previous_log_magnitude = log_magnitude[-1]
    result = {"bandPower": band_power}
    if band_magnitude is not None:
        result["bandMagnitude"] = band_magnitude
    if flux is not None:
        result["positiveLogSpectralFlux"] = flux
    return result


def transient_mask(flux: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if flux.size < 2:
        raise ValueError("Not enough spectral frames for transient detection")
    candidates = flux[1:]
    seed_count = max(1, int(math.ceil(candidates.size * (100.0 - TRANSIENT_PERCENTILE) / 100.0)))
    top_indices = np.argpartition(candidates, -seed_count)[-seed_count:] + 1
    seed = np.zeros(flux.size, dtype=bool)
    seed[top_indices] = True
    mask = seed.copy()
    for distance in range(1, TRANSIENT_DILATION_FRAMES + 1):
        mask[distance:] |= seed[:-distance]
        mask[:-distance] |= seed[distance:]
    threshold = float(np.min(flux[top_indices]))
    return mask, {
        "detectorSource": "officialTorch.drums",
        "detectorFeature": "positive-log-spectral-flux-all-rfft-bins",
        "selection": "top-ranked-frames",
        "requestedPercentile": TRANSIENT_PERCENTILE,
        "seedFrameCount": int(seed.sum()),
        "expandedFrameCount": int(mask.sum()),
        "totalFrameCount": int(mask.size),
        "expandedFrameFraction": float(mask.mean()),
        "dilationFramesEachSide": TRANSIENT_DILATION_FRAMES,
        "threshold": threshold,
        "maximum": float(flux.max(initial=0.0)),
        "detectorReliable": bool(float(flux.max(initial=0.0)) > 1e-12),
    }


def vocal_activity_masks(
    official_vocal_band_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if official_vocal_band_power.size < VOCAL_ACTIVITY_SMOOTHING_FRAMES:
        raise ValueError("Not enough spectral frames for vocal-activity detection")
    floor = max(float(official_vocal_band_power.max(initial=0.0)) * 1e-12, EPSILON)
    log_power_db = 10.0 * np.log10(np.maximum(official_vocal_band_power, floor))
    half = VOCAL_ACTIVITY_SMOOTHING_FRAMES // 2
    padded = np.pad(log_power_db, (half, half), mode="edge")
    smoothed = np.convolve(
        padded,
        np.full(
            VOCAL_ACTIVITY_SMOOTHING_FRAMES,
            1.0 / VOCAL_ACTIVITY_SMOOTHING_FRAMES,
        ),
        mode="valid",
    )
    frame_count = smoothed.size
    active_count = max(1, int(math.floor(frame_count * VOCAL_ACTIVE_FRACTION)))
    inactive_count = max(1, int(math.floor(frame_count * VOCAL_INACTIVE_FRACTION)))
    if active_count + inactive_count > frame_count:
        raise ValueError("Vocal active and inactive selections overlap")
    ranked_indices = np.argsort(smoothed, kind="stable")
    inactive_indices = ranked_indices[:inactive_count]
    active_indices = ranked_indices[-active_count:]
    active = np.zeros(frame_count, dtype=bool)
    inactive = np.zeros(frame_count, dtype=bool)
    active[active_indices] = True
    inactive[inactive_indices] = True
    if np.any(active & inactive):
        raise ValueError("Vocal activity masks unexpectedly overlap")
    dynamic_range_db = float(np.percentile(smoothed, 90) - np.percentile(smoothed, 10))
    return active, inactive, {
        "detectorSource": "officialTorch.vocals",
        "detectorFeature": "smoothed-log-300-to-8000-hz-energy",
        "selection": "fixed-top-and-bottom-ranked-frames",
        "activeFractionRequested": VOCAL_ACTIVE_FRACTION,
        "inactiveFractionRequested": VOCAL_INACTIVE_FRACTION,
        "smoothingFrames": VOCAL_ACTIVITY_SMOOTHING_FRAMES,
        "activeFrameCount": int(active.sum()),
        "inactiveFrameCount": int(inactive.sum()),
        "unclassifiedFrameCount": int(frame_count - active.sum() - inactive.sum()),
        "totalFrameCount": int(frame_count),
        "activeMinimumSmoothedDb": float(smoothed[active].min()),
        "inactiveMaximumSmoothedDb": float(smoothed[inactive].max()),
        "p90MinusP10Db": dynamic_range_db,
        "detectorReliable": bool(dynamic_range_db >= 6.0),
    }


def spectral_cosine_summary(
    left_magnitude: np.ndarray,
    right_magnitude: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if left_magnitude.shape != right_magnitude.shape:
        raise ValueError("Band-magnitude shapes do not match")
    if left_magnitude.shape[0] != mask.size:
        raise ValueError("Band-magnitude frame count does not match the mask")
    left = left_magnitude[mask]
    right = right_magnitude[mask]
    if left.size == 0:
        raise ValueError("Spectral cosine selection is empty")
    row_denominator = np.sqrt(
        np.sum(np.square(left), axis=1) * np.sum(np.square(right), axis=1)
    )
    valid = row_denominator > EPSILON
    row_cosine = np.divide(
        np.sum(left * right, axis=1),
        row_denominator,
        out=np.zeros_like(row_denominator),
        where=valid,
    )
    return {
        "globalMagnitudeCosineSimilarity": correlation(
            left.reshape(-1), right.reshape(-1), centered=False
        ),
        "meanPerFrameMagnitudeCosineSimilarity": (
            float(row_cosine[valid].mean()) if np.any(valid) else None
        ),
        "medianPerFrameMagnitudeCosineSimilarity": (
            float(np.median(row_cosine[valid])) if np.any(valid) else None
        ),
        "validFrameCount": int(valid.sum()),
        "selectedFrameCount": int(mask.sum()),
    }


def frame_selection_to_center_samples(
    audio: np.ndarray, frame_mask: np.ndarray
) -> np.ndarray:
    sample_mask = np.zeros(audio.shape[0], dtype=bool)
    center_offset = (FFT_SIZE - HOP_SIZE) // 2
    for frame_index in np.flatnonzero(frame_mask):
        start = int(frame_index) * HOP_SIZE + center_offset
        end = min(audio.shape[0], start + HOP_SIZE)
        if start < audio.shape[0]:
            sample_mask[start:end] = True
    return audio[sample_mask]


def mean_selected(value: np.ndarray, mask: np.ndarray) -> float:
    selected = value[mask]
    if selected.size == 0:
        raise ValueError("Transient selection is empty")
    return float(selected.mean())


def backend_leakage_metrics(
    backend: BackendAudio,
    mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    vocals_features = spectral_features(backend.stems["vocals"], include_flux=False)
    drums_features = spectral_features(backend.stems["drums"], include_flux=False)
    vocals_band = vocals_features["bandPower"]
    drums_band = drums_features["bandPower"]
    if vocals_band.shape != mask.shape or drums_band.shape != mask.shape:
        raise ValueError(f"Spectral frame mismatch for {backend.input.name}")
    vocals_transient = mean_selected(vocals_band, mask)
    vocals_non_transient = mean_selected(vocals_band, ~mask)
    drums_transient = mean_selected(drums_band, mask)
    drums_non_transient = mean_selected(drums_band, ~mask)
    log_vocals = np.log10(np.maximum(vocals_band, EPSILON))
    log_drums = np.log10(np.maximum(drums_band, EPSILON))
    result = {
        "vocalDrumRelationship": {
            **relationship(backend.stems["vocals"], backend.stems["drums"]),
            "log2To8KhzEnergyEnvelopeCorrelation": correlation(log_vocals, log_drums),
        },
        "transientConditioned2To8Khz": {
            "vocalMeanPowerTransient": vocals_transient,
            "vocalMeanPowerNonTransient": vocals_non_transient,
            "drumMeanPowerTransient": drums_transient,
            "drumMeanPowerNonTransient": drums_non_transient,
            "vocalTransientToNonTransientDb": db10_ratio(
                vocals_transient, vocals_non_transient
            ),
            "drumTransientToNonTransientDb": db10_ratio(
                drums_transient, drums_non_transient
            ),
            "vocalToDrumTransientDb": db10_ratio(vocals_transient, drums_transient),
            "vocalTransientEnergyFraction": float(
                vocals_band[mask].sum() / max(float(vocals_band.sum()), EPSILON)
            ),
            "drumTransientEnergyFraction": float(
                drums_band[mask].sum() / max(float(drums_band.sum()), EPSILON)
            ),
        },
    }
    return result, {"vocalsBandPower": vocals_band, "drumsBandPower": drums_band}


def piano_vocal_backend_metrics(
    backend: BackendAudio,
    official: BackendAudio,
    official_vocal_features: dict[str, np.ndarray],
    active_mask: np.ndarray,
    inactive_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    vocals_features = spectral_features(
        backend.stems["vocals"],
        include_flux=False,
        band_low_hz=PIANO_VOCAL_BAND_LOW_HZ,
        band_high_hz=PIANO_VOCAL_BAND_HIGH_HZ,
        include_band_magnitude=True,
    )
    piano_features = spectral_features(
        backend.stems["piano"],
        include_flux=False,
        band_low_hz=PIANO_VOCAL_BAND_LOW_HZ,
        band_high_hz=PIANO_VOCAL_BAND_HIGH_HZ,
        include_band_magnitude=True,
    )
    for features in (vocals_features, piano_features, official_vocal_features):
        if features["bandPower"].shape != active_mask.shape:
            raise ValueError(f"Piano/vocal spectral frame mismatch for {backend.input.name}")

    piano_band = piano_features["bandPower"]
    own_vocal_band = vocals_features["bandPower"]
    fixed_vocal_band = official_vocal_features["bandPower"]
    piano_active = mean_selected(piano_band, active_mask)
    piano_inactive = mean_selected(piano_band, inactive_mask)
    own_vocal_active = mean_selected(own_vocal_band, active_mask)
    fixed_vocal_active = mean_selected(fixed_vocal_band, active_mask)

    piano_active_samples = frame_selection_to_center_samples(
        backend.stems["piano"], active_mask
    )
    own_vocal_active_samples = frame_selection_to_center_samples(
        backend.stems["vocals"], active_mask
    )
    fixed_vocal_active_samples = frame_selection_to_center_samples(
        official.stems["vocals"], active_mask
    )
    fixed_spectral_active = spectral_cosine_summary(
        piano_features["bandMagnitude"],
        official_vocal_features["bandMagnitude"],
        active_mask,
    )
    fixed_spectral_inactive = spectral_cosine_summary(
        piano_features["bandMagnitude"],
        official_vocal_features["bandMagnitude"],
        inactive_mask,
    )
    own_spectral_active = spectral_cosine_summary(
        piano_features["bandMagnitude"],
        vocals_features["bandMagnitude"],
        active_mask,
    )
    result = {
        "pianoVsFixedOfficialVocals": {
            "fullWaveform": relationship(
                backend.stems["piano"], official.stems["vocals"]
            ),
            "vocalActiveWaveform": relationship(
                piano_active_samples, fixed_vocal_active_samples
            ),
            "vocalActiveSpectralMagnitude": fixed_spectral_active,
            "vocalInactiveSpectralMagnitude": fixed_spectral_inactive,
        },
        "pianoVsOwnBackendVocals": {
            "fullWaveform": relationship(
                backend.stems["piano"], backend.stems["vocals"]
            ),
            "vocalActiveWaveform": relationship(
                piano_active_samples, own_vocal_active_samples
            ),
            "vocalActiveSpectralMagnitude": own_spectral_active,
        },
        "vocalActivityConditioned300To8000Hz": {
            "pianoMeanPowerVocalActive": piano_active,
            "pianoMeanPowerVocalInactive": piano_inactive,
            "fixedOfficialVocalMeanPowerActive": fixed_vocal_active,
            "ownBackendVocalMeanPowerActive": own_vocal_active,
            "pianoActiveToInactiveDb": db10_ratio(piano_active, piano_inactive),
            "pianoToFixedOfficialVocalActiveDb": db10_ratio(
                piano_active, fixed_vocal_active
            ),
            "pianoToOwnBackendVocalActiveDb": db10_ratio(
                piano_active, own_vocal_active
            ),
            "pianoActiveEnergyFractionOfSelectedFrames": float(
                piano_band[active_mask].sum()
                / max(
                    float(
                        piano_band[active_mask].sum()
                        + piano_band[inactive_mask].sum()
                    ),
                    EPSILON,
                )
            ),
        },
    }
    return result, {
        "pianoBandPower": piano_band,
        "pianoBandMagnitude": piano_features["bandMagnitude"],
        "vocalsBandPower": own_vocal_band,
        "vocalsBandMagnitude": vocals_features["bandMagnitude"],
    }


def delta_metrics(
    candidate: BackendAudio,
    reference: BackendAudio,
    official: BackendAudio,
    mask: np.ndarray,
    official_drum_band: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    vocal_delta = (
        candidate.stems["vocals"].astype(np.float64)
        - reference.stems["vocals"].astype(np.float64)
    ) / 32768.0
    drum_delta = (
        candidate.stems["drums"].astype(np.float64)
        - reference.stems["drums"].astype(np.float64)
    ) / 32768.0
    official_drums = official.stems["drums"].astype(np.float64) / 32768.0
    # spectral_features accepts PCM-scale values, so preserve the delta exactly by
    # mapping one float LSB back to the same integer scale without clipping.
    vocal_delta_lsb = vocal_delta * 32768.0
    delta_band = spectral_features(vocal_delta_lsb, include_flux=False)["bandPower"]
    delta_transient = mean_selected(delta_band, mask)
    delta_non_transient = mean_selected(delta_band, ~mask)
    official_drum_transient = mean_selected(official_drum_band, mask)
    result = {
        "candidate": candidate.input.name,
        "reference": reference.input.name,
        "vocalDeltaRms": rms(vocal_delta),
        "drumDeltaRms": rms(drum_delta),
        "vocalDeltaVsDrumDelta": float_relationship(vocal_delta, drum_delta),
        "vocalDeltaVsOfficialDrums": float_relationship(vocal_delta, official_drums),
        "vocalDeltaTransient2To8Khz": {
            "meanPowerTransient": delta_transient,
            "meanPowerNonTransient": delta_non_transient,
            "transientToNonTransientDb": db10_ratio(delta_transient, delta_non_transient),
            "toOfficialDrumTransientDb": db10_ratio(
                delta_transient, official_drum_transient
            ),
            "transientEnergyFraction": float(
                delta_band[mask].sum() / max(float(delta_band.sum()), EPSILON)
            ),
        },
    }
    return result, delta_band


def piano_vocal_delta_metrics(
    candidate: BackendAudio,
    reference: BackendAudio,
    official: BackendAudio,
    active_mask: np.ndarray,
    inactive_mask: np.ndarray,
    official_vocal_features: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    piano_delta = (
        candidate.stems["piano"].astype(np.float64)
        - reference.stems["piano"].astype(np.float64)
    ) / 32768.0
    vocal_delta = (
        candidate.stems["vocals"].astype(np.float64)
        - reference.stems["vocals"].astype(np.float64)
    ) / 32768.0
    removed_from_piano = -piano_delta
    official_vocals = official.stems["vocals"].astype(np.float64) / 32768.0
    delta_features = spectral_features(
        piano_delta * 32768.0,
        include_flux=False,
        band_low_hz=PIANO_VOCAL_BAND_LOW_HZ,
        band_high_hz=PIANO_VOCAL_BAND_HIGH_HZ,
        include_band_magnitude=True,
    )
    delta_band = delta_features["bandPower"]
    delta_active = mean_selected(delta_band, active_mask)
    delta_inactive = mean_selected(delta_band, inactive_mask)
    official_vocal_active_power = mean_selected(
        official_vocal_features["bandPower"], active_mask
    )
    active_removed = frame_selection_to_center_samples(
        removed_from_piano, active_mask
    )
    active_official_vocals = frame_selection_to_center_samples(
        official_vocals, active_mask
    )
    active_piano_delta = frame_selection_to_center_samples(piano_delta, active_mask)
    active_vocal_delta = frame_selection_to_center_samples(vocal_delta, active_mask)
    return {
        "candidate": candidate.input.name,
        "reference": reference.input.name,
        "pianoDeltaRms": rms(piano_delta),
        "vocalDeltaRms": rms(vocal_delta),
        "removedFromPianoVsOfficialVocals": {
            "fullWaveform": float_relationship(removed_from_piano, official_vocals),
            "vocalActiveWaveform": float_relationship(
                active_removed, active_official_vocals
            ),
            "vocalActiveSpectralMagnitude": spectral_cosine_summary(
                delta_features["bandMagnitude"],
                official_vocal_features["bandMagnitude"],
                active_mask,
            ),
        },
        "pianoDeltaVsVocalDelta": {
            "fullWaveform": float_relationship(piano_delta, vocal_delta),
            "vocalActiveWaveform": float_relationship(
                active_piano_delta, active_vocal_delta
            ),
        },
        "pianoDeltaVocalActivityConditioned300To8000Hz": {
            "meanPowerVocalActive": delta_active,
            "meanPowerVocalInactive": delta_inactive,
            "activeToInactiveDb": db10_ratio(delta_active, delta_inactive),
            "toOfficialVocalActiveDb": db10_ratio(
                delta_active, official_vocal_active_power
            ),
            "activeEnergyFractionOfSelectedFrames": float(
                delta_band[active_mask].sum()
                / max(
                    float(
                        delta_band[active_mask].sum()
                        + delta_band[inactive_mask].sum()
                    ),
                    EPSILON,
                )
            ),
        },
    }, delta_band


def parity_pairs(backends: dict[str, BackendAudio]) -> dict[str, Any]:
    definitions = {
        "conversion": ("guitarFtTorch", "s25GuitarFtLiteRt"),
        "weightChange": ("officialTorch", "guitarFtTorch"),
        "totalObserved": ("officialTorch", "s25GuitarFtLiteRt"),
    }
    result: dict[str, Any] = {}
    for pair_name, (reference_name, candidate_name) in definitions.items():
        reference = backends[reference_name]
        candidate = backends[candidate_name]
        per_stem = {
            stem: waveform_comparison(reference.stems[stem], candidate.stems[stem])
            for stem in STEMS
        }
        result[pair_name] = {
            "reference": reference_name,
            "candidate": candidate_name,
            "perStem": per_stem,
            # Preserve the original vocals/drums screen meaning for existing reports.
            "pcm16ParityScreenPassed": all(
                per_stem[stem]["pcm16ParityScreen"]["passed"]
                for stem in VOCAL_DRUM_PARITY_STEMS
            ),
            "vocalDrumPcm16ParityScreenPassed": all(
                per_stem[stem]["pcm16ParityScreen"]["passed"]
                for stem in VOCAL_DRUM_PARITY_STEMS
            ),
            "pianoVocalPcm16ParityScreenPassed": all(
                per_stem[stem]["pcm16ParityScreen"]["passed"]
                for stem in PIANO_VOCAL_PARITY_STEMS
            ),
            "allAnalyzedStemsPcm16ParityScreenPassed": all(
                item["pcm16ParityScreen"]["passed"] for item in per_stem.values()
            ),
        }
    return result


def attribution(
    pairs: dict[str, Any],
    backend_metrics: dict[str, Any],
    deltas: dict[str, Any],
    conversion_delta_band: np.ndarray,
    weight_delta_band: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    conversion_power = mean_selected(conversion_delta_band, mask)
    weight_power = mean_selected(weight_delta_band, mask)
    conversion_vs_weight_db = db10_ratio(conversion_power, weight_power)
    leakage_delta_db = (
        backend_metrics["s25GuitarFtLiteRt"]["transientConditioned2To8Khz"][
            "vocalToDrumTransientDb"
        ]
        - backend_metrics["guitarFtTorch"]["transientConditioned2To8Khz"][
            "vocalToDrumTransientDb"
        ]
    )
    parity_passed = pairs["conversion"]["pcm16ParityScreenPassed"]
    if parity_passed and conversion_vs_weight_db <= -20.0:
        classification = "weight-change-dominant"
        summary = (
            "The S25 output passes the PCM16 conversion screen and its transient-band "
            "vocal delta is at least 20 dB below the guitar-ft weight delta. The measured "
            "vocal/drum change is therefore dominated by the fine-tuned weights."
        )
    elif parity_passed and conversion_vs_weight_db <= -10.0:
        classification = "probably-weight-change-dominant"
        summary = (
            "The S25 output passes the PCM16 conversion screen and its transient-band "
            "vocal delta is materially below the weight delta. A smaller conversion "
            "contribution is still measurable and should be inspected."
        )
    elif not parity_passed:
        classification = "conversion-contribution-not-excluded"
        summary = (
            "The S25 output does not pass the PCM16 conversion screen, so the observed "
            "vocal/drum difference cannot be attributed to the fine-tuned weights alone."
        )
    else:
        classification = "inconclusive"
        summary = (
            "Waveform parity is acceptable, but the conversion delta is not sufficiently "
            "below the weight delta for a strong attribution."
        )
    return {
        "classification": classification,
        "summary": summary,
        "conversionPcm16ParityScreenPassed": parity_passed,
        "conversionVsWeightVocalDeltaTransient2To8KhzDb": conversion_vs_weight_db,
        "s25MinusGuitarFtTorchVocalToDrumTransientDb": leakage_delta_db,
        "weightChangeVocalDeltaVsDrumDeltaCorrelation": deltas["weightChange"][
            "vocalDeltaVsDrumDelta"
        ]["normalizedWaveformCorrelation"],
        "conversionVocalDeltaVsDrumDeltaCorrelation": deltas["conversion"][
            "vocalDeltaVsDrumDelta"
        ]["normalizedWaveformCorrelation"],
        "limitations": [
            "No isolated ground-truth stems are available; correlation and energy ratios are diagnostics, not leakage SDR.",
            "Musical vocals and drums often share timing, so positive envelope correlation can occur without audible contamination.",
            "The 2-8 kHz band emphasizes snare and cymbal attacks but does not measure low-frequency kick leakage.",
            "The PCM16 parity screen cannot replace the float-tensor Torch/LiteRT export gate.",
        ],
    }


def piano_vocal_attribution(
    pairs: dict[str, Any],
    backend_metrics: dict[str, Any],
    deltas: dict[str, Any],
    conversion_delta_band: np.ndarray,
    weight_delta_band: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, Any]:
    conversion_power = mean_selected(conversion_delta_band, active_mask)
    weight_power = mean_selected(weight_delta_band, active_mask)
    conversion_vs_weight_db = db10_ratio(conversion_power, weight_power)
    parity_passed = pairs["conversion"]["pianoVocalPcm16ParityScreenPassed"]

    official_cosine = backend_metrics["officialTorch"]["pianoVsFixedOfficialVocals"][
        "vocalActiveSpectralMagnitude"
    ]["globalMagnitudeCosineSimilarity"]
    guitar_ft_cosine = backend_metrics["guitarFtTorch"][
        "pianoVsFixedOfficialVocals"
    ]["vocalActiveSpectralMagnitude"]["globalMagnitudeCosineSimilarity"]
    s25_cosine = backend_metrics["s25GuitarFtLiteRt"][
        "pianoVsFixedOfficialVocals"
    ]["vocalActiveSpectralMagnitude"]["globalMagnitudeCosineSimilarity"]
    weight_cosine_change = (
        guitar_ft_cosine - official_cosine
        if guitar_ft_cosine is not None and official_cosine is not None
        else None
    )
    conversion_cosine_change = (
        s25_cosine - guitar_ft_cosine
        if s25_cosine is not None and guitar_ft_cosine is not None
        else None
    )
    removed_cosine = deltas["weightChange"]["removedFromPianoVsOfficialVocals"][
        "vocalActiveWaveform"
    ]["waveformCosineSimilarity"]
    weight_power_ratio_change = (
        backend_metrics["guitarFtTorch"][
            "vocalActivityConditioned300To8000Hz"
        ]["pianoToFixedOfficialVocalActiveDb"]
        - backend_metrics["officialTorch"][
            "vocalActivityConditioned300To8000Hz"
        ]["pianoToFixedOfficialVocalActiveDb"]
    )
    conversion_power_ratio_change = (
        backend_metrics["s25GuitarFtLiteRt"][
            "vocalActivityConditioned300To8000Hz"
        ]["pianoToFixedOfficialVocalActiveDb"]
        - backend_metrics["guitarFtTorch"][
            "vocalActivityConditioned300To8000Hz"
        ]["pianoToFixedOfficialVocalActiveDb"]
    )

    if parity_passed and conversion_vs_weight_db <= -20.0:
        classification = "weight-change-dominant"
        summary = (
            "The S25 piano/vocal outputs pass the PCM16 conversion screen and the "
            "conversion piano delta on vocal-active frames is at least 20 dB below the "
            "fine-tune weight delta. The measured piano/vocal change is dominated by "
            "the guitar-ft weights."
        )
    elif parity_passed and conversion_vs_weight_db <= -10.0:
        classification = "probably-weight-change-dominant"
        summary = (
            "The S25 piano/vocal outputs pass the PCM16 conversion screen and the "
            "conversion piano delta is materially below the weight delta, although a "
            "smaller conversion contribution remains measurable."
        )
    elif not parity_passed:
        classification = "conversion-contribution-not-excluded"
        summary = (
            "The S25 piano/vocal outputs do not pass the PCM16 conversion screen, so "
            "the piano change cannot be attributed to the fine-tuned weights alone."
        )
    else:
        classification = "inconclusive"
        summary = (
            "PCM16 parity is acceptable, but the conversion piano delta is not far "
            "enough below the weight delta for a strong attribution."
        )

    if (
        weight_cosine_change is not None
        and weight_cosine_change < 0.0
        and removed_cosine is not None
        and removed_cosine > 0.0
    ):
        listening_consistency = "directionally-consistent-with-reduced-vocal-leakage"
    elif (
        weight_cosine_change is not None
        and weight_cosine_change > 0.0
        and removed_cosine is not None
        and removed_cosine < 0.0
    ):
        listening_consistency = "directionally-opposed-to-reduced-vocal-leakage"
    else:
        listening_consistency = "mixed-or-inconclusive-proxies"

    return {
        "classification": classification,
        "summary": summary,
        "listeningObservationConsistency": listening_consistency,
        "conversionPianoVocalPcm16ParityScreenPassed": parity_passed,
        "conversionVsWeightPianoDeltaVocalActive300To8000HzDb": (
            conversion_vs_weight_db
        ),
        "guitarFtMinusOfficialFixedVocalActiveSpectralCosine": weight_cosine_change,
        "s25MinusGuitarFtFixedVocalActiveSpectralCosine": conversion_cosine_change,
        "guitarFtMinusOfficialPianoToFixedVocalActivePowerDb": (
            weight_power_ratio_change
        ),
        "s25MinusGuitarFtPianoToFixedVocalActivePowerDb": (
            conversion_power_ratio_change
        ),
        "weightChangeRemovedPianoVsOfficialVocalActiveWaveformCosine": (
            removed_cosine
        ),
        "limitations": [
            "No isolated clean piano and vocal ground truth is available; these are cross-stem similarity proxies, not leakage SDR.",
            "Piano and vocals in Imagine are arranged to overlap, so energy-envelope and spectral similarity can remain high without literal waveform leakage.",
            "The removal-direction cosine is more specific: a positive value means the content removed from piano aligns with official vocals, but it still does not prove perceptual improvement.",
            "The vocal-active detector is a fixed ranked-energy proxy from official Torch vocals, not a speech activity model.",
            "The PCM16 parity screen cannot replace the float-tensor Torch/LiteRT export gate.",
        ],
    }


def format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Guitar-ft cross-stem leakage diagnostic: {result['track']}",
        "",
        f"Corpus role: `{result['corpusRole']}`",
        "",
        "## Source identity",
        "",
        f"All three runs use PCM `{result['sharedSource']['selectedPcmSha256']}` with "
        f"`{result['sharedSource']['selectedFrames']}` frames at 44.1 kHz.",
        "",
        "The transient mask is derived once from official Torch drums and reused for all "
        "backends. This prevents backend-specific onset detection from changing the sample "
        "population being compared.",
        "The vocal-active and inactive masks are likewise derived once from official Torch "
        "vocals and reused for every piano comparison.",
        "",
        "## Waveform comparison",
        "",
        "| Difference source | Stem | SNR (dB) | Correlation | Max error (LSB) | PCM16 screen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair_name in ("conversion", "weightChange", "totalObserved"):
        pair = result["waveformPairs"][pair_name]
        for stem in STEMS:
            item = pair["perStem"][stem]
            lines.append(
                f"| {pair_name} | {stem} | {format_number(item['snrDb'])} | "
                f"{format_number(item['correlation'], 7)} | "
                f"{item['maximumAbsoluteErrorLsb']} | "
                f"{'pass' if item['pcm16ParityScreen']['passed'] else 'fail'} |"
            )
    lines.extend(
        [
            "",
            "`conversion` is S25 guitar-ft LiteRT versus guitar-ft Torch. `weightChange` "
            "is guitar-ft Torch versus official Torch. `totalObserved` combines both.",
            "",
            "## Vocal/drum proxies",
            "",
            "| Backend | Waveform correlation | 2-8 kHz envelope correlation | Vocal/drum transient power (dB) | Vocal transient/non-transient (dB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in BACKEND_ORDER:
        item = result["backendLeakageMetrics"][name]
        relationship_value = item["vocalDrumRelationship"]
        transient = item["transientConditioned2To8Khz"]
        lines.append(
            f"| {name} | "
            f"{format_number(relationship_value['normalizedWaveformCorrelation'], 7)} | "
            f"{format_number(relationship_value['log2To8KhzEnergyEnvelopeCorrelation'], 5)} | "
            f"{format_number(transient['vocalToDrumTransientDb'])} | "
            f"{format_number(transient['vocalTransientToNonTransientDb'])} |"
        )
    lines.extend(
        [
            "",
            "Lower vocal/drum transient power is directionally consistent with less drum "
            "energy in vocals, but it is not a quality score because legitimate vocal "
            "content can coincide with drum attacks.",
            "",
            "## Piano/vocal proxies",
            "",
            "| Backend | Fixed-vocal active spectral cosine | Fixed-vocal active waveform cosine | Piano/fixed-vocal active power (dB) | Piano active/inactive power (dB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in BACKEND_ORDER:
        item = result["pianoVocalBackendMetrics"][name]
        fixed = item["pianoVsFixedOfficialVocals"]
        active = item["vocalActivityConditioned300To8000Hz"]
        lines.append(
            f"| {name} | "
            f"{format_number(fixed['vocalActiveSpectralMagnitude']['globalMagnitudeCosineSimilarity'], 6)} | "
            f"{format_number(fixed['vocalActiveWaveform']['waveformCosineSimilarity'], 7)} | "
            f"{format_number(active['pianoToFixedOfficialVocalActiveDb'])} | "
            f"{format_number(active['pianoActiveToInactiveDb'])} |"
        )
    lines.extend(
        [
            "",
            "The fixed-vocal columns compare every piano stem to the same official Torch "
            "vocal reference on the same ranked vocal-active frames. Lower similarity is "
            "directionally consistent with less shared vocal content, but arrangement-level "
            "piano/vocal overlap remains a confounder.",
            "",
            "## Vocal/drum attribution",
            "",
            f"Classification: **{result['attribution']['classification']}**",
            "",
            result["attribution"]["summary"],
            "",
            "Key discriminant: S25 conversion vocal-delta power versus guitar-ft weight-delta "
            f"power on the same transient frames is "
            f"`{format_number(result['attribution']['conversionVsWeightVocalDeltaTransient2To8KhzDb'])} dB`.",
            "",
            "S25 minus guitar-ft Torch vocal/drum transient ratio is "
            f"`{format_number(result['attribution']['s25MinusGuitarFtTorchVocalToDrumTransientDb'])} dB`.",
            "",
            "## Piano/vocal attribution",
            "",
            f"Classification: **{result['pianoVocalAttribution']['classification']}**",
            "",
            result["pianoVocalAttribution"]["summary"],
            "",
            "Listening-observation proxy: "
            f"**{result['pianoVocalAttribution']['listeningObservationConsistency']}**.",
            "",
            "Key discriminant: S25 conversion piano-delta power versus guitar-ft "
            "weight-delta power on the same vocal-active 300-8000 Hz frames is "
            f"`{format_number(result['pianoVocalAttribution']['conversionVsWeightPianoDeltaVocalActive300To8000HzDb'])} dB`.",
            "",
            "Guitar-ft minus official fixed-vocal active spectral cosine is "
            f"`{format_number(result['pianoVocalAttribution']['guitarFtMinusOfficialFixedVocalActiveSpectralCosine'], 6)}`; "
            "the vocal-active waveform cosine between content removed from piano and "
            "official vocals is "
            f"`{format_number(result['pianoVocalAttribution']['weightChangeRemovedPianoVsOfficialVocalActiveWaveformCosine'], 7)}`.",
            "",
            "## Interpretation limits",
            "",
            "### Vocal/drum",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in result["attribution"]["limitations"])
    lines.extend(["", "### Piano/vocal", ""])
    lines.extend(
        f"- {value}" for value in result["pianoVocalAttribution"]["limitations"]
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    inputs = (
        BackendInput("officialTorch", args.official_dir.resolve(), args.official_report.resolve()),
        BackendInput(
            "guitarFtTorch",
            args.guitar_ft_torch_dir.resolve(),
            args.guitar_ft_torch_report.resolve(),
        ),
        BackendInput("s25GuitarFtLiteRt", args.s25_dir.resolve(), args.s25_report.resolve()),
    )
    backends = {value.name: load_backend(value) for value in inputs}
    model_identity_checks = validate_model_identities(args, backends)
    identities = {
        name: backend.report_evidence["source"] for name, backend in backends.items()
    }
    pcm_hashes = {value["selectedPcmSha256"] for value in identities.values()}
    frame_counts = {value["selectedFrames"] for value in identities.values()}
    if len(pcm_hashes) != 1 or len(frame_counts) != 1:
        raise ValueError(f"The three runs do not share one PCM identity: {identities}")
    expected_frames = round(args.expected_duration_seconds * SAMPLE_RATE)
    if args.expected_duration_seconds <= 0.0 or frame_counts != {expected_frames}:
        raise ValueError(
            f"Expected exactly {args.expected_duration_seconds} seconds / "
            f"{expected_frames} frames, found {sorted(frame_counts)}"
        )

    official_flux = spectral_features(
        backends["officialTorch"].stems["drums"], include_flux=True
    )["positiveLogSpectralFlux"]
    mask, detector = transient_mask(official_flux)
    official_vocal_features = spectral_features(
        backends["officialTorch"].stems["vocals"],
        include_flux=False,
        band_low_hz=PIANO_VOCAL_BAND_LOW_HZ,
        band_high_hz=PIANO_VOCAL_BAND_HIGH_HZ,
        include_band_magnitude=True,
    )
    vocal_active_mask, vocal_inactive_mask, vocal_activity_detector = (
        vocal_activity_masks(official_vocal_features["bandPower"])
    )
    backend_metrics: dict[str, Any] = {}
    spectral: dict[str, dict[str, np.ndarray]] = {}
    for name in BACKEND_ORDER:
        backend_metrics[name], spectral[name] = backend_leakage_metrics(
            backends[name], mask
        )
    piano_vocal_backend: dict[str, Any] = {}
    for name in BACKEND_ORDER:
        piano_vocal_backend[name], _ = (
            piano_vocal_backend_metrics(
                backends[name],
                backends["officialTorch"],
                official_vocal_features,
                vocal_active_mask,
                vocal_inactive_mask,
            )
        )

    pairs = parity_pairs(backends)
    deltas: dict[str, Any] = {}
    deltas["weightChange"], weight_delta_band = delta_metrics(
        backends["guitarFtTorch"],
        backends["officialTorch"],
        backends["officialTorch"],
        mask,
        spectral["officialTorch"]["drumsBandPower"],
    )
    deltas["conversion"], conversion_delta_band = delta_metrics(
        backends["s25GuitarFtLiteRt"],
        backends["guitarFtTorch"],
        backends["officialTorch"],
        mask,
        spectral["officialTorch"]["drumsBandPower"],
    )
    deltas["totalObserved"], _ = delta_metrics(
        backends["s25GuitarFtLiteRt"],
        backends["officialTorch"],
        backends["officialTorch"],
        mask,
        spectral["officialTorch"]["drumsBandPower"],
    )
    piano_vocal_deltas: dict[str, Any] = {}
    piano_vocal_deltas["weightChange"], piano_weight_delta_band = (
        piano_vocal_delta_metrics(
            backends["guitarFtTorch"],
            backends["officialTorch"],
            backends["officialTorch"],
            vocal_active_mask,
            vocal_inactive_mask,
            official_vocal_features,
        )
    )
    piano_vocal_deltas["conversion"], piano_conversion_delta_band = (
        piano_vocal_delta_metrics(
            backends["s25GuitarFtLiteRt"],
            backends["guitarFtTorch"],
            backends["officialTorch"],
            vocal_active_mask,
            vocal_inactive_mask,
            official_vocal_features,
        )
    )
    piano_vocal_deltas["totalObserved"], _ = piano_vocal_delta_metrics(
        backends["s25GuitarFtLiteRt"],
        backends["officialTorch"],
        backends["officialTorch"],
        vocal_active_mask,
        vocal_inactive_mask,
        official_vocal_features,
    )
    result = {
        "schemaVersion": 1,
        "status": "complete",
        "scope": "no-ground-truth-vocal-drum-and-piano-vocal-leakage-attribution",
        "track": args.track,
        "corpusRole": args.corpus_role,
        "sharedSource": {
            "selectedPcmSha256": next(iter(pcm_hashes)),
            "selectedFrames": next(iter(frame_counts)),
            "durationSeconds": next(iter(frame_counts)) / SAMPLE_RATE,
        },
        "analysisContract": {
            "sampleRate": SAMPLE_RATE,
            "channelCount": CHANNELS,
            "sampleWidthBytes": SAMPLE_WIDTH,
            "fftSize": FFT_SIZE,
            "hopSize": HOP_SIZE,
            "window": "numpy-hanning",
            "bandHz": [BAND_LOW_HZ, BAND_HIGH_HZ],
            "vocalDrumBandHz": [BAND_LOW_HZ, BAND_HIGH_HZ],
            "pianoVocalBandHz": [
                PIANO_VOCAL_BAND_LOW_HZ,
                PIANO_VOCAL_BAND_HIGH_HZ,
            ],
            "expectedDurationSeconds": args.expected_duration_seconds,
            "modelIdentityChecks": model_identity_checks,
            "transientDetector": detector,
            "vocalActivityDetector": vocal_activity_detector,
            "pcm16ParityScreen": {
                "minimumSnrDb": MINIMUM_SNR_DB,
                "maximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
                "lowSignalReferenceRms": LOW_SIGNAL_REFERENCE_RMS,
                "lowSignalMaximumAbsoluteErrorLsb": LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR_LSB,
                "lowSignalMinimumCorrelation": LOW_SIGNAL_MINIMUM_CORRELATION,
                "warning": "This output-file screen does not replace float-tensor export parity.",
            },
        },
        "backends": {
            name: {
                "stemRoot": str(backend.input.stem_root),
                "report": backend.report_evidence,
                "files": backend.files,
            }
            for name, backend in backends.items()
        },
        "waveformPairs": pairs,
        "backendLeakageMetrics": backend_metrics,
        "deltaDiagnostics": deltas,
        "pianoVocalBackendMetrics": piano_vocal_backend,
        "pianoVocalDeltaDiagnostics": piano_vocal_deltas,
    }
    result["attribution"] = attribution(
        pairs,
        backend_metrics,
        deltas,
        conversion_delta_band,
        weight_delta_band,
        mask,
    )
    result["pianoVocalAttribution"] = piano_vocal_attribution(
        pairs,
        piano_vocal_backend,
        piano_vocal_deltas,
        piano_conversion_delta_band,
        piano_weight_delta_band,
        vocal_active_mask,
    )
    write_atomic(
        args.json.resolve(),
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_atomic(args.markdown.resolve(), render_markdown(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "track": result["track"],
                "classification": result["attribution"]["classification"],
                "conversionPcm16ParityScreenPassed": result["attribution"][
                    "conversionPcm16ParityScreenPassed"
                ],
                "conversionVsWeightTransientBandDeltaDb": result["attribution"][
                    "conversionVsWeightVocalDeltaTransient2To8KhzDb"
                ],
                "pianoVocalClassification": result["pianoVocalAttribution"][
                    "classification"
                ],
                "pianoVocalListeningObservationConsistency": result[
                    "pianoVocalAttribution"
                ]["listeningObservationConsistency"],
                "conversionVsWeightPianoDeltaVocalActiveBandDb": result[
                    "pianoVocalAttribution"
                ]["conversionVsWeightPianoDeltaVocalActive300To8000HzDb"],
                "json": str(args.json.resolve()),
                "markdown": str(args.markdown.resolve()),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
