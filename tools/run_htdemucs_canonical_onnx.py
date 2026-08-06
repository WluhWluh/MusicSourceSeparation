#!/usr/bin/env python3
"""Run the published HTDemucs-6s waveform ONNX graph with the canonical host contract.

The ONNX file is a complete waveform graph.  It is deliberately kept separate
from the project-owned neural-core LiteRT artifact: this runner is a desktop
reference and provenance experiment, not an Android contract adapter.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import wave
from typing import Any, Iterator

import numpy as np


SAMPLE_RATE = 44_100
CHANNELS = 2
BYTES_PER_SAMPLE = 2
BYTES_PER_FRAME = CHANNELS * BYTES_PER_SAMPLE
WINDOW_SAMPLES = 343_980
STRIDE_SAMPLES = 257_985
OVERLAP_SAMPLES = WINDOW_SAMPLES - STRIDE_SAMPLES
STEM_ORDER = ("drums", "bass", "other", "vocals", "guitar", "piano")
STEM_COUNT = len(STEM_ORDER)
EPSILON = np.float32(1e-8)
PCM_SCALE = np.float32(32_768.0)
PCM_OUTPUT_SCALE = np.float32(32_767.0)
IO_CHUNK_FRAMES = 65_536
MODEL_FILE_NAME = "htdemucs_6s_fp16weights.onnx"
MODEL_SHA256 = "7ce55792e2231c93fbf92de95f5fd5b3a5e6c89f7db690dfd693e8f1dce56869"
MODEL_BYTE_SIZE = 136_428_532
FP32_MODEL_SHA256 = "48f8e84945579f8ab340e083339e9221e03785dbe733a52c388200b6d3ca779a"
FP32_MODEL_BYTE_SIZE = 258_159_781
DIAGNOSTIC_MODEL_SHA256 = "c18a8b927ba105da032769697986f57615e06e69f21e99dd89c603cb340db766"
DIAGNOSTIC_MODEL_BYTE_SIZE = 258_180_996
DIAGNOSTIC_VARIANT_ID = "diagnostic-frequency-fp64-transformer-groupnorm-fp64"
SUPPORTED_MODEL_VARIANTS = {
    MODEL_SHA256: {
        "variantId": "stemsplit-waveform-fp16-weight-storage",
        "publishedFileName": "htdemucs_6s_fp16weights.onnx",
        "byteSize": MODEL_BYTE_SIZE,
        "weightStorage": "float16",
        "url": (
            "https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/"
            "49df9b6989cf2150840ea65b0bef77a2e471b678/"
            "htdemucs_6s_fp16weights.onnx?download=true"
        ),
    },
    FP32_MODEL_SHA256: {
        "variantId": "stemsplit-waveform-fp32-weight-storage",
        "publishedFileName": "htdemucs_6s.onnx",
        "byteSize": FP32_MODEL_BYTE_SIZE,
        "weightStorage": "float32",
        "url": (
            "https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/"
            "49df9b6989cf2150840ea65b0bef77a2e471b678/"
            "htdemucs_6s.onnx?download=true"
        ),
    },
}
WAVEFORM_CONTRACT = Path("app/src/main/assets/benchmark-contracts/htdemucs_6s_waveform_7p8s_onnx.json")
CANONICAL_MANIFEST = Path(
    "models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/candidate-manifest.json"
)
FIXTURE_ROOT = Path(
    "models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/fixtures"
)


@dataclass(frozen=True)
class WindowPlan:
    index: int
    track_samples: int
    offset: int
    actual_samples: int
    context_start: int
    context_end: int
    source_start: int
    source_end: int
    pad_left: int
    pad_right: int
    crop_left: int
    crop_right: int

    @property
    def copied_samples(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True)
class Normalization:
    mean: np.float32
    sample_standard_deviation: np.float32
    divisor: np.float32
    selected_pcm_sha256: str


@dataclass(frozen=True)
class Timing:
    wall_ms: float
    cpu_ms: float

    def evidence(self) -> dict[str, float]:
        return {"wallMs": self.wall_ms, "cpuMs": self.cpu_ms}


@dataclass(frozen=True)
class TimedValue:
    value: Any
    timing: Timing


class StemStats:
    def __init__(self) -> None:
        self.count = 0
        self.non_finite = 0
        self.clipped = 0
        self.abs_sum = 0.0
        self.square_sum = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        self.count += int(values.size)
        finite = np.isfinite(values)
        self.non_finite += int(values.size - int(finite.sum()))
        if not finite.all():
            raise ValueError("ONNX output contains NaN or infinity")
        values64 = values.astype(np.float64, copy=False)
        self.clipped += int(np.count_nonzero((values64 < -1.0) | (values64 > 1.0)))
        self.abs_sum += float(np.abs(values64).sum())
        self.square_sum += float(np.square(values64).sum())
        self.minimum = min(self.minimum, float(values64.min(initial=math.inf)))
        self.maximum = max(self.maximum, float(values64.max(initial=-math.inf)))

    def evidence(self) -> dict[str, Any]:
        return {
            "sampleCount": self.count,
            "nonFiniteCount": self.non_finite,
            "clippedSampleCount": self.clipped,
            "absoluteMean": self.abs_sum / self.count if self.count else None,
            "rms": math.sqrt(self.square_sum / self.count) if self.count else None,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
        }


class CanonicalWav:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        with wave.open(str(self.path), "rb") as reader:
            params = reader.getparams()
        if params.nchannels != CHANNELS:
            raise ValueError(f"WAV must be stereo, got {params.nchannels} channels")
        if params.sampwidth != BYTES_PER_SAMPLE:
            raise ValueError(f"WAV must be PCM16, got {params.sampwidth * 8}-bit")
        if params.framerate != SAMPLE_RATE:
            raise ValueError(f"WAV must be {SAMPLE_RATE} Hz, got {params.framerate} Hz")
        if params.comptype != "NONE":
            raise ValueError(f"WAV compression is unsupported: {params.comptype}")
        self.frame_count = int(params.nframes)
        if self.frame_count <= 0:
            raise ValueError("WAV contains no frames")

    @contextmanager
    def reader(self) -> Iterator[wave.Wave_read]:
        with wave.open(str(self.path), "rb") as reader:
            yield reader

    def scan_normalization(self, selected_frames: int) -> Normalization:
        digest = hashlib.sha256()
        reference_sum = 0.0
        remaining = selected_frames
        with self.reader() as reader:
            while remaining:
                count = min(IO_CHUNK_FRAMES, remaining)
                payload = reader.readframes(count)
                expected_bytes = count * BYTES_PER_FRAME
                if len(payload) != expected_bytes:
                    raise ValueError("WAV ended while scanning the selected frames")
                digest.update(payload)
                samples = pcm16_to_float(payload)
                reference = (
                    samples[:, 0].astype(np.float64)
                    + samples[:, 1].astype(np.float64)
                ) * 0.5
                reference_sum += float(reference.sum())
                remaining -= count

        mean_double = reference_sum / selected_frames
        squared_deviation_sum = 0.0
        remaining = selected_frames
        with self.reader() as reader:
            while remaining:
                count = min(IO_CHUNK_FRAMES, remaining)
                payload = reader.readframes(count)
                samples = pcm16_to_float(payload)
                reference = (
                    samples[:, 0].astype(np.float64)
                    + samples[:, 1].astype(np.float64)
                ) * 0.5
                deviation = reference - mean_double
                squared_deviation_sum += float(np.square(deviation).sum())
                remaining -= count

        std = np.float32(math.sqrt(squared_deviation_sum / (selected_frames - 1)))
        mean = np.float32(mean_double)
        divisor = np.float32(std + EPSILON)
        if not np.isfinite(mean) or not np.isfinite(divisor) or divisor <= 0:
            raise ValueError("Normalization parameters are not finite")
        return Normalization(
            mean=mean,
            sample_standard_deviation=std,
            divisor=divisor,
            selected_pcm_sha256=digest.hexdigest(),
        )


def pcm16_to_float(payload: bytes) -> np.ndarray:
    values = np.frombuffer(payload, dtype="<i2")
    if values.size % CHANNELS:
        raise ValueError("PCM payload is not aligned to stereo frames")
    return values.reshape(-1, CHANNELS).astype(np.float32) / PCM_SCALE


def parse_duration_frames(value: str | None, total_frames: int) -> int:
    if value is None:
        if total_frames < 2:
            raise ValueError("Correction-1 normalization requires at least two frames")
        return total_frames
    try:
        seconds = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid duration: {value}") from exc
    if seconds <= 0:
        raise ValueError("Duration must be positive")
    frames_decimal = seconds * SAMPLE_RATE
    if frames_decimal != frames_decimal.to_integral_value():
        raise ValueError("Duration must correspond to an integral 44.1 kHz frame count")
    frames = int(frames_decimal)
    if frames < 2:
        raise ValueError("Correction-1 normalization requires at least two frames")
    if frames > total_frames:
        raise ValueError(f"Requested {frames} frames but input has {total_frames}")
    return frames


def make_window_plans(track_samples: int) -> list[WindowPlan]:
    if track_samples <= 0:
        raise ValueError("Track must contain at least one frame")
    plans: list[WindowPlan] = []
    offset = 0
    while offset < track_samples:
        actual = min(WINDOW_SAMPLES, track_samples - offset)
        delta = WINDOW_SAMPLES - actual
        crop_left = delta // 2
        crop_right = delta - crop_left
        context_start = offset - crop_left
        context_end = context_start + WINDOW_SAMPLES
        source_start = max(0, context_start)
        source_end = min(track_samples, context_end)
        plans.append(
            WindowPlan(
                index=len(plans),
                track_samples=track_samples,
                offset=offset,
                actual_samples=actual,
                context_start=context_start,
                context_end=context_end,
                source_start=source_start,
                source_end=source_end,
                pad_left=source_start - context_start,
                pad_right=context_end - source_end,
                crop_left=crop_left,
                crop_right=crop_right,
            )
        )
        offset += STRIDE_SAMPLES
    return plans


def triangle_weights() -> np.ndarray:
    half = WINDOW_SAMPLES // 2
    values = np.concatenate(
        (
            np.arange(1, half + 1, dtype=np.float32),
            np.arange(WINDOW_SAMPLES - half, 0, -1, dtype=np.float32),
        )
    )
    return values / np.float32(half)


def read_normalized_window(
    reader: wave.Wave_read,
    plan: WindowPlan,
    normalization: Normalization,
) -> np.ndarray:
    output = np.zeros((CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    reader.setpos(plan.source_start)
    payload = reader.readframes(plan.copied_samples)
    if len(payload) != plan.copied_samples * BYTES_PER_FRAME:
        raise ValueError(f"Short read for window {plan.index}")
    samples = pcm16_to_float(payload).T
    normalized = (samples - normalization.mean) / normalization.divisor
    output[:, plan.pad_left : plan.pad_left + plan.copied_samples] = normalized
    return output


def apply_streaming_ola(
    stems: np.ndarray,
    plan: WindowPlan,
    track_samples: int,
    normalization: Normalization,
    weights: np.ndarray,
    carry: np.ndarray,
    carry_weights: np.ndarray,
    carry_length: int,
) -> tuple[np.ndarray, int]:
    if stems.shape != (STEM_COUNT, CHANNELS, WINDOW_SAMPLES):
        raise ValueError(f"Unexpected ONNX output shape: {stems.shape}")
    if plan.offset == 0:
        if carry_length != 0:
            raise ValueError("First OLA window unexpectedly has carry")
    elif not 0 < carry_length <= OVERLAP_SAMPLES:
        raise ValueError(f"Invalid carry length before window {plan.index}: {carry_length}")

    has_next = plan.offset + STRIDE_SAMPLES < track_samples
    finalized = STRIDE_SAMPLES if has_next else plan.actual_samples
    next_carry_length = plan.actual_samples - finalized if has_next else 0
    if not 0 <= next_carry_length <= OVERLAP_SAMPLES:
        raise ValueError("Invalid EOF carry length")

    active = stems[:, :, plan.crop_left : plan.crop_left + plan.actual_samples]
    numerator = active[:, :, :finalized] * weights[:finalized]
    denominator = weights[:finalized].copy()
    if carry_length:
        numerator[:, :, :carry_length] += carry[:, :, :carry_length]
        denominator[:carry_length] += carry_weights[:carry_length]
    if not np.all(denominator > 0):
        raise ValueError("OLA accumulated weight is non-positive")
    normalized = numerator / denominator[None, None, :]
    finalized_values = normalized * normalization.divisor + normalization.mean

    carry.fill(0)
    carry_weights.fill(0)
    if has_next:
        carry[:, :, :next_carry_length] = (
            active[:, :, STRIDE_SAMPLES : STRIDE_SAMPLES + next_carry_length]
            * weights[STRIDE_SAMPLES : STRIDE_SAMPLES + next_carry_length]
        )
        carry_weights[:next_carry_length] = weights[
            STRIDE_SAMPLES : STRIDE_SAMPLES + next_carry_length
        ]
    return np.ascontiguousarray(finalized_values, dtype=np.float32), next_carry_length


def encode_pcm16(values: np.ndarray, stats: StemStats) -> bytes:
    stats.add(values)
    clipped = np.clip(values, -1.0, 1.0)
    # Math.round-compatible for the Android PCM writer (ties are exceptionally rare).
    quantized = np.floor(clipped * PCM_OUTPUT_SCALE + np.float32(0.5))
    quantized = np.clip(quantized, -32768, 32767).astype("<i2")
    interleaved = np.ascontiguousarray(quantized.T)
    return interleaved.tobytes()


def timed(call: Any) -> TimedValue:
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    value = call()
    return TimedValue(
        value=value,
        timing=Timing(
            wall_ms=(time.perf_counter_ns() - wall_start) / 1e6,
            cpu_ms=(time.process_time_ns() - cpu_start) / 1e6,
        ),
    )


def process_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"pid": os.getpid()}
    try:
        import psutil

        process = psutil.Process()
        result["rssBytes"] = process.memory_info().rss
        result["cpuUserSeconds"] = process.cpu_times().user
        result["cpuSystemSeconds"] = process.cpu_times().system
    except ImportError:
        result["rssBytes"] = None
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    error = candidate - reference
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    return {
        "finite": bool(np.isfinite(candidate).all()),
        "snrDb": float(
            20.0
            * np.log10(max(reference_rms, 1e-30) / max(error_rms, 1e-30))
        ),
        "maxAbsoluteError": float(np.max(np.abs(error))),
        "rootMeanSquareError": error_rms,
        "referenceRms": reference_rms,
    }


def validate_fixture(
    session: Any,
    input_name: str,
    output_name: str,
    fixture_root: Path,
    model_variant: dict[str, Any],
) -> dict[str, Any]:
    input_path = fixture_root / "waveform_input.f32le.raw"
    golden_path = fixture_root / "combined_golden.f32le.raw"
    input_values = np.fromfile(input_path, dtype="<f4").reshape(1, CHANNELS, WINDOW_SAMPLES)
    golden = np.fromfile(golden_path, dtype="<f4").reshape(
        1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES
    )
    result = timed(lambda: session.run([output_name], {input_name: input_values})[0])
    candidate = np.asarray(result.value, dtype=np.float32)
    if candidate.shape != golden.shape:
        raise ValueError(f"Fixture output shape mismatch: {candidate.shape}")
    aggregate = metric(golden, candidate)
    per_stem = {
        stem: metric(golden[:, index], candidate[:, index])
        for index, stem in enumerate(STEM_ORDER)
    }
    return {
        "modelVariant": model_variant,
        "inputSha256": sha256_file(input_path),
        "goldenSha256": sha256_file(golden_path),
        "outputShape": list(candidate.shape),
        "timing": result.timing.evidence(),
        "aggregate": aggregate,
        "perStem": per_stem,
        "absoluteErrorGate1e3Passed": aggregate["maxAbsoluteError"] <= 1e-3,
        "uniform80DbGatePassed": aggregate["snrDb"] >= 80.0,
        "interpretation": (
            f"{model_variant['weightStorage'].upper()}-weight-storage waveform ONNX "
            "is behaviorally aligned within the 1e-3 absolute tolerance; the "
            "uniform 80 dB FP32 parity result is reported separately."
        ),
    }


def validate_fixture_subprocess(
    model_path: Path,
    fixture_root: Path,
    threads: int,
    provider: str,
    rewrite_manifest_path: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fixture-only",
        "--model",
        str(model_path),
        "--input",
        str(model_path),
        "--output-dir",
        str(model_path.parent),
        "--fixture-root",
        str(fixture_root),
        "--threads",
        str(threads),
        "--provider",
        provider,
    ]
    if rewrite_manifest_path is not None:
        command.extend(["--rewrite-manifest", str(rewrite_manifest_path)])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Fixture subprocess did not return JSON: {completed.stdout[-1000:]}"
        ) from exc


def inspect_model(model_path: Path) -> dict[str, Any]:
    # protobuf arenas can remain resident after onnx.load(); keep that audit
    # allocation out of the process that measures ORT inference RSS.
    code = r'''
import json, onnx, sys
from collections import Counter
model = onnx.load(sys.argv[1], load_external_data=False)
histogram = Counter(f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node)
print(json.dumps({
    "irVersion": model.ir_version,
    "producerName": model.producer_name,
    "producerVersion": model.producer_version,
    "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
    "nodeCount": len(model.graph.node),
    "initializerCount": len(model.graph.initializer),
    "operatorHistogram": dict(sorted(histogram.items())),
    "externalData": any(item.data_location == onnx.TensorProto.EXTERNAL for item in model.graph.initializer),
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def load_json_identity(path: Path) -> tuple[dict[str, Any], str]:
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def lineage_evidence(
    model_path: Path,
    model_variant: dict[str, Any],
    waveform_contract_path: Path,
    canonical_manifest_path: Path,
) -> dict[str, Any]:
    waveform_contract, contract_sha = load_json_identity(waveform_contract_path)
    canonical_manifest, manifest_sha = load_json_identity(canonical_manifest_path)
    conversion_source = waveform_contract["conversionSource"]
    exporter_input = waveform_contract["conversion"]["exporterInput"]
    canonical_weight = canonical_manifest["provenance"]["canonicalWeight"]
    model_id = canonical_manifest["modelId"]
    expected_input_shape = [1, CHANNELS, WINDOW_SAMPLES]
    expected_output_shape = [1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES]
    contract_input = waveform_contract["tensorContract"]["inputs"][0]
    contract_output = waveform_contract["tensorContract"]["outputs"][0]
    if waveform_contract["contractId"] != "htdemucs_6s_waveform_7p8s_onnx@3":
        raise ValueError("Unexpected waveform ONNX contract ID")
    if conversion_source["sha256"] != MODEL_SHA256:
        raise ValueError("Waveform contract does not identify the pinned export family")
    if exporter_input["sha256"] != "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd":
        raise ValueError("Unexpected ONNX exporter checkpoint identity")
    if contract_input["shape"] != expected_input_shape or contract_output["shape"] != expected_output_shape:
        raise ValueError("Waveform ONNX contract shape mismatch")
    if model_id != "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0":
        raise ValueError("Unexpected canonical neural-core manifest ID")
    if canonical_manifest["artifact"]["sha256"] != (
        "8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7"
    ):
        raise ValueError("Unexpected canonical neural-core artifact identity")
    return {
        "onnxArtifact": {
            "fileName": model_path.name,
            "byteSize": model_path.stat().st_size,
            "sha256": model_variant["sha256"],
            "expectedSha256": model_variant["sha256"],
            "expectedByteSize": model_variant["byteSize"],
            "variantId": model_variant["variantId"],
            "weightStorage": model_variant["weightStorage"],
            "publishedFileName": model_variant["publishedFileName"],
            "hfUrl": model_variant["url"],
            "diagnosticOnly": model_variant.get("diagnosticOnly", False),
            "derivationManifest": model_variant.get("derivationManifest"),
        },
        "waveformContract": {
            "path": str(waveform_contract_path.resolve()),
            "sha256": contract_sha,
            "contractId": waveform_contract["contractId"],
            "conversionSource": conversion_source,
            "selectedArtifactVariant": model_variant,
            "exporterInput": exporter_input,
            "modelRepository": waveform_contract["conversion"]["modelRepository"],
            "modelRevision": waveform_contract["conversion"]["modelRevision"],
            "tensorContract": waveform_contract["tensorContract"],
            "workloadContract": waveform_contract["workloadContract"],
        },
        "canonicalNeuralCore": {
            "path": str(canonical_manifest_path.resolve()),
            "sha256": manifest_sha,
            "modelId": model_id,
            "candidateId": canonical_manifest["candidateId"],
            "artifact": canonical_manifest["artifact"],
            "canonicalWeight": canonical_weight,
            "modelSemantics": canonical_manifest["modelSemantics"],
        },
        "compatibility": {
            "sameModelFamily": True,
            "sameOfficialModelBasename": (
                exporter_input["fileName"].startswith("5c90dfd2")
                and canonical_weight["fileName"].startswith("5c90dfd2")
            ),
            "sameSampleRate": (
                waveform_contract["workloadContract"]["sampleRate"] == SAMPLE_RATE
            ),
            "sameWindowSamples": (
                waveform_contract["workloadContract"]["sampleCount"] == WINDOW_SAMPLES
            ),
            "sameStemOrder": [
                item["semantic"]
                for item in waveform_contract["stemContract"]["bindings"][0]["stems"]
            ]
            == list(STEM_ORDER),
            "sameBoundary": False,
            "sameWeightsByteProven": False,
            "behavioralFixtureCheckRequired": True,
            "contractValidationPassed": True,
            "boundaryExplanation": (
                "ONNX contains waveform STFT/iSTFT and emits six stems; the "
                "canonical TFLite boundary consumes waveform+spectrum and emits "
                "frequency+time branches."
            ),
        },
    }


def open_session(model_path: Path, threads: int, provider: str) -> Any:
    import onnxruntime as ort

    if provider not in ort.get_available_providers():
        raise ValueError(
            f"Provider {provider!r} is unavailable; available={ort.get_available_providers()}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[provider],
    )


def validate_rewrite_manifest(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_sha = load_json_identity(manifest_path)
    actual_size = model_path.stat().st_size
    actual_sha = sha256_file(model_path)
    source = manifest.get("source", {})
    artifact = manifest.get("artifact", {})
    frequency_rewrite = manifest.get("rewrite", {})
    groupnorm_rewrite = manifest.get("transformerGroupNormalizationRewrite", {})

    if manifest.get("schemaVersion") != 1 or manifest.get("status") != "complete":
        raise ValueError("Rewrite manifest is not a complete schemaVersion 1 manifest")
    if source.get("sha256") != FP32_MODEL_SHA256:
        raise ValueError("Rewrite manifest does not identify the pinned FP32 root model")
    if source.get("byteSize") != FP32_MODEL_BYTE_SIZE:
        raise ValueError("Rewrite manifest FP32 root model size mismatch")
    if actual_sha != DIAGNOSTIC_MODEL_SHA256 or actual_size != DIAGNOSTIC_MODEL_BYTE_SIZE:
        raise ValueError(
            "Diagnostic ONNX identity mismatch: "
            f"sha256={actual_sha} byteSize={actual_size}"
        )
    if artifact.get("sha256") != actual_sha or artifact.get("byteSize") != actual_size:
        raise ValueError("Rewrite manifest artifact identity does not match --model")
    if Path(str(artifact.get("path", ""))).name != model_path.name:
        raise ValueError("Rewrite manifest artifact file name does not match --model")
    if frequency_rewrite.get("accumulatorDtype") != "float64":
        raise ValueError("Frequency normalization rewrite is not float64 accumulation")
    if frequency_rewrite.get("weightsChanged") is not False:
        raise ValueError("Frequency normalization rewrite changed model weights")
    if groupnorm_rewrite.get("accumulatorDtype") != "float64":
        raise ValueError("Transformer GroupNorm rewrite is not float64 accumulation")
    if groupnorm_rewrite.get("weightsChanged") is not False:
        raise ValueError("Transformer GroupNorm rewrite changed model weights")

    return {
        "variantId": DIAGNOSTIC_VARIANT_ID,
        "publishedFileName": None,
        "byteSize": actual_size,
        "weightStorage": "float32",
        "url": None,
        "sha256": actual_sha,
        "diagnosticOnly": True,
        "derivationManifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "schemaVersion": manifest["schemaVersion"],
            "status": manifest["status"],
            "source": source,
            "artifact": artifact,
            "rewrite": {
                "scope": frequency_rewrite.get("scope"),
                "accumulatorDtype": frequency_rewrite["accumulatorDtype"],
                "elementwiseDtype": frequency_rewrite.get("elementwiseDtype"),
                "weightsChanged": frequency_rewrite["weightsChanged"],
            },
            "transformerGroupNormalizationRewrite": {
                "scope": groupnorm_rewrite.get("scope"),
                "accumulatorDtype": groupnorm_rewrite["accumulatorDtype"],
                "elementwiseDtype": groupnorm_rewrite.get("elementwiseDtype"),
                "weightsChanged": groupnorm_rewrite["weightsChanged"],
                "learnedAffineChanged": groupnorm_rewrite.get("learnedAffineChanged"),
            },
            "tool": manifest.get("tool"),
        },
    }


def identify_supported_model(
    model_path: Path,
    rewrite_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if rewrite_manifest_path is not None:
        return validate_rewrite_manifest(model_path, rewrite_manifest_path)
    actual_size = model_path.stat().st_size
    actual_sha = sha256_file(model_path)
    expected = SUPPORTED_MODEL_VARIANTS.get(actual_sha)
    if expected is None:
        raise ValueError(f"Unsupported ONNX model SHA-256: {actual_sha}")
    if actual_size != expected["byteSize"]:
        raise ValueError(
            f"ONNX model size mismatch: expected={expected['byteSize']} actual={actual_size}"
        )
    return {**expected, "sha256": actual_sha}


def output_stage_paths(stage: Path) -> dict[str, Path]:
    return {stem: stage / f"{stem}.wav.partial" for stem in STEM_ORDER}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/demucs/onnx") / MODEL_FILE_NAME)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--duration-seconds")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--waveform-contract", type=Path, default=WAVEFORM_CONTRACT)
    parser.add_argument("--canonical-manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument(
        "--rewrite-manifest",
        type=Path,
        help="Authorize a diagnostic derived ONNX artifact through its rewrite manifest",
    )
    parser.add_argument("--skip-fixture-validation", action="store_true")
    parser.add_argument("--fixture-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1 or args.threads > 64:
        raise ValueError("--threads must be in [1,64]")
    model_path = args.model.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    rewrite_manifest_path = (
        args.rewrite_manifest.resolve() if args.rewrite_manifest is not None else None
    )
    if rewrite_manifest_path is not None and not rewrite_manifest_path.is_file():
        raise FileNotFoundError(rewrite_manifest_path)
    model_variant = identify_supported_model(model_path, rewrite_manifest_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory exists; pass --force: {output_dir}")
    stage = output_dir.with_name(output_dir.name + ".partial")
    if stage.exists():
        if not args.force:
            raise FileExistsError(f"Staging directory exists; pass --force: {stage}")

    run_started = time.perf_counter_ns()
    source = CanonicalWav(input_path)
    selected_frames = parse_duration_frames(args.duration_seconds, source.frame_count)
    plans = make_window_plans(selected_frames)
    weights = triangle_weights()
    model_inspection = timed(lambda: inspect_model(model_path))
    source_hash = timed(lambda: sha256_file(input_path))
    normalization_scan = timed(lambda: source.scan_normalization(selected_frames))
    normalization = normalization_scan.value
    lineage = lineage_evidence(
        model_path,
        model_variant,
        args.waveform_contract.resolve(),
        args.canonical_manifest.resolve(),
    )

    fixture_validation = None
    fixture_process_timing = None
    if not args.skip_fixture_validation:
        fixture_process = timed(
            lambda: validate_fixture_subprocess(
                model_path,
                args.fixture_root.resolve(),
                args.threads,
                args.provider,
                rewrite_manifest_path,
            )
        )
        fixture_validation = fixture_process.value
        fixture_process_timing = fixture_process.timing

    session_create = timed(lambda: open_session(model_path, args.threads, args.provider))
    session = session_create.value
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected one ONNX input/output, got {len(inputs)}/{len(outputs)}")
    input_name = inputs[0].name
    output_name = outputs[0].name
    if input_name != "mix" or output_name != "stems":
        raise ValueError(f"Unexpected ONNX bindings: {input_name}/{output_name}")
    if inputs[0].shape != [1, CHANNELS, WINDOW_SAMPLES]:
        raise ValueError(f"Unexpected ONNX input shape: {inputs[0].shape}")

    runtime_session_create = session_create

    # Destructive replacement is delayed until model, contracts, and fixtures pass.
    if stage.exists():
        shutil.rmtree(stage)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    stage.mkdir(parents=True)

    stage_paths = output_stage_paths(stage)
    writers: dict[str, wave.Wave_write] = {}
    for stem, path in stage_paths.items():
        writer = wave.open(str(path), "wb")
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(BYTES_PER_SAMPLE)
        writer.setframerate(SAMPLE_RATE)
        writers[stem] = writer

    carry = np.zeros((STEM_COUNT, CHANNELS, OVERLAP_SAMPLES), dtype=np.float32)
    carry_weights = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)
    carry_length = 0
    stem_stats = {stem: StemStats() for stem in STEM_ORDER}
    window_reports: list[dict[str, Any]] = []
    emitted_frames = 0
    separation_started = time.perf_counter_ns()
    failure: BaseException | None = None
    try:
        with source.reader() as reader:
            for plan in plans:
                window_started = time.perf_counter_ns()
                window_cpu_started = time.process_time_ns()
                prepare = timed(lambda: read_normalized_window(reader, plan, normalization))
                inference = timed(
                    lambda: session.run(
                        [output_name],
                        {input_name: prepare.value[np.newaxis, ...]},
                    )[0]
                )
                model_output = np.asarray(inference.value, dtype=np.float32)
                if model_output.shape != (1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES):
                    raise ValueError(f"Unexpected runtime output shape: {model_output.shape}")
                ola = timed(
                    lambda: apply_streaming_ola(
                        model_output[0],
                        plan,
                        selected_frames,
                        normalization,
                        weights,
                        carry,
                        carry_weights,
                        carry_length,
                    )
                )
                finalized, carry_length = ola.value
                pcm_payloads: dict[str, bytes] = {}
                pcm_start = time.perf_counter_ns()
                pcm_cpu_start = time.process_time_ns()
                for index, stem in enumerate(STEM_ORDER):
                    pcm_payloads[stem] = encode_pcm16(finalized[index], stem_stats[stem])
                pcm_timing = Timing(
                    wall_ms=(time.perf_counter_ns() - pcm_start) / 1e6,
                    cpu_ms=(time.process_time_ns() - pcm_cpu_start) / 1e6,
                )
                write_start = time.perf_counter_ns()
                write_cpu_start = time.process_time_ns()
                for stem in STEM_ORDER:
                    writers[stem].writeframesraw(pcm_payloads[stem])
                write_timing = Timing(
                    wall_ms=(time.perf_counter_ns() - write_start) / 1e6,
                    cpu_ms=(time.process_time_ns() - write_cpu_start) / 1e6,
                )
                emitted_frames += int(finalized.shape[-1])
                window_reports.append(
                    {
                        "index": plan.index,
                        "plan": asdict(plan),
                        "finalizedFrames": int(finalized.shape[-1]),
                        "outputFramesAfterWindow": emitted_frames,
                        "stages": {
                            "prepare": prepare.timing.evidence(),
                            "inference": inference.timing.evidence(),
                            "ola": ola.timing.evidence(),
                            "pcm": pcm_timing.evidence(),
                            "write": write_timing.evidence(),
                        },
                        "total": {
                            "wallMs": (time.perf_counter_ns() - window_started) / 1e6,
                            "cpuMs": (time.process_time_ns() - window_cpu_started) / 1e6,
                        },
                        "process": process_snapshot(),
                    }
                )
    except BaseException as exc:
        failure = exc
    finally:
        close_error: BaseException | None = None
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as exc:  # pragma: no cover - cleanup path
                close_error = close_error or exc
        if close_error is not None and failure is None:
            failure = close_error

    separation_wall_ms = (time.perf_counter_ns() - separation_started) / 1e6
    if failure is not None:
        shutil.rmtree(stage, ignore_errors=True)
        raise failure

    if emitted_frames != selected_frames or carry_length != 0:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError(f"OLA emitted {emitted_frames}/{selected_frames} frames, carry={carry_length}")

    for stem in STEM_ORDER:
        partial = stage / f"{stem}.wav.partial"
        partial.replace(stage / f"{stem}.wav")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(output_dir)

    output_evidence: dict[str, Any] = {}
    for stem in STEM_ORDER:
        path = output_dir / f"{stem}.wav"
        output_evidence[stem] = {
            **stem_stats[stem].evidence(),
            "path": str(path),
            "byteSize": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    total_wall_ms = (time.perf_counter_ns() - run_started) / 1e6
    audio_duration_seconds = selected_frames / SAMPLE_RATE
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "complete",
        "runner": "run_htdemucs_canonical_onnx.py",
        "scope": "desktop-onnx-reference-not-an-android-result",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "providerRequested": args.provider,
            "providersAvailable": list(session.get_providers()),
            "threads": args.threads,
        },
        "provenance": lineage,
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "path": str(input_path),
            "byteSize": input_path.stat().st_size,
            "sha256": source_hash.value,
            "sampleRate": SAMPLE_RATE,
            "channelCount": CHANNELS,
            "sourceFrames": source.frame_count,
            "selectedFrames": selected_frames,
            "durationSeconds": selected_frames / SAMPLE_RATE,
            "selectedPcmSha256": normalization.selected_pcm_sha256,
            "globalMean": float(normalization.mean),
            "sampleStandardDeviation": float(normalization.sample_standard_deviation),
            "normalizationDivisor": float(normalization.divisor),
        },
        "contract": {
            "sampleRate": SAMPLE_RATE,
            "windowSamples": WINDOW_SAMPLES,
            "strideSamples": STRIDE_SAMPLES,
            "overlapSamples": OVERLAP_SAMPLES,
            "overlap": 0.25,
            "transitionPower": 1.0,
            "windowCount": len(plans),
            "tailPadding": "official-demucs-tensor-chunk-centered-zero-pad",
            "tailCrop": "center-crop-to-actual-samples",
            "tailWeightRule": "triangle-prefix",
            "globalNormalization": "stereo-reference-mean-unbiased-std-correction-1",
            "stemOrder": list(STEM_ORDER),
        },
        "modelInspection": model_inspection.value,
        "modelInspectionTiming": model_inspection.timing.evidence(),
        "sourceHashTiming": source_hash.timing.evidence(),
        "normalizationScanTiming": normalization_scan.timing.evidence(),
        "sessionCreateTiming": session_create.timing.evidence(),
        "runtimeSessionCreateTiming": runtime_session_create.timing.evidence(),
        "fixtureProcessTiming": fixture_process_timing.evidence()
        if fixture_process_timing is not None
        else None,
        "bindings": {
            "input": {"name": input_name, "shape": [1, CHANNELS, WINDOW_SAMPLES], "dtype": "float32"},
            "output": {"name": output_name, "shape": [1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES], "dtype": "float32"},
        },
        "fixtureValidation": fixture_validation,
        "windows": window_reports,
        "outputs": output_evidence,
        "processFinal": process_snapshot(),
        "separationWallMs": separation_wall_ms,
        "separationRealtimeFactor": separation_wall_ms / 1000.0 / audio_duration_seconds,
        "totalWallMs": total_wall_ms,
        "totalRealtimeFactor": total_wall_ms / 1000.0 / audio_duration_seconds,
    }
    report_path = output_dir / "report.json"
    write_json_atomic(report_path, report)
    if args.report is not None:
        write_json_atomic(args.report.resolve(), report)
    return report


def main() -> int:
    args = parse_args()
    if args.fixture_only:
        try:
            rewrite_manifest_path = (
                args.rewrite_manifest.resolve()
                if args.rewrite_manifest is not None
                else None
            )
            model_variant = identify_supported_model(
                args.model.resolve(), rewrite_manifest_path
            )
            session = open_session(args.model.resolve(), args.threads, args.provider)
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Fixture session must have one input and one output")
            if inputs[0].name != "mix" or outputs[0].name != "stems":
                raise ValueError("Fixture session bindings do not match mix/stems")
            result = validate_fixture(
                session,
                inputs[0].name,
                outputs[0].name,
                args.fixture_root.resolve(),
                model_variant,
            )
        except Exception as exc:
            print(f"fixture error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    try:
        report = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report["status"],
        "outputDir": str(args.output_dir.resolve()),
        "report": str((args.output_dir / "report.json").resolve()),
        "windowCount": report["contract"]["windowCount"],
        "separationWallMs": report["separationWallMs"],
        "separationRealtimeFactor": report["separationRealtimeFactor"],
        "totalWallMs": report["totalWallMs"],
        "totalRealtimeFactor": report["totalRealtimeFactor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
