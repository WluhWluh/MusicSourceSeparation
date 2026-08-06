#!/usr/bin/env python3
"""Run the pinned HTDemucs four-stem Batch 4A host quality experiment.

The five primary renderings share one 44.1 kHz stereo input, one selected
prefix normalization, and the repository's 7.8 second / 25% overlap OLA.
Official fine-tuned specialists are evaluated separately and only their
documented target stem is retained. The third-party Psytrance neural core is
research-only and has no Torch oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gc
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
from typing import Any, Callable, Iterable
import wave

import numpy as np

from run_htdemucs_canonical_onnx import (
    CanonicalWav,
    StemStats,
    encode_pcm16,
    read_normalized_window,
    sha256_file,
    triangle_weights,
)


SAMPLE_RATE = 44_100
CHANNELS = 2
STEMS = ("drums", "bass", "other", "vocals")
WINDOW_SAMPLES = 343_980
CANONICAL_STRIDE = 257_985
PSY_NATIVE_STRIDE = 171_990
DEFAULT_FRAMES = 1_323_000
DEMUCS_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
PARAMETER_COUNT = 41_984_456
WEIGHT_BYTES = 84_025_440
HF_BASE_REVISION = "bf35a81b663819a8255c8fefee17f9d812b786b5"
HF_FT_REVISION = "478be8a68f85418addd6f7baefd4be76522a4034"
HF_PSY_REVISION = "e725e7eb9204188de4731658e9923dcf049273c4"
HF_CONTROL_REVISION = "12cc9a49c3b5f3badc1b0821ccc26f1a32c1779a"
PSY_ONNX_SHA256 = "d7cfcfaf41dc611dd14d35a206fe14a667bdaa6a1910f4c09a8d8ed43606cc78"
PSY_MANIFEST_SHA256 = "de1a86eb56462976968e56d5bd15ef242bd2dcd9627a5355ecceb0c68e2385ce"
CONTROL_ONNX_SHA256 = "21cdabc8246f5052397647399e48292dc7394475e92335c65c19eb7e90bde6e0"

TRACKS = {
    "athletics-ii": "Athletics - II",
    "john-lennon-imagine": "John Lennon - Imagine",
    "josiah-james-chasing-the-wind": "Josiah James - Chasing The Wind",
    "kygo-ed-sheeran-i-see-fire-kygo-remix": "Kygo & Ed Sheeran - I See Fire (Kygo Remix)",
}

PRIMARY_VARIANTS = (
    "official-base",
    "official-ft-bag",
    "base-plus-vocals",
    "base-plus-drums",
    "psytrance-onnx",
)
SENSITIVITY_VARIANT = "psytrance-onnx-native50-sensitivity"


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
class TorchArtifact:
    model_id: str
    signature: str
    target_stem: str | None
    weight: Path
    weight_sha256: str
    metadata: Path
    metadata_sha256: str
    hf_repository: str
    hf_revision: str


@dataclass(frozen=True)
class TrackContract:
    slug: str
    display_name: str
    path: Path
    source_frames: int
    selected_frames: int
    file_sha256: str
    selected_pcm_sha256: str
    mean: float
    divisor: float
    sample_standard_deviation: float
    upstream_report: Path
    upstream_report_sha256: str
    source_mp3_sha256: str | None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value, dtype=np.float32), allow_pickle=False)
    partial.replace(path)


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def process_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"pid": os.getpid()}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile"}:
                result[f"{key}KiB"] = int(value.strip().split()[0])
    return result


def make_plans(track_samples: int, stride: int) -> list[WindowPlan]:
    if track_samples <= 0 or not 0 < stride <= WINDOW_SAMPLES:
        raise ValueError("Invalid track length or stride")
    plans: list[WindowPlan] = []
    for offset in range(0, track_samples, stride):
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
    return plans


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "fileName": path.name,
        "byteSize": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expected_artifacts(repo: Path) -> tuple[TorchArtifact, ...]:
    base = repo / "models/demucs/official-hf/htdemucs"
    ft = repo / "models/demucs/candidates/htdemucs-ft" / HF_FT_REVISION
    return (
        TorchArtifact(
            "base-955717e8", "955717e8", None,
            base / "955717e8.safetensors",
            "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
            base / "955717e8.json",
            "12540373de858920b60002ebfbe17738860dbf2e89df625dc3b7875af2d28491",
            "adefossez/HTDemucs", HF_BASE_REVISION,
        ),
        TorchArtifact(
            "ft-drums-f7e0c4bc", "f7e0c4bc", "drums",
            ft / "f7e0c4bc.safetensors",
            "2c85ab3c62dd6edd8e0b965e38b16fd1cdde357cc25de6b6bc9ce7c83f60925f",
            ft / "f7e0c4bc.json",
            "29fc60975f3389567f8f2510403c8bfb6fbcfde2a68903a332d82005ee416e02",
            "adefossez/HTDemucs-ft", HF_FT_REVISION,
        ),
        TorchArtifact(
            "ft-bass-d12395a8", "d12395a8", "bass",
            ft / "d12395a8.safetensors",
            "5b01a97567ae9a3178a6236fb520251045c03eb8834bc8c24a4eec11d6c8fb56",
            ft / "d12395a8.json",
            "e20a555ff3f94368b651de8ec7fc9b76297324f94654ce012d8ef403a5a6e0d7",
            "adefossez/HTDemucs-ft", HF_FT_REVISION,
        ),
        TorchArtifact(
            "ft-other-92cfc3b6", "92cfc3b6", "other",
            ft / "92cfc3b6.safetensors",
            "a241863551f30d01c42bd7b97da40839922ead3acb0f1fcab25682f55b4eeb59",
            ft / "92cfc3b6.json",
            "9c4090bdd905340ea80553535118ef61e55b826515c30630ecf118727552d48b",
            "adefossez/HTDemucs-ft", HF_FT_REVISION,
        ),
        TorchArtifact(
            "ft-vocals-04573f0d", "04573f0d", "vocals",
            ft / "04573f0d.safetensors",
            "68854b0d7c2b3274723b5761f6fd9f5aec5f1bcd3f0de7c1669546fdb7871b7c",
            ft / "04573f0d.json",
            "ca2f2646edd82a74c470c398689f44c9e25e33fa8dd0b86daeca3359ecd7a3af",
            "adefossez/HTDemucs-ft", HF_FT_REVISION,
        ),
    )


def verify_torch_artifact(artifact: TorchArtifact) -> dict[str, Any]:
    if artifact.weight.stat().st_size != WEIGHT_BYTES:
        raise ValueError(f"Unexpected weight size: {artifact.weight}")
    if sha256_file(artifact.weight) != artifact.weight_sha256:
        raise ValueError(f"Weight SHA mismatch: {artifact.weight}")
    if sha256_file(artifact.metadata) != artifact.metadata_sha256:
        raise ValueError(f"Metadata SHA mismatch: {artifact.metadata}")
    external = json.loads(artifact.metadata.read_text(encoding="utf-8"))
    from safetensors import safe_open

    with safe_open(str(artifact.weight), framework="pt") as handle:
        embedded = handle.metadata() or {}
        tensor_count = len(handle.keys())
    for key in ("klass", "args", "kwargs"):
        if external.get(key) != embedded.get(key):
            raise ValueError(f"Embedded metadata mismatch for {artifact.signature}: {key}")
    return {
        "modelId": artifact.model_id,
        "signature": artifact.signature,
        "documentedTargetStem": artifact.target_stem,
        "weight": identity(artifact.weight),
        "metadata": identity(artifact.metadata),
        "embeddedMetadataMatchesSidecar": True,
        "tensorCount": tensor_count,
        "hfRepository": artifact.hf_repository,
        "hfRevision": artifact.hf_revision,
    }


def load_torch_model(
    torch: Any,
    artifact: TorchArtifact,
    verified: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from demucs.hf import load_safetensors_model
    from export_htdemucs_litert_candidate import install_deterministic_pos_embedding

    verified = verified or verify_torch_artifact(artifact)
    started = time.perf_counter_ns()
    model = load_safetensors_model(artifact.weight).cpu().float().eval()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    if tuple(model.sources) != STEMS:
        raise ValueError(f"Unexpected stem order for {artifact.signature}: {model.sources}")
    if model.samplerate != SAMPLE_RATE or model.segment != Fraction(39, 5):
        raise ValueError(f"Unexpected workload for {artifact.signature}")
    if not model.use_train_segment:
        raise ValueError(f"Fixed training segment disabled for {artifact.signature}")
    parameter_count = sum(value.numel() for value in model.state_dict().values())
    if parameter_count != PARAMETER_COUNT:
        raise ValueError(f"Unexpected parameter count for {artifact.signature}")
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError(f"Non-finite parameter in {artifact.signature}")
    install_deterministic_pos_embedding(model)
    return model, {
        **verified,
        "loadWallMs": elapsed_ms,
        "loadedParameterDtype": "float32",
        "parameterCount": parameter_count,
        "finiteParameterGatePassed": True,
        "contract": {
            "sampleRate": SAMPLE_RATE,
            "channels": CHANNELS,
            "windowSamples": WINDOW_SAMPLES,
            "stemOrder": list(STEMS),
        },
    }


def load_track_contract(source_root: Path, slug: str, selected_frames: int) -> TrackContract:
    track_root = source_root / "tracks" / slug
    path = track_root / "s25-cpu" / "canonical-input-44100-stereo-pcm16.wav"
    upstream_report = track_root / "s25-cpu" / "report.json"
    source = CanonicalWav(path)
    if source.frame_count < selected_frames:
        raise ValueError(f"{slug} is shorter than the requested prefix")
    normalization = source.scan_normalization(selected_frames)
    source_mp3_sha = None
    if upstream_report.is_file():
        upstream = json.loads(upstream_report.read_text(encoding="utf-8"))
        source_mp3_sha = upstream.get("source", {}).get("fileSha256")
    return TrackContract(
        slug=slug,
        display_name=TRACKS[slug],
        path=path.resolve(),
        source_frames=source.frame_count,
        selected_frames=selected_frames,
        file_sha256=sha256_file(path),
        selected_pcm_sha256=normalization.selected_pcm_sha256,
        mean=float(normalization.mean),
        divisor=float(normalization.divisor),
        sample_standard_deviation=float(normalization.sample_standard_deviation),
        upstream_report=upstream_report.resolve(),
        upstream_report_sha256=sha256_file(upstream_report),
        source_mp3_sha256=source_mp3_sha,
    )


def track_evidence(track: TrackContract) -> dict[str, Any]:
    return {
        "slug": track.slug,
        "displayName": track.display_name,
        "canonicalWav": {
            "path": str(track.path),
            "fileSha256": track.file_sha256,
            "sourceFrames": track.source_frames,
            "sampleRate": SAMPLE_RATE,
            "channelCount": CHANNELS,
            "bitsPerSample": 16,
        },
        "selection": {
            "startFrame": 0,
            "selectedFrames": track.selected_frames,
            "durationSeconds": track.selected_frames / SAMPLE_RATE,
            "selectedPcmSha256": track.selected_pcm_sha256,
        },
        "normalization": {
            "scope": "selected 30-second prefix, shared by all variants",
            "mean": track.mean,
            "sampleStandardDeviation": track.sample_standard_deviation,
            "divisor": track.divisor,
            "epsilon": 1e-8,
        },
        "upstreamDeviceDecodeReport": {
            "path": str(track.upstream_report),
            "sha256": track.upstream_report_sha256,
            "sourceMp3Sha256": track.source_mp3_sha256,
        },
    }


def raw_paths(output_root: Path, model_id: str, slug: str) -> tuple[Path, Path]:
    root = output_root / "raw" / model_id
    return root / f"{slug}.npy", root / f"{slug}.json"


def valid_raw(
    output_root: Path,
    model_id: str,
    track: TrackContract,
    expected_shape: tuple[int, ...],
    tool_sha: str,
    expected_artifact_sha: str | None = None,
    expected_metadata_sha: str | None = None,
) -> bool:
    array_path, report_path = raw_paths(output_root, model_id, track.slug)
    if not array_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        value = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    model = report.get("model", {})
    reported_artifact_sha = (
        model.get("weight", {}).get("sha256")
        or model.get("onnx", {}).get("sha256")
    )
    reported_metadata_sha = model.get("metadata", {}).get("sha256")
    return bool(
        report.get("status") == "complete"
        and report.get("tool", {}).get("sha256") == tool_sha
        and report.get("source", {}).get("selectedPcmSha256") == track.selected_pcm_sha256
        and report.get("output", {}).get("sha256") == sha256_file(array_path)
        and (expected_artifact_sha is None or reported_artifact_sha == expected_artifact_sha)
        and (expected_metadata_sha is None or reported_metadata_sha == expected_metadata_sha)
        and value.shape == expected_shape
        and value.dtype == np.float32
    )


def accumulate_track(
    track: TrackContract,
    stride: int,
    output_count: int,
    infer: Callable[[np.ndarray, WindowPlan], tuple[np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    from run_htdemucs_canonical_onnx import Normalization

    plans = make_plans(track.selected_frames, stride)
    weights = triangle_weights()
    normalization = Normalization(
        mean=np.float32(track.mean),
        sample_standard_deviation=np.float32(track.sample_standard_deviation),
        divisor=np.float32(track.divisor),
        selected_pcm_sha256=track.selected_pcm_sha256,
    )
    numerator = np.zeros(
        (output_count, CHANNELS, track.selected_frames), dtype=np.float32
    )
    denominator = np.zeros(track.selected_frames, dtype=np.float32)
    window_reports: list[dict[str, Any]] = []
    with CanonicalWav(track.path).reader() as reader:
        for plan in plans:
            prepare_started = time.perf_counter_ns()
            window = read_normalized_window(reader, plan, normalization)
            prepare_ms = (time.perf_counter_ns() - prepare_started) / 1e6
            output, detail = infer(window, plan)
            expected = (output_count, CHANNELS, WINDOW_SAMPLES)
            if output.shape != expected:
                raise ValueError(f"Unexpected inference shape: {output.shape} != {expected}")
            finite = np.isfinite(output)
            if not bool(finite.all()):
                raise ValueError(f"Non-finite inference output at {track.slug}/{plan.index}")
            ola_started = time.perf_counter_ns()
            active = output[..., plan.crop_left : plan.crop_left + plan.actual_samples]
            end = plan.offset + plan.actual_samples
            numerator[..., plan.offset:end] += active * weights[: plan.actual_samples]
            denominator[plan.offset:end] += weights[: plan.actual_samples]
            ola_ms = (time.perf_counter_ns() - ola_started) / 1e6
            window_reports.append(
                {
                    "plan": asdict(plan),
                    "prepareWallMs": prepare_ms,
                    "olaAccumulateWallMs": ola_ms,
                    "inference": detail,
                    "allOutputFinite": True,
                    "process": process_snapshot(),
                }
            )
    if not bool((denominator > 0).all()):
        raise ValueError("OLA denominator contains non-positive values")
    finalize_started = time.perf_counter_ns()
    value = numerator / denominator[None, None, :]
    value = value * np.float32(track.divisor) + np.float32(track.mean)
    value = np.ascontiguousarray(value, dtype=np.float32)
    finalize_ms = (time.perf_counter_ns() - finalize_started) / 1e6
    return value, {
        "windowCount": len(plans),
        "windowSamples": WINDOW_SAMPLES,
        "strideSamples": stride,
        "overlapSamples": WINDOW_SAMPLES - stride,
        "overlap": (WINDOW_SAMPLES - stride) / WINDOW_SAMPLES,
        "transitionPower": 1.0,
        "tailPadding": "official-demucs-tensor-chunk-centered-zero-pad",
        "tailCrop": "center-crop-to-actual-samples",
        "tailWeightRule": "triangle-prefix",
        "olaFinalizeWallMs": finalize_ms,
        "windows": window_reports,
    }


def run_torch_artifact(
    torch: Any,
    artifact: TorchArtifact,
    tracks: list[TrackContract],
    output_root: Path,
    tool: dict[str, Any],
) -> dict[str, Any]:
    output_count = 4 if artifact.target_stem is None else 1
    expected = (output_count, CHANNELS, tracks[0].selected_frames)
    verified = verify_torch_artifact(artifact)
    all_raw_valid = all(
        valid_raw(
            output_root,
            artifact.model_id,
            track,
            expected,
            tool["sha256"],
            artifact.weight_sha256,
            artifact.metadata_sha256,
        )
        for track in tracks
    )
    model_report_path = output_root / "raw" / artifact.model_id / "model-report.json"
    if all_raw_valid and model_report_path.is_file():
        existing = json.loads(model_report_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("tool", {}).get("sha256") == tool["sha256"]
            and existing.get("model", {}).get("weight", {}).get("sha256")
            == artifact.weight_sha256
        ):
            return existing

    model, model_report = load_torch_model(torch, artifact, verified)
    track_reports: list[dict[str, Any]] = []
    target_index = STEMS.index(artifact.target_stem) if artifact.target_stem else None
    for track in tracks:
        expected_shape = (output_count, CHANNELS, track.selected_frames)
        if valid_raw(
            output_root,
            artifact.model_id,
            track,
            expected_shape,
            tool["sha256"],
            artifact.weight_sha256,
            artifact.metadata_sha256,
        ):
            report = json.loads(raw_paths(output_root, artifact.model_id, track.slug)[1].read_text(encoding="utf-8"))
            track_reports.append(report)
            continue

        def infer(window: np.ndarray, plan: WindowPlan) -> tuple[np.ndarray, dict[str, Any]]:
            started = time.perf_counter_ns()
            with torch.inference_mode():
                tensor = torch.from_numpy(window[np.newaxis, ...])
                full = model(tensor)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            full_np = full.detach().cpu().numpy().astype(np.float32, copy=False)
            if full_np.shape != (1, 4, CHANNELS, WINDOW_SAMPLES):
                raise ValueError(f"Unexpected Torch output shape: {full_np.shape}")
            selected = full_np[0] if target_index is None else full_np[0, target_index : target_index + 1]
            return np.ascontiguousarray(selected), {
                "backend": "Torch CPU FP32",
                "wallMs": elapsed_ms,
                "retainedStem": artifact.target_stem or "all",
                "discardedNonTargetStemCount": 0 if target_index is None else 3,
            }

        started = time.perf_counter_ns()
        value, contract = accumulate_track(
            track, CANONICAL_STRIDE, output_count, infer
        )
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        array_path, report_path = raw_paths(output_root, artifact.model_id, track.slug)
        save_npy(array_path, value)
        report = {
            "schemaVersion": 1,
            "status": "complete",
            "scope": "Batch 4A raw Torch inference output before PCM quantization",
            "model": model_report,
            "source": track_evidence(track)["selection"] | {
                "canonicalWavFileSha256": track.file_sha256,
            },
            "contract": contract,
            "totalTrackWallMs": wall_ms,
            "realtimeFactor": wall_ms / 1000.0 / (track.selected_frames / SAMPLE_RATE),
            "output": {
                **identity(array_path),
                "shape": list(value.shape),
                "dtype": "float32",
                "allFinite": bool(np.isfinite(value).all()),
            },
            "tool": tool,
        }
        write_json(report_path, report)
        track_reports.append(report)
        print(json.dumps({
            "stage": artifact.model_id,
            "track": track.slug,
            "wallMs": wall_ms,
            "realtimeFactor": report["realtimeFactor"],
        }), flush=True)
    del model
    gc.collect()
    result = {
        "schemaVersion": 1,
        "status": "complete",
        "model": model_report,
        "tracks": [
            {
                "slug": track.slug,
                "rawReport": str(raw_paths(output_root, artifact.model_id, track.slug)[1]),
                "rawReportSha256": sha256_file(raw_paths(output_root, artifact.model_id, track.slug)[1]),
            }
            for track in tracks
        ],
        "tool": tool,
    }
    write_json(model_report_path, result)
    return result


def open_ort_session(ort: Any, model_path: Path, threads: int) -> tuple[Any, dict[str, Any]]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    inputs = [
        {"name": value.name, "shape": value.shape, "type": value.type}
        for value in session.get_inputs()
    ]
    outputs = [
        {"name": value.name, "shape": value.shape, "type": value.type}
        for value in session.get_outputs()
    ]
    abi_valid = bool(
        [item["name"] for item in inputs] == ["input", "x"]
        and [item["name"] for item in outputs] == ["output", "add_67"]
        and all(item["type"] == "tensor(float)" for item in inputs + outputs)
        and inputs[0]["shape"][:2] == [1, 2]
        and inputs[0]["shape"][2] in {"T", WINDOW_SAMPLES}
        and inputs[1]["shape"] == [1, 4, 2048, 336]
        and outputs[0]["shape"] == [1, 4, 4, 2048, 336]
        and len(outputs[1]["shape"]) == 4
        and outputs[1]["shape"][-1] in {"T", WINDOW_SAMPLES}
    )
    if not abi_valid:
        raise ValueError(f"Unexpected Psytrance ONNX ABI: {inputs} / {outputs}")
    return session, {
        "sessionLoadWallMs": elapsed_ms,
        "providers": session.get_providers(),
        "inputs": inputs,
        "outputs": outputs,
    }


def validate_kani_conversion_control(
    torch: Any,
    ort: Any,
    artifacts: tuple[TorchArtifact, ...],
    tracks: list[TrackContract],
    output_root: Path,
    repo: Path,
    threads: int,
    tool: dict[str, Any],
) -> dict[str, Any]:
    from run_htdemucs_canonical_onnx import Normalization
    from export_htdemucs_litert_candidate import (
        as_numpy,
        reconstruct_branches,
        spec_to_channels,
    )

    control_path = (
        repo
        / "models/demucs/conversion/htdemucs-ft-ort"
        / HF_CONTROL_REVISION
        / "htdemucs_ft.onnx"
    )
    if control_path.stat().st_size != 174_266_467:
        raise ValueError("Unexpected Kani conversion-control byte size")
    if sha256_file(control_path) != CONTROL_ONNX_SHA256:
        raise ValueError("Kani conversion-control SHA mismatch")
    track = next(
        (item for item in tracks if item.slug == "josiah-james-chasing-the-wind"),
        tracks[0],
    )
    plan = make_plans(track.selected_frames, CANONICAL_STRIDE)[0]
    normalization = Normalization(
        mean=np.float32(track.mean),
        sample_standard_deviation=np.float32(track.sample_standard_deviation),
        divisor=np.float32(track.divisor),
        selected_pcm_sha256=track.selected_pcm_sha256,
    )
    with CanonicalWav(track.path).reader() as reader:
        window = read_normalized_window(reader, plan, normalization)
    waveform = np.ascontiguousarray(window[np.newaxis, ...], dtype=np.float32)

    drums_artifact = next(item for item in artifacts if item.target_stem == "drums")
    drums_model, drums_model_report = load_torch_model(torch, drums_artifact)
    torch_started = time.perf_counter_ns()
    with torch.inference_mode():
        torch_drums = as_numpy(drums_model(torch.from_numpy(waveform)))[0, 0]
    torch_ms = (time.perf_counter_ns() - torch_started) / 1e6
    del drums_model
    gc.collect()

    dsp_model, dsp_model_report = load_torch_model(torch, artifacts[0])
    session, session_report = open_ort_session(ort, control_path, threads)
    stft_started = time.perf_counter_ns()
    with torch.inference_mode():
        spectrum = as_numpy(
            spec_to_channels(torch, dsp_model._spec(torch.from_numpy(waveform)))
        )
    stft_ms = (time.perf_counter_ns() - stft_started) / 1e6
    ort_started = time.perf_counter_ns()
    frequency, time_branch = session.run(
        ["output", "add_67"], {"input": waveform, "x": spectrum}
    )
    ort_ms = (time.perf_counter_ns() - ort_started) / 1e6
    reconstruct_started = time.perf_counter_ns()
    with torch.inference_mode():
        _, control_full = reconstruct_branches(
            torch,
            dsp_model,
            torch.from_numpy(np.ascontiguousarray(frequency, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(time_branch, dtype=np.float32)),
            WINDOW_SAMPLES,
        )
    control_drums = as_numpy(control_full)[0, 0]
    reconstruct_ms = (time.perf_counter_ns() - reconstruct_started) / 1e6
    comparison = compare_arrays(
        torch_drums,
        control_drums,
        reference_label="official f7e0c4bc Torch drums E2E",
        candidate_label="Kani htdemucs_ft.onnx conversion-control drums E2E",
        interpretation="Conversion-boundary parity gate; not a model-quality comparison.",
    )
    gates = {
        "minimumWaveformDeltaSnrDb": 60.0,
        "maximumAbsoluteDelta": 1e-4,
        "minimumCosineSimilarity": 0.99999,
    }
    accepted = bool(
        comparison["referenceToWaveformDeltaSnrDb"] is not None
        and comparison["referenceToWaveformDeltaSnrDb"] >= gates["minimumWaveformDeltaSnrDb"]
        and comparison["maximumAbsoluteDelta"] <= gates["maximumAbsoluteDelta"]
        and comparison["cosineSimilarity"] is not None
        and comparison["cosineSimilarity"] >= gates["minimumCosineSimilarity"]
    )
    report = {
        "schemaVersion": 1,
        "status": "complete" if accepted else "rejected",
        "scope": "Kani neural-core conversion-control E2E parity gate before Psytrance rendering",
        "accepted": accepted,
        "controlArtifact": {
            **identity(control_path),
            "hfRepository": "Kani95/htdemucs-ft-ort",
            "hfRevision": HF_CONTROL_REVISION,
            "semanticRole": "conversion control shaped from official f7e0c4bc drums specialist; not the four-model FT bag",
        },
        "input": {
            "track": track_evidence(track),
            "windowPlan": asdict(plan),
            "normalizedWaveformDataSha256": hashlib.sha256(waveform.tobytes(order="C")).hexdigest(),
            "spectrumDataSha256": hashlib.sha256(spectrum.tobytes(order="C")).hexdigest(),
        },
        "torchReference": {
            "model": drums_model_report,
            "inferenceWallMs": torch_ms,
            "drumsDataSha256": hashlib.sha256(torch_drums.tobytes(order="C")).hexdigest(),
        },
        "controlRuntime": {
            "hostDspModel": dsp_model_report,
            "session": session_report,
            "stftWallMs": stft_ms,
            "onnxInferenceWallMs": ort_ms,
            "istftAndCombineWallMs": reconstruct_ms,
            "drumsDataSha256": hashlib.sha256(control_drums.tobytes(order="C")).hexdigest(),
        },
        "comparison": comparison,
        "gates": gates,
        "tool": tool,
    }
    report_path = output_root / "kani-conversion-control-validation.json"
    write_json(report_path, report)
    del session
    del dsp_model
    gc.collect()
    if not accepted:
        raise ValueError("Kani conversion-control E2E parity gate failed")
    print(json.dumps({
        "stage": "kani-conversion-control",
        "track": track.slug,
        "accepted": True,
        "waveformDeltaSnrDb": comparison["referenceToWaveformDeltaSnrDb"],
        "maximumAbsoluteDelta": comparison["maximumAbsoluteDelta"],
    }), flush=True)
    return report


def run_psy_onnx(
    torch: Any,
    ort: Any,
    base_artifact: TorchArtifact,
    tracks: list[TrackContract],
    output_root: Path,
    repo: Path,
    threads: int,
    tool: dict[str, Any],
) -> dict[str, Any]:
    psy_root = repo / "models/demucs/candidates/htdemucs-psy-ft" / HF_PSY_REVISION
    model_path = psy_root / "psytrance_ft.onnx"
    manifest_path = psy_root / "manifest.json"
    if sha256_file(model_path) != PSY_ONNX_SHA256:
        raise ValueError("Psytrance ONNX SHA mismatch")
    if sha256_file(manifest_path) != PSY_MANIFEST_SHA256:
        raise ValueError("Psytrance manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("window") != WINDOW_SAMPLES or manifest.get("hop") != PSY_NATIVE_STRIDE:
        raise ValueError("Unexpected Psytrance publisher window contract")

    dsp_model, dsp_report = load_torch_model(torch, base_artifact)
    session, session_report = open_ort_session(ort, model_path, threads)
    from export_htdemucs_litert_candidate import (
        as_numpy,
        reconstruct_branches,
        spec_to_channels,
    )

    model_report = {
        "modelId": "psytrance-onnx-core",
        "researchOnly": True,
        "unverifiedTrainingProvenance": True,
        "noTorchOracle": True,
        "hfRepository": "Kani95/htdemucs-psy-ft",
        "hfRevision": HF_PSY_REVISION,
        "onnx": identity(model_path),
        "publisherManifest": identity(manifest_path),
        "publisherDeclaredHopSamples": PSY_NATIVE_STRIDE,
        "hostDspReference": dsp_report,
        "runtime": session_report,
    }

    def run_one(track: TrackContract, model_id: str, stride: int) -> dict[str, Any]:
        expected_shape = (4, CHANNELS, track.selected_frames)
        if valid_raw(
            output_root,
            model_id,
            track,
            expected_shape,
            tool["sha256"],
            PSY_ONNX_SHA256,
        ):
            return json.loads(raw_paths(output_root, model_id, track.slug)[1].read_text(encoding="utf-8"))

        def infer(window: np.ndarray, plan: WindowPlan) -> tuple[np.ndarray, dict[str, Any]]:
            waveform = np.ascontiguousarray(window[np.newaxis, ...], dtype=np.float32)
            stft_started = time.perf_counter_ns()
            with torch.inference_mode():
                spectrum_tensor = spec_to_channels(torch, dsp_model._spec(torch.from_numpy(waveform)))
            spectrum = as_numpy(spectrum_tensor)
            stft_ms = (time.perf_counter_ns() - stft_started) / 1e6
            ort_started = time.perf_counter_ns()
            frequency, time_branch = session.run(
                ["output", "add_67"], {"input": waveform, "x": spectrum}
            )
            ort_ms = (time.perf_counter_ns() - ort_started) / 1e6
            reconstruction_started = time.perf_counter_ns()
            with torch.inference_mode():
                _, combined = reconstruct_branches(
                    torch,
                    dsp_model,
                    torch.from_numpy(np.ascontiguousarray(frequency, dtype=np.float32)),
                    torch.from_numpy(np.ascontiguousarray(time_branch, dtype=np.float32)),
                    WINDOW_SAMPLES,
                )
            value = as_numpy(combined)[0]
            reconstruction_ms = (time.perf_counter_ns() - reconstruction_started) / 1e6
            return value, {
                "backend": "ONNX Runtime CPU neural core plus Torch official host DSP",
                "stftWallMs": stft_ms,
                "onnxInferenceWallMs": ort_ms,
                "istftAndCombineWallMs": reconstruction_ms,
                "wallMs": stft_ms + ort_ms + reconstruction_ms,
            }

        started = time.perf_counter_ns()
        value, contract = accumulate_track(track, stride, 4, infer)
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        array_path, report_path = raw_paths(output_root, model_id, track.slug)
        save_npy(array_path, value)
        report = {
            "schemaVersion": 1,
            "status": "complete",
            "scope": "Batch 4A raw Psytrance ONNX E2E output before PCM quantization",
            "model": model_report,
            "source": track_evidence(track)["selection"] | {
                "canonicalWavFileSha256": track.file_sha256,
            },
            "contract": contract | {
                "contractId": (
                    "repo-canonical-7p8s-overlap25" if stride == CANONICAL_STRIDE
                    else "publisher-native-7p8s-overlap50-sensitivity"
                ),
                "publisherHopConflictExplicit": stride != PSY_NATIVE_STRIDE,
            },
            "totalTrackWallMs": wall_ms,
            "realtimeFactor": wall_ms / 1000.0 / (track.selected_frames / SAMPLE_RATE),
            "output": {
                **identity(array_path),
                "shape": list(value.shape),
                "dtype": "float32",
                "allFinite": bool(np.isfinite(value).all()),
            },
            "tool": tool,
        }
        write_json(report_path, report)
        print(json.dumps({
            "stage": model_id,
            "track": track.slug,
            "wallMs": wall_ms,
            "realtimeFactor": report["realtimeFactor"],
        }), flush=True)
        return report

    canonical_reports = [
        run_one(track, "psytrance-onnx-canonical25", CANONICAL_STRIDE)
        for track in tracks
    ]
    native_reports: list[dict[str, Any]] = []
    kygo = next((track for track in tracks if track.slug == "kygo-ed-sheeran-i-see-fire-kygo-remix"), None)
    if kygo is not None:
        native_reports.append(
            run_one(kygo, "psytrance-onnx-native50", PSY_NATIVE_STRIDE)
        )
    del session
    del dsp_model
    gc.collect()
    result = {
        "schemaVersion": 1,
        "status": "complete",
        "model": model_report,
        "canonical25Tracks": [item["source"]["selectedPcmSha256"] for item in canonical_reports],
        "native50SensitivityTrackCount": len(native_reports),
    }
    write_json(output_root / "raw" / "psytrance-model-report.json", result)
    return result


def iter_blocks(values: np.ndarray, block: int = 1 << 20) -> Iterable[np.ndarray]:
    flat = np.asarray(values).reshape(-1)
    for offset in range(0, flat.size, block):
        yield flat[offset : offset + block]


def array_stats(values: np.ndarray) -> dict[str, Any]:
    count = 0
    non_finite = 0
    clipped = 0
    total = 0.0
    square = 0.0
    minimum = math.inf
    maximum = -math.inf
    peak = 0.0
    for raw in iter_blocks(values):
        finite = np.isfinite(raw)
        non_finite += int(raw.size - np.count_nonzero(finite))
        if not bool(finite.all()):
            continue
        chunk = raw.astype(np.float64, copy=False)
        count += int(chunk.size)
        total += float(chunk.sum())
        square += float(np.dot(chunk, chunk))
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        peak = max(peak, float(np.abs(chunk).max()))
        clipped += int(np.count_nonzero((chunk < -1.0) | (chunk > 1.0)))
    if non_finite:
        return {
            "sampleCount": int(np.asarray(values).size),
            "finiteCount": count,
            "nonFiniteCount": non_finite,
            "allFinite": False,
        }
    return {
        "sampleCount": count,
        "finiteCount": count,
        "nonFiniteCount": 0,
        "allFinite": True,
        "mean": total / count,
        "rms": math.sqrt(square / count),
        "minimum": minimum,
        "maximum": maximum,
        "peakAbsolute": peak,
        "clippedSampleCount": clipped,
        "clippedSampleFraction": clipped / count,
    }


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
    interpretation: str = "Waveform difference only; not SDR/SIR/SAR or a quality score.",
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Comparison shape mismatch: {reference.shape} != {candidate.shape}")
    n = 0
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = delta2 = 0.0
    max_delta = 0.0
    changed = 0
    for x_raw, y_raw in zip(iter_blocks(reference), iter_blocks(candidate), strict=True):
        if not bool(np.isfinite(x_raw).all() and np.isfinite(y_raw).all()):
            raise ValueError("Non-finite comparison input")
        x = x_raw.astype(np.float64, copy=False)
        y = y_raw.astype(np.float64, copy=False)
        delta = y - x
        n += x.size
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_x2 += float(np.dot(x, x))
        sum_y2 += float(np.dot(y, y))
        sum_xy += float(np.dot(x, y))
        delta2 += float(np.dot(delta, delta))
        max_delta = max(max_delta, float(np.abs(delta).max()))
        changed += int(np.count_nonzero(delta))
    reference_rms = math.sqrt(sum_x2 / n)
    candidate_rms = math.sqrt(sum_y2 / n)
    delta_rms = math.sqrt(delta2 / n)
    cosine_denominator = math.sqrt(sum_x2 * sum_y2)
    covariance = sum_xy - sum_x * sum_y / n
    variance_x = sum_x2 - sum_x * sum_x / n
    variance_y = sum_y2 - sum_y * sum_y / n
    pearson_denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
    return {
        "referenceLabel": reference_label,
        "candidateLabel": candidate_label,
        "sampleCount": n,
        "changedSampleCount": changed,
        "exactlyEqual": changed == 0,
        "referenceRms": reference_rms,
        "candidateRms": candidate_rms,
        "candidateRmsChangeDb": 20.0 * math.log10(max(candidate_rms, 1e-30) / max(reference_rms, 1e-30)),
        "rootMeanSquareDelta": delta_rms,
        "maximumAbsoluteDelta": max_delta,
        "referenceToWaveformDeltaSnrDb": (
            None if delta_rms == 0 else 20.0 * math.log10(max(reference_rms, 1e-30) / delta_rms)
        ),
        "cosineSimilarity": sum_xy / cosine_denominator if cosine_denominator else None,
        "pearsonCorrelation": covariance / pearson_denominator if pearson_denominator else None,
        "interpretation": interpretation,
    }


def seam_metrics(stems: np.ndarray, stride: int) -> dict[str, Any]:
    boundaries = list(range(stride, stems.shape[-1], stride))
    per_stem: dict[str, Any] = {}
    for index, stem in enumerate(STEMS):
        details = []
        for boundary in boundaries:
            left = max(0, boundary - 2048)
            right = min(stems.shape[-1], boundary + 2048)
            local_parts = []
            if boundary - left > 1:
                local_parts.append(np.abs(np.diff(stems[index, :, left:boundary], axis=-1)).reshape(-1))
            if right - boundary > 1:
                local_parts.append(np.abs(np.diff(stems[index, :, boundary:right], axis=-1)).reshape(-1))
            local = np.concatenate(local_parts) if local_parts else np.zeros(1, dtype=np.float32)
            p95 = float(np.percentile(local, 95.0))
            jump = np.abs(stems[index, :, boundary] - stems[index, :, boundary - 1])
            details.append({
                "boundaryFrame": boundary,
                "channelJumpAbsolute": [float(value) for value in jump],
                "maximumJumpAbsolute": float(jump.max()),
                "localDerivativeAbsoluteP95": p95,
                "jumpToLocalDerivativeP95Ratio": float(jump.max()) / max(p95, 1e-30),
            })
        per_stem[stem] = {
            "boundaries": details,
            "maximumJumpToLocalDerivativeP95Ratio": max(
                (item["jumpToLocalDerivativeP95Ratio"] for item in details), default=0.0
            ),
        }
    return {"strideSamples": stride, "boundaryCount": len(boundaries), "perStem": per_stem}


def load_mix(track: TrackContract) -> np.ndarray:
    with wave.open(str(track.path), "rb") as reader:
        payload = reader.readframes(track.selected_frames)
    if hashlib.sha256(payload).hexdigest() != track.selected_pcm_sha256:
        raise ValueError(f"Selected PCM SHA changed for {track.slug}")
    values = np.frombuffer(payload, dtype="<i2").reshape(-1, CHANNELS)
    return np.ascontiguousarray(values.T.astype(np.float32) / np.float32(32768.0))


def reconstruction_metrics(mix: np.ndarray, stems: np.ndarray) -> dict[str, Any]:
    reconstruction = np.zeros_like(mix, dtype=np.float64)
    for index in range(len(STEMS)):
        reconstruction += stems[index].astype(np.float64, copy=False)
    residual = mix.astype(np.float64, copy=False) - reconstruction
    mix_stats = array_stats(mix)
    residual_stats = array_stats(residual)
    comparison = compare_arrays(
        mix,
        reconstruction,
        reference_label="selected input mixture",
        candidate_label="raw sum of four pre-quantized stems",
        interpretation="Mixture reconstruction waveform diagnostic; not SDR/SIR/SAR or a quality score.",
    )
    return {
        "accumulationDtype": "float64",
        "definition": "residual = selected input mixture - raw sum of four pre-quantized stems",
        "noResidualRedistribution": True,
        "noLimitingOrLoudnessMatching": True,
        "mix": mix_stats,
        "stemSum": array_stats(reconstruction),
        "residual": residual_stats,
        "mixVersusStemSum": comparison,
        "residualRmsRelativeToMixDb": 20.0 * math.log10(
            max(residual_stats["rms"], 1e-30) / max(mix_stats["rms"], 1e-30)
        ),
        "mixReconstructionSnrDb": 20.0 * math.log10(
            max(mix_stats["rms"], 1e-30) / max(residual_stats["rms"], 1e-30)
        ),
        "interpretation": "Mixture coherence diagnostic only; not an automatic quality veto.",
    }


def write_pcm16_wav(path: Path, values: np.ndarray) -> dict[str, Any]:
    stats = StemStats()
    payload = encode_pcm16(np.ascontiguousarray(values, dtype=np.float32), stats)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with wave.open(str(partial), "wb") as writer:
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(payload)
    partial.replace(path)
    return {
        **identity(path),
        "frameCount": values.shape[-1],
        "sampleRate": SAMPLE_RATE,
        "channelCount": CHANNELS,
        "sampleFormat": "signed-pcm16-le",
        "pcmSha256": hashlib.sha256(payload).hexdigest(),
        "quantization": "clip [-1,1], multiply 32767, floor(x+0.5), signed PCM16",
        "preQuantization": stats.evidence(),
    }


def verify_rendered_wav(path: Path, expected: dict[str, Any], frames: int) -> bool:
    if not path.is_file() or sha256_file(path) != expected.get("sha256"):
        return False
    digest = hashlib.sha256()
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getframerate() != SAMPLE_RATE
                or reader.getnchannels() != CHANNELS
                or reader.getsampwidth() != 2
                or reader.getcomptype() != "NONE"
                or reader.getnframes() != frames
            ):
                return False
            while True:
                payload = reader.readframes(65_536)
                if not payload:
                    break
                digest.update(payload)
    except (OSError, wave.Error):
        return False
    return digest.hexdigest() == expected.get("pcmSha256")


def load_raw(output_root: Path, model_id: str, slug: str) -> np.ndarray:
    path, _ = raw_paths(output_root, model_id, slug)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def raw_report(output_root: Path, model_id: str, slug: str) -> dict[str, Any]:
    return json.loads(raw_paths(output_root, model_id, slug)[1].read_text(encoding="utf-8"))


def inference_cost(output_root: Path, slug: str, model_ids: list[str]) -> dict[str, Any]:
    reports = [raw_report(output_root, model_id, slug) for model_id in model_ids]
    wall_ms = sum(report["totalTrackWallMs"] for report in reports)
    duration = reports[0]["source"]["durationSeconds"]
    return {
        "rawForwardModelIds": model_ids,
        "forwardCountPerWindow": len(model_ids),
        "measuredConstituentTrackWallMs": wall_ms,
        "constituentRealtimeFactor": wall_ms / 1000.0 / duration,
        "modelLoadExcluded": True,
        "note": "Hybrid and bag costs are the sum of independently measured constituent passes.",
    }


def variant_arrays(output_root: Path, slug: str) -> dict[str, tuple[np.ndarray, dict[str, str]]]:
    base = load_raw(output_root, "base-955717e8", slug)
    drums = load_raw(output_root, "ft-drums-f7e0c4bc", slug)[0]
    bass = load_raw(output_root, "ft-bass-d12395a8", slug)[0]
    other = load_raw(output_root, "ft-other-92cfc3b6", slug)[0]
    vocals = load_raw(output_root, "ft-vocals-04573f0d", slug)[0]
    psy = load_raw(output_root, "psytrance-onnx-canonical25", slug)
    return {
        "official-base": (base, {stem: "base-955717e8" for stem in STEMS}),
        "official-ft-bag": (
            np.stack((drums, bass, other, vocals)),
            {
                "drums": "ft-drums-f7e0c4bc",
                "bass": "ft-bass-d12395a8",
                "other": "ft-other-92cfc3b6",
                "vocals": "ft-vocals-04573f0d",
            },
        ),
        "base-plus-vocals": (
            np.stack((base[0], base[1], base[2], vocals)),
            {"drums": "base-955717e8", "bass": "base-955717e8", "other": "base-955717e8", "vocals": "ft-vocals-04573f0d"},
        ),
        "base-plus-drums": (
            np.stack((drums, base[1], base[2], base[3])),
            {"drums": "ft-drums-f7e0c4bc", "bass": "base-955717e8", "other": "base-955717e8", "vocals": "base-955717e8"},
        ),
        "psytrance-onnx": (psy, {stem: "psytrance-onnx-canonical25" for stem in STEMS}),
    }


def render_variant(
    output_root: Path,
    track: TrackContract,
    variant_id: str,
    stems: np.ndarray,
    assembly: dict[str, str],
    base: np.ndarray,
    mix: np.ndarray,
    stride: int,
    cost: dict[str, Any],
    tool: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = (4, CHANNELS, track.selected_frames)
    if stems.shape != expected or stems.dtype != np.float32 or not bool(np.isfinite(stems).all()):
        raise ValueError(f"Invalid assembled variant {variant_id}: {stems.shape}/{stems.dtype}")
    final = output_root / "tracks" / track.slug / variant_id
    report_path = final / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        wavs_valid = all(
            verify_rendered_wav(
                final / f"{stem}.wav",
                report.get("stems", {}).get(stem, {}).get("outputWav", {}),
                track.selected_frames,
            )
            for stem in STEMS
        )
        if (
            report.get("status") == "complete"
            and report.get("tool", {}).get("sha256") == tool["sha256"]
            and report.get("source", {}).get("selectedPcmSha256") == track.selected_pcm_sha256
            and wavs_valid
        ):
            return report
        raise FileExistsError(f"Invalid existing variant output: {final}")
    stage = final.with_name(final.name + ".partial")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    started = time.perf_counter_ns()
    stem_reports: dict[str, Any] = {}
    energies: dict[str, float] = {}
    for index, stem in enumerate(STEMS):
        stats = array_stats(stems[index])
        energies[stem] = stats["rms"] ** 2 * stats["sampleCount"]
        output = write_pcm16_wav(stage / f"{stem}.wav", stems[index])
        output["path"] = str((final / f"{stem}.wav").resolve())
        stem_reports[stem] = {
            "preQuantization": stats,
            "versusOfficialBase": compare_arrays(
                base[index],
                stems[index],
                reference_label=f"official-base/{stem}",
                candidate_label=f"{variant_id}/{stem}",
            ),
            "outputWav": output,
        }
    total_energy = sum(energies.values())
    for stem in STEMS:
        stem_reports[stem]["energyShareOfRenderedStems"] = energies[stem] / total_energy if total_energy else None
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "scope": "HTDemucs four-stem Batch 4A host quality rendering",
        "variantId": variant_id,
        "source": track_evidence(track)["selection"] | {
            "canonicalWavFileSha256": track.file_sha256,
        },
        "contract": {
            "sampleRate": SAMPLE_RATE,
            "channelCount": CHANNELS,
            "selectedFrames": track.selected_frames,
            "windowSamples": WINDOW_SAMPLES,
            "strideSamples": stride,
            "overlap": (WINDOW_SAMPLES - stride) / WINDOW_SAMPLES,
            "stemOrder": list(STEMS),
            "globalNormalization": "selected-prefix mono-reference mean and correction-1 std",
            "residualRedistribution": "none",
            "postSeparationLoudnessMatching": "none",
        },
        "assembly": {
            "stemSourceModelIds": assembly,
            "officialSpecialistRule": "retain only each specialist's documented target stem",
        },
        "inferenceCost": cost,
        "stems": stem_reports,
        "seams": seam_metrics(stems, stride),
        "mixtureReconstruction": reconstruction_metrics(mix, stems),
        "renderWallMs": (time.perf_counter_ns() - started) / 1e6,
        "tool": tool,
        **(extra or {}),
    }
    write_json(stage / "report.json", report)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(final)
    return report


def render_all(
    output_root: Path,
    tracks: list[TrackContract],
    tool: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for track in tracks:
        arrays = variant_arrays(output_root, track.slug)
        base = arrays["official-base"][0]
        mix = load_mix(track)
        costs = {
            "official-base": inference_cost(output_root, track.slug, ["base-955717e8"]),
            "official-ft-bag": inference_cost(output_root, track.slug, [
                "ft-drums-f7e0c4bc", "ft-bass-d12395a8", "ft-other-92cfc3b6", "ft-vocals-04573f0d"
            ]),
            "base-plus-vocals": inference_cost(output_root, track.slug, ["base-955717e8", "ft-vocals-04573f0d"]),
            "base-plus-drums": inference_cost(output_root, track.slug, ["base-955717e8", "ft-drums-f7e0c4bc"]),
            "psytrance-onnx": inference_cost(output_root, track.slug, ["psytrance-onnx-canonical25"]),
        }
        track_results: dict[str, Any] = {}
        for variant_id in PRIMARY_VARIANTS:
            values, assembly = arrays[variant_id]
            report = render_variant(
                output_root, track, variant_id, np.asarray(values, dtype=np.float32),
                assembly, np.asarray(base), mix, CANONICAL_STRIDE, costs[variant_id], tool,
                {
                    "researchOnly": variant_id == "psytrance-onnx",
                    "qualityGroundTruthAvailable": False,
                },
            )
            report_path = output_root / "tracks" / track.slug / variant_id / "report.json"
            track_results[variant_id] = {
                "report": str(report_path),
                "reportSha256": sha256_file(report_path),
                "allFinite": all(item["preQuantization"]["allFinite"] for item in report["stems"].values()),
                "mixReconstructionSnrDb": report["mixtureReconstruction"]["mixReconstructionSnrDb"],
            }

        if track.slug == "kygo-ed-sheeran-i-see-fire-kygo-remix":
            native = load_raw(output_root, "psytrance-onnx-native50", track.slug)
            canonical_psy = arrays["psytrance-onnx"][0]
            report = render_variant(
                output_root, track, SENSITIVITY_VARIANT, np.asarray(native),
                {stem: "psytrance-onnx-native50" for stem in STEMS},
                np.asarray(base), mix, PSY_NATIVE_STRIDE,
                inference_cost(output_root, track.slug, ["psytrance-onnx-native50"]), tool,
                {
                    "researchOnly": True,
                    "excludedFromPrimaryBlindSet": True,
                    "comparisonVersusCanonical25": {
                        stem: compare_arrays(
                            canonical_psy[index],
                            native[index],
                            reference_label=f"psytrance-canonical25/{stem}",
                            candidate_label=f"psytrance-native50/{stem}",
                            interpretation="OLA contract sensitivity only; not a quality score.",
                        )
                        for index, stem in enumerate(STEMS)
                    },
                    "interpretation": "Publisher-hop sensitivity on an electronic non-Psytrance control; not domain validation.",
                },
            )
            report_path = output_root / "tracks" / track.slug / SENSITIVITY_VARIANT / "report.json"
            track_results[SENSITIVITY_VARIANT] = {
                "report": str(report_path),
                "reportSha256": sha256_file(report_path),
                "mixReconstructionSnrDb": report["mixtureReconstruction"]["mixReconstructionSnrDb"],
            }

        base_reconstruction = mix.astype(np.float64) - sum(
            np.asarray(base[index], dtype=np.float64) for index in range(4)
        )
        hybrid_checks = {}
        for variant_id, changed_stem, specialist in (
            ("base-plus-vocals", "vocals", load_raw(output_root, "ft-vocals-04573f0d", track.slug)[0]),
            ("base-plus-drums", "drums", load_raw(output_root, "ft-drums-f7e0c4bc", track.slug)[0]),
        ):
            candidate = arrays[variant_id][0]
            residual = mix.astype(np.float64) - sum(
                np.asarray(candidate[index], dtype=np.float64) for index in range(4)
            )
            expected_delta = -(
                np.asarray(specialist, dtype=np.float64)
                - np.asarray(base[STEMS.index(changed_stem)], dtype=np.float64)
            )
            actual_delta = residual - base_reconstruction
            hybrid_checks[variant_id] = {
                "identity": f"residual_{variant_id} - residual_base = -({changed_stem}_specialist - {changed_stem}_base)",
                "maximumAbsoluteIdentityError": float(np.max(np.abs(actual_delta - expected_delta))),
            }
        results[track.slug] = {"variants": track_results, "hybridResidualIdentities": hybrid_checks}
        print(json.dumps({"stage": "render", "track": track.slug, "status": "complete"}), flush=True)
        del arrays
        gc.collect()
    return results


def determinism_check(
    torch: Any,
    base_artifact: TorchArtifact,
    track: TrackContract,
    output_root: Path,
) -> dict[str, Any]:
    model, model_report = load_torch_model(torch, base_artifact)

    def infer(window: np.ndarray, plan: WindowPlan) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(torch.from_numpy(window[np.newaxis, ...]))
        return output.detach().cpu().numpy()[0].astype(np.float32, copy=False), {
            "wallMs": (time.perf_counter_ns() - started) / 1e6,
        }

    repeated, contract = accumulate_track(track, CANONICAL_STRIDE, 4, infer)
    reference = load_raw(output_root, "base-955717e8", track.slug)
    comparison = compare_arrays(
        reference,
        repeated,
        reference_label="first official-base run",
        candidate_label="fresh-load official-base rerun",
        interpretation="Determinism comparison.",
    )
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "scope": "fresh model load and full 30-second deterministic rerun",
        "track": track_evidence(track),
        "model": model_report,
        "contract": contract,
        "comparison": comparison,
        "bitExactGatePassed": comparison["exactlyEqual"],
        "repeatDataSha256": hashlib.sha256(repeated.tobytes(order="C")).hexdigest(),
        "referenceDataSha256": hashlib.sha256(np.asarray(reference).tobytes(order="C")).hexdigest(),
        "tool": identity(Path(__file__).resolve()),
    }
    del model
    del repeated
    gc.collect()
    write_json(output_root / "determinism-check.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/htdemucs4-batch4a-host-20260805"),
    )
    parser.add_argument("--demucs-root", type=Path, default=Path(".tmp/demucs-adefossez"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--track", action="append", choices=tuple(TRACKS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.threads <= 64:
        raise ValueError("--threads must be in [1,64]")
    selected_frames_float = args.duration_seconds * SAMPLE_RATE
    if selected_frames_float != int(selected_frames_float) or selected_frames_float < 2:
        raise ValueError("Duration must map to an integral frame count")
    selected_frames = int(selected_frames_float)
    repo = args.repo.resolve()
    source_root = (repo / args.source_root).resolve() if not args.source_root.is_absolute() else args.source_root.resolve()
    output_root = (repo / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    demucs_root = (repo / args.demucs_root).resolve() if not args.demucs_root.is_absolute() else args.demucs_root.resolve()
    if git_revision(demucs_root) != DEMUCS_REVISION:
        raise ValueError("Pinned Demucs checkout revision mismatch")
    if str(demucs_root) not in sys.path:
        sys.path.insert(0, str(demucs_root))
    import torch
    import onnxruntime as ort

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    track_slugs = args.track or list(TRACKS)
    tracks = [load_track_contract(source_root, slug, selected_frames) for slug in track_slugs]
    tool_path = Path(__file__).resolve()
    tool = identity(tool_path)
    artifacts = expected_artifacts(repo)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    progress: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "startedUtc": datetime.now(timezone.utc).isoformat(),
        "tracks": [track_evidence(track) for track in tracks],
        "completedStages": [],
        "tool": tool,
    }
    write_json(progress_path, progress)
    try:
        conversion_control = validate_kani_conversion_control(
            torch,
            ort,
            artifacts,
            tracks,
            output_root,
            repo,
            args.threads,
            tool,
        )
        progress["completedStages"].append("kani-conversion-control")
        write_json(progress_path, progress)
        torch_results = {}
        for artifact in artifacts:
            torch_results[artifact.model_id] = run_torch_artifact(
                torch, artifact, tracks, output_root, tool
            )
            progress["completedStages"].append(artifact.model_id)
            write_json(progress_path, progress)
        psy_result = run_psy_onnx(
            torch, ort, artifacts[0], tracks, output_root, repo, args.threads, tool
        )
        progress["completedStages"].append("psytrance-onnx")
        write_json(progress_path, progress)
        rendered = render_all(output_root, tracks, tool)
        progress["completedStages"].append("render")
        write_json(progress_path, progress)
        deterministic = determinism_check(torch, artifacts[0], tracks[0], output_root)
        progress["completedStages"].append("determinism")
        write_json(progress_path, progress)
        batch_report = {
            "schemaVersion": 1,
            "status": "complete",
            "scope": "HTDemucs four-stem Batch 4A host quality short test",
            "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "onnxRuntime": ort.__version__,
                "numpy": np.__version__,
                "safetensors": package_version("safetensors"),
                "threads": args.threads,
            },
            "demucs": {"path": str(demucs_root), "revision": DEMUCS_REVISION},
            "primaryContract": {
                "contractId": "repo-canonical-7p8s-overlap25",
                "sampleRate": SAMPLE_RATE,
                "windowSamples": WINDOW_SAMPLES,
                "strideSamples": CANONICAL_STRIDE,
                "overlap": 0.25,
                "selectedFrames": selected_frames,
                "stemOrder": list(STEMS),
                "globalNormalizationScope": "each selected prefix, once and shared across variants",
            },
            "psytrancePublisherContractConflict": {
                "publisherManifestHopSamples": PSY_NATIVE_STRIDE,
                "publisherOverlap": 0.5,
                "primaryComparisonUsesCanonical25": True,
                "native50Sensitivity": "Kygo electronic non-Psytrance control only",
            },
            "tracks": {track.slug: track_evidence(track) for track in tracks},
            "rawTorch": torch_results,
            "kaniConversionControl": {
                "report": str(output_root / "kani-conversion-control-validation.json"),
                "reportSha256": sha256_file(output_root / "kani-conversion-control-validation.json"),
                "accepted": conversion_control["accepted"],
                "comparison": conversion_control["comparison"],
            },
            "rawPsytrance": psy_result,
            "rendered": rendered,
            "determinism": {
                "report": str(output_root / "determinism-check.json"),
                "reportSha256": sha256_file(output_root / "determinism-check.json"),
                "bitExactGatePassed": deterministic["bitExactGatePassed"],
            },
            "interpretation": {
                "isolatedGroundTruthAvailable": False,
                "waveformDeltaIsNotQualityMetric": True,
                "noSdrSirSarClaims": True,
                "mixtureResidualIsCoherenceDiagnosticNotQualityVeto": True,
                "psytranceCandidate": "research-only, unverified training provenance, no Torch oracle",
                "kygo": "electronic non-Psytrance control; cannot establish in-domain Psytrance quality",
            },
            "tool": tool,
        }
        write_json(output_root / "batch-report.json", batch_report)
        progress["status"] = "complete"
        progress["completedUtc"] = datetime.now(timezone.utc).isoformat()
        progress["batchReport"] = {
            "path": str(output_root / "batch-report.json"),
            "sha256": sha256_file(output_root / "batch-report.json"),
        }
        write_json(progress_path, progress)
    except BaseException as exc:
        progress["status"] = "failed"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        progress["failedUtc"] = datetime.now(timezone.utc).isoformat()
        write_json(progress_path, progress)
        raise
    print(json.dumps({
        "status": "complete",
        "outputRoot": str(output_root),
        "batchReport": str(output_root / "batch-report.json"),
        "trackCount": len(tracks),
        "variantCount": len(PRIMARY_VARIANTS),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
