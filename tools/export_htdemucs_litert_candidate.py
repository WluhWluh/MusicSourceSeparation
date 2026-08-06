#!/usr/bin/env python3
"""Export a fixed-window HTDemucs neural core to LiteRT.

The model boundary follows the public MIT-licensed demucs-lite neural-core
rewrite: STFT and iSTFT stay on the host, while the real-valued frequency and
time branches are converted together. The canonical input is the maintainer's
safetensors artifact, not a community ONNX model or a pickle checkpoint.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MethodType
from typing import Any, Callable

import numpy as np


PROFILE_SPECS = {
    "smoke_2s": {
        "modelId": "htdemucs_6s_core_smoke_2s_fp32_v1_0_0",
        "sampleCount": 88_200,
    },
    "canonical_7p8s": {
        "modelId": "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0",
        "sampleCount": 343_980,
    },
}
SOURCE_ORDER = ("drums", "bass", "other", "vocals", "guitar", "piano")
EXPECTED_WEIGHT_BYTES = 54_885_744
EXPECTED_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
EXPECTED_METADATA_BYTES = 10_398
EXPECTED_METADATA_SHA256 = "72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e"
EXPECTED_BAG_MANIFEST_BYTES = 21
EXPECTED_BAG_MANIFEST_SHA256 = "207405151270af8fd81c2373c25d27950916682ac91dca7884a11ce13dad6f58"
EXPECTED_BOUNDARY_SOURCE_BYTES = 29_284
EXPECTED_BOUNDARY_SOURCE_SHA256 = "fc9b1debbc2d0e61f523ccd32a22b19f471bc4404d32fadbd79f38c049abcc7b"
EXPECTED_REFERENCE_EXPORTER_BYTES = 5_879
EXPECTED_REFERENCE_EXPORTER_SHA256 = "57b73642c1ac2d399817a8dff4438db587202a14ffa90877c7b9fb0d95f9e506"
EXPECTED_LOADER_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
EXPECTED_DEMUCS_LITE_REVISION = "9a2a17c7a81843c2ae49674986f9e1e8b5f6915f"
MINIMUM_SNR_DB = 80.0
MAXIMUM_ABSOLUTE_ERROR = 1e-3
LOW_SIGNAL_REFERENCE_RMS = 1e-3
LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR = 1e-6

TENSOR_AXES = {
    "waveformInput": ["batch", "channel", "sample"],
    "spectrumInput": ["batch", "feature", "frequency", "frame"],
    "frequencyOutput": ["batch", "stem", "feature", "frequency", "frame"],
    "waveformOutput": ["batch", "stem", "channel", "sample"],
    "combinedOutput": ["batch", "stem", "channel", "sample"],
}

CANONICAL_OVERLAP = 0.25
CANONICAL_TRANSITION_POWER = 1.0
CANONICAL_OLA_TRACK_SAMPLES = 515_970


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demucs-root", type=Path, required=True)
    parser.add_argument("--demucs-lite-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILE_SPECS), default="smoke_2s")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=20_260_803)
    parser.add_argument("--skip-litert", action="store_true")
    parser.add_argument("--reuse-litert", action="store_true")
    args = parser.parse_args()
    profile = PROFILE_SPECS[args.profile]
    if args.samples is None:
        args.samples = profile["sampleCount"]
    elif args.samples != profile["sampleCount"]:
        parser.error(
            f"--profile {args.profile} requires --samples {profile['sampleCount']}"
        )
    args.model_id = profile["modelId"]
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing source artifact: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Unexpected source artifact {path}: bytes={actual_bytes}, sha256={actual_sha256}"
        )


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_git_revision(path: Path, expected_revision: str, label: str) -> dict[str, Any]:
    revision = git_revision(path)
    if revision != expected_revision:
        raise RuntimeError(f"Unexpected {label} revision: {revision}")
    status_lines = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    working_tree_state = "clean"
    if status_lines:
        if any(line.startswith("??") for line in status_lines):
            raise RuntimeError(f"Untracked files in {label} checkout: {path}")
        for cached in (False, True):
            command = ["git", "-C", str(path), "diff", "--ignore-cr-at-eol", "--quiet"]
            if cached:
                command.append("--cached")
            if subprocess.run(command, check=False).returncode != 0:
                raise RuntimeError(f"Semantic changes in {label} checkout: {path}")
        working_tree_state = "line-endings-only"
    return {
        "revision": revision,
        "workingTreeState": working_tree_state,
        "statusEntryCount": len(status_lines),
    }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Metric shape mismatch: {reference.shape} != {candidate.shape}")
    finite = bool(np.isfinite(candidate).all())
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    candidate64 = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = candidate64 - reference64
    signal_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    error_rms = float(np.sqrt(np.mean(delta * delta)))
    snr_db = 20.0 * math.log10(max(signal_rms, 1e-30) / max(error_rms, 1e-30))
    return {
        "finite": finite,
        "maxAbsoluteError": float(np.max(np.abs(delta))),
        "meanAbsoluteError": float(np.mean(np.abs(delta))),
        "rootMeanSquareError": error_rms,
        "signalRootMeanSquare": signal_rms,
        "signalToNoiseDb": snr_db,
        "bitwiseEqual": bool(np.array_equal(reference, candidate)),
    }


def metric_passes(value: dict[str, Any]) -> bool:
    return bool(
        value["finite"]
        and value["signalToNoiseDb"] >= MINIMUM_SNR_DB
        and value["maxAbsoluteError"] <= MAXIMUM_ABSOLUTE_ERROR
    )


def evaluate_metric(
    value: dict[str, Any],
    *,
    require_absolute_gate: bool,
    allow_low_signal_gate: bool,
    low_signal_maximum_absolute_error: float = LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR,
) -> dict[str, Any]:
    low_signal = bool(
        allow_low_signal_gate
        and value["signalRootMeanSquare"] <= LOW_SIGNAL_REFERENCE_RMS
    )
    if low_signal:
        accepted = bool(
            value["finite"]
            and value["maxAbsoluteError"] <= low_signal_maximum_absolute_error
        )
        basis = "low-signal-absolute"
    else:
        accepted = bool(
            value["finite"]
            and value["signalToNoiseDb"] >= MINIMUM_SNR_DB
            and (
                not require_absolute_gate
                or value["maxAbsoluteError"] <= MAXIMUM_ABSOLUTE_ERROR
            )
        )
        basis = "snr-and-absolute" if require_absolute_gate else "snr-only"
    return {
        "accepted": accepted,
        "basis": basis,
        "lowSignal": low_signal,
        "requireAbsoluteGate": require_absolute_gate,
        "maximumAbsoluteError": (
            low_signal_maximum_absolute_error
            if low_signal
            else MAXIMUM_ABSOLUTE_ERROR if require_absolute_gate else None
        ),
    }


def deterministic_mix(torch: Any, samples: int, sample_rate: int, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    phase = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    left = (
        0.18 * torch.sin(2 * torch.pi * 110.0 * phase)
        + 0.07 * torch.sin(2 * torch.pi * 997.0 * phase + 0.31)
        + 0.015 * torch.randn(samples, generator=generator)
    )
    right = (
        0.16 * torch.sin(2 * torch.pi * 164.81 * phase + 0.17)
        + 0.06 * torch.sin(2 * torch.pi * 1501.0 * phase)
        + 0.015 * torch.randn(samples, generator=generator)
    )
    return torch.stack((left, right), dim=0).unsqueeze(0).contiguous()


def deterministic_ola_mix(torch: Any, samples: int, sample_rate: int, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    phase = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    duration = max(samples / float(sample_rate), 1e-6)
    envelope = 0.62 + 0.28 * torch.sin(2 * torch.pi * 0.37 * phase + 0.23)
    chirp_left = torch.sin(
        2 * torch.pi * (73.0 * phase + 0.5 * 910.0 * phase.square() / duration)
    )
    chirp_right = torch.sin(
        2 * torch.pi * (131.0 * phase + 0.5 * 677.0 * phase.square() / duration) + 0.41
    )
    left = (
        0.16 * envelope * chirp_left
        + 0.055 * torch.sin(2 * torch.pi * 1009.0 * phase)
        + 0.012 * torch.randn(samples, generator=generator)
    )
    right = (
        0.15 * envelope.flip(0) * chirp_right
        + 0.05 * torch.sin(2 * torch.pi * 1483.0 * phase + 0.19)
        + 0.012 * torch.randn(samples, generator=generator)
    )
    for index, amplitude in (
        (257_985 - 17, 0.19),
        (257_985 + 29, -0.17),
        (343_980 - 11, 0.15),
        (343_980 + 23, -0.13),
        (samples - 37, 0.11),
    ):
        if 0 <= index < samples:
            left[index] += amplitude
            right[index] -= amplitude * 0.73
    return torch.stack((left, right), dim=0).unsqueeze(0).contiguous()


def demucs_triangle_weight(torch: Any, window_samples: int, device: Any) -> Any:
    weight = torch.cat(
        (
            torch.arange(1, window_samples // 2 + 1, device=device),
            torch.arange(window_samples - window_samples // 2, 0, -1, device=device),
        )
    )
    if len(weight) != window_samples:
        raise RuntimeError(f"Unexpected triangular weight length: {len(weight)}")
    return weight / weight.max()


def demucs_padded_chunk(torch: Any, mix: Any, offset: int, window_samples: int) -> tuple[Any, dict[str, int]]:
    track_samples = int(mix.shape[-1])
    actual_samples = min(track_samples - offset, window_samples)
    delta = window_samples - actual_samples
    context_start = offset - delta // 2
    context_end = context_start + window_samples
    source_start = max(0, context_start)
    source_end = min(track_samples, context_end)
    pad_left = source_start - context_start
    pad_right = context_end - source_end
    padded = torch.nn.functional.pad(
        mix[..., source_start:source_end],
        (pad_left, pad_right),
    )
    if padded.shape[-1] != window_samples:
        raise RuntimeError(f"Unexpected padded chunk length: {padded.shape[-1]}")
    return padded, {
        "offset": offset,
        "actualSamples": actual_samples,
        "contextStart": context_start,
        "contextEnd": context_end,
        "sourceStart": source_start,
        "sourceEnd": source_end,
        "padLeft": pad_left,
        "padRight": pad_right,
        "cropLeft": delta // 2,
        "cropRight": delta - delta // 2,
    }


def run_split_ola(
    torch: Any,
    mix: Any,
    source_count: int,
    window_samples: int,
    overlap: float,
    runner: Callable[[Any, int], tuple[Any, dict[str, Any]]],
) -> tuple[Any, list[dict[str, int]], list[dict[str, Any]]]:
    track_samples = int(mix.shape[-1])
    stride_samples = int((1.0 - overlap) * window_samples)
    weight = demucs_triangle_weight(torch, window_samples, mix.device)
    output = torch.zeros(
        mix.shape[0],
        source_count,
        mix.shape[1],
        track_samples,
        device=mix.device,
        dtype=mix.dtype,
    )
    accumulated_weight = torch.zeros(track_samples, device=mix.device, dtype=mix.dtype)
    plans: list[dict[str, int]] = []
    details: list[dict[str, Any]] = []
    for offset in range(0, track_samples, stride_samples):
        padded, plan = demucs_padded_chunk(torch, mix, offset, window_samples)
        separated, detail = runner(padded, offset)
        crop_left = plan["cropLeft"]
        crop_right = plan["cropRight"]
        active = separated[..., crop_left: separated.shape[-1] - crop_right if crop_right else None]
        actual_samples = plan["actualSamples"]
        if active.shape[-1] != actual_samples:
            raise RuntimeError(f"Unexpected active chunk length: {active.shape[-1]}")
        output[..., offset: offset + actual_samples] += active * weight[:actual_samples]
        accumulated_weight[offset: offset + actual_samples] += weight[:actual_samples]
        plans.append(plan)
        details.append(detail)
    if not bool((accumulated_weight > 0).all()):
        raise RuntimeError("OLA accumulated weight contains non-positive values")
    return output / accumulated_weight, plans, details


def metric_bundle(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    return {
        "aggregate": metrics(reference, candidate),
        "perStem": {
            name: metrics(reference[:, index], candidate[:, index])
            for index, name in enumerate(SOURCE_ORDER)
        },
    }


def metric_bundle_passes(bundle: dict[str, Any]) -> bool:
    return bool(
        metric_passes(bundle["aggregate"])
        and all(metric_passes(value) for value in bundle["perStem"].values())
    )


def evaluate_metric_bundle(
    bundle: dict[str, Any],
    *,
    require_absolute_gate: bool,
    allow_low_signal_gate: bool,
    low_signal_maximum_absolute_error: float = LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR,
) -> dict[str, Any]:
    aggregate = evaluate_metric(
        bundle["aggregate"],
        require_absolute_gate=require_absolute_gate,
        allow_low_signal_gate=False,
        low_signal_maximum_absolute_error=low_signal_maximum_absolute_error,
    )
    per_stem = {
        stem: evaluate_metric(
            value,
            require_absolute_gate=require_absolute_gate,
            allow_low_signal_gate=allow_low_signal_gate,
            low_signal_maximum_absolute_error=low_signal_maximum_absolute_error,
        )
        for stem, value in bundle["perStem"].items()
    }
    return {
        "accepted": bool(aggregate["accepted"] and all(item["accepted"] for item in per_stem.values())),
        "aggregate": aggregate,
        "perStem": per_stem,
    }


def global_normalize(torch: Any, mix: Any) -> tuple[Any, Any, Any]:
    reference = mix.mean(dim=1)
    mean = reference.mean()
    std = reference.std() + 1e-8
    return (mix - mean) / std, mean, std


def spec_to_channels(torch: Any, value: Any) -> Any:
    batch, channels, frequency, frames = value.shape
    return (
        torch.view_as_real(value)
        .permute(0, 1, 4, 2, 3)
        .reshape(batch, channels * 2, frequency, frames)
        .contiguous()
    )


def channels_to_spec(torch: Any, value: Any) -> Any:
    prefix = tuple(value.shape[:-3])
    packed_channels, frequency, frames = value.shape[-3:]
    if packed_channels % 2:
        raise RuntimeError(f"Invalid complex channel count: {packed_channels}")
    prefix_axes = tuple(range(len(prefix)))
    channel_axis = len(prefix)
    complex_axis = channel_axis + 1
    frequency_axis = channel_axis + 2
    frame_axis = channel_axis + 3
    unpacked = value.reshape(
        *prefix, packed_channels // 2, 2, frequency, frames
    ).permute(
        *prefix_axes, channel_axis, frequency_axis, frame_axis, complex_axis
    ).contiguous()
    return torch.view_as_complex(unpacked)


def install_deterministic_pos_embedding(model: Any) -> None:
    from demucs.transformer import create_sin_embedding

    transformer = model.crosstransformer
    if transformer.emb != "sin" or transformer.sin_random_shift != 0:
        raise RuntimeError(
            "The pinned deterministic rewrite only covers sin embedding with zero random shift"
        )

    def deterministic(self: Any, length: int, batch: int, channels: int, device: Any) -> Any:
        del batch
        return create_sin_embedding(
            length,
            channels,
            shift=0,
            device=device,
            max_period=self.max_period,
        )

    transformer._get_pos_embedding = MethodType(deterministic, transformer)


def build_core(torch: Any, model: Any) -> Any:
    class HTDemucsCore(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source
            self.training_length = int(source.segment * source.samplerate)

        def forward(self, mix: Any, spectrum_channels: Any) -> tuple[Any, Any]:
            source = self.source
            x = spectrum_channels
            batch, _, frequency, frames = x.shape

            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)

            xt = mix
            mean_time = xt.mean(dim=(1, 2), keepdim=True)
            std_time = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - mean_time) / (1e-5 + std_time)

            saved = []
            saved_time = []
            lengths = []
            lengths_time = []
            for index, encode in enumerate(source.encoder):
                lengths.append(x.shape[-1])
                inject = None
                if index < len(source.tencoder):
                    lengths_time.append(xt.shape[-1])
                    time_encode = source.tencoder[index]
                    xt = time_encode(xt)
                    if not time_encode.empty:
                        saved_time.append(xt)
                    else:
                        inject = xt
                x = encode(x, inject)
                if index == 0 and source.freq_emb is not None:
                    positions = torch.arange(x.shape[-2], device=x.device)
                    embedding = source.freq_emb(positions).t()[None, :, :, None].expand_as(x)
                    x = x + source.freq_emb_scale * embedding
                saved.append(x)

            if source.crosstransformer:
                if source.bottom_channels:
                    _, channels, bins, steps = x.shape
                    x = x.reshape(batch, channels, bins * steps)
                    x = source.channel_upsampler(x)
                    x = x.reshape(batch, -1, bins, steps)
                    xt = source.channel_upsampler_t(xt)
                x, xt = source.crosstransformer(x, xt)
                if source.bottom_channels:
                    _, channels, bins, steps = x.shape
                    x = x.reshape(batch, channels, bins * steps)
                    x = source.channel_downsampler(x)
                    x = x.reshape(batch, -1, bins, steps)
                    xt = source.channel_downsampler_t(xt)

            for index, decode in enumerate(source.decoder):
                skip = saved.pop()
                x, pre = decode(x, skip, lengths.pop())
                offset = source.depth - len(source.tdecoder)
                if index >= offset:
                    time_decode = source.tdecoder[index - offset]
                    time_length = lengths_time.pop()
                    if time_decode.empty:
                        pre = pre[:, :, 0]
                        xt, _ = time_decode(pre, None, time_length)
                    else:
                        time_skip = saved_time.pop()
                        xt, _ = time_decode(xt, time_skip, time_length)

            stems = len(source.sources)
            x = x.view(batch, stems, -1, frequency, frames)
            x = x * std[:, None] + mean[:, None]
            xt = xt.view(batch, stems, -1, self.training_length)
            xt = xt * std_time[:, None] + mean_time[:, None]
            return x, xt

    return HTDemucsCore(model)


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def reconstruct_branches(torch: Any, model: Any, frequency: Any, waveform: Any, samples: int) -> tuple[Any, Any]:
    """Reconstruct the host iSTFT branch and the final hybrid output."""
    frequency_waveform = model._ispec(
        channels_to_spec(torch, frequency),
        length=samples,
    )
    return frequency_waveform, frequency_waveform + waveform


def write_raw(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    little_endian = np.ascontiguousarray(value, dtype="<f4")
    little_endian.tofile(path)
    return {
        "fileName": path.name,
        "byteSize": path.stat().st_size,
        "sha256": sha256(path),
        "dtype": "float32-le",
        "shape": list(value.shape),
    }


def interpreter_runner(model_path: Path) -> tuple[Callable[..., tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    import ai_edge_litert.interpreter as litert

    interpreter = litert.Interpreter(model_path=str(model_path), num_threads=8)
    interpreter.allocate_tensors()
    signatures = interpreter.get_signature_list()
    if "serving_default" not in signatures:
        raise RuntimeError(f"Missing serving_default signature: {signatures}")
    runner = interpreter.get_signature_runner("serving_default")

    def run(mix: np.ndarray, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = runner(args_0=mix, args_1=spectrum)
        values = [as_numpy(value) for value in outputs.values()]
        frequency = next((value for value in values if value.ndim == 5), None)
        waveform = next((value for value in values if value.ndim == 4), None)
        if frequency is None or waveform is None:
            raise RuntimeError(f"Unexpected LiteRT output shapes: {[list(v.shape) for v in values]}")
        return frequency, waveform

    inspection = {
        "signatures": signatures,
        "inputs": [
            {
                "index": int(item["index"]),
                "name": str(item["name"]),
                "dtype": np.dtype(item["dtype"]).name,
                "shape": [int(value) for value in item["shape"]],
            }
            for item in interpreter.get_input_details()
        ],
        "outputs": [
            {
                "index": int(item["index"]),
                "name": str(item["name"]),
                "dtype": np.dtype(item["dtype"]).name,
                "shape": [int(value) for value in item["shape"]],
            }
            for item in interpreter.get_output_details()
        ],
        "tensorCount": len(interpreter.get_tensor_details()),
    }
    return run, inspection


def validate_canonical_ola(
    torch: Any,
    model: Any,
    core: Any,
    run_litert: Callable[..., tuple[np.ndarray, np.ndarray]],
    fixtures: Path,
    seed: int,
) -> dict[str, Any]:
    from demucs.apply import TensorChunk, apply_model

    window_samples = PROFILE_SPECS["canonical_7p8s"]["sampleCount"]
    stride_samples = int((1.0 - CANONICAL_OVERLAP) * window_samples)
    overlap_samples = window_samples - stride_samples
    mix = deterministic_ola_mix(
        torch,
        CANONICAL_OLA_TRACK_SAMPLES,
        model.samplerate,
        seed + 1,
    )
    normalized_mix, global_mean, global_std = global_normalize(torch, mix)

    padding_plans: list[dict[str, int]] = []
    for offset in range(0, CANONICAL_OLA_TRACK_SAMPLES, stride_samples):
        project_padded, plan = demucs_padded_chunk(
            torch,
            normalized_mix,
            offset,
            window_samples,
        )
        official_padded = TensorChunk(normalized_mix, offset, window_samples).padded(
            window_samples
        )
        if not torch.equal(project_padded, official_padded):
            raise RuntimeError(f"Project tail padding differs from TensorChunk at {offset}")
        padding_plans.append(plan)

    official_started = time.perf_counter()
    with torch.inference_mode():
        official_normalized = apply_model(
            model,
            normalized_mix,
            shifts=0,
            split=True,
            overlap=CANONICAL_OVERLAP,
            transition_power=CANONICAL_TRANSITION_POWER,
            device="cpu",
            num_workers=0,
            segment=Fraction(window_samples, model.samplerate),
        )
    official_seconds = time.perf_counter() - official_started

    def torch_runner(padded: Any, offset: int) -> tuple[Any, dict[str, Any]]:
        del offset
        with torch.inference_mode():
            spectrum = spec_to_channels(torch, model._spec(padded))
            frequency, waveform = core(padded, spectrum)
            frequency_waveform, combined = reconstruct_branches(
                torch,
                model,
                frequency,
                waveform,
                window_samples,
            )
        return combined, {
            "frequencyOutput": as_numpy(frequency),
            "waveformOutput": as_numpy(waveform),
            "frequencyWaveformOutput": as_numpy(frequency_waveform),
            "combinedOutput": as_numpy(combined),
        }

    def lite_runner(padded: Any, offset: int) -> tuple[Any, dict[str, Any]]:
        del offset
        with torch.inference_mode():
            spectrum = spec_to_channels(torch, model._spec(padded))
        frequency_np, waveform_np = run_litert(as_numpy(padded), as_numpy(spectrum))
        with torch.inference_mode():
            frequency_waveform, combined = reconstruct_branches(
                torch,
                model,
                torch.from_numpy(frequency_np),
                torch.from_numpy(waveform_np),
                window_samples,
            )
        return combined, {
            "frequencyOutput": frequency_np,
            "waveformOutput": waveform_np,
            "frequencyWaveformOutput": as_numpy(frequency_waveform),
            "combinedOutput": as_numpy(combined),
        }

    torch_started = time.perf_counter()
    torch_ola_normalized, torch_plans, torch_details = run_split_ola(
        torch,
        normalized_mix,
        len(SOURCE_ORDER),
        window_samples,
        CANONICAL_OVERLAP,
        torch_runner,
    )
    torch_seconds = time.perf_counter() - torch_started
    if torch_plans != padding_plans:
        raise RuntimeError("Torch OLA plan changed between padding and inference")

    litert_started = time.perf_counter()
    litert_ola_normalized, litert_plans, litert_details = run_split_ola(
        torch,
        normalized_mix,
        len(SOURCE_ORDER),
        window_samples,
        CANONICAL_OVERLAP,
        lite_runner,
    )
    litert_seconds = time.perf_counter() - litert_started
    if litert_plans != padding_plans:
        raise RuntimeError("LiteRT OLA plan changed between padding and inference")

    window_metrics = []
    window_metrics_pass = True
    uniform_tensor_gate_pass = True
    for plan, torch_detail, litert_detail in zip(
        padding_plans,
        torch_details,
        litert_details,
        strict=True,
    ):
        layer_metrics = {
            name: metric_bundle(torch_detail[name], litert_detail[name])
            for name in (
                "frequencyOutput",
                "waveformOutput",
                "frequencyWaveformOutput",
                "combinedOutput",
            )
        }
        layer_evaluations = {
            name: evaluate_metric(
                bundle["aggregate"],
                require_absolute_gate=name != "frequencyOutput",
                allow_low_signal_gate=False,
            )
            for name, bundle in layer_metrics.items()
        }
        layer_pass = all(value["accepted"] for value in layer_evaluations.values())
        uniform_tensor_gate_pass = bool(
            uniform_tensor_gate_pass
            and all(metric_bundle_passes(value) for value in layer_metrics.values())
        )
        window_metrics_pass = window_metrics_pass and layer_pass
        window_metrics.append(
            {
                "offset": plan["offset"],
                "actualSamples": plan["actualSamples"],
                "accepted": layer_pass,
                "gateEvaluation": layer_evaluations,
                "liteRtVsTorch": layer_metrics,
            }
        )

    torch_oracle = metric_bundle(
        as_numpy(official_normalized),
        as_numpy(torch_ola_normalized),
    )
    torch_oracle_bitwise = bool(
        torch_oracle["aggregate"]["bitwiseEqual"]
        and all(value["bitwiseEqual"] for value in torch_oracle["perStem"].values())
    )

    torch_final = torch_ola_normalized * global_std + global_mean
    litert_final = litert_ola_normalized * global_std + global_mean
    torch_np = as_numpy(torch_final)
    litert_np = as_numpy(litert_final)
    full_metrics = metric_bundle(torch_np, litert_np)
    overlap_range = (stride_samples, window_samples)
    eof_range = (
        CANONICAL_OLA_TRACK_SAMPLES - overlap_samples,
        CANONICAL_OLA_TRACK_SAMPLES,
    )
    overlap_metrics = metric_bundle(
        torch_np[..., overlap_range[0]:overlap_range[1]],
        litert_np[..., overlap_range[0]:overlap_range[1]],
    )
    eof_metrics = metric_bundle(
        torch_np[..., eof_range[0]:eof_range[1]],
        litert_np[..., eof_range[0]:eof_range[1]],
    )
    full_evaluation = evaluate_metric_bundle(
        full_metrics,
        require_absolute_gate=True,
        allow_low_signal_gate=True,
    )
    overlap_evaluation = evaluate_metric_bundle(
        overlap_metrics,
        require_absolute_gate=True,
        allow_low_signal_gate=True,
    )
    eof_evaluation = evaluate_metric_bundle(
        eof_metrics,
        require_absolute_gate=True,
        allow_low_signal_gate=True,
    )
    uniform_tensor_gate_pass = bool(
        uniform_tensor_gate_pass
        and metric_bundle_passes(full_metrics)
        and metric_bundle_passes(overlap_metrics)
        and metric_bundle_passes(eof_metrics)
    )
    accepted = bool(
        torch_oracle_bitwise
        and window_metrics_pass
        and full_evaluation["accepted"]
        and overlap_evaluation["accepted"]
        and eof_evaluation["accepted"]
    )
    result = {
        "status": "passed" if accepted else "failed",
        "acceptedForDeviceTesting": accepted,
        "uniformTensorGatePassed": uniform_tensor_gate_pass,
        "qualityGatePolicy": {
            "minimumSignalToNoiseDb": MINIMUM_SNR_DB,
            "maximumWaveformAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
            "lowSignalReferenceRms": LOW_SIGNAL_REFERENCE_RMS,
            "lowSignalMaximumAbsoluteError": LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR,
            "lowSignalLatentMaximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
            "frequencyLatentAbsoluteError": "reported-not-gated",
            "windowPerStemMetrics": "reported-not-gated-before-ola",
        },
        "profile": {
            "sampleRate": model.samplerate,
            "windowSamples": window_samples,
            "strideSamples": stride_samples,
            "overlapSamples": overlap_samples,
            "overlap": CANONICAL_OVERLAP,
            "transitionPower": CANONICAL_TRANSITION_POWER,
            "shifts": 0,
            "trackSamples": CANONICAL_OLA_TRACK_SAMPLES,
            "accumulationDtype": "float32",
            "windowOrder": "ascending-offset",
            "tailWeightRule": "triangle-prefix",
        },
        "globalNormalization": {
            "reference": "mean-across-stereo-channels",
            "standardDeviationCorrection": 1,
            "epsilon": 1e-8,
            "mean": float(global_mean),
            "standardDeviation": float(global_std),
        },
        "windowPlans": padding_plans,
        "paddingVsOfficialTensorChunkBitwiseEqual": True,
        "torchCoreOlaVsOfficialApplyModel": torch_oracle,
        "windowParity": window_metrics,
        "liteRtVsTorchOla": {
            "full": {
                "metrics": full_metrics,
                "gateEvaluation": full_evaluation,
            },
            "overlap": {
                "startSample": overlap_range[0],
                "endSample": overlap_range[1],
                "metrics": overlap_metrics,
                "gateEvaluation": overlap_evaluation,
            },
            "eof": {
                "startSample": eof_range[0],
                "endSample": eof_range[1],
                "metrics": eof_metrics,
                "gateEvaluation": eof_evaluation,
            },
        },
        "fixtures": {
            "seed": seed + 1,
            "mixInput": write_raw(fixtures / "ola_mix_input.f32le.raw", as_numpy(mix)),
            "combinedGolden": write_raw(
                fixtures / "ola_combined_golden.f32le.raw",
                torch_np,
            ),
        },
        "seconds": {
            "officialApplyModel": official_seconds,
            "torchCoreOla": torch_seconds,
            "liteRtCoreOla": litert_seconds,
        },
    }
    return result


def main() -> int:
    args = parse_args()
    lightweight_conversion = args.profile == "smoke_2s"
    runtime_constant_folding = lightweight_conversion
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "modelId": args.model_id,
        "profileName": args.profile,
        "status": "running",
        "completedStage": "arguments",
        "qualityGate": {
            "minimumSignalToNoiseDb": MINIMUM_SNR_DB,
            "maximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
            "lowSignalReferenceRms": LOW_SIGNAL_REFERENCE_RMS,
            "lowSignalMaximumAbsoluteError": LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR,
            "lowSignalLatentMaximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
            "uniformTensorGatePassed": False,
            "hostPipelineGatePassed": False,
            "acceptedForDeviceTesting": False,
        },
    }

    def checkpoint(stage: str) -> None:
        report["completedStage"] = stage
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    try:
        verify_file(args.weights, EXPECTED_WEIGHT_BYTES, EXPECTED_WEIGHT_SHA256)
        verify_file(args.metadata, EXPECTED_METADATA_BYTES, EXPECTED_METADATA_SHA256)
        bag_manifest = args.weights.parent / "htdemucs_6s.yaml"
        boundary_source = args.demucs_lite_root / "demucs-for-onnx" / "demucs" / "htdemucs.py"
        reference_exporter = args.demucs_lite_root / "scripts" / "convert-pth-to-onnx-chunked.py"
        requirements_lock = Path(__file__).resolve().parent.parent / "requirements-demucs-litert-export.txt"
        verify_file(bag_manifest, EXPECTED_BAG_MANIFEST_BYTES, EXPECTED_BAG_MANIFEST_SHA256)
        verify_file(boundary_source, EXPECTED_BOUNDARY_SOURCE_BYTES, EXPECTED_BOUNDARY_SOURCE_SHA256)
        verify_file(reference_exporter, EXPECTED_REFERENCE_EXPORTER_BYTES, EXPECTED_REFERENCE_EXPORTER_SHA256)
        if not requirements_lock.is_file():
            raise RuntimeError(f"Missing requirements lock: {requirements_lock}")
        loader_checkout = verify_git_revision(
            args.demucs_root,
            EXPECTED_LOADER_REVISION,
            "Demucs loader",
        )
        wrapper_checkout = verify_git_revision(
            args.demucs_lite_root,
            EXPECTED_DEMUCS_LITE_REVISION,
            "demucs-lite reference",
        )
        loader_revision = loader_checkout["revision"]
        wrapper_revision = wrapper_checkout["revision"]

        sys.path.insert(0, str(args.demucs_root))
        import torch
        from demucs.hf import load_safetensors_model

        report["versions"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "litertTorch": package_version("litert-torch"),
            "aiEdgeLiteRt": package_version("ai-edge-litert"),
            "safetensors": package_version("safetensors"),
        }
        report["provenance"] = {
            "canonicalWeight": {
                "fileName": args.weights.name,
                "byteSize": args.weights.stat().st_size,
                "sha256": sha256(args.weights),
                "format": "safetensors",
            },
            "metadata": {
                "fileName": args.metadata.name,
                "byteSize": args.metadata.stat().st_size,
                "sha256": sha256(args.metadata),
            },
            "bagManifest": {
                "fileName": bag_manifest.name,
                "byteSize": bag_manifest.stat().st_size,
                "sha256": sha256(bag_manifest),
                "format": "yaml",
            },
            "boundarySource": {
                "fileName": boundary_source.name,
                "byteSize": boundary_source.stat().st_size,
                "sha256": sha256(boundary_source),
                "format": "python-source",
            },
            "referenceExporter": {
                "fileName": reference_exporter.name,
                "byteSize": reference_exporter.stat().st_size,
                "sha256": sha256(reference_exporter),
                "format": "python-source",
            },
            "loaderRevision": loader_revision,
            "loaderCheckout": loader_checkout,
            "demucsLiteReferenceRevision": wrapper_revision,
            "demucsLiteReferenceCheckout": wrapper_checkout,
            "exportScript": {
                "fileName": Path(__file__).name,
                "sha256": sha256(Path(__file__)),
            },
            "requirementsLock": {
                "fileName": requirements_lock.name,
                "byteSize": requirements_lock.stat().st_size,
                "sha256": sha256(requirements_lock),
                "format": "pip-requirements",
            },
        }
        report["invocation"] = {
            "arguments": sys.argv[1:],
            "workingDirectory": str(Path.cwd()),
        }
        checkpoint("source-verified")

        model = load_safetensors_model(args.weights).cpu().float().eval()
        if tuple(model.sources) != SOURCE_ORDER:
            raise RuntimeError(f"Unexpected source order: {model.sources}")
        if model.samplerate != 44_100:
            raise RuntimeError(f"Unexpected sample rate: {model.samplerate}")
        if args.samples < 44_100:
            raise RuntimeError("Smoke windows shorter than one second are intentionally rejected")
        original_segment = model.segment
        if args.profile == "canonical_7p8s" and original_segment != Fraction(39, 5):
            raise RuntimeError(f"Unexpected canonical segment: {original_segment}")
        model.segment = Fraction(args.samples, model.samplerate)
        model.use_train_segment = True
        install_deterministic_pos_embedding(model)
        core = build_core(torch, model).cpu().eval()
        checkpoint("model-loaded")

        mix = deterministic_mix(torch, args.samples, model.samplerate, args.seed)
        with torch.inference_mode():
            torch_started = time.perf_counter()
            spectrum_complex = model._spec(mix)
            spectrum = spec_to_channels(torch, spectrum_complex)
            reference = model(mix)
            frequency, waveform = core(mix, spectrum)
            frequency_reconstruction, reconstructed = reconstruct_branches(
                torch,
                model,
                frequency,
                waveform,
                args.samples,
            )
            torch_seconds = time.perf_counter() - torch_started

        mix_np = as_numpy(mix)
        spectrum_np = as_numpy(spectrum)
        frequency_np = as_numpy(frequency)
        waveform_np = as_numpy(waveform)
        frequency_reconstruction_np = as_numpy(frequency_reconstruction)
        reference_np = as_numpy(reference)
        reconstructed_np = as_numpy(reconstructed)
        torch_reconstruction = metrics(reference_np, reconstructed_np)
        if not metric_passes(torch_reconstruction):
            raise RuntimeError(f"Torch core reconstruction failed: {torch_reconstruction}")

        expected_shapes = {
            "waveformInput": [1, 2, args.samples],
            "spectrumInput": [1, 4, 2048, math.ceil(args.samples / 1024)],
            "frequencyOutput": [1, 6, 4, 2048, math.ceil(args.samples / 1024)],
            "waveformOutput": [1, 6, 2, args.samples],
            "combinedOutput": [1, 6, 2, args.samples],
        }
        actual_shapes = {
            "waveformInput": list(mix_np.shape),
            "spectrumInput": list(spectrum_np.shape),
            "frequencyOutput": list(frequency_np.shape),
            "waveformOutput": list(waveform_np.shape),
            "combinedOutput": list(reference_np.shape),
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(f"Unexpected ABI: {actual_shapes}")
        report["profile"] = {
            "name": args.profile,
            "sampleRate": model.samplerate,
            "sampleCount": args.samples,
            "segment": {
                "numerator": model.segment.numerator,
                "denominator": model.segment.denominator,
            },
            "originalSegment": str(original_segment),
            "useTrainSegment": model.use_train_segment,
            "sourceOrder": list(SOURCE_ORDER),
            "complexFeatureOrder": ["left.real", "left.imag", "right.real", "right.imag"],
        }
        report["abi"] = actual_shapes
        report["tensorContract"] = {
            name: {
                "dtype": "float32",
                "shape": actual_shapes[name],
                "axes": axes,
            }
            for name, axes in TENSOR_AXES.items()
        }
        report["conversionRecipe"] = {
            "strictExport": True,
            "lightweightConversion": lightweight_conversion,
            "runtimeConstantFolding": runtime_constant_folding,
            "enableX64": False,
            "deterministicPositionalEmbedding": True,
            "liteRtConversionInput": "project-owned PyTorch neural-core module",
            "onnxRole": "diagnostic-only" if args.onnx_output is not None else "not-generated",
        }
        report["seconds"] = {"torchReferenceAndCore": torch_seconds}
        report["torchCoreReconstruction"] = torch_reconstruction
        report["fixtures"] = {
            "seed": args.seed,
            "waveformInput": write_raw(args.fixtures / "waveform_input.f32le.raw", mix_np),
            "spectrumInput": write_raw(args.fixtures / "spectrum_input.f32le.raw", spectrum_np),
            "frequencyGolden": write_raw(args.fixtures / "frequency_golden.f32le.raw", frequency_np),
            "waveformGolden": write_raw(args.fixtures / "waveform_golden.f32le.raw", waveform_np),
            "frequencyWaveformGolden": write_raw(
                args.fixtures / "frequency_waveform_golden.f32le.raw",
                frequency_reconstruction_np,
            ),
            "combinedGolden": write_raw(args.fixtures / "combined_golden.f32le.raw", reference_np),
        }
        checkpoint("torch-parity-passed")

        if args.onnx_output is not None:
            import onnx
            import onnxruntime as ort

            args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
            onnx_started = time.perf_counter()
            torch.onnx.export(
                core,
                (mix, spectrum),
                str(args.onnx_output),
                input_names=["mix", "spectrum_channels"],
                output_names=["frequency_channels", "time_waveform"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
                external_data=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
            report["seconds"]["onnxExport"] = time.perf_counter() - onnx_started
            onnx.checker.check_model(str(args.onnx_output))
            onnx_model = onnx.load(str(args.onnx_output), load_external_data=False)
            operator_histogram: dict[str, int] = {}
            for node in onnx_model.graph.node:
                name = f"{node.domain or 'ai.onnx'}::{node.op_type}"
                operator_histogram[name] = operator_histogram.get(name, 0) + 1
            external_data_count = sum(
                bool(initializer.external_data)
                for initializer in onnx_model.graph.initializer
            )

            ort_started = time.perf_counter()
            session = ort.InferenceSession(
                str(args.onnx_output), providers=["CPUExecutionProvider"]
            )
            onnx_frequency, onnx_waveform = (
                as_numpy(value)
                for value in session.run(
                    ["frequency_channels", "time_waveform"],
                    {"mix": mix_np, "spectrum_channels": spectrum_np},
                )
            )
            report["seconds"]["onnxRuntimeCore"] = time.perf_counter() - ort_started
            with torch.inference_mode():
                onnx_frequency_waveform_tensor, onnx_combined_tensor = reconstruct_branches(
                    torch,
                    model,
                    torch.from_numpy(onnx_frequency),
                    torch.from_numpy(onnx_waveform),
                    args.samples,
                )
                onnx_frequency_waveform = as_numpy(onnx_frequency_waveform_tensor)
                onnx_combined = as_numpy(onnx_combined_tensor)
            onnx_per_stem_frequency_waveform = {
                name: metrics(
                    frequency_reconstruction_np[:, index],
                    onnx_frequency_waveform[:, index],
                )
                for index, name in enumerate(SOURCE_ORDER)
            }
            onnx_per_stem_combined = {
                name: metrics(reference_np[:, index], onnx_combined[:, index])
                for index, name in enumerate(SOURCE_ORDER)
            }
            onnx_metrics = {
                "frequencyOutput": metrics(frequency_np, onnx_frequency),
                "waveformOutput": metrics(waveform_np, onnx_waveform),
                "frequencyWaveformOutput": metrics(
                    frequency_reconstruction_np,
                    onnx_frequency_waveform,
                ),
                "combinedOutput": metrics(reference_np, onnx_combined),
                "perStemFrequencyWaveform": onnx_per_stem_frequency_waveform,
                "perStemCombined": onnx_per_stem_combined,
            }
            onnx_gated = [
                onnx_metrics["frequencyOutput"],
                onnx_metrics["waveformOutput"],
                onnx_metrics["frequencyWaveformOutput"],
                onnx_metrics["combinedOutput"],
                *onnx_per_stem_frequency_waveform.values(),
                *onnx_per_stem_combined.values(),
            ]
            if not all(metric_passes(value) for value in onnx_gated):
                raise RuntimeError(f"ONNX output did not pass the host quality gate: {onnx_metrics}")
            report["onnxArtifact"] = {
                "fileName": args.onnx_output.name,
                "byteSize": args.onnx_output.stat().st_size,
                "sha256": sha256(args.onnx_output),
                "opset": 17,
                "producerName": onnx_model.producer_name,
                "producerVersion": onnx_model.producer_version,
                "externalDataTensorCount": external_data_count,
                "operatorCount": len(onnx_model.graph.node),
                "operatorHistogram": dict(sorted(operator_histogram.items())),
            }
            report["onnxVsTorch"] = onnx_metrics
            checkpoint("onnx-parity-passed")

        if args.skip_litert:
            report["status"] = "torch-only-passed"
            checkpoint("complete")
            return 0

        if not args.reuse_litert:
            import litert_torch

            if args.output.exists():
                args.output.unlink()
            conversion_started = time.perf_counter()
            lite_model = litert_torch.convert(
                core,
                (mix, spectrum),
                strict_export=True,
                lightweight_conversion=lightweight_conversion,
                enable_x64=False,
                runtime_constant_folding=runtime_constant_folding,
            )
            lite_model.export(str(args.output))
            report["seconds"]["liteRtConversion"] = time.perf_counter() - conversion_started
        elif not args.output.is_file():
            raise RuntimeError("--reuse-litert requires an existing --output")
        checkpoint("litert-exported")

        run_litert, inspection = interpreter_runner(args.output)
        lite_started = time.perf_counter()
        lite_frequency, lite_waveform = run_litert(mix_np, spectrum_np)
        report["seconds"]["liteRtCore"] = time.perf_counter() - lite_started
        with torch.inference_mode():
            lite_frequency_waveform_tensor, lite_combined_tensor = reconstruct_branches(
                torch,
                model,
                torch.from_numpy(lite_frequency),
                torch.from_numpy(lite_waveform),
                args.samples,
            )
            lite_frequency_waveform = as_numpy(lite_frequency_waveform_tensor)
            lite_combined = as_numpy(lite_combined_tensor)

        raw_frequency = metrics(frequency_np, lite_frequency)
        raw_waveform = metrics(waveform_np, lite_waveform)
        frequency_waveform_metric = metrics(
            frequency_reconstruction_np,
            lite_frequency_waveform,
        )
        combined = metrics(reference_np, lite_combined)
        per_stem_frequency = {
            name: metrics(frequency_np[:, index], lite_frequency[:, index])
            for index, name in enumerate(SOURCE_ORDER)
        }
        per_stem_waveform = {
            name: metrics(waveform_np[:, index], lite_waveform[:, index])
            for index, name in enumerate(SOURCE_ORDER)
        }
        per_stem_frequency_waveform = {
            name: metrics(
                frequency_reconstruction_np[:, index],
                lite_frequency_waveform[:, index],
            )
            for index, name in enumerate(SOURCE_ORDER)
        }
        per_stem_combined = {
            name: metrics(reference_np[:, index], lite_combined[:, index])
            for index, name in enumerate(SOURCE_ORDER)
        }
        gated_metrics = [
            raw_frequency,
            raw_waveform,
            frequency_waveform_metric,
            combined,
            *per_stem_frequency.values(),
            *per_stem_waveform.values(),
            *per_stem_frequency_waveform.values(),
            *per_stem_combined.values(),
        ]
        uniform_tensor_gate_pass = all(metric_passes(value) for value in gated_metrics)
        single_window_bundles = {
            "frequencyOutput": {
                "aggregate": raw_frequency,
                "perStem": per_stem_frequency,
            },
            "waveformOutput": {
                "aggregate": raw_waveform,
                "perStem": per_stem_waveform,
            },
            "frequencyWaveformOutput": {
                "aggregate": frequency_waveform_metric,
                "perStem": per_stem_frequency_waveform,
            },
            "combinedOutput": {
                "aggregate": combined,
                "perStem": per_stem_combined,
            },
        }
        single_window_evaluation = {
            name: evaluate_metric_bundle(
                bundle,
                require_absolute_gate=name != "frequencyOutput",
                allow_low_signal_gate=True,
                low_signal_maximum_absolute_error=(
                    MAXIMUM_ABSOLUTE_ERROR
                    if name == "frequencyOutput"
                    else LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR
                ),
            )
            for name, bundle in single_window_bundles.items()
        }
        host_pipeline_gate_pass = all(
            value["accepted"] for value in single_window_evaluation.values()
        )
        accepted = (
            uniform_tensor_gate_pass
            if args.profile == "smoke_2s"
            else host_pipeline_gate_pass
        )
        report["liteRtArtifact"] = {
            "fileName": args.output.name,
            "byteSize": args.output.stat().st_size,
            "sha256": sha256(args.output),
            "flatBufferIdentifier": args.output.read_bytes()[4:8].decode("ascii", errors="replace"),
            "inspection": inspection,
        }
        report["liteRtVsTorch"] = {
            "frequencyOutput": raw_frequency,
            "waveformOutput": raw_waveform,
            "frequencyWaveformOutput": frequency_waveform_metric,
            "combinedOutput": combined,
            "perStemFrequencyOutput": per_stem_frequency,
            "perStemWaveformOutput": per_stem_waveform,
            "perStemFrequencyWaveform": per_stem_frequency_waveform,
            "perStemCombined": per_stem_combined,
        }
        report["singleWindowGateEvaluation"] = single_window_evaluation
        if args.profile == "canonical_7p8s":
            report["canonicalOlaValidation"] = validate_canonical_ola(
                torch,
                model,
                core,
                run_litert,
                args.fixtures,
                args.seed,
            )
            report["seconds"]["canonicalOla"] = report["canonicalOlaValidation"]["seconds"]
            accepted = bool(
                accepted
                and report["canonicalOlaValidation"]["acceptedForDeviceTesting"]
            )
            uniform_tensor_gate_pass = bool(
                uniform_tensor_gate_pass
                and report["canonicalOlaValidation"]["uniformTensorGatePassed"]
            )
            host_pipeline_gate_pass = accepted
        report["qualityGate"]["uniformTensorGatePassed"] = uniform_tensor_gate_pass
        report["qualityGate"]["hostPipelineGatePassed"] = host_pipeline_gate_pass
        report["qualityGate"]["acceptedForDeviceTesting"] = accepted
        if not accepted:
            raise RuntimeError("LiteRT output did not pass the host quality gate")
        report["status"] = "passed"
        checkpoint("complete")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "failed", "stage": report["completedStage"], "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
