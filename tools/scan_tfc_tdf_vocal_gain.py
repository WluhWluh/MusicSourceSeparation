#!/usr/bin/env python3
"""Scan post-separation vocal gains and write unnormalized listening renders.

This is an inference-side diagnostic only.  Given a mixture and an already
rendered vocal estimate V_hat, each candidate is assembled as:

    vocals_g = g * V_hat
    accompaniment_g = mixture - vocals_g

The script deliberately does not normalize either output.  FLAC/PCM_16
serialization clips values outside [-1, 1]; the report records those counts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf

from tfc_tdf_short_window import (
    ShortWindowContract,
    render_with_backend,
    sha256_array,
)


DEFAULT_GAINS = (1.00, 1.05, 1.10, 1.20, 1.30)
DEFAULT_CHUNK_FRAMES = 262_144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    estimate_group = parser.add_mutually_exclusive_group(required=True)
    estimate_group.add_argument("--vocals-estimate", type=Path)
    estimate_group.add_argument("--onnx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAINS),
        help="Multipliers applied to V_hat (default: 1.00 1.05 1.10 1.20 1.30)",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="Frames processed per streaming chunk",
    )
    parser.add_argument(
        "--model-id",
        default="tfc_tdf_default_vocals_core_fp32@audio-1",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expected-audio-pcm-sha256")
    parser.add_argument("--expected-vhat-sha256")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Compute the report without writing listening files",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gain_label(gain: float) -> str:
    """Create a stable Windows-safe directory name for the requested sweep."""
    text = f"{gain:.2f}"
    return f"gain-{text.replace('-', 'm').replace('.', 'p')}x"


def rms_db(power: float, count: int) -> float:
    return 20.0 * math.log10(max(math.sqrt(power / max(count, 1)), 1e-12))


@dataclass
class GainAccumulator:
    gain: float
    sample_count: int = 0
    vocal_power: float = 0.0
    accompaniment_power: float = 0.0
    mixture_power: float = 0.0
    projection_dot: float = 0.0
    estimate_power: float = 0.0
    baseline_delta_power: float = 0.0
    reconstruction_max_abs_error: float = 0.0
    vocal_max_abs: float = 0.0
    accompaniment_max_abs: float = 0.0
    mixture_max_abs: float = 0.0
    vocal_min: float = math.inf
    vocal_max: float = -math.inf
    accompaniment_min: float = math.inf
    accompaniment_max: float = -math.inf
    vocal_clipped_samples: int = 0
    accompaniment_clipped_samples: int = 0
    vocal_hash: Any = field(default_factory=hashlib.sha256)
    accompaniment_hash: Any = field(default_factory=hashlib.sha256)

    def update(
        self,
        mixture: np.ndarray,
        estimate: np.ndarray,
        vocals: np.ndarray,
        accompaniment: np.ndarray,
        baseline_accompaniment: np.ndarray,
    ) -> None:
        mixture64 = mixture.astype(np.float64)
        estimate64 = estimate.astype(np.float64)
        vocals64 = vocals.astype(np.float64)
        accompaniment64 = accompaniment.astype(np.float64)
        baseline64 = baseline_accompaniment.astype(np.float64)
        error = vocals64 + accompaniment64 - mixture64

        self.sample_count += int(vocals.size)
        self.vocal_power += float(np.sum(vocals64 * vocals64))
        self.accompaniment_power += float(np.sum(accompaniment64 * accompaniment64))
        self.mixture_power += float(np.sum(mixture64 * mixture64))
        self.estimate_power += float(np.sum(estimate64 * estimate64))
        self.projection_dot += float(np.sum(accompaniment64 * estimate64))
        self.baseline_delta_power += float(
            np.sum((accompaniment64 - baseline64) ** 2)
        )
        self.reconstruction_max_abs_error = max(
            self.reconstruction_max_abs_error,
            float(np.max(np.abs(error))),
        )

        self.vocal_max_abs = max(self.vocal_max_abs, float(np.max(np.abs(vocals64))))
        self.accompaniment_max_abs = max(
            self.accompaniment_max_abs,
            float(np.max(np.abs(accompaniment64))),
        )
        self.mixture_max_abs = max(self.mixture_max_abs, float(np.max(np.abs(mixture64))))
        self.vocal_min = min(self.vocal_min, float(np.min(vocals64)))
        self.vocal_max = max(self.vocal_max, float(np.max(vocals64)))
        self.accompaniment_min = min(
            self.accompaniment_min,
            float(np.min(accompaniment64)),
        )
        self.accompaniment_max = max(
            self.accompaniment_max,
            float(np.max(accompaniment64)),
        )
        self.vocal_clipped_samples += int(np.count_nonzero(np.abs(vocals64) > 1.0))
        self.accompaniment_clipped_samples += int(
            np.count_nonzero(np.abs(accompaniment64) > 1.0)
        )

        self.vocal_hash.update(
            np.ascontiguousarray(vocals.astype("<f4", copy=False)).tobytes()
        )
        self.accompaniment_hash.update(
            np.ascontiguousarray(accompaniment.astype("<f4", copy=False)).tobytes()
        )

    def result(
        self,
        baseline_estimate_power: float,
        baseline_accompaniment_power: float,
        baseline_projection_db: float,
    ) -> dict[str, object]:
        projection_gain = self.projection_dot / max(self.estimate_power, 1e-12)
        projection_power = (self.projection_dot**2) / max(self.estimate_power, 1e-12)
        projection_db = 10.0 * math.log10(
            max(projection_power, 1e-12) / max(self.accompaniment_power, 1e-12)
        )
        return {
            "gain": self.gain,
            "vocal": {
                "rmsDbfs": rms_db(self.vocal_power, self.sample_count),
                "peak": self.vocal_max_abs,
                "min": self.vocal_min,
                "max": self.vocal_max,
                "pcm16ClipSamples": self.vocal_clipped_samples,
                "pcm16ClipFraction": self.vocal_clipped_samples
                / max(self.sample_count, 1),
                "rawFloat32Sha256": self.vocal_hash.hexdigest(),
            },
            "accompaniment": {
                "rmsDbfs": rms_db(self.accompaniment_power, self.sample_count),
                "peak": self.accompaniment_max_abs,
                "min": self.accompaniment_min,
                "max": self.accompaniment_max,
                "pcm16ClipSamples": self.accompaniment_clipped_samples,
                "pcm16ClipFraction": self.accompaniment_clipped_samples
                / max(self.sample_count, 1),
                "rawFloat32Sha256": self.accompaniment_hash.hexdigest(),
            },
            "diagnostics": {
                "vocalRmsDeltaDbFrom1x": rms_db(
                    self.vocal_power, self.sample_count
                )
                - rms_db(baseline_estimate_power, self.sample_count),
                "accompanimentRmsDeltaDbFrom1x": rms_db(
                    self.accompaniment_power, self.sample_count
                )
                - rms_db(baseline_accompaniment_power, self.sample_count),
                "accompanimentProjectionOntoVhatGain": projection_gain,
                "accompanimentProjectionOntoVhatDb": projection_db,
                "accompanimentProjectionDeltaDbFrom1x": projection_db
                - baseline_projection_db,
                "accompanimentDifferenceFrom1xRmsDbfs": rms_db(
                    self.baseline_delta_power, self.sample_count
                ),
                "mixtureRmsDbfs": rms_db(self.mixture_power, self.sample_count),
                "mixturePeak": self.mixture_max_abs,
            },
            "reconstructionMaxAbsErrorFloat32": self.reconstruction_max_abs_error,
        }


def load_stereo(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"Expected stereo input, got {audio.shape}")
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def make_onnx_backend(path: Path, threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_detail = session.get_inputs()[0]
    output_detail = session.get_outputs()[0]
    expected_shape = [1, 4, 1025, 128]
    if list(input_detail.shape) != expected_shape:
        raise ValueError(f"Unexpected ONNX input shape: {input_detail.shape}")
    if list(output_detail.shape) != expected_shape:
        raise ValueError(f"Unexpected ONNX output shape: {output_detail.shape}")

    def run(value: np.ndarray) -> np.ndarray:
        return session.run([output_detail.name], {input_detail.name: value})[0]

    return run


def load_or_render_estimate(
    args: argparse.Namespace,
    mixture: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if args.vocals_estimate is not None:
        path = args.vocals_estimate.resolve()
        estimate, estimate_rate = load_stereo(path)
        if estimate_rate != sample_rate or estimate.shape != mixture.shape:
            raise ValueError(
                "Vocal estimate does not match mixture: "
                f"rate={estimate_rate}/{sample_rate} shape={estimate.shape}/{mixture.shape}"
            )
        return estimate, {
            "kind": "pre-rendered-audio",
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    if sample_rate != 44_100:
        raise ValueError(f"ONNX render requires 44100 Hz input, got {sample_rate}")
    onnx_path = args.onnx.resolve()
    contract = ShortWindowContract(
        num_frames=128,
        left_context_hops=5,
        right_context_hops=5,
    )
    backend = make_onnx_backend(onnx_path, args.threads)
    started = time.perf_counter()
    rendered = render_with_backend(
        mixture,
        contract,
        backend,
        assembly_mode="isolated",
    )
    return rendered.audio, {
        "kind": "onnxruntime-float32-render",
        "file": str(onnx_path),
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256_file(onnx_path),
        "threads": args.threads,
        "providers": ["CPUExecutionProvider"],
        "windowCount": rendered.window_count,
        "elapsedSeconds": time.perf_counter() - started,
        "contract": contract.as_dict(assembly="isolated-zero-padded"),
    }


def main() -> int:
    args = parse_args()
    if not args.gains or any(
        not math.isfinite(value) or value <= 0.0 for value in args.gains
    ):
        raise ValueError("gains must be finite positive numbers")
    if args.chunk_frames <= 0:
        raise ValueError("chunk-frames must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")

    mixture_path = args.audio.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mixture, sample_rate = load_stereo(mixture_path)
    channels = mixture.shape[1]
    frames = mixture.shape[0]
    mixture_pcm_sha256 = sha256_array(mixture)
    if (
        args.expected_audio_pcm_sha256 is not None
        and mixture_pcm_sha256.lower() != args.expected_audio_pcm_sha256.lower()
    ):
        raise ValueError(
            "Decoded mixture identity mismatch: "
            f"{mixture_pcm_sha256} != {args.expected_audio_pcm_sha256}"
        )
    estimate, estimate_identity = load_or_render_estimate(args, mixture, sample_rate)
    estimate_pcm_sha256 = sha256_array(estimate)
    if (
        args.expected_vhat_sha256 is not None
        and estimate_pcm_sha256.lower() != args.expected_vhat_sha256.lower()
    ):
        raise ValueError(
            "V_hat identity mismatch: "
            f"{estimate_pcm_sha256} != {args.expected_vhat_sha256}"
        )

    # Keep one baseline copy only for the current chunk.  All candidate files
    # are generated from the same input samples and estimate.
    accumulators = [GainAccumulator(float(gain)) for gain in args.gains]
    output_files: list[tuple[object, object] | None] = [None] * len(accumulators)

    if not args.no_audio:
        for index, accumulator in enumerate(accumulators):
            gain_dir = output_dir / gain_label(accumulator.gain)
            gain_dir.mkdir(parents=True, exist_ok=True)
            vocal_path = gain_dir / "vocals.flac"
            accompaniment_path = gain_dir / "accompaniment.flac"
            vocal_file = sf.SoundFile(
                vocal_path,
                mode="w",
                samplerate=sample_rate,
                channels=channels,
                format="FLAC",
                subtype="PCM_16",
            )
            accompaniment_file = sf.SoundFile(
                accompaniment_path,
                mode="w",
                samplerate=sample_rate,
                channels=channels,
                format="FLAC",
                subtype="PCM_16",
            )
            output_files[index] = (vocal_file, accompaniment_file)

    try:
        for start in range(0, frames, args.chunk_frames):
            mixture_chunk = mixture[start : start + args.chunk_frames]
            estimate_chunk = estimate[start : start + args.chunk_frames]
            baseline_accompaniment = mixture_chunk - estimate_chunk
            for index, accumulator in enumerate(accumulators):
                vocals = estimate_chunk * np.float32(accumulator.gain)
                accompaniment = mixture_chunk - vocals
                accumulator.update(
                    mixture_chunk,
                    estimate_chunk,
                    vocals,
                    accompaniment,
                    baseline_accompaniment,
                )
                if output_files[index] is not None:
                    vocal_file, accompaniment_file = output_files[index]
                    vocal_file.write(np.clip(vocals, -1.0, 1.0))
                    accompaniment_file.write(np.clip(accompaniment, -1.0, 1.0))
    finally:
        for pair in output_files:
            if pair is not None:
                pair[0].close()
                pair[1].close()

    baseline = next(
        (item for item in accumulators if math.isclose(item.gain, 1.0)),
        None,
    )
    if baseline is None:
        raise ValueError("The diagnostic requires a 1.0 baseline gain")
    baseline_projection_power = (
        baseline.projection_dot**2 / max(baseline.estimate_power, 1e-12)
    )
    baseline_projection_db = 10.0 * math.log10(
        max(baseline_projection_power, 1e-12)
        / max(baseline.accompaniment_power, 1e-12)
    )
    results: list[dict[str, object]] = []
    for accumulator in accumulators:
        result = accumulator.result(
            baseline.estimate_power,
            baseline.accompaniment_power,
            baseline_projection_db,
        )
        if not args.no_audio:
            gain_dir = output_dir / gain_label(accumulator.gain)
            result["files"] = {
                "vocals": {
                    "file": str(gain_dir / "vocals.flac"),
                    "bytes": (gain_dir / "vocals.flac").stat().st_size,
                    "sha256": sha256_file(gain_dir / "vocals.flac"),
                    "format": "FLAC/PCM_16",
                },
                "accompaniment": {
                    "file": str(gain_dir / "accompaniment.flac"),
                    "bytes": (gain_dir / "accompaniment.flac").stat().st_size,
                    "sha256": sha256_file(gain_dir / "accompaniment.flac"),
                    "format": "FLAC/PCM_16",
                },
            }
        results.append(result)

    report = {
        "schemaVersion": 1,
        "status": "completed",
        "purpose": "diagnostic-vocal-gain-sweep",
        "formula": {
            "vocals": "gain * V_hat",
            "accompaniment": "mixture - gain * V_hat",
            "normalization": "none",
            "pcm16Serialization": "clip to [-1, 1] only at FLAC write",
        },
        "sourceAudio": {
            "file": str(mixture_path),
            "bytes": mixture_path.stat().st_size,
            "sha256": sha256_file(mixture_path),
            "sampleRate": sample_rate,
            "channels": channels,
            "frames": frames,
            "decodedPcmFloat32Sha256": mixture_pcm_sha256,
        },
        "vocalEstimate": {
            "modelId": args.model_id,
            "rawFloat32Sha256": estimate_pcm_sha256,
            "source": estimate_identity,
        },
        "checkpoint": None
        if args.checkpoint is None
        else {
            "file": str(args.checkpoint.resolve()),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint.resolve()),
        },
        "gains": [float(value) for value in args.gains],
        "candidates": results,
        "interpretation": (
            "A lower accompaniment projection onto V_hat can indicate that the "
            "estimate was under-scaled, but it is not a separation-quality or "
            "vocal-leakage measurement without isolated ground-truth stems. "
            "Increasing gain also subtracts any instrumental bleed present in V_hat."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "onnxruntime": package_version("onnxruntime"),
            "soundfile": package_version("soundfile"),
            "tool": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "candidates": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
