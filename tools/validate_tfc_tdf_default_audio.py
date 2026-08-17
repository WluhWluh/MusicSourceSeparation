#!/usr/bin/env python3
"""Compare PyTorch, ONNX, and TFLite TFC-TDF audio end to end."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly

from tfc_tdf_default_model import (
    DEFAULT_CONFIG,
    TfcTdfNchwWrapper,
    load_default_checkpoint,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "tfc-tdf" / "default-compact"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "tfc-tdf-default-validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_MODEL_DIR / "tfc_tdf_default_vocals_core_fp32.onnx",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=DEFAULT_MODEL_DIR / "tfc_tdf_default_vocals_core_fp32.tflite",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--short-start-seconds", type=float, default=0.0)
    parser.add_argument("--short-seconds", type=float, default=30.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def periodic_hann(size: int) -> np.ndarray:
    return np.hanning(size + 1)[:-1].astype(np.float32)


def stft_centered(wave: np.ndarray) -> np.ndarray:
    expected = (DEFAULT_CONFIG.model_input_samples, 2)
    if wave.shape != expected:
        raise ValueError(f"Expected waveform {expected}, got {wave.shape}")
    channels = wave.T
    pad = DEFAULT_CONFIG.n_fft // 2
    padded = np.pad(channels, ((0, 0), (pad, pad)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(
        padded,
        DEFAULT_CONFIG.n_fft,
        axis=1,
    )[:, :: DEFAULT_CONFIG.hop_length, :]
    if frames.shape[1] != DEFAULT_CONFIG.num_frames:
        raise ValueError(f"Unexpected STFT frame count: {frames.shape}")
    windowed = frames * periodic_hann(DEFAULT_CONFIG.n_fft)[None, None, :]
    spectrum = np.fft.rfft(windowed, n=DEFAULT_CONFIG.n_fft, axis=-1)
    spectrum = spectrum.astype(np.complex64).transpose(0, 2, 1)
    packed = np.stack(
        [
            spectrum[0].real,
            spectrum[1].real,
            spectrum[0].imag,
            spectrum[1].imag,
        ],
        axis=0,
    )
    return packed[None].astype(np.float32)


def istft_centered(packed: np.ndarray) -> np.ndarray:
    expected = (1, 4, DEFAULT_CONFIG.frequency_bins, DEFAULT_CONFIG.num_frames)
    if packed.shape != expected:
        raise ValueError(f"Expected spectrum {expected}, got {packed.shape}")
    left = packed[0, 0] + 1j * packed[0, 2]
    right = packed[0, 1] + 1j * packed[0, 3]
    spectrum = np.stack([left, right], axis=0).transpose(0, 2, 1)
    frames = np.fft.irfft(spectrum, n=DEFAULT_CONFIG.n_fft, axis=-1)
    frames = frames.real.astype(np.float32)

    window = periodic_hann(DEFAULT_CONFIG.n_fft)
    padded_samples = (
        DEFAULT_CONFIG.n_fft
        + DEFAULT_CONFIG.hop_length * (DEFAULT_CONFIG.num_frames - 1)
    )
    output = np.zeros((2, padded_samples), dtype=np.float32)
    window_sum = np.zeros(padded_samples, dtype=np.float32)
    for frame_index in range(DEFAULT_CONFIG.num_frames):
        start = frame_index * DEFAULT_CONFIG.hop_length
        output[:, start : start + DEFAULT_CONFIG.n_fft] += (
            frames[:, frame_index, :] * window
        )
        window_sum[start : start + DEFAULT_CONFIG.n_fft] += window * window
    nonzero = window_sum > 1e-8
    output[:, nonzero] /= window_sum[nonzero]
    pad = DEFAULT_CONFIG.n_fft // 2
    output = output[:, pad : pad + DEFAULT_CONFIG.model_input_samples]
    return output.T


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.maximum_absolute_error = 0.0
        self.absolute_error_sum = 0.0
        self.error_power = 0.0
        self.reference_power = 0.0
        self.candidate_power = 0.0
        self.dot_product = 0.0

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"Metric shape mismatch: {reference.shape} != {candidate.shape}")
        if not np.isfinite(candidate).all():
            raise ValueError("Candidate contains non-finite values")
        reference64 = reference.astype(np.float64)
        candidate64 = candidate.astype(np.float64)
        error = candidate64 - reference64
        self.count += reference.size
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(np.max(np.abs(error))),
        )
        self.absolute_error_sum += float(np.sum(np.abs(error)))
        self.error_power += float(np.sum(error * error))
        self.reference_power += float(np.sum(reference64 * reference64))
        self.candidate_power += float(np.sum(candidate64 * candidate64))
        self.dot_product += float(np.sum(reference64 * candidate64))

    def result(self) -> dict[str, float]:
        snr = float("inf") if self.error_power == 0.0 else float(
            10.0 * np.log10(self.reference_power / self.error_power)
        )
        cosine = float(
            self.dot_product
            / math.sqrt(self.reference_power * self.candidate_power)
        )
        return {
            "maxAbsError": self.maximum_absolute_error,
            "meanAbsError": self.absolute_error_sum / self.count,
            "rmse": math.sqrt(self.error_power / self.count),
            "signalRms": math.sqrt(self.reference_power / self.count),
            "snrDb": snr,
            "cosineSimilarity": cosine,
        }


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    accumulator = MetricAccumulator()
    accumulator.update(reference, candidate)
    return accumulator.result()


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, repeats=2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if sample_rate != DEFAULT_CONFIG.sample_rate:
        divisor = math.gcd(sample_rate, DEFAULT_CONFIG.sample_rate)
        audio = resample_poly(
            audio,
            DEFAULT_CONFIG.sample_rate // divisor,
            sample_rate // divisor,
            axis=0,
        ).astype(np.float32)
        sample_rate = DEFAULT_CONFIG.sample_rate
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def sha256_array(value: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(value.astype("<f4", copy=False))
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def dsp_oracle(first_window: np.ndarray) -> dict[str, Any]:
    numpy_spectrum = stft_centered(first_window)
    wave_tensor = torch.from_numpy(first_window.T)
    window = torch.hann_window(DEFAULT_CONFIG.n_fft, periodic=True)
    torch_channels = [
        torch.stft(
            wave_tensor[channel],
            n_fft=DEFAULT_CONFIG.n_fft,
            hop_length=DEFAULT_CONFIG.hop_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        for channel in range(2)
    ]
    torch_spectrum = torch.stack(
        [
            torch_channels[0].real,
            torch_channels[1].real,
            torch_channels[0].imag,
            torch_channels[1].imag,
        ],
        dim=0,
    )[None].numpy()

    numpy_wave = istft_centered(numpy_spectrum)
    reconstructed_channels = []
    for channel in range(2):
        complex_spectrum = torch.complex(
            torch.from_numpy(numpy_spectrum[0, channel]),
            torch.from_numpy(numpy_spectrum[0, channel + 2]),
        )
        reconstructed_channels.append(
            torch.istft(
                complex_spectrum,
                n_fft=DEFAULT_CONFIG.n_fft,
                hop_length=DEFAULT_CONFIG.hop_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=DEFAULT_CONFIG.model_input_samples,
                return_complex=False,
            )
        )
    torch_wave = torch.stack(reconstructed_channels, dim=1).numpy()
    result = {
        "featureOrder": [
            "left.real",
            "right.real",
            "left.imag",
            "right.imag",
        ],
        "numpyVsTorchStft": metrics(torch_spectrum, numpy_spectrum),
        "numpyVsTorchIstft": metrics(torch_wave, numpy_wave),
        "numpyRoundTrip": metrics(first_window, numpy_wave),
        "torchRoundTrip": metrics(first_window, torch_wave),
    }
    if result["numpyVsTorchStft"]["snrDb"] < 90.0:
        raise ValueError(f"STFT oracle failed: {result}")
    if result["numpyVsTorchIstft"]["snrDb"] < 90.0:
        raise ValueError(f"iSTFT oracle failed: {result}")
    return result


def load_interpreter(path: Path, threads: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
    return Interpreter(model_path=str(path), num_threads=threads)


def create_backends(
    checkpoint: Path,
    onnx_path: Path,
    tflite_path: Path,
    threads: int,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    torch_model, _ = load_default_checkpoint(checkpoint)
    torch_wrapper = TfcTdfNchwWrapper(torch_model).eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    onnx_session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    onnx_input = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.get_outputs()[0].name

    interpreter = load_interpreter(tflite_path, threads)
    interpreter.allocate_tensors()
    tflite_input = interpreter.get_input_details()[0]
    tflite_output = interpreter.get_output_details()[0]
    if tuple(int(value) for value in tflite_input["shape"]) != (1, 1025, 128, 4):
        raise ValueError(f"Unexpected TFLite input: {tflite_input}")
    if tuple(int(value) for value in tflite_output["shape"]) != (1, 1025, 128, 4):
        raise ValueError(f"Unexpected TFLite output: {tflite_output}")

    def run_pytorch(value: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return torch_wrapper(torch.from_numpy(value)).numpy()

    def run_onnx(value: np.ndarray) -> np.ndarray:
        return onnx_session.run([onnx_output], {onnx_input: value})[0]

    def run_tflite(value: np.ndarray) -> np.ndarray:
        nhwc = np.ascontiguousarray(np.transpose(value, (0, 2, 3, 1)))
        interpreter.set_tensor(int(tflite_input["index"]), nhwc)
        interpreter.invoke()
        output_nhwc = interpreter.get_tensor(int(tflite_output["index"]))
        return np.ascontiguousarray(np.transpose(output_nhwc, (0, 3, 1, 2)))

    return {
        "pytorch": run_pytorch,
        "onnx": run_onnx,
        "tflite": run_tflite,
    }


def write_stem(path: Path, audio: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        DEFAULT_CONFIG.sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": "FLAC/PCM_16",
        "rawFloat32Sha256": sha256_array(audio),
    }


def validate_fixture(
    fixture_name: str,
    audio: np.ndarray,
    output_dir: Path,
    backends: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> dict[str, Any]:
    useful_samples = DEFAULT_CONFIG.useful_samples
    window_count = math.ceil(audio.shape[0] / useful_samples)
    output_parts: dict[str, list[np.ndarray]] = {name: [] for name in backends}
    backend_seconds = {name: 0.0 for name in backends}
    neural_pairs = {
        "pytorchVsOnnx": MetricAccumulator(),
        "pytorchVsTflite": MetricAccumulator(),
        "onnxVsTflite": MetricAccumulator(),
    }
    dsp_seconds = 0.0

    for window_index in range(window_count):
        start = window_index * useful_samples
        segment = audio[start : start + useful_samples]
        window = np.zeros((DEFAULT_CONFIG.model_input_samples, 2), dtype=np.float32)
        window[
            DEFAULT_CONFIG.trim_samples : DEFAULT_CONFIG.trim_samples + len(segment)
        ] = segment
        dsp_started = time.perf_counter()
        spectrum = stft_centered(window)
        dsp_seconds += time.perf_counter() - dsp_started

        predictions: dict[str, np.ndarray] = {}
        for backend_name, backend in backends.items():
            started = time.perf_counter()
            prediction = backend(spectrum)
            backend_seconds[backend_name] += time.perf_counter() - started
            predictions[backend_name] = prediction

        neural_pairs["pytorchVsOnnx"].update(
            predictions["pytorch"], predictions["onnx"]
        )
        neural_pairs["pytorchVsTflite"].update(
            predictions["pytorch"], predictions["tflite"]
        )
        neural_pairs["onnxVsTflite"].update(
            predictions["onnx"], predictions["tflite"]
        )

        for backend_name, prediction in predictions.items():
            dsp_started = time.perf_counter()
            reconstructed = istft_centered(prediction)
            dsp_seconds += time.perf_counter() - dsp_started
            output_parts[backend_name].append(
                reconstructed[
                    DEFAULT_CONFIG.trim_samples : -DEFAULT_CONFIG.trim_samples
                ]
            )

    vocals = {
        name: np.concatenate(parts, axis=0)[: audio.shape[0]].astype(np.float32)
        for name, parts in output_parts.items()
    }
    instrumental = {
        name: (audio - stem).astype(np.float32)
        for name, stem in vocals.items()
    }
    comparisons = {
        "vocals": {
            "pytorchVsOnnx": metrics(vocals["pytorch"], vocals["onnx"]),
            "pytorchVsTflite": metrics(vocals["pytorch"], vocals["tflite"]),
            "onnxVsTflite": metrics(vocals["onnx"], vocals["tflite"]),
        },
        "instrumental": {
            "pytorchVsOnnx": metrics(
                instrumental["pytorch"], instrumental["onnx"]
            ),
            "pytorchVsTflite": metrics(
                instrumental["pytorch"], instrumental["tflite"]
            ),
            "onnxVsTflite": metrics(
                instrumental["onnx"], instrumental["tflite"]
            ),
        },
    }
    gate = {"minimumSnrDb": 90.0, "maximumAbsoluteError": 1e-4}
    passed = all(
        comparison["snrDb"] >= gate["minimumSnrDb"]
        and comparison["maxAbsError"] <= gate["maximumAbsoluteError"]
        for stem_comparisons in comparisons.values()
        for comparison in stem_comparisons.values()
    )
    if not passed:
        raise ValueError(f"{fixture_name} audio parity failed: {comparisons}")

    artifacts: dict[str, Any] = {}
    fixture_dir = output_dir / fixture_name
    for backend_name in backends:
        artifacts[backend_name] = {
            "vocals": write_stem(
                fixture_dir / f"{backend_name}-vocals.flac",
                vocals[backend_name],
            ),
            "instrumental": write_stem(
                fixture_dir / f"{backend_name}-instrumental.flac",
                instrumental[backend_name],
            ),
            "residualReconstructionMaxAbsError": float(
                np.max(
                    np.abs(
                        vocals[backend_name] + instrumental[backend_name] - audio
                    )
                )
            ),
        }
    return {
        "name": fixture_name,
        "sampleRate": DEFAULT_CONFIG.sample_rate,
        "samples": int(audio.shape[0]),
        "durationSeconds": audio.shape[0] / DEFAULT_CONFIG.sample_rate,
        "pcmFloat32Sha256": sha256_array(audio),
        "windowCount": window_count,
        "windowContract": {
            "inputSamples": DEFAULT_CONFIG.model_input_samples,
            "trimSamplesPerSide": DEFAULT_CONFIG.trim_samples,
            "usefulSamples": useful_samples,
        },
        "neuralTensorComparisons": {
            name: accumulator.result()
            for name, accumulator in neural_pairs.items()
        },
        "audioComparisons": comparisons,
        "gate": gate,
        "passed": passed,
        "timingSeconds": {
            "backendInference": backend_seconds,
            "sharedAndReconstructionDsp": dsp_seconds,
        },
        "artifacts": artifacts,
    }


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = load_audio(args.audio.resolve())
    if sample_rate != DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"Unexpected sample rate after resampling: {sample_rate}")
    first_window = np.zeros((DEFAULT_CONFIG.model_input_samples, 2), dtype=np.float32)
    first_segment = audio[: DEFAULT_CONFIG.useful_samples]
    first_window[
        DEFAULT_CONFIG.trim_samples : DEFAULT_CONFIG.trim_samples + len(first_segment)
    ] = first_segment
    oracle = dsp_oracle(first_window)
    backends = create_backends(
        args.checkpoint.resolve(),
        args.onnx.resolve(),
        args.tflite.resolve(),
        args.threads,
    )

    short_start = int(round(args.short_start_seconds * DEFAULT_CONFIG.sample_rate))
    short_length = int(round(args.short_seconds * DEFAULT_CONFIG.sample_rate))
    short_audio = audio[short_start : short_start + short_length]
    if len(short_audio) != short_length:
        raise ValueError("Short fixture extends beyond the source audio")
    fixtures = [
        validate_fixture("short-30s", short_audio, output_dir, backends),
    ]
    if not args.skip_full:
        fixtures.append(validate_fixture("full-song", audio, output_dir, backends))

    report = {
        "schemaVersion": 1,
        "candidateId": "tfc_tdf_default_vocals_core_fp32@audio-1",
        "status": (
            "short-and-full-audio-parity-passed"
            if not args.skip_full
            else "short-audio-parity-passed"
        ),
        "sourceAudio": {
            "file": args.audio.name,
            "bytes": args.audio.stat().st_size,
            "sha256": sha256_file(args.audio),
            "decodedSamples": int(audio.shape[0]),
            "decodedSampleRate": sample_rate,
            "decodedPcmFloat32Sha256": sha256_array(audio),
        },
        "models": {
            "checkpoint": {
                "bytes": args.checkpoint.stat().st_size,
                "sha256": sha256_file(args.checkpoint),
            },
            "onnx": {
                "file": args.onnx.name,
                "bytes": args.onnx.stat().st_size,
                "sha256": sha256_file(args.onnx),
            },
            "tflite": {
                "file": args.tflite.name,
                "bytes": args.tflite.stat().st_size,
                "sha256": sha256_file(args.tflite),
            },
        },
        "dspOracle": oracle,
        "fixtures": fixtures,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "soundfile": package_version("soundfile"),
            "onnxruntime": package_version("onnxruntime"),
            "aiEdgeLiteRt": package_version("ai-edge-litert"),
            "validator": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    report_path = output_dir / "audio-validation-report.json"
    json_write(report_path, report)
    summary = {
        "status": report["status"],
        "dspOracle": oracle,
        "fixtures": [
            {
                "name": fixture["name"],
                "windowCount": fixture["windowCount"],
                "audioComparisons": fixture["audioComparisons"],
                "timingSeconds": fixture["timingSeconds"],
            }
            for fixture in fixtures
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
