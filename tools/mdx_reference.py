#!/usr/bin/env python3
"""Reference utilities for MDX-style ONNX music source separation models.

This script is a desktop reference aid. It is not optimized for Android and it
does not ship model weights. It exists so model tensor contracts and DSP
behavior can be tested before porting pieces into the app.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class MdxParams:
    sample_rate: int = 44_100
    n_fft: int = 6_144
    hop_length: int = 1_024
    dim_f: int = 2_048
    dim_t_power: int = 8
    trim: int = 3_072

    @property
    def dim_t(self) -> int:
        return 2**self.dim_t_power

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def chunk_size(self) -> int:
        return self.hop_length * (self.dim_t - 1)

    @property
    def generation_size(self) -> int:
        return self.chunk_size - 2 * self.trim


def inspect_model(model_path: Path) -> dict:
    model = onnx.load(model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def graph_value_to_dict(value) -> dict:
        tensor_type = value.type.tensor_type
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param or "?")
        return {
            "name": value.name,
            "elem_type": tensor_type.elem_type,
            "shape": dims,
        }

    return {
        "path": str(model_path),
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
            for opset in model.opset_import
        ],
        "graph_inputs": [graph_value_to_dict(value) for value in model.graph.input],
        "graph_outputs": [graph_value_to_dict(value) for value in model.graph.output],
        "ort_inputs": [
            {"name": value.name, "type": value.type, "shape": value.shape}
            for value in session.get_inputs()
        ],
        "ort_outputs": [
            {"name": value.name, "type": value.type, "shape": value.shape}
            for value in session.get_outputs()
        ],
    }


def smoke_run_model(model_path: Path, params: MdxParams) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy = np.zeros((1, 4, params.dim_f, params.dim_t), dtype=np.float32)
    output = session.run([output_name], {input_name: dummy})[0]
    return {
        "input_shape": list(dummy.shape),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_abs_mean": float(np.mean(np.abs(output))),
    }


def periodic_hann(size: int) -> np.ndarray:
    return np.hanning(size + 1)[:-1].astype(np.float32)


def stft_centered(waves: np.ndarray, params: MdxParams) -> np.ndarray:
    """Convert `[batch, 2, samples]` waveform chunks to MDX STFT tensors."""
    if waves.ndim != 3 or waves.shape[1] != 2 or waves.shape[2] != params.chunk_size:
        raise ValueError(f"Expected waveform shape [batch, 2, {params.chunk_size}], got {waves.shape}.")

    batch = waves.shape[0]
    flat = waves.reshape(batch * 2, params.chunk_size)
    pad = params.n_fft // 2
    padded = np.pad(flat, ((0, 0), (pad, pad)), mode="reflect")
    frame_count = 1 + (padded.shape[1] - params.n_fft) // params.hop_length
    window = periodic_hann(params.n_fft)
    spectra = np.empty((batch * 2, params.n_bins, frame_count), dtype=np.complex64)

    for frame_index in range(frame_count):
        start = frame_index * params.hop_length
        frame = padded[:, start : start + params.n_fft] * window
        spectra[:, :, frame_index] = np.fft.rfft(frame, n=params.n_fft, axis=1)

    real_imag = np.stack([spectra.real, spectra.imag], axis=1)
    mdx = real_imag.reshape(batch, 2, 2, params.n_bins, frame_count)
    mdx = mdx.reshape(batch, 4, params.n_bins, frame_count)
    return mdx[:, :, : params.dim_f, :].astype(np.float32)


def istft_centered(spec: np.ndarray, params: MdxParams) -> np.ndarray:
    """Convert MDX output tensors back to `[batch, 2, samples]` waveform chunks."""
    if spec.ndim != 4 or spec.shape[1] != 4 or spec.shape[2] != params.dim_f:
        raise ValueError(f"Expected spectrum shape [batch, 4, {params.dim_f}, frames], got {spec.shape}.")

    batch, _, _, frame_count = spec.shape
    if frame_count != params.dim_t:
        raise ValueError(f"Expected {params.dim_t} frames, got {frame_count}.")

    if params.dim_f > params.n_bins:
        raise ValueError("dim_f cannot exceed the full FFT bin count.")

    full = np.zeros((batch, 4, params.n_bins, frame_count), dtype=np.float32)
    full[:, :, : params.dim_f, :] = spec
    split = full.reshape(batch, 2, 2, params.n_bins, frame_count)
    split = split.reshape(batch * 2, 2, params.n_bins, frame_count)
    spectra = split[:, 0, :, :] + 1j * split[:, 1, :, :]

    window = periodic_hann(params.n_fft)
    padded_length = params.chunk_size + params.n_fft
    output = np.zeros((batch * 2, padded_length), dtype=np.float32)
    window_sum = np.zeros(padded_length, dtype=np.float32)

    for frame_index in range(frame_count):
        start = frame_index * params.hop_length
        frame = np.fft.irfft(spectra[:, :, frame_index], n=params.n_fft, axis=1).real.astype(np.float32)
        output[:, start : start + params.n_fft] += frame * window
        window_sum[start : start + params.n_fft] += window * window

    nonzero = window_sum > 1e-8
    output[:, nonzero] /= window_sum[nonzero]

    pad = params.n_fft // 2
    cropped = output[:, pad : pad + params.chunk_size]
    return cropped.reshape(batch, 2, params.chunk_size)


def load_wav_stereo(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    sample_rate, data = wavfile.read(path)
    audio = np.asarray(data)

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, repeats=2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    audio = pcm_to_float(audio)
    if sample_rate != target_sample_rate:
        gcd = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(audio, target_sample_rate // gcd, sample_rate // gcd, axis=0).astype(np.float32)
        sample_rate = target_sample_rate

    return audio.T.astype(np.float32), sample_rate


def pcm_to_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32).clip(-1.0, 1.0)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if audio.dtype == np.int32:
        return (audio.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    raise ValueError(f"Unsupported WAV dtype: {audio.dtype}")


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    return (audio.clip(-1.0, 1.0) * 32767.0).astype(np.int16)


def build_windows(mix: np.ndarray, params: MdxParams) -> tuple[np.ndarray, int]:
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError(f"Expected stereo mix shape [2, samples], got {mix.shape}.")

    n_sample = mix.shape[1]
    pad = params.generation_size - (n_sample % params.generation_size)
    if pad == params.generation_size:
        pad = 0
    mix_padded = np.concatenate(
        [
            np.zeros((2, params.trim), dtype=np.float32),
            mix,
            np.zeros((2, pad), dtype=np.float32),
            np.zeros((2, params.trim), dtype=np.float32),
        ],
        axis=1,
    )

    windows = []
    for start in range(0, n_sample + pad, params.generation_size):
        windows.append(mix_padded[:, start : start + params.chunk_size])
    return np.stack(windows, axis=0).astype(np.float32), pad


def separate_wav(
    model_path: Path,
    input_path: Path,
    output_dir: Path,
    params: MdxParams,
    denoise: bool,
    limit_seconds: float | None,
) -> dict:
    mix, sample_rate = load_wav_stereo(input_path, params.sample_rate)
    if limit_seconds is not None:
        limit_samples = int(limit_seconds * params.sample_rate)
        mix = mix[:, :limit_samples]

    windows, pad = build_windows(mix, params)
    spec = stft_centered(windows, params)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if denoise:
        pred = -session.run([output_name], {input_name: -spec})[0] * 0.5
        pred += session.run([output_name], {input_name: spec})[0] * 0.5
    else:
        pred = session.run([output_name], {input_name: spec})[0]

    target_windows = istft_centered(pred.astype(np.float32), params)
    target = target_windows[:, :, params.trim : -params.trim]
    target = target.transpose(1, 0, 2).reshape(2, -1)
    if pad:
        target = target[:, :-pad]
    target = target[:, : mix.shape[1]]
    instrumental = (mix[:, : target.shape[1]] - target).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    vocals_path = output_dir / f"{stem}_vocals.wav"
    instrumental_path = output_dir / f"{stem}_instrumental.wav"
    wavfile.write(vocals_path, sample_rate, float_to_pcm16(target.T))
    wavfile.write(instrumental_path, sample_rate, float_to_pcm16(instrumental.T))

    return {
        "input_path": str(input_path),
        "sample_rate": sample_rate,
        "input_samples": int(mix.shape[1]),
        "windows": int(windows.shape[0]),
        "model_input_shape": list(spec.shape),
        "vocals_path": str(vocals_path),
        "instrumental_path": str(instrumental_path),
    }


def self_test(params: MdxParams) -> dict:
    rng = np.random.default_rng(seed=1234)
    full_band_params = replace(params, dim_f=params.n_bins)
    waves = rng.normal(0.0, 0.02, size=(2, 2, params.chunk_size)).astype(np.float32)

    full_spec = stft_centered(waves, full_band_params)
    full_reconstructed = istft_centered(full_spec, full_band_params)
    full_error = waves - full_reconstructed

    model_spec = stft_centered(waves, params)
    model_reconstructed = istft_centered(model_spec, params)
    model_error = waves - model_reconstructed

    return {
        "params": asdict(params),
        "input_shape": list(waves.shape),
        "full_band": {
            "spectrum_shape": list(full_spec.shape),
            "reconstructed_shape": list(full_reconstructed.shape),
            "max_abs_error": float(np.max(np.abs(full_error))),
            "mean_abs_error": float(np.mean(np.abs(full_error))),
        },
        "model_band_truncated": {
            "spectrum_shape": list(model_spec.shape),
            "reconstructed_shape": list(model_reconstructed.shape),
            "max_abs_error": float(np.max(np.abs(model_error))),
            "mean_abs_error": float(np.mean(np.abs(model_error))),
            "note": "This path is expected to be lossy because the model uses 2048 of 3073 rFFT bins.",
        },
    }


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser("inspect", help="Inspect an ONNX model contract.")
    inspect.add_argument("--model", type=existing_path, required=True)
    inspect.add_argument("--smoke-run", action="store_true", help="Run one zero-input inference.")

    self_test_parser = subcommands.add_parser("self-test", help="Check local STFT/ISTFT reconstruction.")
    self_test_parser.set_defaults(command="self-test")

    separate = subcommands.add_parser("separate", help="Run a short desktop reference separation on a WAV file.")
    separate.add_argument("--model", type=existing_path, required=True)
    separate.add_argument("--input", type=existing_path, required=True)
    separate.add_argument("--output-dir", type=Path, default=Path("outputs/reference"))
    separate.add_argument("--no-denoise", dest="denoise", action="store_false", default=True)
    separate.add_argument("--limit-seconds", type=float, default=20.0)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    params = MdxParams()

    if args.command == "inspect":
        result = {"model": inspect_model(args.model), "params": asdict(params)}
        if args.smoke_run:
            result["smoke_run"] = smoke_run_model(args.model, params)
        print_json(result)
        return 0

    if args.command == "self-test":
        print_json(self_test(params))
        return 0

    if args.command == "separate":
        print_json(
            separate_wav(
                model_path=args.model,
                input_path=args.input,
                output_dir=args.output_dir,
                params=params,
                denoise=args.denoise,
                limit_seconds=args.limit_seconds,
            )
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
