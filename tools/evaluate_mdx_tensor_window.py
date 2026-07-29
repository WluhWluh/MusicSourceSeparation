#!/usr/bin/env python3
"""Prepare an MDX benchmark window and evaluate two output tensors as audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from mdx_reference import (
    MdxParams,
    build_windows,
    float_to_pcm16,
    istft_centered,
    load_wav_stereo,
    stft_centered,
)


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_tensor(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"Expected {expected} values for {shape}, got {values.size}: {path}")
    return values.reshape(shape)


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    error = candidate64 - reference64
    rmse = float(np.sqrt(np.mean(error * error)))
    signal_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    denominator = float(np.sqrt(np.sum(reference64 * reference64) * np.sum(candidate64 * candidate64)))
    return {
        "maxAbsError": float(np.max(np.abs(error))),
        "meanAbsError": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "signalRms": signal_rms,
        "snrDb": float(20.0 * np.log10(signal_rms / rmse)) if rmse > 0.0 else None,
        "cosineSimilarity": float(np.sum(reference64 * candidate64) / denominator)
        if denominator > 0.0
        else 1.0,
        "referencePeak": float(np.max(np.abs(reference64))),
        "candidatePeak": float(np.max(np.abs(candidate64))),
    }


def write_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    wavfile.write(path, sample_rate, float_to_pcm16(audio.T))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=existing_file, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--reference-output", type=existing_file)
    parser.add_argument("--candidate-output", type=existing_file)
    parser.add_argument("--model-output-scale", type=float, default=1.0)
    parser.add_argument(
        "--model-output-stem",
        choices=("vocals", "instrumental"),
        default="vocals",
    )
    parser.add_argument("--difference-gain", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--n-fft", type=int, default=6_144)
    parser.add_argument("--hop-length", type=int, default=1_024)
    parser.add_argument("--dim-f", type=int, default=2_048)
    parser.add_argument("--dim-t-power", type=int, default=8)
    parser.add_argument("--trim", type=int, default=3_072)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.reference_output is None) != (args.candidate_output is None):
        raise SystemExit("--reference-output and --candidate-output must be supplied together.")
    if args.window_index < 0:
        raise SystemExit("--window-index cannot be negative.")
    if args.difference_gain <= 0.0:
        raise SystemExit("--difference-gain must be positive.")

    params = MdxParams(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        dim_f=args.dim_f,
        dim_t_power=args.dim_t_power,
        trim=args.trim,
    )
    mix, sample_rate = load_wav_stereo(args.source_wav, params.sample_rate)
    windows, pad = build_windows(mix, params)
    if args.window_index >= windows.shape[0]:
        raise SystemExit(
            f"Window index {args.window_index} is outside the source range 0..{windows.shape[0] - 1}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    waveform_window = windows[args.window_index : args.window_index + 1]
    model_input = stft_centered(waveform_window, params)
    input_path = args.output_dir / "input-nchw-f32.bin"
    model_input.astype("<f4").tofile(input_path)
    stable_mix = waveform_window[0, :, params.trim : -params.trim]
    mixture_path = args.output_dir / "mixture.wav"
    write_wav(mixture_path, sample_rate, stable_mix)

    report: dict[str, object] = {
        "schemaVersion": 1,
        "source": {
            "path": str(args.source_wav),
            "bytes": args.source_wav.stat().st_size,
            "sha256": sha256(args.source_wav),
        },
        "params": asdict(params),
        "windowIndex": args.window_index,
        "sourceWindowCount": int(windows.shape[0]),
        "sourcePadSamples": pad,
        "stableSamples": int(stable_mix.shape[1]),
        "durationSeconds": stable_mix.shape[1] / sample_rate,
        "input": {
            "path": str(input_path),
            "shapeNchw": list(model_input.shape),
            "bytes": input_path.stat().st_size,
            "sha256": sha256(input_path),
        },
        "mixtureWav": str(mixture_path),
    }

    if args.reference_output is not None and args.candidate_output is not None:
        shape = (1, 4, params.dim_f, params.dim_t)
        reference_tensor = read_tensor(args.reference_output, shape)
        candidate_tensor = read_tensor(args.candidate_output, shape)
        reference_model_stem = (
            istft_centered(reference_tensor, params)[0, :, params.trim : -params.trim]
            * args.model_output_scale
        )
        candidate_model_stem = (
            istft_centered(candidate_tensor, params)[0, :, params.trim : -params.trim]
            * args.model_output_scale
        )
        reference_residual = stable_mix - reference_model_stem
        candidate_residual = stable_mix - candidate_model_stem
        if args.model_output_stem == "vocals":
            reference_vocals, candidate_vocals = reference_model_stem, candidate_model_stem
            reference_instrumental, candidate_instrumental = reference_residual, candidate_residual
        else:
            reference_instrumental, candidate_instrumental = reference_model_stem, candidate_model_stem
            reference_vocals, candidate_vocals = reference_residual, candidate_residual

        listening_files = {
            "referenceVocals": args.output_dir / "reference-vocals.wav",
            "candidateVocals": args.output_dir / "candidate-vocals.wav",
            "referenceInstrumental": args.output_dir / "reference-instrumental.wav",
            "candidateInstrumental": args.output_dir / "candidate-instrumental.wav",
            "modelStemDifferenceAmplified": args.output_dir / "model-stem-difference-amplified.wav",
        }
        write_wav(listening_files["referenceVocals"], sample_rate, reference_vocals)
        write_wav(listening_files["candidateVocals"], sample_rate, candidate_vocals)
        write_wav(listening_files["referenceInstrumental"], sample_rate, reference_instrumental)
        write_wav(listening_files["candidateInstrumental"], sample_rate, candidate_instrumental)
        write_wav(
            listening_files["modelStemDifferenceAmplified"],
            sample_rate,
            (candidate_model_stem - reference_model_stem) * args.difference_gain,
        )
        report["evaluation"] = {
            "referenceOutput": {
                "path": str(args.reference_output),
                "bytes": args.reference_output.stat().st_size,
                "sha256": sha256(args.reference_output),
            },
            "candidateOutput": {
                "path": str(args.candidate_output),
                "bytes": args.candidate_output.stat().st_size,
                "sha256": sha256(args.candidate_output),
            },
            "modelOutputScale": args.model_output_scale,
            "modelOutputStem": args.model_output_stem,
            "tensor": comparison(reference_tensor, candidate_tensor),
            "vocalsWaveform": comparison(reference_vocals, candidate_vocals),
            "instrumentalWaveform": comparison(reference_instrumental, candidate_instrumental),
            "differenceGain": args.difference_gain,
            "listeningFiles": {key: str(path) for key, path in listening_files.items()},
        }

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
