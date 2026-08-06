#!/usr/bin/env python3
"""Localize HTDemucs waveform ONNX error across pre-export Torch patches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
from typing import Any, Callable

import numpy as np


SAMPLE_RATE = 44_100
CHANNELS = 2
WINDOW_SAMPLES = 343_980
STEM_ORDER = ("drums", "bass", "other", "vocals", "guitar", "piano")
STEM_COUNT = len(STEM_ORDER)
EXPECTED_DEMUCS_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
EXPECTED_EXPORTER_REVISION = "85db5c80aba33f0f2bdf88034a4be6539feec85b"
EXPECTED_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
EXPECTED_FIXTURE_SHA256 = {
    "waveform_input.f32le.raw": "9515d42e72e96036e34f0193d1302e8cccbf80c4e9e7084d1fcba26e17fea40e",
    "spectrum_input.f32le.raw": "9d01b6cf6943c61724f6bb7edefc8f6156f9749f1ae9bc4f84508a62738ee5e0",
    "frequency_golden.f32le.raw": "86e916da6041bf0038adcd770ed986ddbe54e52e264d934f11d8186b7616dcbb",
    "frequency_waveform_golden.f32le.raw": "5bb82c18f2f778b6cf66c28c663d76921f37c4bb69128351e3ce0b4229498ba8",
    "waveform_golden.f32le.raw": "be0934e5be603d5fd17a69cf92d3786e21053128cdfb4fa0f4f78166aebea474",
    "combined_golden.f32le.raw": "7d5fd3585cbdbb46551e9e05883da9aa880702a72efaa71402072957d037264a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cannot read git revision for {path}: {result.stderr}")
    return result.stdout.strip()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    error = candidate64 - reference64
    signal_rms = float(np.sqrt(np.mean(np.square(reference64))))
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    return {
        "sampleCount": int(reference64.size),
        "finite": bool(np.isfinite(candidate64).all()),
        "bitwiseEqual": bool(np.array_equal(reference, candidate)),
        "signalRootMeanSquare": signal_rms,
        "rootMeanSquareError": error_rms,
        "meanAbsoluteError": float(np.mean(np.abs(error))),
        "maximumAbsoluteError": float(np.max(np.abs(error))),
        "signalToNoiseDb": float(
            20.0 * math.log10(max(signal_rms, 1e-30) / max(error_rms, 1e-30))
        ),
    }


def stem_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    return {
        "aggregate": metric(reference, candidate),
        "perStem": {
            stem: metric(reference[:, index], candidate[:, index])
            for index, stem in enumerate(STEM_ORDER)
        },
    }


def as_numpy(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)


def apply_mha_patch(model: Any, nn: Any, replacement: Callable[..., Any]) -> None:
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.forward = types.MethodType(replacement, module)


def restore_mha(model: Any, nn: Any) -> None:
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.forward = types.MethodType(nn.MultiheadAttention.forward, module)


def restore_position_embedding(model: Any, transformer_module: Any) -> None:
    for module in model.modules():
        if isinstance(module, transformer_module.CrossTransformerEncoder):
            module.sin_random_shift = 0
            module._get_pos_embedding = types.MethodType(
                transformer_module.CrossTransformerEncoder._get_pos_embedding,
                module,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demucs-root", type=Path, required=True)
    parser.add_argument("--exporter-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path, required=True)
    parser.add_argument("--onnx-run-report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.threads > 64:
        raise ValueError("--threads must be in [1,64]")
    demucs_root = args.demucs_root.resolve()
    exporter_root = args.exporter_root.resolve()
    weights = args.weights.resolve()
    fixtures = args.fixtures.resolve()
    onnx_output_path = args.onnx_output.resolve()
    onnx_run_report_path = args.onnx_run_report.resolve()

    demucs_revision = git_revision(demucs_root)
    exporter_revision = git_revision(exporter_root)
    if demucs_revision != EXPECTED_DEMUCS_REVISION:
        raise ValueError(f"Unexpected Demucs revision: {demucs_revision}")
    if exporter_revision != EXPECTED_EXPORTER_REVISION:
        raise ValueError(f"Unexpected exporter revision: {exporter_revision}")
    if sha256_file(weights) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Official safetensors identity mismatch")
    for name, expected_sha in EXPECTED_FIXTURE_SHA256.items():
        actual_sha = sha256_file(fixtures / name)
        if actual_sha != expected_sha:
            raise ValueError(f"Fixture identity mismatch for {name}: {actual_sha}")

    sys.path.insert(0, str(exporter_root / "src"))
    sys.path.insert(0, str(demucs_root))
    import torch
    import torch.nn as nn
    import demucs.transformer as transformer_module
    from demucs.hf import load_safetensors_model
    from demucs_onnx.export.mha import onnx_friendly_mha_forward
    from demucs_onnx.export.patch import patch_htdemucs_for_onnx
    from demucs_onnx.export.pos_embed import disable_random_pos_shift
    from demucs_onnx.export.segment import coerce_segment_to_float

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    def load_model() -> Any:
        model = load_safetensors_model(weights).cpu().float().eval()
        if tuple(model.sources) != STEM_ORDER:
            raise ValueError(f"Unexpected source order: {model.sources}")
        return model

    waveform = np.fromfile(
        fixtures / "waveform_input.f32le.raw", dtype="<f4"
    ).reshape(1, CHANNELS, WINDOW_SAMPLES)
    spectrum = np.fromfile(
        fixtures / "spectrum_input.f32le.raw", dtype="<f4"
    ).reshape(1, CHANNELS * 2, 2048, 336)
    frequency = np.fromfile(
        fixtures / "frequency_golden.f32le.raw", dtype="<f4"
    ).reshape(1, STEM_COUNT, CHANNELS * 2, 2048, 336)
    frequency_waveform = np.fromfile(
        fixtures / "frequency_waveform_golden.f32le.raw", dtype="<f4"
    ).reshape(1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES)
    time_waveform = np.fromfile(
        fixtures / "waveform_golden.f32le.raw", dtype="<f4"
    ).reshape(1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES)
    combined = np.fromfile(
        fixtures / "combined_golden.f32le.raw", dtype="<f4"
    ).reshape(1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES)
    onnx_output = np.fromfile(onnx_output_path, dtype="<f4").reshape(
        1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES
    )
    onnx_run = json.loads(onnx_run_report_path.read_text(encoding="utf-8"))
    if onnx_run["output"]["sha256"] != sha256_file(onnx_output_path):
        raise ValueError("ONNX output identity does not match its isolated-run report")
    if onnx_run["model"]["weightStorage"] != "float32":
        raise ValueError("The localization input must come from the pinned FP32 ONNX")

    variants: dict[str, Callable[[Any], None]] = {
        "original": lambda model: None,
        "segment_float_only": lambda model: coerce_segment_to_float(model),
        "deterministic_position_only": lambda model: disable_random_pos_shift(model),
        "manual_mha_only": lambda model: apply_mha_patch(
            model, nn, onnx_friendly_mha_forward
        ),
    }

    def spectral_only(model: Any) -> None:
        original_segment = model.segment
        patch_htdemucs_for_onnx(model)
        model.segment = original_segment
        restore_position_embedding(model, transformer_module)
        restore_mha(model, nn)

    def spectral_and_mha(model: Any) -> None:
        spectral_only(model)
        apply_mha_patch(model, nn, onnx_friendly_mha_forward)

    variants["real_stft_istft_only"] = spectral_only
    variants["real_stft_istft_plus_manual_mha"] = spectral_and_mha
    variants["all_export_patches"] = lambda model: patch_htdemucs_for_onnx(model)

    variant_reports: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, patch in variants.items():
        model = load_model()
        patch(model)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = as_numpy(model(torch.from_numpy(waveform)))
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        outputs[name] = output
        variant_reports[name] = {
            "wallMs": wall_ms,
            "versusFrozenOriginalTorch": stem_comparison(combined, output),
            "versusFp32Onnx": stem_comparison(onnx_output, output),
        }
        del model
        gc.collect()

    stage_model = load_model()
    with torch.inference_mode():
        original_spectrum_complex = stage_model._spec(torch.from_numpy(waveform))
        original_spectrum_channels = as_numpy(
            stage_model._magnitude(original_spectrum_complex)
        )
    patch_htdemucs_for_onnx(stage_model)
    with torch.inference_mode():
        patched_spectrum_real = stage_model._spec(torch.from_numpy(waveform))
        patched_spectrum_channels = as_numpy(
            stage_model._magnitude(patched_spectrum_real)
        )
        patched_frequency_real = stage_model._mask(
            patched_spectrum_real,
            torch.from_numpy(frequency),
        )
        patched_frequency_waveform = as_numpy(
            stage_model._ispec(patched_frequency_real, length=WINDOW_SAMPLES)
        )
    patched_recombined = patched_frequency_waveform + time_waveform

    stage_reports = {
        "frozenSpectrumVsOriginalTorchSpec": metric(
            spectrum, original_spectrum_channels
        ),
        "realStftVsOriginalTorchSpec": metric(
            original_spectrum_channels, patched_spectrum_channels
        ),
        "realStftVsFrozenSpectrum": metric(spectrum, patched_spectrum_channels),
        "realIstftVsOriginalTorchIstft": stem_comparison(
            frequency_waveform, patched_frequency_waveform
        ),
        "realIstftRecombinedWithFrozenTimeBranchVsCombinedGolden": stem_comparison(
            combined, patched_recombined
        ),
    }

    all_patched = outputs["all_export_patches"]
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "provenance": {
            "demucsRoot": str(demucs_root),
            "demucsRevision": demucs_revision,
            "exporterRoot": str(exporter_root),
            "exporterRevision": exporter_revision,
            "weight": {
                "path": str(weights),
                "sha256": EXPECTED_WEIGHT_SHA256,
            },
            "fp32OnnxOutput": {
                "path": str(onnx_output_path),
                "sha256": sha256_file(onnx_output_path),
                "runReport": str(onnx_run_report_path),
                "model": onnx_run["model"],
            },
        },
        "fixtures": {
            name: {
                "path": str(fixtures / name),
                "sha256": expected_sha,
            }
            for name, expected_sha in EXPECTED_FIXTURE_SHA256.items()
        },
        "variantComparisons": variant_reports,
        "stageComparisons": stage_reports,
        "keyComparisons": {
            "originalTorchVsFrozenGolden": stem_comparison(
                combined, outputs["original"]
            ),
            "allPatchedTorchVsOriginalTorch": stem_comparison(
                outputs["original"], all_patched
            ),
            "fp32OnnxVsAllPatchedTorch": stem_comparison(
                all_patched, onnx_output
            ),
            "fp32OnnxVsOriginalTorch": stem_comparison(
                outputs["original"], onnx_output
            ),
        },
    }
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": "complete",
        "output": str(args.output.resolve()),
        "allPatchedVsOriginalSnrDb": report["keyComparisons"]
        ["allPatchedTorchVsOriginalTorch"]["aggregate"]["signalToNoiseDb"],
        "onnxVsAllPatchedSnrDb": report["keyComparisons"]
        ["fp32OnnxVsAllPatchedTorch"]["aggregate"]["signalToNoiseDb"],
        "realStftSnrDb": stage_reports["realStftVsOriginalTorchSpec"]
        ["signalToNoiseDb"],
        "realIstftSnrDb": stage_reports["realIstftVsOriginalTorchIstft"]
        ["aggregate"]["signalToNoiseDb"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
