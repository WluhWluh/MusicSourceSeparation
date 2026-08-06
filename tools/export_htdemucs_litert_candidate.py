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


MODEL_ID = "htdemucs_6s_core_smoke_2s_fp32_v1_0_0"
SOURCE_ORDER = ("drums", "bass", "other", "vocals", "guitar", "piano")
EXPECTED_WEIGHT_BYTES = 54_885_744
EXPECTED_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
EXPECTED_METADATA_BYTES = 10_398
EXPECTED_METADATA_SHA256 = "72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e"
EXPECTED_LOADER_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
EXPECTED_DEMUCS_LITE_REVISION = "9a2a17c7a81843c2ae49674986f9e1e8b5f6915f"
MINIMUM_SNR_DB = 80.0
MAXIMUM_ABSOLUTE_ERROR = 1e-3


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
    parser.add_argument("--samples", type=int, default=88_200)
    parser.add_argument("--seed", type=int, default=20_260_803)
    parser.add_argument("--skip-litert", action="store_true")
    parser.add_argument("--reuse-litert", action="store_true")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "modelId": MODEL_ID,
        "status": "running",
        "completedStage": "arguments",
        "qualityGate": {
            "minimumSignalToNoiseDb": MINIMUM_SNR_DB,
            "maximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
            "acceptedForDeviceTesting": False,
        },
    }

    def checkpoint(stage: str) -> None:
        report["completedStage"] = stage
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    try:
        verify_file(args.weights, EXPECTED_WEIGHT_BYTES, EXPECTED_WEIGHT_SHA256)
        verify_file(args.metadata, EXPECTED_METADATA_BYTES, EXPECTED_METADATA_SHA256)
        loader_revision = git_revision(args.demucs_root)
        wrapper_revision = git_revision(args.demucs_lite_root)
        if loader_revision != EXPECTED_LOADER_REVISION:
            raise RuntimeError(f"Unexpected Demucs loader revision: {loader_revision}")
        if wrapper_revision != EXPECTED_DEMUCS_LITE_REVISION:
            raise RuntimeError(f"Unexpected demucs-lite revision: {wrapper_revision}")

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
            "loaderRevision": loader_revision,
            "demucsLiteReferenceRevision": wrapper_revision,
            "exportScript": {
                "fileName": Path(__file__).name,
                "sha256": sha256(Path(__file__)),
            },
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
            reconstructed = model._ispec(
                channels_to_spec(torch, frequency), length=args.samples
            ) + waveform
            torch_seconds = time.perf_counter() - torch_started

        mix_np = as_numpy(mix)
        spectrum_np = as_numpy(spectrum)
        frequency_np = as_numpy(frequency)
        waveform_np = as_numpy(waveform)
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
        report["seconds"] = {"torchReferenceAndCore": torch_seconds}
        report["torchCoreReconstruction"] = torch_reconstruction
        report["fixtures"] = {
            "seed": args.seed,
            "waveformInput": write_raw(args.fixtures / "waveform_input.f32le.raw", mix_np),
            "spectrumInput": write_raw(args.fixtures / "spectrum_input.f32le.raw", spectrum_np),
            "frequencyGolden": write_raw(args.fixtures / "frequency_golden.f32le.raw", frequency_np),
            "waveformGolden": write_raw(args.fixtures / "waveform_golden.f32le.raw", waveform_np),
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
                onnx_combined = as_numpy(
                    model._ispec(
                        channels_to_spec(torch, torch.from_numpy(onnx_frequency)),
                        length=args.samples,
                    )
                    + torch.from_numpy(onnx_waveform)
                )
            onnx_metrics = {
                "frequencyOutput": metrics(frequency_np, onnx_frequency),
                "waveformOutput": metrics(waveform_np, onnx_waveform),
                "combinedOutput": metrics(reference_np, onnx_combined),
                "perStemCombined": {
                    name: metrics(reference_np[:, index], onnx_combined[:, index])
                    for index, name in enumerate(SOURCE_ORDER)
                },
            }
            if not all(
                metric_passes(onnx_metrics[key])
                for key in ("frequencyOutput", "waveformOutput", "combinedOutput")
            ):
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
                lightweight_conversion=True,
                enable_x64=False,
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
            lite_combined = as_numpy(
                model._ispec(
                    channels_to_spec(torch, torch.from_numpy(lite_frequency)),
                    length=args.samples,
                )
                + torch.from_numpy(lite_waveform)
            )

        raw_frequency = metrics(frequency_np, lite_frequency)
        raw_waveform = metrics(waveform_np, lite_waveform)
        combined = metrics(reference_np, lite_combined)
        per_stem = {
            name: metrics(reference_np[:, index], lite_combined[:, index])
            for index, name in enumerate(SOURCE_ORDER)
        }
        accepted = all(metric_passes(value) for value in (raw_frequency, raw_waveform, combined))
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
            "combinedOutput": combined,
            "perStemCombined": per_stem,
        }
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
