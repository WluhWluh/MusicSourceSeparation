#!/usr/bin/env python3
"""Expose selected HTDemucs ONNX intermediates and compare with patched Torch."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable

import numpy as np


EXPECTED_MODEL_SHA256 = "48f8e84945579f8ab340e083339e9221e03785dbe733a52c388200b6d3ca779a"
EXPECTED_FIXTURE_SHA256 = "9515d42e72e96036e34f0193d1302e8cccbf80c4e9e7084d1fcba26e17fea40e"
EXPECTED_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
PROBES = {
    "realStftCos": "/real_stft/Conv_output_0",
    "realStftSin": "/real_stft/Conv_1_output_0",
    "spectrumChannels": "/Reshape_2_output_0",
    "frequencyMean": "/ReduceMean_output_0",
    "frequencyVariance": "/Div_output_0",
    "frequencyStd": "/Sqrt_output_0",
    "frequencyDivisor": "/Add_output_0",
    "normalizedFrequencyInput": "/Div_1_output_0",
    "timeMean": "/ReduceMean_3_output_0",
    "timeVariance": "/Div_2_output_0",
    "timeStd": "/Sqrt_1_output_0",
    "timeDivisor": "/Add_1_output_0",
    "normalizedTimeInput": "/Div_3_output_0",
    "timeEncoder0": "/tencoder.0/Mul_2_output_0",
    "timeEncoder1": "/tencoder.1/Mul_2_output_0",
    "timeEncoder2": "/tencoder.2/Mul_2_output_0",
    "timeEncoder3": "/tencoder.3/Mul_2_output_0",
    "frequencyEncoder0": "/encoder.0/Mul_2_output_0",
    "frequencyEncoder1": "/encoder.1/Mul_2_output_0",
    "frequencyEncoder2": "/encoder.2/Mul_2_output_0",
    "frequencyEncoder3": "/encoder.3/Mul_2_output_0",
    "frequencyTransformer": "/crosstransformer/Transpose_8_output_0",
    "frequencyDecoder0": "/decoder.0/Mul_2_output_0",
    "frequencyDecoder1": "/decoder.1/Mul_2_output_0",
    "frequencyDecoder2": "/decoder.2/Mul_2_output_0",
    "frequencyDecoder": "/decoder.3/Slice_output_0",
    "denormalizedFrequency": "/Reshape_4_output_0",
    "istftInput": "/Pad_2_output_0",
    "timeDecoder": "/tdecoder.3/Slice_output_0",
    "istftCos": "/real_istft/ConvTranspose_output_0",
    "istftSin": "/real_istft/ConvTranspose_1_output_0",
    "istftSum": "/real_istft/Add_output_0",
    "istftWaveform": "/Slice_5_output_0",
    "denormalizedTime": "/Add_4_output_0",
    "stems": "stems",
}

TRANSFORMER_PROBES = {
    "transformerFrequencyInput": "/crosstransformer/Add_8_output_0",
    "transformerTimeInput": "/crosstransformer/Add_9_output_0",
    "transformerTimeEncoded": "/crosstransformer/Transpose_6_output_0",
    "transformerTimeNormalized": (
        "/crosstransformer/norm_in_t/LayerNormalization_output_0"
    ),
    "transformerTimePosition": "/crosstransformer/Transpose_7_output_0",
    "transformerTimePositionScaled": "/crosstransformer/Mul_19_output_0",
    **{
        f"transformerFrequencyLayer{index}": (
            f"/crosstransformer/layers.{index}/norm_out/Transpose_1_output_0"
        )
        for index in range(5)
    },
    **{
        f"transformerTimeLayer{index}": (
            f"/crosstransformer/layers_t.{index}/norm_out/Transpose_1_output_0"
            if index < 4
            else "/crosstransformer/layers_t.4/norm_out/Add_output_0"
        )
        for index in range(5)
    },
}
PROBES.update(TRANSFORMER_PROBES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "shape": list(reference.shape),
        "finite": bool(np.isfinite(candidate64).all()),
        "signalRootMeanSquare": signal_rms,
        "rootMeanSquareError": error_rms,
        "meanAbsoluteError": float(np.mean(np.abs(error))),
        "maximumAbsoluteError": float(np.max(np.abs(error))),
        "signalToNoiseDb": float(
            20.0
            * np.log10(max(signal_rms, 1e-30) / max(error_rms, 1e-30))
        ),
    }


def add_probe_outputs(source: Path, destination: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(source), load_external_data=False)
    produced = {output for node in model.graph.node for output in node.output}
    available = {item.name for item in model.graph.output}
    for label, tensor_name in PROBES.items():
        if tensor_name not in produced and tensor_name not in available:
            raise ValueError(f"Probe tensor is absent: {label}={tensor_name}")
        if tensor_name not in available:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    tensor_name, onnx.TensorProto.FLOAT, None
                )
            )
            available.add(tensor_name)
    onnx.save(model, str(destination))
    return {
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "probeModel": {
            "path": str(destination),
            "byteSize": destination.stat().st_size,
            "sha256": sha256_file(destination),
        },
        "outputs": PROBES,
    }


def run_ort(probe_model: Path, waveform: np.ndarray, threads: int) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(probe_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    result = session.run(list(PROBES.values()), {"mix": waveform})
    return {
        label: np.asarray(value, dtype=np.float32)
        for label, value in zip(PROBES, result, strict=True)
    }


def run_ort_with_optimization(
    probe_model: Path,
    waveform: np.ndarray,
    threads: int,
    optimization: str,
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options = ort.SessionOptions()
    options.graph_optimization_level = levels[optimization]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(probe_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    result = session.run(list(PROBES.values()), {"mix": waveform})
    return {
        label: np.asarray(value, dtype=np.float32)
        for label, value in zip(PROBES, result, strict=True)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--demucs-root", type=Path, required=True)
    parser.add_argument("--exporter-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--rewrite-manifest",
        type=Path,
        help="Required when --model is a derived frequency-normalization rewrite.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    fixture_path = args.fixture.resolve()
    weights = args.weights.resolve()
    output_dir = args.output_dir.resolve()
    stage = output_dir.with_name(output_dir.name + ".partial")
    if (output_dir.exists() or stage.exists()) and not args.force:
        raise FileExistsError(f"Output exists; pass --force: {output_dir}")
    model_sha256 = sha256_file(model_path)
    rewrite_manifest = None
    if model_sha256 != EXPECTED_MODEL_SHA256:
        if args.rewrite_manifest is None:
            raise ValueError(
                "Derived ONNX identity requires --rewrite-manifest provenance"
            )
        rewrite_manifest_path = args.rewrite_manifest.resolve()
        rewrite_manifest = json.loads(
            rewrite_manifest_path.read_text(encoding="utf-8")
        )
        if rewrite_manifest.get("status") != "complete":
            raise ValueError("Rewrite manifest is not complete")
        if rewrite_manifest.get("source", {}).get("sha256") != EXPECTED_MODEL_SHA256:
            raise ValueError("Rewrite manifest source identity mismatch")
        if rewrite_manifest.get("artifact", {}).get("sha256") != model_sha256:
            raise ValueError("Rewrite manifest artifact identity mismatch")
        if rewrite_manifest.get("artifact", {}).get("byteSize") != model_path.stat().st_size:
            raise ValueError("Rewrite manifest artifact size mismatch")
        rewrite_manifest = {
            "path": str(rewrite_manifest_path),
            "sha256": sha256_file(rewrite_manifest_path),
            "content": rewrite_manifest,
        }
    elif args.rewrite_manifest is not None:
        raise ValueError("--rewrite-manifest is only valid for a derived model")
    if sha256_file(fixture_path) != EXPECTED_FIXTURE_SHA256:
        raise ValueError("Fixture identity mismatch")
    if sha256_file(weights) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Weight identity mismatch")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    probe_model = stage / "htdemucs_6s.fp32.probes.onnx"
    probe_identity = add_probe_outputs(model_path, probe_model)
    waveform = np.fromfile(fixture_path, dtype="<f4").reshape(1, 2, 343_980)
    ort_values_by_optimization = {
        optimization: run_ort_with_optimization(
            probe_model, waveform, args.threads, optimization
        )
        for optimization in ("disabled", "basic", "extended", "all")
    }
    ort_values = ort_values_by_optimization["all"]

    demucs_root = args.demucs_root.resolve()
    exporter_root = args.exporter_root.resolve()
    sys.path.insert(0, str(exporter_root / "src"))
    sys.path.insert(0, str(demucs_root))
    import torch
    import onnx
    import onnxruntime as ort
    from demucs.hf import load_safetensors_model
    from demucs_onnx.export.patch import patch_htdemucs_for_onnx

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = patch_htdemucs_for_onnx(
        load_safetensors_model(weights).cpu().float().eval()
    )
    captured: dict[str, np.ndarray] = {}

    def save(name: str, value: Any) -> None:
        captured[name] = np.ascontiguousarray(
            value.detach().cpu().numpy(), dtype=np.float32
        )

    def decoder_hook(_module: Any, _inputs: Any, output: Any) -> None:
        save("frequencyDecoder", output[0])

    def time_decoder_hook(_module: Any, _inputs: Any, output: Any) -> None:
        save("timeDecoder", output[0])

    def real_stft_hook(module: Any, inputs: Any, output: Any) -> None:
        value = inputs[0]
        other = tuple(value.shape[:-1])
        length = value.shape[-1]
        flat = value.reshape(-1, 1, length)
        pad = module.n_fft // 2
        padded = torch.nn.functional.pad(flat, (pad, pad), mode="reflect")
        with torch.inference_mode():
            cos = torch.nn.functional.conv1d(
                padded, module.cos_kernel, stride=module.hop_length
            )
            sin = torch.nn.functional.conv1d(
                padded, module.sin_kernel, stride=module.hop_length
            )
        save("realStftCos", cos)
        save("realStftSin", sin)

    def istft_hook(_module: Any, inputs: Any, output: Any) -> None:
        z = inputs[0]
        other = tuple(z.shape[:-3])
        frequencies = z.shape[-2]
        frames = z.shape[-1]
        flat = z.reshape(-1, 2, frequencies, frames)
        with torch.inference_mode():
            cos = torch.nn.functional.conv_transpose1d(
                flat[:, 0], model.real_istft.inv_cos,
                stride=model.real_istft.hop_length,
            )
            sin = torch.nn.functional.conv_transpose1d(
                flat[:, 1], model.real_istft.inv_sin,
                stride=model.real_istft.hop_length,
            )
        save("istftCos", cos)
        save("istftSin", sin)
        save("istftSum", cos + sin)

    hooks = []

    def tensor_hook(name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            save(name, output[0] if isinstance(output, tuple) else output)
        return hook

    def transformer_input_hook(name: str) -> Callable[..., None]:
        def hook(_module: Any, inputs: Any) -> None:
            save(name, inputs[0])
        return hook

    def final_time_transformer_hook(
        _module: Any, _inputs: Any, output: Any
    ) -> None:
        save("transformerTimeLayer4", output.transpose(1, 2))

    for index, module in enumerate(model.encoder):
        hooks.append(module.register_forward_hook(tensor_hook(f"frequencyEncoder{index}")))
    for index, module in enumerate(model.tencoder):
        hooks.append(module.register_forward_hook(tensor_hook(f"timeEncoder{index}")))
    hooks.append(
        model.crosstransformer.register_forward_hook(
            lambda _module, _inputs, output: save("frequencyTransformer", output[0])
        )
    )
    for index, module in enumerate(model.decoder[:-1]):
        hooks.append(module.register_forward_hook(tensor_hook(f"frequencyDecoder{index}")))
    hooks.extend([
        model.crosstransformer.layers[0].register_forward_pre_hook(
            transformer_input_hook("transformerFrequencyInput")
        ),
        model.crosstransformer.layers_t[0].register_forward_pre_hook(
            transformer_input_hook("transformerTimeInput")
        ),
    ])
    for index, module in enumerate(model.crosstransformer.layers):
        hooks.append(
            module.register_forward_hook(
                tensor_hook(f"transformerFrequencyLayer{index}")
            )
        )
    for index, module in enumerate(model.crosstransformer.layers_t):
        hooks.append(
            module.register_forward_hook(
                final_time_transformer_hook
                if index == 4
                else tensor_hook(f"transformerTimeLayer{index}")
            )
        )
    hooks.extend([
        model.decoder[-1].register_forward_hook(decoder_hook),
        model.tdecoder[-1].register_forward_hook(time_decoder_hook),
        model.real_stft.register_forward_hook(real_stft_hook),
        model.real_istft.register_forward_hook(istft_hook),
    ])
    with torch.inference_mode():
        stems = model(torch.from_numpy(waveform))
    for hook in hooks:
        hook.remove()
    save("stems", stems)

    meant = torch.from_numpy(waveform).mean(dim=(1, 2), keepdim=True)
    stdt = torch.from_numpy(waveform).std(dim=(1, 2), keepdim=True)
    raw_time = torch.from_numpy(captured["timeDecoder"])
    batch = raw_time.shape[0]
    denormalized_time = raw_time.view(batch, len(model.sources), -1, 343_980)
    denormalized_time = denormalized_time * stdt[:, None] + meant[:, None]
    save("denormalizedTime", denormalized_time)

    with torch.inference_mode():
        spectrum = model._spec(torch.from_numpy(waveform))
        magnitude = model._magnitude(spectrum)
        save("spectrumChannels", magnitude)
        mean_frequency = magnitude.mean(dim=(1, 2, 3), keepdim=True)
        std_frequency = magnitude.std(dim=(1, 2, 3), keepdim=True)
        save("frequencyMean", mean_frequency)
        save("frequencyVariance", std_frequency * std_frequency)
        save("frequencyStd", std_frequency)
        save("frequencyDivisor", std_frequency + 1e-5)
        save(
            "normalizedFrequencyInput",
            (magnitude - mean_frequency) / (1e-5 + std_frequency),
        )
        mix_tensor = torch.from_numpy(waveform)
        mean_time = mix_tensor.mean(dim=(1, 2), keepdim=True)
        std_time = mix_tensor.std(dim=(1, 2), keepdim=True)
        save("timeMean", mean_time)
        save("timeVariance", std_time * std_time)
        save("timeStd", std_time)
        save("timeDivisor", std_time + 1e-5)
        save(
            "normalizedTimeInput",
            (mix_tensor - mean_time) / (1e-5 + std_time),
        )
        encoded_time = torch.from_numpy(captured["timeEncoder3"]).transpose(1, 2)
        save("transformerTimeEncoded", encoded_time)
        normalized_encoded_time = model.crosstransformer.norm_in_t(encoded_time)
        save("transformerTimeNormalized", normalized_encoded_time)
        batch_size, time_steps, channels = encoded_time.shape
        position = model.crosstransformer._get_pos_embedding(
            time_steps,
            batch_size,
            channels,
            encoded_time.device,
        ).permute(1, 0, 2)
        save("transformerTimePosition", position)
        save(
            "transformerTimePositionScaled",
            model.crosstransformer.weight_pos_embed * position,
        )
        raw_frequency = torch.from_numpy(captured["frequencyDecoder"])
        frequency = raw_frequency.view(
            raw_frequency.shape[0], len(model.sources), -1,
            raw_frequency.shape[-2], raw_frequency.shape[-1],
        )
        frequency = frequency * std_frequency[:, None] + mean_frequency[:, None]
        save("denormalizedFrequency", frequency)
        frequency_real = model._mask(spectrum, frequency)
        istft_input = torch.nn.functional.pad(frequency_real, (0, 0, 0, 1))
        istft_input = torch.nn.functional.pad(istft_input, (2, 2))
        save("istftInput", istft_input)
        save(
            "istftWaveform",
            model._ispec(frequency_real, length=343_980),
        )

    comparisons = {}
    for label in PROBES:
        left = captured[label]
        right = ort_values[label]
        if (
            label == "denormalizedFrequency"
            and left.ndim == 5
            and right.ndim == 6
            and right.shape[2] * right.shape[3] == left.shape[2]
        ):
            right = right.reshape(left.shape)
        if left.shape != right.shape:
            raise ValueError(
                f"Probe shape mismatch for {label}: Torch={left.shape}, ORT={right.shape}"
            )
        comparisons[label] = metric(left, right)

    optimization_comparisons = {}
    for optimization, values in ort_values_by_optimization.items():
        optimization_comparisons[optimization] = {}
        for label in PROBES:
            left = captured[label]
            right = values[label]
            if (
                label == "denormalizedFrequency"
                and left.ndim == 5
                and right.ndim == 6
                and right.shape[2] * right.shape[3] == left.shape[2]
            ):
                right = right.reshape(left.shape)
            optimization_comparisons[optimization][label] = metric(left, right)

    report = {
        "schemaVersion": 1,
        "status": "complete",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "provider": "CPUExecutionProvider",
            "threads": args.threads,
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "provenance": {
            "model": probe_identity,
            "rewriteManifest": rewrite_manifest,
            "fixture": {
                "path": str(fixture_path),
                "sha256": EXPECTED_FIXTURE_SHA256,
            },
            "weights": {
                "path": str(weights),
                "sha256": EXPECTED_WEIGHT_SHA256,
            },
        },
        "comparisons": comparisons,
        "optimizationComparisons": optimization_comparisons,
        "interpretation": (
            "The earliest probe with a large error identifies whether the dominant "
            "ORT residual exists before or inside the real iSTFT subgraph."
        ),
    }
    write_json_atomic(stage / "report.json", report)
    del model
    gc.collect()
    probe_model.unlink()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    stage.replace(output_dir)
    print(json.dumps({
        "status": "complete",
        "report": str((output_dir / "report.json").resolve()),
        "comparisons": {
            label: {
                "snrDb": item["signalToNoiseDb"],
                "maxAbsoluteError": item["maximumAbsoluteError"],
            }
            for label, item in comparisons.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
