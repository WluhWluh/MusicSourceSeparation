#!/usr/bin/env python3
"""Isolate HTDemucs transformer layer zero ONNX numerical behavior."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


EXPECTED_SOURCE_SHA256 = "48f8e84945579f8ab340e083339e9221e03785dbe733a52c388200b6d3ca779a"
EXPECTED_FIXTURE_SHA256 = "9515d42e72e96036e34f0193d1302e8cccbf80c4e9e7084d1fcba26e17fea40e"
EXPECTED_WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
EXPECTED_DEMUCS_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
EXPECTED_EXPORTER_REVISION = "85db5c80aba33f0f2bdf88034a4be6539feec85b"
INPUT_NAME = "/crosstransformer/Add_8_output_0"
PROBES = {
    "norm1": "/crosstransformer/layers.0/norm1/LayerNormalization_output_0",
    "qkvProjection": "/crosstransformer/layers.0/self_attn/Add_output_0",
    "query": "/crosstransformer/layers.0/self_attn/Transpose_1_output_0",
    "scaledQuery": "/crosstransformer/layers.0/self_attn/Mul_4_output_0",
    "transposedKey": "/crosstransformer/layers.0/self_attn/Transpose_3_output_0",
    "value": "/crosstransformer/layers.0/self_attn/Transpose_2_output_0",
    "attentionLogits": "/crosstransformer/layers.0/self_attn/MatMul_1_output_0",
    "attentionWeights": "/crosstransformer/layers.0/self_attn/Softmax_output_0",
    "attentionContext": "/crosstransformer/layers.0/self_attn/MatMul_2_output_0",
    "attentionProjection": "/crosstransformer/layers.0/self_attn/Transpose_5_output_0",
    "scaledAttention": "/crosstransformer/layers.0/gamma_1/Mul_output_0",
    "attentionResidual": "/crosstransformer/layers.0/Add_output_0",
    "norm2": "/crosstransformer/layers.0/norm2/LayerNormalization_output_0",
    "linear1": "/crosstransformer/layers.0/linear1/Add_output_0",
    "gelu": "/crosstransformer/layers.0/Mul_1_output_0",
    "linear2": "/crosstransformer/layers.0/linear2/Add_output_0",
    "scaledFeedForward": "/crosstransformer/layers.0/gamma_2/Mul_output_0",
    "feedForwardResidual": "/crosstransformer/layers.0/Add_2_output_0",
    "groupNormTransposed": "/crosstransformer/layers.0/norm_out/Transpose_output_0",
    "groupNormReshaped": "/crosstransformer/layers.0/norm_out/Reshape_output_0",
    "groupNormNormalized": "/crosstransformer/layers.0/norm_out/InstanceNormalization_output_0",
    "groupNormRestored": "/crosstransformer/layers.0/norm_out/Reshape_1_output_0",
    "groupNormScaled": "/crosstransformer/layers.0/norm_out/Mul_output_0",
    "groupNormShifted": "/crosstransformer/layers.0/norm_out/Add_output_0",
    "output": "/crosstransformer/layers.0/norm_out/Transpose_1_output_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rewrite-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--demucs-root", type=Path, required=True)
    parser.add_argument("--exporter-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


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
            20.0 * math.log10(max(signal_rms, 1e-30) / max(error_rms, 1e-30))
        ),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.resolve()
    manifest_path = args.rewrite_manifest.resolve()
    fixture = args.fixture.resolve()
    weights = args.weights.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Rewrite manifest is not complete")
    if manifest.get("source", {}).get("sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("Rewrite source identity mismatch")
    if manifest.get("artifact", {}).get("sha256") != sha256_file(model):
        raise ValueError("Rewrite artifact identity mismatch")
    if manifest.get("artifact", {}).get("byteSize") != model.stat().st_size:
        raise ValueError("Rewrite artifact size mismatch")
    if manifest.get("rewrite", {}).get("accumulatorDtype") != "float64":
        raise ValueError("Transformer localization requires the FP64 reduction rewrite")
    if sha256_file(fixture) != EXPECTED_FIXTURE_SHA256:
        raise ValueError("Fixture identity mismatch")
    if sha256_file(weights) != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Official safetensors identity mismatch")
    demucs_root = args.demucs_root.resolve()
    exporter_root = args.exporter_root.resolve()
    if git_revision(demucs_root) != EXPECTED_DEMUCS_REVISION:
        raise ValueError("Demucs revision mismatch")
    if git_revision(exporter_root) != EXPECTED_EXPORTER_REVISION:
        raise ValueError("Exporter revision mismatch")
    return {
        "model": model,
        "manifestPath": manifest_path,
        "manifest": manifest,
        "fixture": fixture,
        "weights": weights,
        "demucsRoot": demucs_root,
        "exporterRoot": exporter_root,
    }


def extract_subgraphs(model: Path, final_path: Path, probes_path: Path) -> None:
    import onnx

    onnx.utils.extract_model(
        str(model),
        str(final_path),
        [INPUT_NAME],
        [PROBES["output"]],
        check_model=True,
    )
    onnx.utils.extract_model(
        str(model),
        str(probes_path),
        [INPUT_NAME],
        list(PROBES.values()),
        check_model=True,
    )


def open_session(model: Path, threads: int, optimization: str) -> Any:
    import onnxruntime as ort

    levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options = ort.SessionOptions()
    options.graph_optimization_level = levels[optimization]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )


def as_numpy(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)


def torch_references(layer: Any, layer_input: Any, torch: Any) -> dict[str, np.ndarray]:
    functional = torch.nn.functional
    norm1 = layer.norm1(layer_input)
    attention = layer.self_attn
    sequence = norm1.transpose(0, 1)
    qkv = functional.linear(
        sequence, attention.in_proj_weight, attention.in_proj_bias
    )
    query, key, value = qkv.chunk(3, dim=-1)
    target_length, batch_size, embed_dim = query.shape
    head_count = attention.num_heads
    head_dim = embed_dim // head_count
    query = query.contiguous().view(
        target_length, batch_size * head_count, head_dim
    ).transpose(0, 1)
    key = key.contiguous().view(
        target_length, batch_size * head_count, head_dim
    ).transpose(0, 1)
    value = value.contiguous().view(
        target_length, batch_size * head_count, head_dim
    ).transpose(0, 1)
    scaled_query = query * (head_dim ** -0.5)
    transposed_key = key.transpose(1, 2)
    logits = torch.bmm(scaled_query, transposed_key)
    weights = functional.softmax(logits, dim=-1)
    context = torch.bmm(weights, value)
    projected_sequence = context.transpose(0, 1).contiguous().view(
        target_length, batch_size, embed_dim
    )
    projected_sequence = functional.linear(
        projected_sequence,
        attention.out_proj.weight,
        attention.out_proj.bias,
    )
    projected = projected_sequence.transpose(0, 1)
    scaled_attention = layer.gamma_1(projected)
    attention_residual = layer_input + scaled_attention
    norm2 = layer.norm2(attention_residual)
    linear1 = layer.linear1(norm2)
    gelu = functional.gelu(linear1)
    linear2 = layer.linear2(gelu)
    scaled_feed_forward = layer.gamma_2(linear2)
    feed_forward_residual = attention_residual + scaled_feed_forward
    output = layer.norm_out(feed_forward_residual)
    transposed = feed_forward_residual.transpose(1, 2)
    reshaped = transposed.reshape(transposed.shape[0], 1, -1)
    normalized = functional.instance_norm(
        reshaped,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        use_input_stats=True,
        momentum=0.1,
        eps=layer.norm_out.eps,
    )
    restored = normalized.reshape(transposed.shape)
    scaled = restored * layer.norm_out.weight[:, None]
    shifted = scaled + layer.norm_out.bias[:, None]
    return {
        "norm1": as_numpy(norm1),
        "qkvProjection": as_numpy(qkv),
        "query": as_numpy(query),
        "scaledQuery": as_numpy(scaled_query),
        "transposedKey": as_numpy(transposed_key),
        "value": as_numpy(value),
        "attentionLogits": as_numpy(logits),
        "attentionWeights": as_numpy(weights),
        "attentionContext": as_numpy(context),
        "attentionProjection": as_numpy(projected),
        "scaledAttention": as_numpy(scaled_attention),
        "attentionResidual": as_numpy(attention_residual),
        "norm2": as_numpy(norm2),
        "linear1": as_numpy(linear1),
        "gelu": as_numpy(gelu),
        "linear2": as_numpy(linear2),
        "scaledFeedForward": as_numpy(scaled_feed_forward),
        "feedForwardResidual": as_numpy(feed_forward_residual),
        "groupNormTransposed": as_numpy(transposed),
        "groupNormReshaped": as_numpy(reshaped),
        "groupNormNormalized": as_numpy(normalized),
        "groupNormRestored": as_numpy(restored),
        "groupNormScaled": as_numpy(scaled),
        "groupNormShifted": as_numpy(shifted),
        "output": as_numpy(output),
    }


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.threads > 64:
        raise ValueError("--threads must be in [1,64]")
    identity = validate_inputs(args)
    output_dir = args.output_dir.resolve()
    stage = output_dir.with_name(output_dir.name + ".partial")
    if (output_dir.exists() or stage.exists()) and not args.force:
        raise FileExistsError(f"Output exists; pass --force: {output_dir}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    final_model = stage / "transformer-layer0.final.onnx"
    probe_model = stage / "transformer-layer0.probes.onnx"
    extract_subgraphs(identity["model"], final_model, probe_model)

    sys.path.insert(0, str(identity["exporterRoot"] / "src"))
    sys.path.insert(0, str(identity["demucsRoot"]))
    import torch
    from demucs.hf import load_safetensors_model
    from demucs_onnx.export.patch import patch_htdemucs_for_onnx

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = patch_htdemucs_for_onnx(
        load_safetensors_model(identity["weights"]).cpu().float().eval()
    )
    captured: dict[str, Any] = {}

    def capture_input(_module: Any, inputs: Any) -> None:
        captured["input"] = inputs[0].detach().clone()

    hook = model.crosstransformer.layers[0].register_forward_pre_hook(capture_input)
    waveform = np.fromfile(identity["fixture"], dtype="<f4").reshape(1, 2, 343_980)
    with torch.inference_mode():
        model(torch.from_numpy(waveform))
    hook.remove()
    layer_input = captured["input"]
    with torch.inference_mode():
        references = torch_references(
            model.crosstransformer.layers[0], layer_input, torch
        )
    input_numpy = as_numpy(layer_input)

    probe_session = open_session(probe_model, args.threads, "disabled")
    probe_values = probe_session.run(
        list(PROBES.values()), {INPUT_NAME: input_numpy}
    )
    comparisons: dict[str, Any] = {}
    for label, candidate in zip(PROBES, probe_values, strict=True):
        candidate_array = np.asarray(candidate, dtype=np.float32)
        if references[label].shape != candidate_array.shape:
            raise ValueError(
                f"Shape mismatch for {label}: "
                f"Torch={references[label].shape} ORT={candidate_array.shape}"
            )
        comparisons[label] = metric(references[label], candidate_array)

    final_comparisons: dict[str, Any] = {}
    for optimization in ("disabled", "all"):
        session = open_session(final_model, args.threads, optimization)
        candidate = session.run([PROBES["output"]], {INPUT_NAME: input_numpy})[0]
        final_comparisons[optimization] = metric(references["output"], candidate)

    report = {
        "schemaVersion": 1,
        "status": "complete",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": args.threads,
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "provenance": {
            "model": {
                "path": str(identity["model"]),
                "byteSize": identity["model"].stat().st_size,
                "sha256": sha256_file(identity["model"]),
            },
            "rewriteManifest": {
                "path": str(identity["manifestPath"]),
                "sha256": sha256_file(identity["manifestPath"]),
            },
            "fixture": {
                "path": str(identity["fixture"]),
                "sha256": EXPECTED_FIXTURE_SHA256,
            },
            "weights": {
                "path": str(identity["weights"]),
                "sha256": EXPECTED_WEIGHT_SHA256,
            },
            "demucsRevision": EXPECTED_DEMUCS_REVISION,
            "exporterRevision": EXPECTED_EXPORTER_REVISION,
            "subgraphs": {
                "final": {
                    "path": str(final_model),
                    "byteSize": final_model.stat().st_size,
                    "sha256": sha256_file(final_model),
                },
                "probes": {
                    "path": str(probe_model),
                    "byteSize": probe_model.stat().st_size,
                    "sha256": sha256_file(probe_model),
                },
            },
        },
        "isolation": {
            "input": INPUT_NAME,
            "inputShape": list(input_numpy.shape),
            "inputSource": "exact patched-Torch transformer layer-zero input",
            "outputs": PROBES,
        },
        "comparisonsWithOptimizationsDisabled": comparisons,
        "finalOutputComparisons": final_comparisons,
    }
    write_json_atomic(stage / "report.json", report)
    del model, references, probe_values, probe_session
    gc.collect()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    stage.replace(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str((output_dir / "report.json").resolve()),
                "comparisons": {
                    label: {
                        "snrDb": value["signalToNoiseDb"],
                        "maxAbsoluteError": value["maximumAbsoluteError"],
                    }
                    for label, value in comparisons.items()
                },
                "finalOutputComparisons": final_comparisons,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
