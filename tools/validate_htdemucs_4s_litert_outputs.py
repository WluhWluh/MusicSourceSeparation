#!/usr/bin/env python3
"""Validate official four-stem device outputs against frozen Torch goldens."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import export_htdemucs_4s_litert_candidate as exporter


MODEL_FAMILY = "htdemucs-4s"
MODEL_SPEC = exporter.configure_model_family(MODEL_FAMILY)
MODEL_ID = MODEL_SPEC["modelIds"]["canonical_7p8s"]
PROFILE_SPECS = {
    "canonical_7p8s": {
        "modelId": MODEL_ID,
        "sampleCount": exporter.PROFILE_SPECS["canonical_7p8s"]["sampleCount"],
    },
}
SOURCE_ORDER = MODEL_SPEC["sourceOrder"]
EXPECTED_WEIGHT_BYTES = MODEL_SPEC["weightBytes"]
EXPECTED_WEIGHT_SHA256 = MODEL_SPEC["weightSha256"]
LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR = exporter.LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR
LOW_SIGNAL_REFERENCE_RMS = exporter.LOW_SIGNAL_REFERENCE_RMS
MAXIMUM_ABSOLUTE_ERROR = exporter.MAXIMUM_ABSOLUTE_ERROR
MINIMUM_SNR_DB = exporter.MINIMUM_SNR_DB
channels_to_spec = exporter.channels_to_spec
evaluate_metric_bundle = exporter.evaluate_metric_bundle
metric_bundle = exporter.metric_bundle
verify_file = exporter.verify_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demucs-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--frequency-output", type=Path, required=True)
    parser.add_argument("--waveform-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_SPECS), default="canonical_7p8s")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def read_raw(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype("<f4").itemsize
    if not path.is_file() or path.stat().st_size != expected_bytes:
        actual = path.stat().st_size if path.is_file() else None
        raise RuntimeError(f"Unexpected raw tensor size for {path}: {actual} != {expected_bytes}")
    return np.memmap(path, mode="r", dtype="<f4", shape=shape)


def main() -> int:
    args = parse_args()
    profile = PROFILE_SPECS[args.profile]
    samples = int(profile["sampleCount"])
    frames = (samples + 1023) // 1024
    if args.profile == "canonical_7p8s":
        if args.manifest is None:
            raise RuntimeError("--manifest is required for canonical_7p8s validation")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest["modelId"] != profile["modelId"]:
            raise RuntimeError("Host manifest model ID does not match the selected profile")
        gate = manifest["hostValidation"]["qualityGate"]
        if not gate["hostPipelineGatePassed"] or not gate["acceptedForDeviceTesting"]:
            raise RuntimeError("Host manifest has not passed the pipeline admission gate")
    else:
        manifest = None
    verify_file(args.weights, EXPECTED_WEIGHT_BYTES, EXPECTED_WEIGHT_SHA256)
    sys.path.insert(0, str(args.demucs_root))

    import torch
    from demucs.hf import load_safetensors_model

    source_count = len(SOURCE_ORDER)
    frequency_shape = (1, source_count, 4, 2048, frames)
    waveform_shape = (1, source_count, 2, samples)
    golden_frequency = read_raw(args.fixtures / "frequency_golden.f32le.raw", frequency_shape)
    golden_waveform = read_raw(args.fixtures / "waveform_golden.f32le.raw", waveform_shape)
    golden_frequency_waveform = (
        read_raw(args.fixtures / "frequency_waveform_golden.f32le.raw", waveform_shape)
        if (args.fixtures / "frequency_waveform_golden.f32le.raw").is_file()
        else None
    )
    golden_combined = read_raw(args.fixtures / "combined_golden.f32le.raw", waveform_shape)
    candidate_frequency = read_raw(args.frequency_output, frequency_shape)
    candidate_waveform = read_raw(args.waveform_output, waveform_shape)

    model = load_safetensors_model(args.weights).cpu().float().eval()
    model.segment = Fraction(39, 5) if args.profile == "canonical_7p8s" else Fraction(2, 1)
    model.use_train_segment = True
    with torch.inference_mode():
        frequency_tensor = torch.from_numpy(np.array(candidate_frequency, copy=True))
        waveform_tensor = torch.from_numpy(np.array(candidate_waveform, copy=True))
        candidate_frequency_waveform = model._ispec(
            channels_to_spec(torch, frequency_tensor),
            length=samples,
        ).cpu().numpy()
        candidate_combined = candidate_frequency_waveform + waveform_tensor.cpu().numpy()

    bundles = {
        "frequencyOutput": metric_bundle(golden_frequency, candidate_frequency),
        "waveformOutput": metric_bundle(golden_waveform, candidate_waveform),
        "combinedOutput": metric_bundle(golden_combined, candidate_combined),
    }
    if golden_frequency_waveform is not None:
        bundles["frequencyWaveformOutput"] = metric_bundle(
            golden_frequency_waveform,
            candidate_frequency_waveform,
        )
    evaluations = {
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
        for name, bundle in bundles.items()
    }
    accepted = all(value["accepted"] for value in evaluations.values())
    report: dict[str, Any] = {
        "status": "passed" if accepted else "failed",
        "modelId": profile["modelId"],
        "profile": args.profile,
        "gate": {
            "minimumSignalToNoiseDb": MINIMUM_SNR_DB,
            "maximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
            "lowSignalReferenceRms": LOW_SIGNAL_REFERENCE_RMS,
            "lowSignalMaximumAbsoluteError": LOW_SIGNAL_MAXIMUM_ABSOLUTE_ERROR,
            "lowSignalLatentMaximumAbsoluteError": MAXIMUM_ABSOLUTE_ERROR,
        },
        "metrics": bundles,
        "gateEvaluation": evaluations,
        "inputs": {
            "frequencyOutput": str(args.frequency_output),
            "waveformOutput": str(args.waveform_output),
            "fixtures": str(args.fixtures),
            "hostManifest": str(args.manifest) if args.manifest else None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
