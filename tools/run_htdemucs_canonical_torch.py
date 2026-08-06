#!/usr/bin/env python3
"""Run official HTDemucs-6s safetensors with the canonical full-song contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import wave
from typing import Any

import numpy as np

from run_htdemucs_canonical_onnx import (
    BYTES_PER_SAMPLE,
    CHANNELS,
    OVERLAP_SAMPLES,
    SAMPLE_RATE,
    STEM_COUNT,
    STEM_ORDER,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    CanonicalWav,
    StemStats,
    Timing,
    apply_streaming_ola,
    encode_pcm16,
    make_window_plans,
    metric,
    output_stage_paths,
    parse_duration_frames,
    read_normalized_window,
    sha256_file,
    timed,
    triangle_weights,
    write_json_atomic,
)


WEIGHT_PATH = Path("models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors")
WEIGHT_SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
WEIGHT_BYTE_SIZE = 54_885_744
METADATA_PATH = Path("models/demucs/official-hf/htdemucs_6s/5c90dfd2.json")
METADATA_SHA256 = "72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e"
DEMUCS_ROOT = Path(".tmp/demucs-adefossez")
DEMUCS_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
CANONICAL_MANIFEST = Path(
    "models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/candidate-manifest.json"
)
FIXTURE_ROOT = Path(
    "models/demucs/generated/htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0/fixtures"
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def process_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"pid": os.getpid()}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile"}:
                fields = value.strip().split()
                result[f"{key}KiB"] = int(fields[0])
    return result


def load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    weight = args.weights.resolve()
    metadata = args.metadata.resolve()
    demucs_root = args.demucs_root.resolve()
    if weight.stat().st_size != WEIGHT_BYTE_SIZE or sha256_file(weight) != WEIGHT_SHA256:
        raise ValueError("Official safetensors identity mismatch")
    if sha256_file(metadata) != METADATA_SHA256:
        raise ValueError("Official HTDemucs metadata identity mismatch")
    revision = git_revision(demucs_root)
    if revision != DEMUCS_REVISION:
        raise ValueError(f"Demucs checkout is {revision}, expected {DEMUCS_REVISION}")

    sys.path.insert(0, str(demucs_root))
    import torch
    from demucs.hf import load_safetensors_model
    from export_htdemucs_litert_candidate import install_deterministic_pos_embedding

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = load_safetensors_model(weight).cpu().float().eval()
    if tuple(model.sources) != STEM_ORDER:
        raise ValueError(f"Unexpected stem order: {tuple(model.sources)}")
    if model.samplerate != SAMPLE_RATE or model.segment != Fraction(39, 5):
        raise ValueError(
            f"Unexpected official workload: rate={model.samplerate}, segment={model.segment}"
        )
    if not model.use_train_segment:
        raise ValueError("Official model must retain its fixed training segment")
    install_deterministic_pos_embedding(model)
    return torch, model, {
        "weight": {
            "path": str(weight),
            "fileName": weight.name,
            "byteSize": weight.stat().st_size,
            "sha256": WEIGHT_SHA256,
            "format": "safetensors",
        },
        "metadata": {
            "path": str(metadata),
            "fileName": metadata.name,
            "byteSize": metadata.stat().st_size,
            "sha256": METADATA_SHA256,
        },
        "loader": {
            "path": str(demucs_root),
            "revision": revision,
        },
        "deterministicPositionEmbedding": "sin-shift-zero",
        "weightMutation": "none",
    }


def infer(torch: Any, model: Any, waveform: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        output = model(torch.from_numpy(waveform[np.newaxis, ...]))
    value = output.detach().cpu().numpy().astype(np.float32, copy=False)
    expected = (1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES)
    if value.shape != expected:
        raise ValueError(f"Unexpected Torch output shape: {value.shape}, expected {expected}")
    return value


def validate_fixture(
    torch: Any,
    model: Any,
    fixture_root: Path,
) -> dict[str, Any]:
    input_path = fixture_root / "waveform_input.f32le.raw"
    golden_path = fixture_root / "combined_golden.f32le.raw"
    waveform = np.fromfile(input_path, dtype="<f4").reshape(CHANNELS, WINDOW_SAMPLES)
    golden = np.fromfile(golden_path, dtype="<f4").reshape(
        1, STEM_COUNT, CHANNELS, WINDOW_SAMPLES
    )
    result = timed(lambda: infer(torch, model, waveform))
    aggregate = metric(golden, result.value)
    per_stem = {
        stem: metric(golden[:, index], result.value[:, index])
        for index, stem in enumerate(STEM_ORDER)
    }
    return {
        "inputSha256": sha256_file(input_path),
        "goldenSha256": sha256_file(golden_path),
        "timing": result.timing.evidence(),
        "aggregate": aggregate,
        "perStem": per_stem,
        "aggregate80DbGatePassed": aggregate["snrDb"] >= 80.0,
        "uniform80DbGatePassed": aggregate["snrDb"] >= 80.0
        and all(value["snrDb"] >= 80.0 for value in per_stem.values()),
        "absoluteErrorGate1e3Passed": aggregate["maxAbsoluteError"] <= 1e-3,
        "interpretation": (
            "This is the unmodified official safetensors payload evaluated through "
            "the deterministic canonical host boundary. Low-signal stems remain "
            "subject to the separate absolute-error gate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--duration-seconds")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weights", type=Path, default=WEIGHT_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--demucs-root", type=Path, default=DEMUCS_ROOT)
    parser.add_argument("--canonical-manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--skip-fixture-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1 or args.threads > 64:
        raise ValueError("--threads must be in [1, 64]")
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    stage = output_dir.with_name(output_dir.name + ".partial")
    if (stage.exists() or output_dir.exists()) and not args.force:
        raise FileExistsError("Output or staging directory exists; pass --force")
    if stage.exists():
        shutil.rmtree(stage)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    stage.mkdir(parents=True)

    run_started = time.perf_counter_ns()
    source = CanonicalWav(input_path)
    selected_frames = parse_duration_frames(args.duration_seconds, source.frame_count)
    plans = make_window_plans(selected_frames)
    weights = triangle_weights()
    source_hash = timed(lambda: sha256_file(input_path))
    normalization_scan = timed(lambda: source.scan_normalization(selected_frames))
    normalization = normalization_scan.value
    model_load = timed(lambda: load_model(args))
    torch, model, provenance = model_load.value

    manifest_path = args.canonical_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_weight = manifest["provenance"]["canonicalWeight"]
    if canonical_weight["sha256"] != WEIGHT_SHA256:
        raise ValueError("Canonical manifest does not identify the same original weight")

    fixture_validation = None
    if not args.skip_fixture_validation:
        fixture_validation = validate_fixture(torch, model, args.fixture_root.resolve())

    writers: dict[str, wave.Wave_write] = {}
    for stem, path in output_stage_paths(stage).items():
        writer = wave.open(str(path), "wb")
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(BYTES_PER_SAMPLE)
        writer.setframerate(SAMPLE_RATE)
        writers[stem] = writer

    carry = np.zeros((STEM_COUNT, CHANNELS, OVERLAP_SAMPLES), dtype=np.float32)
    carry_weights = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)
    carry_length = 0
    emitted_frames = 0
    stats = {stem: StemStats() for stem in STEM_ORDER}
    window_reports: list[dict[str, Any]] = []
    separation_started = time.perf_counter_ns()
    failure: BaseException | None = None
    try:
        with source.reader() as reader:
            for plan in plans:
                window_started = time.perf_counter_ns()
                window_cpu_started = time.process_time_ns()
                prepare = timed(lambda: read_normalized_window(reader, plan, normalization))
                inference = timed(lambda: infer(torch, model, prepare.value))
                ola = timed(
                    lambda: apply_streaming_ola(
                        inference.value[0],
                        plan,
                        selected_frames,
                        normalization,
                        weights,
                        carry,
                        carry_weights,
                        carry_length,
                    )
                )
                finalized, carry_length = ola.value
                pcm_wall = time.perf_counter_ns()
                pcm_cpu = time.process_time_ns()
                payloads = {
                    stem: encode_pcm16(finalized[index], stats[stem])
                    for index, stem in enumerate(STEM_ORDER)
                }
                pcm_timing = Timing(
                    wall_ms=(time.perf_counter_ns() - pcm_wall) / 1e6,
                    cpu_ms=(time.process_time_ns() - pcm_cpu) / 1e6,
                )
                write_wall = time.perf_counter_ns()
                write_cpu = time.process_time_ns()
                for stem in STEM_ORDER:
                    writers[stem].writeframesraw(payloads[stem])
                write_timing = Timing(
                    wall_ms=(time.perf_counter_ns() - write_wall) / 1e6,
                    cpu_ms=(time.process_time_ns() - write_cpu) / 1e6,
                )
                emitted_frames += int(finalized.shape[-1])
                window_reports.append(
                    {
                        "index": plan.index,
                        "plan": asdict(plan),
                        "finalizedFrames": int(finalized.shape[-1]),
                        "outputFramesAfterWindow": emitted_frames,
                        "stages": {
                            "prepare": prepare.timing.evidence(),
                            "inference": inference.timing.evidence(),
                            "ola": ola.timing.evidence(),
                            "pcm": pcm_timing.evidence(),
                            "write": write_timing.evidence(),
                        },
                        "total": {
                            "wallMs": (time.perf_counter_ns() - window_started) / 1e6,
                            "cpuMs": (time.process_time_ns() - window_cpu_started) / 1e6,
                        },
                        "process": process_snapshot(),
                    }
                )
    except BaseException as exc:
        failure = exc
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except BaseException as exc:
                failure = failure or exc
    separation_wall_ms = (time.perf_counter_ns() - separation_started) / 1e6
    if failure is not None:
        shutil.rmtree(stage, ignore_errors=True)
        raise failure
    if emitted_frames != selected_frames or carry_length != 0:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError(f"OLA emitted {emitted_frames}/{selected_frames}, carry={carry_length}")

    for stem in STEM_ORDER:
        (stage / f"{stem}.wav.partial").replace(stage / f"{stem}.wav")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(output_dir)
    outputs = {}
    for stem in STEM_ORDER:
        path = output_dir / f"{stem}.wav"
        outputs[stem] = {
            **stats[stem].evidence(),
            "path": str(path),
            "byteSize": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    duration = selected_frames / SAMPLE_RATE
    total_wall_ms = (time.perf_counter_ns() - run_started) / 1e6
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "runner": "run_htdemucs_canonical_torch.py",
        "scope": "official-original-safetensors-torch-fp32-reference",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "safetensors": package_version("safetensors"),
            "threads": args.threads,
        },
        "provenance": {
            **provenance,
            "canonicalManifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "modelId": manifest["modelId"],
                "candidateId": manifest["candidateId"],
            },
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "path": str(input_path),
            "byteSize": input_path.stat().st_size,
            "sha256": source_hash.value,
            "sampleRate": SAMPLE_RATE,
            "channelCount": CHANNELS,
            "sourceFrames": source.frame_count,
            "selectedFrames": selected_frames,
            "durationSeconds": duration,
            "selectedPcmSha256": normalization.selected_pcm_sha256,
            "globalMean": float(normalization.mean),
            "sampleStandardDeviation": float(normalization.sample_standard_deviation),
            "normalizationDivisor": float(normalization.divisor),
        },
        "contract": {
            "sampleRate": SAMPLE_RATE,
            "windowSamples": WINDOW_SAMPLES,
            "strideSamples": STRIDE_SAMPLES,
            "overlapSamples": OVERLAP_SAMPLES,
            "overlap": 0.25,
            "transitionPower": 1.0,
            "windowCount": len(plans),
            "tailPadding": "official-demucs-tensor-chunk-centered-zero-pad",
            "tailCrop": "center-crop-to-actual-samples",
            "tailWeightRule": "triangle-prefix",
            "globalNormalization": "stereo-reference-mean-unbiased-std-correction-1",
            "stemOrder": list(STEM_ORDER),
        },
        "modelLoadTiming": model_load.timing.evidence(),
        "sourceHashTiming": source_hash.timing.evidence(),
        "normalizationScanTiming": normalization_scan.timing.evidence(),
        "fixtureValidation": fixture_validation,
        "windows": window_reports,
        "outputs": outputs,
        "processFinal": process_snapshot(),
        "separationWallMs": separation_wall_ms,
        "separationRealtimeFactor": separation_wall_ms / 1000.0 / duration,
        "totalWallMs": total_wall_ms,
        "totalRealtimeFactor": total_wall_ms / 1000.0 / duration,
    }
    write_json_atomic(output_dir / "report.json", report)
    if args.report is not None:
        write_json_atomic(args.report.resolve(), report)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputDir": str(args.output_dir.resolve()),
                "windowCount": report["contract"]["windowCount"],
                "separationWallMs": report["separationWallMs"],
                "separationRealtimeFactor": report["separationRealtimeFactor"],
                "totalWallMs": report["totalWallMs"],
                "totalRealtimeFactor": report["totalRealtimeFactor"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
