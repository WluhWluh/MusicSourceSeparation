#!/usr/bin/env python3
"""Evaluate a static short TFC-TDF window on a complete host audio fixture."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tfc_tdf_short_window import (
    ShortWindowContract,
    load_audio,
    render_with_backend,
    seam_metrics,
    sha256_array,
    sha256_file,
    vocal_residual_proxy,
    waveform_metrics,
    write_flac,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    ROOT / "models" / "tfc-tdf" / "compact-24" / "tfc_tdf_default_vocals_core_fp32_f24.onnx"
)
DEFAULT_REFERENCE = (
    ROOT / "models" / "tfc-tdf" / "default-compact" / "tfc_tdf_default_vocals_core_fp32.onnx"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "tfc-tdf-24-frame-host-evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--candidate-onnx", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--reference-onnx", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trim-hops", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-audio-output", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_backend(path: Path, expected_frames: int, threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path.resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_detail = session.get_inputs()[0]
    output_detail = session.get_outputs()[0]
    expected_shape = [1, 4, 1025, expected_frames]
    actual_input = [int(value) for value in input_detail.shape]
    actual_output = [int(value) for value in output_detail.shape]
    if actual_input != expected_shape or actual_output != expected_shape:
        raise ValueError(
            f"Unexpected ONNX shape for {path}: {actual_input} -> {actual_output}; "
            f"expected {expected_shape}"
        )

    def run(value: np.ndarray) -> np.ndarray:
        return session.run(
            [output_detail.name],
            {input_detail.name: value},
        )[0]

    return run


def main() -> int:
    args = parse_args()
    if any(value < 0 for value in args.trim_hops):
        raise ValueError("trim-hops cannot be negative")
    if not args.trim_hops:
        raise ValueError("at least one trim-hops value is required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_audio, sample_rate = load_audio(args.audio.resolve())
    start = int(round(args.start_seconds * sample_rate))
    if start < 0 or start >= source_audio.shape[0]:
        raise ValueError("start-seconds is outside the source")
    if args.duration_seconds is None:
        end = source_audio.shape[0]
    else:
        requested = int(round(args.duration_seconds * sample_rate))
        end = min(source_audio.shape[0], start + requested)
    audio = np.ascontiguousarray(source_audio[start:end], dtype=np.float32)
    if audio.shape[0] == 0:
        raise ValueError("selected audio range is empty")

    candidate_backend = make_backend(args.candidate_onnx, 24, args.threads)
    reference_backend = make_backend(args.reference_onnx, 128, args.threads)
    reference_contract = ShortWindowContract(
        num_frames=128,
        left_context_hops=5,
        right_context_hops=5,
    )
    reference_render = render_with_backend(
        audio,
        reference_contract,
        reference_backend,
        assembly_mode="isolated",
    )
    reference_vocals = reference_render.audio
    reference_instrumental = audio - reference_vocals
    reference_artifacts: dict[str, object] = {}
    if not args.no_audio_output:
        reference_artifacts = {
            "vocals": write_flac(
                output_dir / "reference-128-isolated-vocals.flac",
                reference_vocals,
                sample_rate,
            ),
            "instrumental": write_flac(
                output_dir / "reference-128-isolated-instrumental.flac",
                reference_instrumental,
                sample_rate,
            ),
        }

    candidates: list[dict[str, object]] = []
    for trim_hops in args.trim_hops:
        contract = ShortWindowContract(
            num_frames=24,
            left_context_hops=trim_hops,
            right_context_hops=trim_hops,
        )
        rendered = render_with_backend(
            audio,
            contract,
            candidate_backend,
            assembly_mode="continuous",
        )
        candidate_vocals = rendered.audio
        candidate_instrumental = audio - candidate_vocals
        candidate_dir = output_dir / f"trim-{trim_hops}"
        artifacts: dict[str, object] = {}
        if not args.no_audio_output:
            artifacts = {
                "vocals": write_flac(
                    candidate_dir / "vocals.flac",
                    candidate_vocals,
                    sample_rate,
                ),
                "instrumental": write_flac(
                    candidate_dir / "instrumental.flac",
                    candidate_instrumental,
                    sample_rate,
                ),
            }
        reference_proxy = vocal_residual_proxy(
            audio,
            reference_vocals,
            reference_vocals,
        )
        candidate_proxy = vocal_residual_proxy(
            audio,
            reference_vocals,
            candidate_vocals,
        )
        candidates.append(
            {
                "trimHops": trim_hops,
                "contract": contract.as_dict(),
                "windowCount": rendered.window_count,
                "elapsedSeconds": rendered.elapsed_seconds,
                "windowsPerOutputMinute": rendered.window_count
                / max(audio.shape[0] / sample_rate / 60.0, 1e-9),
                "audioComparisonsTo128Reference": {
                    "vocals": waveform_metrics(reference_vocals, candidate_vocals),
                    "instrumental": waveform_metrics(
                        reference_instrumental,
                        candidate_instrumental,
                    ),
                },
                "seams": seam_metrics(candidate_vocals, rendered.boundaries),
                "sourceSeams": seam_metrics(audio, rendered.boundaries),
                "vocalResidualProxy": {
                    "candidate": candidate_proxy,
                    "referenceBaseline": reference_proxy,
                    "candidateMinusReferenceInstrumentalProjectionDb": candidate_proxy[
                        "candidateInstrumentalReferenceVocalProjectionDb"
                    ]
                    - reference_proxy[
                        "referenceInstrumentalReferenceVocalProjectionDb"
                    ],
                    "interpretation": (
                        "No isolated vocal ground truth is available; these are "
                        "zero-lag projections against the 128-frame estimate, "
                        "not SDR or a separation-quality claim."
                    ),
                },
                "artifacts": artifacts,
            }
        )

    report = {
        "schemaVersion": 1,
        "candidateId": "tfc-tdf-static-24-frame-host-evaluation@1",
        "status": "completed",
        "sourceAudio": {
            "file": args.audio.name,
            "bytes": args.audio.stat().st_size,
            "sha256": sha256_file(args.audio.resolve()),
            "selectedStartSeconds": args.start_seconds,
            "selectedDurationSeconds": audio.shape[0] / sample_rate,
            "sampleRate": sample_rate,
            "samples": int(audio.shape[0]),
            "decodedPcmFloat32Sha256": sha256_array(audio),
        },
        "models": {
            "candidateOnnx": {
                "file": args.candidate_onnx.name,
                "bytes": args.candidate_onnx.stat().st_size,
                "sha256": sha256_file(args.candidate_onnx.resolve()),
                "shape": [1, 4, 1025, 24],
            },
            "referenceOnnx": {
                "file": args.reference_onnx.name,
                "bytes": args.reference_onnx.stat().st_size,
                "sha256": sha256_file(args.reference_onnx.resolve()),
                "shape": [1, 4, 1025, 128],
            },
        },
        "reference": {
            "contract": reference_contract.as_dict(assembly="isolated-zero-padded"),
            "windowCount": reference_render.window_count,
            "elapsedSeconds": reference_render.elapsed_seconds,
            "assemblyMode": "isolated",
            "artifacts": reference_artifacts,
        },
        "candidates": candidates,
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "soundfile": package_version("soundfile"),
            "onnxruntime": package_version("onnxruntime"),
            "evaluator": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        {
            "trimHops": item["trimHops"],
            "strideHops": item["contract"]["strideHops"],
            "efficiency": item["contract"]["outputEfficiency"],
            "windowCount": item["windowCount"],
            "vocalsSnrTo128Db": item["audioComparisonsTo128Reference"]["vocals"]["snrDb"],
            "seamP95Ratio": item["seams"].get("p95Ratio"),
            "vocalProjectionDeltaDb": item["vocalResidualProxy"][
                "candidateMinusReferenceInstrumentalProjectionDb"
            ],
        }
        for item in candidates
    ]
    print(json.dumps({"report": str(report_path), "candidates": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
