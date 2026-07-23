#!/usr/bin/env python3
"""Compare exported Android Phase 7 stems with desktop ORT references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


def read_pcm16(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    raw = np.asarray(raw)
    if raw.dtype != np.int16:
        raise ValueError(f"{path} is {raw.dtype}, expected signed PCM16")
    if raw.ndim == 1:
        raw = raw[:, None]
    return sample_rate, raw, raw.astype(np.float32) / 32768.0


def compare_stem(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape[1] != expected.shape[1]:
        raise ValueError(f"channel mismatch: {actual.shape} vs {expected.shape}")
    frame_count = min(len(actual), len(expected))
    actual_raw = actual[:frame_count].astype(np.int64)
    expected_raw = expected[:frame_count].astype(np.int64)
    delta_raw = actual_raw - expected_raw
    delta = delta_raw.astype(np.float64) / 32768.0
    expected_float = expected_raw.astype(np.float64) / 32768.0
    actual_float = actual_raw.astype(np.float64) / 32768.0
    error_energy = float(np.sum(delta * delta, dtype=np.float64))
    signal_energy = float(np.sum(expected_float * expected_float, dtype=np.float64))
    snr_db = math.inf if error_energy == 0 else 10.0 * math.log10(
        signal_energy / error_energy
    )
    cosine = float(
        np.sum(actual_float * expected_float, dtype=np.float64)
        / (
            np.linalg.norm(actual_float)
            * np.linalg.norm(expected_float)
        )
    )
    return {
        "actualFrames": len(actual),
        "expectedFrames": len(expected),
        "frameDelta": len(actual) - len(expected),
        "channels": actual.shape[1],
        "maxPcm16Delta": int(np.max(np.abs(delta_raw))) if frame_count else 0,
        "quantizedEquivalent": bool(np.all(np.abs(delta_raw) <= 1)),
        "maxAbsError": float(np.max(np.abs(delta))) if frame_count else 0.0,
        "meanAbsError": float(np.mean(np.abs(delta))) if frame_count else 0.0,
        "snrDb": snr_db,
        "cosineSimilarity": cosine,
    }


def reconstruction_metrics(
    actual_vocals: np.ndarray,
    actual_instrumental: np.ndarray,
    expected_vocals: np.ndarray,
    expected_instrumental: np.ndarray,
    source: np.ndarray,
) -> dict[str, Any]:
    frame_count = min(
        len(actual_vocals),
        len(actual_instrumental),
        len(expected_vocals),
        len(expected_instrumental),
        len(source),
    )
    actual_error = (
        actual_vocals[:frame_count].astype(np.float64) / 32768.0
        + actual_instrumental[:frame_count].astype(np.float64) / 32768.0
        - source[:frame_count].astype(np.float64) / 32768.0
    )
    expected_error = (
        expected_vocals[:frame_count].astype(np.float64) / 32768.0
        + expected_instrumental[:frame_count].astype(np.float64) / 32768.0
        - source[:frame_count].astype(np.float64) / 32768.0
    )
    return {
        "frames": frame_count,
        "actualMaxAbsError": float(np.max(np.abs(actual_error))),
        "actualMeanAbsError": float(np.mean(np.abs(actual_error))),
        "actualRmsError": float(np.sqrt(np.mean(actual_error * actual_error))),
        "referenceMaxAbsError": float(np.max(np.abs(expected_error))),
        "referenceMeanAbsError": float(np.mean(np.abs(expected_error))),
        "referenceRmsError": float(
            np.sqrt(np.mean(expected_error * expected_error))
        ),
    }


def join_discontinuity(
    actual: np.ndarray,
    expected: np.ndarray,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    boundaries: list[int] = []
    if manifest:
        segments = manifest.get("segmentPlan", {}).get("segments", [])
        boundaries = [
            int(segment["playbackEndFrame"])
            for segment in segments[:-1]
            if "playbackEndFrame" in segment
        ]
    values: list[float] = []
    for boundary in boundaries:
        if boundary <= 0 or boundary >= min(len(actual), len(expected)):
            continue
        actual_step = (
            actual[boundary].astype(np.float64)
            - actual[boundary - 1].astype(np.float64)
        ) / 32768.0
        expected_step = (
            expected[boundary].astype(np.float64)
            - expected[boundary - 1].astype(np.float64)
        ) / 32768.0
        values.append(float(np.max(np.abs(actual_step - expected_step))))
    return {
        "boundaryCount": len(boundaries),
        "maxJoinDiscontinuity": max(values, default=0.0),
        "boundaryValues": values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual-prefix", default="")
    parser.add_argument("--reference-prefix", default="_-_Coast_Town__decoded")
    parser.add_argument("--fixture-id", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_dir = args.actual_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    actual_paths = {
        "vocals": actual_dir / f"{args.actual_prefix}vocals.wav",
        "instrumental": actual_dir / f"{args.actual_prefix}instrumental.wav",
    }
    reference_paths = {
        "vocals": reference_dir / f"{args.reference_prefix}_vocals.wav",
        "instrumental": reference_dir / f"{args.reference_prefix}_instrumental.wav",
    }

    actual: dict[str, tuple[int, np.ndarray]] = {}
    expected: dict[str, tuple[int, np.ndarray]] = {}
    stem_reports: dict[str, dict[str, Any]] = {}
    for semantic in ("vocals", "instrumental"):
        actual_rate, actual_raw, _ = read_pcm16(actual_paths[semantic])
        expected_rate, expected_raw, _ = read_pcm16(reference_paths[semantic])
        if actual_rate != expected_rate:
            raise ValueError(
                f"{semantic} sample rate mismatch: {actual_rate} vs {expected_rate}"
            )
        actual[semantic] = (actual_rate, actual_raw)
        expected[semantic] = (expected_rate, expected_raw)
        stem_reports[semantic] = compare_stem(actual_raw, expected_raw)

    source_rate = None
    source_raw = None
    reconstruction = None
    if args.source:
        source_rate, source_raw, _ = read_pcm16(args.source.resolve())
        if source_rate != actual["vocals"][0]:
            raise ValueError(f"source sample rate mismatch: {source_rate}")
        reconstruction = reconstruction_metrics(
            actual["vocals"][1],
            actual["instrumental"][1],
            expected["vocals"][1],
            expected["instrumental"][1],
            source_raw,
        )

    manifest = None
    if args.cache_manifest:
        manifest = json.loads(args.cache_manifest.resolve().read_text(encoding="utf-8"))

    joins = {
        semantic: join_discontinuity(
            actual[semantic][1], expected[semantic][1], manifest
        )
        for semantic in ("vocals", "instrumental")
    }
    result = {
        "schemaVersion": "phase7-audio-compare-v1",
        "fixtureId": args.fixture_id,
        "actual": {"directory": str(actual_dir), "sampleRate": actual["vocals"][0]},
        "reference": {"directory": str(reference_dir), "sampleRate": expected["vocals"][0]},
        "stems": stem_reports,
        "joins": joins,
        "reconstruction": reconstruction,
        "summary": {
            "allFrameCountsExact": all(
                report["frameDelta"] == 0 for report in stem_reports.values()
            ),
            "allQuantizedEquivalent": all(
                report["quantizedEquivalent"] for report in stem_reports.values()
            ),
            "maxAbsError": max(
                report["maxAbsError"] for report in stem_reports.values()
            ),
            "minimumSnrDb": min(report["snrDb"] for report in stem_reports.values()),
            "maxJoinDiscontinuity": max(
                report["maxJoinDiscontinuity"] for report in joins.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
