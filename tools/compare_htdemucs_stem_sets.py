#!/usr/bin/env python3
"""Compare two or more full-length PCM16 HTDemucs stem sets without loading them whole."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import wave
from typing import Any

import numpy as np


STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
SAMPLE_RATE = 44_100
CHANNELS = 2
SAMPLE_WIDTH = 2
CHUNK_FRAMES = 65_536


@dataclass
class Accumulator:
    count: int = 0
    mismatch_count: int = 0
    reference_sum: float = 0.0
    candidate_sum: float = 0.0
    reference_square_sum: float = 0.0
    candidate_square_sum: float = 0.0
    cross_sum: float = 0.0
    error_square_sum: float = 0.0
    absolute_error_sum: float = 0.0
    maximum_absolute_error_lsb: int = 0

    def add(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        reference_i32 = reference.astype(np.int32, copy=False)
        candidate_i32 = candidate.astype(np.int32, copy=False)
        error = candidate_i32 - reference_i32
        reference_f64 = reference_i32.astype(np.float64)
        candidate_f64 = candidate_i32.astype(np.float64)
        error_f64 = error.astype(np.float64)
        self.count += int(reference.size)
        self.mismatch_count += int(np.count_nonzero(error))
        self.reference_sum += float(reference_f64.sum())
        self.candidate_sum += float(candidate_f64.sum())
        self.reference_square_sum += float(np.square(reference_f64).sum())
        self.candidate_square_sum += float(np.square(candidate_f64).sum())
        self.cross_sum += float((reference_f64 * candidate_f64).sum())
        self.error_square_sum += float(np.square(error_f64).sum())
        self.absolute_error_sum += float(np.abs(error_f64).sum())
        self.maximum_absolute_error_lsb = max(
            self.maximum_absolute_error_lsb,
            int(np.abs(error).max(initial=0)),
        )

    def merge(self, other: "Accumulator") -> None:
        self.count += other.count
        self.mismatch_count += other.mismatch_count
        self.reference_sum += other.reference_sum
        self.candidate_sum += other.candidate_sum
        self.reference_square_sum += other.reference_square_sum
        self.candidate_square_sum += other.candidate_square_sum
        self.cross_sum += other.cross_sum
        self.error_square_sum += other.error_square_sum
        self.absolute_error_sum += other.absolute_error_sum
        self.maximum_absolute_error_lsb = max(
            self.maximum_absolute_error_lsb,
            other.maximum_absolute_error_lsb,
        )

    def evidence(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("Cannot summarize an empty comparison")
        reference_rms_lsb = math.sqrt(self.reference_square_sum / self.count)
        candidate_rms_lsb = math.sqrt(self.candidate_square_sum / self.count)
        error_rms_lsb = math.sqrt(self.error_square_sum / self.count)
        covariance = self.cross_sum - (
            self.reference_sum * self.candidate_sum / self.count
        )
        reference_variance = self.reference_square_sum - (
            self.reference_sum * self.reference_sum / self.count
        )
        candidate_variance = self.candidate_square_sum - (
            self.candidate_sum * self.candidate_sum / self.count
        )
        denominator = math.sqrt(max(reference_variance * candidate_variance, 0.0))
        correlation = covariance / denominator if denominator > 0 else 1.0
        return {
            "sampleCount": self.count,
            "finite": True,
            "differentSampleCount": self.mismatch_count,
            "differentSampleFraction": self.mismatch_count / self.count,
            "referenceRms": reference_rms_lsb / 32768.0,
            "candidateRms": candidate_rms_lsb / 32768.0,
            "meanAbsoluteError": self.absolute_error_sum / self.count / 32768.0,
            "meanAbsoluteErrorLsb": self.absolute_error_sum / self.count,
            "rootMeanSquareError": error_rms_lsb / 32768.0,
            "rootMeanSquareErrorLsb": error_rms_lsb,
            "maximumAbsoluteError": self.maximum_absolute_error_lsb / 32768.0,
            "maximumAbsoluteErrorLsb": self.maximum_absolute_error_lsb,
            "snrDb": 20.0
            * math.log10(max(reference_rms_lsb, 1e-30) / max(error_rms_lsb, 1e-30)),
            "correlation": correlation,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_binding(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, Path(path_text)


def parse_stems(value: str) -> tuple[str, ...]:
    stems = tuple(part.strip() for part in value.split(",") if part.strip())
    if not stems:
        raise argparse.ArgumentTypeError("Provide at least one stem")
    if len(stems) != len(set(stems)):
        raise argparse.ArgumentTypeError("Stem names must be unique")
    unknown = sorted(set(stems) - set(STEMS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stems: {', '.join(unknown)}")
    return stems


def wav_contract(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        evidence = {
            "sampleRate": reader.getframerate(),
            "channelCount": reader.getnchannels(),
            "sampleWidthBytes": reader.getsampwidth(),
            "frameCount": reader.getnframes(),
            "compressionType": reader.getcomptype(),
        }
    expected = (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, "NONE")
    actual = (
        evidence["sampleRate"],
        evidence["channelCount"],
        evidence["sampleWidthBytes"],
        evidence["compressionType"],
    )
    if actual != expected:
        raise ValueError(f"Unexpected WAV contract for {path}: {actual}")
    return evidence


def compare_wavs(reference_path: Path, candidate_path: Path) -> tuple[Accumulator, dict[str, Any]]:
    reference_contract = wav_contract(reference_path)
    candidate_contract = wav_contract(candidate_path)
    if reference_contract != candidate_contract:
        raise ValueError(
            f"WAV contract mismatch: {reference_path} vs {candidate_path}"
        )
    accumulator = Accumulator()
    with wave.open(str(reference_path), "rb") as reference_reader, wave.open(
        str(candidate_path), "rb"
    ) as candidate_reader:
        remaining = reference_contract["frameCount"]
        while remaining:
            frames = min(CHUNK_FRAMES, remaining)
            reference_bytes = reference_reader.readframes(frames)
            candidate_bytes = candidate_reader.readframes(frames)
            expected_bytes = frames * CHANNELS * SAMPLE_WIDTH
            if len(reference_bytes) != expected_bytes or len(candidate_bytes) != expected_bytes:
                raise ValueError("Short PCM read during comparison")
            reference = np.frombuffer(reference_bytes, dtype="<i2")
            candidate = np.frombuffer(candidate_bytes, dtype="<i2")
            accumulator.add(reference, candidate)
            remaining -= frames
    return accumulator, {
        "contract": reference_contract,
        "reference": {
            "path": str(reference_path.resolve()),
            "byteSize": reference_path.stat().st_size,
            "sha256": sha256_file(reference_path),
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "byteSize": candidate_path.stat().st_size,
            "sha256": sha256_file(candidate_path),
        },
    }


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_source(report: dict[str, Any]) -> dict[str, Any]:
    source = report["source"]
    canonical = source.get("canonicalPcm", {})
    return {
        "selectedFrames": source.get("selectedFrames"),
        "durationSeconds": source.get("durationSeconds"),
        "selectedPcmSha256": source.get("selectedPcmSha256"),
        "canonicalPcmSha256": canonical.get("pcmSha256"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", type=parse_binding, required=True)
    parser.add_argument("--set-report", action="append", type=parse_binding, default=[])
    parser.add_argument("--primary-reference", required=True)
    parser.add_argument(
        "--stems",
        type=parse_stems,
        default=STEMS,
        help="Comma-separated stem order (default: six-stem HTDemucs order)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stems = args.stems
    sets = dict(args.set)
    reports = dict(args.set_report)
    if len(sets) < 2 or len(sets) != len(args.set):
        raise ValueError("Provide at least two uniquely named stem sets")
    if args.primary_reference not in sets:
        raise ValueError("--primary-reference must name one of the stem sets")
    for name, root in sets.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing stem set {name}: {root}")
        for stem in stems:
            if not (root / f"{stem}.wav").is_file():
                raise FileNotFoundError(root / f"{stem}.wav")

    report_evidence = {}
    source_identities = {}
    for name, path in reports.items():
        report = load_report(path)
        report_evidence[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "status": report.get("status"),
            "source": report_source(report),
        }
        source_identities[name] = report_evidence[name]["source"]["selectedPcmSha256"]
    non_null_identities = {value for value in source_identities.values() if value}
    if len(non_null_identities) > 1:
        raise ValueError(f"Stem sets do not share one selected PCM SHA: {source_identities}")

    primary = args.primary_reference
    pair_order = [(primary, name) for name in sets if name != primary]
    non_primary = [name for name in sets if name != primary]
    pair_order.extend(itertools.combinations(non_primary, 2))
    pairs = {}
    for reference_name, candidate_name in pair_order:
        aggregate = Accumulator()
        stem_results = {}
        for stem in stems:
            stem_accumulator, files = compare_wavs(
                sets[reference_name] / f"{stem}.wav",
                sets[candidate_name] / f"{stem}.wav",
            )
            aggregate.merge(stem_accumulator)
            stem_results[stem] = {
                **stem_accumulator.evidence(),
                "files": files,
            }
        pair_id = f"{candidate_name}_vs_{reference_name}"
        pairs[pair_id] = {
            "reference": reference_name,
            "candidate": candidate_name,
            "aggregate": aggregate.evidence(),
            "perStem": stem_results,
        }

    result = {
        "schemaVersion": 1,
        "status": "complete",
        "primaryReference": primary,
        "stemOrder": list(stems),
        "sets": {name: str(path.resolve()) for name, path in sets.items()},
        "reports": report_evidence,
        "sharedSelectedPcmSha256": next(iter(non_null_identities), None),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(args.output)
    print(json.dumps({
        "status": "complete",
        "output": str(args.output.resolve()),
        "pairs": {
            name: {
                "snrDb": value["aggregate"]["snrDb"],
                "maxAbsoluteErrorLsb": value["aggregate"]["maximumAbsoluteErrorLsb"],
                "correlation": value["aggregate"]["correlation"],
            }
            for name, value in pairs.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
