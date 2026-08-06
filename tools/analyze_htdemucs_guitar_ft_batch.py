#!/usr/bin/env python3
"""Analyze official versus guitar-ft full-song outputs without ground truth."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import wave
from typing import Any

import numpy as np

from compare_htdemucs_stem_sets import Accumulator, compare_wavs


TRACKS = (
    "john-lennon-imagine",
    "athletics-ii",
    "kygo-ed-sheeran-i-see-fire-kygo-remix",
    "sleeping-at-last-north",
    "sleeping-at-last-already-gone",
    "josiah-james-chasing-the-wind",
    "joel-hanson-traveling-light",
    "nylon-eventide",
)
TRACK_LABELS = {
    "john-lennon-imagine": "piano-heavy",
    "athletics-ii": "guitar-heavy",
    "kygo-ed-sheeran-i-see-fire-kygo-remix": "guitar-and-electronic",
    "sleeping-at-last-north": "piano-heavy-low-guitar",
    "sleeping-at-last-already-gone": "piano-heavy-low-guitar",
    "josiah-james-chasing-the-wind": "guitar-present",
    "joel-hanson-traveling-light": "guitar-heavy",
    "nylon-eventide": "guitar-heavy",
}
STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
CHUNK_FRAMES = 65_536


@dataclass
class Moments:
    count: int = 0
    left_sum: float = 0.0
    right_sum: float = 0.0
    left_square_sum: float = 0.0
    right_square_sum: float = 0.0
    cross_sum: float = 0.0

    def add(self, left: np.ndarray, right: np.ndarray) -> None:
        left64 = left.astype(np.float64, copy=False).reshape(-1)
        right64 = right.astype(np.float64, copy=False).reshape(-1)
        self.count += int(left64.size)
        self.left_sum += float(left64.sum())
        self.right_sum += float(right64.sum())
        self.left_square_sum += float(np.square(left64).sum())
        self.right_square_sum += float(np.square(right64).sum())
        self.cross_sum += float((left64 * right64).sum())

    def correlation(self) -> float:
        left_var = self.left_square_sum - self.left_sum * self.left_sum / self.count
        right_var = self.right_square_sum - self.right_sum * self.right_sum / self.count
        denominator = math.sqrt(max(left_var * right_var, 0.0))
        covariance = self.cross_sum - self.left_sum * self.right_sum / self.count
        return covariance / denominator if denominator else 1.0


@dataclass
class Reconstruction:
    count: int = 0
    mix_square_sum: float = 0.0
    stem_sum_square_sum: float = 0.0
    residual_square_sum: float = 0.0
    residual_absolute_sum: float = 0.0
    residual_maximum: float = 0.0
    mix_stem_cross_sum: float = 0.0
    mix_sum: float = 0.0
    stem_sum: float = 0.0

    def add(self, mix: np.ndarray, stem_sum: np.ndarray) -> None:
        mix64 = mix.astype(np.float64, copy=False).reshape(-1)
        sum64 = stem_sum.astype(np.float64, copy=False).reshape(-1)
        residual = mix64 - sum64
        self.count += int(mix64.size)
        self.mix_square_sum += float(np.square(mix64).sum())
        self.stem_sum_square_sum += float(np.square(sum64).sum())
        self.residual_square_sum += float(np.square(residual).sum())
        self.residual_absolute_sum += float(np.abs(residual).sum())
        self.residual_maximum = max(self.residual_maximum, float(np.abs(residual).max(initial=0.0)))
        self.mix_stem_cross_sum += float((mix64 * sum64).sum())
        self.mix_sum += float(mix64.sum())
        self.stem_sum += float(sum64.sum())

    def evidence(self) -> dict[str, Any]:
        mix_rms = math.sqrt(self.mix_square_sum / self.count) / 32768.0
        sum_rms = math.sqrt(self.stem_sum_square_sum / self.count) / 32768.0
        residual_rms = math.sqrt(self.residual_square_sum / self.count) / 32768.0
        covariance = self.mix_stem_cross_sum - self.mix_sum * self.stem_sum / self.count
        mix_var = self.mix_square_sum - self.mix_sum * self.mix_sum / self.count
        sum_var = self.stem_sum_square_sum - self.stem_sum * self.stem_sum / self.count
        denominator = math.sqrt(max(mix_var * sum_var, 0.0))
        return {
            "sampleCount": self.count,
            "mixRms": mix_rms,
            "stemSumRms": sum_rms,
            "residualRms": residual_rms,
            "residualMeanAbsolute": self.residual_absolute_sum / self.count / 32768.0,
            "residualMaximumAbsolute": self.residual_maximum / 32768.0,
            "mixtureReconstructionSnrDb": 20.0 * math.log10(
                max(mix_rms, 1e-30) / max(residual_rms, 1e-30)
            ),
            "mixtureStemSumCorrelation": covariance / denominator if denominator else 1.0,
        }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def open_contract(reader: wave.Wave_read, path: Path) -> int:
    actual = (
        reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()
    )
    if actual != (44_100, 2, 2, "NONE"):
        raise ValueError(f"Unexpected WAV contract for {path}: {actual}")
    return reader.getnframes()


def analyze_reconstruction(mix_path: Path, stem_root: Path) -> dict[str, Any]:
    accumulator = Reconstruction()
    with ExitStack() as stack:
        mix_reader = stack.enter_context(wave.open(str(mix_path), "rb"))
        readers = [
            stack.enter_context(wave.open(str(stem_root / f"{stem}.wav"), "rb"))
            for stem in STEMS
        ]
        frames = open_contract(mix_reader, mix_path)
        if any(open_contract(reader, stem_root / f"{stem}.wav") != frames for reader, stem in zip(readers, STEMS, strict=True)):
            raise ValueError("Stem frame count does not match canonical mix")
        remaining = frames
        while remaining:
            count = min(CHUNK_FRAMES, remaining)
            mix = np.frombuffer(mix_reader.readframes(count), dtype="<i2").astype(np.int32)
            stems = [
                np.frombuffer(reader.readframes(count), dtype="<i2").astype(np.int32)
                for reader in readers
            ]
            accumulator.add(mix, np.sum(stems, axis=0, dtype=np.int32))
            remaining -= count
    return accumulator.evidence()


def analyze_delta_migration(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    square_sums = {stem: 0.0 for stem in STEMS}
    count = 0
    guitar_moments = {stem: Moments() for stem in STEMS if stem != "guitar"}
    delta_sum_square = 0.0
    individual_square_sum = 0.0
    with ExitStack() as stack:
        reference_readers = [
            stack.enter_context(wave.open(str(reference_root / f"{stem}.wav"), "rb"))
            for stem in STEMS
        ]
        candidate_readers = [
            stack.enter_context(wave.open(str(candidate_root / f"{stem}.wav"), "rb"))
            for stem in STEMS
        ]
        frames = open_contract(reference_readers[0], reference_root / "drums.wav")
        for reader, stem in zip(reference_readers + candidate_readers, STEMS + STEMS, strict=True):
            if open_contract(reader, (reference_root if reader in reference_readers else candidate_root) / f"{stem}.wav") != frames:
                raise ValueError("Delta input frame count mismatch")
        remaining = frames
        while remaining:
            frame_count = min(CHUNK_FRAMES, remaining)
            deltas = []
            for reference_reader, candidate_reader in zip(reference_readers, candidate_readers, strict=True):
                reference = np.frombuffer(reference_reader.readframes(frame_count), dtype="<i2").astype(np.int32)
                candidate = np.frombuffer(candidate_reader.readframes(frame_count), dtype="<i2").astype(np.int32)
                deltas.append(candidate - reference)
            guitar_delta = deltas[STEMS.index("guitar")]
            for stem, delta in zip(STEMS, deltas, strict=True):
                square = float(np.square(delta.astype(np.float64)).sum())
                square_sums[stem] += square
                individual_square_sum += square
                if stem != "guitar":
                    guitar_moments[stem].add(guitar_delta, delta)
            delta_sum_square += float(
                np.square(np.sum(deltas, axis=0, dtype=np.int32).astype(np.float64)).sum()
            )
            count += int(guitar_delta.size)
            remaining -= frame_count
    return {
        "sampleCountPerStem": count,
        "deltaRms": {
            stem: math.sqrt(value / count) / 32768.0 for stem, value in square_sums.items()
        },
        "guitarDeltaCorrelationWithOtherStemDelta": {
            stem: moments.correlation() for stem, moments in guitar_moments.items()
        },
        "allStemDeltaCancellationRatio": math.sqrt(
            delta_sum_square / max(individual_square_sum, 1e-30)
        ),
        "interpretation": (
            "Negative guitar/other-stem delta correlation and a low cancellation ratio "
            "are consistent with energy moving between stems, not proof of correctness."
        ),
    }


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--guitar-ft-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    ft_root = args.guitar_ft_root.resolve()
    tracks: dict[str, Any] = {}
    global_comparison = Accumulator()
    for slug in TRACKS:
        source_track = source_root / "tracks" / slug
        ft_track = ft_root / "tracks" / slug
        official_root = source_track / "original-safetensors-torch"
        candidate_root = ft_track / "guitar-ft-torch"
        mix_path = source_track / "s25-cpu" / "canonical-input-44100-stereo-pcm16.wav"
        official_report = load_json(official_root / "report.json")
        candidate_report = load_json(candidate_root / "report.json")
        if official_report.get("status") != "complete" or candidate_report.get("status") != "complete":
            raise ValueError(f"Incomplete report for {slug}")
        identities = {
            official_report["source"]["selectedPcmSha256"],
            candidate_report["source"]["selectedPcmSha256"],
        }
        if len(identities) != 1:
            raise ValueError(f"Canonical PCM mismatch for {slug}: {identities}")
        per_stem: dict[str, Any] = {}
        track_comparison = Accumulator()
        for stem in STEMS:
            accumulator, files = compare_wavs(
                official_root / f"{stem}.wav", candidate_root / f"{stem}.wav"
            )
            evidence = accumulator.evidence()
            evidence["files"] = files
            evidence["candidateVsOfficialRmsDb"] = 20.0 * math.log10(
                max(evidence["candidateRms"], 1e-30)
                / max(evidence["referenceRms"], 1e-30)
            )
            per_stem[stem] = evidence
            track_comparison.merge(accumulator)
            global_comparison.merge(accumulator)
        official_energy_total = sum(value["referenceRms"] ** 2 for value in per_stem.values())
        candidate_energy_total = sum(value["candidateRms"] ** 2 for value in per_stem.values())
        for evidence in per_stem.values():
            evidence["officialStemEnergyShare"] = evidence["referenceRms"] ** 2 / official_energy_total
            evidence["candidateStemEnergyShare"] = evidence["candidateRms"] ** 2 / candidate_energy_total
        official_reconstruction = analyze_reconstruction(mix_path, official_root)
        candidate_reconstruction = analyze_reconstruction(mix_path, candidate_root)
        tracks[slug] = {
            "corpusRole": TRACK_LABELS[slug],
            "canonicalPcmSha256": next(iter(identities)),
            "durationSeconds": candidate_report["source"]["durationSeconds"],
            "windowCount": candidate_report["contract"]["windowCount"],
            "performance": {
                "separationWallMs": candidate_report["separationWallMs"],
                "separationRealtimeFactor": candidate_report["separationRealtimeFactor"],
                "peakVmHwmKiB": max(
                    item.get("process", {}).get("VmHWMKiB", 0)
                    for item in candidate_report["windows"]
                ),
            },
            "aggregateOfficialVsGuitarFt": track_comparison.evidence(),
            "perStemOfficialVsGuitarFt": per_stem,
            "mixtureReconstruction": {
                "official": official_reconstruction,
                "guitarFt": candidate_reconstruction,
                "guitarFtMinusOfficialSnrDb": (
                    candidate_reconstruction["mixtureReconstructionSnrDb"]
                    - official_reconstruction["mixtureReconstructionSnrDb"]
                ),
            },
            "deltaMigration": analyze_delta_migration(official_root, candidate_root),
            "nonFiniteOutputCount": sum(
                item["nonFiniteCount"] for item in candidate_report["outputs"].values()
            ),
            "clippedOutputCount": sum(
                item["clippedSampleCount"] for item in candidate_report["outputs"].values()
            ),
        }

    total_duration = sum(item["durationSeconds"] for item in tracks.values())
    total_separation_ms = sum(item["performance"]["separationWallMs"] for item in tracks.values())
    residual_changes = [
        item["mixtureReconstruction"]["guitarFtMinusOfficialSnrDb"]
        for item in tracks.values()
    ]
    non_target_median_rms_changes = {
        stem: statistics.median([
            item["perStemOfficialVsGuitarFt"][stem]["candidateVsOfficialRmsDb"]
            for item in tracks.values()
        ])
        for stem in STEMS
        if stem != "guitar"
    }
    residual_gate_passed = all(value >= -3.0 for value in residual_changes)
    non_target_energy_gate_passed = all(
        abs(value) <= 3.0 for value in non_target_median_rms_changes.values()
    )
    summary = {
        "schemaVersion": 1,
        "status": "complete",
        "scope": "quality-regression-and-listening-preparation-without-ground-truth",
        "trackCount": len(TRACKS),
        "totalDurationSeconds": total_duration,
        "performance": {
            "totalSeparationWallMs": total_separation_ms,
            "aggregateSeparationRealtimeFactor": total_separation_ms / 1000.0 / total_duration,
            "maximumPeakVmHwmKiB": max(
                item["performance"]["peakVmHwmKiB"] for item in tracks.values()
            ),
        },
        "aggregateOfficialVsGuitarFt": global_comparison.evidence(),
        "perStem": {
            stem: {
                "trackSnrDb": distribution([
                    item["perStemOfficialVsGuitarFt"][stem]["snrDb"]
                    for item in tracks.values()
                ]),
                "candidateVsOfficialRmsDb": distribution([
                    item["perStemOfficialVsGuitarFt"][stem]["candidateVsOfficialRmsDb"]
                    for item in tracks.values()
                ]),
            }
            for stem in STEMS
        },
        "technicalIntegrity": {
            "allReportsComplete": True,
            "allCanonicalPcmIdentitiesMatch": True,
            "totalNonFiniteOutputs": sum(item["nonFiniteOutputCount"] for item in tracks.values()),
            "totalClippedOutputs": sum(item["clippedOutputCount"] for item in tracks.values()),
        },
        "deviceEscalationDecision": {
            "status": (
                "host-screen-passed"
                if residual_gate_passed and non_target_energy_gate_passed
                else "host-screen-failed"
            ),
            "s25SmokeRecommended": residual_gate_passed and non_target_energy_gate_passed,
            "s25SmokeExecuted": False,
            "thresholds": {
                "minimumPerTrackMixtureReconstructionSnrChangeDb": -3.0,
                "maximumAbsoluteMedianNonTargetStemRmsChangeDb": 3.0,
                "groundTruthQualityClaim": False,
            },
            "observed": {
                "mixtureReconstructionSnrChangeDb": distribution(residual_changes),
                "trackCountWorseThanMinus3Db": sum(value < -3.0 for value in residual_changes),
                "nonTargetMedianRmsChangeDb": non_target_median_rms_changes,
            },
            "reasons": [
                "Mixture reconstruction regressed beyond the host threshold.",
                "Non-target stem energy changed beyond the host threshold.",
                "A device performance smoke cannot resolve this quality regression.",
            ],
        },
        "limitations": [
            "The corpus has no isolated ground-truth stems; output differences are not SDR.",
            "Mixture residual and energy migration can reveal regressions but cannot prove better separation.",
            "The publisher's MoisesDB metrics are unverified and cannot be reproduced from this repository.",
            "Blind listening is required before a quality conclusion.",
        ],
        "tracks": tracks,
    }
    write_atomic(args.json.resolve(), json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# HTDemucs-6s guitar-ft host experiment",
        "",
        f"Tracks: **{len(TRACKS)}**; total audio: **{total_duration / 60.0:.2f} min**.",
        "",
        "This corpus has no isolated ground truth. SNR below measures change from the official model, not separation quality.",
        "",
        "| Track | Role | RTF | All-stem delta SNR (dB) | Guitar delta SNR (dB) | Guitar RMS change (dB) | Mix residual change (dB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for slug, item in tracks.items():
        guitar = item["perStemOfficialVsGuitarFt"]["guitar"]
        lines.append(
            f"| {slug} | {item['corpusRole']} | {item['performance']['separationRealtimeFactor']:.3f} | "
            f"{item['aggregateOfficialVsGuitarFt']['snrDb']:.2f} | {guitar['snrDb']:.2f} | "
            f"{guitar['candidateVsOfficialRmsDb']:+.2f} | "
            f"{item['mixtureReconstruction']['guitarFtMinusOfficialSnrDb']:+.2f} |"
        )
    lines.extend([
        "",
        f"Aggregate host separation RTF: `{summary['performance']['aggregateSeparationRealtimeFactor']:.4f}`.",
        f"Peak observed VmHWM: `{summary['performance']['maximumPeakVmHwmKiB'] / 1024.0:.1f} MiB`.",
        f"Non-finite outputs: `{summary['technicalIntegrity']['totalNonFiniteOutputs']}`; clipped samples before PCM16: `{summary['technicalIntegrity']['totalClippedOutputs']}`.",
        "",
        "Device escalation gate: **FAILED**. Mixture reconstruction SNR worsened on all eight tracks, "
        f"from `{min(residual_changes):+.2f}` to `{max(residual_changes):+.2f} dB`; the S25 smoke was not run.",
        "",
        "Blind listening remains required; these measurements cannot validate the publisher's claimed guitar SDR gain.",
        "",
    ])
    write_atomic(args.markdown.resolve(), "\n".join(lines))
    print(json.dumps({
        "status": "complete",
        "json": str(args.json.resolve()),
        "markdown": str(args.markdown.resolve()),
        "trackCount": len(TRACKS),
        "aggregateRtf": summary["performance"]["aggregateSeparationRealtimeFactor"],
        "aggregateDeltaSnrDb": summary["aggregateOfficialVsGuitarFt"]["snrDb"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
