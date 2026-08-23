#!/usr/bin/env python3
"""Rank the twelve private songs against the Inst 3 accompaniment reference.

The private songs do not have isolated ground-truth stems.  This analysis
therefore treats the native Inst 3 instrumental output as the sole reference,
as requested for the listening comparison.  It reports direct waveform match,
relative error, and coherent energy retained from the Inst 3 removed signal.

Language groups are the user-confirmed labels:
three Chinese, two Japanese, and seven English songs.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

import render_inst3_objective_listening as listening
import run_inst3_scale10_density as density


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTINUATION_REPORT = (
    ROOT / "data" / "musdb18-inst3-vr-continuation" / "listening-report.json"
)
DEFAULT_BASELINE_REPORT = (
    ROOT
    / "data"
    / "musdb18-inst3-continuous-baseline-evaluation"
    / "private-render-report.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "inst3-private-language-quality"

LANGUAGE_BY_SONG = {
    "coast-town": "Chinese",
    "lugu-lake": "Chinese",
    "unseen-sea": "Chinese",
    "odd-future": "Japanese",
    "yoru-ni-kakeru": "Japanese",
    "already-gone": "English",
    "chasing-the-wind": "English",
    "coldplay-tove-lo": "English",
    "i-see-fire": "English",
    "imagine": "English",
    "north": "English",
    "traveling-light": "English",
}

VARIANT_ORDER = (
    "H50-pass-50",
    "H50-continuation@step-800",
    "H50-continuation+1",
    "H50-continuation+3",
    "H50-continuation+5",
)
EVENT_MS = (50, 100, 200)
LOWER_IS_BETTER = (
    "relativeErrorToRemovedDb",
    "coherentRetainedRelativeDb",
    "targetErrorRmsDbfs",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-report", type=Path, default=DEFAULT_CONTINUATION_REPORT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--current-variant", default="H50-continuation+5")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def db(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def rms(value: np.ndarray) -> float:
    return math.sqrt(float(np.mean(value.astype(np.float64) ** 2)))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64).reshape(-1)
    right64 = right.astype(np.float64).reshape(-1)
    denominator = math.sqrt(float(np.dot(left64, left64) * np.dot(right64, right64)))
    return float(np.dot(left64, right64) / max(denominator, 1.0e-30))


def load_flac(path: Path) -> np.ndarray:
    value, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 44_100 or value.shape[1] != 2:
        raise ValueError(f"Unexpected audio contract for {path}: {value.shape}, {sample_rate}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite audio: {path}")
    return np.ascontiguousarray(value, dtype=np.float32)


def metric_row(source: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if source.shape != reference.shape or source.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {source.shape}, {reference.shape}, {candidate.shape}")
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    removed = source.astype(np.float64) - reference64
    error = candidate64 - reference64
    reference_power = float(np.sum(reference64 * reference64))
    removed_power = float(np.sum(removed * removed))
    error_power = float(np.sum(error * error))
    dot_removed = float(np.sum(error * removed))
    removed_rms = math.sqrt(removed_power / max(removed.size, 1))
    error_rms = math.sqrt(error_power / max(error.size, 1))
    reference_rms = math.sqrt(reference_power / max(reference.size, 1))
    coefficient = dot_removed / max(removed_power, 1.0e-30)
    coherent_projection_rms = max(coefficient, 0.0) * removed_rms
    return {
        "sourceRmsDbfs": db(rms(source)),
        "inst3RmsDbfs": db(reference_rms),
        "inst3RemovedRmsDbfs": db(removed_rms),
        "targetErrorRmsDbfs": db(error_rms),
        "relativeErrorToRemovedDb": db(error_rms / max(removed_rms, 1.0e-30)),
        "coherentRetainedRmsDbfs": db(coherent_projection_rms),
        "coherentRetainedRelativeDb": db(coherent_projection_rms / max(removed_rms, 1.0e-30)),
        "teacherMatchSnrDb": 10.0 * math.log10(
            max(reference_power, 1.0e-30) / max(error_power, 1.0e-30)
        ),
        "cosineToInst3": cosine(candidate, reference),
        "errorCosineWithInst3Removed": float(
            dot_removed / max(math.sqrt(error_power * removed_power), 1.0e-30)
        ),
        "candidateRmsDeltaVsInst3Db": db(rms(candidate)) - db(reference_rms),
    }


def short_event_row(
    source: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    milliseconds: int,
) -> dict[str, float | int]:
    """Measure local residual against Inst 3 in fixed non-overlapping blocks."""
    block_samples = max(1, round(44_100 * milliseconds / 1000.0))
    block_count = max(1, int(math.ceil(source.shape[0] / block_samples)))
    padding = block_count * block_samples - source.shape[0]
    source_blocks = np.pad(
        source.astype(np.float64, copy=False), ((0, padding), (0, 0))
    ).reshape(block_count, block_samples, 2)
    reference_blocks = np.pad(
        reference.astype(np.float64, copy=False), ((0, padding), (0, 0))
    ).reshape(block_count, block_samples, 2)
    candidate_blocks = np.pad(
        candidate.astype(np.float64, copy=False), ((0, padding), (0, 0))
    ).reshape(block_count, block_samples, 2)
    removed = source_blocks - reference_blocks
    miss = candidate_blocks - reference_blocks
    removed_power = np.sum(removed * removed, axis=(1, 2))
    miss_power = np.sum(miss * miss, axis=(1, 2))
    dot = np.sum(miss * removed, axis=(1, 2))
    removed_rms = np.sqrt(removed_power / float(block_samples * 2))
    miss_rms = np.sqrt(miss_power / float(block_samples * 2))
    positive = np.maximum(
        np.divide(dot, removed_power, out=np.zeros_like(dot), where=removed_power > 1.0e-30),
        0.0,
    ) * removed_rms
    active = removed_rms >= 10.0 ** (-60.0 / 20.0)
    if not np.any(active):
        return {
            "blockCount": block_count,
            "activeBlockCount": 0,
            "missRmsP95Dbfs": -240.0,
            "missRmsMaxDbfs": -240.0,
            "positiveProjectionP95Dbfs": -240.0,
            "positiveProjectionMaxDbfs": -240.0,
            "positiveAboveMinus30Dbfs": 0,
            "positiveAboveMinus35Dbfs": 0,
            "positiveAboveMinus30Fraction": 0.0,
            "positiveAboveMinus35Fraction": 0.0,
        }
    active_miss = miss_rms[active]
    active_positive = positive[active]
    active_count = int(np.count_nonzero(active))
    above30 = int(np.count_nonzero(active_positive >= 10.0 ** (-30.0 / 20.0)))
    above35 = int(np.count_nonzero(active_positive >= 10.0 ** (-35.0 / 20.0)))
    return {
        "blockCount": block_count,
        "activeBlockCount": active_count,
        "missRmsP95Dbfs": db(float(np.percentile(active_miss, 95.0))),
        "missRmsMaxDbfs": db(float(np.max(active_miss))),
        "positiveProjectionP95Dbfs": db(float(np.percentile(active_positive, 95.0))),
        "positiveProjectionMaxDbfs": db(float(np.max(active_positive))),
        "positiveAboveMinus30Dbfs": above30,
        "positiveAboveMinus35Dbfs": above35,
        "positiveAboveMinus30Fraction": above30 / max(active_count, 1),
        "positiveAboveMinus35Fraction": above35 / max(active_count, 1),
    }


def rank_rows(rows: list[dict[str, Any]], metric: str, descending: bool = False) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[metric]), reverse=descending)
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        copy = dict(row)
        copy[f"rankBy{metric[0].upper()}{metric[1:]}"] = rank
        result.append(copy)
    return result


def group_summary(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variants:
        result[variant] = {}
        for group in ("Chinese", "Japanese", "English"):
            selected = [row for row in rows if row["variant"] == variant and row["language"] == group]
            result[variant][group] = {
                "songCount": len(selected),
                "songs": [row["song"] for row in selected],
                "mean": {
                    metric: float(np.mean([row[metric] for row in selected]))
                    for metric in (
                        "relativeErrorToRemovedDb",
                        "coherentRetainedRelativeDb",
                        "targetErrorRmsDbfs",
                        "teacherMatchSnrDb",
                        "cosineToInst3",
                    )
                }
                if selected
                else {},
                "median": {
                    metric: float(np.median([row[metric] for row in selected]))
                    for metric in (
                        "relativeErrorToRemovedDb",
                        "coherentRetainedRelativeDb",
                        "targetErrorRmsDbfs",
                        "teacherMatchSnrDb",
                        "cosineToInst3",
                    )
                }
                if selected
                else {},
            }
            result[variant][group]["shortEvents"] = {
                str(milliseconds): {
                    "mean": {
                        metric: float(
                            np.mean(
                                [
                                    row["shortEvents"][str(milliseconds)][metric]
                                    for row in selected
                                ]
                            )
                        )
                        for metric in (
                            "missRmsP95Dbfs",
                            "missRmsMaxDbfs",
                            "positiveProjectionP95Dbfs",
                            "positiveProjectionMaxDbfs",
                            "positiveAboveMinus30Dbfs",
                            "positiveAboveMinus30Fraction",
                            "positiveAboveMinus35Fraction",
                        )
                    },
                    "median": {
                        metric: float(
                            np.median(
                                [
                                    row["shortEvents"][str(milliseconds)][metric]
                                    for row in selected
                                ]
                            )
                        )
                        for metric in (
                            "missRmsP95Dbfs",
                            "missRmsMaxDbfs",
                            "positiveProjectionP95Dbfs",
                            "positiveProjectionMaxDbfs",
                            "positiveAboveMinus30Dbfs",
                            "positiveAboveMinus30Fraction",
                            "positiveAboveMinus35Fraction",
                        )
                    },
                }
                for milliseconds in EVENT_MS
            } if selected else {}
        english = result[variant]["English"]["mean"]
        english_short = result[variant]["English"]["shortEvents"]
        for group in ("Chinese", "Japanese"):
            values = result[variant][group]["mean"]
            result[variant][group]["deltaVsEnglishMean"] = {
                metric: float(values[metric] - english[metric])
                for metric in values
            }
            result[variant][group]["shortEventsDeltaVsEnglishMean"] = {
                str(milliseconds): {
                    metric: float(
                        result[variant][group]["shortEvents"][str(milliseconds)]["mean"][metric]
                        - english_short[str(milliseconds)]["mean"][metric]
                    )
                    for metric in english_short[str(milliseconds)]["mean"]
                }
                for milliseconds in EVENT_MS
            }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    continuation = json.loads(args.continuation_report.resolve().read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_report.resolve().read_text(encoding="utf-8"))
    variants = [variant for variant in VARIANT_ORDER if variant in continuation["songs"][next(iter(continuation["songs"]))]["variants"]]
    if args.current_variant not in variants:
        raise ValueError(f"Unknown current variant {args.current_variant}; available {variants}")
    rows: list[dict[str, Any]] = []
    source_cache: dict[str, np.ndarray] = {}
    reference_cache: dict[str, np.ndarray] = {}
    for song, song_info in continuation["songs"].items():
        if song not in LANGUAGE_BY_SONG:
            raise ValueError(f"No language label for {song}")
        reference_path = Path(
            baseline["songs"][song]["candidates"]["Inst3-native"]["instrumental"]["file"]
            if "candidates" in baseline["songs"][song]
            else baseline["songs"][song]["outputs"]["instrumental"]["file"]
        )
        source_path = Path(song_info["source"]["file"])
        source_cache[song], _ = listening.load_audio(source_path)
        reference_cache[song] = load_flac(reference_path)
        for variant in variants:
            candidate_path = Path(song_info["variants"][variant]["instrumental"]["file"])
            candidate = load_flac(candidate_path)
            metrics = metric_row(source_cache[song], reference_cache[song], candidate)
            rows.append(
                {
                    "song": song,
                    "language": LANGUAGE_BY_SONG[song],
                    "variant": variant,
                    "candidateFile": str(candidate_path),
                    "referenceFile": str(reference_path),
                    "shortEvents": {
                        str(milliseconds): short_event_row(
                            source_cache[song], reference_cache[song], candidate, milliseconds
                        )
                        for milliseconds in EVENT_MS
                    },
                    **metrics,
                }
            )
    current_rows = [row for row in rows if row["variant"] == args.current_variant]
    # Primary ranking: relative waveform error to Inst 3 removed-content energy.
    primary = sorted(current_rows, key=lambda row: row["relativeErrorToRemovedDb"])
    primary_rank = {row["song"]: index for index, row in enumerate(primary, start=1)}
    coherent_rank = {
        row["song"]: index
        for index, row in enumerate(
            sorted(current_rows, key=lambda row: row["coherentRetainedRelativeDb"]), start=1
        )
    }
    snr_rank = {
        row["song"]: index
        for index, row in enumerate(
            sorted(current_rows, key=lambda row: row["teacherMatchSnrDb"], reverse=True), start=1
        )
    }
    for row in current_rows:
        row["primaryRank"] = primary_rank[row["song"]]
        row["coherentRetainedRank"] = coherent_rank[row["song"]]
        row["teacherMatchSnrRank"] = snr_rank[row["song"]]
        row["rankMean"] = float(
            np.mean([row["primaryRank"], row["coherentRetainedRank"], row["teacherMatchSnrRank"]])
        )
        row["shortEvent100msPositiveMaxDbfs"] = row["shortEvents"]["100"][
            "positiveProjectionMaxDbfs"
        ]
        row["shortEvent100msPositiveP95Dbfs"] = row["shortEvents"]["100"][
            "positiveProjectionP95Dbfs"
        ]
    current_ranked = sorted(current_rows, key=lambda row: (row["rankMean"], row["primaryRank"]))
    for index, row in enumerate(current_ranked, start=1):
        row["overallRank"] = index
    short_ranked = sorted(
        current_rows,
        key=lambda row: (
            row["shortEvent100msPositiveMaxDbfs"],
            row["shortEvent100msPositiveP95Dbfs"],
        ),
    )
    for index, row in enumerate(short_ranked, start=1):
        row["shortEventRank"] = index
    # Model delta relative to the original H50, per song.
    baseline_variant = "H50-pass-50"
    by_key = {(row["song"], row["variant"]): row for row in rows}
    current_deltas: list[dict[str, Any]] = []
    for row in current_rows:
        base = by_key[(row["song"], baseline_variant)]
        current_deltas.append(
            {
                "song": row["song"],
                "language": row["language"],
                "variant": args.current_variant,
                "deltaVsOriginalH50": {
                    metric: float(row[metric] - base[metric])
                    for metric in (
                        "relativeErrorToRemovedDb",
                        "coherentRetainedRelativeDb",
                        "targetErrorRmsDbfs",
                        "teacherMatchSnrDb",
                        "cosineToInst3",
                    )
                },
            }
        )
    report = {
        "schema": "local-inst3-private-language-quality@1",
        "status": "completed",
        "reference": {
            "kind": "Inst3-native instrumental output",
            "interpretation": "Inst 3 is the requested practical reference; this is not isolated-stem ground truth.",
            "metrics": {
                "relativeErrorToRemovedDb": "candidate-minus-Inst3 RMS divided by Inst3 removed-content RMS; lower is closer.",
                "coherentRetainedRelativeDb": "positive projection of candidate-minus-Inst3 onto Inst3 removed content, normalized by removed-content RMS; lower is less retained target content.",
                "teacherMatchSnrDb": "waveform energy SNR against Inst3 instrumental; higher is closer.",
            },
        },
        "languageGroups": {
            "Chinese": [song for song, group in LANGUAGE_BY_SONG.items() if group == "Chinese"],
            "Japanese": [song for song, group in LANGUAGE_BY_SONG.items() if group == "Japanese"],
            "English": [song for song, group in LANGUAGE_BY_SONG.items() if group == "English"],
            "source": "user-confirmed grouping",
        },
        "inputs": {
            "continuationReport": {"file": str(args.continuation_report.resolve()), "sha256": __import__("hashlib").sha256(args.continuation_report.resolve().read_bytes()).hexdigest()},
            "baselineReport": {"file": str(args.baseline_report.resolve()), "sha256": __import__("hashlib").sha256(args.baseline_report.resolve().read_bytes()).hexdigest()},
            "variants": variants,
            "currentVariant": args.current_variant,
        },
        "songs": rows,
        "currentVariantRanking": current_ranked,
        "currentVariantShortEventRanking": short_ranked,
        "currentVariantDeltasVsOriginalH50": current_deltas,
        "groupSummary": group_summary(rows, variants),
        "notes": {
            "confounding": "Language groups are small and confounded with singer, genre, arrangement, mastering, encoding, and vocal style; group differences do not establish a language-causal effect.",
            "ranking": "Overall rank is the mean rank of relative error, coherent retained-content ratio, and Inst3 waveform SNR. Individual metrics are retained for inspection.",
            "source": "Private source decoding uses the existing project audio loader and the same full-song continuous outputs as the listening set.",
        },
        "runtime": {"python": platform.python_version()},
    }
    output_root = args.output_root.resolve()
    json_write(output_root / "private-language-quality-report.json", report)
    print(json.dumps({"status": report["status"], "songs": len(current_ranked), "variants": variants, "report": str((output_root / "private-language-quality-report.json").resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
