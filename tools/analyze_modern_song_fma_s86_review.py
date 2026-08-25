#!/usr/bin/env python3
"""Compare the all-FMA S86 review with the earlier H50 review rounds.

The current and earlier files do not contain the same candidate locations:
the current five events per song were selected after rendering S86, while the
older files were selected from H50.  This report therefore has three separate
views: raw mark rates, same-song aggregate rates, and one-to-one nearest-time
matches with an explicit tolerance.  None of those views is treated as an
objective vocal ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "human-review-template.csv"
DEFAULT_CURRENT_REPORT = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "event-listening-report.json"
DEFAULT_PRIOR_BATCH1 = ROOT / "data" / "modern-song-inst3-event-listening" / "human-review-template.csv"
DEFAULT_PRIOR_BATCH1_REPORT = ROOT / "data" / "modern-song-inst3-event-listening" / "event-listening-report.json"
DEFAULT_PRIOR_BATCH2 = ROOT / "data" / "modern-song-batch2-inst3-event-listening" / "human-review-template.csv"
DEFAULT_PRIOR_BATCH2_REPORT = ROOT / "data" / "modern-song-batch2-inst3-event-listening" / "event-listening-report.json"
DEFAULT_PRIOR_BATCH2_FULL = ROOT / "data" / "modern-song-batch2-full-event-listening" / "human-review-template.csv"
DEFAULT_PRIOR_BATCH2_FULL_REPORT = ROOT / "data" / "modern-song-batch2-full-event-listening" / "event-listening-report.json"
DEFAULT_PRESSURE_REPORT = ROOT / "data" / "modern-song-s-event-pressure" / "reports" / "pressure-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "review-analysis.json"

MARKS = ("S", "R", "K", "I")
AGGRESSIVE = frozenset(("S", "R"))
METRICS = (50, 100, 200)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-csv", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--current-report", type=Path, default=DEFAULT_CURRENT_REPORT)
    parser.add_argument("--prior-batch1-csv", type=Path, default=DEFAULT_PRIOR_BATCH1)
    parser.add_argument("--prior-batch1-report", type=Path, default=DEFAULT_PRIOR_BATCH1_REPORT)
    parser.add_argument("--prior-batch2-csv", type=Path, default=DEFAULT_PRIOR_BATCH2)
    parser.add_argument("--prior-batch2-report", type=Path, default=DEFAULT_PRIOR_BATCH2_REPORT)
    parser.add_argument("--prior-batch2-full-csv", type=Path, default=DEFAULT_PRIOR_BATCH2_FULL)
    parser.add_argument("--prior-batch2-full-report", type=Path, default=DEFAULT_PRIOR_BATCH2_FULL_REPORT)
    parser.add_argument("--pressure-report", type=Path, default=DEFAULT_PRESSURE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--match-tolerance-seconds", type=float, default=0.5)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def stats(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": ordered[max(0, math.ceil(len(ordered) * 0.10) - 1)],
        "p90": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.90) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def load_rows(path: Path, default_batch: str | None = None) -> list[dict[str, Any]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"CSV is empty: {path}")
    required = {"mark", "eventId", "sourceOrder", "centerSeconds"}
    missing = required - set(source_rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        row["mark"] = (source.get("mark") or "").strip().upper()
        if row["mark"] not in MARKS:
            raise ValueError(f"Invalid mark {row['mark']!r} in {path}: {source.get('eventId')}")
        row["batch"] = (source.get("sourceBatch") or default_batch or "").strip()
        if not row["batch"]:
            raise ValueError(f"Cannot determine source batch for {source.get('eventId')}")
        row["order"] = int(source["sourceOrder"])
        row["center"] = float(source["centerSeconds"])
        row["role"] = (source.get("priorSplitRole") or source.get("splitRole") or "").strip()
        row["categoryValue"] = (source.get("category") or "").strip() or "<metadata-blank>"
        for milliseconds in METRICS:
            field = f"positiveProjection{milliseconds}Dbfs"
            if source.get(field, "") not in (None, ""):
                row[f"projection{milliseconds}"] = float(source[field])
        rows.append(row)
    return rows


def song_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["batch"]), int(row["order"]))


def group_by(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def mark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["mark"] for row in rows)
    return {
        "eventCount": len(rows),
        "songCount": len({song_key(row) for row in rows}),
        "marks": {mark: counts[mark] for mark in MARKS},
        "fractions": {mark: counts[mark] / len(rows) for mark in MARKS},
        "aggressiveCountSPlusR": counts["S"] + counts["R"],
        "aggressiveFractionSPlusR": (counts["S"] + counts["R"]) / len(rows),
    }


def grouped_mark_summary(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    groups = group_by(rows, lambda row: row.get(key_name) or "<metadata-blank>")
    result = []
    for name, values in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = mark_summary(values)
        summary["name"] = str(name)
        result.append(summary)
    return result


def song_fraction_comparison(
    current: dict[tuple[str, int], list[dict[str, Any]]],
    prior: dict[tuple[str, int], list[dict[str, Any]]],
    shared: list[tuple[str, int]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"sharedSongCount": len(shared)}
    for label, predicate in (
        ("S", lambda mark: mark == "S"),
        ("R", lambda mark: mark == "R"),
        ("aggressiveSPlusR", lambda mark: mark in AGGRESSIVE),
        ("K", lambda mark: mark == "K"),
        ("I", lambda mark: mark == "I"),
    ):
        current_values = [
            sum(predicate(row["mark"]) for row in current[key]) / len(current[key])
            for key in shared
        ]
        prior_values = [
            sum(predicate(row["mark"]) for row in prior[key]) / len(prior[key])
            for key in shared
        ]
        deltas = [a - b for a, b in zip(current_values, prior_values)]
        result[label] = {
            "currentMean": statistics.fmean(current_values),
            "currentMedian": statistics.median(current_values),
            "priorMean": statistics.fmean(prior_values),
            "priorMedian": statistics.median(prior_values),
            "meanDeltaCurrentMinusPrior": statistics.fmean(deltas),
            "songsCurrentHigher": sum(delta > 1.0e-12 for delta in deltas),
            "songsPriorHigher": sum(delta < -1.0e-12 for delta in deltas),
            "songsEqual": sum(abs(delta) <= 1.0e-12 for delta in deltas),
        }
    return result


def nearest_matches(
    current: dict[tuple[str, int], list[dict[str, Any]]],
    prior: dict[tuple[str, int], list[dict[str, Any]]],
    shared: list[tuple[str, int]],
    tolerance: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for key in sorted(shared):
        left = current[key]
        right = prior[key]
        candidates = sorted(
            (
                abs(float(a["center"]) - float(b["center"])),
                i,
                j,
            )
            for i, a in enumerate(left)
            for j, b in enumerate(right)
        )
        used_left: set[int] = set()
        used_right: set[int] = set()
        for distance, i, j in candidates:
            if distance > tolerance or i in used_left or j in used_right:
                continue
            used_left.add(i)
            used_right.add(j)
            matches.append(
                {
                    "song": {"batch": key[0], "sourceOrder": key[1]},
                    "current": left[i],
                    "prior": right[j],
                    "centerDistanceSeconds": distance,
                }
            )
    return matches


def binary_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    p_a = sum(a for a, _ in pairs) / len(pairs)
    p_b = sum(b for _, b in pairs) / len(pairs)
    expected = p_a * p_b + (1.0 - p_a) * (1.0 - p_b)
    return 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def match_summary(matches: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = Counter((item["current"]["mark"], item["prior"]["mark"]) for item in matches)
    aggressive_pairs = [
        (item["current"]["mark"] in AGGRESSIVE, item["prior"]["mark"] in AGGRESSIVE)
        for item in matches
    ]
    s_pairs = [(item["current"]["mark"] == "S", item["prior"]["mark"] == "S") for item in matches]
    i_pairs = [(item["current"]["mark"] == "I", item["prior"]["mark"] == "I") for item in matches]
    return {
        "matchedEventCount": len(matches),
        "matchedSongCount": len({(item["song"]["batch"], item["song"]["sourceOrder"]) for item in matches}),
        "centerDistanceSeconds": stats(item["centerDistanceSeconds"] for item in matches),
        "markMatrixCurrentToPrior": {
            f"{current}->{prior}": matrix[current, prior]
            for current in MARKS
            for prior in MARKS
            if matrix[current, prior]
        },
        "exactMarkAgreementFraction": (
            sum(matrix[mark, mark] for mark in MARKS) / len(matches) if matches else None
        ),
        "aggressiveBinary": {
            "bothAggressive": sum(a and b for a, b in aggressive_pairs),
            "currentOnlyAggressive": sum(a and not b for a, b in aggressive_pairs),
            "priorOnlyAggressive": sum(not a and b for a, b in aggressive_pairs),
            "neitherAggressive": sum(not a and not b for a, b in aggressive_pairs),
            "agreementFraction": (
                sum(a == b for a, b in aggressive_pairs) / len(aggressive_pairs)
                if aggressive_pairs
                else None
            ),
            "cohenKappa": binary_kappa(aggressive_pairs),
        },
        "sBinary": {
            "agreementFraction": sum(a == b for a, b in s_pairs) / len(s_pairs) if s_pairs else None,
            "cohenKappa": binary_kappa(s_pairs),
        },
        "iBinary": {
            "agreementFraction": sum(a == b for a, b in i_pairs) / len(i_pairs) if i_pairs else None,
            "cohenKappa": binary_kappa(i_pairs),
        },
    }


def current_pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    songs = group_by(rows, song_key)
    role_events: dict[str, dict[str, int]] = {}
    for role, values in group_by(rows, lambda row: row["role"]).items():
        counts = Counter(row["mark"] for row in values)
        role_events[str(role)] = {mark: counts[mark] for mark in MARKS}

    def song_pool(max_i: int | None = None, require_i_less_than_aggressive: bool = False) -> dict[str, Any]:
        selected_songs: list[tuple[str, int]] = []
        selected_events = 0
        selected_aggressive = 0
        by_role = Counter()
        for key, values in songs.items():
            counts = Counter(row["mark"] for row in values)
            aggressive = counts["S"] + counts["R"]
            if aggressive == 0:
                continue
            if max_i is not None and counts["I"] > max_i:
                continue
            if require_i_less_than_aggressive and counts["I"] >= aggressive:
                continue
            selected_songs.append(key)
            selected_events += len(values)
            selected_aggressive += aggressive
            by_role[values[0]["role"]] += 1
        return {
            "songCount": len(selected_songs),
            "eventCount": selected_events,
            "aggressiveEventCount": selected_aggressive,
            "byRole": dict(by_role),
            "songs": [
                {"batch": key[0], "sourceOrder": key[1]}
                for key in sorted(selected_songs)
            ],
        }

    return {
        "roleEventMarks": role_events,
        "strictNoI": song_pool(max_i=0),
        "allowAtMostOneI": song_pool(max_i=1),
        "IIsNotMajority": song_pool(require_i_less_than_aggressive=True),
        "eventLevelAggressiveByRole": {
            role: sum(values[mark] for mark in AGGRESSIVE)
            for role, values in role_events.items()
        },
    }


def mark_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mark in MARKS:
        values = [row for row in rows if row["mark"] == mark]
        result[mark] = {
            "count": len(values),
            **{
                f"positiveProjection{milliseconds}Dbfs": stats(
                    row[f"projection{milliseconds}"]
                    for row in values
                    if f"projection{milliseconds}" in row
                )
                for milliseconds in METRICS
            },
        }
    return result


def pressure_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = read_json(path)
    evaluations = report.get("evaluations", [])
    result = []
    for item in evaluations:
        delta = item.get("deltaVsH50", {})
        result.append(
            {
                "pass": item.get("pass"),
                "positiveProjectionP95DeltaMeanDb": delta.get("positiveProjectionP95Dbfs", {}).get("meanDeltaDb"),
                "positiveProjectionP95DeltaMedianDb": delta.get("positiveProjectionP95Dbfs", {}).get("medianDeltaDb"),
                "positiveProjectionMaxDeltaMeanDb": delta.get("positiveProjectionMaxDbfs", {}).get("meanDeltaDb"),
                "missRmsP95DeltaMeanDb": delta.get("missRmsP95Dbfs", {}).get("meanDeltaDb"),
                "missRmsMaxDeltaMeanDb": delta.get("missRmsMaxDbfs", {}).get("meanDeltaDb"),
            }
        )
    return {
        "file": str(path.resolve()),
        "schema": report.get("schema"),
        "evaluations": result,
        "interpretation": "The pass-6-to-pass-10 projection-tail changes are small while raw-miss metrics continue to worsen; this is a plateau/over-removal warning, not a reason to extend S-only training blindly.",
    }


def extract_model_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    report = read_json(path)
    model = report.get("model", {})
    if not isinstance(model, dict):
        model = {}
    result: dict[str, Any] = {}
    for key, value in model.items():
        if isinstance(value, dict):
            result[key] = {
                field: value[field]
                for field in ("file", "sha256", "assembly", "step")
                if field in value
            }
    checkpoint = report.get("checkpoint")
    if isinstance(checkpoint, dict):
        result["checkpoint"] = {
            field: checkpoint[field]
            for field in ("file", "sha256", "variant", "step", "globalStep")
            if field in checkpoint
        }
    return result


def make_comparison(
    current: list[dict[str, Any]],
    prior: list[dict[str, Any]],
    prior_name: str,
    tolerance: float,
) -> dict[str, Any]:
    current_groups = group_by(current, song_key)
    prior_groups = group_by(prior, song_key)
    shared = sorted(set(current_groups) & set(prior_groups))
    matches = nearest_matches(current_groups, prior_groups, shared, tolerance)
    return {
        "priorName": prior_name,
        "priorSummary": mark_summary(prior),
        "sharedSongCount": len(shared),
        "sameSongFractions": song_fraction_comparison(current_groups, prior_groups, shared),
        "nearestTimeMatching": {
            "toleranceSeconds": tolerance,
            "currentRowsOnSharedSongs": sum(len(current_groups[key]) for key in shared),
            "priorRowsOnSharedSongs": sum(len(prior_groups[key]) for key in shared),
            "summary": match_summary(matches),
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.match_tolerance_seconds <= 0:
        raise ValueError("match tolerance must be positive")
    current = load_rows(args.current_csv)
    prior_b1 = load_rows(args.prior_batch1_csv, "batch1")
    prior_b2 = load_rows(args.prior_batch2_csv, "batch2")
    prior_b2_full = load_rows(args.prior_batch2_full_csv, "batch2")
    current_groups = group_by(current, song_key)
    report = {
        "schema": "local-modern-song-fma-s86-review-analysis@1",
        "status": "completed",
        "inputs": {
            "current": {"csv": str(args.current_csv.resolve()), "eventCount": len(current), "songCount": len(current_groups), "model": extract_model_reference(args.current_report)},
            "priorH50Batch1": {"csv": str(args.prior_batch1_csv.resolve()), "eventCount": len(prior_b1), "songCount": len(group_by(prior_b1, song_key)), "model": extract_model_reference(args.prior_batch1_report)},
            "priorH50Batch2FirstPass": {"csv": str(args.prior_batch2_csv.resolve()), "eventCount": len(prior_b2), "songCount": len(group_by(prior_b2, song_key)), "model": extract_model_reference(args.prior_batch2_report)},
            "priorH50Batch2FullPass": {"csv": str(args.prior_batch2_full_csv.resolve()), "eventCount": len(prior_b2_full), "songCount": len(group_by(prior_b2_full, song_key)), "model": extract_model_reference(args.prior_batch2_full_report)},
        },
        "semantics": {
            "S": "Especially conspicuous residual vocal; a subset of R.",
            "R": "Further vocal, harmony, spoken, or vocal-effect removal desired.",
            "K": "Current result acceptable for the reviewed event.",
            "I": "H50/S86 removed extra non-vocal content relative to Inst 3; this does not mean Inst 3 removed non-vocal content.",
            "comparisonCaveat": "Current candidates were rescanned and selected after S86 rendering, so raw mark-rate changes are descriptive. Nearest-time matches use one-to-one matching within the stated tolerance and remain subjective evidence.",
        },
        "current": {
            "overall": mark_summary(current),
            "byCategory": grouped_mark_summary(current, "categoryValue"),
            "byRole": grouped_mark_summary(current, "role"),
            "byBatch": grouped_mark_summary(current, "batch"),
            "byMarkMetrics": mark_metric_summary(current),
            "poolSummary": current_pool_summary(current),
        },
        "comparisons": [
            make_comparison(current, prior_b1, "prior H50 batch 1, 16 events/song", args.match_tolerance_seconds),
            make_comparison(current, prior_b2, "prior H50 batch 2 first pass, 5 events/song", args.match_tolerance_seconds),
            make_comparison(current, prior_b2_full, "prior H50 batch 2 selected full pass, 16 events/song", args.match_tolerance_seconds),
        ],
        "pressureContext": pressure_summary(args.pressure_report),
        "recommendation": {
            "decision": "Do not continue unconstrained S-only pressure training. First run a small residual-vocal pilot from the current S/R labels with strict song splits and a frozen S86 control; defer accompaniment restoration to a separate stage.",
            "vocalStage": [
                "Use current S/R events as residual examples, not old events that are now K.",
                "Keep current S86-event-only@pass-5 as the source and control; use Inst 3 residual target only in the reviewed event core and anchor to S86 elsewhere.",
                "Initially use only batch-2 train songs with no I event, plus a separately frozen train split from the previously pending batch-1 songs. Keep calibration/holdout songs evaluation-only.",
                "Weight S above R by sampling priority, cap one event per song per pass, and keep the external budget at roughly 5-10% of updates. Do not raise learning rate or remove the anchor in the first pilot.",
                "Stop if the 12-song continuous listening set is unchanged or if I-like damage increases; the prior S-only pressure curve already shows a numerical plateau after pass 5.",
            ],
            "restorationStage": [
                "Do not use FMA I labels as pseudo-ground-truth instrumental targets; they only say the current output is more aggressive than Inst 3 at that clip.",
                "Use MUSDB true instrumental (drums+bass+other) as the restoration reference on a separate restoration arm, with song-disjoint validation and a MUSDB-only control.",
                "Use FMA Inst 3 as a bounded preference target: on reviewed I regions move toward Inst 3, while on S/R regions retain the vocal-removal target. Keep K regions anchored to the current S86 output.",
                "Evaluate restoration and vocal removal separately before combining them; compare I-region error, S/R residual projection, ordinary instrumental SDR, low-vocal SDR, seams, and blind listening.",
            ],
        },
    }
    json_write(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "current": report["current"]["overall"],
        "comparisons": [
            {
                "priorName": item["priorName"],
                "sharedSongCount": item["sharedSongCount"],
                "sameSongAggressive": item["sameSongFractions"]["aggressiveSPlusR"],
                "matchedEvents": item["nearestTimeMatching"]["summary"]["matchedEventCount"],
            }
            for item in report["comparisons"]
        ],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
