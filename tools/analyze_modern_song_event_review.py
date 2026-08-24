#!/usr/bin/env python3
"""Analyze S/R/K/I labels for the modern-song event listening set.

``S`` is a strict subset of ``R``: it denotes a particularly conspicuous
residual where loudness and/or clear consonants make the lyric easy to hear.
This tool keeps S separate for sampling and evaluation while preserving the
same Inst 3 target semantics as R.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "modern-song-inst3-event-listening" / "human-review-template.csv"
DEFAULT_REPORT = ROOT / "data" / "modern-song-inst3-event-listening" / "event-listening-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-inst3-event-listening" / "human-review-analysis.json"
DEFAULT_S_CSV = ROOT / "data" / "modern-song-inst3-event-listening" / "s-focused-event-candidates.csv"
MARKS = ("S", "R", "K", "I")
RESOLUTIONS = (50, 100, 200)
CLUSTER_GAP_SECONDS = 3.0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--s-csv", type=Path, default=DEFAULT_S_CSV)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def parse_rows(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "mark",
        "eventId",
        "category",
        "artistName",
        "trackName",
        "eventScoreDbfs",
        "positiveProjection50Dbfs",
        "positiveProjection100Dbfs",
        "positiveProjection200Dbfs",
        "centerSeconds",
    }
    if not rows:
        raise ValueError("Modern event review CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Review CSV missing columns: {sorted(missing)}")
    for row in rows:
        mark = (row.get("mark") or "").strip().upper()
        if mark not in MARKS:
            raise ValueError(f"Invalid mark {mark!r} for {row.get('eventId')}")
        row["mark"] = mark
        for key in (
            "eventScoreDbfs",
            "positiveProjection50Dbfs",
            "positiveProjection100Dbfs",
            "positiveProjection200Dbfs",
            "centerSeconds",
        ):
            row[key] = float(row[key])
    return rows


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "<blank>")].append(row)
    return groups


def summarize_group(name: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["mark"] for row in values)
    return {
        "name": name,
        "count": len(values),
        "marks": {mark: counts[mark] for mark in MARKS},
        "fractions": {mark: counts[mark] / len(values) for mark in MARKS},
        "aggressiveCount": counts["S"] + counts["R"],
        "aggressiveFraction": (counts["S"] + counts["R"]) / len(values),
        "eventScoreDbfs": stats(row["eventScoreDbfs"] for row in values),
        "sEventScoreDbfs": stats(row["eventScoreDbfs"] for row in values if row["mark"] == "S"),
    }


def clusters(values: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(values, key=lambda row: float(row["centerSeconds"]))
    result: list[list[dict[str, Any]]] = []
    for row in ordered:
        if not result or float(row["centerSeconds"]) - float(result[-1][-1]["centerSeconds"]) >= CLUSTER_GAP_SECONDS:
            result.append([row])
        else:
            result[-1].append(row)
    return result


def write_s_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "priorityRank",
        "mark",
        "eventId",
        "category",
        "artistName",
        "trackName",
        "centerSeconds",
        "eventScoreDbfs",
        "positiveProjection50Dbfs",
        "positiveProjection100Dbfs",
        "positiveProjection200Dbfs",
        "pairedFile",
        "h50File",
        "inst3File",
        "notes",
    )
    with path.resolve().open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "priorityRank": rank,
                    "mark": row["mark"],
                    "eventId": row["eventId"],
                    "category": row["category"],
                    "artistName": row["artistName"],
                    "trackName": row["trackName"],
                    "centerSeconds": f"{row['centerSeconds']:.3f}",
                    "eventScoreDbfs": f"{row['eventScoreDbfs']:.2f}",
                    "positiveProjection50Dbfs": f"{row['positiveProjection50Dbfs']:.2f}",
                    "positiveProjection100Dbfs": f"{row['positiveProjection100Dbfs']:.2f}",
                    "positiveProjection200Dbfs": f"{row['positiveProjection200Dbfs']:.2f}",
                    "pairedFile": row.get("pairedFile", ""),
                    "h50File": row.get("h50File", ""),
                    "inst3File": row.get("inst3File", ""),
                    "notes": row.get("notes", ""),
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = parse_rows(args.csv)
    event_report = json.loads(args.event_report.resolve().read_text(encoding="utf-8"))
    report_ids = {event["eventId"] for event in event_report.get("events", [])}
    missing = [row["eventId"] for row in rows if row["eventId"] not in report_ids]
    if missing:
        raise ValueError(f"Review rows missing from event report: {missing[:5]}")
    by_mark = {mark: [row for row in rows if row["mark"] == mark] for mark in MARKS}
    by_category = [summarize_group(name, values) for name, values in sorted(group_rows(rows, "category").items())]
    by_song = group_rows(rows, "trackName")
    song_summaries = []
    s_clusters_total = 0
    multi_event_clusters = 0
    for name, values in sorted(by_song.items(), key=lambda item: (-sum(row["mark"] == "S" for row in item[1]), item[0])):
        s_values = [row for row in values if row["mark"] == "S"]
        song_clusters = clusters(s_values)
        s_clusters_total += len(song_clusters)
        multi_event_clusters += sum(len(cluster) > 1 for cluster in song_clusters)
        item = summarize_group(name, values)
        item.update(
            {
                "category": values[0]["category"],
                "artistName": values[0]["artistName"],
                "sClusterCount": len(song_clusters),
                "sMultiEventClusterCount": sum(len(cluster) > 1 for cluster in song_clusters),
                "sCentersSeconds": [float(row["centerSeconds"]) for row in s_values],
            }
        )
        song_summaries.append(item)
    s_rows = sorted(
        by_mark["S"],
        key=lambda row: (
            -float(row["eventScoreDbfs"]),
            -float(row["positiveProjection50Dbfs"]),
            row["trackName"],
            float(row["centerSeconds"]),
        ),
    )
    write_s_csv(args.s_csv, s_rows)
    r_scores = [row["eventScoreDbfs"] for row in by_mark["R"]]
    s_scores = [row["eventScoreDbfs"] for row in by_mark["S"]]
    r_median = statistics.median(r_scores)
    result = {
        "schema": "local-modern-song-human-review-analysis@1",
        "status": "completed",
        "input": {
            "csv": str(args.csv.resolve()),
            "eventReport": str(args.event_report.resolve()),
            "rowCount": len(rows),
            "eventCount": len(report_ids),
            "missingEventRows": missing,
        },
        "overall": {
            "marks": {mark: len(by_mark[mark]) for mark in MARKS},
            "fractions": {mark: len(by_mark[mark]) / len(rows) for mark in MARKS},
            "aggressiveCountSPlusR": len(by_mark["S"]) + len(by_mark["R"]),
            "aggressiveFractionSPlusR": (len(by_mark["S"]) + len(by_mark["R"])) / len(rows),
        },
        "byMark": {
            mark: {
                "count": len(by_mark[mark]),
                "eventScoreDbfs": stats(row["eventScoreDbfs"] for row in by_mark[mark]),
                "positiveProjection": {
                    str(ms): stats(row[f"positiveProjection{ms}Dbfs"] for row in by_mark[mark])
                    for ms in RESOLUTIONS
                },
            }
            for mark in MARKS
        },
        "sVsR": {
            "sCount": len(by_mark["S"]),
            "rCount": len(by_mark["R"]),
            "eventScoreMeanDeltaSMinusRDb": statistics.fmean(s_scores) - statistics.fmean(r_scores),
            "eventScoreMedianDeltaSMinusRDb": statistics.median(s_scores) - statistics.median(r_scores),
            "fractionSAboveRMedian": sum(value > r_median for value in s_scores) / len(s_scores),
            "positiveProjectionMeanDeltaSMinusRDb": {
                str(ms): statistics.fmean(row[f"positiveProjection{ms}Dbfs"] for row in by_mark["S"])
                - statistics.fmean(row[f"positiveProjection{ms}Dbfs"] for row in by_mark["R"])
                for ms in RESOLUTIONS
            },
            "interpretation": "S is a high-priority perceptual subset of R, not a different target semantic.",
        },
        "byCategory": by_category,
        "bySong": song_summaries,
        "sClustering": {
            "clusterGapSeconds": CLUSTER_GAP_SECONDS,
            "songsWithS": sum(item["marks"]["S"] > 0 for item in song_summaries),
            "songCount": len(song_summaries),
            "sEventCount": len(by_mark["S"]),
            "sClusterCount": s_clusters_total,
            "multiEventClusterCount": multi_event_clusters,
            "recommendation": "cap first pilot at one event per S cluster and at most two S clusters per song",
        },
        "recommendedExperiment": {
            "decision": "run one S-focused pilot before acquiring more data or adding more R dose",
            "reason": "The Y-filtered modern pool now has 210 high-confidence S events across 39 songs; it is sufficient to test salience-aware sampling, while S is measurably stronger than ordinary R.",
            "arms": {
                "S0-control": "equal-budget MUSDB-only continuation from H50-continuation+5 step 1600",
                "S1-focused": "same MUSDB schedule plus 64 S-event records/pass; one event per 3-second S cluster, max two clusters/song, balanced by category",
                "optional-S2": "same S pool with 50 ms core + 50 ms guard versus 100 ms core + 25 ms guard; only after S1 smoke is stable",
            },
            "trainingContract": {
                "studentSemantic": "residual-vocals",
                "target": "V_T = mixture - Inst3Instrumental",
                "eventCore": "S uses the same Inst3 residual target as R; S only changes sampling priority",
                "nonEvent": "H50 anchor outside event core and guard",
                "assembly": "128-frame continuous-context overlap-save",
                "learningRate": 1e-6,
                "batchSize": 4,
                "batchNorm": "freeze running statistics",
                "passes": 5,
                "externalFraction": 64 / 704,
            },
            "split": "freeze an artist/song-disjoint 16-20 song S-rich training pool and 10-12 song S-containing holdout before selecting training events; remaining Y songs are controls/stress tests",
            "evaluation": [
                "S holdout 50/100/200 ms target error, p95 and max",
                "ordinary R, K and I events separately",
                "12 private modern songs with continuous A/B listening",
                "new modern song holdout, not used for event selection",
                "20 MUSDB calibration/internal-test songs, seam and mechanical-noise checks",
            ],
            "goNoGo": {
                "machine": "at least 0.5 dB improvement on held-out S 100 ms p95 and max direction, with no I/K deterioration over 0.1 dB",
                "listening": "known abrupt hotspots must become repeatedly distinguishable or audibly less intelligible in blind A/B; a small numeric gain alone is insufficient",
                "stop": "if S-focused training is machine-positive but blind-indistinguishable again, stop increasing external data and investigate temporal resolution/loss/model capacity",
            },
        },
        "sFocusedCsv": str(args.s_csv.resolve()),
        "notes": [
            "S labels are user-provided perceptual severity labels based on residual loudness and consonant intelligibility.",
            "S and R share the same aggressive-removal target; S is a sampling/priority label.",
            "The present Y pool is style-filtered but still FMA-heavy and lacks verified Mandarin modern-pop coverage.",
        ],
    }
    json_write(args.output, result)
    print(json.dumps({"status": result["status"], "marks": result["overall"]["marks"], "sClusters": s_clusters_total, "sFocusedCsv": str(args.s_csv.resolve()), "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
