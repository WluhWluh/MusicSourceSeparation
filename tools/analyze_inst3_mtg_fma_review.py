#!/usr/bin/env python3
"""Summarize the human R/K/I review and propose conservative pilot pools.

The human mark is the authority for the desired direction relative to the
current H50 output. ``R`` means H50 should remove more content like Inst 3;
``I`` means H50 removed extra non-vocal content and should restore content up
to the Inst 3 reference; ``K`` means preserve the current result. Numeric
Inst 3 projection is reported for diagnostics only.
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
DEFAULT_CSV = ROOT / "data" / "inst3-mtg-fma-event-listening" / "human-review-template.csv"
DEFAULT_REPORT = ROOT / "data" / "inst3-mtg-fma-event-listening" / "mtg-fma-event-listening-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "inst3-mtg-fma-event-listening" / "human-review-analysis.json"
MARKS = ("R", "K", "I")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_rows(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Review CSV is empty")
    required = {"mark", "eventId", "sourceSet", "role", "trackName", "eventScoreDbfs"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Review CSV missing columns: {sorted(missing)}")
    for row in rows:
        mark = (row.get("mark") or "").strip().upper()
        if mark not in MARKS:
            raise ValueError(f"Every row must have R, K, or I; invalid mark for {row.get('eventId')}: {mark!r}")
        row["mark"] = mark
        for key in (
            "eventScoreDbfs",
            "positiveProjection50Dbfs",
            "positiveProjection100Dbfs",
            "positiveProjection200Dbfs",
        ):
            row[key] = float(row[key])
    return rows


def fraction(count: int, total: int) -> float:
    return count / max(total, 1)


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    result = []
    for name, values in sorted(groups.items()):
        counts = Counter(row["mark"] for row in values)
        scores = [float(row["eventScoreDbfs"]) for row in values]
        r_fraction = fraction(counts["R"], len(values))
        i_fraction = fraction(counts["I"], len(values))
        result.append(
            {
                "name": name,
                "count": len(values),
                "marks": {mark: counts[mark] for mark in MARKS},
                "fractions": {mark: fraction(counts[mark], len(values)) for mark in MARKS},
                "rToI": fraction(counts["R"], counts["R"] + counts["I"]),
                "meanEventScoreDbfs": statistics.fmean(scores),
                "medianEventScoreDbfs": statistics.median(scores),
                "humanConfidence": (
                    "r-dominant" if r_fraction >= 0.5 and i_fraction <= 0.125
                    else "i-dominant" if i_fraction >= 0.5
                    else "neutral" if counts["R"] == 0 and counts["I"] == 0
                    else "mixed"
                ),
            }
        )
    return result


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mark in MARKS:
        values = sorted(float(row["eventScoreDbfs"]) for row in rows if row["mark"] == mark)
        result[mark] = {
            "count": len(values),
            "meanDbfs": statistics.fmean(values),
            "medianDbfs": statistics.median(values),
            "minDbfs": min(values),
            "maxDbfs": max(values),
        }
    return result


def conservative_song_policy(summary: dict[str, Any]) -> str:
    marks = summary["marks"]
    r_fraction = summary["fractions"]["R"]
    i_fraction = summary["fractions"]["I"]
    if marks["R"] == 0 and marks["I"] == 0:
        return "neutral-control"
    if i_fraction >= 0.5:
        return "restoration-or-validation-only"
    if marks["R"] >= 4 and i_fraction <= 0.125:
        return "reviewed-r-candidate"
    if marks["R"] >= 1 and i_fraction <= 0.25:
        return "small-r-candidate"
    return "ambiguous-review-only"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = parse_rows(args.csv)
    event_report = json.loads(args.event_report.resolve().read_text(encoding="utf-8"))
    report_event_ids = {event["eventId"] for event in event_report.get("events", [])}
    missing_events = [row["eventId"] for row in rows if row["eventId"] not in report_event_ids]
    if missing_events:
        raise ValueError(f"Review rows missing from event report: {missing_events[:3]}")
    by_song = group_summary(rows, "trackName")
    for song in by_song:
        song["policy"] = conservative_song_policy(song)
        song["sourceSet"] = next(row["sourceSet"] for row in rows if row["trackName"] == song["name"])
        song["role"] = next(row["role"] for row in rows if row["trackName"] == song["name"])
        song["source"] = next(row["source"] for row in rows if row["trackName"] == song["name"])
        song["slug"] = next(row["eventId"].split("-", 1)[1].rsplit("-", 1)[0] for row in rows if row["trackName"] == song["name"])
    r_candidate_songs = {
        item["name"]
        for item in by_song
        if item["policy"] in {"reviewed-r-candidate", "small-r-candidate"}
        and item["role"] != "supplement-holdout"
    }
    restoration_songs = {
        item["name"] for item in by_song if item["marks"]["I"] > 0
    }
    recommended_r_events = [
        {
            "eventId": row["eventId"],
            "trackName": row["trackName"],
            "sourceSet": row["sourceSet"],
            "role": row["role"],
            "mark": row["mark"],
            "eventScoreDbfs": row["eventScoreDbfs"],
            "centerSeconds": float(row["centerSeconds"]),
        }
        for row in rows
        if row["mark"] == "R" and row["trackName"] in r_candidate_songs
    ]
    recommended_i_events = [
        {
            "eventId": row["eventId"],
            "trackName": row["trackName"],
            "sourceSet": row["sourceSet"],
            "role": row["role"],
            "mark": row["mark"],
            "eventScoreDbfs": row["eventScoreDbfs"],
            "centerSeconds": float(row["centerSeconds"]),
        }
        for row in rows
        if row["mark"] == "I" and row["trackName"] in restoration_songs
    ]
    result = {
        "schema": "local-inst3-mtg-fma-human-review-analysis@1",
        "status": "completed",
        "input": {
            "csv": str(args.csv.resolve()),
            "eventReport": str(args.event_report.resolve()),
            "rowCount": len(rows),
            "eventCount": len(report_event_ids),
            "missingEventRows": missing_events,
        },
        "overall": {
            "marks": {mark: sum(row["mark"] == mark for row in rows) for mark in MARKS},
            "fractions": {mark: fraction(sum(row["mark"] == mark for row in rows), len(rows)) for mark in MARKS},
        },
        "bySourceSet": group_summary(rows, "sourceSet"),
        "byRole": group_summary(rows, "role"),
        "bySong": sorted(by_song, key=lambda item: (-item["fractions"]["R"], item["name"])),
        "scoreByMark": score_summary(rows),
        "recommendedPolicy": {
            "humanMarkAuthority": True,
            "doNotUseNumericRankingAlone": True,
            "reviewedRCandidateSongs": sorted(r_candidate_songs),
            "restorationReviewSongs": sorted(restoration_songs),
            "recommendedRCountBeforeSampling": len(recommended_r_events),
            "recommendedICountBeforeSampling": len(recommended_i_events),
            "sampling": "sample marked R events for aggressive-removal correction and marked I events for a separate restoration/safety arm; preserve K and non-event context at H50; keep whole songs disjoint from holdout",
            "target": "Use Inst3 residual target inside reviewed R and I event cores (opposite signed corrections); preserve H50 on K events and outside event cores",
            "arms": {
                "R-only": "Inst3 target on R events; H50 anchor on I/K and context",
                "R-plus-I": "Inst3 target on both R and I events; H50 anchor on K and context",
            },
        },
        "recommendedREvents": recommended_r_events,
        "recommendedIEvents": recommended_i_events,
        "notes": [
            "The track pool is a stress-test set, not a representative modern-pop distribution.",
            "I marks indicate H50 over-removal relative to Inst 3, not an Inst 3 error. High-I songs may still be poor modern-pop representatives and should be held out or given a separate corrective arm until domain fit is confirmed.",
            "The existing MUSDB18 backbone and private modern listening set should remain primary product-facing evidence.",
        ],
    }
    json_write(args.output, result)
    print(json.dumps({"status": result["status"], "rows": len(rows), "recommendedRSongs": len(r_candidate_songs), "recommendedREvents": len(recommended_r_events), "restorationSongs": len(restoration_songs), "recommendedIEvents": len(recommended_i_events), "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
