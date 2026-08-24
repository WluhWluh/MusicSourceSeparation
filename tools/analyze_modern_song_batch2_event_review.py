#!/usr/bin/env python3
"""Analyze the first-pass labels for the second FMA candidate batch.

S is a perceptual severity subset of R. The CSV stores them as mutually
exclusive marks, so all aggressive-removal counts in this report use S + R.
The report also freezes a deterministic 50-70 song set for a second pass with
16 events per song: every song containing S, plus one R-only song per category.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "modern-song-batch2-inst3-event-listening" / "human-review-template.csv"
DEFAULT_EVENT_REPORT = ROOT / "data" / "modern-song-batch2-inst3-event-listening" / "event-listening-report.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "data" / "modern-song-original-candidates-batch2" / "source-manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "modern-song-batch2-inst3-event-listening"
MARKS = ("S", "R", "K", "I")
CATEGORIES = ("pop-synth", "hiphop-rnb", "dance-electronic", "latin-modern", "pop-rock", "singer-songwriter")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_EVENT_REPORT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_key(song: dict[str, Any]) -> str:
    value = "batch2-full-review-v1\0{}\0{}\0{}".format(
        song["sourceOrder"], song["sourceId"], song["artistName"]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def parse_inputs(csv_path: Path, event_report_path: Path, source_manifest_path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    with csv_path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "mark", "eventId", "serial", "sourceOrder", "splitRole", "category", "languageCode",
        "artistName", "trackName", "centerSeconds", "eventScoreDbfs",
        "positiveProjection50Dbfs", "positiveProjection100Dbfs", "positiveProjection200Dbfs",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Review CSV missing required columns: {sorted(required - set(rows[0] if rows else {}))}")
    event_report = json.loads(event_report_path.resolve().read_text(encoding="utf-8"))
    report_ids = {str(event["eventId"]) for event in event_report.get("events", [])}
    by_song: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mark = (row.get("mark") or "").strip().upper()
        if mark not in MARKS:
            raise ValueError(f"Invalid mark {mark!r} for {row.get('eventId')}")
        if row["eventId"] not in report_ids:
            raise ValueError(f"Review row is absent from event report: {row['eventId']}")
        row["mark"] = mark
        row["sourceOrderInt"] = int(row["sourceOrder"])
        for key in ("centerSeconds", "eventScoreDbfs", "positiveProjection50Dbfs", "positiveProjection100Dbfs", "positiveProjection200Dbfs"):
            row[key + "Float"] = float(row[key])
        by_song[row["sourceOrderInt"]].append(row)
    if len(rows) != len(by_song) * 5 or any(len(values) != 5 for values in by_song.values()):
        raise ValueError("Expected exactly five first-pass events per song")
    manifest = json.loads(source_manifest_path.resolve().read_text(encoding="utf-8"))
    records = {int(record["order"]): record for record in manifest.get("records", [])}
    if len(records) != 118:
        raise ValueError(f"Expected 118 source records, found {len(records)}")
    for order, values in by_song.items():
        if order not in records:
            raise ValueError(f"Unknown source order in review: {order}")
        if any(value["category"] != values[0]["category"] for value in values):
            raise ValueError(f"Inconsistent category for source order {order}")
    return rows, by_song, records


def song_summary(values: list[dict[str, Any]], source_record: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["mark"] for row in values)
    first = values[0]
    return {
        "sourceOrder": int(first["sourceOrder"]),
        "sourceId": source_record["sourceId"],
        "source": source_record.get("source", "fma"),
        "artistName": first["artistName"],
        "trackName": first["trackName"],
        "slug": source_record["slug"],
        "category": first["category"],
        "languageCode": first.get("languageCode", ""),
        "splitRole": first["splitRole"],
        "rawSha256": source_record["download"]["sha256"],
        "sourceFile": source_record["download"]["file"],
        "license": source_record["license"],
        "licenseUrl": source_record["licenseUrl"],
        "counts": {mark: counts[mark] for mark in MARKS},
        "aggressiveCount": counts["S"] + counts["R"],
        "aggressiveFraction": (counts["S"] + counts["R"]) / len(values),
        "priorityScore": counts["S"] * 3 + counts["R"],
        "eventScoreDbfs": stats(row["eventScoreDbfsFloat"] for row in values),
        "sEventScoreDbfs": stats(row["eventScoreDbfsFloat"] for row in values if row["mark"] == "S"),
        "firstPassEventIds": [row["eventId"] for row in values],
    }


def choose_full_annotation_songs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for song in summaries:
        if song["counts"]["S"] > 0:
            copy = dict(song)
            copy["selectionReason"] = "all songs with at least one S event"
            selected[song["sourceOrder"]] = copy
    for category in CATEGORIES:
        candidates = [
            song for song in summaries
            if song["category"] == category and song["counts"]["S"] == 0 and song["counts"]["R"] > 0
        ]
        if not candidates:
            raise ValueError(f"No R-only candidate for category {category}")
        chosen = sorted(
            candidates,
            key=lambda song: (-song["counts"]["R"], -song["eventScoreDbfs"]["mean"], stable_key(song)),
        )[0]
        copy = dict(chosen)
        copy["selectionReason"] = f"highest-priority R-only context in category {category}"
        selected[chosen["sourceOrder"]] = copy
    result = sorted(selected.values(), key=lambda song: song["sourceOrder"])
    if not 50 <= len(result) <= 70:
        raise ValueError(f"Full annotation selection outside requested range: {len(result)}")
    return result


def group_summary(summaries: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for song in summaries:
        groups[str(song.get(key) or "<metadata-blank>")].append(song)
    result = []
    for name, songs in sorted(groups.items()):
        counts = Counter()
        for song in songs:
            counts.update(song["counts"])
        event_count = len(songs) * 5
        result.append(
            {
                "name": name,
                "songCount": len(songs),
                "eventCount": event_count,
                "marks": {mark: counts[mark] for mark in MARKS},
                "aggressiveCount": counts["S"] + counts["R"],
                "aggressiveFraction": (counts["S"] + counts["R"]) / event_count,
            }
        )
    return result


def write_selection_csv(path: Path, songs: list[dict[str, Any]]) -> None:
    fields = (
        "selected", "selectionReason", "sourceOrder", "splitRole", "category", "languageCode",
        "artistName", "trackName", "S", "R", "K", "I", "aggressiveCount", "sourceFile",
        "license", "licenseUrl", "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for song in songs:
            counts = song["counts"]
            writer.writerow(
                {
                    "selected": "Y",
                    "selectionReason": song["selectionReason"],
                    "sourceOrder": song["sourceOrder"],
                    "splitRole": song["splitRole"],
                    "category": song["category"],
                    "languageCode": song.get("languageCode", ""),
                    "artistName": song["artistName"],
                    "trackName": song["trackName"],
                    "S": counts["S"],
                    "R": counts["R"],
                    "K": counts["K"],
                    "I": counts["I"],
                    "aggressiveCount": song["aggressiveCount"],
                    "sourceFile": song["sourceFile"],
                    "license": song["license"],
                    "licenseUrl": song["licenseUrl"],
                    "notes": "",
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows, by_song, source_records = parse_inputs(args.csv, args.event_report, args.source_manifest)
    summaries = [song_summary(values, source_records[order]) for order, values in sorted(by_song.items())]
    selected = choose_full_annotation_songs(summaries)
    counts = Counter(row["mark"] for row in rows)
    report = {
        "schema": "local-modern-song-batch2-human-review-analysis@1",
        "status": "completed",
        "input": {
            "csv": str(args.csv.resolve()),
            "csvSha256": sha256_file(args.csv.resolve()),
            "eventReport": str(args.event_report.resolve()),
            "sourceManifest": str(args.source_manifest.resolve()),
            "eventCount": len(rows),
            "songCount": len(summaries),
            "eventsPerSong": 5,
        },
        "semantics": {
            "S": "perceptually conspicuous residual vocal; S is a subset of R",
            "R": "residual vocal, harmony, spoken content, or vocal effect desired removed",
            "K": "current result acceptable",
            "I": "H50 removed extra non-vocal content relative to Inst 3; I is not an assertion that Inst 3 removed non-vocal content",
            "aggressiveCount": "S + R because S and R are stored as exclusive CSV marks",
        },
        "overall": {
            "marks": {mark: counts[mark] for mark in MARKS},
            "fractions": {mark: counts[mark] / len(rows) for mark in MARKS},
            "aggressiveCountSPlusR": counts["S"] + counts["R"],
            "aggressiveFractionSPlusR": (counts["S"] + counts["R"]) / len(rows),
            "songsWithS": sum(song["counts"]["S"] > 0 for song in summaries),
            "songsWithR": sum(song["counts"]["R"] > 0 for song in summaries),
            "songsWithAggressive": sum(song["aggressiveCount"] > 0 for song in summaries),
            "songsWithI": sum(song["counts"]["I"] > 0 for song in summaries),
            "songsWithoutSOrR": sum(song["aggressiveCount"] == 0 for song in summaries),
        },
        "byCategory": group_summary(summaries, "category"),
        "bySplit": group_summary(summaries, "splitRole"),
        "byLanguage": group_summary(summaries, "languageCode"),
        "byMarkMetrics": {
            mark: {
                "count": counts[mark],
                "eventScoreDbfs": stats(row["eventScoreDbfsFloat"] for row in rows if row["mark"] == mark),
                "positiveProjectionDbfs": {
                    str(ms): stats(row[f"positiveProjection{ms}DbfsFloat"] for row in rows if row["mark"] == mark)
                    for ms in (50, 100, 200)
                },
            }
            for mark in MARKS
        },
        "songs": summaries,
        "fullAnnotationSelection": {
            "selectionRule": "all S-containing songs plus the highest-priority R-only song in each category",
            "songCount": len(selected),
            "eventsIfSixteenPerSong": len(selected) * 16,
            "categoryCounts": dict(Counter(song["category"] for song in selected)),
            "splitCounts": dict(Counter(song["splitRole"] for song in selected)),
            "songs": selected,
        },
        "nextStep": {
            "action": "prepare 16 temporally distributed events for the frozen selection; keep all other songs as unexpanded controls",
            "doNotDo": ["do not move songs between train/calibration/holdout", "do not assign any event to training before second-pass review", "do not infer language from titles"],
            "controlCandidates": [song for song in summaries if song["aggressiveCount"] == 0 or song["counts"]["I"] >= 2],
        },
    }
    output_root = args.output_root.resolve()
    report_path = output_root / "human-review-analysis.json"
    selection_json = output_root / "full-annotation-selection.json"
    selection_csv = output_root / "full-annotation-selection.csv"
    json_write(report_path, report)
    json_write(selection_json, {
        "schema": "local-modern-song-batch2-full-annotation-selection@1",
        "status": "frozen",
        "analysis": {"file": str(report_path.resolve()), "sha256": sha256_file(report_path)},
        "selectionRule": report["fullAnnotationSelection"]["selectionRule"],
        "songCount": len(selected),
        "eventsPerSong": 16,
        "songs": selected,
    })
    write_selection_csv(selection_csv, selected)
    print(json.dumps({"status": report["status"], "marks": report["overall"]["marks"], "songsWithS": report["overall"]["songsWithS"], "aggressiveFraction": report["overall"]["aggressiveFractionSPlusR"], "fullAnnotationSongs": len(selected), "fullAnnotationEvents": len(selected) * 16, "report": str(report_path), "selectionCsv": str(selection_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
