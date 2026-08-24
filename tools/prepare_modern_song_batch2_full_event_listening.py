#!/usr/bin/env python3
"""Prepare the second, 16-event human review pass for FMA batch 2.

The song selection is frozen by analyze_modern_song_batch2_event_review.py.
This script reuses the already rendered full-song Inst 3 and H50 continuous
caches, scans each selected song for 16 temporally distributed candidates,
and writes the same 4.6-second paired listening format as the first pass.
No model inference is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import prepare_modern_song_event_listening as base
import render_inst3_mtg_fma_event_listening as event_tools


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "data" / "modern-song-batch2-inst3-event-listening" / "full-annotation-selection.json"
DEFAULT_CACHE_ROOT = ROOT / "data" / "modern-song-batch2-inst3-event-listening"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-batch2-full-event-listening"
SAMPLE_RATE = 44_100
CANDIDATES_PER_SONG = 16
COVERAGE_BINS = 8
SCAN_HOP_MS = 50


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def file_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_selection(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if value.get("status") != "frozen":
        raise ValueError(f"Selection is not frozen: {path}")
    songs = value.get("songs")
    if not isinstance(songs, list) or not 50 <= len(songs) <= 70:
        raise ValueError(f"Expected 50-70 selected songs, got {len(songs) if isinstance(songs, list) else None}")
    if len({int(song["sourceOrder"]) for song in songs}) != len(songs):
        raise ValueError("Selection contains duplicate source orders")
    return value


def cache_paths(cache_root: Path, song: dict[str, Any]) -> tuple[Path, Path]:
    prefix = f"{int(song['sourceOrder']):03d}-{song['slug']}"
    return (
        cache_root / "full-song" / "h50ContinuationPlus5" / f"{prefix}.flac",
        cache_root / "full-song" / "inst3Instrumental" / f"{prefix}.flac",
    )


def load_source(song: dict[str, Any]) -> np.ndarray:
    path = Path(song["sourceFile"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != song["rawSha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    audio = event_tools.load_audio(path)
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid source audio: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def write_review_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fields = (
        "mark", "eventId", "serial", "sourceOrder", "splitRole", "selectionReason",
        "category", "languageCode", "artistName", "trackName", "centerSeconds",
        "eventScoreDbfs", "positiveProjection50Dbfs", "positiveProjection100Dbfs",
        "positiveProjection200Dbfs", "pairedFile", "h50File", "inst3File", "mixtureFile",
        "license", "licenseUrl", "notes",
    )
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {row["eventId"]: row for row in csv.DictReader(handle) if row.get("eventId")}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            previous = existing.get(event["eventId"], {})
            metrics = event["metrics"]
            writer.writerow(
                {
                    "mark": (previous.get("mark") or "").strip().upper(),
                    "eventId": event["eventId"],
                    "serial": event["serial"],
                    "sourceOrder": event["sourceOrder"],
                    "splitRole": event["splitRole"],
                    "selectionReason": event["selectionReason"],
                    "category": event["category"],
                    "languageCode": event.get("languageCode", ""),
                    "artistName": event["artistName"],
                    "trackName": event["trackName"],
                    "centerSeconds": f"{event['centerSeconds']:.3f}",
                    "eventScoreDbfs": f"{event['eventScoreDbfs']:.2f}",
                    "positiveProjection50Dbfs": f"{metrics['50']['positiveProjectionDbfs']:.2f}",
                    "positiveProjection100Dbfs": f"{metrics['100']['positiveProjectionDbfs']:.2f}",
                    "positiveProjection200Dbfs": f"{metrics['200']['positiveProjectionDbfs']:.2f}",
                    "pairedFile": event["outputs"]["pairedListening"]["file"],
                    "h50File": event["outputs"]["h50ContinuationPlus5"]["file"],
                    "inst3File": event["outputs"]["inst3Instrumental"]["file"],
                    "mixtureFile": event["outputs"]["mixture"]["file"],
                    "license": event["license"],
                    "licenseUrl": event["licenseUrl"],
                    "notes": previous.get("notes", ""),
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    selection = load_selection(args.selection)
    songs = selection["songs"]
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for song in songs:
        h50_path, inst3_path = cache_paths(cache_root, song)
        if not h50_path.is_file() or not inst3_path.is_file():
            raise FileNotFoundError(f"Missing full-song cache for {song['sourceOrder']}: {h50_path}, {inst3_path}")
    report_path = output_root / "event-listening-report.json"
    started = time.perf_counter()
    all_events: list[dict[str, Any]] = []
    song_reports: list[dict[str, Any]] = []
    for song_index, song in enumerate(songs, start=1):
        h50_path, inst3_path = cache_paths(cache_root, song)
        source = load_source(song)
        h50 = base.load_flac(h50_path)
        inst3 = base.load_flac(inst3_path)
        if source.shape != h50.shape or source.shape != inst3.shape:
            raise ValueError(f"Shape mismatch for source order {song['sourceOrder']}: {source.shape}, {h50.shape}, {inst3.shape}")
        selected = event_tools.scan_candidates(
            source,
            inst3,
            h50,
            song["slug"],
            SCAN_HOP_MS,
            CANDIDATES_PER_SONG,
            COVERAGE_BINS,
        )
        event_ids: list[str] = []
        for candidate in selected:
            serial = len(all_events) + 1
            index = int(candidate["songCandidateIndex"])
            event_id = f"batch2-full-{int(song['sourceOrder']):03d}-{song['slug']}-{index:02d}"
            start = int(candidate["snippetStartSamples"])
            end = int(candidate["snippetEndSamples"])
            prefix = f"{serial:04d}-{event_id}"
            outputs = {
                "mixture": base.write_flac(output_root / "events" / "mixture" / f"{prefix}.flac", source[start:end]),
                "h50ContinuationPlus5": base.write_flac(output_root / "events" / "h50ContinuationPlus5" / f"{prefix}.flac", h50[start:end]),
                "inst3Instrumental": base.write_flac(output_root / "events" / "inst3Instrumental" / f"{prefix}.flac", inst3[start:end]),
                "pairedListening": base.write_flac(output_root / "events" / "pairedListening" / f"{prefix}.flac", base.make_pair(h50[start:end], inst3[start:end])),
            }
            event = {
                "eventId": event_id,
                "serial": serial,
                "sourceOrder": int(song["sourceOrder"]),
                "splitRole": song["splitRole"],
                "selectionReason": song["selectionReason"],
                "category": song["category"],
                "languageCode": song.get("languageCode", ""),
                "artistName": song["artistName"],
                "trackName": song["trackName"],
                "slug": song["slug"],
                "source": song.get("source", "fma"),
                "sourceId": song["sourceId"],
                "license": song["license"],
                "licenseUrl": song["licenseUrl"],
                "sourceUrl": song.get("sourceUrl", ""),
                "rawSha256": song["rawSha256"],
                "centerSamples": int(candidate["centerSamples"]),
                "centerSeconds": float(candidate["centerSeconds"]),
                "snippetStartSamples": start,
                "snippetEndSamples": end,
                "snippetDurationSeconds": (end - start) / SAMPLE_RATE,
                "coverageFraction": float(candidate["coverageFraction"]),
                "eventScoreDbfs": float(candidate["eventScoreDbfs"]),
                "metrics": candidate["metrics"],
                "outputs": outputs,
                "humanReview": {"mark": None, "notes": ""},
            }
            all_events.append(event)
            event_ids.append(event_id)
        song_reports.append(
            {
                "sourceOrder": int(song["sourceOrder"]),
                "splitRole": song["splitRole"],
                "selectionReason": song["selectionReason"],
                "category": song["category"],
                "artistName": song["artistName"],
                "trackName": song["trackName"],
                "songDurationSeconds": source.shape[0] / SAMPLE_RATE,
                "candidateCount": len(selected),
                "eventIds": event_ids,
                "firstPassCounts": song["counts"],
            }
        )
        print(f"full events {song_index}/{len(songs)}: {song['artistName']} - {song['trackName']}", flush=True)
        del source, h50, inst3
    report = {
        "schema": "local-modern-song-batch2-full-event-listening@1",
        "status": "completed",
        "selection": {
            "file": str(args.selection.resolve()),
            "sha256": sha256_file(args.selection.resolve()),
            "songCount": len(songs),
            "candidatesPerSong": CANDIDATES_PER_SONG,
            "eventCount": len(all_events),
            "coverageBins": COVERAGE_BINS,
            "scanHopMs": SCAN_HOP_MS,
            "snippetDurationSeconds": 2.0,
            "pairedDurationSeconds": 4.6,
            "pairedOrder": "H50-continuation+5 2 s, 300 ms silence, Inst 3 2 s, 300 ms trailing silence",
            "edgePolicy": "only complete two-second snippets; no song-edge padding",
        },
        "cache": {
            "root": str(cache_root),
            "source": "reused full-song PCM16 FLAC caches from first pass",
            "modelInferencePerformed": False,
        },
        "songs": song_reports,
        "events": all_events,
        "review": {
            "status": "awaiting-human-listening",
            "markKey": {
                "S": "especially conspicuous residual vocal; S is a subset of R",
                "R": "further vocal, harmony, spoken, or vocal-effect removal is desired",
                "K": "current result is acceptable",
                "I": "H50 removed extra non-vocal content relative to Inst 3; it is not a claim that Inst 3 removed non-vocal content",
                "blank": "not reviewed",
            },
            "csv": str((output_root / "human-review-template.csv").resolve()),
        },
        "runtime": {
            "elapsedSeconds": time.perf_counter() - started,
            "platform": platform.platform(),
        },
    }
    json_write(report_path, report)
    write_review_csv(output_root / "human-review-template.csv", all_events)
    (output_root / "human-review-key.txt").write_text(
        "S = especially conspicuous R: loud residual or clear consonants make the lyric easy to hear\n"
        "R = H50 still retains vocal/harmony/vocal-effect content to remove\n"
        "K = current H50 result is acceptable\n"
        "I = H50 removed extra non-vocal content retained by Inst 3\n"
        "blank = not reviewed\n\n"
        "Pair order: H50-continuation+5 2 s, silence 300 ms, Inst 3 2 s, trailing silence 300 ms.\n"
        "This is a second-pass 16-event annotation set; do not assign events to training before review.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "songs": len(songs), "events": len(all_events), "report": str(report_path), "reviewCsv": str((output_root / "human-review-template.csv").resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
