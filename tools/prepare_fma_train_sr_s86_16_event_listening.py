#!/usr/bin/env python3
"""Prepare a second 16-event listening pass for train FMA songs with S/R marks.

The song pool is selected from the completed all-FMA S86 review.  A song is
included when ``priorSplitRole`` is ``train`` and at least one of its existing
five-event marks is ``S`` or ``R``.  Existing ``I`` marks do not exclude the
song: they are useful evidence for the later accompaniment-restoration stage.

The script reuses the continuous S86 full-song cache and native Inst 3 cache;
it does not run model inference.  Each song receives 16 temporally distributed
two-second candidates.  Listening pairs are written as H50 (2 s), 300 ms
silence, Inst 3 (2 s), and 300 ms trailing silence (4.6 s total).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import prepare_all_fma_s86_event_listening as base
import render_inst3_mtg_fma_event_listening as event_tools


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPORT = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "event-listening-report.json"
DEFAULT_SOURCE_CSV = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "human-review-template.csv"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-train-sr-s86-event-pass5-16"

SAMPLE_RATE = 44_100
SNIPPET_SAMPLES = SAMPLE_RATE * 2
GAP_SAMPLES = round(SAMPLE_RATE * 0.3)
PAIR_SAMPLES = SNIPPET_SAMPLES * 2 + GAP_SAMPLES * 2
EVENT_MS = (50, 100, 200)
CANDIDATES_PER_SONG = 16
COVERAGE_BINS = 8
SCAN_HOP_MS = 50
CONTRACT = base.CONTRACT
SCHEMA = "local-inst3-fma-train-sr-s86-16-event-listening@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-hop-ms", type=int, default=SCAN_HOP_MS)
    parser.add_argument("--coverage-bins", type=int, default=COVERAGE_BINS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    required = {"mark", "sourceBatch", "sourceOrder", "eventId"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return rows


def song_key(value: dict[str, Any] | dict[str, str]) -> tuple[str, int]:
    return str(value["sourceBatch"]), int(value["sourceOrder"])


def select_songs(report: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    songs = report.get("songs")
    if not isinstance(songs, list) or not songs:
        raise ValueError("Source report has no songs")
    marks_by_song: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        marks_by_song.setdefault(song_key(row), []).append((row.get("mark") or "").strip().upper())
    selected: list[dict[str, Any]] = []
    for song in songs:
        key = song_key(song)
        marks = marks_by_song.get(key, [])
        if str(song.get("priorSplitRole", "")).strip().lower() != "train":
            continue
        if not any(mark in {"S", "R"} for mark in marks):
            continue
        selected.append(song)
    selected.sort(key=lambda item: (str(item["sourceBatch"]), int(item["sourceOrder"])))
    return selected


def load_audio(path: Path) -> np.ndarray:
    audio = event_tools.load_audio(path)
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio {path}: {audio.shape}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def make_pair(h50: np.ndarray, inst3: np.ndarray) -> np.ndarray:
    if h50.shape != (SNIPPET_SAMPLES, 2) or inst3.shape != h50.shape:
        raise ValueError(f"Expected two two-second stereo clips, got {h50.shape}, {inst3.shape}")
    silence = np.zeros((GAP_SAMPLES, 2), dtype=np.float32)
    result = np.concatenate((h50, silence, inst3, silence), axis=0)
    if result.shape != (PAIR_SAMPLES, 2) or not np.isfinite(result).all():
        raise ValueError(f"Invalid pair shape: {result.shape}")
    return np.ascontiguousarray(result, dtype=np.float32)


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio for {path}: {audio.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": 2,
    }


def write_review_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fields = (
        "mark", "eventId", "serial", "sourceBatch", "sourceOrder", "priorSplitRole",
        "category", "languageCode", "artistName", "trackName", "centerSeconds",
        "eventScoreDbfs", "positiveProjection50Dbfs", "positiveProjection100Dbfs",
        "positiveProjection200Dbfs", "pairedFile", "h50File", "inst3File", "mixtureFile",
        "license", "licenseUrl", "notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            metrics = event["metrics"]
            writer.writerow(
                {
                    "mark": "",
                    "eventId": event["eventId"],
                    "serial": event["serial"],
                    "sourceBatch": event["sourceBatch"],
                    "sourceOrder": event["sourceOrder"],
                    "priorSplitRole": event["priorSplitRole"],
                    "category": event["category"],
                    "languageCode": event["languageCode"],
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
                    "notes": "",
                }
            )


def process_song(song: dict[str, Any], serial_start: int, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = load_audio(Path(song["sourcePath"]))
    h50 = load_audio(Path(song["h50Path"]))
    teacher = load_audio(Path(song["teacherPath"]))
    if source.shape != h50.shape or source.shape != teacher.shape:
        raise ValueError(f"Source/H50/Inst3 shape mismatch for {song['sourceBatch']} {song['sourceOrder']}")
    selected = event_tools.scan_candidates(
        source,
        teacher,
        h50,
        f"{song['sourceBatch']}-{song['slug']}",
        args.scan_hop_ms,
        CANDIDATES_PER_SONG,
        args.coverage_bins,
    )
    events: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(selected, start=1):
        serial = serial_start + candidate_index - 1
        event_id = (
            f"fma-train-sr-s86-pass5-16-{song['sourceBatch']}-"
            f"{int(song['sourceOrder']):03d}-{song['slug']}-{candidate_index:02d}"
        )
        start = int(candidate["snippetStartSamples"])
        end = int(candidate["snippetEndSamples"])
        h50_clip = np.ascontiguousarray(h50[start:end], dtype=np.float32)
        inst3_clip = np.ascontiguousarray(teacher[start:end], dtype=np.float32)
        mixture_clip = np.ascontiguousarray(source[start:end], dtype=np.float32)
        if h50_clip.shape != (SNIPPET_SAMPLES, 2):
            raise ValueError(f"Invalid candidate slice for {event_id}: {h50_clip.shape}")
        prefix = f"{serial:04d}-{event_id}"
        outputs = {
            "h50ContinuationPlus5": write_flac(args.output_root / "events" / "h50ContinuationPlus5" / f"{prefix}.flac", h50_clip),
            "inst3Instrumental": write_flac(args.output_root / "events" / "inst3Instrumental" / f"{prefix}.flac", inst3_clip),
            "mixture": write_flac(args.output_root / "events" / "mixture" / f"{prefix}.flac", mixture_clip),
            "pairedListening": write_flac(args.output_root / "events" / "pairedListening" / f"{prefix}.flac", make_pair(h50_clip, inst3_clip)),
        }
        events.append(
            {
                "eventId": event_id,
                "serial": serial,
                "sourceBatch": song["sourceBatch"],
                "sourceOrder": int(song["sourceOrder"]),
                "sourceId": song["sourceId"],
                "priorSplitRole": song["priorSplitRole"],
                "category": song.get("category", ""),
                "languageCode": song.get("languageCode", ""),
                "artistName": song.get("artistName", ""),
                "trackName": song.get("trackName", ""),
                "slug": song["slug"],
                "license": song.get("license", ""),
                "licenseUrl": song.get("licenseUrl", ""),
                "sourceUrl": song.get("sourceUrl", ""),
                "sourceSha256": song["sourceSha256"],
                "centerSamples": int(candidate["centerSamples"]),
                "centerSeconds": float(candidate["centerSeconds"]),
                "snippetStartSamples": start,
                "snippetEndSamples": end,
                "coverageFraction": float(candidate["coverageFraction"]),
                "eventScoreDbfs": float(candidate["eventScoreDbfs"]),
                "metrics": candidate["metrics"],
                "outputs": outputs,
                "humanReview": {"mark": None, "notes": ""},
            }
        )
    return (
        {
            "sourceBatch": song["sourceBatch"],
            "sourceOrder": int(song["sourceOrder"]),
            "sourceId": song["sourceId"],
            "slug": song["slug"],
            "artistName": song.get("artistName", ""),
            "trackName": song.get("trackName", ""),
            "priorSplitRole": song["priorSplitRole"],
            "category": song.get("category", ""),
            "languageCode": song.get("languageCode", ""),
            "candidateCount": len(events),
            "eventIds": [event["eventId"] for event in events],
        },
        events,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scan_hop_ms <= 0 or args.coverage_bins <= 0:
        raise ValueError("scan hop and coverage bins must be positive")
    source_report = read_json(args.source_report)
    source_rows = read_rows(args.source_csv)
    songs = select_songs(source_report, source_rows)
    if args.output_root.exists() and not args.force:
        existing = args.output_root / "event-listening-report.json"
        if existing.is_file():
            raise FileExistsError(f"Output already exists; use --force to rebuild: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    song_reports: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for index, song in enumerate(songs, start=1):
        report, song_events = process_song(song, len(events) + 1, args)
        song_reports.append(report)
        events.extend(song_events)
        print(f"train S/R 16-event {index}/{len(songs)}: {song['artistName']} - {song['trackName']}", flush=True)
    if len(events) != len(songs) * CANDIDATES_PER_SONG:
        raise AssertionError(f"Unexpected event count: {len(events)}")
    selected_songs = {
        "schema": SCHEMA + ".selection",
        "selectionRule": "priorSplitRole=train and at least one current five-event mark is S or R; songs with I are retained",
        "sourceReport": str(args.source_report.resolve()),
        "sourceReportSha256": sha256_file(args.source_report),
        "sourceCsv": str(args.source_csv.resolve()),
        "sourceCsvSha256": sha256_file(args.source_csv),
        "songCount": len(songs),
        "eventCount": len(events),
        "eventsPerSong": CANDIDATES_PER_SONG,
        "songs": songs,
    }
    json_write(args.output_root / "selected-songs.json", selected_songs)
    write_review_csv(args.output_root / "human-review-template.csv", events)
    source_checkpoint = source_report.get("checkpoint", {})
    if not isinstance(source_checkpoint, dict):
        raise ValueError("Source report checkpoint metadata is invalid")
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": {
            "sourceReport": str(args.source_report.resolve()),
            "sourceCsv": str(args.source_csv.resolve()),
            "songCount": len(songs),
            "eventCount": len(events),
            "eventsPerSong": CANDIDATES_PER_SONG,
            "includesSongsWithI": True,
        },
        "checkpoint": source_checkpoint,
        "contract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "candidateScan": {
            "scanHopMs": args.scan_hop_ms,
            "coverageBins": args.coverage_bins,
            "eventMilliseconds": list(EVENT_MS),
            "selection": "continuous full-song source/Inst3/H50 scan; one strongest event per temporal bin, then score-ranked fill",
            "edgePolicy": "complete two-second snippets only; no zero padding or song-edge candidates",
        },
        "pairFormat": {
            "h50Seconds": 2.0,
            "leadingSilenceSeconds": 0.3,
            "inst3Seconds": 2.0,
            "trailingSilenceSeconds": 0.3,
            "sampleRate": SAMPLE_RATE,
            "frames": PAIR_SAMPLES,
        },
        "markSemantics": {
            "S": "Especially conspicuous residual vocal; subset of R.",
            "R": "Further vocal, harmony, spoken, or vocal-effect removal desired.",
            "K": "Current S86 result is acceptable for this event.",
            "I": "S86 removed extra non-vocal content relative to Inst 3; I does not mean Inst 3 removed non-vocal content.",
            "blank": "Not reviewed yet.",
        },
        "songs": song_reports,
        "events": events,
        "runtime": {"elapsedSeconds": time.perf_counter() - started, "platform": platform.platform()},
    }
    json_write(args.output_root / "event-listening-report.json", report)
    (args.output_root / "human-review-key.txt").write_text(
        "S = especially conspicuous R: loud residual or clear consonants make the lyric easy to hear\n"
        "R = S86 still retains vocal/harmony/vocal-effect content to remove\n"
        "K = current S86 result is acceptable\n"
        "I = S86 removed extra non-vocal content retained by Inst 3\n"
        "blank = not reviewed\n\n"
        "Pair order: S86-event-only@pass-5 2 s, silence 300 ms, Inst 3 2 s, trailing silence 300 ms.\n"
        "Selection: train songs with at least one S/R mark in the preceding five-event review; songs with I are retained.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "songs": len(songs), "events": len(events), "output": str(args.output_root.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
