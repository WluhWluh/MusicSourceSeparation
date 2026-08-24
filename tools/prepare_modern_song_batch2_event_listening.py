#!/usr/bin/env python3
"""Freeze and prepare human event review for the second FMA song batch.

The style review is the only selection input: Y songs are frozen into a
category/language-stratified song split, while N songs remain excluded. Each
selected song gets five temporally distributed two-second candidates. Each
candidate is written as H50-continuation+5, 300 ms silence, Inst 3, and 300
ms trailing silence. This tool prepares listening artifacts only; it does
not assign any event to training until the CSV is reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

import prepare_modern_song_event_listening as base
import run_inst3_distill_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "data" / "modern-song-original-candidates-batch2" / "source-manifest.json"
DEFAULT_STYLE_REVIEW = ROOT / "data" / "modern-song-original-candidates-batch2" / "original-style-review.csv"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-batch2-inst3-event-listening"
DEFAULT_CHECKPOINT = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
EXPECTED_SOURCE_COUNT = 118
CANDIDATES_PER_SONG = 5
COVERAGE_BINS = 5
SCAN_HOP_MS = 50
SPLIT_WEIGHTS = {"train": 70, "calibration": 20, "holdout": 30}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--style-review", type=Path, default=DEFAULT_STYLE_REVIEW)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument("--force-h50", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": base.sha256_file(path)}


def load_batch_records(manifest_path: Path, review_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    records = manifest.get("records")
    if manifest.get("status") != "completed" or not isinstance(records, list) or len(records) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"Expected completed {EXPECTED_SOURCE_COUNT}-song manifest")
    with review_path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if len(review_rows) != len(records):
        raise ValueError(f"Style review has {len(review_rows)} rows, expected {len(records)}")
    by_order = {int(record["order"]): record for record in records}
    all_records: list[dict[str, Any]] = []
    for row in review_rows:
        mark = (row.get("keep") or "").strip().upper()
        if mark not in {"Y", "N"}:
            raise ValueError(f"Every style-review row must be Y or N; order {row.get('order')!r} is {mark!r}")
        order = int(row["order"])
        record = dict(by_order[order])
        source_path = Path(record["download"]["file"])
        if source_path.resolve() != Path(row["file"]).resolve():
            raise ValueError(f"Style-review path mismatch for order {order}")
        if not source_path.is_file() or base.sha256_file(source_path) != record["download"]["sha256"]:
            raise ValueError(f"Original source hash mismatch for order {order}")
        record["styleReview"] = mark
        record["languageClass"] = record.get("languageCode") or "<metadata-blank>"
        all_records.append(record)
    selected = [record for record in all_records if record["styleReview"] == "Y"]
    if not selected:
        raise ValueError("No Y songs in style review")
    return all_records, selected


def stable_key(record: dict[str, Any]) -> str:
    value = "batch2-split-v1\0{}\0{}\0{}\0{}".format(
        record["order"], record["sourceId"], record["artistName"], record["trackName"]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def allocate_counts(total: int, weights: dict[str, int | float]) -> dict[str, int]:
    if total < 0 or not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("Invalid count allocation")
    weight_sum = float(sum(weights.values()))
    raw = {key: total * float(value) / weight_sum for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(weights, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


def freeze_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    targets = allocate_counts(len(records), SPLIT_WEIGHTS)
    remaining = dict(targets)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["category"], record["languageClass"])].append(record)
    assigned: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        values = sorted(groups[group_key], key=stable_key)
        group_counts = allocate_counts(len(values), remaining)
        cursor = 0
        for role in ("train", "calibration", "holdout"):
            for record in values[cursor : cursor + group_counts[role]]:
                copy = dict(record)
                copy["splitRole"] = role
                copy["splitKey"] = stable_key(record)
                assigned.append(copy)
            cursor += group_counts[role]
        for role, count in group_counts.items():
            remaining[role] -= count
    if any(value != 0 for value in remaining.values()):
        raise AssertionError(f"Split allocation did not consume targets: {remaining}")
    role_by_order = {record["order"]: record["splitRole"] for record in assigned}
    return {
        "algorithm": "category-language stratified stable SHA-256 order",
        "weights": SPLIT_WEIGHTS,
        "selectedSongCount": len(records),
        "targetCounts": targets,
        "actualCounts": dict(Counter(record["splitRole"] for record in assigned)),
        "roleByOrder": role_by_order,
        "records": assigned,
    }


def write_song_classification(path: Path, records: list[dict[str, Any]], role_by_order: dict[int, str]) -> None:
    fields = (
        "selection", "splitRole", "order", "category", "languageCode", "languageClass",
        "artistName", "trackName", "sourceId", "genreTags", "durationSecondsMetadata",
        "license", "licenseUrl", "file", "notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: int(item["order"])):
            writer.writerow(
                {
                    "selection": record["styleReview"],
                    "splitRole": role_by_order.get(int(record["order"]), "style-rejected"),
                    "order": record["order"],
                    "category": record["category"],
                    "languageCode": record.get("languageCode", ""),
                    "languageClass": record["languageClass"],
                    "artistName": record["artistName"],
                    "trackName": record["trackName"],
                    "sourceId": record["sourceId"],
                    "genreTags": json.dumps(record.get("genreTags", []), ensure_ascii=False),
                    "durationSecondsMetadata": record.get("durationSeconds", ""),
                    "license": record["license"],
                    "licenseUrl": record["licenseUrl"],
                    "file": record["download"]["file"],
                    "notes": "" if record["styleReview"] == "Y" else "excluded by original style review",
                }
            )


def write_event_review(path: Path, events: list[dict[str, Any]], role_by_order: dict[int, str]) -> None:
    fields = (
        "mark", "eventId", "serial", "sourceOrder", "splitRole", "category", "languageCode",
        "artistName", "trackName", "centerSeconds", "eventScoreDbfs",
        "positiveProjection50Dbfs", "positiveProjection100Dbfs", "positiveProjection200Dbfs",
        "pairedFile", "h50File", "inst3File", "mixtureFile", "license", "licenseUrl", "notes",
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
                    "splitRole": role_by_order[event["sourceOrder"]],
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
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("threads and inference-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    all_records, selected = load_batch_records(args.source_manifest, args.style_review)
    split = freeze_split(selected)
    split_records = split["records"]
    role_by_order = {int(record["order"]): record["splitRole"] for record in split_records}
    selected_by_order = {int(record["order"]): record for record in split_records}
    for record in all_records:
        if int(record["order"]) in selected_by_order:
            record.update(selected_by_order[int(record["order"])])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    classification = {
        "schema": "local-modern-song-batch2-classification@1",
        "status": "frozen",
        "sourceManifest": file_metadata(args.source_manifest.resolve()),
        "styleReview": file_metadata(args.style_review.resolve()),
        "sourceSongCount": len(all_records),
        "selectedSongCount": len(selected),
        "rejectedSongCount": len(all_records) - len(selected),
        "selectionRule": "all and only original-style-review keep=Y",
        "languagePolicy": "Use FMA languageCode only; blank means metadata-blank and is not inferred from title or artist.",
        "categoryCounts": dict(Counter(record["category"] for record in selected)),
        "languageCounts": dict(Counter(record["languageClass"] for record in selected)),
        "records": [
            {
                "order": record["order"],
                "selection": record["styleReview"],
                "splitRole": role_by_order.get(int(record["order"]), "style-rejected"),
                "category": record["category"],
                "languageCode": record.get("languageCode", ""),
                "languageClass": record["languageClass"],
                "artistName": record["artistName"],
                "trackName": record["trackName"],
                "sourceId": record["sourceId"],
            }
            for record in sorted(all_records, key=lambda item: int(item["order"]))
        ],
    }
    json_write(output_root / "song-classification.json", classification)
    json_write(output_root / "song-split.json", split)
    write_song_classification(output_root / "song-classification.csv", all_records, role_by_order)
    started = time.perf_counter()
    teacher_caches = base.prepare_teacher_caches(selected, args, output_root)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    songs, events = base.prepare_h50_and_events(
        selected,
        args,
        output_root,
        device,
        candidates_per_song=CANDIDATES_PER_SONG,
        coverage_bins=COVERAGE_BINS,
        scan_hop_ms=SCAN_HOP_MS,
    )
    for event in events:
        event["splitRole"] = role_by_order[event["sourceOrder"]]
        event["languageCode"] = selected_by_order[event["sourceOrder"]].get("languageCode", "")
        event["humanReview"] = {"mark": None, "notes": ""}
    for song in songs:
        song["splitRole"] = role_by_order[song["order"]]
        song["languageCode"] = selected_by_order[song["order"]].get("languageCode", "")
    report = {
        "schema": "local-modern-song-batch2-event-listening@1",
        "status": "completed",
        "sourceSelection": file_metadata(output_root / "song-classification.json"),
        "split": {
            "file": str((output_root / "song-split.json").resolve()),
            "sha256": base.sha256_file(output_root / "song-split.json"),
            "algorithm": split["algorithm"],
            "targetCounts": split["targetCounts"],
            "actualCounts": split["actualCounts"],
        },
        "model": {
            "h50ContinuationPlus5": file_metadata(args.checkpoint.resolve()),
            "architecture": file_metadata(args.architecture_checkpoint.resolve()),
            "assembly": "continuous-context-overlap-save",
            "inst3": {
                "file": str(pilot.DEFAULT_TEACHER.resolve()),
                "sha256": base.sha256_file(pilot.DEFAULT_TEACHER.resolve()),
                "contractId": "uvr_mdxnet_inst_3@2",
                "fullSongCaches": teacher_caches,
            },
        },
        "selection": {
            "songCount": len(selected),
            "candidatesPerSong": CANDIDATES_PER_SONG,
            "eventCount": len(events),
            "coverageBins": COVERAGE_BINS,
            "scanHopMs": SCAN_HOP_MS,
            "eventResolutionsMs": [50, 100, 200],
            "snippetDurationSeconds": 2.0,
            "pairedDurationSeconds": 4.6,
            "pairedOrder": "H50-continuation+5 2 s, 300 ms silence, Inst 3 2 s, 300 ms trailing silence",
            "edgePolicy": "only complete two-second snippets; no song-edge padding",
        },
        "songs": songs,
        "events": events,
        "review": {
            "status": "awaiting-human-listening",
            "markKey": {
                "S": "especially conspicuous residual vocal; S is a subset of R",
                "R": "further vocal, harmony, spoken, or vocal-effect removal is desired",
                "K": "current H50 result is acceptable",
                "I": "H50 removed extra non-vocal content that should be retained relative to Inst 3",
                "blank": "not reviewed",
            },
            "csv": str((output_root / "human-review-template.csv").resolve()),
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "elapsedSeconds": time.perf_counter() - started,
        },
        "licenseDisposition": {
            "scope": "local non-commercial research; preserve per-track attribution and license",
            "audio": "do not redistribute original or derived listening files",
            "training": "do not assign reviewed events to training until human CSV review is complete",
        },
    }
    report_path = output_root / "event-listening-report.json"
    json_write(report_path, report)
    write_event_review(output_root / "human-review-template.csv", events, role_by_order)
    (output_root / "human-review-key.txt").write_text(
        "S = especially conspicuous R: loud residual or clear consonants make the lyric easy to hear\n"
        "R = H50 still retains vocal/harmony/vocal-effect content to remove\n"
        "K = current H50 result is acceptable\n"
        "I = H50 removed extra non-vocal content retained by Inst 3\n"
        "blank = not reviewed\n\n"
        "Pair order: H50-continuation+5 2 s, silence 300 ms, Inst 3 2 s, trailing silence 300 ms.\n"
        "The split is song-level and frozen before event marks are collected.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "sourceSongs": len(all_records), "selectedSongs": len(selected), "events": len(events), "split": split["actualCounts"], "report": str(report_path), "reviewCsv": str((output_root / "human-review-template.csv").resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
