#!/usr/bin/env python3
"""Prepare reviewed modern songs for H50-versus-Inst3 event annotation.

Only songs marked Y in the original-style review are processed.  Each selected
song receives 16 temporally distributed two-second candidates.  The paired
listening file is H50 (2 s), 300 ms silence, Inst 3 (2 s), and a final 300 ms
silence, for a total of 4.6 seconds.

Full-song teacher and student renders are PCM16 FLAC annotation caches.  Any
later training run must rematerialize native teacher windows from the original
source rather than treating these listening files as float training targets.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as event_tools
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_teacher_oracle as oracle
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "data" / "modern-song-original-candidates" / "source-manifest.json"
DEFAULT_STYLE_REVIEW = ROOT / "data" / "modern-song-original-candidates" / "original-style-review.csv"
DEFAULT_CHECKPOINT = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-inst3-event-listening"
SAMPLE_RATE = 44_100
CANDIDATES_PER_SONG = 16
COVERAGE_BINS = 8
SCAN_HOP_MS = 50
SNIPPET_SAMPLES = 2 * SAMPLE_RATE
GAP_SAMPLES = round(0.3 * SAMPLE_RATE)
PAIR_SAMPLES = 2 * SNIPPET_SAMPLES + 2 * GAP_SAMPLES


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
    parser.add_argument("--force-events", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_selected_records(manifest_path: Path, review_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or len(manifest.get("records", [])) != 72:
        raise ValueError("Expected completed 72-song original manifest")
    with review_path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if len(review_rows) != 72:
        raise ValueError("Expected 72 style-review rows")
    marks = {(row.get("keep") or "").strip().upper() for row in review_rows}
    if not marks <= {"Y", "N"} or "" in marks:
        raise ValueError(f"Every style-review row must be Y or N, got {marks}")
    by_order = {int(record["order"]): record for record in manifest["records"]}
    selected = []
    for row in review_rows:
        if (row.get("keep") or "").strip().upper() != "Y":
            continue
        order = int(row["order"])
        record = dict(by_order[order])
        source_path = Path(record["download"]["file"])
        if source_path.resolve() != Path(row["file"]).resolve():
            raise ValueError(f"Style-review path mismatch for order {order}")
        if not source_path.is_file() or sha256_file(source_path) != record["download"]["sha256"]:
            raise ValueError(f"Original source hash mismatch for order {order}")
        record["styleReview"] = "Y"
        record["role"] = "pending-event-review"
        selected.append(record)
    if len(selected) != 57:
        raise ValueError(f"Expected 57 Y songs, got {len(selected)}")
    return selected


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid stereo audio for {path}: {audio.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": 2,
    }


def load_flac(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid cached FLAC {path}: {audio.shape}, {rate}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def make_pair(h50: np.ndarray, inst3: np.ndarray) -> np.ndarray:
    if h50.shape != (SNIPPET_SAMPLES, 2) or inst3.shape != (SNIPPET_SAMPLES, 2):
        raise ValueError(f"Expected two 2-second clips, got {h50.shape}, {inst3.shape}")
    silence = np.zeros((GAP_SAMPLES, 2), dtype=np.float32)
    pair = np.concatenate((h50, silence, inst3, silence), axis=0)
    if pair.shape != (PAIR_SAMPLES, 2):
        raise ValueError(f"Unexpected pair shape {pair.shape}")
    return np.ascontiguousarray(pair, dtype=np.float32)


def cache_valid(path: Path, metadata_path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            return False
        audio, rate = sf.info(path), SAMPLE_RATE
        return audio.samplerate == rate and audio.channels == 2 and audio.frames > 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def prepare_teacher_caches(
    records: list[dict[str, Any]], args: argparse.Namespace, output_root: Path
) -> dict[str, dict[str, Any]]:
    teacher_hash = sha256_file(pilot.DEFAULT_TEACHER.resolve())
    session, providers = pilot.make_teacher_session(
        pilot.DEFAULT_TEACHER.resolve(), args.threads, True
    )
    result: dict[str, dict[str, Any]] = {}
    try:
        for index, record in enumerate(records, start=1):
            slug = record["slug"]
            path = output_root / "full-song" / "inst3Instrumental" / f"{record['order']:03d}-{slug}.flac"
            metadata_path = path.with_suffix(".json")
            expected = {
                "schema": "local-modern-song-inst3-cache@1",
                "rawSha256": record["download"]["sha256"],
                "teacherSha256": teacher_hash,
                "teacherContractId": "uvr_mdxnet_inst_3@2",
            }
            reused = not args.force_teacher and cache_valid(path, metadata_path, expected)
            if reused:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                source, rate = listening.load_audio(Path(record["download"]["file"]))
                if rate != SAMPLE_RATE:
                    raise ValueError(f"Unexpected decoded rate for {slug}")
                instrumental, _residual, timing = oracle.render_teacher_audio(
                    session, source, progress_label=f"teacher {index}/{len(records)} {slug}"
                )
                output = write_flac(path, instrumental)
                metadata = {
                    **expected,
                    "slug": slug,
                    "order": record["order"],
                    "providers": list(providers),
                    "timing": timing,
                    "output": output,
                    "annotationCache": "PCM16 FLAC; regenerate native teacher windows for training",
                }
                json_write(metadata_path, metadata)
                del source, instrumental, _residual
                gc.collect()
            result[slug] = metadata
            print(f"teacher cache {'reuse' if reused else 'write'} {index}/{len(records)}: {record['artistName']} - {record['trackName']}", flush=True)
    finally:
        del session
    return result


def prepare_h50_and_events(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
    device: torch.device,
    candidates_per_song: int = CANDIDATES_PER_SONG,
    coverage_bins: int = COVERAGE_BINS,
    scan_hop_ms: int = SCAN_HOP_MS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_hash = sha256_file(args.checkpoint.resolve())
    model, model_source = local.load_h50_model(
        args.architecture_checkpoint.resolve(), args.checkpoint.resolve(), device
    )
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    all_events: list[dict[str, Any]] = []
    song_reports: list[dict[str, Any]] = []
    try:
        for song_index, record in enumerate(records, start=1):
            slug = record["slug"]
            source, rate = listening.load_audio(Path(record["download"]["file"]))
            if rate != SAMPLE_RATE:
                raise ValueError(f"Unexpected decoded rate for {slug}")
            teacher_path = output_root / "full-song" / "inst3Instrumental" / f"{record['order']:03d}-{slug}.flac"
            teacher = load_flac(teacher_path)
            if source.shape != teacher.shape:
                raise ValueError(f"Source/teacher shape mismatch for {slug}")
            h50_path = output_root / "full-song" / "h50ContinuationPlus5" / f"{record['order']:03d}-{slug}.flac"
            h50_metadata_path = h50_path.with_suffix(".json")
            expected = {
                "schema": "local-modern-song-h50-cache@1",
                "rawSha256": record["download"]["sha256"],
                "checkpointSha256": checkpoint_hash,
                "assembly": "continuous-context-overlap-save",
            }
            reused = not args.force_h50 and cache_valid(h50_path, h50_metadata_path, expected)
            if reused:
                h50 = load_flac(h50_path)
                h50_metadata = json.loads(h50_metadata_path.read_text(encoding="utf-8"))
            else:
                residual, timing = continuous.render_student(
                    model, source, contract, device, args.inference_batch_size
                )
                h50 = np.ascontiguousarray(source - residual, dtype=np.float32)
                output = write_flac(h50_path, h50)
                h50_metadata = {
                    **expected,
                    "slug": slug,
                    "order": record["order"],
                    "modelSource": model_source,
                    "timing": timing,
                    "output": output,
                }
                json_write(h50_metadata_path, h50_metadata)
                # Always rank from the persisted annotation cache so resume is deterministic.
                h50 = load_flac(h50_path)
                del residual
            selected = event_tools.scan_candidates(
                source,
                teacher,
                h50,
                slug,
                scan_hop_ms,
                candidates_per_song,
                coverage_bins,
            )
            event_ids = []
            for event in selected:
                event_id = f"modern-{record['order']:03d}-{slug}-{int(event['songCandidateIndex']):02d}"
                serial = len(all_events) + 1
                start = int(event["snippetStartSamples"])
                end = int(event["snippetEndSamples"])
                prefix = f"{serial:04d}-{event_id}"
                event_paths = {
                    "mixture": output_root / "events" / "mixture" / f"{prefix}.flac",
                    "h50ContinuationPlus5": output_root / "events" / "h50ContinuationPlus5" / f"{prefix}.flac",
                    "inst3Instrumental": output_root / "events" / "inst3Instrumental" / f"{prefix}.flac",
                    "pairedListening": output_root / "events" / "pairedListening" / f"{prefix}.flac",
                }
                outputs = {
                    "mixture": write_flac(event_paths["mixture"], source[start:end]),
                    "h50ContinuationPlus5": write_flac(event_paths["h50ContinuationPlus5"], h50[start:end]),
                    "inst3Instrumental": write_flac(event_paths["inst3Instrumental"], teacher[start:end]),
                    "pairedListening": write_flac(event_paths["pairedListening"], make_pair(h50[start:end], teacher[start:end])),
                }
                report = {
                    "eventId": event_id,
                    "serial": serial,
                    "sourceOrder": record["order"],
                    "category": record["category"],
                    "source": record["source"],
                    "sourceId": record["sourceId"],
                    "artistName": record["artistName"],
                    "trackName": record["trackName"],
                    "slug": slug,
                    "role": "pending-event-review",
                    "license": record["license"],
                    "licenseUrl": record["licenseUrl"],
                    "sourceUrl": record["sourceUrl"],
                    "rawSha256": record["download"]["sha256"],
                    "centerSamples": int(event["centerSamples"]),
                    "centerSeconds": float(event["centerSeconds"]),
                    "coverageFraction": float(event["coverageFraction"]),
                    "eventScoreDbfs": float(event["eventScoreDbfs"]),
                    "metrics": event["metrics"],
                    "outputs": outputs,
                    "humanReview": {"mark": None, "notes": ""},
                }
                all_events.append(report)
                event_ids.append(event_id)
            song_reports.append(
                {
                    "order": record["order"],
                    "category": record["category"],
                    "artistName": record["artistName"],
                    "trackName": record["trackName"],
                    "slug": slug,
                    "durationSecondsDecoded": source.shape[0] / SAMPLE_RATE,
                    "candidateCount": len(selected),
                    "eventIds": event_ids,
                    "h50": h50_metadata,
                }
            )
            print(f"events {song_index}/{len(records)}: {record['artistName']} - {record['trackName']}", flush=True)
            del source, teacher, h50
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return song_reports, all_events


def write_review_csv(path: Path, events: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {
                row["eventId"]: {
                    "mark": (row.get("mark") or "").strip().upper(),
                    "notes": row.get("notes") or "",
                }
                for row in csv.DictReader(handle)
                if row.get("eventId")
            }
    fields = (
        "mark",
        "eventId",
        "serial",
        "sourceOrder",
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
        "mixtureFile",
        "license",
        "licenseUrl",
        "notes",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            previous = existing.get(event["eventId"], {})
            writer.writerow(
                {
                    "mark": previous.get("mark", ""),
                    "eventId": event["eventId"],
                    "serial": event["serial"],
                    "sourceOrder": event["sourceOrder"],
                    "category": event["category"],
                    "artistName": event["artistName"],
                    "trackName": event["trackName"],
                    "centerSeconds": f"{event['centerSeconds']:.3f}",
                    "eventScoreDbfs": f"{event['eventScoreDbfs']:.2f}",
                    "positiveProjection50Dbfs": f"{event['metrics']['50']['positiveProjectionDbfs']:.2f}",
                    "positiveProjection100Dbfs": f"{event['metrics']['100']['positiveProjectionDbfs']:.2f}",
                    "positiveProjection200Dbfs": f"{event['metrics']['200']['positiveProjectionDbfs']:.2f}",
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("threads and inference-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    records = load_selected_records(args.source_manifest, args.style_review)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_manifest = {
        "schema": "local-modern-song-selected-originals@1",
        "status": "frozen",
        "sourceManifest": checkpoint_metadata(args.source_manifest.resolve()),
        "styleReview": checkpoint_metadata(args.style_review.resolve()),
        "selection": "all and only rows marked Y",
        "songCount": len(records),
        "categoryCounts": dict(Counter(record["category"] for record in records)),
        "records": records,
        "role": "pending-event-review; no train/validation assignment",
    }
    json_write(output_root / "selected-source-manifest.json", selected_manifest)
    started = time.perf_counter()
    teacher_caches = prepare_teacher_caches(records, args, output_root)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    songs, events = prepare_h50_and_events(records, args, output_root, device)
    report = {
        "schema": "local-modern-song-inst3-event-listening@1",
        "status": "completed",
        "sourceSelection": checkpoint_metadata(output_root / "selected-source-manifest.json"),
        "model": {
            "h50ContinuationPlus5": checkpoint_metadata(args.checkpoint.resolve()),
            "architecture": checkpoint_metadata(args.architecture_checkpoint.resolve()),
            "assembly": "continuous-context-overlap-save",
            "inst3": {
                "file": str(pilot.DEFAULT_TEACHER.resolve()),
                "sha256": sha256_file(pilot.DEFAULT_TEACHER.resolve()),
                "contractId": "uvr_mdxnet_inst_3@2",
                "fullSongCaches": teacher_caches,
            },
        },
        "selection": {
            "songCount": len(records),
            "candidatesPerSong": CANDIDATES_PER_SONG,
            "eventCount": len(events),
            "coverageBins": COVERAGE_BINS,
            "scanHopMs": SCAN_HOP_MS,
            "eventResolutionsMs": [50, 100, 200],
            "snippetDurationSeconds": 2.0,
            "pairedDurationSeconds": 4.6,
            "pairedOrder": "H50 2 s, 300 ms silence, Inst 3 2 s, 300 ms trailing silence",
            "edgePolicy": "only complete two-second snippets; no song-edge padding",
        },
        "songs": songs,
        "events": events,
        "review": {
            "status": "awaiting-human-listening",
            "markKey": {
                "S": "H50 retains especially conspicuous vocal content; S is a high-priority subset of R",
                "R": "H50 retains vocal/harmony/vocal-effect content that should be removed further",
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
            "scope": "local research; preserve per-track attribution and license",
            "audio": "do not redistribute original or derived listening files from this repository",
            "training": "regenerate native float teacher windows only after train/validation split is frozen",
        },
    }
    report_path = output_root / "event-listening-report.json"
    json_write(report_path, report)
    write_review_csv(output_root / "human-review-template.csv", events)
    (output_root / "human-review-key.txt").write_text(
        "S = especially conspicuous R: loud residual or clear consonants make the lyric easy to hear\n"
        "R = H50 still retains vocal/harmony/vocal-effect content to remove\n"
        "K = current H50 result is acceptable\n"
        "I = H50 removed extra non-vocal content retained by Inst 3\n"
        "blank = not reviewed\n\n"
        "Pair order: H50 2 s, silence 300 ms, Inst 3 2 s, trailing silence 300 ms.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "songs": len(songs), "events": len(events), "report": str(report_path), "reviewCsv": report["review"]["csv"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
