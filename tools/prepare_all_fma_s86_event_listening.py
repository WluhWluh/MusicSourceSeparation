#!/usr/bin/env python3
"""Prepare five continuous S86-versus-Inst 3 hotspot clips for every FMA Y song.

The two downloaded FMA candidate batches have independent style-review files.
This tool joins them back to their source manifests, keeps only songs marked
``Y``, and refuses to process a style-rejected song.  The student is rendered
with the same 128-frame continuous overlap-save contract used by the S86
training runs.  Inst 3 is reused from the validated native full-song cache.

Generated artifacts are local listening material.  They are intentionally not
training records and the official MUSDB18 final test is not touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
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
import run_inst3_distill_pilot as pilot
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH1_MANIFEST = ROOT / "data" / "modern-song-original-candidates" / "source-manifest.json"
DEFAULT_BATCH1_REVIEW = ROOT / "data" / "modern-song-original-candidates" / "original-style-review.csv"
DEFAULT_BATCH1_TEACHER_ROOT = ROOT / "data" / "modern-song-inst3-event-listening"
DEFAULT_BATCH2_MANIFEST = ROOT / "data" / "modern-song-original-candidates-batch2" / "source-manifest.json"
DEFAULT_BATCH2_REVIEW = ROOT / "data" / "modern-song-batch2-inst3-event-listening" / "song-classification.csv"
DEFAULT_BATCH2_TEACHER_ROOT = ROOT / "data" / "modern-song-batch2-inst3-event-listening"
DEFAULT_CHECKPOINT = ROOT / "data" / "modern-song-s-r-continuation" / "runs" / "S86-event-only" / "step-3360.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-all-s86-event-pass5"

SAMPLE_RATE = 44_100
SNIPPET_SAMPLES = SAMPLE_RATE * 2
GAP_SAMPLES = round(SAMPLE_RATE * 0.3)
PAIR_SAMPLES = SNIPPET_SAMPLES * 2 + GAP_SAMPLES * 2
EVENT_MS = (50, 100, 200)
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
CHECKPOINT_FORMAT = "local-inst3-s-r-continuation-checkpoint@1"
SCHEMA = "local-inst3-fma-all-s86-event-listening@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch1-manifest", type=Path, default=DEFAULT_BATCH1_MANIFEST)
    parser.add_argument("--batch1-review", type=Path, default=DEFAULT_BATCH1_REVIEW)
    parser.add_argument("--batch1-teacher-root", type=Path, default=DEFAULT_BATCH1_TEACHER_ROOT)
    parser.add_argument("--batch2-manifest", type=Path, default=DEFAULT_BATCH2_MANIFEST)
    parser.add_argument("--batch2-review", type=Path, default=DEFAULT_BATCH2_REVIEW)
    parser.add_argument("--batch2-teacher-root", type=Path, default=DEFAULT_BATCH2_TEACHER_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates-per-song", type=int, default=5)
    parser.add_argument("--scan-hop-ms", type=int, default=50)
    parser.add_argument("--coverage-bins", type=int, default=5)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "song"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_manifest(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if value.get("status") != "completed":
        raise ValueError(f"Source manifest is not completed: {path}")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Source manifest has no records: {path}")
    return records


def read_review(path: Path) -> dict[int, dict[str, str]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Style review is empty: {path}")
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            order = int((row.get("order") or "").strip())
        except ValueError as error:
            raise ValueError(f"Review row has invalid order in {path}: {row}") from error
        result[order] = {str(key): str(value or "") for key, value in row.items()}
    return result


def review_mark(row: dict[str, str]) -> str:
    return (row.get("keep") or row.get("selection") or "").strip().upper()


def select_y_records(
    records: list[dict[str, Any]],
    reviews: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    """Validate a frozen review and return only records marked Y."""
    selected: list[dict[str, Any]] = []
    for record in records:
        order = int(record["order"])
        review = reviews.get(order)
        if review is None:
            raise ValueError(f"Missing review row for order {order}")
        mark = review_mark(review)
        if mark not in {"Y", "N"}:
            raise ValueError(f"Unexpected style mark {mark!r} for order {order}")
        if mark == "Y":
            selected.append(record)
    return selected


def source_hash(record: dict[str, Any]) -> str:
    value = record.get("download", {}).get("sha256")
    if not value:
        raise ValueError(f"Source record has no download hash: {record.get('slug')}")
    return str(value).lower()


def build_song_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = (
        (
            "batch1",
            args.batch1_manifest,
            args.batch1_review,
            args.batch1_teacher_root,
        ),
        (
            "batch2",
            args.batch2_manifest,
            args.batch2_review,
            args.batch2_teacher_root,
        ),
    )
    selected: list[dict[str, Any]] = []
    all_source_ids: list[str] = []
    for batch, manifest_path, review_path, teacher_root in specs:
        records = read_manifest(manifest_path)
        reviews = read_review(review_path)
        seen_orders = {int(record["order"]) for record in records}
        for record in select_y_records(records, reviews):
            order = int(record["order"])
            review = reviews[order]
            mark = review_mark(review)
            source_id = str(record.get("sourceId", ""))
            if not source_id:
                raise ValueError(f"Selected record has no sourceId: {batch} order {order}")
            source_path = Path(record["download"]["file"]).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            expected_hash = source_hash(record)
            actual_hash = sha256_file(source_path).lower()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Source hash mismatch for {batch} order {order}: {actual_hash} != {expected_hash}"
                )
            slug = str(record["slug"])
            teacher_path = (
                teacher_root.resolve()
                / "full-song"
                / "inst3Instrumental"
                / f"{order:03d}-{slug}.flac"
            )
            teacher_meta_path = teacher_path.with_suffix(".json")
            if not teacher_path.is_file() or not teacher_meta_path.is_file():
                raise FileNotFoundError(f"Missing Inst 3 cache for {batch} order {order}: {teacher_path}")
            teacher_meta = read_json(teacher_meta_path)
            if str(teacher_meta.get("rawSha256", "")).lower() != expected_hash:
                raise ValueError(f"Inst 3 cache raw hash mismatch for {teacher_path}")
            selected.append(
                {
                    "sourceBatch": batch,
                    "sourceOrder": order,
                    "sourceId": source_id,
                    "sourcePath": str(source_path),
                    "sourceSha256": expected_hash,
                    "slug": slug,
                    "artistName": str(record.get("artistName", "")),
                    "trackName": str(record.get("trackName", "")),
                    "albumName": str(record.get("albumName", "")),
                    "category": str(record.get("category", "")),
                    "languageCode": str(record.get("languageCode", "")),
                    "genreTags": record.get("genreTags", []),
                    "license": str(record.get("license", "")),
                    "licenseUrl": str(record.get("licenseUrl", "")),
                    "sourceUrl": str(record.get("sourceUrl", "")),
                    "priorSplitRole": str(review.get("splitRole", record.get("role", "unassigned"))),
                    "teacherPath": str(teacher_path),
                    "teacherMetaPath": str(teacher_meta_path),
                    "styleMark": mark,
                }
            )
        if len(seen_orders) != len(records):
            raise AssertionError("Duplicate source order in manifest")
    selected.sort(key=lambda item: (item["sourceBatch"], int(item["sourceOrder"])))
    ids = [item["sourceId"] for item in selected]
    if len(selected) != 158:
        raise ValueError(f"Expected 158 Y songs after style filtering, got {len(selected)}")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sourceId among selected Y songs")
    return selected


def load_audio(path: Path) -> np.ndarray:
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except RuntimeError:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        audio = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)
        rate = SAMPLE_RATE
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(rate) // divisor,
            axis=0,
        ).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio {path}: {audio.shape}")
    return audio


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio for {path}: {audio.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": 2,
        "peak": float(np.max(np.abs(audio))),
    }


def load_model(
    checkpoint_path: Path,
    architecture_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unexpected checkpoint format: {payload.get('format')}")
    if payload.get("variant") != "S86-event-only" or int(payload.get("step", -1)) != 3360:
        raise ValueError(
            f"Expected S86-event-only step 3360, got {payload.get('variant')} step {payload.get('step')}"
        )
    state = payload.get("stateDict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no stateDict")
    model, architecture = pilot.make_model(architecture_path.resolve(), device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        "format": payload.get("format"),
        "variant": payload.get("variant"),
        "step": int(payload["step"]),
        "globalStep": int(payload.get("globalStep", payload["step"])),
        "checkpoint": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "architecture": architecture,
    }


def render_h50(
    model: torch.nn.Module,
    source: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual, timing = continuous.render_student(model, source, CONTRACT, device, batch_size)
    candidate = np.ascontiguousarray(source - residual, dtype=np.float32)
    if candidate.shape != source.shape or not np.isfinite(candidate).all():
        raise ValueError("Invalid continuous S86 output")
    return candidate, timing


def valid_cached_h50(
    audio_path: Path,
    metadata_path: Path,
    source: dict[str, Any],
    checkpoint_sha256: str,
) -> np.ndarray | None:
    if not audio_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        contract = metadata.get("contract", {})
        if metadata.get("sourceSha256") != source["sourceSha256"]:
            return None
        if metadata.get("checkpointSha256") != checkpoint_sha256:
            return None
        if contract != CONTRACT.as_dict(assembly="continuous-context-overlap-save"):
            return None
        audio = load_audio(audio_path)
        if int(metadata.get("frames", -1)) != audio.shape[0]:
            return None
        return audio
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def make_pair(h50: np.ndarray, inst3: np.ndarray) -> np.ndarray:
    if h50.shape != (SNIPPET_SAMPLES, 2) or inst3.shape != h50.shape:
        raise ValueError(f"Expected two 2-second stereo clips, got {h50.shape}, {inst3.shape}")
    silence = np.zeros((GAP_SAMPLES, 2), dtype=np.float32)
    pair = np.concatenate((h50, silence, inst3, silence), axis=0)
    if pair.shape != (PAIR_SAMPLES, 2) or not np.isfinite(pair).all():
        raise ValueError(f"Invalid pair shape: {pair.shape}")
    return np.ascontiguousarray(pair, dtype=np.float32)


def selected_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ms in EVENT_MS:
        prefix = str(ms)
        projection = [float(row["metrics"][prefix]["positiveProjectionDbfs"]) for row in rows]
        miss = [float(row["metrics"][prefix]["missRmsDbfs"]) for row in rows]
        result[prefix] = {
            "count": len(rows),
            "positiveProjectionP95Dbfs": float(np.percentile(projection, 95.0)),
            "positiveProjectionMaxDbfs": float(max(projection)),
            "missRmsP95Dbfs": float(np.percentile(miss, 95.0)),
            "missRmsMaxDbfs": float(max(miss)),
        }
    return result


def global_selected_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ms in EVENT_MS:
        key = str(ms)
        projection = np.asarray(
            [float(event["metrics"][key]["positiveProjectionDbfs"]) for event in events],
            dtype=np.float64,
        )
        miss = np.asarray(
            [float(event["metrics"][key]["missRmsDbfs"]) for event in events],
            dtype=np.float64,
        )
        result[key] = {
            "count": len(events),
            "positiveProjectionP50Dbfs": float(np.percentile(projection, 50.0)),
            "positiveProjectionP90Dbfs": float(np.percentile(projection, 90.0)),
            "positiveProjectionP95Dbfs": float(np.percentile(projection, 95.0)),
            "positiveProjectionMaxDbfs": float(np.max(projection)),
            "positiveProjectionAtOrAboveMinus20Dbfs": int(np.count_nonzero(projection >= -20.0)),
            "positiveProjectionAtOrAboveMinus25Dbfs": int(np.count_nonzero(projection >= -25.0)),
            "missRmsP50Dbfs": float(np.percentile(miss, 50.0)),
            "missRmsP90Dbfs": float(np.percentile(miss, 90.0)),
            "missRmsP95Dbfs": float(np.percentile(miss, 95.0)),
            "missRmsMaxDbfs": float(np.max(miss)),
        }
    return result


def csv_marks_blank(path: Path) -> bool:
    if not path.is_file():
        return True
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return all(not (row.get("mark") or "").strip() for row in csv.DictReader(handle))


def write_review_csv(path: Path, events: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("eventId"):
                        existing[row["eventId"]] = row
        except (OSError, csv.Error):
            existing = {}
    fields = (
        "mark", "eventId", "serial", "sourceBatch", "sourceOrder", "priorSplitRole",
        "category", "languageCode", "artistName", "trackName", "centerSeconds",
        "eventScoreDbfs", "positiveProjection50Dbfs", "positiveProjection100Dbfs",
        "positiveProjection200Dbfs", "pairedFile", "h50File", "inst3File",
        "mixtureFile", "license", "licenseUrl", "notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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
                    "notes": previous.get("notes", ""),
                }
            )


def process_song(
    song: dict[str, Any],
    serial_start: int,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = load_audio(Path(song["sourcePath"]))
    teacher = load_audio(Path(song["teacherPath"]))
    if source.shape != teacher.shape:
        raise ValueError(
            f"Source/Inst3 shape mismatch for {song['sourceBatch']} {song['sourceOrder']} "
            f"{source.shape} != {teacher.shape}"
        )
    h50_dir = args.output_root.resolve() / "full-song" / "h50ContinuationPlus5"
    h50_path = h50_dir / f"{song['sourceBatch']}-{int(song['sourceOrder']):03d}-{song['slug']}.flac"
    h50_meta_path = h50_path.with_suffix(".json")
    h50 = None if args.force else valid_cached_h50(h50_path, h50_meta_path, song, checkpoint_sha256)
    if h50 is None:
        h50, timing = render_h50(model, source, device, args.inference_batch_size)
        h50_output = write_flac(h50_path, h50)
        json_write(
            h50_meta_path,
            {
                "schema": "local-inst3-fma-s86-full-song-cache@1",
                "sourceBatch": song["sourceBatch"],
                "sourceOrder": song["sourceOrder"],
                "sourceSha256": song["sourceSha256"],
                "checkpointSha256": checkpoint_sha256,
                "checkpoint": str(args.checkpoint.resolve()),
                "variant": "S86-event-only@pass-5",
                "contract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
                "frames": int(h50.shape[0]),
                "output": h50_output,
                "timing": timing,
            },
        )
        reused_h50 = False
    else:
        timing = {"reused": True}
        reused_h50 = True
    candidates = event_tools.scan_candidates(
        source,
        teacher,
        h50,
        f"{song['sourceBatch']}-{song['slug']}",
        args.scan_hop_ms,
        args.candidates_per_song,
        args.coverage_bins,
    )
    if len(candidates) != args.candidates_per_song:
        raise ValueError(
            f"Expected {args.candidates_per_song} candidates for {song['sourceBatch']} {song['slug']}, "
            f"got {len(candidates)}"
        )
    events: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        serial = serial_start + candidate_index - 1
        event_id = (
            f"fma-all-s86-pass5-{song['sourceBatch']}-{int(song['sourceOrder']):03d}-"
            f"{safe_name(song['slug'])}-{candidate_index:02d}"
        )
        start = int(candidate["snippetStartSamples"])
        end = int(candidate["snippetEndSamples"])
        h50_clip = np.ascontiguousarray(h50[start:end], dtype=np.float32)
        inst3_clip = np.ascontiguousarray(teacher[start:end], dtype=np.float32)
        mixture_clip = np.ascontiguousarray(source[start:end], dtype=np.float32)
        if h50_clip.shape != (SNIPPET_SAMPLES, 2):
            raise ValueError(f"Invalid candidate slice for {event_id}: {h50_clip.shape}")
        base = f"{serial:04d}-{event_id}"
        outputs = {
            "h50ContinuationPlus5": write_flac(
                args.output_root / "events" / "h50ContinuationPlus5" / f"{base}.flac", h50_clip
            ),
            "inst3Instrumental": write_flac(
                args.output_root / "events" / "inst3Instrumental" / f"{base}.flac", inst3_clip
            ),
            "mixture": write_flac(
                args.output_root / "events" / "mixture" / f"{base}.flac", mixture_clip
            ),
            "pairedListening": write_flac(
                args.output_root / "events" / "pairedListening" / f"{base}.flac",
                make_pair(h50_clip, inst3_clip),
            ),
        }
        event = {
            "eventId": event_id,
            "serial": serial,
            "sourceBatch": song["sourceBatch"],
            "sourceOrder": song["sourceOrder"],
            "sourceId": song["sourceId"],
            "priorSplitRole": song["priorSplitRole"],
            "category": song["category"],
            "languageCode": song["languageCode"],
            "artistName": song["artistName"],
            "trackName": song["trackName"],
            "slug": song["slug"],
            "license": song["license"],
            "licenseUrl": song["licenseUrl"],
            "centerSamples": int(candidate["centerSamples"]),
            "centerSeconds": float(candidate["centerSeconds"]),
            "snippetStartSamples": start,
            "snippetEndSamples": end,
            "coverageFraction": float(candidate["coverageFraction"]),
            "eventScoreDbfs": float(candidate["eventScoreDbfs"]),
            "metrics": candidate["metrics"],
            "outputs": outputs,
        }
        events.append(event)
    song_report = {
        "sourceBatch": song["sourceBatch"],
        "sourceOrder": song["sourceOrder"],
        "sourceId": song["sourceId"],
        "slug": song["slug"],
        "artistName": song["artistName"],
        "trackName": song["trackName"],
        "category": song["category"],
        "languageCode": song["languageCode"],
        "priorSplitRole": song["priorSplitRole"],
        "sourcePath": song["sourcePath"],
        "sourceSha256": song["sourceSha256"],
        "teacherPath": song["teacherPath"],
        "h50Path": str(h50_path.resolve()),
        "h50Reused": reused_h50,
        "h50Timing": timing,
        "candidateCount": len(events),
        "selectedSummary": selected_summary(events),
        "eventIds": [event["eventId"] for event in events],
    }
    return song_report, events


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidates_per_song != 5:
        raise ValueError("This annotation preparation contract requires exactly 5 events per song")
    if args.inference_batch_size <= 0 or args.threads <= 0:
        raise ValueError("inference batch size and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    songs = build_song_records(args)
    if args.max_songs is not None:
        if args.max_songs <= 0:
            raise ValueError("--max-songs must be positive")
        songs = songs[: args.max_songs]
    checkpoint_sha256 = sha256_file(args.checkpoint)
    model, checkpoint_meta = load_model(args.checkpoint, args.architecture_checkpoint, device)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    selected_songs_path = args.output_root / "selected-songs.json"
    json_write(
        selected_songs_path,
        {
            "schema": "local-inst3-fma-y-selection@1",
            "selectionRule": "keep/selection == Y in both frozen FMA style reviews; N excluded",
            "songCount": len(songs),
            "songs": songs,
        },
    )
    progress_path = args.output_root / "progress.json"
    progress: dict[str, Any] = {}
    if progress_path.is_file() and not args.force:
        try:
            previous = read_json(progress_path)
            if previous.get("checkpointSha256") == checkpoint_sha256:
                progress = previous
        except (OSError, ValueError, json.JSONDecodeError):
            progress = {}
    song_reports: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, song in enumerate(songs, start=1):
        key = f"{song['sourceBatch']}:{song['sourceOrder']}"
        print(f"S86 continuous {index}/{len(songs)}: {song['sourceBatch']} {song['sourceOrder']} {song['slug']}", flush=True)
        song_report, song_events = process_song(
            song,
            len(events) + 1,
            model,
            device,
            args,
            checkpoint_sha256,
        )
        song_reports.append(song_report)
        events.extend(song_events)
        progress = {
            "schema": SCHEMA + ".progress",
            "checkpointSha256": checkpoint_sha256,
            "completedSongs": index,
            "songReports": song_reports,
            "events": events,
            "elapsedSeconds": time.perf_counter() - started,
        }
        json_write(progress_path, progress)
        write_review_csv(args.output_root / "human-review-template.csv", events)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(song_reports) != len(songs) or len(events) != len(songs) * args.candidates_per_song:
        raise AssertionError(f"Unexpected output counts: songs={len(song_reports)} events={len(events)}")
    review_csv_path = args.output_root / "human-review-template.csv"
    json_write(
        args.output_root / "event-listening-report.json",
        {
            "schema": SCHEMA,
            "status": "completed",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection": {
                "batches": ["batch1", "batch2"],
                "styleMark": "Y",
                "excludedStyleMark": "N",
                "songCount": len(song_reports),
                "eventCount": len(events),
                "eventsPerSong": args.candidates_per_song,
            },
            "checkpoint": checkpoint_meta,
            "architecture": checkpoint_meta["architecture"],
            "contract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
            "candidateScan": {
                "scanHopMs": args.scan_hop_ms,
                "coverageBins": args.coverage_bins,
                "eventMilliseconds": list(EVENT_MS),
                "selection": "event_tools.scan_candidates; distributed temporal coverage, then score",
            },
            "selectedHotspotSummary": global_selected_summary(events),
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
                "K": "Current H50 result is acceptable for this event.",
                "I": "H50 removed extra non-vocal content relative to Inst 3; this does not mean Inst 3 removed non-vocal content.",
                "blank": "Not reviewed yet.",
            },
            "songs": song_reports,
            "events": events,
            "validation": {
                "allSelectedSongsAreY": True,
                "styleRejectedSongsIncluded": 0,
                "uniqueSourceIds": len({song["sourceId"] for song in songs}),
                "candidateCountHistogram": dict(Counter(song["candidateCount"] for song in song_reports)),
                "pairFrames": PAIR_SAMPLES,
                "allMarksInitiallyBlank": csv_marks_blank(review_csv_path),
                "elapsedSeconds": time.perf_counter() - started,
            },
        },
    )
    Path(args.output_root / "human-review-key.txt").write_text(
        "S86-event-only@pass-5 versus native Inst 3\n"
        "\n"
        "Each paired file is: H50 2.0 s, 300 ms silence, Inst 3 2.0 s, 300 ms silence.\n"
        "Enter one mark in the CSV mark column: S, R, K, or I. Leave blank if not reviewed.\n"
        "S = especially conspicuous residual vocal and is a subset of R.\n"
        "R = further vocal/harmony/spoken/vocal-effect removal is desired.\n"
        "K = current result is acceptable.\n"
        "I = H50 removed extra non-vocal content relative to Inst 3; it does not mean Inst 3 removed non-vocal content.\n",
        encoding="utf-8",
    )
    print(
        f"completed {len(song_reports)} Y songs / {len(events)} events in "
        f"{time.perf_counter() - started:.1f}s: {args.output_root.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
