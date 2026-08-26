#!/usr/bin/env python3
"""Survey new FMA songs for concentrated, likely conspicuous vocal leaks.

The tool deliberately separates acquisition, inference, numeric ranking, and
retention.  It downloads a balanced batch from the already frozen FMA
candidate rules, renders native Inst 3 and the current residual-vocals
checkpoint with continuous 128-frame assembly, and keeps only songs with
multiple high-ranking short-event candidates.  The numeric ranking is a
screening proxy for an S mark; it is not a replacement for listening.

Non-retained source audio and intermediate renders are deleted after the
batch is ranked.  Retained songs keep their raw source, full H50/Inst 3
renders, and five temporally separated paired listening clips.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
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
import run_inst3_fma_sr_event_only_continuation as sr_continuation
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FMA_METADATA = ROOT / ".tmp" / "fma-metadata" / "fma_metadata.zip"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "fma-s-leakage-survey"
DEFAULT_CHECKPOINT = (
    ROOT
    / "data"
    / "modern-song-fma-sr-event-only-continuation"
    / "extensions"
    / "from-step-5120"
    / "runs"
    / "step-5648.pt"
)
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
SAMPLE_RATE = 44_100
EVENT_MS = (50, 100, 200)
SNIPPET_SAMPLES = 2 * SAMPLE_RATE
PAIR_GAP_SAMPLES = round(0.3 * SAMPLE_RATE)
PAIR_SAMPLES = 2 * SNIPPET_SAMPLES + 2 * PAIR_GAP_SAMPLES
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
CHECKPOINT_FORMAT = sr_continuation.CHECKPOINT_FORMAT
CATEGORY_ORDER = (
    "pop-synth",
    "hiphop-rnb",
    "dance-electronic",
    "latin-modern",
    "pop-rock",
    "singer-songwriter",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fma-metadata-zip", type=Path, default=DEFAULT_FMA_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--max-retain-per-batch", type=int, default=12)
    parser.add_argument("--candidates-per-song", type=int, default=16)
    parser.add_argument("--coverage-bins", type=int, default=8)
    parser.add_argument("--scan-hop-ms", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower() or "song"


def artist_key(value: Any) -> str:
    """Normalize case and repeated whitespace for artist-level deduplication."""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def db(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def load_audio(path: Path) -> np.ndarray:
    audio, rate = listening.load_audio(path)
    if rate != SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Unexpected audio contract for {path}: {audio.shape}, {rate}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite source audio: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio for {path}: {audio.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": 2,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def make_pair(h50: np.ndarray, inst3: np.ndarray) -> np.ndarray:
    if h50.shape != (SNIPPET_SAMPLES, 2) or inst3.shape != (SNIPPET_SAMPLES, 2):
        raise ValueError(f"Expected two 2-second clips, got {h50.shape}, {inst3.shape}")
    silence = np.zeros((PAIR_GAP_SAMPLES, 2), dtype=np.float32)
    pair = np.concatenate((h50, silence, inst3, silence), axis=0)
    if pair.shape != (PAIR_SAMPLES, 2):
        raise ValueError(f"Unexpected pair shape: {pair.shape}")
    return np.ascontiguousarray(pair, dtype=np.float32)


def manifest_records(path: Path) -> list[dict[str, Any]]:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return []
    records = value.get("records")
    return records if isinstance(records, list) else []


def previously_used_fma(root: Path) -> tuple[set[str], set[str], set[str]]:
    """Exclude earlier artists/IDs and URLs that already failed to download."""
    artists: set[str] = set()
    source_ids: set[str] = set()
    failed_ids: set[str] = set()
    data_root = root / "data"
    for path in data_root.rglob("source-manifest.json"):
        try:
            manifest = read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        for record in manifest_records(path):
            if str(record.get("source", "")).casefold() != "fma":
                continue
            artist = artist_key(record.get("artistName", ""))
            source_id = str(record.get("sourceId", "")).strip()
            if artist:
                artists.add(artist)
            if source_id:
                source_ids.add(source_id)
        for failure in manifest.get("failures", []):
            source_id = str(failure.get("sourceId", "")).strip()
            if source_id:
                failed_ids.add(source_id)
    return artists, source_ids, failed_ids


def next_batch_root(output_root: Path) -> Path:
    numbers = []
    for path in output_root.glob("batch-*"):
        match = re.fullmatch(r"batch-(\d+)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    return output_root / f"batch-{number:03d}"


def select_candidates(args: argparse.Namespace, batch_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import download_modern_song_candidates as downloader

    used_artists, used_ids, failed_ids = previously_used_fma(ROOT)
    candidates = downloader.load_fma_candidates(args.fma_metadata_zip.resolve(), used_artists)
    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORY_ORDER}
    for candidate in candidates:
        if str(candidate.get("sourceId", "")) in used_ids or str(candidate.get("sourceId", "")) in failed_ids:
            continue
        by_category.setdefault(candidate["category"], []).append(candidate)
    for values in by_category.values():
        values.sort(key=lambda item: (-float(item["rankingScore"]), item["artistName"].casefold(), item["sourceId"]))

    selected: list[dict[str, Any]] = []
    selected_artists = set(used_artists)
    skipped_categories: dict[str, str] = {}
    for category in CATEGORY_ORDER:
        values = by_category.get(category, [])
        if not values:
            skipped_categories[category] = "resource pool exhausted"
            continue
        count = 0
        for candidate in values:
            candidate_artist_key = artist_key(candidate["artistName"])
            if candidate_artist_key in selected_artists:
                continue
            selected_artists.add(candidate_artist_key)
            value = dict(candidate)
            value["batch"] = batch_root.name
            value["order"] = len(selected) + 1
            value["role"] = "numeric-S-screening"
            value["slug"] = safe_name(f"fma-{candidate['artistName']}-{candidate['trackName']}")
            selected.append(value)
            count += 1
            if count >= args.per_category:
                break
        if count < args.per_category:
            skipped_categories[category] = f"only {count}/{args.per_category} new artists available"

    if not selected:
        raise RuntimeError("No new FMA candidates remain under the frozen selection rules")
    selection = {
        "schema": "local-fma-s-leakage-survey-selection@1",
        "batch": batch_root.name,
        "requestedPerCategory": args.per_category,
        "selectedCount": len(selected),
        "categoryCounts": dict(Counter(item["category"] for item in selected)),
        "availableCounts": {category: len(by_category.get(category, [])) for category in CATEGORY_ORDER},
        "skippedCategories": skipped_categories,
        "artistPolicy": "one track per artist; every earlier FMA artist excluded, including style-N tracks",
        "sourcePolicy": "existing frozen FMA candidate rules; CC Attribution-family, no ND, vocal evidence, 150-330 s",
        "usedArtistCountBeforeBatch": len(used_artists),
        "usedSourceIdCountBeforeBatch": len(used_ids),
        "failedSourceIdCountBeforeBatch": len(failed_ids),
        "records": selected,
    }
    return selected, selection


def download_batch(args: argparse.Namespace, batch_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = batch_root / "source-manifest.json"
    if manifest_path.is_file() and not args.force:
        manifest = read_json(manifest_path)
        records = manifest.get("records")
        if manifest.get("status") in {"downloaded", "processing", "completed"} and isinstance(records, list) and records:
            return records, manifest
    records, selection = select_candidates(args, batch_root)
    raw_root = batch_root / "raw-originals"
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        **selection,
        "status": "downloading",
        "audioProcessing": {
            "decoded": False,
            "teacherRun": False,
            "studentRun": False,
            "metricsComputed": False,
            "snippetsGenerated": False,
        },
        "failures": failures,
        "records": completed,
    }
    json_write(manifest_path, manifest)
    import download_modern_song_candidates as downloader

    for candidate in records:
        destination = raw_root / f"{int(candidate['order']):03d}-{candidate['slug']}.mp3"
        try:
            value = dict(candidate)
            value["download"] = downloader.download_raw(candidate["downloadUrl"], destination, args.force)
            completed.append(value)
            manifest["records"] = completed
            json_write(manifest_path, manifest)
            print(
                f"downloaded {len(completed)}/{len(records)} {candidate['category']}: "
                f"{candidate['artistName']} - {candidate['trackName']}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - retain failure and continue batch
            failures.append(
                {
                    "artistName": candidate.get("artistName"),
                    "trackName": candidate.get("trackName"),
                    "sourceId": candidate.get("sourceId"),
                    "url": candidate.get("downloadUrl"),
                    "error": repr(error),
                }
            )
            json_write(manifest_path, manifest)
            print(f"download failed {candidate['artistName']} - {candidate['trackName']}: {error}", flush=True)
    if not completed:
        raise RuntimeError("All candidate downloads failed")
    manifest["status"] = "downloaded"
    manifest["downloadSummary"] = {
        "fileCount": len(completed),
        "categoryCounts": dict(Counter(item["category"] for item in completed)),
        "totalBytes": sum(int(item["download"]["bytes"]) for item in completed),
        "sha256Recorded": True,
    }
    manifest["records"] = completed
    json_write(manifest_path, manifest)
    return completed, manifest


def load_checkpoint_model(
    checkpoint: Path, architecture: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unexpected checkpoint format: {payload.get('format')}")
    model, architecture_meta = pilot.make_model(architecture.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "file": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "format": payload.get("format"),
        "step": int(payload.get("step", -1)),
        "continuationPass": int(payload.get("continuationPass", -1)),
        "variant": payload.get("variant"),
        "architecture": architecture_meta.get("checkpoint"),
    }


def event_proxy(event: dict[str, Any]) -> float:
    """Higher means a stronger numeric S screening signal."""
    metrics = event["metrics"]
    p100 = float(metrics["100"]["positiveProjectionDbfs"])
    p200 = float(metrics["200"]["positiveProjectionDbfs"])
    score = float(event["eventScoreDbfs"])
    # 200 ms was the most stable projection for S versus non-S in the prior
    # reviewed FMA set; shorter scales retain sensitivity to consonant bursts.
    return 0.55 * p200 + 0.30 * p100 + 0.15 * score


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return -240.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def select_event_rows(events: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: (-float(item["sProxyDbfs"]), int(item["centerSamples"])))
    selected: list[dict[str, Any]] = []
    centers: list[int] = []
    for separation in (round(1.0 * SAMPLE_RATE), round(0.5 * SAMPLE_RATE), 0):
        for event in ordered:
            if event in selected:
                continue
            center = int(event["centerSamples"])
            if all(abs(center - other) >= separation for other in centers):
                selected.append(event)
                centers.append(center)
            if len(selected) >= count:
                return selected
    return selected


def write_event_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fields = (
        "mark", "eventId", "serial", "sourceOrder", "category", "artistName", "trackName",
        "centerSeconds", "eventScoreDbfs", "sProxyDbfs", "positiveProjection50Dbfs",
        "positiveProjection100Dbfs", "positiveProjection200Dbfs", "pairedFile", "h50File",
        "inst3File", "mixtureFile", "license", "licenseUrl", "notes",
    )
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
                    "sourceOrder": event["sourceOrder"],
                    "category": event["category"],
                    "artistName": event["artistName"],
                    "trackName": event["trackName"],
                    "centerSeconds": f"{float(event['centerSeconds']):.3f}",
                    "eventScoreDbfs": f"{float(event['eventScoreDbfs']):.2f}",
                    "sProxyDbfs": f"{float(event['sProxyDbfs']):.2f}",
                    "positiveProjection50Dbfs": f"{float(metrics['50']['positiveProjectionDbfs']):.2f}",
                    "positiveProjection100Dbfs": f"{float(metrics['100']['positiveProjectionDbfs']):.2f}",
                    "positiveProjection200Dbfs": f"{float(metrics['200']['positiveProjectionDbfs']):.2f}",
                    "pairedFile": event["outputs"]["pairedListening"]["file"],
                    "h50File": event["outputs"]["h50ContinuationPlus5"]["file"],
                    "inst3File": event["outputs"]["inst3Instrumental"]["file"],
                    "mixtureFile": event["outputs"]["mixture"]["file"],
                    "license": event["license"],
                    "licenseUrl": event["licenseUrl"],
                    "notes": "numeric S proxy; human mark pending",
                }
            )


def write_review_key(path: Path) -> None:
    path.write_text(
        "FMA numeric-S screening event review\n"
        "\n"
        "Each paired file is: H50-continuation+5 2.0 s, 300 ms silence, "
        "Inst 3 instrumental 2.0 s, 300 ms trailing silence.\n"
        "\n"
        "Enter one mark in the CSV mark column:\n"
        "S = especially conspicuous residual vocal; subset of R\n"
        "R = further vocal, harmony, spoken, or vocal-effect removal is desired\n"
        "K = current result is acceptable\n"
        "I = H50 removed extra non-vocal content relative to Inst 3; this does "
        "not mean Inst 3 removed non-vocal content\n"
        "\n"
        "sProxyDbfs and projection values are numerical screening signals only; "
        "they do not replace listening. Leave mark blank until reviewed.\n",
        encoding="utf-8",
    )


def process_batch(args: argparse.Namespace, batch_root: Path, records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    report_path = batch_root / "survey-report.json"
    existing = read_json(report_path) if report_path.is_file() and not args.force else {}
    report: dict[str, Any] = existing or {
        "schema": "local-fma-s-leakage-survey@1",
        "status": "processing",
        "batch": batch_root.name,
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "checkpoint": {},
        "selection": {
            "proxy": "0.55*p200 + 0.30*p100 + 0.15*eventScore; higher is a numeric S screening signal",
            "eventCountPerSong": args.candidates_per_song,
            "topDecileRule": "retain songs with at least two events in the batch top 10% of sProxy",
            "pairFormat": "H50 2.0 s + 300 ms silence + Inst 3 2.0 s + 300 ms trailing silence",
        },
        "songs": {},
        "failures": [],
    }
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, checkpoint_meta = load_checkpoint_model(args.checkpoint.resolve(), args.architecture_checkpoint.resolve(), device)
    report["checkpoint"] = checkpoint_meta
    report["runtime"] = {
        "device": str(device),
        "threads": args.threads,
        "inferenceBatchSize": args.inference_batch_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    json_write(report_path, report)
    session, providers = pilot.make_teacher_session(pilot.DEFAULT_TEACHER.resolve(), args.threads, True)
    report["teacher"] = {
        "file": str(pilot.DEFAULT_TEACHER.resolve()),
        "sha256": sha256_file(pilot.DEFAULT_TEACHER),
        "contractId": "uvr_mdxnet_inst_3@2",
        "providers": list(providers),
    }
    work_root = batch_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, record in enumerate(records, start=1):
            slug = record["slug"]
            if slug in report["songs"] and report["songs"][slug].get("status") == "completed" and not args.force:
                print(f"reuse processed {index}/{len(records)}: {slug}", flush=True)
                continue
            raw_path = Path(record["download"]["file"])
            try:
                print(f"process {index}/{len(records)}: {record['artistName']} - {record['trackName']}", flush=True)
                source = load_audio(raw_path)
                teacher, teacher_residual, teacher_timing = oracle.render_teacher_audio(
                    session, source, progress_label=f"Inst3 {index}/{len(records)} {slug}"
                )
                residual, h50_timing = continuous.render_student(
                    model, source, CONTRACT, device, args.inference_batch_size
                )
                h50 = np.ascontiguousarray(source - residual, dtype=np.float32)
                if teacher.shape != source.shape or h50.shape != source.shape:
                    raise ValueError(f"shape mismatch source={source.shape} teacher={teacher.shape} h50={h50.shape}")
                candidates = event_tools.scan_candidates(
                    source, teacher, h50, slug, args.scan_hop_ms, args.candidates_per_song, args.coverage_bins
                )
                for event_index, event in enumerate(candidates, start=1):
                    event["eventIndex"] = event_index
                    event["sProxyDbfs"] = event_proxy(event)
                work_h50 = work_root / f"{slug}-h50-instrumental.flac"
                work_residual = work_root / f"{slug}-h50-residual.flac"
                work_inst3 = work_root / f"{slug}-inst3-instrumental.flac"
                write_flac(work_h50, h50)
                write_flac(work_residual, residual)
                write_flac(work_inst3, teacher)
                report["songs"][slug] = {
                    "status": "completed",
                    "source": {
                        "source": record["source"],
                        "sourceId": record["sourceId"],
                        "sourceOrder": record.get("order", ""),
                        "artistName": record["artistName"],
                        "trackName": record["trackName"],
                        "category": record["category"],
                        "languageCode": record.get("languageCode", ""),
                        "durationSeconds": float(source.shape[0] / SAMPLE_RATE),
                        "license": record["license"],
                        "licenseUrl": record["licenseUrl"],
                        "sourceUrl": record["sourceUrl"],
                        "rawFile": str(raw_path.resolve()),
                        "rawSha256": record["download"]["sha256"],
                    },
                    "timing": {"teacher": teacher_timing, "h50": h50_timing},
                    "workOutputs": {
                        "h50ContinuationPlus5": str(work_h50.resolve()),
                        "h50Residual": str(work_residual.resolve()),
                        "inst3Instrumental": str(work_inst3.resolve()),
                    },
                    "events": candidates,
                }
                manifest["status"] = "processing"
                manifest["audioProcessing"] = {"decoded": True, "teacherRun": True, "studentRun": True, "metricsComputed": True, "snippetsGenerated": False}
                json_write(batch_root / "source-manifest.json", manifest)
                json_write(report_path, report)
                del source, teacher, teacher_residual, residual, h50
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as error:  # noqa: BLE001 - preserve per-song evidence
                report["failures"].append({"slug": slug, "artistName": record.get("artistName"), "trackName": record.get("trackName"), "error": repr(error)})
                report["songs"][slug] = {"status": "failed", "error": repr(error), "source": {"artistName": record.get("artistName"), "trackName": record.get("trackName"), "category": record.get("category")}}
                json_write(report_path, report)
                print(f"process failed {slug}: {error}", flush=True)
    finally:
        del session, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finalize_batch(args, batch_root, records, manifest, report)
    return report


def finalize_batch(args: argparse.Namespace, batch_root: Path, records: list[dict[str, Any]], manifest: dict[str, Any], report: dict[str, Any]) -> None:
    completed = [value for value in report["songs"].values() if value.get("status") == "completed"]
    all_events = []
    for slug, song in report["songs"].items():
        if song.get("status") != "completed":
            continue
        for event in song.get("events", []):
            all_events.append({"songSlug": slug, **event})
    values = [float(event["sProxyDbfs"]) for event in all_events]
    top_decile = percentile(values, 0.90)
    top_quartile = percentile(values, 0.75)
    for event in all_events:
        event["topDecile"] = float(event["sProxyDbfs"]) >= top_decile
        event["topQuartile"] = float(event["sProxyDbfs"]) >= top_quartile
    song_rows = []
    for slug, song in report["songs"].items():
        if song.get("status") != "completed":
            continue
        events = [event for event in all_events if event["songSlug"] == slug]
        proxies = sorted((float(event["sProxyDbfs"]) for event in events), reverse=True)
        row = {
            "slug": slug,
            **song["source"],
            "candidateCount": len(events),
            "topDecileCount": sum(bool(event["topDecile"]) for event in events),
            "topQuartileCount": sum(bool(event["topQuartile"]) for event in events),
            "maxSProxyDbfs": proxies[0] if proxies else -240.0,
            "top3MeanSProxyDbfs": float(np.mean(proxies[:3])) if proxies else -240.0,
        }
        song_rows.append(row)
    eligible = [row for row in song_rows if row["topDecileCount"] >= 2]
    eligible.sort(key=lambda row: (-row["topDecileCount"], -row["topQuartileCount"], -row["top3MeanSProxyDbfs"], row["slug"]))
    retained = eligible[: args.max_retain_per_batch]
    retained_slugs = {row["slug"] for row in retained}
    retained_root = batch_root / "retained"
    event_root = retained_root / "events"
    retained_events: list[dict[str, Any]] = []
    serial = 1
    for row in retained:
        slug = row["slug"]
        song = report["songs"][slug]
        song_root = retained_root / "songs" / slug
        full_root = song_root / "full"
        full_root.mkdir(parents=True, exist_ok=True)
        for key, filename in (("h50ContinuationPlus5", "h50-instrumental.flac"), ("h50Residual", "h50-residual.flac"), ("inst3Instrumental", "inst3-instrumental.flac")):
            source_path = Path(song["workOutputs"][key])
            destination = full_root / filename
            if source_path.is_file():
                shutil.move(str(source_path), str(destination))
            song.setdefault("retainedOutputs", {})[key] = str(destination.resolve())
        source_audio = load_audio(Path(song["source"]["rawFile"]))
        h50_audio = load_audio(full_root / "h50-instrumental.flac")
        inst3_audio = load_audio(full_root / "inst3-instrumental.flac")
        chosen = select_event_rows([event for event in all_events if event["songSlug"] == slug], 5)
        for event_index, event in enumerate(chosen, start=1):
            start = int(event["snippetStartSamples"])
            end = int(event["snippetEndSamples"])
            h50_clip = h50_audio[start:end]
            inst3_clip = inst3_audio[start:end]
            mixture_clip = source_audio[start:end]
            prefix = f"{serial:04d}-{slug}-{event_index:02d}"
            outputs = {
                "mixture": write_flac(event_root / "mixture" / f"{prefix}.flac", mixture_clip),
                "h50ContinuationPlus5": write_flac(event_root / "h50ContinuationPlus5" / f"{prefix}.flac", h50_clip),
                "inst3Instrumental": write_flac(event_root / "inst3Instrumental" / f"{prefix}.flac", inst3_clip),
                "pairedListening": write_flac(event_root / "pairedListening" / f"{prefix}.flac", make_pair(h50_clip, inst3_clip)),
            }
            retained_event = {
                "eventId": f"{batch_root.name}-{slug}-{event_index:02d}",
                "serial": serial,
                "sourceOrder": row.get("sourceOrder", ""),
                "category": row["category"],
                "artistName": row["artistName"],
                "trackName": row["trackName"],
                "centerSeconds": event["centerSeconds"],
                "eventScoreDbfs": event["eventScoreDbfs"],
                "sProxyDbfs": event["sProxyDbfs"],
                "metrics": event["metrics"],
                "topDecile": event["topDecile"],
                "outputs": outputs,
                "license": row["license"],
                "licenseUrl": row["licenseUrl"],
                "notes": "numeric S proxy; human mark pending",
            }
            retained_events.append(retained_event)
            serial += 1
        song["retained"] = True
        song["retainedEventIds"] = [event["eventId"] for event in retained_events if event["artistName"] == row["artistName"] and event["trackName"] == row["trackName"]]
        del source_audio, h50_audio, inst3_audio
    for row in song_rows:
        if row["slug"] not in retained_slugs:
            song = report["songs"][row["slug"]]
            raw_path = Path(song["source"].get("rawFile", ""))
            if raw_path.is_file():
                raw_path.unlink()
            for path in song.get("workOutputs", {}).values():
                Path(path).unlink(missing_ok=True)
            song["retained"] = False
    shutil.rmtree(batch_root / "work", ignore_errors=True)
    write_event_csv(retained_root / "human-review-template.csv", retained_events)
    write_review_key(retained_root / "human-review-key.txt")
    with (retained_root / "retained-songs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("slug", "category", "artistName", "trackName", "sourceId", "topDecileCount", "topQuartileCount", "maxSProxyDbfs", "top3MeanSProxyDbfs", "rawFile", "license", "licenseUrl")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in retained:
            writer.writerow({key: row.get(key, "") for key in fields})
    report["ranking"] = {
        "eventCount": len(all_events),
        "topDecileThresholdDbfs": top_decile,
        "topQuartileThresholdDbfs": top_quartile,
        "songRows": sorted(song_rows, key=lambda row: (-row["topDecileCount"], -row["maxSProxyDbfs"], row["slug"])),
        "eligibleCount": len(eligible),
        "retainedCount": len(retained),
        "retained": retained,
        "rule": "retain at most max-retain-per-batch songs with >=2 events in batch top 10% of numeric S proxy",
        "caution": "S proxy is numerical screening only; confirm with paired listening and do not treat it as an S label",
    }
    report["status"] = "completed"
    report["retainedEvents"] = retained_events
    report["finishedAtEpoch"] = time.time()
    json_write(batch_root / "survey-report.json", report)
    manifest["status"] = "completed"
    manifest["audioProcessing"] = {"decoded": True, "teacherRun": True, "studentRun": True, "metricsComputed": True, "snippetsGenerated": bool(retained_events)}
    manifest["retention"] = {"retainedSongCount": len(retained), "retainedEventCount": len(retained_events), "report": str((batch_root / "survey-report.json").resolve())}
    json_write(batch_root / "source-manifest.json", manifest)
    update_cumulative(args.output_root.resolve())
    print(json.dumps({"status": report["status"], "batch": batch_root.name, "processedSongs": len(song_rows), "retainedSongs": len(retained), "retainedEvents": len(retained_events), "topDecileThresholdDbfs": top_decile, "report": str((batch_root / "survey-report.json").resolve()), "listeningCsv": str((retained_root / "human-review-template.csv").resolve())}, ensure_ascii=False, indent=2), flush=True)


def update_cumulative(output_root: Path) -> None:
    rows = []
    for report_path in sorted(output_root.glob("batch-*/survey-report.json")):
        try:
            report = read_json(report_path)
        except Exception:  # noqa: BLE001
            continue
        for row in report.get("ranking", {}).get("retained", []):
            rows.append({"batch": report.get("batch"), **row})
    json_write(output_root / "cumulative-retained.json", {"schema": "local-fma-s-leakage-survey-cumulative@1", "retainedSongCount": len(rows), "songs": rows})
    with (output_root / "cumulative-retained.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("batch", "slug", "category", "artistName", "trackName", "topDecileCount", "topQuartileCount", "maxSProxyDbfs")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.per_category <= 0 or args.max_retain_per_batch <= 0 or args.candidates_per_song <= 0:
        raise ValueError("batch and candidate counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_root = args.batch_root.resolve() if args.batch_root else next_batch_root(output_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        records, selection = select_candidates(args, batch_root)
        print(json.dumps({
            "batch": batch_root.name,
            "records": len(records),
            "categories": dict(Counter(item["category"] for item in records)),
            "available": selection["availableCounts"],
            "skippedCategories": selection["skippedCategories"],
        }, ensure_ascii=False, indent=2))
        return 0
    records, manifest = download_batch(args, batch_root)
    process_batch(args, batch_root, records, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
