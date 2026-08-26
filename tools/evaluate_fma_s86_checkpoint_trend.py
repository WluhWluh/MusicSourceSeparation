#!/usr/bin/env python3
"""Measure S86 FMA continuation checkpoints on unused style-approved songs.

The primary metric is the 100 ms positive projection of the candidate
instrumental error onto the content removed by native Inst 3.  The event
centres are the 5 fixed centres per song from the existing all-S86 FMA
listening manifest, so every checkpoint is compared on exactly the same
regions.

This tool intentionally writes metrics only.  It does not write rendered
audio.  The source list is the all-S86 FMA review; songs present in any
recorded S86 training selection are excluded at song level.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as external
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_REPORT = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "event-listening-report.json"
DEFAULT_SELECTED_SONGS = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "selected-songs.json"
DEFAULT_ARCHITECTURE = ROOT / "models" / "tfc-tdf" / "source" / "vocals_epoch=891.ckpt"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-s86-checkpoint-trend-all-unused-y"
DEFAULT_FIRST_SOURCE = ROOT / "data" / "modern-song-s-r-continuation" / "runs" / "S86-event-only" / "step-3360.pt"
DEFAULT_FIRST_ROOT = ROOT / "data" / "modern-song-fma-sr-event-only-continuation"
DEFAULT_SECOND_ROOT = ROOT / "data" / "modern-song-fma-s-leakage-survey-continuation"
DEFAULT_TRAINING_SELECTIONS = (
    ROOT / "data" / "modern-song-s-only-stress" / "s-pool-selection.json",
    ROOT / "data" / "modern-song-s-r-continuation" / "selection.json",
    ROOT / "data" / "modern-song-fma-sr-event-only-continuation" / "selection.json",
    ROOT / "data" / "modern-song-fma-s-leakage-survey-continuation" / "pool-selection.json",
)
SAMPLE_RATE = 44_100
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
EVENT_MS = (50, 100, 200)


def default_checkpoints() -> list[dict[str, Any]]:
    first_root = DEFAULT_FIRST_ROOT / "runs"
    extension_root = DEFAULT_FIRST_ROOT / "extensions" / "from-step-5120" / "runs"
    second_root = DEFAULT_SECOND_ROOT / "runs"
    return [
        {
            "key": "first-p00-step-3360",
            "stage": "first",
            "pass": 0,
            "localPass": 0,
            "step": 3360,
            "path": DEFAULT_FIRST_SOURCE,
        },
        {
            "key": "first-p01-step-3536",
            "stage": "first",
            "pass": 1,
            "localPass": 1,
            "step": 3536,
            "path": first_root / "step-3536.pt",
        },
        {
            "key": "first-p03-step-3888",
            "stage": "first",
            "pass": 3,
            "localPass": 3,
            "step": 3888,
            "path": first_root / "step-3888.pt",
        },
        {
            "key": "first-p05-step-4240",
            "stage": "first",
            "pass": 5,
            "localPass": 5,
            "step": 4240,
            "path": first_root / "step-4240.pt",
        },
        {
            "key": "first-p08-step-4768",
            "stage": "first",
            "pass": 8,
            "localPass": 8,
            "step": 4768,
            "path": first_root / "step-4768.pt",
        },
        {
            "key": "first-p10-step-5120",
            "stage": "first",
            "pass": 10,
            "localPass": 10,
            "step": 5120,
            "path": first_root / "step-5120.pt",
        },
        {
            "key": "first-p11-step-5296",
            "stage": "first",
            "pass": 11,
            "localPass": 11,
            "step": 5296,
            "path": extension_root / "step-5296.pt",
        },
        {
            "key": "first-p13-step-5648",
            "stage": "first",
            "pass": 13,
            "localPass": 13,
            "step": 5648,
            "path": extension_root / "step-5648.pt",
        },
        {
            "key": "second-p01-step-5824",
            "stage": "second",
            "pass": 14,
            "localPass": 1,
            "step": 5824,
            "path": second_root / "step-5824.pt",
        },
        {
            "key": "second-p05-step-6528",
            "stage": "second",
            "pass": 18,
            "localPass": 5,
            "step": 6528,
            "path": second_root / "step-6528.pt",
        },
        {
            "key": "second-p08-step-7056",
            "stage": "second",
            "pass": 21,
            "localPass": 8,
            "step": 7056,
            "path": second_root / "step-7056.pt",
        },
        {
            "key": "second-p10-step-7408",
            "stage": "second",
            "pass": 23,
            "localPass": 10,
            "step": 7408,
            "path": second_root / "step-7408.pt",
        },
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-songs", type=Path, default=DEFAULT_SELECTED_SONGS)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_EVENT_REPORT)
    parser.add_argument(
        "--training-selection",
        type=Path,
        action="append",
        default=None,
        help="Selection JSON to exclude; may be supplied more than once",
    )
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
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


def checkpoint_meta(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "key": item["key"],
        "stage": item["stage"],
        "pass": int(item["pass"]),
        "localPass": int(item["localPass"]),
        "step": int(item["step"]),
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": payload.get("format"),
        "variant": payload.get("variant"),
        "continuationPass": payload.get("continuationPass"),
        "sourceStep": payload.get("sourceStep"),
    }


def normalize_slug(value: str) -> str:
    """Normalize the several source/event key formats used by the runs."""
    slug = str(value).split("::")[-1]
    slug = re.sub(r"^modern-\d+-", "", slug)
    slug = re.sub(r"^batch2-full-\d+-", "", slug)
    slug = re.sub(r"^batch2-\d+-", "", slug)
    slug = re.sub(r"^fma-train-sr-s86-pass5-16-", "", slug)
    slug = re.sub(r"^\d+-", "", slug)
    return slug


def slug_from_event_id(event_id: str) -> str:
    slug = normalize_slug(event_id)
    slug = re.sub(r"-\d+$", "", slug)
    return slug


def load_training_slugs(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    """Collect song-level exclusions from every S86-related selection file."""
    trained: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for path in paths:
        path = path.resolve()
        payload = read_json(path)
        found: set[str] = set()
        for field in ("songKeys", "eventIds", "sEventIds", "rEventIds"):
            values = payload.get(field, [])
            if isinstance(values, list):
                for value in values:
                    text = str(value)
                    found.add(
                        slug_from_event_id(text)
                        if field in {"eventIds", "sEventIds", "rEventIds"}
                        else normalize_slug(text)
                    )
        for field in ("songs", "currentSongs", "previousSongs"):
            values = payload.get(field)
            if isinstance(values, dict):
                found.update(normalize_slug(str(key)) for key in values)
            elif isinstance(values, list):
                for value in values:
                    if isinstance(value, dict) and value.get("slug"):
                        found.add(normalize_slug(str(value["slug"])))
                    elif isinstance(value, str):
                        found.add(normalize_slug(value))
        # Non-FMA keys are harmless, but retaining only the normalized FMA
        # namespace makes the exclusion audit easier to read.
        found = {slug for slug in found if slug.startswith("fma-")}
        trained.update(found)
        provenance.append(
            {
                "file": str(path),
                "sha256": sha256_file(path),
                "songCount": len(found),
                "songs": sorted(found),
            }
        )
    return trained, provenance


def load_targets(
    selected_songs_path: Path,
    event_report_path: Path,
    training_selection_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    """Load every style-approved FMA song not present in a training selection."""
    selected_payload = read_json(selected_songs_path)
    selected_rows = selected_payload.get("songs", [])
    selected_by_slug = {
        normalize_slug(str(row["slug"])): row
        for row in selected_rows
        if isinstance(row, dict) and row.get("slug") and str(row.get("styleMark", "")).upper() == "Y"
    }
    if not selected_by_slug:
        raise ValueError(f"No Y songs found in {selected_songs_path}")

    event_report = read_json(event_report_path)
    report_songs = {
        normalize_slug(str(row["slug"])): row
        for row in event_report.get("songs", [])
        if isinstance(row, dict) and row.get("slug")
    }
    events_by_slug: dict[str, list[dict[str, Any]]] = {}
    for event in event_report.get("events", []):
        if not isinstance(event, dict) or not event.get("slug"):
            continue
        events_by_slug.setdefault(normalize_slug(str(event["slug"])), []).append(event)

    trained_slugs, training_provenance = load_training_slugs(training_selection_paths)
    targets: list[dict[str, Any]] = []
    for slug in sorted(selected_by_slug):
        if slug in trained_slugs:
            continue
        song = report_songs.get(slug)
        if song is None:
            raise FileNotFoundError(f"Y song has no event-report entry: {slug}")
        events = sorted(events_by_slug.get(slug, []), key=lambda item: int(item["centerSamples"]))
        if len(events) != 5:
            raise ValueError(f"Expected 5 frozen events for {slug}, got {len(events)}")
        source_path = Path(song["sourcePath"]).resolve()
        teacher_path = Path(song["teacherPath"]).resolve()
        if not source_path.is_file() or not teacher_path.is_file():
            raise FileNotFoundError(f"Missing source or teacher for {slug}")
        selected = selected_by_slug[slug]
        targets.append(
            {
                "slug": slug,
                "artistName": str(song.get("artistName", selected.get("artistName", ""))),
                "trackName": str(song.get("trackName", selected.get("trackName", ""))),
                "source": "fma",
                "sourceId": selected.get("sourceId", song.get("sourceId")),
                "sourceBatch": selected.get("sourceBatch", song.get("sourceBatch")),
                "sourceOrder": selected.get("sourceOrder", song.get("sourceOrder")),
                "category": selected.get("category", song.get("category")),
                "languageCode": selected.get("languageCode", song.get("languageCode", "")),
                "priorSplitRole": selected.get("priorSplitRole", song.get("priorSplitRole")),
                "testStratum": "unused-y",
                "sourcePath": str(source_path),
                "teacherPath": str(teacher_path),
                "sourceSha256": selected.get("sourceSha256", song.get("sourceSha256")),
                "teacherSha256": sha256_file(teacher_path),
                "license": selected.get("license", song.get("license")),
                "licenseUrl": selected.get("licenseUrl", song.get("licenseUrl")),
                "sourceUrl": selected.get("sourceUrl", song.get("sourceUrl")),
                "events": [
                    {
                        "eventId": event["eventId"],
                        "centerSamples": int(event["centerSamples"]),
                        "centerSeconds": float(event["centerSeconds"]),
                    }
                    for event in events
                ],
            }
        )
    return targets, trained_slugs, training_provenance


def verify_unseen(targets: list[dict[str, Any]], trained_slugs: set[str]) -> None:
    overlap = sorted(trained_slugs.intersection(target["slug"] for target in targets))
    if overlap:
        raise ValueError(f"Test songs overlap a training pool: {overlap}")


def load_model(
    architecture_checkpoint: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, architecture = pilot.make_model(architecture_checkpoint.resolve(), device)
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    state = payload.get("stateDict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no stateDict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "architecture": architecture["checkpoint"],
        "format": payload.get("format"),
        "variant": payload.get("variant"),
        "step": payload.get("step"),
        "continuationPass": payload.get("continuationPass"),
    }


def load_audio_file(path: Path) -> np.ndarray:
    audio, sample_rate = listening.load_audio(path)
    if sample_rate != SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Unexpected audio contract for {path}: {audio.shape}, {sample_rate}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_teacher_file(path: Path) -> np.ndarray:
    # Earlier manifests store the teacher as NPZ; the all-FMA event report
    # stores the same full-song instrumental as FLAC.
    if path.suffix.lower() == ".npz":
        audio = external.load_teacher(path)
    else:
        audio = load_audio_file(path)
    return np.ascontiguousarray(audio, dtype=np.float32)


def event_bounds(center: int, milliseconds: int, total_samples: int) -> tuple[int, int]:
    length = max(1, round(SAMPLE_RATE * milliseconds / 1000.0))
    start = center - length // 2
    end = start + length
    if start < 0 or end > total_samples:
        raise ValueError(
            f"Event block outside source: center={center}, ms={milliseconds}, samples={total_samples}"
        )
    return start, end


def render_student_event_regions(
    model: torch.nn.Module,
    audio: np.ndarray,
    targets: list[dict[str, Any]],
    contract: ShortWindowContract,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer only assembly windows intersecting the fixed metric blocks.

    The selected output samples are identical to ``render_student`` for the
    same song: starts remain on the global stride grid and each input uses
    continuous left/right context. Samples outside event blocks are unused.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    window_count = math.ceil(audio.shape[0] / contract.stride_samples)
    needed: set[int] = set()
    for target in targets:
        for event in target["events"]:
            for milliseconds in EVENT_MS:
                start, end = event_bounds(
                    int(event["centerSamples"]), milliseconds, audio.shape[0]
                )
                first = max(0, start // contract.stride_samples)
                last = min(window_count - 1, (end - 1) // contract.stride_samples)
                needed.update(range(first, last + 1))
    indices = sorted(needed)
    residual = np.zeros_like(audio)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            starts = [index * contract.stride_samples for index in batch_indices]
            lengths = [min(contract.stride_samples, audio.shape[0] - start) for start in starts]
            spectra = np.concatenate(
                [
                    continuous.stft_centered(
                        continuous.assemble_input(
                            audio,
                            start,
                            length,
                            contract,
                            mode="continuous",
                        ),
                        contract,
                    )
                    for start, length in zip(starts, lengths)
                ],
                axis=0,
            )
            predictions = model(torch.from_numpy(spectra).to(device)).detach().cpu().numpy()
            for offset, (start, length) in enumerate(zip(starts, lengths)):
                reconstructed = continuous.istft_centered(
                    predictions[offset : offset + 1], contract
                )
                begin = contract.left_context_samples
                residual[start : start + length] = reconstructed[begin : begin + length]
    if not np.isfinite(residual).all():
        raise ValueError("Student residual contains non-finite values")
    return residual, {
        "windowCount": window_count,
        "evaluatedWindowCount": len(indices),
        "boundaries": max(0, window_count - 1),
        "elapsedSeconds": time.perf_counter() - started,
        "inferenceBatchSize": batch_size,
    }


def db(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def summarize(values: list[float], song_values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    array = np.asarray(values, dtype=np.float64)
    per_song = np.asarray(song_values, dtype=np.float64)
    return {
        "eventCount": int(array.size),
        "songCount": int(per_song.size),
        "pooledP50Dbfs": float(np.percentile(array, 50.0)),
        "pooledP90Dbfs": float(np.percentile(array, 90.0)),
        "pooledP95Dbfs": float(np.percentile(array, 95.0)),
        "pooledMaxDbfs": float(np.max(array)),
        "meanPerSongP95Dbfs": float(np.mean(per_song)),
        "medianPerSongP95Dbfs": float(np.median(per_song)),
        "minPerSongP95Dbfs": float(np.min(per_song)),
        "maxPerSongP95Dbfs": float(np.max(per_song)),
    }


def evaluate_song(
    source: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    target: dict[str, Any],
) -> dict[str, Any]:
    event_values: dict[str, list[float]] = {str(ms): [] for ms in EVENT_MS}
    event_rows: list[dict[str, Any]] = []
    for event in target["events"]:
        row: dict[str, Any] = {
            "eventId": event["eventId"],
            "centerSamples": event["centerSamples"],
            "centerSeconds": event["centerSeconds"],
            "metrics": {},
        }
        for ms in EVENT_MS:
            metric = external.block_metric(
                source,
                teacher,
                candidate,
                int(event["centerSamples"]),
                ms,
            )
            row["metrics"][str(ms)] = metric
            event_values[str(ms)].append(float(metric["positiveProjectionDbfs"]))
        event_rows.append(row)

    short_summary: dict[str, Any] = {}
    for ms in EVENT_MS:
        values = event_values[str(ms)]
        short_summary[str(ms)] = {
            "p50Dbfs": float(np.percentile(values, 50.0)),
            "p90Dbfs": float(np.percentile(values, 90.0)),
            "p95Dbfs": float(np.percentile(values, 95.0)),
            "maxDbfs": float(np.max(values)),
            "meanDbfs": float(np.mean(values)),
        }
    return {
        "slug": target["slug"],
        "artistName": target["artistName"],
        "trackName": target["trackName"],
        "sourceBatch": target.get("sourceBatch"),
        "sourceOrder": target.get("sourceOrder"),
        "category": target.get("category"),
        "languageCode": target.get("languageCode", ""),
        "priorSplitRole": target.get("priorSplitRole"),
        "testStratum": target["testStratum"],
        "frames": int(source.shape[0]),
        "durationSeconds": float(source.shape[0] / SAMPLE_RATE),
        "eventCount": len(event_rows),
        "eventSummary": short_summary,
        # The broad trend run intentionally infers only event-intersecting
        # windows; a whole-song metric would be invalid for the sparse output.
        "wholeSong": None,
        "eventRows": event_rows,
    }


def evaluate_checkpoint(
    item: dict[str, Any],
    targets: list[dict[str, Any]],
    architecture_checkpoint: Path,
    device: torch.device,
    inference_batch_size: int,
) -> dict[str, Any]:
    checkpoint_path = Path(item["path"]).resolve()
    print(f"load {item['key']}: {checkpoint_path.name}", flush=True)
    model, model_meta = load_model(architecture_checkpoint, checkpoint_path, device)
    songs: list[dict[str, Any]] = []
    all_values: dict[str, list[float]] = {str(ms): [] for ms in EVENT_MS}
    all_song_p95: dict[str, list[float]] = {str(ms): [] for ms in EVENT_MS}
    started = time.perf_counter()
    try:
        for index, target in enumerate(targets, start=1):
            print(
                f"render {item['key']} {index}/{len(targets)}: {target['artistName']} - {target['trackName']}",
                flush=True,
            )
            source = load_audio_file(Path(target["sourcePath"]))
            source_rate = SAMPLE_RATE
            if source_rate != SAMPLE_RATE:
                raise ValueError(f"Unexpected source rate for {target['slug']}: {source_rate}")
            teacher = load_teacher_file(Path(target["teacherPath"]))
            if source.shape != teacher.shape:
                raise ValueError(f"Source/teacher shape mismatch for {target['slug']}: {source.shape} {teacher.shape}")
            residual, timing = render_student_event_regions(
                model,
                source,
                [target],
                CONTRACT,
                device,
                inference_batch_size,
            )
            candidate = np.ascontiguousarray(source - residual, dtype=np.float32)
            result = evaluate_song(source, teacher, candidate, target)
            result["render"] = timing
            for ms in EVENT_MS:
                values = [
                    float(row["metrics"][str(ms)]["positiveProjectionDbfs"])
                    for row in result["eventRows"]
                ]
                all_values[str(ms)].extend(values)
                all_song_p95[str(ms)].append(float(result["eventSummary"][str(ms)]["p95Dbfs"]))
            songs.append(result)
            del source, teacher, residual, candidate
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {
        str(ms): summarize(all_values[str(ms)], all_song_p95[str(ms)])
        for ms in EVENT_MS
    }
    by_stratum: dict[str, Any] = {"all-unused-y": {}}
    strata = [
        ("all-unused-y", songs),
        *sorted(
            (
                f"prior-role-{role}",
                [song for song in songs if song.get("priorSplitRole") == role],
            )
            for role in {str(song.get("priorSplitRole", "unknown")) for song in songs}
        ),
    ]
    for stratum, selected in strata:
        if not selected:
            by_stratum.pop(stratum, None)
            continue
        if stratum not in by_stratum:
            by_stratum[stratum] = {}
        for ms in EVENT_MS:
            values = [
                float(row["metrics"][str(ms)]["positiveProjectionDbfs"])
                for song in selected
                for row in song["eventRows"]
            ]
            song_p95 = [float(song["eventSummary"][str(ms)]["p95Dbfs"]) for song in selected]
            by_stratum[stratum][str(ms)] = summarize(values, song_p95)

    return {
        "checkpoint": checkpoint_meta(checkpoint_path, item),
        "model": model_meta,
        "elapsedSeconds": time.perf_counter() - started,
        "aggregate": aggregate,
        "byStratum": by_stratum,
        "songs": songs,
    }


def build_trend(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    baseline = results[0]
    baseline_by_stratum = baseline["byStratum"]
    trend: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "key": result["checkpoint"]["key"],
            "stage": result["checkpoint"]["stage"],
            "pass": result["checkpoint"]["pass"],
            "localPass": result["checkpoint"]["localPass"],
            "step": result["checkpoint"]["step"],
            "elapsedSeconds": result["elapsedSeconds"],
            "metrics": {},
        }
        for ms in EVENT_MS:
            current = result["aggregate"][str(ms)]
            base = baseline["aggregate"][str(ms)]
            row["metrics"][str(ms)] = {
                "pooledP95Dbfs": current["pooledP95Dbfs"],
                "deltaVsSourceDb": current["pooledP95Dbfs"] - base["pooledP95Dbfs"],
                "meanPerSongP95Dbfs": current["meanPerSongP95Dbfs"],
                "meanPerSongDeltaVsSourceDb": current["meanPerSongP95Dbfs"] - base["meanPerSongP95Dbfs"],
                "pooledMaxDbfs": current["pooledMaxDbfs"],
            }
        row["strata"] = {}
        for stratum, current_values in result["byStratum"].items():
            row["strata"][stratum] = {}
            for ms in EVENT_MS:
                current = current_values[str(ms)]
                baseline_values = baseline_by_stratum.get(stratum, {}).get(str(ms))
                row["strata"][stratum][str(ms)] = {
                    "pooledP95Dbfs": current["pooledP95Dbfs"],
                    "deltaVsSourceDb": (
                        None
                        if baseline_values is None
                        else current["pooledP95Dbfs"] - baseline_values["pooledP95Dbfs"]
                    ),
                    "meanPerSongP95Dbfs": current["meanPerSongP95Dbfs"],
                }
        trend.append(row)
    return trend


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
    training_selection_paths = tuple(
        path.resolve() for path in (args.training_selection or DEFAULT_TRAINING_SELECTIONS)
    )
    for path in training_selection_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    targets, trained_slugs, training_provenance = load_targets(
        args.selected_songs.resolve(),
        args.event_report.resolve(),
        training_selection_paths,
    )
    verify_unseen(targets, trained_slugs)
    if not targets:
        raise ValueError("No unused Y songs remain for evaluation")
    selected_payload = read_json(args.selected_songs.resolve())
    all_y_slugs = {
        normalize_slug(str(row["slug"]))
        for row in selected_payload.get("songs", [])
        if isinstance(row, dict)
        and row.get("slug")
        and str(row.get("styleMark", "")).upper() == "Y"
    }
    excluded_y_slugs = sorted(all_y_slugs.intersection(trained_slugs))
    checkpoint_items = default_checkpoints()
    for item in checkpoint_items:
        if not Path(item["path"]).resolve().is_file():
            raise FileNotFoundError(item["path"])
    checkpoint_meta_list = [checkpoint_meta(Path(item["path"]), item) for item in checkpoint_items]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "checkpoint-trend-report.json"
    manifest = {
        "schema": "local-inst3-fma-s86-checkpoint-trend@2",
        "status": "running",
        "objective": "100 ms positive projection p95 relative to native Inst 3 on all unused Y FMA event centres",
        "metricDefinition": {
            "removedContent": "mixture - Inst3 instrumental",
            "candidateError": "candidate instrumental - Inst3 instrumental",
            "positiveProjection": "max(dot(candidateError, removedContent) / power(removedContent), 0) * rms(removedContent)",
            "primaryAggregation": "p95 over 5 fixed event centres per song, pooled and as mean of per-song p95",
            "lowerIsBetter": True,
        },
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "evaluationMode": {
            "name": "event-intersecting-continuous-windows",
            "description": "Only global stride windows intersecting a 50/100/200 ms event block are inferred; their samples match full continuous overlap-save assembly.",
            "wholeSongMetrics": "omitted because non-event samples are not inferred",
        },
        "testSet": {
            "songCount": len(targets),
            "eventCount": sum(len(target["events"]) for target in targets),
            "songs": targets,
            "selectionRationale": "Every style-approved Y song in the all-S86 FMA list whose normalized slug is absent from the recorded S86 training selections.",
            "sourceSongCount": len(selected_payload.get("songs", [])),
            "styleApprovedCount": len(all_y_slugs),
            "excludedTrainingSongCount": len(excluded_y_slugs),
            "excludedTrainingSongs": excluded_y_slugs,
            "otherKnownTrainingSongsOutsideSourceList": sorted(trained_slugs.difference(all_y_slugs)),
        },
        "provenance": {
            "selectedSongs": {"file": str(args.selected_songs.resolve()), "sha256": sha256_file(args.selected_songs)},
            "eventReport": {"file": str(args.event_report.resolve()), "sha256": sha256_file(args.event_report)},
            "architectureCheckpoint": {"file": str(args.architecture_checkpoint.resolve()), "sha256": sha256_file(args.architecture_checkpoint)},
            "trainingSelections": training_provenance,
        },
        "checkpoints": checkpoint_meta_list,
        "runtime": {
            "device": str(device),
            "threads": args.threads,
            "inferenceBatchSize": args.inference_batch_size,
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "results": [],
    }
    if report_path.is_file() and not args.force:
        try:
            previous = read_json(report_path)
            if (
                previous.get("schema") == manifest["schema"]
                and [item["sha256"] for item in previous.get("checkpoints", [])]
                == [item["sha256"] for item in manifest["checkpoints"]]
                and previous.get("testSet", {}).get("songCount") == len(targets)
            ):
                manifest["results"] = previous.get("results", [])
                print(f"resuming {len(manifest['results'])} checkpoint result(s)", flush=True)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    json_write(report_path, manifest)
    completed_keys = {result.get("checkpoint", {}).get("key") for result in manifest["results"]}
    for item in checkpoint_items:
        if item["key"] in completed_keys and not args.force:
            print(f"skip {item['key']}", flush=True)
            continue
        result = evaluate_checkpoint(
            item,
            targets,
            args.architecture_checkpoint.resolve(),
            device,
            args.inference_batch_size,
        )
        manifest["results"] = [
            old for old in manifest["results"] if old.get("checkpoint", {}).get("key") != item["key"]
        ] + [result]
        manifest["results"].sort(key=lambda value: int(value["checkpoint"]["step"]))
        json_write(report_path, manifest)
        print(
            json.dumps(
                {
                    "checkpoint": item["key"],
                    "p95_100ms": result["aggregate"]["100"]["pooledP95Dbfs"],
                    "meanSongP95_100ms": result["aggregate"]["100"]["meanPerSongP95Dbfs"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest["results"].sort(key=lambda value: int(value["checkpoint"]["step"]))
    manifest["trend"] = build_trend(manifest["results"])
    manifest["status"] = "completed"
    manifest["completedAtEpoch"] = time.time()
    json_write(report_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "songs": len(targets),
                "checkpoints": len(manifest["results"]),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
