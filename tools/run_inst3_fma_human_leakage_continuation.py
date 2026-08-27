#!/usr/bin/env python3
"""Continue step 7408 on newly hand-marked FMA vocal-leakage events.

Each browser mark is converted into a training event by searching the interval
from 1.5 seconds before the mark through 0.5 seconds after it.  The selected
center maximizes the existing 100 ms positive-projection metric against Inst 3.
Centers within 50 ms are merged so repeated clicks on one audible leak do not
silently increase its training weight.

Only these newly selected events are used for training.  Every event is
repeated equally over time without song-level balancing.  The loss keeps the
step-7408 residual output outside a 100 ms core plus 25 ms linear guard while
moving the event region toward the Inst 3 residual target.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as external
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_fma_survey_continuation as prior
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "modern-song-fma-s-leakage-survey-continuation"
DEFAULT_REVIEW_PAIRS = (
    (
        DATA_ROOT / "listening-fma-training-category-sample-20" / "step-7408",
        "flac-leakage-review-2026-08-26T21-08-52-120Z.json",
    ),
    (
        DATA_ROOT / "listening-fma-training-category-remainder-51" / "step-7408",
        "flac-leakage-review-2026-08-27T01-00-52-290Z.json",
    ),
)
DEFAULT_SOURCE = DATA_ROOT / "runs" / "step-7408.pt"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-human-leakage-continuation"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
ALL_FMA_SELECTION = ROOT / "data" / "modern-song-fma-all-s86-event-pass5" / "selected-songs.json"
SURVEY_ROOT = ROOT / "data" / "fma-s-leakage-survey"

SAMPLE_RATE = 44_100
SOURCE_STEP = 7_408
SOURCE_PASS = 23
RECORDS_PER_PASS = 704
BATCH_SIZE = 4
CORE_MS = 100
GUARD_MS = 25
SEARCH_BEFORE_SECONDS = 1.5
SEARCH_AFTER_SECONDS = 0.5
SEARCH_HOP_MS = 10
DEDUPLICATION_MS = 50
DEFAULT_MAX_PASSES = 30
DEFAULT_MIN_PASSES = 3
DEFAULT_PATIENCE = 2
DEFAULT_MIN_IMPROVEMENT_DB = 0.10
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
CHECKPOINT_FORMAT = "local-inst3-fma-human-leakage-continuation-checkpoint@1"
SCHEMA = "local-inst3-fma-human-leakage-continuation@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    parser.add_argument("--min-passes", type=int, default=DEFAULT_MIN_PASSES)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-improvement-db", type=float, default=DEFAULT_MIN_IMPROVEMENT_DB)
    parser.add_argument("--records-per-pass", type=int, default=RECORDS_PER_PASS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def _teacher_entry(
    index: dict[str, list[dict[str, Any]]],
    *,
    slug: str,
    source_path: str,
    source_sha256: str,
    teacher_path: str,
    provenance: str,
) -> None:
    source = Path(source_path).resolve()
    teacher = Path(teacher_path).resolve()
    if not source.is_file() or not teacher.is_file() or not source_sha256:
        return
    index.setdefault(source_sha256.lower(), []).append(
        {
            "slug": slug,
            "sourcePath": str(source),
            "sourceSha256": source_sha256.lower(),
            "teacherPath": str(teacher),
            "provenance": provenance,
        }
    )


def load_teacher_index() -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    all_fma = prior.read_json(ALL_FMA_SELECTION)
    for song in all_fma.get("songs", []):
        _teacher_entry(
            index,
            slug=str(song.get("slug", "")),
            source_path=str(song.get("sourcePath", "")),
            source_sha256=str(song.get("sourceSha256", "")),
            teacher_path=str(song.get("teacherPath", "")),
            provenance=str(ALL_FMA_SELECTION.resolve()),
        )
    for report_path in sorted(SURVEY_ROOT.glob("batch-*/survey-report.json")):
        report = prior.read_json(report_path)
        for slug, song in (report.get("songs") or {}).items():
            source = song.get("source") or {}
            outputs = song.get("retainedOutputs") or {}
            _teacher_entry(
                index,
                slug=str(slug),
                source_path=str(source.get("rawFile", "")),
                source_sha256=str(source.get("rawSha256", "")),
                teacher_path=str(outputs.get("inst3Instrumental", "")),
                provenance=str(report_path.resolve()),
            )
    return index


def resolve_teacher(
    teacher_index: dict[str, list[dict[str, Any]]], source_sha256: str, slug: str
) -> dict[str, Any]:
    matches = teacher_index.get(source_sha256.lower(), [])
    if not matches:
        raise ValueError(f"No Inst 3 teacher matches {slug} ({source_sha256})")
    same_slug = [item for item in matches if item["slug"] == slug]
    selected = same_slug[0] if same_slug else matches[0]
    return selected


def search_mark_hotspot(
    source: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    mark_seconds: float,
    *,
    search_hop_ms: int = SEARCH_HOP_MS,
) -> dict[str, Any]:
    if source.shape != teacher.shape or source.shape != candidate.shape:
        raise ValueError("Source, teacher, and candidate must have identical shapes")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"Expected stereo audio, got {source.shape}")
    hop = max(1, round(SAMPLE_RATE * search_hop_ms / 1000.0))
    half_useful = CONTRACT.useful_samples // 2
    half_metric = round(SAMPLE_RATE * CORE_MS / 1000.0) // 2
    first_valid = max(half_useful, half_metric)
    last_valid = min(source.shape[0] - (CONTRACT.useful_samples - half_useful), source.shape[0] - half_metric - 1)
    requested_start = round((mark_seconds - SEARCH_BEFORE_SECONDS) * SAMPLE_RATE)
    requested_end = round((mark_seconds + SEARCH_AFTER_SECONDS) * SAMPLE_RATE)
    search_start = max(first_valid, requested_start)
    search_end = min(last_valid, requested_end)
    if search_end < search_start:
        raise ValueError(f"Mark at {mark_seconds:.6f}s has no valid continuous-context search interval")
    first = ((search_start + hop - 1) // hop) * hop
    centers = list(range(first, search_end + 1, hop))
    if not centers:
        centers = [min(max(round(mark_seconds * SAMPLE_RATE), search_start), search_end)]
    mark_samples = round(mark_seconds * SAMPLE_RATE)
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int] | None = None
    for center in centers:
        metrics = external.block_metric(source, teacher, candidate, center, CORE_MS)
        key = (
            float(metrics["positiveProjectionDbfs"]),
            float(metrics["missRmsDbfs"]),
            float(metrics["removedRmsDbfs"]),
            -abs(center - mark_samples),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "centerSamples": center,
                "centerSeconds": center / SAMPLE_RATE,
                "metrics100": metrics,
            }
    assert best is not None
    best.update(
        {
            "markSeconds": mark_seconds,
            "markSamples": mark_samples,
            "requestedSearchStartSeconds": requested_start / SAMPLE_RATE,
            "requestedSearchEndSeconds": requested_end / SAMPLE_RATE,
            "searchStartSeconds": search_start / SAMPLE_RATE,
            "searchEndSeconds": search_end / SAMPLE_RATE,
            "searchHopMs": search_hop_ms,
            "candidateCount": len(centers),
        }
    )
    return best


def deduplicate_hotspots(
    rows: Sequence[dict[str, Any]], deduplication_ms: int = DEDUPLICATION_MS
) -> list[dict[str, Any]]:
    separation = round(SAMPLE_RATE * deduplication_ms / 1000.0)
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["metrics100"]["positiveProjectionDbfs"]),
            -float(row["metrics100"]["missRmsDbfs"]),
            int(row["centerSamples"]),
            int(row["sourceMarkIndex"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        nearest = next(
            (
                event
                for event in selected
                if abs(int(event["centerSamples"]) - int(row["centerSamples"])) <= separation
            ),
            None,
        )
        source_mark = {
            "sourceMarkIndex": int(row["sourceMarkIndex"]),
            "markSeconds": float(row["markSeconds"]),
            "markSamples": int(row["markSamples"]),
            "selectedCenterSamplesBeforeDeduplication": int(row["centerSamples"]),
            "selectedCenterSecondsBeforeDeduplication": float(row["centerSeconds"]),
            "metrics100BeforeDeduplication": row["metrics100"],
        }
        if nearest is None:
            event = dict(row)
            event["sourceMarks"] = [source_mark]
            selected.append(event)
        else:
            nearest["sourceMarks"].append(source_mark)
    for event in selected:
        event["sourceMarks"].sort(key=lambda item: item["sourceMarkIndex"])
        event["mergedMarkCount"] = len(event["sourceMarks"])
    return sorted(selected, key=lambda row: (int(row["centerSamples"]), int(row["sourceMarkIndex"])))


def load_review_pool(
    review_pairs: Sequence[tuple[Path, str]] = DEFAULT_REVIEW_PAIRS,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    teacher_index = load_teacher_index()
    groups: dict[str, dict[str, Any]] = {}
    review_audit: list[dict[str, Any]] = []
    total_marks = 0
    provisional_count = 0
    for directory, review_name in review_pairs:
        directory = directory.resolve()
        review_path = directory / review_name
        render_path = directory / "render-report.json"
        selection_path = directory / "selection.json"
        review = prior.read_json(review_path)
        render = prior.read_json(render_path)
        if review.get("schema") != "flac-leakage-review@1":
            raise ValueError(f"Unexpected review schema: {review_path}")
        if render.get("status") != "completed":
            raise ValueError(f"Render report is not complete: {render_path}")
        songs = render.get("songs") or {}
        by_file: dict[str, tuple[str, dict[str, Any]]] = {}
        for slug, song in songs.items():
            file_name = Path(str((song.get("instrumental") or {}).get("file", ""))).name
            if not file_name or file_name in by_file:
                raise ValueError(f"Blank or duplicate rendered instrumental file name in {render_path}")
            by_file[file_name] = (str(slug), song)
        marked_tracks = 0
        report_marks = 0
        for track in review.get("tracks", []):
            marks = list(track.get("marks") or [])
            if not marks:
                continue
            file_name = str(track.get("fileName", ""))
            mapped = by_file.get(file_name)
            if mapped is None:
                raise ValueError(f"Reviewed file is absent from render report: {file_name}")
            slug, song = mapped
            source_info = song.get("source") or {}
            source_path = Path(str(source_info.get("file", ""))).resolve()
            candidate_path = Path(str((song.get("instrumental") or {}).get("file", ""))).resolve()
            source_sha256 = str(source_info.get("sha256", "")).lower()
            teacher_info = resolve_teacher(teacher_index, source_sha256, slug)
            teacher_path = Path(teacher_info["teacherPath"]).resolve()
            for path in (source_path, candidate_path, teacher_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            source = external.load_audio(source_path)
            teacher = external.load_audio(teacher_path)
            candidate = external.load_audio(candidate_path)
            if source.shape != teacher.shape or source.shape != candidate.shape:
                raise ValueError(f"Audio length mismatch for {slug}: {source.shape}, {teacher.shape}, {candidate.shape}")
            provisional: list[dict[str, Any]] = []
            for mark_index, mark in enumerate(marks, start=1):
                if str(mark.get("mark", "")) != "remove-vocal":
                    raise ValueError(f"Unexpected mark type in {review_path}: {mark}")
                seconds = float(mark["timeSeconds"])
                if not math.isfinite(seconds):
                    raise ValueError(f"Non-finite mark time in {review_path}")
                hotspot = search_mark_hotspot(source, teacher, candidate, seconds)
                hotspot.update(
                    {
                        "sourceMarkIndex": mark_index,
                        "reviewFileName": file_name,
                        "reviewPath": str(review_path),
                    }
                )
                provisional.append(hotspot)
            selected = deduplicate_hotspots(provisional)
            key = f"human::{slug}"
            if key in groups:
                raise ValueError(f"Song occurs in more than one review report: {slug}")
            events: list[dict[str, Any]] = []
            for event_number, hotspot in enumerate(selected, start=1):
                center = int(hotspot["centerSamples"])
                metrics = {
                    str(milliseconds): external.block_metric(source, teacher, candidate, center, milliseconds)
                    for milliseconds in (50, 100, 200)
                }
                events.append(
                    {
                        **hotspot,
                        "eventId": f"human-leakage-{slug}-{event_number:03d}",
                        "mark": "remove-vocal",
                        "metricsAtSelection": metrics,
                        "reviewSet": directory.parent.name,
                        "category": str(song.get("category", "")),
                    }
                )
            groups[key] = {
                "key": key,
                "slug": slug,
                "sourceBatch": str(song.get("selectedForStage", "")),
                "sourceOrder": int(song.get("index", 0)),
                "artistName": str(song.get("artistName", "")),
                "trackName": str(song.get("trackName", "")),
                "category": str(song.get("category", "")),
                "sourcePath": str(source_path),
                "sourceSha256": source_sha256,
                "teacherPath": str(teacher_path),
                "teacherProvenance": teacher_info["provenance"],
                "candidatePath": str(candidate_path),
                "reviewPath": str(review_path),
                "renderReportPath": str(render_path),
                "selectionPath": str(selection_path),
                "events": events,
            }
            marked_tracks += 1
            report_marks += len(marks)
            total_marks += len(marks)
            provisional_count += len(provisional)
            del source, teacher, candidate
        review_audit.append(
            {
                "directory": str(directory),
                "reviewPath": str(review_path),
                "reviewSha256": prior.sha256_file(review_path),
                "renderReportPath": str(render_path),
                "renderReportSha256": prior.sha256_file(render_path),
                "selectionPath": str(selection_path),
                "selectionSha256": prior.sha256_file(selection_path),
                "markedTrackCount": marked_tracks,
                "markCount": report_marks,
            }
        )
    groups = {key: groups[key] for key in sorted(groups)}
    records: list[dict[str, Any]] = []
    for group in groups.values():
        for index, event in enumerate(group["events"]):
            records.append({"key": group["key"], "index": index, **event})
    if not records:
        raise ValueError("The review reports contain no usable marks")
    selection: dict[str, Any] = {
        "schema": SCHEMA + ".selection",
        "reviewAudit": review_audit,
        "selectedSongCount": len(groups),
        "rawMarkCount": total_marks,
        "provisionalEventCount": provisional_count,
        "selectedEventCount": len(records),
        "mergedMarkCount": total_marks - len(records),
        "categoryCounts": dict(sorted(Counter(group["category"] for group in groups.values()).items())),
        "eventsPerSong": dict(sorted(Counter(record["key"] for record in records).items())),
        "songKeys": sorted(groups),
        "eventIds": [record["eventId"] for record in records],
        "eventSelection": {
            "searchIntervalRelativeToMarkSeconds": [-SEARCH_BEFORE_SECONDS, SEARCH_AFTER_SECONDS],
            "searchHopMs": SEARCH_HOP_MS,
            "primaryMetric": "100ms positiveProjectionDbfs against Inst 3",
            "tieBreakers": ["100ms missRmsDbfs", "100ms removedRmsDbfs", "distance to manual mark"],
            "deduplicationMs": DEDUPLICATION_MS,
            "deduplicationRule": "strongest center retained; nearby source marks attached as provenance",
        },
        "trainingBoundary": "only newly hand-marked events; no old FMA events, MUSDB windows, or song balancing",
    }
    selection["selectionSha256"] = prior.canonical_sha256(selection)
    return groups, records, selection


def load_source_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    source_path = args.source_checkpoint.resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-fma-s-leakage-survey-continuation-checkpoint@1":
        raise ValueError(f"Unexpected source format: {payload.get('format')}")
    if int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(f"Expected source step {SOURCE_STEP}, got {payload.get('step')}")
    if not isinstance(payload.get("stateDict"), dict) or not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Source checkpoint lacks model or optimizer state")
    model, architecture = pilot.make_model(args.architecture_checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    prior.move_optimizer_state(optimizer, device)
    for optimizer_group in optimizer.param_groups:
        optimizer_group["lr"] = args.learning_rate
    source = {
        "checkpoint": prior.checkpoint_metadata(source_path),
        "architecture": architecture["checkpoint"],
        "sourceStep": int(payload["step"]),
        "sourcePass": int(payload.get("continuationPass", SOURCE_PASS)),
        "sourceVariant": payload.get("variant"),
        "optimizerRestored": True,
    }
    return model, optimizer, source, payload


def validate_cache(path: Path, expected_records: int) -> None:
    prior.validate_cache(path, expected_records)


def prepare_caches(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    device: torch.device,
) -> dict[str, Path]:
    cache_root = args.output_root.resolve() / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, group in groups.items():
        path = cache_root / f"{prior.safe_name(key)}.npz"
        metadata_path = path.with_suffix(".json")
        if not args.force_cache and path.is_file() and metadata_path.is_file():
            try:
                metadata = prior.read_json(metadata_path)
                if (
                    metadata.get("selectionSha256") == selection["selectionSha256"]
                    and metadata.get("sourceCheckpointSha256") == selection["sourceCheckpoint"]["sha256"]
                ):
                    validate_cache(path, len(group["events"]))
                    paths[key] = path
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    missing = [key for key in groups if key not in paths]
    if not missing:
        print(json.dumps({"event": "cache-reuse", "groups": len(paths)}, sort_keys=True), flush=True)
        return paths

    anchor_model, anchor_optimizer, anchor_source, _ = load_source_model(args, device)
    del anchor_optimizer
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    try:
        for number, key in enumerate(missing, start=1):
            group = groups[key]
            source_path = Path(group["sourcePath"])
            teacher_path = Path(group["teacherPath"])
            source = external.load_audio(source_path)
            teacher = external.load_audio(teacher_path)
            if source.shape != teacher.shape:
                raise ValueError(f"Source/teacher shape mismatch for {key}")
            if prior.sha256_file(source_path).lower() != group["sourceSha256"]:
                raise ValueError(f"Source hash mismatch for {key}")
            input_rows: list[np.ndarray] = []
            target_rows: list[np.ndarray] = []
            mask_rows: list[np.ndarray] = []
            for event in group["events"]:
                center = int(event["centerSamples"])
                start = center - CONTRACT.useful_samples // 2
                end = start + CONTRACT.useful_samples
                if start < 0 or end > source.shape[0]:
                    raise ValueError(f"Event context out of bounds: {event['eventId']}")
                assembled = local.assemble_input(source, start, CONTRACT.useful_samples, CONTRACT, mode="continuous")
                input_rows.append(np.ascontiguousarray(local.stft_centered(assembled, CONTRACT)[0], dtype=np.float32))
                target_rows.append(np.ascontiguousarray(source[start:end] - teacher[start:end], dtype=np.float32))
                mask_rows.append(
                    np.ascontiguousarray(
                        prior.c1.event_mask(CONTRACT.useful_samples, center - start, CORE_MS, GUARD_MS),
                        dtype=np.float32,
                    )
                )
            input_array = np.stack(input_rows).astype(np.float32)
            target_array = np.stack(target_rows).astype(np.float32)
            mask_array = np.stack(mask_rows).astype(np.float32)
            anchor_array = prior.infer_anchor(
                anchor_model, input_array, device, args.inference_batch_size, window
            )
            path = cache_root / f"{prior.safe_name(key)}.npz"
            temporary = path.with_name(path.name + ".tmp.npz")
            np.savez_compressed(
                temporary,
                inputSpec=input_array,
                targetAudio=target_array,
                anchorAudio=anchor_array,
                eventMask=mask_array,
            )
            temporary.replace(path)
            metadata = {
                "schema": SCHEMA + ".cache",
                "selectionSha256": selection["selectionSha256"],
                "sourceCheckpointSha256": selection["sourceCheckpoint"]["sha256"],
                "anchorCheckpoint": anchor_source["checkpoint"],
                "key": key,
                "slug": group["slug"],
                "sourcePath": str(source_path),
                "teacherPath": str(teacher_path),
                "sourceSha256": group["sourceSha256"],
                "teacherSha256": prior.sha256_file(teacher_path),
                "recordCount": len(group["events"]),
                "events": group["events"],
                "targetSemantic": "mixture - Inst3 instrumental",
                "anchorSemantic": "step-7408 residual-vocals output",
                "eventMask": "100ms core + 25ms linear guard",
                "contract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
                "cache": prior.checkpoint_metadata(path),
            }
            prior.json_write(path.with_suffix(".json"), metadata)
            validate_cache(path, len(group["events"]))
            paths[key] = path
            print(
                json.dumps(
                    {"event": "cache-prepared", "index": number, "total": len(missing), "key": key, "records": len(group["events"])},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            del source, teacher, input_array, target_array, anchor_array, mask_array
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del anchor_model, window
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prior.json_write(args.output_root / "selection-with-cache.json", {**selection, "cacheCount": len(paths)})
    return paths


def build_equal_schedule(
    records: Sequence[dict[str, Any]], passes: int, seed: int, records_per_pass: int = RECORDS_PER_PASS
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    items = [(str(record["key"]), int(record["index"])) for record in records]
    if not items or len(set(items)) != len(items):
        raise ValueError("Schedule records must be non-empty and unique")
    if records_per_pass < len(items):
        raise ValueError("records-per-pass must be at least the number of unique events")
    base_repeat, remainder = divmod(records_per_pass, len(items))
    extra_order = [items[int(index)] for index in np.random.default_rng(seed + 17).permutation(len(items))]
    schedule: list[tuple[str, int]] = []
    pass_rows: list[dict[str, Any]] = []
    cumulative: Counter[tuple[str, int]] = Counter()
    for pass_index in range(passes):
        values = items * base_repeat
        if remainder:
            offset = (pass_index * remainder) % len(items)
            values.extend(extra_order[(offset + index) % len(items)] for index in range(remainder))
        rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97)
        ordered = [values[int(index)] for index in rng.permutation(len(values))]
        schedule.extend(ordered)
        counts = Counter(ordered)
        cumulative.update(ordered)
        per_event = [counts[item] for item in items]
        cumulative_values = [cumulative[item] for item in items]
        pass_rows.append(
            {
                "pass": pass_index + 1,
                "recordCount": len(ordered),
                "uniqueEventCount": len(items),
                "eventCountMin": min(per_event),
                "eventCountMax": max(per_event),
                "cumulativeEventCountMin": min(cumulative_values),
                "cumulativeEventCountMax": max(cumulative_values),
            }
        )
    summary = {
        "recordsPerPass": records_per_pass,
        "uniqueEventCount": len(items),
        "baseRepeat": base_repeat,
        "remainder": remainder,
        "passes": pass_rows,
        "recordCount": len(schedule),
        "weightedInterpretation": "all new events are equally repeated over time; no song-level balancing",
        "scheduleSha256": prior.canonical_sha256(schedule),
    }
    return schedule, summary


def required_production_windows(events: Sequence[dict[str, Any]], sample_count: int) -> list[int]:
    indices: set[int] = set()
    half = round(SAMPLE_RATE * 0.200 / 2.0)
    for event in events:
        center = int(event["centerSamples"])
        start = center - half
        end = center + half
        if start < 0 or end > sample_count:
            raise ValueError(f"Event block is outside song bounds: {event['eventId']}")
        indices.update(range(start // CONTRACT.stride_samples, (end - 1) // CONTRACT.stride_samples + 1))
    return sorted(indices)


def infer_production_windows(
    model: torch.nn.Module,
    source: np.ndarray,
    window_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    segments: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for begin in range(0, len(window_indices), batch_size):
            batch_indices = list(window_indices[begin : begin + batch_size])
            starts = [index * CONTRACT.stride_samples for index in batch_indices]
            lengths = [min(CONTRACT.stride_samples, source.shape[0] - start) for start in starts]
            spectra = np.concatenate(
                [
                    continuous.stft_centered(
                        continuous.assemble_input(source, start, length, CONTRACT, mode="continuous"),
                        CONTRACT,
                    )
                    for start, length in zip(starts, lengths)
                ],
                axis=0,
            )
            predictions = model(torch.from_numpy(spectra).to(device)).detach().cpu().numpy()
            for offset, (index, length) in enumerate(zip(batch_indices, lengths)):
                reconstructed = continuous.istft_centered(predictions[offset : offset + 1], CONTRACT)
                start = CONTRACT.left_context_samples
                segment = np.ascontiguousarray(reconstructed[start : start + length], dtype=np.float32)
                if not np.isfinite(segment).all():
                    raise ValueError(f"Non-finite production segment for window {index}")
                segments[index] = segment
    return segments


def extract_production_block(
    segments: dict[int, np.ndarray], start: int, end: int
) -> np.ndarray:
    rows: list[np.ndarray] = []
    position = start
    while position < end:
        index = position // CONTRACT.stride_samples
        window_start = index * CONTRACT.stride_samples
        segment = segments[index]
        local_start = position - window_start
        take = min(end - position, segment.shape[0] - local_start)
        if take <= 0:
            raise ValueError("Unable to assemble production event block")
        rows.append(segment[local_start : local_start + take])
        position += take
    result = np.concatenate(rows, axis=0)
    if result.shape != (end - start, 2):
        raise ValueError(f"Unexpected assembled block shape: {result.shape}")
    return result


def evaluate_model(
    model: torch.nn.Module,
    groups: dict[str, dict[str, Any]],
    device: torch.device,
    inference_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    all_rows: list[dict[str, Any]] = []
    by_review: dict[str, list[dict[str, Any]]] = {}
    by_category: dict[str, list[dict[str, Any]]] = {}
    songs: list[dict[str, Any]] = []
    started = time.perf_counter()
    inferred_windows = 0
    for key, group in groups.items():
        source = external.load_audio(Path(group["sourcePath"]))
        teacher = external.load_audio(Path(group["teacherPath"]))
        if source.shape != teacher.shape:
            raise ValueError(f"Evaluation source/teacher mismatch for {key}")
        window_indices = required_production_windows(group["events"], source.shape[0])
        segments = infer_production_windows(model, source, window_indices, device, inference_batch_size)
        inferred_windows += len(window_indices)
        song_rows: list[dict[str, Any]] = []
        for event in group["events"]:
            event_metrics: dict[str, dict[str, float]] = {}
            center = int(event["centerSamples"])
            for milliseconds in (50, 100, 200):
                length = max(1, round(SAMPLE_RATE * milliseconds / 1000.0))
                start = center - length // 2
                end = start + length
                residual = extract_production_block(segments, start, end)
                source_block = source[start:end]
                teacher_block = teacher[start:end]
                candidate_block = np.ascontiguousarray(source_block - residual, dtype=np.float32)
                event_metrics[str(milliseconds)] = external.block_metric(
                    source_block, teacher_block, candidate_block, length // 2, milliseconds
                )
            row = {
                "eventId": event["eventId"],
                "song": key,
                "category": group["category"],
                "reviewPath": group["reviewPath"],
                "metrics": event_metrics,
            }
            all_rows.append(row)
            song_rows.append(row)
            by_review.setdefault(group["reviewPath"], []).append(row)
            by_category.setdefault(group["category"], []).append(row)
        songs.append(
            {
                "key": key,
                "song": f"{group['artistName']} - {group['trackName']}",
                "category": group["category"],
                "eventCount": len(song_rows),
                "productionWindowCount": len(window_indices),
                "metrics": prior.summarize_rows(song_rows),
            }
        )
        del source, teacher, segments
    result = {
        "elapsedSeconds": time.perf_counter() - started,
        "poolRole": "training-pool-event-evaluation",
        "assembly": "exact fixed-stride continuous-context windows intersecting each event block",
        "inferredProductionWindowCount": inferred_windows,
        "all": prior.summarize_rows(all_rows),
        "byReview": {name: prior.summarize_rows(rows) for name, rows in sorted(by_review.items())},
        "byCategory": {name: prior.summarize_rows(rows) for name, rows in sorted(by_category.items())},
        "songs": songs,
        "eventRows": all_rows,
    }
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    return result


def train_pass(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    store: prior.GpuCacheStore,
    schedule: Sequence[tuple[str, int]],
    pass_index: int,
    args: argparse.Namespace,
    device: torch.device,
    max_updates: int | None = None,
) -> list[dict[str, Any]]:
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    updates_per_pass = args.records_per_pass // args.batch_size
    updates = updates_per_pass if max_updates is None else min(updates_per_pass, max_updates)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    begin = pass_index * args.records_per_pass
    for update in range(updates):
        batch_started = time.perf_counter()
        items = schedule[begin + update * args.batch_size : begin + (update + 1) * args.batch_size]
        input_tensor, target_tensor, anchor_tensor, mask_tensor = store.batch(items)
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        predicted_full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = predicted_full[:, trim : trim + CONTRACT.useful_samples]
        forward_seconds = time.perf_counter() - forward_started
        event_losses = local.charbonnier_per_record(predicted, target_tensor, mask_tensor)
        anchor_losses = local.charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor)
        loss = (event_losses + args.anchor_beta * anchor_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at pass {pass_index + 1}, update {update + 1}")
        backward_started = time.perf_counter()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        if not math.isfinite(gradient):
            raise FloatingPointError(f"Non-finite gradient at pass {pass_index + 1}, update {update + 1}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "pass": pass_index + 1,
            "update": update + 1,
            "globalStep": SOURCE_STEP + pass_index * updates_per_pass + update + 1,
            "loss": float(loss.detach().cpu()),
            "eventLoss": float(event_losses.mean().detach().cpu()),
            "anchorLoss": float(anchor_losses.mean().detach().cpu()),
            "gradientNormBeforeClip": gradient,
            "forwardSeconds": forward_seconds,
            "backwardSeconds": time.perf_counter() - backward_started,
            "batchSeconds": time.perf_counter() - batch_started,
            "cudaAllocatedBytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
            "cudaReservedBytes": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None,
        }
        history.append(row)
        if update == 0 or (update + 1) % args.log_every == 0 or update + 1 == updates:
            print(json.dumps({"event": "progress", **row, "updates": updates}, sort_keys=True), flush=True)
    print(
        json.dumps({"event": "pass-complete", "pass": pass_index + 1, "elapsedSeconds": time.perf_counter() - started}, sort_keys=True),
        flush=True,
    )
    del window
    return history


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    contract_id: str,
    schedule_hash: str,
    selection: dict[str, Any],
    source: dict[str, Any],
    history: list[dict[str, Any]],
    local_pass: int,
    global_step: int,
) -> dict[str, Any]:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": "FMA-human-leakage-event-continuation",
        "runContractId": contract_id,
        "sourceCheckpoint": source["checkpoint"]["file"],
        "sourceStep": SOURCE_STEP,
        "sourcePass": source["sourcePass"],
        "localPass": local_pass,
        "step": global_step,
        "globalStep": global_step,
        "continuationPass": int(source["sourcePass"]) + local_pass,
        "recordsPerPass": args.records_per_pass,
        "updatesPerPass": args.records_per_pass // args.batch_size,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_hash,
        "selectionSha256": selection["selectionSha256"],
        "lossContract": "Inst3 residual target on 100ms+25ms event mask plus step-7408 anchor outside mask",
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "stateDict": prior.cpu_tree(model.state_dict()),
        "optimizerStateDict": prior.cpu_tree(optimizer.state_dict()),
        "history": history,
    }
    prior.atomic_torch_save(path, payload)
    return prior.checkpoint_metadata(path)


def load_latest_checkpoint(
    run_root: Path, contract_id: str, selection_hash: str, schedule_hash: str
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in run_root.glob("step-*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("format") == CHECKPOINT_FORMAT
                and payload.get("runContractId") == contract_id
                and payload.get("selectionSha256") == selection_hash
                and payload.get("scheduleSha256") == schedule_hash
            ):
                candidates.append((int(payload.get("localPass", -1)), path, payload))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def make_report(
    args: argparse.Namespace,
    selection: dict[str, Any],
    schedule_summary: dict[str, Any],
    contract_id: str,
    contract_payload: dict[str, Any],
    source: dict[str, Any],
    history: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    status: str,
    gpu_cache_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "selection": selection,
        "schedule": schedule_summary,
        "contract": {"id": contract_id, "payload": contract_payload},
        "source": source,
        "training": {
            "maxPasses": args.max_passes,
            "completedPasses": len(history) // (args.records_per_pass // args.batch_size),
            "history": history,
        },
        "evaluations": evaluations,
        "stop": {"patience": args.patience, "minImprovementDb": args.min_improvement_db},
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpuCacheBytes": gpu_cache_bytes,
        },
    }


def run_training(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    paths: dict[str, Path],
    schedule: list[tuple[str, int]],
    schedule_summary: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    contract_payload = {
        "schema": SCHEMA,
        "sourceCheckpointSha256": selection["sourceCheckpoint"]["sha256"],
        "selectionSha256": selection["selectionSha256"],
        "scheduleSha256": schedule_summary["scheduleSha256"],
        "recordsPerPass": args.records_per_pass,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "loss": "Inst3 residual target on event mask plus step-7408 anchor outside mask",
        "eventWeighting": "all selected events equal; no song balancing",
        "officialFinalTestUsed": False,
    }
    contract_id = prior.canonical_sha256(contract_payload)
    run_root = args.output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "reports" / "training-report.json"
    prior.set_seed(args.seed)
    model, optimizer, source, _ = load_source_model(args, device)
    store = prior.GpuCacheStore(paths, groups, device)
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    completed_passes = 0

    if args.resume and not args.smoke_only:
        found = load_latest_checkpoint(run_root, contract_id, selection["selectionSha256"], schedule_summary["scheduleSha256"])
        if found is not None:
            checkpoint_path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            prior.move_optimizer_state(optimizer, device)
            for optimizer_group in optimizer.param_groups:
                optimizer_group["lr"] = args.learning_rate
            completed_passes = int(payload.get("localPass", 0))
            history = list(payload.get("history", []))
            if report_path.is_file():
                evaluations = list(prior.read_json(report_path).get("evaluations", []))
            print(json.dumps({"event": "resume", "pass": completed_passes, "checkpoint": str(checkpoint_path)}, sort_keys=True), flush=True)

    if args.smoke_only:
        monitor = prior.GpuMonitor(device)
        monitor.start()
        smoke_history = train_pass(model, optimizer, store, schedule, 0, args, device, args.smoke_updates)
        result = {
            "schema": SCHEMA + ".smoke",
            "status": "smoke-completed",
            "updates": len(smoke_history),
            "history": smoke_history,
            "selection": selection,
            "schedule": schedule_summary,
            "gpu": {"cacheBytes": store.bytes, "trainingMonitor": monitor.stop()},
        }
        del model, optimizer, store
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    if not evaluations:
        baseline = evaluate_model(model, groups, device, args.inference_batch_size)
        evaluations.append(
            {
                "pass": 0,
                "globalStep": SOURCE_STEP,
                "metrics": baseline,
                "improvementFromPreviousDb": None,
                "improvementFromReferenceDb": None,
                "meaningfulImprovement": None,
                "role": "step-7408-source-baseline",
            }
        )
        print(json.dumps({"event": "baseline", "p95_100ms": baseline["all"]["100"]["positiveProjectionP95Dbfs"]}, sort_keys=True), flush=True)
        prior.json_write(
            report_path,
            make_report(args, selection, schedule_summary, contract_id, contract_payload, source, history, evaluations, "running", store.bytes),
        )

    reference_p95 = float(evaluations[0]["metrics"]["all"]["100"]["positiveProjectionP95Dbfs"])
    previous_p95 = float(evaluations[-1]["metrics"]["all"]["100"]["positiveProjectionP95Dbfs"])
    best_p95 = min(float(row["metrics"]["all"]["100"]["positiveProjectionP95Dbfs"]) for row in evaluations)
    no_improvement = 0
    for row in evaluations[1:]:
        if bool(row.get("meaningfulImprovement")):
            reference_p95 = float(row["metrics"]["all"]["100"]["positiveProjectionP95Dbfs"])
            no_improvement = 0
        else:
            no_improvement += 1

    try:
        for pass_index in range(completed_passes, args.max_passes):
            monitor = prior.GpuMonitor(device)
            monitor.start()
            pass_history = train_pass(model, optimizer, store, schedule, pass_index, args, device)
            training_gpu = monitor.stop()
            history.extend(pass_history)
            local_pass = pass_index + 1
            global_step = SOURCE_STEP + local_pass * (args.records_per_pass // args.batch_size)
            checkpoint_path = run_root / f"step-{global_step}.pt"
            checkpoint_info = save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                args,
                contract_id,
                schedule_summary["scheduleSha256"],
                selection,
                source,
                history,
                local_pass,
                global_step,
            )
            metrics = evaluate_model(model, groups, device, args.inference_batch_size)
            current_p95 = float(metrics["all"]["100"]["positiveProjectionP95Dbfs"])
            improvement_previous = previous_p95 - current_p95
            improvement_reference = reference_p95 - current_p95
            meaningful = improvement_reference >= args.min_improvement_db
            best_p95 = min(best_p95, current_p95)
            evaluations.append(
                {
                    "pass": local_pass,
                    "globalStep": global_step,
                    "checkpoint": checkpoint_info,
                    "metrics": metrics,
                    "improvementFromPreviousDb": improvement_previous,
                    "improvementFromReferenceDb": improvement_reference,
                    "meaningfulImprovement": meaningful,
                    "bestP95Dbfs": best_p95,
                    "trainingGpuMonitor": training_gpu,
                    "role": "human-leakage-event-continuation",
                }
            )
            if meaningful:
                reference_p95 = current_p95
                no_improvement = 0
            else:
                no_improvement += 1
            previous_p95 = current_p95
            report = make_report(
                args, selection, schedule_summary, contract_id, contract_payload, source, history, evaluations, "running", store.bytes
            )
            prior.json_write(report_path, report)
            print(
                json.dumps(
                    {
                        "event": "evaluation",
                        "pass": local_pass,
                        "globalStep": global_step,
                        "p95_100ms": current_p95,
                        "improvementFromPreviousDb": improvement_previous,
                        "improvementFromReferenceDb": improvement_reference,
                        "meaningfulImprovement": meaningful,
                        "trainingGpuP50": (training_gpu.get("gpuUtilizationPercent") or {}).get("p50"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if local_pass >= args.min_passes and no_improvement >= args.patience:
                report["status"] = "completed"
                report["stop"] = {
                    "reason": "100ms projection p95 plateau on newly selected training events",
                    "stoppedAfterPass": local_pass,
                    "consecutiveNonMeaningfulPasses": no_improvement,
                    "minImprovementDb": args.min_improvement_db,
                    "patience": args.patience,
                    "bestP95Dbfs": best_p95,
                }
                prior.json_write(report_path, report)
                break
        else:
            report = prior.read_json(report_path)
            report["status"] = "completed"
            report["stop"] = {
                "reason": "max-passes guard",
                "maxPasses": args.max_passes,
                "minImprovementDb": args.min_improvement_db,
                "patience": args.patience,
                "bestP95Dbfs": best_p95,
            }
            prior.json_write(report_path, report)
    finally:
        del model, optimizer, store
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prior.read_json(report_path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_passes <= 0 or args.min_passes <= 0 or args.min_passes > args.max_passes:
        raise ValueError("Invalid pass limits")
    if args.patience <= 0 or args.min_improvement_db < 0:
        raise ValueError("Invalid plateau settings")
    if args.batch_size <= 0 or args.records_per_pass <= 0 or args.records_per_pass % args.batch_size:
        raise ValueError("records-per-pass must be positive and divisible by batch-size")
    if args.smoke_updates <= 0:
        raise ValueError("smoke-updates must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    groups, records, selection = load_review_pool()
    selection["sourceCheckpoint"] = prior.checkpoint_metadata(args.source_checkpoint)
    selection["sourceStep"] = SOURCE_STEP
    selection["selectionSha256"] = prior.canonical_sha256(
        {key: value for key, value in selection.items() if key != "selectionSha256"}
    )
    prior.json_write(args.output_root / "event-selection.json", selection)
    prior.json_write(
        args.output_root / "event-manifest.json",
        {
            "schema": SCHEMA + ".event-manifest",
            "selectionSha256": selection["selectionSha256"],
            "groups": groups,
        },
    )
    print(
        json.dumps(
            {
                "event": "selection-complete",
                "songs": len(groups),
                "rawMarks": selection["rawMarkCount"],
                "selectedEvents": len(records),
                "mergedMarks": selection["mergedMarkCount"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.selection_only:
        return 0
    paths = prepare_caches(args, groups, selection, device)
    schedule, schedule_summary = build_equal_schedule(records, args.max_passes, args.seed, args.records_per_pass)
    prior.json_write(args.output_root / "schedule.json", {"summary": schedule_summary, "schedule": schedule})
    result = run_training(args, groups, selection, paths, schedule, schedule_summary, device)
    if args.smoke_only:
        report_path = args.output_root / "reports" / "smoke-report.json"
        prior.json_write(report_path, result)
        print(json.dumps({"status": result["status"], "report": str(report_path)}, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "report": str(args.output_root / "reports" / "training-report.json"),
                    "completedPasses": result.get("training", {}).get("completedPasses"),
                    "stop": result.get("stop"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
