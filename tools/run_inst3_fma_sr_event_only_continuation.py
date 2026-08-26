#!/usr/bin/env python3
"""Continue S86-event-only using the newly reviewed FMA train S/R events.

This is a bounded local research run.  It starts from the canonical
``S86-event-only@pass-5`` checkpoint, uses only S/R rows from the new 16-event
review, and keeps the original masked-event plus anchor loss.  Each S event is
sampled about twice as often as each R event; no song-level balancing is used.
The runner evaluates the same continuous full-song event pool after every
pass and stops after a patience window without a meaningful 100 ms projection
p95 improvement.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as external
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_mtg_fma_c1 as c1
import run_inst3_s_r_continuation as prior
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_ROOT = ROOT / "data" / "modern-song-fma-train-sr-s86-event-pass5-16"
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "modern-song-s-r-continuation"
    / "runs"
    / "S86-event-only"
    / "step-3360.pt"
)
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-sr-event-only-continuation"

SAMPLE_RATE = 44_100
SOURCE_STEP = 3_360
SOURCE_PASS = 5
RECORDS_PER_PASS = 704
BATCH_SIZE = 4
CORE_MS = 100
GUARD_MS = 25
DEFAULT_MAX_PASSES = 10
DEFAULT_MIN_PASSES = 3
DEFAULT_PATIENCE = 2
DEFAULT_MIN_IMPROVEMENT_DB = 0.10
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
CHECKPOINT_FORMAT = "local-inst3-fma-sr-event-only-continuation-checkpoint@1"
SCHEMA = "local-inst3-fma-sr-event-only-continuation@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    parser.add_argument("--min-passes", type=int, default=DEFAULT_MIN_PASSES)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-improvement-db", type=float, default=DEFAULT_MIN_IMPROVEMENT_DB)
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
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "-" for char in value).strip("-") or "song"


def load_pool(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = args.annotation_root.resolve()
    report = read_json(root / "event-listening-report.json")
    selection = read_json(root / "selected-songs.json")
    rows_path = root / "human-review-template.csv"
    with rows_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events_by_id = {str(event["eventId"]): event for event in report.get("events", [])}
    songs_by_key = {
        (str(song["sourceBatch"]), int(song["sourceOrder"])): song
        for song in selection.get("songs", [])
    }
    records: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        mark = (row.get("mark") or "").strip().upper()
        if mark not in {"S", "R"}:
            continue
        event_id = str(row.get("eventId", ""))
        event = events_by_id.get(event_id)
        if event is None:
            raise ValueError(f"Marked event is missing from report: {event_id}")
        key = (str(event["sourceBatch"]), int(event["sourceOrder"]))
        song = songs_by_key.get(key)
        if song is None or str(song.get("priorSplitRole")) != "train":
            raise ValueError(f"Marked event is not a selected train song: {event_id}")
        source_path = Path(song["sourcePath"]).resolve()
        teacher_path = Path(song["teacherPath"]).resolve()
        h50_path = Path(song["h50Path"]).resolve()
        for path in (source_path, teacher_path, h50_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        cache_key = f"external::{event['sourceBatch']}::{int(event['sourceOrder']):03d}-{safe_name(event['slug'])}"
        group = grouped.setdefault(
            cache_key,
            {
                "key": cache_key,
                "sourceBatch": event["sourceBatch"],
                "sourceOrder": int(event["sourceOrder"]),
                "slug": event["slug"],
                "artistName": event.get("artistName", ""),
                "trackName": event.get("trackName", ""),
                "sourcePath": str(source_path),
                "teacherPath": str(teacher_path),
                "h50Path": str(h50_path),
                "sourceSha256": str(song["sourceSha256"]).lower(),
                "events": [],
            },
        )
        if any(item["eventId"] == event_id for item in group["events"]):
            raise ValueError(f"Duplicate marked event: {event_id}")
        group["events"].append(
            {
                "eventId": event_id,
                "mark": mark,
                "centerSamples": int(event["centerSamples"]),
                "centerSeconds": float(event["centerSeconds"]),
                "eventScoreDbfs": float(event["eventScoreDbfs"]),
                "serial": int(event["serial"]),
                "category": event.get("category", ""),
            }
        )
    if not records and not grouped:
        raise ValueError("The new annotation CSV contains no S/R events")
    groups = {key: grouped[key] for key in sorted(grouped)}
    for group in groups.values():
        group["events"].sort(key=lambda item: (int(item["centerSamples"]), item["eventId"]))
        for index, event in enumerate(group["events"]):
            records.append({"key": group["key"], "index": index, **event})
    s_records = [record for record in records if record["mark"] == "S"]
    r_records = [record for record in records if record["mark"] == "R"]
    if not s_records or not r_records:
        raise ValueError(f"Both S and R pools are required, got S={len(s_records)} R={len(r_records)}")
    selection = {
        "schema": SCHEMA + ".selection",
        "annotationRoot": str(root),
        "annotationReportSha256": sha256_file(root / "event-listening-report.json"),
        "annotationCsvSha256": sha256_file(rows_path),
        "selectedSongCount": len(groups),
        "recordCount": len(records),
        "sRecordCount": len(s_records),
        "rRecordCount": len(r_records),
        "songKeys": sorted(groups),
        "eventIds": [record["eventId"] for record in records],
        "markSemantics": {
            "S": "especially conspicuous residual vocal; subset of R",
            "R": "further vocal/harmony/spoken/vocal-effect removal desired",
            "K": "excluded from training",
            "I": "excluded from training but does not exclude a song's S/R events",
        },
    }
    selection["selectionSha256"] = canonical_sha256(selection)
    return groups, records, selection


def validate_arrays(path: Path, expected_records: int) -> None:
    with np.load(path) as values:
        required = ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
        if any(name not in values for name in required):
            raise ValueError(f"Missing cache array in {path}")
        if values["inputSpec"].shape != (expected_records, 4, 1025, 128):
            raise ValueError(f"Unexpected input shape in {path}: {values['inputSpec'].shape}")
        if values["targetAudio"].shape != (expected_records, CONTRACT.useful_samples, 2):
            raise ValueError(f"Unexpected target shape in {path}: {values['targetAudio'].shape}")
        if values["anchorAudio"].shape != values["targetAudio"].shape:
            raise ValueError(f"Anchor shape mismatch in {path}")
        if values["eventMask"].shape != (expected_records, CONTRACT.useful_samples):
            raise ValueError(f"Unexpected mask shape in {path}: {values['eventMask'].shape}")
        if any(not np.isfinite(values[name]).all() for name in required):
            raise ValueError(f"Non-finite cache value in {path}")


def load_source_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    source_path = args.source_checkpoint.resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-s-r-continuation-checkpoint@1":
        raise ValueError(f"Unexpected source format: {payload.get('format')}")
    if payload.get("variant") != "S86-event-only" or int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(f"Expected S86-event-only step {SOURCE_STEP}, got {payload.get('variant')} {payload.get('step')}")
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Source checkpoint has no optimizer state")
    model, architecture = pilot.make_model(args.architecture_checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    prior.continuation.move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    source = {
        "checkpoint": checkpoint_metadata(source_path),
        "architecture": architecture["checkpoint"],
        "sourceStep": int(payload["step"]),
        "sourcePass": SOURCE_PASS,
        "sourceVariant": payload.get("variant"),
        "optimizerRestored": True,
    }
    return model, optimizer, source, payload


def infer_anchor(
    model: torch.nn.Module,
    input_specs: np.ndarray,
    device: torch.device,
    batch_size: int,
    window: torch.Tensor,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, input_specs.shape[0], batch_size):
            tensor = torch.from_numpy(input_specs[begin : begin + batch_size]).to(device)
            full = local.torch_packed_istft(model(tensor), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            useful = full[:, trim : trim + CONTRACT.useful_samples]
            outputs.append(np.ascontiguousarray(useful.cpu().numpy(), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def prepare_caches(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    device: torch.device,
) -> dict[str, Path]:
    cache_root = args.output_root.resolve() / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_root.resolve() / "selection.json"
    existing = None
    if selection_path.is_file() and not args.force_cache:
        try:
            existing = read_json(selection_path)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
    paths: dict[str, Path] = {}
    if existing and existing.get("selectionSha256") == selection["selectionSha256"]:
        valid = True
        for key, group in groups.items():
            path = cache_root / f"{safe_name(key)}.npz"
            if not path.is_file():
                valid = False
                break
            try:
                validate_arrays(path, len(group["events"]))
            except (OSError, ValueError):
                valid = False
                break
            paths[key] = path
        if valid and len(paths) == len(groups):
            print(f"reuse {len(paths)} event cache files", flush=True)
            return paths

    anchor_model, anchor_optimizer, anchor_source, _payload = load_source_model(args, device)
    del anchor_optimizer, _payload
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    cache_metadata: dict[str, Any] = {}
    try:
        for number, (key, group) in enumerate(groups.items(), start=1):
            source_path = Path(group["sourcePath"])
            teacher_path = Path(group["teacherPath"])
            h50_path = Path(group["h50Path"])
            source = external.load_audio(source_path)
            teacher = external.load_audio(teacher_path)
            h50 = external.load_audio(h50_path)
            if source.shape != teacher.shape or source.shape != h50.shape:
                raise ValueError(f"Source/teacher/H50 shape mismatch for {key}")
            if sha256_file(source_path) != group["sourceSha256"]:
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
                mask_rows.append(np.ascontiguousarray(c1.event_mask(CONTRACT.useful_samples, center - start, CORE_MS, GUARD_MS), dtype=np.float32))
            input_array = np.stack(input_rows).astype(np.float32)
            target_array = np.stack(target_rows).astype(np.float32)
            mask_array = np.stack(mask_rows).astype(np.float32)
            anchor_array = infer_anchor(anchor_model, input_array, device, args.inference_batch_size, window)
            if anchor_array.shape != target_array.shape or not np.isfinite(anchor_array).all():
                raise ValueError(f"Invalid anchor output for {key}")
            path = cache_root / f"{safe_name(key)}.npz"
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
                "key": key,
                "poolType": "S" if any(event["mark"] == "S" for event in group["events"]) else "R",
                "recordCount": len(group["events"]),
                "sourcePath": str(source_path),
                "teacherPath": str(teacher_path),
                "h50Path": str(h50_path),
                "sourceSha256": group["sourceSha256"],
                "teacherSha256": sha256_file(teacher_path),
                "h50Sha256": sha256_file(h50_path),
                "anchorCheckpoint": anchor_source["checkpoint"],
                "contract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
                "coreMs": CORE_MS,
                "guardMs": GUARD_MS,
                "events": group["events"],
                "cache": checkpoint_metadata(path),
            }
            json_write(path.with_suffix(".json"), metadata)
            paths[key] = path
            cache_metadata[key] = metadata
            print(f"prepared cache {number}/{len(groups)}: {key} ({len(group['events'])} records)", flush=True)
            del source, teacher, h50, input_array, target_array, anchor_array, mask_array
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del anchor_model, window
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection_with_cache = {
        **selection,
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint),
        "anchorSemantic": "S86-event-only@pass-5 residual output",
        "targetSemantic": "mixture - Inst3 instrumental on the event mask",
        "cacheContract": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "cacheMetadata": cache_metadata,
    }
    selection_with_cache["selectionSha256"] = selection["selectionSha256"]
    json_write(selection_path, selection_with_cache)
    return paths


def build_schedule(
    records: Sequence[dict[str, Any]],
    passes: int,
    seed: int,
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    s_items = [(record["key"], int(record["index"])) for record in records if record["mark"] == "S"]
    r_items = [(record["key"], int(record["index"])) for record in records if record["mark"] == "R"]
    if not s_items or not r_items:
        raise ValueError("Schedule needs both S and R records")
    unit_count = 2 * len(s_items) + len(r_items)
    repeat = RECORDS_PER_PASS // unit_count
    if repeat <= 0:
        raise ValueError("Record pool is too large for one weighted repetition")
    s_repeat = 2 * repeat
    r_repeat = repeat
    base_count = len(s_items) * s_repeat + len(r_items) * r_repeat
    remainder = RECORDS_PER_PASS - base_count
    if remainder < 0 or remainder > len(s_items):
        raise ValueError(f"Unexpected weighted schedule remainder: {remainder}")
    result: list[tuple[str, int]] = []
    details: list[dict[str, Any]] = []
    label_map = {(record["key"], int(record["index"])): record["mark"] for record in records}
    for pass_index in range(passes):
        rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97)
        s_permutation = [s_items[int(index)] for index in rng.permutation(len(s_items))]
        extras = s_permutation[:remainder]
        values = s_items * s_repeat + r_items * r_repeat + extras
        order = rng.permutation(len(values))
        ordered = [values[int(index)] for index in order]
        result.extend(ordered)
        counts = Counter(ordered)
        s_counts = [counts[item] for item in s_items]
        r_counts = [counts[item] for item in r_items]
        details.append(
            {
                "pass": pass_index + 1,
                "recordCount": len(ordered),
                "sRecords": sum(label_map[item] == "S" for item in ordered),
                "rRecords": sum(label_map[item] == "R" for item in ordered),
                "sUniqueRecords": len(s_items),
                "rUniqueRecords": len(r_items),
                "sRepeatBase": s_repeat,
                "rRepeatBase": r_repeat,
                "extraSRecords": len(extras),
                "sCountMin": min(s_counts),
                "sCountMax": max(s_counts),
                "rCountMin": min(r_counts),
                "rCountMax": max(r_counts),
                "sToRBaseRepeatRatio": s_repeat / r_repeat,
                "extraEventIds": [
                    next(record["eventId"] for record in records if (record["key"], int(record["index"])) == item)
                    for item in extras
                ],
            }
        )
    summary = {
        "recordsPerPass": RECORDS_PER_PASS,
        "passes": details,
        "sUniqueRecords": len(s_items),
        "rUniqueRecords": len(r_items),
        "sRepeatBase": s_repeat,
        "rRepeatBase": r_repeat,
        "weightedInterpretation": "each S event is sampled about twice per R event; no song-level balancing",
        "recordCount": len(result),
        "updatesPerPass": RECORDS_PER_PASS // BATCH_SIZE,
        "scheduleSha256": canonical_sha256(result),
    }
    return result, summary


def load_cache_store(paths: dict[str, Path]) -> local.CacheStore:
    store = local.CacheStore(paths, max_open=len(paths))
    store.preload()
    return store


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    contract_id: str,
    schedule_hash: str,
    selection: dict[str, Any],
    history: list[dict[str, Any]],
    pass_number: int,
    global_step: int,
) -> dict[str, Any]:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": "FMA-SR-event-only",
        "runContractId": contract_id,
        "step": global_step,
        "globalStep": global_step,
        "sourceStep": SOURCE_STEP,
        "sourcePass": SOURCE_PASS,
        "continuationPass": pass_number,
        "recordsPerPass": RECORDS_PER_PASS,
        "updatesPerPass": RECORDS_PER_PASS // args.batch_size,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_hash,
        "selectionSha256": selection["selectionSha256"],
        "lossContract": "S86-event-only masked Inst3 target plus S86 anchor outside 100ms+25ms event mask",
        "stateDict": local.hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": local.hard.cpu_tree(optimizer.state_dict()),
        "history": history,
    }
    return local.hard.atomic_torch_save(path, payload)


def evaluate_model(
    model: torch.nn.Module,
    groups: dict[str, dict[str, Any]],
    device: torch.device,
    inference_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    by_mark: dict[str, list[dict[str, Any]]] = {"S": [], "R": []}
    all_rows: list[dict[str, Any]] = []
    song_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for key, group in groups.items():
        source = external.load_audio(Path(group["sourcePath"]))
        teacher = external.load_audio(Path(group["teacherPath"]))
        residual, timing = continuous.render_student(model, source, CONTRACT, device, inference_batch_size)
        candidate = np.ascontiguousarray(source - residual, dtype=np.float32)
        song_rows: list[dict[str, Any]] = []
        for event in group["events"]:
            metrics = {
                str(ms): external.block_metric(
                    source,
                    teacher,
                    candidate,
                    int(event["centerSamples"]),
                    ms,
                )
                for ms in (50, 100, 200)
            }
            row = {"eventId": event["eventId"], "mark": event["mark"], "metrics": metrics}
            all_rows.append(row)
            by_mark[event["mark"]].append(row)
            song_rows.append(row)
        song_summaries.append(
            {
                "key": key,
                "song": f"{group['artistName']} - {group['trackName']}",
                "eventCount": len(song_rows),
                "render": timing,
                "p95_100ms": float(np.percentile([row["metrics"]["100"]["positiveProjectionDbfs"] for row in song_rows], 95.0)),
                "max_100ms": float(max(row["metrics"]["100"]["positiveProjectionDbfs"] for row in song_rows)),
            }
        )
        del source, teacher, residual, candidate
    def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {"count": len(rows)}
        for ms in (50, 100, 200):
            projection = np.asarray([row["metrics"][str(ms)]["positiveProjectionDbfs"] for row in rows], dtype=np.float64)
            miss = np.asarray([row["metrics"][str(ms)]["missRmsDbfs"] for row in rows], dtype=np.float64)
            output[str(ms)] = {
                "positiveProjectionP50Dbfs": float(np.percentile(projection, 50.0)),
                "positiveProjectionP90Dbfs": float(np.percentile(projection, 90.0)),
                "positiveProjectionP95Dbfs": float(np.percentile(projection, 95.0)),
                "positiveProjectionMaxDbfs": float(np.max(projection)),
                "missRmsP95Dbfs": float(np.percentile(miss, 95.0)),
                "missRmsMaxDbfs": float(np.max(miss)),
            }
        return output
    result = {
        "elapsedSeconds": time.perf_counter() - started,
        "all": summarize(all_rows),
        "byMark": {mark: summarize(rows) for mark, rows in by_mark.items()},
        "songs": song_summaries,
        "eventRows": all_rows,
    }
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    return result


def train_pass(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    store: local.CacheStore,
    schedule: Sequence[tuple[str, int]],
    pass_index: int,
    args: argparse.Namespace,
    device: torch.device,
    max_updates: int | None = None,
) -> list[dict[str, Any]]:
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    updates = RECORDS_PER_PASS // args.batch_size
    if max_updates is not None:
        updates = min(updates, max_updates)
    history: list[dict[str, Any]] = []
    begin_pass = pass_index * RECORDS_PER_PASS
    started = time.perf_counter()
    for update in range(updates):
        items = schedule[begin_pass + update * args.batch_size : begin_pass + (update + 1) * args.batch_size]
        input_array, target_array, anchor_array, mask_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        anchor_tensor = torch.from_numpy(anchor_array).to(device)
        mask_tensor = torch.from_numpy(mask_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = full[:, trim : trim + CONTRACT.useful_samples]
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
        backward_seconds = time.perf_counter() - backward_started
        row = {
            "pass": pass_index + 1,
            "update": update + 1,
            "globalStep": SOURCE_STEP + (pass_index + 1) * updates - (updates - update - 1),
            "loss": float(loss.detach().cpu()),
            "eventLoss": float(event_losses.mean().detach().cpu()),
            "anchorLoss": float(anchor_losses.mean().detach().cpu()),
            "gradientNormBeforeClip": gradient,
            "forwardSeconds": forward_seconds,
            "backwardSeconds": backward_seconds,
            "cudaAllocatedBytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
        }
        history.append(row)
        if update == 0 or (update + 1) % args.log_every == 0 or update + 1 == updates:
            print(json.dumps({"event": "progress", **row, "updates": updates}, sort_keys=True), flush=True)
    print(json.dumps({"event": "pass-complete", "pass": pass_index + 1, "elapsedSeconds": time.perf_counter() - started}, sort_keys=True), flush=True)
    del window
    return history


def save_report(path: Path, report: dict[str, Any]) -> None:
    json_write(path, report)


def run_training(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    selection: dict[str, Any],
    paths: dict[str, Path],
    schedule: list[tuple[str, int]],
    schedule_summary: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    schedule_hash = schedule_summary["scheduleSha256"]
    contract_payload = {
        "schema": SCHEMA,
        "sourceCheckpointSha256": selection["sourceCheckpoint"]["sha256"],
        "selectionSha256": selection["selectionSha256"],
        "scheduleSha256": schedule_hash,
        "recordsPerPass": RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "loss": "event-mask Inst3 target plus S86 anchor outside mask",
        "officialFinalTestUsed": False,
    }
    contract_id = canonical_sha256(contract_payload)
    output_root = args.output_root.resolve()
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "reports" / "training-report.json"
    model, optimizer, source, _payload = load_source_model(args, device)
    store = load_cache_store(paths)
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    completed_passes = 0
    if args.resume and not args.smoke_only:
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        for path in run_root.glob("step-*.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("runContractId") == contract_id:
                    candidates.append((int(payload.get("continuationPass", -1)), path, payload))
            except (OSError, ValueError, RuntimeError, TypeError):
                continue
        if candidates:
            completed_passes, path, payload = max(candidates, key=lambda item: item[0])
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            prior.continuation.move_optimizer_state(optimizer, device)
            for group_state in optimizer.param_groups:
                group_state["lr"] = args.learning_rate
            history = list(payload.get("history", []))
            if report_path.is_file():
                old_report = read_json(report_path)
                evaluations = list(old_report.get("evaluations", []))
            print(json.dumps({"event": "resume", "pass": completed_passes, "checkpoint": str(path)}, sort_keys=True), flush=True)
    if args.smoke_only:
        model.train()
        sweep.freeze_batchnorm_running_statistics(model)
        smoke_history = train_pass(model, optimizer, store, schedule, 0, args, device, args.smoke_updates)
        return {"status": "smoke-completed", "updates": len(smoke_history), "selection": selection, "schedule": schedule_summary}
    if not evaluations:
        baseline = evaluate_model(model, groups, device, args.inference_batch_size)
        evaluations.append({"pass": 0, "globalStep": SOURCE_STEP, "metrics": baseline, "improvementDb": None})
        print(json.dumps({"event": "baseline", "pass": 0, "p95_100ms": baseline["all"]["100"]["positiveProjectionP95Dbfs"]}, sort_keys=True), flush=True)
    no_improvement = 0
    previous_p95 = float(evaluations[-1]["metrics"]["all"]["100"]["positiveProjectionP95Dbfs"])
    for pass_index in range(completed_passes, args.max_passes):
        pass_history = train_pass(model, optimizer, store, schedule, pass_index, args, device)
        history.extend(pass_history)
        global_step = SOURCE_STEP + (pass_index + 1) * (RECORDS_PER_PASS // args.batch_size)
        checkpoint_path = run_root / f"step-{global_step}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            args,
            contract_id,
            schedule_hash,
            selection,
            history,
            pass_index + 1,
            global_step,
        )
        metrics = evaluate_model(model, groups, device, args.inference_batch_size)
        current_p95 = float(metrics["all"]["100"]["positiveProjectionP95Dbfs"])
        improvement = previous_p95 - current_p95
        evaluations.append(
            {
                "pass": pass_index + 1,
                "globalStep": global_step,
                "checkpoint": checkpoint_metadata(checkpoint_path),
                "metrics": metrics,
                "improvementDb": improvement,
                "meaningfulImprovement": improvement >= args.min_improvement_db,
            }
        )
        report = {
            "schema": SCHEMA,
            "status": "running",
            "selection": selection,
            "schedule": schedule_summary,
            "contract": {"id": contract_id, "payload": contract_payload},
            "source": source,
            "training": {"completedPasses": pass_index + 1, "history": history},
            "evaluations": evaluations,
            "stop": {"patience": args.patience, "minImprovementDb": args.min_improvement_db},
            "environment": {
                "device": str(device),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        save_report(report_path, report)
        print(json.dumps({"event": "evaluation", "pass": pass_index + 1, "globalStep": global_step, "p95_100ms": current_p95, "improvementDb": improvement}, sort_keys=True), flush=True)
        if improvement >= args.min_improvement_db:
            no_improvement = 0
        else:
            no_improvement += 1
        previous_p95 = current_p95
        if pass_index + 1 >= args.min_passes and no_improvement >= args.patience:
            report["status"] = "completed"
            report["stop"] = {
                "reason": "100ms projection p95 plateau",
                "stoppedAfterPass": pass_index + 1,
                "consecutiveNonMeaningfulPasses": no_improvement,
                "minImprovementDb": args.min_improvement_db,
                "patience": args.patience,
            }
            save_report(report_path, report)
            break
    else:
        report = read_json(report_path) if report_path.is_file() else {}
        report["status"] = "completed"
        report["stop"] = {
            "reason": "max-passes guard",
            "maxPasses": args.max_passes,
            "minImprovementDb": args.min_improvement_db,
            "patience": args.patience,
        }
        save_report(report_path, report)
    del model, optimizer, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return read_json(report_path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_passes <= 0 or args.min_passes <= 0 or args.min_passes > args.max_passes:
        raise ValueError("invalid pass limits")
    if args.patience <= 0 or args.min_improvement_db < 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid stopping or batch settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    groups, records, selection = load_pool(args)
    selection["sourceCheckpoint"] = checkpoint_metadata(args.source_checkpoint)
    json_write(args.output_root / "selected-pool.json", selection)
    paths = prepare_caches(args, groups, selection, device)
    schedule, schedule_summary = build_schedule(records, args.max_passes, args.seed)
    json_write(args.output_root / "schedule.json", {"summary": schedule_summary, "schedule": schedule})
    if args.smoke_only:
        result = run_training(args, groups, records, selection, paths, schedule, schedule_summary, device)
        json_write(args.output_root / "reports" / "smoke-report.json", result)
        print(json.dumps({"status": "smoke-completed", "report": str((args.output_root / 'reports' / 'smoke-report.json').resolve())}, indent=2), flush=True)
        return 0
    result = run_training(args, groups, records, selection, paths, schedule, schedule_summary, device)
    print(json.dumps({"status": result.get("status"), "report": str((args.output_root / 'reports' / 'training-report.json').resolve()), "completedPasses": result.get("training", {}).get("completedPasses")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
