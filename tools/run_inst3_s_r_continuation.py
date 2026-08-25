#!/usr/bin/env python3
"""Continue the best S-only model with a balanced stable-S/stable-R pilot.

The experiment starts from the continuous ``S86-masked-anchor`` checkpoint.
The first two arms keep the same 640-record MUSDB base schedule and 64-record
external budget.  The control uses reviewed S events for all external records;
the treatment replaces half of those records with two-pass consensus
stable-aggressive (R) events.  Two additional arms remove the MUSDB base
entirely and spend the same 704-record budget on S-only or combined S+R event
records.

External records use the residual-vocals contract:

* the 100 ms core plus a 25 ms guard follows ``V_T = mixture - Inst3``;
* the remaining useful samples are anchored to the S86 source output.

All caches are rebuilt or re-anchored locally so the old H50 anchor cannot
silently enter this continuation.  Rendering uses continuous overlap-save
assembly.  Generated data stays under the ignored ``data`` tree.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import render_inst3_mtg_fma_event_listening as external
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_mtg_fma_c1 as c1
import run_inst3_s_only_stress as stress
import run_inst3_vr_continuation as continuation
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H50_SOURCE = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-continuation"
    / "runs"
    / "H50-continuation-plus5"
    / "step-1600.pt"
)
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "modern-song-s-only-stress"
    / "runs"
    / "S86-masked-anchor"
    / "step-2480.pt"
)
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_S_OLD_ROOT = ROOT / "data" / "modern-song-s-pilot"
DEFAULT_S_CURRENT_ROOT = ROOT / "data" / "modern-song-s-combined-pilot"
DEFAULT_R_POOL = ROOT / "data" / "modern-song-batch2-full-event-listening" / "two-pass-event-pools.csv"
DEFAULT_R_REPORT = ROOT / "data" / "modern-song-batch2-full-event-listening" / "event-listening-report.json"
DEFAULT_R_MANIFEST = ROOT / "data" / "modern-song-original-candidates-batch2" / "source-manifest.json"
DEFAULT_R_FULL_ROOT = ROOT / "data" / "modern-song-batch2-inst3-event-listening"
DEFAULT_MUSDB_CACHE = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local"
DEFAULT_MUSDB_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_BASELINE_REPORT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "continuous-baseline-report.json"
DEFAULT_SAMPLES = ROOT / "data" / "samples"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-s-r-continuation"

SAMPLE_RATE = 44_100
SOURCE_STEP = 2_480
PASSES = 5
MUSDB_RECORDS_PER_PASS = 640
EXTERNAL_RECORDS_PER_PASS = 64
RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS + EXTERNAL_RECORDS_PER_PASS
BATCH_SIZE = 4
CORE_MS = 100
GUARD_MS = 25
S_RECORDS = 86
R_RECORDS = 46
CHECKPOINT_FORMAT = "local-inst3-s-r-continuation-checkpoint@1"
SCHEMA = "local-inst3-s-r-continuation@1"
ARMS = (
    "S86-continuation-control",
    "SR-balanced",
    "S86-event-only",
    "SR-event-only",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--h50-source", type=Path, default=DEFAULT_H50_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--s-old-root", type=Path, default=DEFAULT_S_OLD_ROOT)
    parser.add_argument("--s-current-root", type=Path, default=DEFAULT_S_CURRENT_ROOT)
    parser.add_argument("--r-pool", type=Path, default=DEFAULT_R_POOL)
    parser.add_argument("--r-report", type=Path, default=DEFAULT_R_REPORT)
    parser.add_argument("--r-manifest", type=Path, default=DEFAULT_R_MANIFEST)
    parser.add_argument("--r-full-root", type=Path, default=DEFAULT_R_FULL_ROOT)
    parser.add_argument("--musdb-cache-root", type=Path, default=DEFAULT_MUSDB_CACHE)
    parser.add_argument("--musdb-manifest", type=Path, default=DEFAULT_MUSDB_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--milestones", default="1,3,5")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-prep", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--all-milestones", action="store_true", help="Evaluate and render pass 1/3/5 instead of pass 5 only")
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


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "-" for char in value).strip("-") or "song"


def parse_milestones(raw: str, passes: int) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values or values[-1] != passes or values[0] <= 0:
        raise ValueError("milestones must be sorted, positive, unique, and end at passes")
    return values


def validate_arrays(path: Path, expected_records: int | None = None) -> dict[str, Any]:
    required = ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
    with np.load(path) as values:
        if any(name not in values for name in required):
            raise ValueError(f"Missing arrays in {path}")
        arrays = {name: values[name] for name in required}
        records = int(arrays["inputSpec"].shape[0])
        if expected_records is not None and records != expected_records:
            raise ValueError(f"Expected {expected_records} records in {path}, got {records}")
        if arrays["inputSpec"].shape[1:] != (4, 1025, 128):
            raise ValueError(f"Unexpected input shape in {path}: {arrays['inputSpec'].shape}")
        if arrays["targetAudio"].shape[1:] != (119808, 2):
            raise ValueError(f"Unexpected target shape in {path}: {arrays['targetAudio'].shape}")
        if arrays["anchorAudio"].shape != arrays["targetAudio"].shape:
            raise ValueError(f"Anchor shape mismatch in {path}")
        if arrays["eventMask"].shape != (records, 119808):
            raise ValueError(f"Unexpected mask shape in {path}: {arrays['eventMask'].shape}")
        if any(not np.isfinite(arrays[name]).all() for name in required):
            raise ValueError(f"Non-finite cache values in {path}")
    return {"file": str(path.resolve()), "sha256": sha256_file(path), "records": records}


def load_required_arrays(path: Path) -> dict[str, np.ndarray]:
    validate_arrays(path)
    with np.load(path) as values:
        return {
            name: np.ascontiguousarray(values[name], dtype=np.float32)
            for name in ("inputSpec", "targetAudio", "eventMask")
        }


def load_source_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    path = args.source_checkpoint.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-s-only-stress-checkpoint@1":
        raise ValueError(f"Unexpected S86 source format: {payload.get('format')}")
    if int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(f"Expected S86 source step {SOURCE_STEP}, got {payload.get('step')}")
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("S86 source has no optimizer state")
    model, architecture = pilot.make_model(args.checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    continuation.move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    return model, optimizer, {
        "checkpoint": checkpoint_metadata(path),
        "architecture": architecture["checkpoint"],
        "sourceStep": int(payload["step"]),
        "sourceVariant": payload.get("variant"),
        "optimizerRestored": True,
    }


def infer_anchor(
    model: torch.nn.Module,
    input_spec: np.ndarray,
    device: torch.device,
    batch_size: int,
    window: torch.Tensor,
) -> np.ndarray:
    if input_spec.ndim != 4 or input_spec.shape[1:] != (4, 1025, 128):
        raise ValueError(f"Invalid input spec batch: {input_spec.shape}")
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, input_spec.shape[0], batch_size):
            batch = torch.from_numpy(input_spec[begin : begin + batch_size]).to(device)
            full = local.torch_packed_istft(model(batch), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            useful = full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
            outputs.append(np.ascontiguousarray(useful.detach().cpu().numpy(), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def write_cache(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        inputSpec=arrays["inputSpec"],
        targetAudio=arrays["targetAudio"],
        anchorAudio=arrays["anchorAudio"],
        eventMask=arrays["eventMask"],
    )
    temporary.replace(path)
    info = {
        **metadata,
        "cache": checkpoint_metadata(path),
        "shapes": {name: list(arrays[name].shape) for name in ("inputSpec", "targetAudio", "anchorAudio", "eventMask")},
    }
    json_write(path.with_suffix(".json"), info)
    return info


def prepare_reanchored_cache(
    *,
    input_path: Path,
    output_path: Path,
    anchor_model: torch.nn.Module,
    device: torch.device,
    inference_batch_size: int,
    window: torch.Tensor,
    metadata: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    metadata_path = output_path.with_suffix(".json")
    input_source_sha = sha256_file(input_path)
    metadata = {**metadata, "inputSourceSha256": input_source_sha}
    if not force and output_path.is_file() and metadata_path.is_file():
        try:
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                old.get("anchorSourceSha256") == metadata.get("anchorSourceSha256")
                and old.get("inputSourceSha256") == metadata.get("inputSourceSha256")
            ):
                validate_arrays(output_path, expected_records=int(old["recordCount"]))
                return old
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    arrays = load_required_arrays(input_path)
    arrays["anchorAudio"] = infer_anchor(
        anchor_model,
        arrays["inputSpec"],
        device,
        inference_batch_size,
        window,
    )
    metadata = {
        **metadata,
        "recordCount": int(arrays["inputSpec"].shape[0]),
        "inputSource": str(input_path.resolve()),
        "inputSourceSha256": input_source_sha,
    }
    return write_cache(output_path, arrays, metadata)


def load_s_pool(args: argparse.Namespace) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    h50_sha = sha256_file(args.h50_source.resolve())
    old_paths, old_records, _ = stress.load_pool_selection(args.s_old_root, h50_sha, "previous")
    current_paths, current_records, _ = stress.load_pool_selection(args.s_current_root, h50_sha, "current")
    if len(old_records) != 32 or len(current_records) != 54:
        raise ValueError(f"Expected 32 old and 54 current S records, got {len(old_records)} and {len(current_records)}")
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    key_map: dict[str, str] = {}
    for pool_name, source_paths, source_records in (
        ("previous", old_paths, old_records),
        ("current", current_paths, current_records),
    ):
        for old_key, path in sorted(source_paths.items()):
            slug = safe_slug(old_key.split("::")[-1])
            new_key = f"external::S::{pool_name}::{slug}"
            if new_key in paths:
                raise ValueError(f"Duplicate S cache key: {new_key}")
            paths[new_key] = path
            key_map[old_key] = new_key
        for record in source_records:
            copy = dict(record)
            copy["key"] = key_map[record["key"]]
            copy["poolType"] = "S"
            copy["songKey"] = copy["key"]
            records.append(copy)
    if len(paths) != 38 or len(records) != S_RECORDS:
        raise ValueError(f"Expected 38 S songs and 86 records, got {len(paths)} and {len(records)}")
    return paths, records


def load_r_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    with args.r_pool.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("pool") == "stable-aggressive" and row.get("splitRole") == "train"
    ]
    if len(selected) != R_RECORDS:
        raise ValueError(f"Expected {R_RECORDS} stable-aggressive train rows, got {len(selected)}")
    report = json.loads(args.r_report.resolve().read_text(encoding="utf-8"))
    events = {str(event["eventId"]): event for event in report.get("events", [])}
    manifest = json.loads(args.r_manifest.resolve().read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Incomplete R manifest: {args.r_manifest}")
    by_order = {int(record["order"]): record for record in manifest.get("records", [])}
    return selected, events, by_order


def prepare_r_caches(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    anchor_model: torch.nn.Module,
    device: torch.device,
    window: torch.Tensor,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    rows, events, manifest = load_r_rows(args)
    grouped: dict[tuple[int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        event_id = row.get("secondEventId", "")
        event = events.get(event_id)
        if event is None:
            raise ValueError(f"Missing event report row: {event_id}")
        order = int(row["sourceOrder"])
        slug = str(event["slug"])
        if order not in manifest or manifest[order].get("slug") != slug:
            raise ValueError(f"R manifest mismatch for {order}/{slug}")
        grouped[(order, slug)].append((row, event))

    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for (order, slug), items in sorted(grouped.items()):
        source_record = manifest[order]
        source_path = Path(source_record["download"]["file"]).resolve()
        teacher_path = args.r_full_root.resolve() / "full-song" / "inst3Instrumental" / f"{order:03d}-{slug}.flac"
        h50_path = args.r_full_root.resolve() / "full-song" / "h50ContinuationPlus5" / f"{order:03d}-{slug}.flac"
        for path in (source_path, teacher_path, h50_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        source = external.load_audio(source_path)
        teacher = external.load_audio(teacher_path)
        h50 = external.load_audio(h50_path)
        if source.shape != teacher.shape or source.shape != h50.shape:
            raise ValueError(f"R source/teacher/H50 shape mismatch for {slug}")
        input_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        event_metadata: list[dict[str, Any]] = []
        for row, event in sorted(items, key=lambda pair: int(pair[1]["centerSamples"])):
            center = int(event["centerSamples"])
            start = center - contract.useful_samples // 2
            end = start + contract.useful_samples
            if start < 0 or end > source.shape[0]:
                raise ValueError(f"R event context out of bounds: {event['eventId']}")
            assembled = local.assemble_input(source, start, contract.useful_samples, contract, mode="continuous")
            input_rows.append(np.ascontiguousarray(local.stft_centered(assembled, contract)[0], dtype=np.float32))
            target_rows.append(np.ascontiguousarray(source[start:end] - teacher[start:end], dtype=np.float32))
            mask_rows.append(np.ascontiguousarray(c1.event_mask(contract.useful_samples, center - start, CORE_MS, GUARD_MS), dtype=np.float32))
            event_metadata.append(
                {
                    "eventId": event["eventId"],
                    "centerSamples": center,
                    "startSamples": start,
                    "category": row.get("category", ""),
                    "artistName": row.get("artistName", ""),
                    "trackName": row.get("trackName", ""),
                    "firstMark": row.get("firstMark", ""),
                    "secondMark": row.get("secondMark", ""),
                }
            )
        arrays = {
            "inputSpec": np.stack(input_rows).astype(np.float32),
            "targetAudio": np.stack(target_rows).astype(np.float32),
            "eventMask": np.stack(mask_rows).astype(np.float32),
        }
        key = f"external::R::{order:03d}-{safe_slug(slug)}"
        output_path = args.output_root.resolve() / "cache" / "external" / "R" / f"{order:03d}-{safe_slug(slug)}.npz"
        metadata = {
            "schema": "local-inst3-s-r-external-cache@1",
            "poolType": "R",
            "key": key,
            "slug": slug,
            "sourceOrder": order,
            "recordCount": len(items),
            "selectedEvents": event_metadata,
            "rawSource": str(source_path),
            "teacherFile": str(teacher_path),
            "h50File": str(h50_path),
            "rawSha256": source_record["download"].get("sha256"),
            "teacherSha256": sha256_file(teacher_path),
            "h50Sha256": sha256_file(h50_path),
            "anchorSourceSha256": sha256_file(args.source_checkpoint.resolve()),
        }
        # Reuse the same re-anchoring path as the S and MUSDB caches.
        temporary_input = output_path.with_name(output_path.name + ".unanchored.npz")
        temporary_input.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            temporary_input,
            inputSpec=arrays["inputSpec"],
            targetAudio=arrays["targetAudio"],
            anchorAudio=np.zeros_like(arrays["targetAudio"]),
            eventMask=arrays["eventMask"],
        )
        prepare_reanchored_cache(
            input_path=temporary_input,
            output_path=output_path,
            anchor_model=anchor_model,
            device=device,
            inference_batch_size=args.inference_batch_size,
            window=window,
            metadata=metadata,
            force=args.force_prep,
        )
        temporary_input.unlink(missing_ok=True)
        paths[key] = output_path
        for index, event in enumerate(event_metadata):
            records.append(
                {
                    "key": key,
                    "index": index,
                    "poolType": "R",
                    "songKey": key,
                    "slug": slug,
                    "eventId": event["eventId"],
                    "centerSamples": event["centerSamples"],
                    "category": event["category"],
                    "artistName": event["artistName"],
                    "trackName": event["trackName"],
                }
            )
        del source, teacher, h50
        gc.collect()
        print(f"prepared R cache {order:03d}-{slug} ({len(items)} events)", flush=True)
    if len(records) != R_RECORDS:
        raise AssertionError(f"R cache record count mismatch: {len(records)}")
    return paths, records


def prepare_all_caches(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    device: torch.device,
) -> tuple[dict[str, Path], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    anchor_model, _optimizer, anchor_source = load_source_model(args, device)
    anchor_model.eval()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    paths: dict[str, Path] = {}
    s_records: list[dict[str, Any]] = []
    r_records: list[dict[str, Any]] = []
    try:
        s_source_paths, s_records = load_s_pool(args)
        for index, (key, input_path) in enumerate(sorted(s_source_paths.items()), start=1):
            pool_name = key.split("::")[2]
            slug = key.split("::")[-1]
            output_path = args.output_root.resolve() / "cache" / "external" / "S" / f"{pool_name}-{slug}.npz"
            metadata = {
                "schema": "local-inst3-s-r-external-cache@1",
                "poolType": "S",
                "key": key,
                "slug": slug,
                "recordCount": None,
                "anchorSourceSha256": sha256_file(args.source_checkpoint.resolve()),
                "h50SourceSha256": sha256_file(args.h50_source.resolve()),
            }
            prepare_reanchored_cache(
                input_path=input_path,
                output_path=output_path,
                anchor_model=anchor_model,
                device=device,
                inference_batch_size=args.inference_batch_size,
                window=window,
                metadata=metadata,
                force=args.force_prep,
            )
            paths[key] = output_path
            if index == 1 or index % 10 == 0 or index == len(s_source_paths):
                print(f"prepared S caches {index}/{len(s_source_paths)}", flush=True)

        musdb_selection = json.loads((args.musdb_cache_root.resolve() / "selection.json").read_text(encoding="utf-8"))
        records_per_song = int(musdb_selection["contract"]["recordsPerSong"])
        musdb_source_paths = {
            slug: args.musdb_cache_root.resolve() / "cache" / "train" / f"{slug}.npz"
            for slug in sorted(musdb_selection["songs"])
        }
        if len(musdb_source_paths) * records_per_song != MUSDB_RECORDS_PER_PASS:
            raise ValueError("MUSDB cache does not contain the expected 80 x 8 records")
        for index, (slug, input_path) in enumerate(sorted(musdb_source_paths.items()), start=1):
            key = f"musdb::{safe_slug(slug)}"
            output_path = args.output_root.resolve() / "cache" / "musdb" / f"{safe_slug(slug)}.npz"
            metadata = {
                "schema": "local-inst3-s-r-musdb-cache@1",
                "poolType": "MUSDB",
                "key": key,
                "slug": slug,
                "recordCount": records_per_song,
                "anchorSourceSha256": sha256_file(args.source_checkpoint.resolve()),
                "inputSourceSha256": sha256_file(input_path),
            }
            prepare_reanchored_cache(
                input_path=input_path,
                output_path=output_path,
                anchor_model=anchor_model,
                device=device,
                inference_batch_size=args.inference_batch_size,
                window=window,
                metadata=metadata,
                force=args.force_prep,
            )
            paths[key] = output_path
            if index == 1 or index % 10 == 0 or index == len(musdb_source_paths):
                print(f"prepared MUSDB caches {index}/{len(musdb_source_paths)}", flush=True)

        r_paths, r_records = prepare_r_caches(args, contract, anchor_model, device, window)
        paths.update(r_paths)
    finally:
        del anchor_model, _optimizer, window
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for path in paths.values():
        validate_arrays(path)
    if len(s_records) != S_RECORDS or len(r_records) != R_RECORDS:
        raise AssertionError("Unexpected S/R record count after cache preparation")
    selection = {
        "schema": "local-inst3-s-r-continuation-selection@1",
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "h50Source": checkpoint_metadata(args.h50_source.resolve()),
        "architecture": checkpoint_metadata(args.checkpoint.resolve()),
        "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
        "sRecordCount": len(s_records),
        "rRecordCount": len(r_records),
        "musdbRecordCount": MUSDB_RECORDS_PER_PASS,
        "cacheCount": len(paths),
        "sEventIds": [record["eventId"] for record in s_records],
        "rEventIds": [record["eventId"] for record in r_records],
        "anchorSemantic": "S86-masked-anchor-pass-5 residual output",
        "targetSemantic": "Inst3 residual output on external event masks; full Inst3 residual on MUSDB base",
        "anchorSourceMetadata": anchor_source,
    }
    selection["selectionSha256"] = canonical_sha256(selection)
    json_write(args.output_root.resolve() / "selection.json", selection)
    return paths, s_records, r_records, selection


def choose_balanced(
    records: Sequence[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
    max_per_song: int = 2,
) -> list[tuple[str, int]]:
    if count <= 0 or count > len(records):
        raise ValueError(f"Invalid balanced sample count {count} for {len(records)} records")
    order = rng.permutation(len(records))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    deferred: list[dict[str, Any]] = []
    for raw_index in order:
        record = records[int(raw_index)]
        song = str(record.get("songKey", record["key"]))
        if counts[song] < max_per_song:
            selected.append(record)
            counts[song] += 1
            if len(selected) == count:
                break
        else:
            deferred.append(record)
    if len(selected) < count:
        selected.extend(deferred[: count - len(selected)])
    if len(selected) != count:
        raise AssertionError(f"Could not select {count} balanced records")
    return [(record["key"], int(record["index"])) for record in selected]


def build_schedules(
    paths: dict[str, Path],
    s_records: Sequence[dict[str, Any]],
    r_records: Sequence[dict[str, Any]],
    passes: int,
    seed: int,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, Any]]:
    musdb_keys = sorted(key for key in paths if key.startswith("musdb::"))
    if len(musdb_keys) * 8 != MUSDB_RECORDS_PER_PASS:
        raise ValueError(f"Expected 80 eight-record MUSDB caches, got {len(musdb_keys)}")
    base = [(key, index) for key in musdb_keys for index in range(8)]
    schedules = {arm: [] for arm in ARMS}
    details: list[dict[str, Any]] = []
    combined_records = list(s_records) + list(r_records)
    for pass_index in range(passes):
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        base_order = [base[int(index)] for index in rng.permutation(len(base))]

        # The MUSDB arms keep the established 640 + 64 schedule.  The S
        # subset is shared so the treatment replaces, rather than adds to,
        # half of the control's external events.
        control_s = choose_balanced(s_records, EXTERNAL_RECORDS_PER_PASS, rng, max_per_song=2)
        treatment_s = control_s[: EXTERNAL_RECORDS_PER_PASS // 2]
        treatment_r = choose_balanced(r_records, EXTERNAL_RECORDS_PER_PASS // 2, rng, max_per_song=2)
        musdb_extras = {
            ARMS[0]: control_s,
            ARMS[1]: treatment_s + treatment_r,
        }

        # Event-only control: reproduce the S86 density (86 records x 8,
        # plus 16 deterministic balanced repeats).  Event-only treatment:
        # expose every consensus S and R record five times, then add 44
        # balanced repeats to reach the same 704-record compute budget.
        s_base_values = [(record["key"], int(record["index"])) for record in s_records]
        s_event_values = s_base_values * (RECORDS_PER_PASS // len(s_records))
        s_event_values.extend(choose_balanced(s_records, RECORDS_PER_PASS - len(s_event_values), rng, max_per_song=2))
        sr_event_values = [
            (record["key"], int(record["index"]))
            for record in combined_records
        ] * (RECORDS_PER_PASS // len(combined_records))
        remainder = RECORDS_PER_PASS - len(sr_event_values)
        sr_event_values.extend(
            choose_balanced(s_records, remainder // 2, rng, max_per_song=2)
        )
        sr_event_values.extend(
            choose_balanced(r_records, remainder - remainder // 2, rng, max_per_song=2)
        )
        arm_values = {
            ARMS[0]: base_order + musdb_extras[ARMS[0]],
            ARMS[1]: base_order + musdb_extras[ARMS[1]],
            ARMS[2]: s_event_values,
            ARMS[3]: sr_event_values,
        }
        for arm_index, arm in enumerate(ARMS):
            values = arm_values[arm]
            if len(values) != RECORDS_PER_PASS:
                raise AssertionError(f"{arm} pass has {len(values)} records")
            arm_rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97 + arm_index * 1009)
            schedules[arm].extend(values[int(index)] for index in arm_rng.permutation(len(values)))

        def ids(items: Sequence[tuple[str, int]], pool: Sequence[dict[str, Any]]) -> list[str]:
            selected = set(items)
            return [record["eventId"] for record in pool if (record["key"], int(record["index"])) in selected]

        details.append(
            {
                "pass": pass_index + 1,
                "musdbRecords": len(base_order),
                "musdbControlSRecords": len(control_s),
                "musdbTreatmentSRecords": len(treatment_s),
                "musdbTreatmentRRecords": len(treatment_r),
                "eventOnlySRecords": len(s_event_values),
                "eventOnlySUniqueRecords": len(set(s_event_values)),
                "eventOnlySRRecords": len(sr_event_values),
                "eventOnlySRUniqueRecords": len(set(sr_event_values)),
                "controlEventIds": ids(control_s, s_records),
                "treatmentSEventIds": ids(treatment_s, s_records),
                "treatmentREventIds": ids(treatment_r, r_records),
            }
        )
    if any(len(schedule) != passes * RECORDS_PER_PASS for schedule in schedules.values()):
        raise AssertionError("Schedule length mismatch")
    summary = {
        "passes": details,
        "recordsPerPass": RECORDS_PER_PASS,
        "updatesPerPass": RECORDS_PER_PASS // BATCH_SIZE,
        "schedules": {
            arm: {
                "recordCount": len(schedule),
                "sha256": canonical_sha256(schedule),
                "sourceCounts": dict(Counter("musdb" if key.startswith("musdb::") else "external" for key, _ in schedule)),
                "poolCounts": dict(
                    Counter(
                        "MUSDB" if key.startswith("musdb::")
                        else ("S" if "::S::" in key else "R")
                        for key, _ in schedule
                    )
                ),
                "uniqueExternalRecords": len({(key, index) for key, index in schedule if key.startswith("external::")}),
            }
            for arm, schedule in schedules.items()
        },
    }
    return schedules, summary


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    arm: str,
    local_step: int,
    contract_id: str,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    source: dict[str, Any],
    schedule_sha256: str,
    pool_counts: dict[str, int],
) -> dict[str, Any]:
    global_step = SOURCE_STEP + local_step
    payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": arm,
        "runContractId": contract_id,
        "step": global_step,
        "globalStep": global_step,
        "sourceStep": SOURCE_STEP,
        "localStep": local_step,
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_sha256,
        "poolCounts": pool_counts,
        "stateDict": local.hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": local.hard.cpu_tree(optimizer.state_dict()),
        "history": history,
        "source": source,
    }
    return local.hard.atomic_torch_save(path, payload)


def schedule_pool_counts(schedule: Sequence[tuple[str, int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key, _ in schedule:
        if key.startswith("musdb::"):
            counts["MUSDB"] += 1
        elif "::S::" in key:
            counts["S"] += 1
        elif "::R::" in key:
            counts["R"] += 1
        else:
            counts["external"] += 1
    return dict(counts)


def train_arm(
    arm: str,
    paths: dict[str, Path],
    schedule: list[tuple[str, int]],
    args: argparse.Namespace,
    device: torch.device,
    milestones: tuple[int, ...],
    common_contract: dict[str, Any],
    smoke_updates: int | None = None,
) -> dict[str, Any]:
    model, optimizer, source = load_source_model(args, device)
    store = local.CacheStore(paths, max_open=len(paths))
    store.preload()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    updates_per_pass = RECORDS_PER_PASS // args.batch_size
    total_updates = args.passes * updates_per_pass
    if smoke_updates is not None:
        total_updates = min(smoke_updates, total_updates)
    schedule_sha = canonical_sha256(schedule)
    pool_counts = schedule_pool_counts(schedule[:RECORDS_PER_PASS])
    contract = {
        **common_contract,
        "arm": arm,
        "scheduleSha256": schedule_sha,
        "lossContract": "MUSDB full Inst3 residual; external 100ms+25ms Inst3 event plus S86 anchor outside event",
    }
    contract_id = canonical_sha256(contract)
    run_root = args.output_root.resolve() / "runs" / arm
    run_root.mkdir(parents=True, exist_ok=True)
    current = 0
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    if args.resume and smoke_updates is None:
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        for path in run_root.glob("step-*.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("runContractId") == contract_id:
                    candidates.append((int(payload.get("localStep", -1)), path, payload))
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                continue
        if candidates:
            current, path, payload = max(candidates, key=lambda item: item[0])
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            continuation.move_optimizer_state(optimizer, device)
            history = list(payload.get("history", []))
            print(json.dumps({"event": "resume", "arm": arm, "localStep": current, "file": str(path)}), flush=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    milestone_steps = {value * updates_per_pass: value for value in milestones}
    if smoke_updates is None:
        for step, pass_count in milestone_steps.items():
            path = run_root / f"step-{SOURCE_STEP + step}.pt"
            if path.is_file():
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if payload.get("runContractId") == contract_id:
                        checkpoints[str(pass_count)] = checkpoint_metadata(path)
                except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                    pass
    started = time.perf_counter()
    while current < total_updates:
        items = schedule[current * args.batch_size : (current + 1) * args.batch_size]
        input_array, target_array, anchor_array, mask_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        anchor_tensor = torch.from_numpy(anchor_array).to(device)
        mask_tensor = torch.from_numpy(mask_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
        forward_seconds = time.perf_counter() - forward_started
        full_losses = local.charbonnier_per_record(predicted, target_tensor, torch.ones_like(mask_tensor))
        event_losses = local.charbonnier_per_record(predicted, target_tensor, mask_tensor)
        anchor_losses = local.charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor)
        musdb_flags = torch.tensor(
            [key.startswith("musdb::") for key, _ in items],
            dtype=torch.bool,
            device=device,
        )
        record_losses = torch.where(
            musdb_flags,
            full_losses,
            event_losses + args.anchor_beta * anchor_losses,
        )
        loss = record_losses.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {arm} loss at update {current + 1}")
        backward_started = time.perf_counter()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        if not math.isfinite(gradient):
            raise FloatingPointError(f"Non-finite {arm} gradient at update {current + 1}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        current += 1
        external_flags = ~musdb_flags
        history.append(
            {
                "localStep": current,
                "globalStep": SOURCE_STEP + current,
                "pass": current / updates_per_pass,
                "loss": float(loss.detach().cpu()),
                "musdbLoss": float(full_losses[musdb_flags].mean().detach().cpu()) if bool(musdb_flags.any()) else None,
                "externalEventLoss": float(event_losses[external_flags].mean().detach().cpu()) if bool(external_flags.any()) else None,
                "externalAnchorLoss": float(anchor_losses[external_flags].mean().detach().cpu()) if bool(external_flags.any()) else None,
                "externalRecords": int(external_flags.sum().item()),
                "gradientNormBeforeClip": gradient,
                "forwardSeconds": forward_seconds,
                "backwardSeconds": backward_seconds,
            }
        )
        if current == 1 or current % 100 == 0:
            gpu_memory = None
            if device.type == "cuda":
                gpu_memory = int(torch.cuda.memory_allocated(device))
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "arm": arm,
                        "update": current,
                        "totalUpdates": total_updates,
                        "pass": current / updates_per_pass,
                        "loss": history[-1]["loss"],
                        "forwardSeconds": forward_seconds,
                        "backwardSeconds": backward_seconds,
                        "cudaAllocatedBytes": gpu_memory,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if smoke_updates is None and current in milestone_steps:
            pass_count = milestone_steps[current]
            metadata = save_checkpoint(
                run_root / f"step-{SOURCE_STEP + current}.pt",
                model,
                optimizer,
                arm,
                current,
                contract_id,
                args,
                history,
                source,
                schedule_sha,
                pool_counts,
            )
            checkpoints[str(pass_count)] = metadata
            print(json.dumps({"event": "milestone", "arm": arm, "pass": pass_count, "step": SOURCE_STEP + current}, sort_keys=True), flush=True)
        elif smoke_updates is None and current % args.state_interval == 0:
            save_checkpoint(
                run_root / f"step-{SOURCE_STEP + current}.pt",
                model,
                optimizer,
                arm,
                current,
                contract_id,
                args,
                history,
                source,
                schedule_sha,
                pool_counts,
            )
    result = {
        "status": "smoke-completed" if smoke_updates is not None else "completed",
        "arm": arm,
        "updates": total_updates,
        "updatesPerPass": updates_per_pass,
        "recordsPerPass": RECORDS_PER_PASS,
        "externalRecordsPerPass": EXTERNAL_RECORDS_PER_PASS,
        "elapsedSeconds": time.perf_counter() - started,
        "history": history,
        "milestoneCheckpoints": checkpoints,
        "contract": {"id": contract_id, "payload": contract},
    }
    del model, optimizer, store, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_checkpoint_paths(
    args: argparse.Namespace,
    training: dict[str, dict[str, Any]],
    milestones: tuple[int, ...],
) -> dict[str, Path]:
    paths = {"S86-masked-anchor-source": args.source_checkpoint.resolve()}
    for arm in ARMS:
        for pass_count in milestones:
            metadata = training[arm]["milestoneCheckpoints"].get(str(pass_count))
            if metadata is None:
                raise ValueError(f"Missing checkpoint for {arm} pass {pass_count}")
            paths[f"{arm}@pass-{pass_count}"] = Path(metadata["file"])
    return paths


def select_report_checkpoints(
    checkpoint_paths: dict[str, Path],
    milestones: tuple[int, ...],
    all_milestones: bool,
) -> dict[str, Path]:
    if all_milestones:
        return checkpoint_paths
    final_pass = milestones[-1]
    return {
        name: path
        for name, path in checkpoint_paths.items()
        if name == "S86-masked-anchor-source" or name.endswith(f"@pass-{final_pass}")
    }


def evaluate_musdb(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    checkpoint_paths: dict[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    manifest = json.loads(args.musdb_manifest.resolve().read_text(encoding="utf-8"))
    entries = sorted(
        [entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}],
        key=lambda item: (item["role"], item["member"]),
    )
    eval_args = argparse.Namespace(
        baseline_report=args.baseline_report.resolve(),
        output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        oracle_root=args.oracle_root.resolve(),
        inference_batch_size=args.inference_batch_size,
    )
    return local.evaluate_trained_models(eval_args, contract, checkpoint_paths, entries, device)


def render_listening(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    checkpoint_paths: dict[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    listening_args = argparse.Namespace(
        samples_root=args.samples_root.resolve(),
        output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        force=args.force_listening,
        inference_batch_size=args.inference_batch_size,
    )
    return local.render_listening(listening_args, contract, checkpoint_paths, device)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_milestones(args.milestones, args.passes)
    if args.passes <= 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid passes or batch size")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0:
        raise ValueError("invalid learning rate, anchor beta, or thread count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for path in (
        args.source_checkpoint,
        args.h50_source,
        args.checkpoint,
        args.r_pool,
        args.r_report,
        args.r_manifest,
        args.musdb_cache_root / "selection.json",
        args.musdb_manifest,
        args.oracle_root,
        args.baseline_report,
        args.samples_root,
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path)
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    paths, s_records, r_records, selection = prepare_all_caches(args, contract, device)
    schedules, schedule_summary = build_schedules(paths, s_records, r_records, args.passes, args.seed)
    json_write(args.output_root / "schedule.json", schedule_summary)
    if args.smoke_only:
        smoke = {
            arm: train_arm(
                arm,
                paths,
                schedules[arm],
                args,
                device,
                milestones,
                {"schema": SCHEMA, "selectionSha256": selection["selectionSha256"], "scheduleSha256": schedule_summary["schedules"][arm]["sha256"]},
                args.smoke_updates,
            )
            for arm in ARMS
        }
        report = {
            "schema": SCHEMA,
            "status": "smoke-completed",
            "selection": selection,
            "schedule": schedule_summary,
            "smoke": smoke,
            "environment": {
                "device": str(device),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        report_path = args.output_root / "reports" / "smoke-report.json"
        json_write(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False, indent=2), flush=True)
        return 0

    common_contract = {
        "schema": SCHEMA,
        "selectionSha256": selection["selectionSha256"],
        "scheduleSummarySha256": canonical_sha256(schedule_summary),
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "h50Source": checkpoint_metadata(args.h50_source.resolve()),
        "architecture": checkpoint_metadata(args.checkpoint.resolve()),
        "passes": args.passes,
        "milestones": list(milestones),
        "recordsPerPass": RECORDS_PER_PASS,
        "musdbRecordsPerPass": MUSDB_RECORDS_PER_PASS,
        "externalRecordsPerPass": EXTERNAL_RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixture - Inst3 instrumental",
        "anchorSemantic": "S86-masked-anchor pass-5 residual output",
        "eventCoreMs": CORE_MS,
        "eventGuardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "officialFinalTestUsed": False,
    }
    training: dict[str, Any] = {}
    for arm in ARMS:
        training[arm] = train_arm(
            arm,
            paths,
            schedules[arm],
            args,
            device,
            milestones,
            common_contract,
        )
    checkpoint_paths = build_checkpoint_paths(args, training, milestones)
    report_checkpoint_paths = select_report_checkpoints(checkpoint_paths, milestones, args.all_milestones)
    evaluation = None
    if not args.skip_evaluation:
        evaluation = evaluate_musdb(args, contract, report_checkpoint_paths, device)
    listening_report = None
    if not args.skip_listening:
        listening_report = render_listening(args, contract, report_checkpoint_paths, device)
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": selection,
        "schedule": schedule_summary,
        "training": training,
        "checkpoints": {name: checkpoint_metadata(path) for name, path in checkpoint_paths.items()},
        "evaluation": evaluation,
        "listening": listening_report,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    report_path = args.output_root / "reports" / "s-r-continuation-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "checkpoints": report["checkpoints"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
