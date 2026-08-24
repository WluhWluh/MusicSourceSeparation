#!/usr/bin/env python3
"""Train a balanced S-event pilot from two reviewed FMA batches.

The control and treatment use the same MUSDB records and update budget.  The
treatment replaces the 64-record MUSDB extra sample with a combined external
pool: 24 records/pass from the previously used S pool and 40 records/pass
from the current two-pass stable-S pool.  Both pools keep the C1 residual
target and the H50 non-event anchor unchanged.
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
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as external
import run_inst3_distill_pilot as pilot
import run_inst3_mtg_fma_c1 as c1
import run_inst3_vr_continuous_topk_local as local
import run_inst3_modern_s_pilot as previous_s
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS_ROOT = ROOT / "data" / "modern-song-s-pilot"
DEFAULT_CURRENT_POOL = ROOT / "data" / "modern-song-batch2-full-event-listening" / "two-pass-event-pools.csv"
DEFAULT_CURRENT_REPORT = ROOT / "data" / "modern-song-batch2-full-event-listening" / "event-listening-report.json"
DEFAULT_CURRENT_MANIFEST = ROOT / "data" / "modern-song-original-candidates-batch2" / "source-manifest.json"
DEFAULT_CURRENT_FULL_ROOT = ROOT / "data" / "modern-song-batch2-inst3-event-listening"
DEFAULT_MUSDB_CACHE = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local"
DEFAULT_MUSDB_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_SOURCE = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_BASELINE_REPORT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "continuous-baseline-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-s-combined-pilot"

SAMPLE_RATE = 44_100
SOURCE_STEP = 1600
PASSES = 5
MUSDB_RECORDS_PER_PASS = 640
EXTERNAL_RECORDS_PER_PASS = 64
RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS + EXTERNAL_RECORDS_PER_PASS
PREVIOUS_EXTERNAL_PER_PASS = 24
CURRENT_EXTERNAL_PER_PASS = EXTERNAL_RECORDS_PER_PASS - PREVIOUS_EXTERNAL_PER_PASS
CORE_MS = 100
GUARD_MS = 25
ARMS = ("C0-combined-control", "C1-combined-stable-S")
CHECKPOINT_FORMAT = "local-inst3-modern-s-combined-pilot-checkpoint@1"
SCHEMA = "local-inst3-modern-s-combined-pilot@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, default=DEFAULT_PREVIOUS_ROOT)
    parser.add_argument("--current-pool", type=Path, default=DEFAULT_CURRENT_POOL)
    parser.add_argument("--current-report", type=Path, default=DEFAULT_CURRENT_REPORT)
    parser.add_argument("--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST)
    parser.add_argument("--current-full-root", type=Path, default=DEFAULT_CURRENT_FULL_ROOT)
    parser.add_argument("--musdb-cache-root", type=Path, default=DEFAULT_MUSDB_CACHE)
    parser.add_argument("--musdb-manifest", type=Path, default=DEFAULT_MUSDB_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--milestones", default="1,3,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-prep", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-musdb-evaluation", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=4)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def configure_c1() -> None:
    c1.ARMS = ARMS
    c1.CHECKPOINT_FORMAT = CHECKPOINT_FORMAT
    c1.RECORDS_PER_PASS = RECORDS_PER_PASS
    c1.MUSDB_RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS
    c1.EXTRA_RECORDS_PER_PASS = EXTERNAL_RECORDS_PER_PASS
    c1.SOURCE_STEP = SOURCE_STEP


def validate_array_cache(path: Path, expected_records: int | None = None) -> dict[str, list[int]]:
    with np.load(path) as values:
        required = ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
        if any(name not in values for name in required):
            raise ValueError(f"Missing arrays in {path}")
        shapes = {name: list(values[name].shape) for name in required}
        if any(not np.isfinite(values[name]).all() for name in required):
            raise ValueError(f"Non-finite array in {path}")
        records = int(values["inputSpec"].shape[0])
        if expected_records is not None and records != expected_records:
            raise ValueError(f"Expected {expected_records} records in {path}, got {records}")
        if any(int(values[name].shape[0]) != records for name in required):
            raise ValueError(f"Record dimension mismatch in {path}")
    return shapes


def load_previous_pool(root: Path, source_checkpoint: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    selection_path = root.resolve() / "external-selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for property_item in selection.get("songs", {}).values():
        slug = property_item["slug"]
        contract = property_item.get("contract", {})
        if contract.get("sourceCheckpointSha256") != sha256_file(source_checkpoint.resolve()):
            raise ValueError(f"Previous S cache uses a different H50 checkpoint: {slug}")
        contract_info = contract.get("contract", {})
        if contract_info.get("numFrames") != 128 or contract_info.get("assembly") != "continuous-context-overlap-save":
            raise ValueError(f"Previous S cache contract mismatch: {slug}")
        path = root.resolve() / "cache" / "external" / f"{slug}.npz"
        validate_array_cache(path, expected_records=int(property_item.get("recordCount", 2)))
        key = f"external::previous::{slug}"
        paths[key] = path
        for index, event in enumerate(property_item.get("selectedEvents", [])):
            records.append(
                {
                    "key": key,
                    "index": index,
                    "poolSource": "previous",
                    "slug": slug,
                    "eventId": event.get("eventId", ""),
                    "centerSamples": int(event.get("centerSamples", 0)),
                    "category": property_item.get("category", ""),
                    "artistName": property_item.get("artistName", ""),
                    "trackName": property_item.get("trackName", property_item.get("slug", "")),
                }
            )
    if len(paths) != 16 or len(records) != 32:
        raise ValueError(f"Expected 16 previous songs and 32 records, got {len(paths)} / {len(records)}")
    return paths, records


def load_manifest_records(path: Path) -> dict[int, dict[str, Any]]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise ValueError(f"Incomplete source manifest: {path}")
    return {int(record["order"]): record for record in value.get("records", [])}


def load_current_rows(pool_path: Path, report_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    with pool_path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if row.get("pool") != "stable-S" or row.get("splitRole") != "train":
            continue
        if row.get("firstMark") != "S" or row.get("secondMark") != "S":
            raise ValueError(f"Non-S row in stable-S pool: {row.get('secondEventId')}")
        selected.append(row)
    report = json.loads(report_path.resolve().read_text(encoding="utf-8"))
    events = {str(event["eventId"]): event for event in report.get("events", [])}
    if len(selected) != 54:
        raise ValueError(f"Expected 54 current train stable-S rows, got {len(selected)}")
    return selected, events


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in ".-_" else "-" for character in value).strip("-") or "song"


def prepare_current_pool(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    source_checkpoint: Path,
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    rows, events = load_current_rows(args.current_pool, args.current_report)
    manifest = load_manifest_records(args.current_manifest)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event = events.get(row["secondEventId"])
        if event is None:
            raise ValueError(f"Missing current event {row['secondEventId']}")
        order = int(row["sourceOrder"])
        grouped[(order, str(event["slug"]))].append({"row": row, "event": event})
    output_root = args.output_root.resolve() / "cache" / "external" / "current"
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    song_metadata: dict[str, Any] = {}
    for (order, slug), items in sorted(grouped.items()):
        record = manifest.get(order)
        if record is None or record.get("slug") != slug:
            raise ValueError(f"Manifest mismatch for current event group {order}/{slug}")
        source_path = Path(record["download"]["file"]).resolve()
        h50_path = args.current_full_root.resolve() / "full-song" / "h50ContinuationPlus5" / f"{order:03d}-{slug}.flac"
        teacher_path = args.current_full_root.resolve() / "full-song" / "inst3Instrumental" / f"{order:03d}-{slug}.flac"
        for path in (source_path, h50_path, teacher_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        key = f"external::current::{order:03d}-{safe_slug(slug)}"
        output_path = output_root / f"{order:03d}-{safe_slug(slug)}.npz"
        metadata_path = output_path.with_suffix(".json")
        contract_payload = {
            "schema": "local-inst3-modern-s-combined-current-cache@1",
            "sourceCheckpointSha256": sha256_file(source_checkpoint.resolve()),
            "rawSha256": record["download"]["sha256"],
            "h50Sha256": sha256_file(h50_path),
            "teacherSha256": sha256_file(teacher_path),
            "eventIds": [item["event"]["eventId"] for item in items],
            "coreMs": CORE_MS,
            "guardMs": GUARD_MS,
            "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
        }
        contract_id = canonical_sha256(contract_payload)
        reused = False
        if not args.force_prep and output_path.is_file() and metadata_path.is_file():
            try:
                old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                validate_array_cache(output_path, expected_records=len(items))
                reused = old_metadata.get("contractId") == contract_id
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                reused = False
        if not reused:
            source = external.load_audio(source_path)
            h50 = external.load_audio(h50_path)
            teacher = external.load_audio(teacher_path)
            if source.shape != h50.shape or source.shape != teacher.shape:
                raise ValueError(f"Current source/H50/teacher shape mismatch for {slug}")
            inputs, targets, anchors, masks = [], [], [], []
            event_metadata = []
            for item in sorted(items, key=lambda item: int(item["event"]["centerSamples"])):
                event = item["event"]
                center = int(event["centerSamples"])
                start = center - contract.useful_samples // 2
                end = start + contract.useful_samples
                if start < 0 or end > source.shape[0]:
                    raise ValueError(f"Event context out of bounds: {event['eventId']}")
                window = local.assemble_input(source, start, contract.useful_samples, contract, mode="continuous")
                inputs.append(local.stft_centered(window, contract)[0])
                targets.append(np.ascontiguousarray(source[start:end] - teacher[start:end], dtype=np.float32))
                anchors.append(np.ascontiguousarray(source[start:end] - h50[start:end], dtype=np.float32))
                masks.append(c1.event_mask(contract.useful_samples, center - start, CORE_MS, GUARD_MS))
                event_metadata.append(
                    {
                        "eventId": event["eventId"],
                        "centerSamples": center,
                        "startSamples": start,
                        "eventScoreDbfs": float(event["eventScoreDbfs"]),
                        "category": item["row"]["category"],
                        "artistName": item["row"]["artistName"],
                        "trackName": item["row"]["trackName"],
                    }
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(output_path.name + ".tmp.npz")
            np.savez_compressed(
                temporary,
                inputSpec=np.stack(inputs).astype(np.float32),
                targetAudio=np.stack(targets).astype(np.float32),
                anchorAudio=np.stack(anchors).astype(np.float32),
                eventMask=np.stack(masks).astype(np.float32),
            )
            temporary.replace(output_path)
            metadata = {
                "schema": contract_payload["schema"],
                "contractId": contract_id,
                "contract": contract_payload,
                "poolSource": "current-stable-S",
                "key": key,
                "slug": slug,
                "sourceOrder": order,
                "category": record["category"],
                "artistName": record["artistName"],
                "trackName": record["trackName"],
                "recordCount": len(items),
                "selectedEvents": event_metadata,
                "sourceFile": str(source_path),
                "h50File": str(h50_path),
                "teacherFile": str(teacher_path),
                "cache": checkpoint_metadata(output_path),
            }
            json_write(metadata_path, metadata)
            del source, h50, teacher
            gc.collect()
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths[key] = output_path
        for index, event in enumerate(metadata["selectedEvents"]):
            records.append(
                {
                    "key": key,
                    "index": index,
                    "poolSource": "current",
                    "slug": slug,
                    "eventId": event["eventId"],
                    "centerSamples": int(event["centerSamples"]),
                    "category": metadata["category"],
                    "artistName": metadata["artistName"],
                    "trackName": metadata["trackName"],
                }
            )
        song_metadata[key] = metadata
        print(f"current cache {'reuse' if reused else 'write'}: {record['artistName']} - {record['trackName']} ({len(items)} events)", flush=True)
    if len(records) != 54:
        raise AssertionError(f"Current cache record count mismatch: {len(records)}")
    return paths, records, song_metadata


def choose_pool_items(items: list[dict[str, Any]], count: int, rng: np.random.Generator, max_per_song: int = 2) -> list[tuple[str, int]]:
    if count <= 0 or count > len(items):
        raise ValueError(f"Invalid pool sample size {count} for {len(items)} records")
    order = rng.permutation(len(items))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    deferred: list[dict[str, Any]] = []
    for index in order:
        item = items[int(index)]
        song = f"{item['poolSource']}::{item['slug']}"
        if counts[song] < max_per_song:
            selected.append(item)
            counts[song] += 1
            if len(selected) == count:
                break
        else:
            deferred.append(item)
    if len(selected) < count:
        selected.extend(deferred[: count - len(selected)])
    if len(selected) != count:
        raise AssertionError(f"Could not select {count} pool items")
    return [(item["key"], int(item["index"])) for item in selected]


def build_combined_schedules(
    musdb_slugs: list[str],
    records_per_song: int,
    previous_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    passes: int,
    seed: int,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, Any]]:
    base = [(slug, index) for slug in sorted(musdb_slugs) for index in range(records_per_song)]
    if len(base) != MUSDB_RECORDS_PER_PASS:
        raise ValueError(f"Expected {MUSDB_RECORDS_PER_PASS} MUSDB records, got {len(base)}")
    schedules = {arm: [] for arm in ARMS}
    pass_details = []
    for pass_index in range(passes):
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        base_order = [base[int(index)] for index in rng.permutation(len(base))]
        control_extra = [base_order[int(index)] for index in rng.choice(len(base_order), size=EXTERNAL_RECORDS_PER_PASS, replace=False)]
        previous_extra = choose_pool_items(previous_records, PREVIOUS_EXTERNAL_PER_PASS, rng, max_per_song=2)
        current_extra = choose_pool_items(current_records, CURRENT_EXTERNAL_PER_PASS, rng, max_per_song=2)
        treatment_extra = previous_extra + current_extra
        for arm, extra in ((ARMS[0], control_extra), (ARMS[1], treatment_extra)):
            values = base_order + extra
            arm_rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97)
            schedules[arm].extend(values[int(index)] for index in arm_rng.permutation(len(values)))
        pass_details.append(
            {
                "pass": pass_index + 1,
                "controlMusdbExtra": len(control_extra),
                "treatmentPreviousExternal": len(previous_extra),
                "treatmentCurrentExternal": len(current_extra),
                "previousEventIds": [item["eventId"] for item in previous_records if (item["key"], item["index"]) in set(previous_extra)],
                "currentEventIds": [item["eventId"] for item in current_records if (item["key"], item["index"]) in set(current_extra)],
            }
        )
    return schedules, {"passes": pass_details}


def schedule_summary(schedule: list[tuple[str, int]]) -> dict[str, Any]:
    external = [(key, index) for key, index in schedule if key.startswith("external::")]
    return {
        "recordCount": len(schedule),
        "sha256": canonical_sha256(schedule),
        "sourceCounts": dict(Counter("external" if key.startswith("external::") else "musdb" for key, _ in schedule)),
        "externalUniqueRecords": len(set(external)),
        "externalBySource": dict(Counter("previous" if "::previous::" in key else "current" for key, _ in external)),
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_c1()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.passes <= 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid passes or batch size")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0:
        raise ValueError("invalid optimizer/runtime arguments")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    required = (
        args.previous_root / "external-selection.json",
        args.current_pool,
        args.current_report,
        args.current_manifest,
        args.musdb_cache_root / "selection.json",
        args.musdb_manifest,
        args.source_checkpoint,
        args.checkpoint,
        args.baseline_report,
    )
    for path in required:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    previous_paths, previous_records = load_previous_pool(args.previous_root, args.source_checkpoint)
    current_paths, current_records, current_metadata = prepare_current_pool(args, contract, args.source_checkpoint)
    external_paths = {**previous_paths, **current_paths}
    musdb_paths, records_per_song, musdb_selection = c1.musdb_cache_paths(args.musdb_cache_root)
    schedules, schedule_details = build_combined_schedules(
        list(musdb_paths), records_per_song, previous_records, current_records, args.passes, args.seed
    )
    all_paths = {**musdb_paths, **external_paths}
    external_selection = {
        "schema": "local-inst3-modern-s-combined-selection@1",
        "previousRecordCount": len(previous_records),
        "currentRecordCount": len(current_records),
        "previousSongCount": len(previous_paths),
        "currentSongCount": len(current_paths),
        "currentSongs": current_metadata,
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
    }
    external_selection["selectionSha256"] = canonical_sha256(external_selection)
    json_write(args.output_root / "external-selection.json", external_selection)
    common_contract = {
        "schema": SCHEMA,
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "musdbSelectionSha256": canonical_sha256(musdb_selection),
        "externalSelectionSha256": external_selection["selectionSha256"],
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "musdbRecordsPerPass": MUSDB_RECORDS_PER_PASS,
        "externalRecordsPerPass": EXTERNAL_RECORDS_PER_PASS,
        "previousExternalPerPass": PREVIOUS_EXTERNAL_PER_PASS,
        "currentExternalPerPass": CURRENT_EXTERNAL_PER_PASS,
        "externalFraction": EXTERNAL_RECORDS_PER_PASS / RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "officialFinalTestUsed": False,
    }
    milestones = c1.parse_milestones(args.milestones, args.passes)
    if args.smoke_only:
        smoke = {arm: c1.train_arm(arm, all_paths, schedules[arm], args, device, milestones, common_contract, args.smoke_updates) for arm in ARMS}
        report = {"schema": SCHEMA, "status": "smoke-completed", "contract": common_contract, "schedules": {arm: schedule_summary(schedules[arm]) for arm in ARMS}, "scheduleDetails": schedule_details, "smoke": smoke}
        json_write(args.output_root / "reports" / "combined-smoke.json", report)
        print(json.dumps({"status": report["status"], "arms": list(smoke)}, indent=2))
        return 0
    training = {arm: c1.train_arm(arm, all_paths, schedules[arm], args, device, milestones, common_contract) for arm in ARMS}
    final_paths = {arm: Path(training[arm]["milestoneCheckpoints"][str(args.passes)]["file"]) for arm in ARMS}
    checkpoint_paths = {"Source-H50-continuation+5": args.source_checkpoint.resolve(), **final_paths}
    musdb_evaluation = None
    if not args.skip_musdb_evaluation:
        manifest = json.loads(args.musdb_manifest.resolve().read_text(encoding="utf-8"))
        entries = sorted([entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}], key=lambda item: (item["role"], item["member"]))
        eval_args = argparse.Namespace(baseline_report=args.baseline_report.resolve(), output_root=args.output_root, checkpoint=args.checkpoint.resolve(), oracle_root=args.oracle_root.resolve(), inference_batch_size=args.inference_batch_size)
        musdb_evaluation = local.evaluate_trained_models(eval_args, contract, checkpoint_paths, entries, device)
        json_write(args.output_root / "reports" / "musdb-evaluation.json", musdb_evaluation)
    listening_report = None
    private_analysis = None
    if not args.skip_listening:
        listening_args = argparse.Namespace(samples_root=ROOT / "data" / "samples", output_root=args.output_root, checkpoint=args.checkpoint.resolve(), force=args.force_listening, inference_batch_size=args.inference_batch_size)
        listening_report = local.render_listening(listening_args, contract, checkpoint_paths, device)
        private_analysis = c1.analyze_private(listening_report, args.baseline_report)
        json_write(args.output_root / "reports" / "private-inst3-analysis.json", private_analysis)
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "selection": {"previousRecordCount": len(previous_records), "currentRecordCount": len(current_records), "previousSongCount": len(previous_paths), "currentSongCount": len(current_paths), "previousPoolRoot": str(args.previous_root.resolve()), "currentPool": str(args.current_pool.resolve())},
        "scheduleDetails": schedule_details,
        "schedules": {arm: schedule_summary(schedules[arm]) for arm in ARMS},
        "training": training,
        "checkpoints": {name: c1.checkpoint_metadata(path) for name, path in checkpoint_paths.items()},
        "musdbEvaluation": musdb_evaluation,
        "listening": listening_report,
        "privateInst3Analysis": private_analysis,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchCuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "runner": checkpoint_metadata(Path(__file__).resolve())},
        "licenseDisposition": {"scope": "local non-commercial research only", "originals": "source audio and derived caches remain local", "weights": "teacher-derived checkpoints remain local pending rights review"},
    }
    report_path = args.output_root / "reports" / "combined-s-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "checkpoints": {name: str(path) for name, path in final_paths.items()}, "listening": 0 if listening_report is None else listening_report.get("outputCount", 0)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
