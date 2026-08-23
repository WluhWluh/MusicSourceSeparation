#!/usr/bin/env python3
"""Run an equal-budget MUSDB control and reviewed-R MTG/FMA C1 pilot.

Both arms start from H50-continuation+5 and restore its AdamW state.  Every
pass contains the same 640 MUSDB continuous records plus 64 equal-budget
records.  C0 repeats deterministic MUSDB records; C1 repeats 16 human-reviewed
R events (four events from each of four external songs).  External targets use
Inst 3 only on a 100 ms core with a 25 ms soft guard and anchor the remaining
useful span to the source H50 output.
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
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch

import analyze_inst3_private_language_quality as quality
import evaluate_inst3_continuous_baseline as continuous
import render_inst3_mtg_fma_event_listening as external
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_vr_continuation as continuation
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "musdb18-inst3-vr-continuation" / "runs" / "H50-continuation-plus5" / "step-1600.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_MUSDB_CACHE = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local"
DEFAULT_REVIEW_CSV = ROOT / "data" / "inst3-mtg-fma-event-listening" / "human-review-template.csv"
DEFAULT_EVENT_REPORT = ROOT / "data" / "inst3-mtg-fma-event-listening" / "mtg-fma-event-listening-report.json"
DEFAULT_EXPANSION_MANIFEST = ROOT / "data" / "inst3-mtg-fma-expansion" / "source-manifest.json"
DEFAULT_SUPPLEMENT_MANIFEST = ROOT / "data" / "inst3-mtg-fma-supplement" / "source-manifest.json"
DEFAULT_EXPANSION_OUTPUT = ROOT / "data" / "inst3-mtg-fma-expansion-evaluation" / "all-candidates"
DEFAULT_EVENT_ROOT = ROOT / "data" / "inst3-mtg-fma-event-listening"
DEFAULT_MUSDB_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_BASELINE_REPORT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "continuous-baseline-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "inst3-mtg-fma-c1"

SAMPLE_RATE = 44_100
SCHEMA = "local-inst3-mtg-fma-c1@1"
CHECKPOINT_FORMAT = "local-inst3-mtg-fma-c1-checkpoint@1"
ARMS = ("C0-musdb-control", "C1-reviewed-R")
SOURCE_STEP = 1600
PASSES = 5
MUSDB_RECORDS_PER_PASS = 640
EXTRA_RECORDS_PER_PASS = 64
RECORDS_PER_PASS = MUSDB_RECORDS_PER_PASS + EXTRA_RECORDS_PER_PASS
EVENTS_PER_EXTERNAL_SONG = 4
EXTERNAL_REPEAT = 4
CORE_MS = 100
GUARD_MS = 25
EXTERNAL_SONGS = (
    "mtg-jamendo-fabrice-collette-c-tait-comme-danser",
    "mtg-jamendo-burnogson-chacun-son-tour",
    "fma-stray-akuma",
    "fma-mr.-mrs.-smith-cold-black-oil",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--musdb-cache-root", type=Path, default=DEFAULT_MUSDB_CACHE)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_EVENT_REPORT)
    parser.add_argument("--expansion-manifest", type=Path, default=DEFAULT_EXPANSION_MANIFEST)
    parser.add_argument("--supplement-manifest", type=Path, default=DEFAULT_SUPPLEMENT_MANIFEST)
    parser.add_argument("--expansion-output", type=Path, default=DEFAULT_EXPANSION_OUTPUT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--musdb-manifest", type=Path, default=DEFAULT_MUSDB_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
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
    parser.add_argument("--skip-external-evaluation", action="store_true")
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
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_milestones(raw: str, passes: int) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or tuple(sorted(set(values))) != values or values[-1] != passes:
        raise ValueError("milestones must be sorted, unique, and end at passes")
    return values


def event_mask(length: int, center: int, core_ms: int = CORE_MS, guard_ms: int = GUARD_MS) -> np.ndarray:
    if length <= 0 or not 0 <= center < length:
        raise ValueError("invalid event mask coordinates")
    core = round(SAMPLE_RATE * core_ms / 1000.0)
    guard = round(SAMPLE_RATE * guard_ms / 1000.0)
    core_start = max(0, center - core // 2)
    core_end = min(length, core_start + core)
    mask = np.zeros(length, dtype=np.float32)
    mask[core_start:core_end] = 1.0
    left = max(0, core_start - guard)
    right = min(length, core_end + guard)
    if core_start > left:
        mask[left:core_start] = np.linspace(0.0, 1.0, core_start - left, endpoint=False, dtype=np.float32)
    if right > core_end:
        mask[core_end:right] = np.linspace(1.0, 0.0, right - core_end, endpoint=False, dtype=np.float32)
    return mask


def load_review_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        mark = (row.get("mark") or "").strip().upper()
        if mark not in {"R", "K", "I"}:
            raise ValueError(f"Invalid review mark {mark!r} for {row.get('eventId')}")
        row["mark"] = mark
        row["eventScoreDbfs"] = float(row["eventScoreDbfs"])
        result[row["eventId"]] = row
    return result


def manifest_records(*paths: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        value = json.loads(path.resolve().read_text(encoding="utf-8"))
        if value.get("status") != "completed":
            raise ValueError(f"Incomplete manifest: {path}")
        for record in value["records"]:
            if record["slug"] in result:
                raise ValueError(f"Duplicate manifest slug: {record['slug']}")
            copy = dict(record)
            copy["manifestPath"] = str(path.resolve())
            copy["manifestKind"] = "expansion" if "expansion" in path.name or "expansion" in str(path.parent.name) else "supplement"
            result[record["slug"]] = copy
    return result


def load_full_h50(record: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    slug = record["slug"]
    if record["manifestKind"] == "expansion":
        path = args.expansion_output.resolve() / f"{slug}-instrumental.flac"
    else:
        path = args.event_root.resolve() / "full-song" / "h50ContinuationPlus5" / f"{slug}-instrumental.flac"
    return quality.load_flac(path)


def select_external_events(
    source_samples: int,
    slug: str,
    review: dict[str, dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
    contract: ShortWindowContract,
) -> list[dict[str, Any]]:
    candidates = []
    margin = contract.useful_samples // 2 + max(contract.left_context_samples, contract.right_context_samples)
    for event_id, row in review.items():
        event = events_by_id.get(event_id)
        if row["mark"] != "R" or event is None or event["slug"] != slug:
            continue
        center = int(event["centerSamples"])
        if margin <= center < source_samples - margin:
            candidates.append({**event, "eventScoreDbfs": float(row["eventScoreDbfs"])})
    candidates.sort(key=lambda item: (-float(item["eventScoreDbfs"]), int(item["centerSamples"])))
    if len(candidates) < EVENTS_PER_EXTERNAL_SONG:
        raise ValueError(f"Only {len(candidates)} eligible R events for {slug}")
    return candidates[:EVENTS_PER_EXTERNAL_SONG]


def prepare_external_caches(args: argparse.Namespace, contract: ShortWindowContract) -> tuple[dict[str, Path], dict[str, Any]]:
    review = load_review_rows(args.review_csv)
    event_report = json.loads(args.event_report.resolve().read_text(encoding="utf-8"))
    events_by_id = {event["eventId"]: event for event in event_report["events"]}
    source_model_sha = sha256_file(args.source_checkpoint.resolve())
    report_model_sha = event_report["model"]["h50ContinuationPlus5"]["sha256"]
    if source_model_sha != report_model_sha:
        raise ValueError("Review H50 checkpoint does not match the C1 source checkpoint")
    records = manifest_records(args.expansion_manifest, args.supplement_manifest)
    root = args.output_root.resolve() / "cache" / "external"
    paths: dict[str, Path] = {}
    selections: dict[str, Any] = {}
    for slug in EXTERNAL_SONGS:
        record = records[slug]
        source = external.load_audio(Path(record["download"]["file"]))
        teacher = external.load_teacher(Path(record["teacher"]["file"]))
        h50 = load_full_h50(record, args)
        if source.shape != teacher.shape or source.shape != h50.shape:
            raise ValueError(f"External source/teacher/H50 shape mismatch for {slug}")
        selected = select_external_events(source.shape[0], slug, review, events_by_id, contract)
        key = f"external::{slug}"
        path = root / f"{slug}.npz"
        metadata_path = root / f"{slug}.json"
        contract_payload = {
            "schema": "local-inst3-mtg-fma-c1-external-cache@1",
            "sourceCheckpointSha256": source_model_sha,
            "architectureSha256": sha256_file(args.checkpoint.resolve()),
            "reviewCsvSha256": sha256_file(args.review_csv.resolve()),
            "eventReportSha256": sha256_file(args.event_report.resolve()),
            "rawSha256": record["download"]["sha256"],
            "teacherSha256": record["teacher"]["sha256"],
            "eventIds": [item["eventId"] for item in selected],
            "coreMs": CORE_MS,
            "guardMs": GUARD_MS,
            "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
        }
        contract_id = canonical_sha256(contract_payload)
        reused = False
        if not args.force_prep and path.is_file() and metadata_path.is_file():
            try:
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                with np.load(path) as values:
                    reused = previous.get("contractId") == contract_id and values["inputSpec"].shape[0] == EVENTS_PER_EXTERNAL_SONG
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                reused = False
        if not reused:
            inputs, targets, anchors, masks, starts = [], [], [], [], []
            event_rows = []
            for event in selected:
                center = int(event["centerSamples"])
                start = center - contract.useful_samples // 2
                end = start + contract.useful_samples
                input_spec = local.stft_centered(local.assemble_input(source, start, contract.useful_samples, contract, mode="continuous"), contract)[0]
                targets.append(np.ascontiguousarray(source[start:end] - teacher[start:end], dtype=np.float32))
                anchors.append(np.ascontiguousarray(source[start:end] - h50[start:end], dtype=np.float32))
                masks.append(event_mask(contract.useful_samples, center - start))
                inputs.append(np.ascontiguousarray(input_spec, dtype=np.float32))
                starts.append(start)
                event_rows.append({"eventId": event["eventId"], "centerSamples": center, "startSamples": start, "eventScoreDbfs": event["eventScoreDbfs"]})
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp.npz")
            np.savez_compressed(
                temporary,
                inputSpec=np.stack(inputs).astype(np.float32),
                targetAudio=np.stack(targets).astype(np.float32),
                anchorAudio=np.stack(anchors).astype(np.float32),
                eventMask=np.stack(masks).astype(np.float32),
            )
            temporary.replace(path)
            metadata = {
                "schema": contract_payload["schema"],
                "contractId": contract_id,
                "contract": contract_payload,
                "slug": slug,
                "trackName": record["trackName"],
                "role": record["role"],
                "recordCount": EVENTS_PER_EXTERNAL_SONG,
                "selectedEvents": event_rows,
                "cache": checkpoint_metadata(path),
            }
            json_write(metadata_path, metadata)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths[key] = path
        selections[key] = metadata
        print(f"external cache {'reuse' if reused else 'write'}: {record['artistName']} - {record['trackName']}", flush=True)
        del source, teacher, h50
        gc.collect()
    payload = {"schema": "local-inst3-mtg-fma-c1-selection@1", "songs": selections, "selectionSha256": canonical_sha256(selections)}
    json_write(args.output_root.resolve() / "external-selection.json", payload)
    return paths, payload


def musdb_cache_paths(root: Path) -> tuple[dict[str, Path], int, dict[str, Any]]:
    selection_path = root.resolve() / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records_per_song = int(selection["contract"]["recordsPerSong"])
    paths = {slug: root.resolve() / "cache" / "train" / f"{slug}.npz" for slug in sorted(selection["songs"])}
    if len(paths) * records_per_song != MUSDB_RECORDS_PER_PASS or any(not path.is_file() for path in paths.values()):
        raise ValueError("Expected complete 80 x 8 MUSDB continuous cache")
    return paths, records_per_song, selection


def build_schedules(
    musdb_slugs: Sequence[str],
    records_per_song: int,
    external_keys: Sequence[str],
    passes: int,
    seed: int,
) -> dict[str, list[tuple[str, int]]]:
    base = [(slug, index) for slug in sorted(musdb_slugs) for index in range(records_per_song)]
    external_items = [(key, index) for key in sorted(external_keys) for index in range(EVENTS_PER_EXTERNAL_SONG)]
    if len(base) != MUSDB_RECORDS_PER_PASS or len(external_items) * EXTERNAL_REPEAT != EXTRA_RECORDS_PER_PASS:
        raise ValueError("Unexpected schedule source count")
    result = {arm: [] for arm in ARMS}
    for pass_index in range(passes):
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        base_order = [base[int(index)] for index in rng.permutation(len(base))]
        c0_extra = [base_order[int(index)] for index in rng.choice(len(base_order), size=EXTRA_RECORDS_PER_PASS, replace=False)]
        c1_extra = external_items * EXTERNAL_REPEAT
        for arm, extra in ((ARMS[0], c0_extra), (ARMS[1], c1_extra)):
            items = base_order + list(extra)
            arm_rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97)
            result[arm].extend(items[int(index)] for index in arm_rng.permutation(len(items)))
    return result


def load_source(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    payload = torch.load(args.source_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-vr-continuation-checkpoint@1" or int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError("C1 requires H50-continuation+5 step 1600")
    model, architecture = pilot.make_model(args.checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    continuation.move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    return model, optimizer, {"checkpoint": checkpoint_metadata(args.source_checkpoint.resolve()), "architecture": architecture["checkpoint"], "optimizerRestored": True}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    arm: str,
    global_step: int,
    local_step: int,
    contract_id: str,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
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
        "seed": args.seed,
        "stateDict": local.hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": local.hard.cpu_tree(optimizer.state_dict()),
        "history": history,
        "source": source,
    }
    return local.hard.atomic_torch_save(path, payload)


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
    model, optimizer, source = load_source(args, device)
    used_keys = sorted({key for key, _index in schedule})
    used_paths = {key: paths[key] for key in used_keys}
    store = local.CacheStore(used_paths, max_open=len(used_paths))
    store.preload()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    updates_per_pass = RECORDS_PER_PASS // args.batch_size
    total_updates = args.passes * updates_per_pass
    if smoke_updates is not None:
        total_updates = min(smoke_updates, total_updates)
    contract = {**common_contract, "arm": arm, "scheduleSha256": canonical_sha256(schedule)}
    contract_id = canonical_sha256(contract)
    run_root = args.output_root.resolve() / "runs" / arm
    run_root.mkdir(parents=True, exist_ok=True)
    current = 0
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    if args.resume and smoke_updates is None:
        candidates = []
        for path in run_root.glob("step-*.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("runContractId") == contract_id:
                    candidates.append((int(payload["localStep"]), path, payload))
            except (OSError, KeyError, ValueError, RuntimeError, TypeError):
                pass
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
        for local_step, pass_count in milestone_steps.items():
            path = run_root / f"step-{SOURCE_STEP + local_step}.pt"
            if path.is_file():
                checkpoints[str(pass_count)] = checkpoint_metadata(path)
    started = time.perf_counter()
    while current < total_updates:
        items = schedule[current * args.batch_size : (current + 1) * args.batch_size]
        input_array, target_array, anchor_array, mask_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        anchor_tensor = torch.from_numpy(anchor_array).to(device)
        mask_tensor = torch.from_numpy(mask_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
        record_losses, full_losses, event_losses, anchor_losses = [], [], [], []
        for index, (key, _cache_index) in enumerate(items):
            if key.startswith("external::"):
                event_loss = local.charbonnier_per_record(predicted[index : index + 1], target_tensor[index : index + 1], mask_tensor[index : index + 1]).mean()
                anchor_loss = local.charbonnier_per_record(predicted[index : index + 1], anchor_tensor[index : index + 1], 1.0 - mask_tensor[index : index + 1]).mean()
                record_losses.append(event_loss + args.anchor_beta * anchor_loss)
                event_losses.append(event_loss)
                anchor_losses.append(anchor_loss)
            else:
                full_loss = local.charbonnier_per_record(predicted[index : index + 1], target_tensor[index : index + 1], torch.ones_like(mask_tensor[index : index + 1])).mean()
                record_losses.append(full_loss)
                full_losses.append(full_loss)
        loss = torch.stack(record_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {arm} loss")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        if not math.isfinite(gradient):
            raise FloatingPointError(f"Non-finite {arm} gradient")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current += 1
        history.append({
            "localStep": current,
            "globalStep": SOURCE_STEP + current,
            "pass": current / updates_per_pass,
            "loss": float(loss.detach().cpu()),
            "musdbLoss": None if not full_losses else float(torch.stack(full_losses).mean().detach().cpu()),
            "externalEventLoss": None if not event_losses else float(torch.stack(event_losses).mean().detach().cpu()),
            "externalAnchorLoss": None if not anchor_losses else float(torch.stack(anchor_losses).mean().detach().cpu()),
            "externalRecords": sum(key.startswith("external::") for key, _ in items),
            "gradientNormBeforeClip": gradient,
        })
        if current == 1 or current % 100 == 0:
            print(json.dumps({"event": "progress", "arm": arm, "update": current, "total": total_updates, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        if smoke_updates is None and current in milestone_steps:
            pass_count = milestone_steps[current]
            metadata = save_checkpoint(run_root / f"step-{SOURCE_STEP + current}.pt", model, optimizer, arm, SOURCE_STEP + current, current, contract_id, args, history, source)
            checkpoints[str(pass_count)] = metadata
            print(json.dumps({"event": "milestone", "arm": arm, "pass": pass_count, "step": SOURCE_STEP + current}, sort_keys=True), flush=True)
        elif smoke_updates is None and current % args.state_interval == 0:
            save_checkpoint(run_root / f"step-{SOURCE_STEP + current}.pt", model, optimizer, arm, SOURCE_STEP + current, current, contract_id, args, history, source)
    result = {
        "status": "smoke-completed" if smoke_updates is not None else "completed",
        "arm": arm,
        "updates": total_updates,
        "updatesPerPass": updates_per_pass,
        "recordsPerPass": RECORDS_PER_PASS,
        "externalRecordsPerPass": 0 if arm == ARMS[0] else EXTRA_RECORDS_PER_PASS,
        "externalFraction": 0.0 if arm == ARMS[0] else EXTRA_RECORDS_PER_PASS / RECORDS_PER_PASS,
        "elapsedSeconds": time.perf_counter() - started,
        "history": history,
        "milestoneCheckpoints": checkpoints,
        "contract": {"id": contract_id, "payload": contract},
    }
    del model, optimizer, store, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def rms(value: np.ndarray) -> float:
    return math.sqrt(float(np.mean(value.astype(np.float64) ** 2)))


def db_ratio(value: float, reference: float) -> float:
    return 20.0 * math.log10(max(value, 1.0e-30) / max(reference, 1.0e-30))


def movement_coefficient(base: np.ndarray, target: np.ndarray, candidate: np.ndarray) -> float:
    desired = target.astype(np.float64) - base.astype(np.float64)
    movement = candidate.astype(np.float64) - base.astype(np.float64)
    return float(np.sum(movement * desired) / max(float(np.sum(desired * desired)), 1.0e-30))


def analyze_private(listening_report: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.resolve().read_text(encoding="utf-8"))
    variants = ("Source-H50-continuation+5", *ARMS)
    rows = []
    for song, info in listening_report["songs"].items():
        source = external.load_audio(Path(info["source"]["file"]))
        reference = quality.load_flac(Path(baseline["privateListening"]["songs"][song]["candidates"]["Inst3-native"]["instrumental"]["file"]))
        for variant in variants:
            candidate = quality.load_flac(Path(info["variants"][variant]["instrumental"]["file"]))
            rows.append({"song": song, "variant": variant, **quality.metric_row(source, reference, candidate), "shortEvents": {str(ms): quality.short_event_row(source, reference, candidate, ms) for ms in quality.EVENT_MS}})
    by_variant = {}
    for variant in variants:
        values = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "songCount": len(values),
            "mean": {key: statistics.fmean(row[key] for row in values) for key in ("teacherMatchSnrDb", "relativeErrorToRemovedDb", "coherentRetainedRelativeDb", "targetErrorRmsDbfs")},
            "shortEvents": {str(ms): {key: statistics.fmean(row["shortEvents"][str(ms)][key] for row in values) for key in ("missRmsP95Dbfs", "missRmsMaxDbfs", "positiveProjectionP95Dbfs", "positiveProjectionMaxDbfs")} for ms in quality.EVENT_MS},
        }
    deltas = {}
    for variant in ARMS:
        deltas[variant] = {
            "vsSource": {
                "perSong": {
                    song: {
                        key: next(row[key] for row in rows if row["song"] == song and row["variant"] == variant) - next(row[key] for row in rows if row["song"] == song and row["variant"] == "Source-H50-continuation+5")
                        for key in ("teacherMatchSnrDb", "relativeErrorToRemovedDb", "coherentRetainedRelativeDb")
                    }
                    for song in listening_report["songs"]
                }
            }
        }
    return {"rows": rows, "aggregate": by_variant, "deltas": deltas}


def analyze_review_events(
    args: argparse.Namespace,
    checkpoint_paths: dict[str, Path],
    contract: ShortWindowContract,
    device: torch.device,
) -> dict[str, Any]:
    review = load_review_rows(args.review_csv)
    event_report = json.loads(args.event_report.resolve().read_text(encoding="utf-8"))
    events_by_slug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in event_report["events"]:
        copy = dict(event)
        copy["mark"] = review[event["eventId"]]["mark"]
        events_by_slug[event["slug"]].append(copy)
    records = manifest_records(args.expansion_manifest, args.supplement_manifest)
    timings: dict[str, dict[str, Any]] = {variant: {} for variant in checkpoint_paths}
    models = {
        variant: local.load_h50_model(args.checkpoint.resolve(), path.resolve(), device)[0]
        for variant, path in checkpoint_paths.items()
    }
    rows = []
    try:
        for song_index, (slug, events) in enumerate(sorted(events_by_slug.items()), start=1):
            record = records[slug]
            source = external.load_audio(Path(record["download"]["file"]))
            teacher = external.load_teacher(Path(record["teacher"]["file"]))
            base = load_full_h50(record, args)
            candidates = {}
            for variant, model in models.items():
                residual, timing = continuous.render_student(model, source, contract, device, args.inference_batch_size)
                candidates[variant] = np.ascontiguousarray(source - residual, dtype=np.float32)
                timings[variant][slug] = timing
            for event in events:
                center = int(event["centerSamples"])
                start, end = center - SAMPLE_RATE, center + SAMPLE_RATE
                base_block, target_block = base[start:end], teacher[start:end]
                base_error = rms(base_block - target_block)
                for variant in checkpoint_paths:
                    candidate = candidates[variant][start:end]
                    local_metrics = {}
                    for milliseconds in (50, 100, 200):
                        length = round(SAMPLE_RATE * milliseconds / 1000.0)
                        local_start = center - length // 2
                        local_end = local_start + length
                        local_base = base[local_start:local_end]
                        local_target = teacher[local_start:local_end]
                        local_candidate = candidates[variant][local_start:local_end]
                        local_metrics[str(milliseconds)] = {
                            "targetErrorDeltaDbVsSource": db_ratio(
                                rms(local_candidate - local_target),
                                rms(local_base - local_target),
                            ),
                            "movementTowardInst3Coefficient": movement_coefficient(
                                local_base, local_target, local_candidate
                            ),
                        }
                    rows.append({
                        "eventId": event["eventId"], "slug": slug, "trackName": record["trackName"], "mark": event["mark"], "variant": variant,
                        "targetErrorDeltaDbVsSource": db_ratio(rms(candidate - target_block), base_error),
                        "changeFromSourceRmsDbfs": quality.db(rms(candidate - base_block)),
                        "movementTowardInst3Coefficient": movement_coefficient(base_block, target_block, candidate),
                        "local": local_metrics,
                    })
            print(f"review eval {song_index}/{len(events_by_slug)}: {slug}", flush=True)
            del source, teacher, base, candidates
            gc.collect()
    finally:
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = {}
    for variant in checkpoint_paths:
        aggregate[variant] = {}
        for mark in ("R", "K", "I"):
            values = [row for row in rows if row["variant"] == variant and row["mark"] == mark]
            aggregate[variant][mark] = {
                "count": len(values),
                "meanTargetErrorDeltaDbVsSource": statistics.fmean(row["targetErrorDeltaDbVsSource"] for row in values),
                "medianTargetErrorDeltaDbVsSource": statistics.median(row["targetErrorDeltaDbVsSource"] for row in values),
                "improvedTowardInst3": sum(row["targetErrorDeltaDbVsSource"] < 0.0 for row in values),
                "meanMovementTowardInst3Coefficient": statistics.fmean(row["movementTowardInst3Coefficient"] for row in values),
                "meanChangeFromSourceRmsDbfs": statistics.fmean(row["changeFromSourceRmsDbfs"] for row in values),
                "local": {
                    str(milliseconds): {
                        "meanTargetErrorDeltaDbVsSource": statistics.fmean(
                            row["local"][str(milliseconds)]["targetErrorDeltaDbVsSource"]
                            for row in values
                        ),
                        "medianTargetErrorDeltaDbVsSource": statistics.median(
                            row["local"][str(milliseconds)]["targetErrorDeltaDbVsSource"]
                            for row in values
                        ),
                        "improvedTowardInst3": sum(
                            row["local"][str(milliseconds)]["targetErrorDeltaDbVsSource"] < 0.0
                            for row in values
                        ),
                        "meanMovementTowardInst3Coefficient": statistics.fmean(
                            row["local"][str(milliseconds)]["movementTowardInst3Coefficient"]
                            for row in values
                        ),
                    }
                    for milliseconds in (50, 100, 200)
                },
            }
    return {"rows": rows, "aggregate": aggregate, "renderTimings": timings}


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    milestones = parse_milestones(args.milestones, args.passes)
    if args.passes <= 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid passes or batch size")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("invalid optimizer/runtime arguments")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in (args.source_checkpoint, args.checkpoint, args.review_csv, args.event_report, args.musdb_manifest, args.baseline_report):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    musdb_paths, records_per_song, musdb_selection = musdb_cache_paths(args.musdb_cache_root)
    external_paths, external_selection = prepare_external_caches(args, contract)
    schedules = build_schedules(list(musdb_paths), records_per_song, list(external_paths), args.passes, args.seed)
    all_paths = {**musdb_paths, **external_paths}
    common_contract = {
        "schema": SCHEMA,
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "musdbSelectionSha256": canonical_sha256(musdb_selection),
        "externalSelectionSha256": external_selection["selectionSha256"],
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "musdbRecordsPerPass": MUSDB_RECORDS_PER_PASS,
        "extraRecordsPerPass": EXTRA_RECORDS_PER_PASS,
        "externalFractionC1": EXTRA_RECORDS_PER_PASS / RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "externalCoreMs": CORE_MS,
        "externalGuardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "officialFinalTestUsed": False,
    }
    if args.smoke_only:
        smoke = {arm: train_arm(arm, all_paths, schedules[arm], args, device, milestones, common_contract, args.smoke_updates) for arm in ARMS}
        report = {"schema": SCHEMA, "status": "smoke-completed", "contract": common_contract, "smoke": smoke}
        json_write(args.output_root / "reports" / "c1-smoke.json", report)
        print(json.dumps({"status": report["status"], "arms": list(smoke)}, indent=2))
        return 0
    training = {arm: train_arm(arm, all_paths, schedules[arm], args, device, milestones, common_contract) for arm in ARMS}
    final_paths = {
        arm: Path(training[arm]["milestoneCheckpoints"][str(args.passes)]["file"])
        for arm in ARMS
    }
    checkpoint_paths = {"Source-H50-continuation+5": args.source_checkpoint.resolve(), **final_paths}
    musdb_evaluation = None
    if not args.skip_musdb_evaluation:
        manifest = json.loads(args.musdb_manifest.resolve().read_text(encoding="utf-8"))
        eval_entries = sorted([entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}], key=lambda item: (item["role"], item["member"]))
        eval_args = argparse.Namespace(baseline_report=args.baseline_report.resolve(), output_root=args.output_root, checkpoint=args.checkpoint.resolve(), oracle_root=args.oracle_root.resolve(), inference_batch_size=args.inference_batch_size)
        musdb_evaluation = local.evaluate_trained_models(eval_args, contract, checkpoint_paths, eval_entries, device)
    review_evaluation = None
    if not args.skip_external_evaluation:
        review_evaluation = analyze_review_events(args, checkpoint_paths, contract, device)
        json_write(args.output_root / "reports" / "review-event-evaluation.json", review_evaluation)
    listening_report = None
    private_analysis = None
    if not args.skip_listening:
        listening_args = argparse.Namespace(samples_root=ROOT / "data" / "samples", output_root=args.output_root, checkpoint=args.checkpoint.resolve(), force=args.force_listening, inference_batch_size=args.inference_batch_size)
        listening_report = local.render_listening(listening_args, contract, checkpoint_paths, device)
        private_analysis = analyze_private(listening_report, args.baseline_report)
        json_write(args.output_root / "reports" / "private-inst3-analysis.json", private_analysis)
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "selection": external_selection,
        "schedules": {arm: {"recordCount": len(schedule), "sha256": canonical_sha256(schedule), "sourceCounts": dict(Counter("external" if key.startswith("external::") else "musdb" for key, _ in schedule))} for arm, schedule in schedules.items()},
        "training": training,
        "checkpoints": {name: checkpoint_metadata(path) for name, path in checkpoint_paths.items()},
        "musdbEvaluation": musdb_evaluation,
        "reviewEventEvaluation": review_evaluation,
        "listening": listening_report,
        "privateInst3Analysis": private_analysis,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchCuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "runner": checkpoint_metadata(Path(__file__).resolve())},
        "licenseDisposition": {"scope": "local non-commercial research only", "audio": "MTG/FMA and MUSDB audio/caches are not published", "weights": "teacher-derived checkpoints remain local pending rights review"},
    }
    report_path = args.output_root / "reports" / "c1-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "checkpoints": {key: str(value) for key, value in final_paths.items()}, "listening": 0 if listening_report is None else listening_report.get("outputCount", 0)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
