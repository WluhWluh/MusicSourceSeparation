#!/usr/bin/env python3
"""Run the all-S, no-stabilizer Inst 3 stress test.

The experiment starts both arms from the same H50-continuation+5 checkpoint
and restores its AdamW state.  The two arms consume exactly the same 86
reviewed-S records and schedule:

* ``S86-masked-anchor``: Inst 3 target on the 100 ms core and 25 ms guard,
  H50 target outside the event with beta=0.25.
* ``S86-max-aggressive``: Inst 3 target over the complete useful window and
  no H50 anchor.

All generated audio and checkpoints stay in the local data tree.  Evaluation
uses continuous overlap-save rendering so isolated zero-padded windows do not
create artificial short leaks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch

import analyze_inst3_private_language_quality as quality
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_vr_continuation as continuation
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_ROOT = ROOT / "data" / "modern-song-s-pilot"
DEFAULT_COMBINED_ROOT = ROOT / "data" / "modern-song-s-combined-pilot"
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-continuation"
    / "runs"
    / "H50-continuation-plus5"
    / "step-1600.pt"
)
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_MUSDB_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_BASELINE_REPORT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "continuous-baseline-report.json"
DEFAULT_SAMPLES = ROOT / "data" / "samples"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-s-only-stress"
DEFAULT_EVENT_REPORT = ROOT / "data" / "musdb18-inst3-training-event-listening" / "event-listening-report.json"

SAMPLE_RATE = 44_100
SOURCE_STEP = 1_600
PASSES = 5
S_RECORDS = 86
RECORDS_PER_PASS = 704
REPEAT_FACTOR = 8
BALANCED_EXTRA = 16
BATCH_SIZE = 4
CORE_MS = 100
GUARD_MS = 25
ARMS = ("S86-masked-anchor", "S86-max-aggressive")
SOURCE_VARIANT = "H50-source"
CHECKPOINT_FORMAT = "local-inst3-s-only-stress-checkpoint@1"
SCHEMA = "local-inst3-s-only-stress@1"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--combined-root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--musdb-manifest", type=Path, default=DEFAULT_MUSDB_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--event-report", type=Path, default=DEFAULT_EVENT_REPORT)
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
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-musdb-evaluation", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--skip-event-clips", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_milestones(raw: str, passes: int) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or tuple(sorted(set(values))) != values or values[-1] != passes:
        raise ValueError("milestones must be sorted, unique, and end at passes")
    if values[0] <= 0:
        raise ValueError("stress milestones must be positive pass numbers")
    return values


def validate_cache(path: Path, expected_records: int) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as values:
        required = ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
        if any(name not in values for name in required):
            raise ValueError(f"Missing required arrays in {path}")
        shapes = {name: list(values[name].shape) for name in required}
        if int(values["inputSpec"].shape[0]) != expected_records:
            raise ValueError(
                f"Expected {expected_records} records in {path}, got {values['inputSpec'].shape[0]}"
            )
        if values["inputSpec"].shape[1:] != (4, 1025, 128):
            raise ValueError(f"Unexpected input shape in {path}: {values['inputSpec'].shape}")
        if values["targetAudio"].shape[1:] != (119808, 2):
            raise ValueError(f"Unexpected target shape in {path}: {values['targetAudio'].shape}")
        if values["anchorAudio"].shape != values["targetAudio"].shape:
            raise ValueError(f"Anchor/target shape mismatch in {path}")
        if values["eventMask"].shape != (expected_records, 119808):
            raise ValueError(f"Unexpected event mask shape in {path}: {values['eventMask'].shape}")
        for name in required:
            if not np.isfinite(values[name]).all():
                raise ValueError(f"Non-finite values in {path}:{name}")
        mask_sum = float(np.sum(values["eventMask"]))
    return {"file": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "shapes": shapes, "eventMaskSamples": mask_sum}


def load_pool_selection(
    root: Path,
    source_sha256: str,
    pool_name: str,
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    selection_path = root.resolve() / "external-selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = selection.get("songs") or selection.get("currentSongs")
    if not isinstance(entries, dict):
        raise ValueError(f"No song selection in {selection_path}")
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    songs: dict[str, Any] = {}
    for original_key, metadata in sorted(entries.items()):
        contract = metadata.get("contract", {})
        observed_source = contract.get("sourceCheckpointSha256")
        if observed_source != source_sha256:
            raise ValueError(
                f"{pool_name} cache source mismatch for {original_key}: {observed_source} != {source_sha256}"
            )
        record_count = int(metadata.get("recordCount", 0))
        if record_count <= 0:
            raise ValueError(f"Invalid record count for {original_key}")
        cache_path = Path(metadata["cache"]["file"]).resolve()
        cache_info = validate_cache(cache_path, record_count)
        key = f"external::{pool_name}::{metadata.get('slug', original_key)}"
        if key in paths:
            raise ValueError(f"Duplicate cache key: {key}")
        paths[key] = cache_path
        selected_events = metadata.get("selectedEvents", [])
        if len(selected_events) != record_count:
            raise ValueError(f"Event metadata count mismatch for {original_key}")
        song_info = {
            "pool": pool_name,
            "originalKey": original_key,
            "key": key,
            "slug": metadata.get("slug", original_key),
            "artistName": metadata.get("artistName", ""),
            "trackName": metadata.get("trackName", metadata.get("slug", original_key)),
            "recordCount": record_count,
            "cache": cache_info,
            "sourceSelection": metadata,
        }
        songs[key] = song_info
        for index, event in enumerate(selected_events):
            records.append(
                {
                    "key": key,
                    "index": index,
                    "pool": pool_name,
                    "songKey": key,
                    "slug": song_info["slug"],
                    "artistName": song_info["artistName"],
                    "trackName": song_info["trackName"],
                    "eventId": event.get("eventId", f"{key}-{index}"),
                    "centerSamples": int(event.get("centerSamples", 0)),
                }
            )
    return paths, records, {
        "selectionFile": str(selection_path.resolve()),
        "selectionSha256": sha256_file(selection_path),
        "pool": pool_name,
        "songCount": len(paths),
        "recordCount": len(records),
        "songs": songs,
    }


def load_s_pool(args: argparse.Namespace) -> tuple[dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    source_sha = sha256_file(args.source_checkpoint.resolve())
    old_paths, old_records, old_info = load_pool_selection(args.old_root, source_sha, "previous")
    current_paths, current_records, current_info = load_pool_selection(args.combined_root, source_sha, "current")
    paths = {**old_paths, **current_paths}
    records = old_records + current_records
    event_ids = [record["eventId"] for record in records]
    if len(paths) != 38 or len(records) != S_RECORDS or len(set(event_ids)) != S_RECORDS:
        raise ValueError(
            f"Expected 38 songs and 86 unique S records, got {len(paths)} and {len(records)}"
        )
    info = {
        "schema": "local-inst3-s-only-stress-selection@1",
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "previous": old_info,
        "current": current_info,
        "songCount": len(paths),
        "recordCount": len(records),
        "eventIds": event_ids,
        "contract": ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5).as_dict(
            assembly="continuous-context-overlap-save"
        ),
    }
    info["selectionSha256"] = canonical_sha256(info)
    return paths, records, info


def build_balanced_extras(records: Sequence[dict[str, Any]], pass_index: int) -> list[tuple[str, int]]:
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item["songKey"], item["index"])):
        groups[record["songKey"]].append((record["key"], int(record["index"])))
    songs = sorted(groups)
    if not songs:
        raise ValueError("S pool is empty")
    extras: list[tuple[str, int]] = []
    start = (pass_index * 7) % len(songs)
    for offset in range(BALANCED_EXTRA):
        song = songs[(start + offset) % len(songs)]
        event_list = groups[song]
        extras.append(event_list[(pass_index + offset) % len(event_list)])
    return extras


def build_schedule(
    records: Sequence[dict[str, Any]], passes: int, seed: int
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    base = [(record["key"], int(record["index"])) for record in sorted(records, key=lambda item: (item["key"], item["index"]))]
    if len(base) != S_RECORDS:
        raise ValueError(f"Expected {S_RECORDS} base S records, got {len(base)}")
    result: list[tuple[str, int]] = []
    pass_details: list[dict[str, Any]] = []
    for pass_index in range(passes):
        extras = build_balanced_extras(records, pass_index)
        values = base * REPEAT_FACTOR + extras
        if len(values) != RECORDS_PER_PASS:
            raise AssertionError(f"Schedule pass has {len(values)} records")
        rng = np.random.default_rng(seed + pass_index * 1_000_003 + 97)
        ordered = [values[int(index)] for index in rng.permutation(len(values))]
        result.extend(ordered)
        pass_details.append(
            {
                "pass": pass_index + 1,
                "baseRecords": len(base),
                "repeatFactor": REPEAT_FACTOR,
                "balancedExtraRecords": len(extras),
                "extraKeys": [f"{key}#{index}" for key, index in extras],
                "recordCount": len(ordered),
            }
        )
    return result, {"passes": pass_details, "recordCount": len(result), "recordsPerPass": RECORDS_PER_PASS, "updates": len(result) // BATCH_SIZE, "sha256": canonical_sha256(result)}


def load_source(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    path = args.source_checkpoint.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-vr-continuation-checkpoint@1":
        raise ValueError(f"Unexpected source format: {payload.get('format')}")
    if int(payload.get("step", -1)) != SOURCE_STEP:
        raise ValueError(f"Expected source step {SOURCE_STEP}, got {payload.get('step')}")
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Source checkpoint has no optimizer state")
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
        "optimizerRestored": True,
        "sourceVariant": payload.get("variant"),
    }


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
    schedule_hash: str,
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
        "lossMode": "masked-anchor" if arm == ARMS[0] else "full-inst3-no-anchor",
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_hash,
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
    store = local.CacheStore(paths, max_open=len(paths))
    store.preload()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    updates_per_pass = RECORDS_PER_PASS // args.batch_size
    total_updates = args.passes * updates_per_pass
    if smoke_updates is not None:
        total_updates = min(smoke_updates, total_updates)
    contract = {
        **common_contract,
        "arm": arm,
        "lossMode": "masked-anchor" if arm == ARMS[0] else "full-inst3-no-anchor",
        "scheduleSha256": canonical_sha256(schedule),
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
        for local_step, pass_count in milestone_steps.items():
            path = run_root / f"step-{SOURCE_STEP + local_step}.pt"
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
        predicted_full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = predicted_full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
        forward_seconds = time.perf_counter() - forward_started
        event_losses = local.charbonnier_per_record(predicted, target_tensor, mask_tensor)
        anchor_losses = local.charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor)
        full_losses = local.charbonnier_per_record(predicted, target_tensor, torch.ones_like(mask_tensor))
        if arm == ARMS[0]:
            loss = (event_losses + args.anchor_beta * anchor_losses).mean()
        else:
            loss = full_losses.mean()
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
        history.append(
            {
                "localStep": current,
                "globalStep": SOURCE_STEP + current,
                "pass": current / updates_per_pass,
                "loss": float(loss.detach().cpu()),
                "eventLoss": float(event_losses.mean().detach().cpu()),
                "anchorLoss": float(anchor_losses.mean().detach().cpu()),
                "fullInst3Loss": float(full_losses.mean().detach().cpu()),
                "gradientNormBeforeClip": gradient,
                "forwardSeconds": forward_seconds,
                "backwardSeconds": backward_seconds,
                "records": len(items),
            }
        )
        if current == 1 or current % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "arm": arm,
                        "update": current,
                        "totalUpdates": total_updates,
                        "loss": history[-1]["loss"],
                        "forwardSeconds": forward_seconds,
                        "backwardSeconds": backward_seconds,
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
                SOURCE_STEP + current,
                current,
                contract_id,
                args,
                history,
                source,
                canonical_sha256(schedule),
            )
            checkpoints[str(pass_count)] = metadata
            print(json.dumps({"event": "milestone", "arm": arm, "pass": pass_count, "step": SOURCE_STEP + current}, sort_keys=True), flush=True)
        elif smoke_updates is None and current % args.state_interval == 0:
            save_checkpoint(
                run_root / f"step-{SOURCE_STEP + current}.pt",
                model,
                optimizer,
                arm,
                SOURCE_STEP + current,
                current,
                contract_id,
                args,
                history,
                source,
                canonical_sha256(schedule),
            )
    result = {
        "status": "smoke-completed" if smoke_updates is not None else "completed",
        "arm": arm,
        "updates": total_updates,
        "updatesPerPass": updates_per_pass,
        "recordsPerPass": RECORDS_PER_PASS,
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
    source: Path, training: dict[str, dict[str, Any]], milestones: tuple[int, ...]
) -> dict[str, Path]:
    paths: dict[str, Path] = {SOURCE_VARIANT: source.resolve()}
    for arm in ARMS:
        for pass_count in milestones:
            metadata = training[arm]["milestoneCheckpoints"].get(str(pass_count))
            if metadata is None:
                raise ValueError(f"Missing checkpoint for {arm} pass {pass_count}")
            paths[f"{arm}@pass-{pass_count}"] = Path(metadata["file"])
    return paths


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


def summarize_source_relative(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Compare every evaluated arm with this run's H50-continuation+5 source."""
    source = evaluation[SOURCE_VARIANT]["perSong"]
    whole_metrics = (
        "instrumentalSdrDb",
        "accompanimentErrorRmsDbfs",
        "positiveVocalProjectionDbfs",
        "teacherResidualMissRmsDbfs",
        "teacherRemovedPositiveProjectionDbfs",
    )
    result: dict[str, Any] = {"sourceVariant": SOURCE_VARIANT, "variants": {}}
    for variant, value in evaluation.items():
        if variant == SOURCE_VARIANT:
            continue
        per_song: dict[str, Any] = {}
        for slug, candidate in value["perSong"].items():
            source_song = source[slug]
            per_song[slug] = {
                "wholeSong": {
                    metric: float(candidate["wholeSong"][metric] - source_song["wholeSong"][metric])
                    for metric in whole_metrics
                },
                "regions": {
                    region: {
                        metric: float(candidate["regions"][region][metric] - source_song["regions"][region][metric])
                        for metric in whole_metrics
                    }
                    for region in candidate["regions"]
                },
                "events": {
                    str(milliseconds): {
                        metric: float(
                            candidate["events"][str(milliseconds)]["all"][metric]
                            - source_song["events"][str(milliseconds)]["all"][metric]
                        )
                        for metric in (
                            "missRmsP95Dbfs",
                            "missRmsMaxDbfs",
                            "positiveProjectionP95Dbfs",
                            "positiveProjectionMaxDbfs",
                        )
                        if metric in candidate["events"][str(milliseconds)]["all"]
                        and metric in source_song["events"][str(milliseconds)]["all"]
                    }
                    for milliseconds in (50, 100, 200)
                },
            }
        aggregate = {
            "wholeSong": {
                metric: float(np.mean([row["wholeSong"][metric] for row in per_song.values()]))
                for metric in whole_metrics
            },
            "events": {},
        }
        for milliseconds in (50, 100, 200):
            aggregate["events"][str(milliseconds)] = {}
            for metric in (
                "missRmsP95Dbfs",
                "missRmsMaxDbfs",
                "positiveProjectionP95Dbfs",
                "positiveProjectionMaxDbfs",
            ):
                values = [
                    row["events"][str(milliseconds)][metric]
                    for row in per_song.values()
                    if metric in row["events"][str(milliseconds)]
                ]
                aggregate["events"][str(milliseconds)][metric] = {
                    "songCount": len(values),
                    "meanDeltaDb": float(np.mean(values)) if values else None,
                    "medianDeltaDb": float(np.median(values)) if values else None,
                    "improvedSongs": int(sum(value < 0.0 for value in values)),
                    "worstDeltaDb": float(np.max(values)) if values else None,
                }
        result["variants"][variant] = {"aggregate": aggregate, "perSong": per_song}
    return result


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


def analyze_private(
    listening_report: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.resolve().read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for song, info in listening_report["songs"].items():
        source, source_rate = listening.load_audio(Path(info["source"]["file"]))
        if source_rate != SAMPLE_RATE or source.shape[1] != 2:
            raise ValueError(f"Unexpected private source contract for {song}: {source.shape}, {source_rate}")
        reference = quality.load_flac(
            Path(baseline["privateListening"]["songs"][song]["candidates"]["Inst3-native"]["instrumental"]["file"])
        )
        for variant, outputs in info["variants"].items():
            candidate = quality.load_flac(Path(outputs["instrumental"]["file"]))
            rows.append(
                {
                    "song": song,
                    "variant": variant,
                    **quality.metric_row(source, reference, candidate),
                    "shortEvents": {
                        str(ms): quality.short_event_row(source, reference, candidate, ms)
                        for ms in (50, 100, 200)
                    },
                }
            )
    aggregates: dict[str, Any] = {}
    for variant in listening_report["variants"]:
        values = [row for row in rows if row["variant"] == variant]
        aggregates[variant] = {
            "songCount": len(values),
            "mean": {
                key: statistics.fmean(row[key] for row in values)
                for key in (
                    "teacherMatchSnrDb",
                    "relativeErrorToRemovedDb",
                    "coherentRetainedRelativeDb",
                    "targetErrorRmsDbfs",
                )
            },
            "shortEvents": {
                str(ms): {
                    key: statistics.fmean(row["shortEvents"][str(ms)][key] for row in values)
                    for key in (
                        "missRmsP95Dbfs",
                        "missRmsMaxDbfs",
                        "positiveProjectionP95Dbfs",
                        "positiveProjectionMaxDbfs",
                    )
                }
                for ms in (50, 100, 200)
            },
        }
    source_variant = SOURCE_VARIANT
    deltas: dict[str, Any] = {}
    for variant in listening_report["variants"]:
        if variant == source_variant:
            continue
        deltas[variant] = {
            "vsSource": {
                song: {
                    key: next(row[key] for row in rows if row["song"] == song and row["variant"] == variant)
                    - next(row[key] for row in rows if row["song"] == song and row["variant"] == source_variant)
                    for key in ("teacherMatchSnrDb", "relativeErrorToRemovedDb", "coherentRetainedRelativeDb")
                }
                for song in listening_report["songs"]
            }
        }
    return {"rows": rows, "aggregate": aggregates, "deltas": deltas}


def _block_metrics(source: np.ndarray, reference: np.ndarray, candidate: np.ndarray, milliseconds: int) -> list[dict[str, float | int]]:
    block_samples = max(1, round(SAMPLE_RATE * milliseconds / 1000.0))
    count = int(math.ceil(source.shape[0] / block_samples))
    padding = count * block_samples - source.shape[0]
    source_blocks = np.pad(source.astype(np.float64), ((0, padding), (0, 0))).reshape(count, block_samples, 2)
    reference_blocks = np.pad(reference.astype(np.float64), ((0, padding), (0, 0))).reshape(count, block_samples, 2)
    candidate_blocks = np.pad(candidate.astype(np.float64), ((0, padding), (0, 0))).reshape(count, block_samples, 2)
    removed = source_blocks - reference_blocks
    miss = candidate_blocks - reference_blocks
    removed_power = np.sum(removed * removed, axis=(1, 2))
    miss_power = np.sum(miss * miss, axis=(1, 2))
    dot = np.sum(miss * removed, axis=(1, 2))
    removed_rms = np.sqrt(removed_power / float(block_samples * 2))
    miss_rms = np.sqrt(miss_power / float(block_samples * 2))
    coefficient = np.divide(dot, removed_power, out=np.zeros_like(dot), where=removed_power > 1.0e-30)
    positive = np.maximum(coefficient, 0.0) * removed_rms
    rows = []
    for index in np.argsort(-positive):
        if removed_rms[index] < 10.0 ** (-60.0 / 20.0):
            continue
        rows.append(
            {
                "blockIndex": int(index),
                "centerSamples": int(min(index * block_samples + block_samples // 2, source.shape[0] - 1)),
                "positiveProjectionDbfs": quality.db(float(positive[index])),
                "missRmsDbfs": quality.db(float(miss_rms[index])),
                "teacherRemovedRmsDbfs": quality.db(float(removed_rms[index])),
            }
        )
    return rows


def _crop_center(audio: np.ndarray, center: int, samples: int) -> np.ndarray:
    start = int(center) - samples // 2
    end = start + samples
    left = max(0, -start)
    right = max(0, end - audio.shape[0])
    clipped_start = max(0, start)
    clipped_end = min(audio.shape[0], end)
    result = np.zeros((samples, audio.shape[1]), dtype=np.float32)
    result[left : samples - right] = audio[clipped_start:clipped_end]
    return result


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path), "frames": int(audio.shape[0]), "sampleRate": SAMPLE_RATE, "channels": int(audio.shape[1])}


def render_event_clips(
    args: argparse.Namespace,
    listening_report: dict[str, Any],
    checkpoint_paths: dict[str, Path],
) -> dict[str, Any]:
    """Create 2 s H50 -> candidate -> Inst3 comparisons at current hotspots."""
    baseline = json.loads(args.baseline_report.resolve().read_text(encoding="utf-8"))
    output = args.output_root.resolve() / "event-listening"
    clips_root = output / "clips"
    silence = np.zeros((round(SAMPLE_RATE * 0.3), 2), dtype=np.float32)
    clip_samples = round(SAMPLE_RATE * 2.0)
    rows: list[dict[str, Any]] = []
    serial = 0
    songs = listening_report["songs"]
    for song_name, info in songs.items():
        source, source_rate = listening.load_audio(Path(info["source"]["file"]))
        if source_rate != SAMPLE_RATE or source.shape[1] != 2:
            raise ValueError(f"Unexpected private source contract for {song_name}: {source.shape}, {source_rate}")
        reference = quality.load_flac(Path(baseline["privateListening"]["songs"][song_name]["candidates"]["Inst3-native"]["instrumental"]["file"]))
        base = quality.load_flac(Path(info["variants"][SOURCE_VARIANT]["instrumental"]["file"]))
        candidates = {
            variant: quality.load_flac(Path(info["variants"][variant]["instrumental"]["file"]))
            for variant in info["variants"]
            if variant != SOURCE_VARIANT
        }
        selected: list[dict[str, Any]] = []
        for event in _block_metrics(source, reference, base, 100):
            center = int(event["centerSamples"])
            if any(abs(center - int(old["centerSamples"])) < SAMPLE_RATE for old in selected):
                continue
            selected.append(event)
            if len(selected) >= 2:
                break
        for event_index, event in enumerate(selected, start=1):
            center = int(event["centerSamples"])
            h50_clip = _crop_center(base, center, clip_samples)
            inst3_clip = _crop_center(reference, center, clip_samples)
            for variant, candidate in candidates.items():
                candidate_clip = _crop_center(candidate, center, clip_samples)
                serial += 1
                joined = np.concatenate((h50_clip, silence, candidate_clip, silence, inst3_clip), axis=0)
                event_id = f"{song_name}-hotspot-{event_index:02d}"
                file_path = clips_root / f"{serial:04d}-{event_id}-{variant}.flac"
                audio_info = write_flac(file_path, joined)
                rows.append(
                    {
                        "serial": serial,
                        "eventId": event_id,
                        "song": song_name,
                        "variant": variant,
                        "centerSamples": center,
                        "centerSeconds": center / SAMPLE_RATE,
                        "positiveProjectionDbfsH50": event["positiveProjectionDbfs"],
                        "missRmsDbfsH50": event["missRmsDbfs"],
                        "segments": {"h50": 0, "candidate": 2.3, "inst3": 4.6, "segmentSeconds": 2.0, "silenceSeconds": 0.3},
                        "audio": audio_info,
                    }
                )
        del source, reference, base, candidates
    report = {
        "schema": "local-inst3-s-only-stress-event-listening@1",
        "description": "2 s H50, 300 ms silence, 2 s candidate, 300 ms silence, 2 s Inst 3",
        "songCount": len(songs),
        "clipCount": len(rows),
        "rows": rows,
        "outputRoot": str(output.resolve()),
    }
    json_write(output / "event-listening-report.json", report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    milestones = parse_milestones(args.milestones, args.passes)
    if args.passes <= 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("invalid passes or batch size")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"This frozen stress contract requires batch size {BATCH_SIZE}")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("invalid optimizer/runtime arguments")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    required = (
        args.old_root / "external-selection.json",
        args.combined_root / "external-selection.json",
        args.source_checkpoint,
        args.checkpoint,
        args.musdb_manifest,
        args.oracle_root,
        args.baseline_report,
        args.samples_root,
    )
    for path in required:
        if not path.resolve().exists():
            raise FileNotFoundError(path)
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    paths, records, selection = load_s_pool(args)
    schedule, schedule_summary = build_schedule(records, args.passes, args.seed)
    json_write(args.output_root / "s-pool-selection.json", selection)
    json_write(args.output_root / "schedule.json", {"summary": schedule_summary, "schedule": schedule})
    common_contract = {
        "schema": SCHEMA,
        "sourceCheckpoint": checkpoint_metadata(args.source_checkpoint.resolve()),
        "architectureCheckpoint": checkpoint_metadata(args.checkpoint.resolve()),
        "selectionSha256": selection["selectionSha256"],
        "scheduleSha256": schedule_summary["sha256"],
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "baseSRecords": S_RECORDS,
        "repeatFactor": REPEAT_FACTOR,
        "balancedExtraRecords": BALANCED_EXTRA,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "assembly": "continuous-context-overlap-save",
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixture - Inst3-instrumental",
        "officialFinalTestUsed": False,
        "musdbTrainingUsed": False,
        "sourceOptimizerRestored": True,
    }
    if args.smoke_only:
        training = {
            arm: train_arm(arm, paths, schedule, args, device, milestones, common_contract, args.smoke_updates)
            for arm in ARMS
        }
        report = {
            "schema": SCHEMA,
            "status": "smoke-completed",
            "contract": common_contract,
            "selection": selection,
            "schedule": schedule_summary,
            "training": training,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchCuda": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        report_path = args.output_root / "reports" / "stress-smoke.json"
        json_write(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False, indent=2), flush=True)
        return 0
    training = {
        arm: train_arm(arm, paths, schedule, args, device, milestones, common_contract)
        for arm in ARMS
    }
    checkpoint_paths = build_checkpoint_paths(args.source_checkpoint, training, milestones)
    musdb_evaluation = None
    source_relative_evaluation = None
    if not args.skip_musdb_evaluation:
        musdb_evaluation = evaluate_musdb(args, contract, checkpoint_paths, device)
        source_relative_evaluation = summarize_source_relative(musdb_evaluation)
        json_write(args.output_root / "reports" / "musdb-evaluation.json", musdb_evaluation)
        json_write(args.output_root / "reports" / "musdb-source-relative.json", source_relative_evaluation)
    listening_report = None
    private_analysis = None
    event_report = None
    if not args.skip_listening:
        existing_listening = args.output_root / "listening-report.json"
        if not args.force_listening and existing_listening.is_file():
            try:
                cached_listening = json.loads(existing_listening.read_text(encoding="utf-8"))
                if (
                    cached_listening.get("songCount") == 12
                    and cached_listening.get("variants") == list(checkpoint_paths)
                    and all(len(song.get("variants", {})) == len(checkpoint_paths) for song in cached_listening.get("songs", {}).values())
                ):
                    listening_report = cached_listening
                    print(json.dumps({"event": "reuse-listening", "file": str(existing_listening)}), flush=True)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                listening_report = None
        if listening_report is None:
            listening_report = render_listening(args, contract, checkpoint_paths, device)
        private_analysis = analyze_private(listening_report, args.baseline_report)
        json_write(args.output_root / "reports" / "private-inst3-analysis.json", private_analysis)
        if not args.skip_event_clips:
            event_report = render_event_clips(args, listening_report, checkpoint_paths)
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "selection": selection,
        "schedule": schedule_summary,
        "training": training,
        "checkpoints": {name: checkpoint_metadata(path) for name, path in checkpoint_paths.items()},
        "musdbEvaluation": musdb_evaluation,
        "musdbSourceRelative": source_relative_evaluation,
        "listening": listening_report,
        "privateInst3Analysis": private_analysis,
        "eventListening": event_report,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runner": checkpoint_metadata(Path(__file__).resolve()),
        },
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "originals": "source audio and derived caches remain local",
            "weights": "teacher-derived checkpoints remain local pending rights review",
        },
    }
    report_path = args.output_root / "reports" / "stress-report.json"
    json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "checkpoints": {name: str(path) for name, path in checkpoint_paths.items()},
                "listeningOutputs": 0 if listening_report is None else listening_report.get("outputCount", 0),
                "eventClips": 0 if event_report is None else event_report.get("clipCount", 0),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
