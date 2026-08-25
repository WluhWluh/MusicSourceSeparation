#!/usr/bin/env python3
"""Continue the all-S event-only arm until its projection tail plateaus.

The run starts from ``S86-event-only@pass-5`` and preserves its exact loss,
cache contract, optimizer state, and deterministic schedule.  Every new pass
is checkpointed.  Continuous MUSDB evaluation is performed only at the end
of a requested chunk; this runner intentionally has no listening renderer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_s_r_continuation as sr
import run_inst3_vr_continuation as continuation
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "modern-song-s-r-continuation"
    / "runs"
    / "S86-event-only"
    / "step-3360.pt"
)
DEFAULT_CACHE_ROOT = ROOT / "data" / "modern-song-s-r-continuation" / "cache" / "external" / "S"
DEFAULT_H50_SOURCE = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-continuation"
    / "runs"
    / "H50-continuation-plus5"
    / "step-1600.pt"
)
DEFAULT_OLD_S_ROOT = ROOT / "data" / "modern-song-s-pilot"
DEFAULT_CURRENT_S_ROOT = ROOT / "data" / "modern-song-s-combined-pilot"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_MANIFEST = ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_BASELINE_REPORT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "continuous-baseline-report.json"
DEFAULT_PRIOR_REPORT = ROOT / "data" / "modern-song-s-r-continuation" / "reports" / "s-r-continuation-report.json"
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-s-event-pressure"

SAMPLE_RATE = 44_100
RECORDS_PER_PASS = 704
BATCH_SIZE = 4
UPDATES_PER_PASS = RECORDS_PER_PASS // BATCH_SIZE
BASE_PRESSURE_STEP = 3_360
BASE_PRESSURE_PASS = 5
S_RECORDS = 86
R_RECORDS = 46
REPEAT_FACTOR = 8
BALANCED_EXTRA = 16
CORE_MS = 100
GUARD_MS = 25
ANCHOR_BETA = 0.25
CHECKPOINT_FORMAT = "local-inst3-s-event-pressure-checkpoint@1"
SCHEMA = "local-inst3-s-event-pressure@1"
SCHEDULE_HASH_IN_PRIOR_RUN = "832e2d77f71407cd258fcd68050ab1aafc7a97841c2242b6a3ebc917585c63b7"
EPSILON = 1.0e-4


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--h50-source", type=Path, default=DEFAULT_H50_SOURCE)
    parser.add_argument("--s-old-root", type=Path, default=DEFAULT_OLD_S_ROOT)
    parser.add_argument("--s-current-root", type=Path, default=DEFAULT_CURRENT_S_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=5, help="Additional passes in this invocation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=ANCHOR_BETA)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--state-interval", type=int, default=UPDATES_PER_PASS)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--evaluate-passes", help="Evaluate existing pressure checkpoints, e.g. 6,8,10; skips training")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
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


def choose_balanced(
    records: Sequence[dict[str, Any]], count: int, rng: np.random.Generator, max_per_song: int = 2
) -> list[tuple[str, int]]:
    if count <= 0 or count > len(records):
        raise ValueError(f"Invalid balanced sample count {count} for {len(records)} records")
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw_index in rng.permutation(len(records)):
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


def load_s_records(args: argparse.Namespace) -> tuple[dict[str, Path], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the original record order and map it to the re-anchored S caches."""
    source_args = argparse.Namespace(
        h50_source=args.h50_source,
        s_old_root=args.s_old_root,
        s_current_root=args.s_current_root,
    )
    _source_paths, s_records = sr.load_s_pool(source_args)
    cache_paths: dict[str, Path] = {}
    for metadata_path in sorted(args.cache_root.resolve().glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = str(metadata.get("key", ""))
        cache = Path(metadata["cache"]["file"]).resolve()
        if key and cache.is_file():
            cache_paths[key] = cache
    if len(cache_paths) != 38 or len(s_records) != S_RECORDS:
        raise ValueError(f"Expected 38 S caches and 86 records, got {len(cache_paths)} and {len(s_records)}")
    for record in s_records:
        key = str(record["key"])
        if key not in cache_paths:
            raise ValueError(f"Missing re-anchored S cache for {key}")
        with np.load(cache_paths[key]) as values:
            count = int(values["inputSpec"].shape[0])
        if int(record["index"]) >= count:
            raise ValueError(f"S record index out of range: {key}/{record['index']}/{count}")

    # The R records are not trained, but their exact pool shape is needed to
    # reproduce the RNG consumption of the original four-arm schedule.
    r_records: list[dict[str, Any]] = []
    r_root = args.cache_root.resolve().parent / "R"
    for metadata_path in sorted(r_root.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = str(metadata["key"])
        for index, event in enumerate(metadata.get("selectedEvents", [])):
            r_records.append(
                {
                    "key": key,
                    "index": index,
                    "songKey": key,
                    "eventId": event.get("eventId", f"{key}-{index}"),
                }
            )
    if len(r_records) != R_RECORDS:
        raise ValueError(f"Expected {R_RECORDS} R records for schedule reproduction, got {len(r_records)}")
    return cache_paths, s_records, r_records


def build_future_schedule(
    s_records: Sequence[dict[str, Any]],
    r_records: Sequence[dict[str, Any]],
    start_pass_index: int,
    passes: int,
    seed: int,
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """Re-run the committed schedule builder and take only future S passes."""
    dummy_paths = {f"musdb::song-{index:02d}": Path(f"song-{index:02d}.npz") for index in range(80)}
    total_passes = start_pass_index + passes
    schedules, summary = sr.build_schedules(dummy_paths, s_records, r_records, total_passes, seed)
    full = schedules["S86-event-only"]
    begin = start_pass_index * RECORDS_PER_PASS
    end = (start_pass_index + passes) * RECORDS_PER_PASS
    future = full[begin:end]
    if len(future) != passes * RECORDS_PER_PASS:
        raise AssertionError(f"Future schedule has {len(future)} records")
    if any("::S::" not in key for key, _ in future):
        raise AssertionError("Pressure schedule contains a non-S record")
    return future, {
        "priorScheduleSha256": canonical_sha256(full[: BASE_PRESSURE_PASS * RECORDS_PER_PASS]),
        "fullScheduleSha256": canonical_sha256(full),
        "futureScheduleSha256": canonical_sha256(future),
        "startPassIndex": start_pass_index,
        "passes": passes,
        "recordCount": len(future),
        "recordsPerPass": RECORDS_PER_PASS,
        "updatesPerPass": UPDATES_PER_PASS,
        "uniqueRecords": len(set(future)),
        "poolCounts": dict(Counter("S" if "::S::" in key else "other" for key, _ in future)),
        "builder": "run_inst3_s_r_continuation.build_schedules",
    }


def pressure_pass_from_step(step: int) -> int:
    if step < BASE_PRESSURE_STEP or (step - BASE_PRESSURE_STEP) % UPDATES_PER_PASS:
        raise ValueError(f"Step {step} is not aligned to the pressure pass boundary")
    return BASE_PRESSURE_PASS + (step - BASE_PRESSURE_STEP) // UPDATES_PER_PASS


def load_model_optimizer(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    source_path = args.source_checkpoint.resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("format") not in {
        "local-inst3-s-r-continuation-checkpoint@1",
        CHECKPOINT_FORMAT,
    }:
        raise ValueError(f"Unexpected pressure source format: {payload.get('format')}")
    if payload.get("variant") not in {"S86-event-only", "S86-event-only-pressure"}:
        raise ValueError(f"Pressure source is not S86-event-only: {payload.get('variant')}")
    source_step = int(payload.get("step", -1))
    source_pass = pressure_pass_from_step(source_step)
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError("Pressure source has no optimizer state")
    model, architecture = pilot.make_model(args.checkpoint.resolve(), device)
    model.load_state_dict(payload["stateDict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizerStateDict"])
    continuation.move_optimizer_state(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    return model, optimizer, {
        "sourceCheckpoint": checkpoint_metadata(source_path),
        "sourceVariant": payload.get("variant"),
        "sourceStep": source_step,
        "sourcePass": source_pass,
        "sourceOptimizerRestored": True,
        "architecture": architecture["checkpoint"],
    }, payload


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    source: dict[str, Any],
    history: list[dict[str, Any]],
    global_step: int,
    pressure_pass: int,
    contract_id: str,
    schedule_sha256: str,
) -> dict[str, Any]:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": "S86-event-only-pressure",
        "runContractId": contract_id,
        "step": global_step,
        "globalStep": global_step,
        "pressurePass": pressure_pass,
        "recordsPerPass": RECORDS_PER_PASS,
        "updatesPerPass": UPDATES_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "coreMs": CORE_MS,
        "guardMs": GUARD_MS,
        "seed": args.seed,
        "scheduleSha256": schedule_sha256,
        "lossContract": "S-only 100ms+25ms Inst3 event target plus S86 anchor outside event",
        "stateDict": local.hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": local.hard.cpu_tree(optimizer.state_dict()),
        "history": history,
        "source": source,
    }
    return local.hard.atomic_torch_save(path, payload)


def train_chunk(
    args: argparse.Namespace,
    device: torch.device,
    cache_paths: dict[str, Path],
    schedule: list[tuple[str, int]],
    schedule_info: dict[str, Any],
) -> dict[str, Any]:
    model, optimizer, source, source_payload = load_model_optimizer(args, device)
    source_step = int(source["sourceStep"])
    source_pass = int(source["sourcePass"])
    total_updates = len(schedule) // args.batch_size
    if len(schedule) % args.batch_size:
        raise ValueError("Pressure schedule is not divisible by batch size")
    contract_payload = {
        "schema": SCHEMA,
        "sourceCheckpointSha256": source["sourceCheckpoint"]["sha256"],
        "cacheRoot": str(args.cache_root.resolve()),
        "cacheKeysSha256": canonical_sha256(sorted(str(key) for key in cache_paths)),
        "startPass": source_pass,
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "scheduleSha256": schedule_info["futureScheduleSha256"],
        "assembly": "continuous-context-overlap-save",
        "officialFinalTestUsed": False,
    }
    contract_id = canonical_sha256(contract_payload)
    store = local.CacheStore(cache_paths, max_open=len(cache_paths))
    store.preload()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    output_run = args.output_root.resolve() / "runs" / "S86-event-only"
    output_run.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    current = 0
    next_pass = source_pass + 1
    checkpoints: list[dict[str, Any]] = []
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
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
        event_losses = local.charbonnier_per_record(predicted, target_tensor, mask_tensor)
        anchor_losses = local.charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor)
        loss = (event_losses + args.anchor_beta * anchor_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite pressure loss at update {current + 1}")
        backward_started = time.perf_counter()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        if not math.isfinite(gradient):
            raise FloatingPointError(f"Non-finite pressure gradient at update {current + 1}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        current += 1
        global_step = source_step + current
        fractional_pass = source_pass + current / UPDATES_PER_PASS
        history.append(
            {
                "update": current,
                "globalStep": global_step,
                "pressurePass": fractional_pass,
                "loss": float(loss.detach().cpu()),
                "eventLoss": float(event_losses.mean().detach().cpu()),
                "anchorLoss": float(anchor_losses.mean().detach().cpu()),
                "gradientNormBeforeClip": gradient,
                "forwardSeconds": forward_seconds,
                "backwardSeconds": backward_seconds,
            }
        )
        if current == 1 or current % 100 == 0:
            allocated = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "update": current,
                        "totalUpdates": total_updates,
                        "globalStep": global_step,
                        "pressurePass": fractional_pass,
                        "loss": history[-1]["loss"],
                        "forwardSeconds": forward_seconds,
                        "backwardSeconds": backward_seconds,
                        "cudaAllocatedBytes": allocated,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if current % UPDATES_PER_PASS == 0:
            pass_number = source_pass + current // UPDATES_PER_PASS
            metadata = save_checkpoint(
                output_run / f"step-{global_step}.pt",
                model,
                optimizer,
                args,
                source,
                history,
                global_step,
                pass_number,
                contract_id,
                schedule_info["futureScheduleSha256"],
            )
            checkpoints.append({"pass": pass_number, "step": global_step, "checkpoint": metadata})
            print(json.dumps({"event": "milestone", "pass": pass_number, "step": global_step}, sort_keys=True), flush=True)
        elif args.state_interval > 0 and current % args.state_interval == 0:
            save_checkpoint(
                output_run / f"step-{global_step}.pt",
                model,
                optimizer,
                args,
                source,
                history,
                global_step,
                source_pass + current // UPDATES_PER_PASS,
                contract_id,
                schedule_info["futureScheduleSha256"],
            )
    result = {
        "status": "completed",
        "source": source,
        "sourcePayloadFormat": source_payload.get("format"),
        "sourcePass": source_pass,
        "finalPass": source_pass + args.passes,
        "sourceStep": source_step,
        "finalStep": source_step + total_updates,
        "passes": args.passes,
        "recordsPerPass": RECORDS_PER_PASS,
        "updatesPerPass": UPDATES_PER_PASS,
        "updates": total_updates,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "optimizerRestored": True,
        "schedule": schedule_info,
        "contract": {"id": contract_id, "payload": contract_payload},
        "checkpoints": checkpoints,
        "elapsedSeconds": time.perf_counter() - started,
        "historySummary": {
            "firstLoss": history[0]["loss"] if history else None,
            "lastLoss": history[-1]["loss"] if history else None,
            "meanLoss": float(np.mean([row["loss"] for row in history])) if history else None,
        },
    }
    del model, optimizer, store, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def absolute_event_metric(evaluation: dict[str, Any], metric: str) -> float:
    values = [
        float(song["events"]["100"]["all"][metric])
        for song in evaluation["perSong"].values()
    ]
    return float(np.mean(values))


def evaluate_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: Path,
    pressure_pass: int,
) -> dict[str, Any]:
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    entries = sorted(
        [entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}],
        key=lambda item: (item["role"], item["member"]),
    )
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    label = f"S86-event-only-pressure-pass-{pressure_pass}"
    eval_args = argparse.Namespace(
        baseline_report=args.baseline_report.resolve(),
        output_root=args.output_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        oracle_root=args.oracle_root.resolve(),
        inference_batch_size=args.inference_batch_size,
    )
    result = local.evaluate_trained_models(eval_args, contract, {label: checkpoint.resolve()}, entries, device)[label]
    return {
        "label": label,
        "pass": pressure_pass,
        "checkpoint": checkpoint_metadata(checkpoint),
        "songCount": result["songCount"],
        "absolute100msPositiveProjectionP95Dbfs": absolute_event_metric(result, "positiveProjectionP95Dbfs"),
        "absolute100msPositiveProjectionMaxDbfs": absolute_event_metric(result, "positiveProjectionMaxDbfs"),
        "absolute100msMissRmsP95Dbfs": absolute_event_metric(result, "missRmsP95Dbfs"),
        "absolute100msMissRmsMaxDbfs": absolute_event_metric(result, "missRmsMaxDbfs"),
        "instrumentalSdrDb": result["aggregatePerSongMean"]["wholeSong"]["instrumentalSdrDb"],
        "deltaVsH50": result["eventDirectionsVsH50"]["100"],
        "elapsedSeconds": result["elapsedSeconds"],
    }


def prior_base_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.prior_report.resolve().is_file():
        raise FileNotFoundError(args.prior_report)
    report = json.loads(args.prior_report.resolve().read_text(encoding="utf-8"))
    source = report["evaluation"]["S86-event-only@pass-5"]
    return {
        "label": "S86-event-only@pass-5",
        "pass": BASE_PRESSURE_PASS,
        "isBase": True,
        "checkpoint": source["checkpoint"],
        "songCount": source["songCount"],
        "absolute100msPositiveProjectionP95Dbfs": absolute_event_metric(source, "positiveProjectionP95Dbfs"),
        "absolute100msPositiveProjectionMaxDbfs": absolute_event_metric(source, "positiveProjectionMaxDbfs"),
        "absolute100msMissRmsP95Dbfs": absolute_event_metric(source, "missRmsP95Dbfs"),
        "absolute100msMissRmsMaxDbfs": absolute_event_metric(source, "missRmsMaxDbfs"),
        "instrumentalSdrDb": source["aggregatePerSongMean"]["wholeSong"]["instrumentalSdrDb"],
        "deltaVsH50": source["eventDirectionsVsH50"]["100"],
        "elapsedSeconds": source.get("elapsedSeconds"),
    }


def merge_evaluation(report: dict[str, Any], evaluation: dict[str, Any]) -> None:
    report["evaluations"] = [
        item for item in report["evaluations"] if item.get("label") != evaluation.get("label")
    ]
    lower = [
        item
        for item in report["evaluations"]
        if int(item.get("pass", -1)) < int(evaluation["pass"])
    ]
    prior = max(lower, key=lambda item: int(item["pass"])) if lower else None
    evaluation["deltaFromPrevious100msP95Db"] = (
        None
        if prior is None
        else evaluation["absolute100msPositiveProjectionP95Dbfs"]
        - prior["absolute100msPositiveProjectionP95Dbfs"]
    )
    base = next(
        (item for item in report["evaluations"] if item.get("isBase") or int(item.get("pass", -1)) == BASE_PRESSURE_PASS),
        None,
    )
    evaluation["deltaFromBaseS86Pass5_100msP95Db"] = (
        None
        if base is None
        else evaluation["absolute100msPositiveProjectionP95Dbfs"]
        - base["absolute100msPositiveProjectionP95Dbfs"]
    )
    report["evaluations"].append(evaluation)
    report["evaluations"].sort(key=lambda item: int(item.get("pass", -1)))


def update_report(
    args: argparse.Namespace,
    device: torch.device,
    chunk: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> Path:
    report_path = args.output_root.resolve() / "reports" / "pressure-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "schema": SCHEMA,
            "status": "in-progress",
            "baseCheckpoint": checkpoint_metadata(DEFAULT_SOURCE if DEFAULT_SOURCE.is_file() else args.source_checkpoint),
            "chunks": [],
            "evaluations": [],
            "notes": {
                "objective": "Continue S86-event-only until 100 ms projection p95 plateaus",
                "assembly": "continuous-context-overlap-save",
                "audioRendering": "intentionally omitted during pressure test",
                "officialFinalTestUsed": False,
            },
        }
    if not any(item.get("pass") == BASE_PRESSURE_PASS and item.get("isBase") for item in report["evaluations"]):
        report["evaluations"].insert(0, prior_base_evaluation(args))
    if chunk:
        report["chunks"].append(chunk)
    if evaluation is not None:
        merge_evaluation(report, evaluation)
    report["status"] = "completed-chunk"
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "cudaAvailable": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "runner": {"file": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    json_write(report_path, report)
    return report_path


def validate_args(args: argparse.Namespace) -> None:
    if args.passes <= 0 or args.batch_size <= 0 or RECORDS_PER_PASS % args.batch_size:
        raise ValueError("passes and batch-size must be positive and divide 704")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0:
        raise ValueError("invalid learning rate, anchor beta, or threads")
    if args.inference_batch_size <= 0 or args.state_interval <= 0:
        raise ValueError("invalid inference batch size or state interval")


def parse_pass_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values or values[0] <= BASE_PRESSURE_PASS:
        raise ValueError("evaluate-passes must be sorted and greater than the base pass")
    return values


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
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
        args.checkpoint,
        args.manifest,
        args.oracle_root,
        args.baseline_report,
        args.prior_report,
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path)
    if args.evaluate_passes:
        evaluations: list[dict[str, Any]] = []
        for pressure_pass in parse_pass_list(args.evaluate_passes):
            step = BASE_PRESSURE_STEP + (pressure_pass - BASE_PRESSURE_PASS) * UPDATES_PER_PASS
            checkpoint = args.output_root / "runs" / "S86-event-only" / f"step-{step}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            evaluation = evaluate_checkpoint(args, device, checkpoint, pressure_pass)
            evaluations.append(evaluation)
            print(json.dumps({"event": "evaluation", **evaluation}, sort_keys=True), flush=True)
        report_path = update_report(args, device, {}, None)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for evaluation in evaluations:
            merge_evaluation(report, evaluation)
        report["status"] = "evaluation-only"
        json_write(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path), "evaluatedPasses": list(parse_pass_list(args.evaluate_passes))}, indent=2), flush=True)
        return 0
    for path in (args.cache_root, args.h50_source, args.s_old_root, args.s_current_root):
        if not path.resolve().exists():
            raise FileNotFoundError(path)
    cache_paths, s_records, r_records = load_s_records(args)
    source_payload = torch.load(args.source_checkpoint.resolve(), map_location="cpu", weights_only=False)
    source_step = int(source_payload.get("step", -1))
    source_pass = pressure_pass_from_step(source_step)
    schedule, schedule_info = build_future_schedule(s_records, r_records, source_pass, args.passes, args.seed)
    if source_pass == BASE_PRESSURE_PASS and schedule_info["priorScheduleSha256"] != SCHEDULE_HASH_IN_PRIOR_RUN:
        raise AssertionError(
            f"Prior S-only schedule mismatch: {schedule_info['priorScheduleSha256']} != {SCHEDULE_HASH_IN_PRIOR_RUN}"
        )
    if args.smoke_only:
        model, optimizer, source, _payload = load_model_optimizer(args, device)
        store = local.CacheStore(cache_paths, max_open=len(cache_paths))
        window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
        losses: list[float] = []
        for update in range(args.smoke_updates):
            items = schedule[update * args.batch_size : (update + 1) * args.batch_size]
            input_array, target_array, anchor_array, mask_array = store.batch(items)
            input_tensor = torch.from_numpy(input_array).to(device)
            target_tensor = torch.from_numpy(target_array).to(device)
            anchor_tensor = torch.from_numpy(anchor_array).to(device)
            mask_tensor = torch.from_numpy(mask_array).to(device)
            optimizer.zero_grad(set_to_none=True)
            full = local.torch_packed_istft(model(input_tensor), window)
            predicted = full[:, pilot.DEFAULT_CONFIG.trim_samples : pilot.DEFAULT_CONFIG.trim_samples + pilot.DEFAULT_CONFIG.useful_samples]
            loss = (
                local.charbonnier_per_record(predicted, target_tensor, mask_tensor)
                + args.anchor_beta * local.charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor)
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pressure smoke loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            if not torch.isfinite(gradient):
                raise FloatingPointError("Non-finite pressure smoke gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"status": "smoke-completed", "updates": args.smoke_updates, "losses": losses, "schedule": schedule_info}, indent=2))
        return 0
    chunk = train_chunk(args, device, cache_paths, schedule, schedule_info)
    evaluation = None
    if not args.skip_evaluation:
        latest = chunk["checkpoints"][-1]
        evaluation = evaluate_checkpoint(args, device, Path(latest["checkpoint"]["file"]), int(latest["pass"]))
        print(json.dumps({"event": "evaluation", **evaluation}, sort_keys=True), flush=True)
    report_path = update_report(args, device, chunk, evaluation)
    print(json.dumps({"status": "completed", "report": str(report_path), "finalStep": chunk["finalStep"], "evaluated": evaluation is not None}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
