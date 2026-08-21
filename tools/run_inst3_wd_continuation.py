#!/usr/bin/env python3
"""Continue the W-D direct-instrumental cell without mutating the matrix run.

The parent matrix established the first 100 complete passes for W-D with a
pretrained TFC-TDF body and a reset direct-instrumental output head.  This
runner verifies the parent state and its schedule prefix, restores the parent
model/AdamW/RNG state exactly, and extends the same deterministic window
schedule to a bounded 200-pass continuation.

All generated states, checkpoints, and reports are local non-commercial
research artifacts under the ignored data tree.  The original four-cell matrix
directory is read-only input to this experiment.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import run_inst3_aggressive_scale10 as scale10
import run_inst3_distill_pilot as pilot
import run_inst3_initialization_output_matrix as matrix
import run_inst3_scale10_density as density
import run_inst3_distill_stability_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "inst3-wd-continuation@1"
STATE_FORMAT = "local-inst3-wd-continuation-state@1"
CHECKPOINT_FORMAT = "local-inst3-wd-continuation-checkpoint@1"
DEFAULT_PARENT_ROOT = ROOT / "data" / "musdb18-inst3-init-output-matrix"
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-wd-continuation"
DEFAULT_DATA_ROOT = matrix.DEFAULT_DATA_ROOT
DEFAULT_EVAL_ROOT = matrix.DEFAULT_EVAL_ROOT
DEFAULT_MANIFEST = matrix.DEFAULT_MANIFEST
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_SEEDS = (891,)
DEFAULT_START_PASSES = 100
DEFAULT_PASSES = 200
DEFAULT_MILESTONE_PASSES = (100, 125, 150, 200)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--parent-experiment-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--start-passes", type=int, default=DEFAULT_START_PASSES)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument(
        "--milestone-passes",
        default=",".join(str(value) for value in DEFAULT_MILESTONE_PASSES),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--train-windows-per-song", type=int, default=32)
    parser.add_argument("--holdout-windows-per-song", type=int, default=16)
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--train-probe-windows-per-song", type=int, default=8)
    parser.add_argument("--probe-song-count", type=int, default=8)
    parser.add_argument("--state-interval-updates", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="Pause every selected run after this total update count; smoke testing only",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    seeds = matrix.parse_csv_ints(args.seeds)
    milestones = matrix.parse_csv_ints(args.milestone_passes)
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    if (
        args.start_passes <= 0
        or args.passes <= args.start_passes
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or args.train_windows_per_song <= 0
        or args.holdout_windows_per_song <= 0
        or args.eval_windows_per_song <= 0
        or args.train_probe_windows_per_song <= 0
        or args.probe_song_count <= 0
        or args.state_interval_updates <= 0
        or args.threads <= 0
    ):
        raise ValueError("Passes, counts, intervals, threads, and learning rate must be positive")
    if milestones[0] != args.start_passes or milestones[-1] != args.passes:
        raise ValueError("Milestones must start at --start-passes and end at --passes")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("Milestones must be sorted and unique")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        raise ValueError("--stop-after-updates must be positive")
    return seeds, milestones


def parent_state_path(parent_root: Path, seed: int) -> Path:
    return parent_root / "runs" / f"seed-{seed}" / "W-D" / "training-state.pt"


def expected_parent_schedule(
    *, window_count: int, start_passes: int, seed: int
) -> np.ndarray:
    return matrix.build_sample_schedule(
        window_count, start_passes, matrix.make_schedule_seed(seed)
    )


def validate_parent_payload(
    *,
    payload: dict[str, Any],
    parent_path: Path,
    seed: int,
    args: argparse.Namespace,
    schedule: np.ndarray,
    updates_per_pass: int,
) -> dict[str, Any]:
    if payload.get("format") not in {matrix.STATE_FORMAT, matrix.CHECKPOINT_FORMAT}:
        raise ValueError(f"Unexpected parent state format: {payload.get('format')}")
    if payload.get("status") != "completed":
        raise ValueError(f"Parent state is not completed: {parent_path}")
    expected_update = args.start_passes * updates_per_pass
    if int(payload.get("update", -1)) != expected_update:
        raise ValueError(
            f"Parent update {payload.get('update')} does not equal {expected_update}"
        )
    run_contract = payload.get("runContract", {})
    contract = run_contract.get("contract", {})
    if contract.get("experimentId") != matrix.EXPERIMENT_ID:
        raise ValueError("Parent state is not from the initialization/output matrix")
    if contract.get("variant") != {
        "key": "W-D",
        "initialization": "pretrained-body",
        "output_semantic": "direct-instrumental",
        "reset_output_head": True,
    }:
        raise ValueError("Parent state is not the W-D direct-instrumental cell")
    expected_contract = {
        "seed": seed,
        "passes": args.start_passes,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "trainWindowsPerSong": args.train_windows_per_song,
        "holdoutWindowsPerSong": args.holdout_windows_per_song,
        "evalWindowsPerSong": args.eval_windows_per_song,
        "trainWindowCount": updates_per_pass * args.batch_size,
        "scheduleSeed": matrix.make_schedule_seed(seed),
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"Parent contract mismatch for {key}: {contract.get(key)!r} != {expected!r}"
            )
    prefix = expected_parent_schedule(
        window_count=updates_per_pass * args.batch_size,
        start_passes=args.start_passes,
        seed=seed,
    )
    expected_hash = matrix.schedule_sha256(prefix)
    if contract.get("scheduleSha256") != expected_hash:
        raise ValueError("Parent schedule hash does not match its frozen schedule")
    if not np.array_equal(schedule[: len(prefix)], prefix):
        raise ValueError("Continuation schedule does not preserve the parent prefix")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != expected_update:
        raise ValueError("Parent history does not cover exactly the parent updates")
    milestones = payload.get("milestoneResults", {})
    if str(args.start_passes) not in milestones:
        raise ValueError("Parent state is missing its final milestone")
    if "baseline" not in payload or "rngState" not in payload:
        raise ValueError("Parent state does not contain baseline or RNG state")
    return run_contract


def make_run_contract(
    *,
    args: argparse.Namespace,
    seed: int,
    schedule: np.ndarray,
    parent_path: Path,
    parent_sha256: str,
    parent_contract: dict[str, Any],
    window_report: dict[str, Any],
    train_window_count: int,
    device: torch.device,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    parent_value = parent_contract["contract"]
    value = {
        "experimentId": EXPERIMENT_ID,
        "variant": "W-D",
        "seed": seed,
        "parentRunContractId": parent_contract["id"],
        "parentStatePath": str(parent_path.resolve()),
        "parentStateSha256": parent_sha256,
        "parentUpdate": args.start_passes * (train_window_count // args.batch_size),
        "startPasses": args.start_passes,
        "passes": args.passes,
        "milestonePasses": list(matrix.parse_csv_ints(args.milestone_passes)),
        "scheduleSeed": matrix.make_schedule_seed(seed),
        "continuationScheduleSha256": matrix.schedule_sha256(schedule),
        "parentScheduleSha256": parent_value["scheduleSha256"],
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "optimizer": "AdamW(weight_decay=0)",
        "gradientClipNorm": 1.0,
        "precision": "float32",
        "batchNormPolicy": "updated in train mode for W-D",
        "trainWindowCount": train_window_count,
        "trainWindowsPerSong": args.train_windows_per_song,
        "holdoutWindowsPerSong": args.holdout_windows_per_song,
        "evalWindowsPerSong": args.eval_windows_per_song,
        "manifestSha256": pilot.sha256_file(args.manifest.resolve()),
        "selectionSha256": pilot.sha256_file(
            args.data_root.resolve() / "scale10-selection.json"
        ),
        "windowSelectionSha256": density.canonical_sha256(window_report),
        "initialCheckpointSha256": pilot.sha256_file(args.checkpoint.resolve()),
        "device": str(device),
        "torchVersion": torch.__version__,
        "runnerSha256": pilot.sha256_file(runner_path),
    }
    return {"id": density.canonical_sha256(value), "contract": value}


def find_resume_path(run_root: Path, milestone_updates: Sequence[int]) -> Path | None:
    candidates = [run_root / "training-state.pt"]
    candidates.extend(
        run_root / f"pass-{update}.pt" for update in milestone_updates if update > 0
    )
    best: tuple[int, float, Path] | None = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            update = int(payload["update"])
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        candidate = (update, path.stat().st_mtime, path)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else None


def validate_continuation_payload(
    *,
    payload: dict[str, Any],
    run_contract: dict[str, Any],
) -> None:
    if payload.get("format") not in {STATE_FORMAT, CHECKPOINT_FORMAT}:
        raise ValueError(f"Unexpected continuation state format: {payload.get('format')}")
    if payload.get("runContract", {}).get("id") != run_contract["id"]:
        raise ValueError("Continuation state contract does not match this run")
    parent = payload.get("parent", {})
    if parent.get("stateSha256") != run_contract["contract"]["parentStateSha256"]:
        raise ValueError("Continuation state has a different parent state")


def checkpoint_payload(
    *,
    state_format: str,
    status: str,
    run_contract: dict[str, Any],
    parent: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    history: list[dict[str, float | int]],
    milestones: dict[str, Any],
    baseline: dict[str, Any],
    continuation_elapsed_seconds: float,
    parent_elapsed_seconds: float,
    resume_count: int,
) -> dict[str, Any]:
    return {
        "format": state_format,
        "status": status,
        "runContract": run_contract,
        "parent": parent,
        "update": update,
        "state_dict": density.cpu_tree(model.state_dict()),
        "optimizer_state_dict": density.cpu_tree(optimizer.state_dict()),
        "history": history,
        "milestoneResults": milestones,
        "baseline": baseline,
        "continuationElapsedSeconds": continuation_elapsed_seconds,
        "parentElapsedSeconds": parent_elapsed_seconds,
        "resumeCount": resume_count,
        "rngState": density.capture_rng_state(),
    }


def continuation_parent_metadata(
    *, parent_path: Path, parent_sha256: str, parent_contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        "statePath": str(parent_path.resolve()),
        "stateSha256": parent_sha256,
        "runContractId": parent_contract["id"],
        "sourceExperimentId": parent_contract["contract"]["experimentId"],
        "sourceUpdate": parent_contract["contract"]["passes"]
        * (parent_contract["contract"]["trainWindowCount"] // parent_contract["contract"]["batchSize"]),
    }


def prepare_data(
    args: argparse.Namespace,
) -> tuple[
    list[pilot.WindowRecord],
    dict[str, list[pilot.WindowRecord]],
    dict[str, list[pilot.WindowRecord]],
    dict[str, list[pilot.WindowRecord]],
    list[pilot.WindowRecord],
    dict[str, Any],
]:
    manifest_path = args.manifest.resolve()
    manifest = scale10.validate_manifest(manifest_path)
    selection_path = args.data_root.resolve() / "scale10-selection.json"
    selection = scale10.read_json(selection_path)
    selected_train_entries, eval_entries = scale10.select_entries(
        manifest, {str(item["slug"]) for item in selection["entries"]}
    )
    if selection.get("trainCount") != 10 or len(selected_train_entries) != 10:
        raise ValueError("This continuation requires exactly ten selected train songs")
    (
        train_songs,
        eval_songs,
        train_records,
        eval_records,
        holdout_records,
        _selected_records,
        window_report,
    ) = scale10.prepare_compact_data(
        args.data_root.resolve(),
        args.eval_root.resolve(),
        selected_train_entries,
        eval_entries,
        train_count=args.train_windows_per_song,
        holdout_count=args.holdout_windows_per_song,
        eval_count=args.eval_windows_per_song,
        seed=891,
    )
    if len(train_records) != 320:
        raise ValueError(f"Expected 320 training windows, found {len(train_records)}")
    if any(song.role == "final-test" for song in train_songs + eval_songs):
        raise AssertionError("Official final-test data entered the continuation")
    train_probe_records = matrix.records_by_song_subset(
        train_records, args.train_probe_windows_per_song
    )
    probe_records: list[pilot.WindowRecord] = []
    for records in list(eval_records.values())[: args.probe_song_count]:
        if records:
            probe_records.append(records[0])
    if len(probe_records) != args.probe_song_count:
        raise ValueError("Unable to construct the fixed output probe")
    return (
        train_records,
        train_probe_records,
        eval_records,
        holdout_records,
        probe_records,
        window_report,
    )


def evaluate_milestone(
    *,
    model: torch.nn.Module,
    spec: matrix.VariantSpec,
    seed: int,
    pass_count: int,
    train_probe_records: dict[str, list[pilot.WindowRecord]],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    probe_records: list[pilot.WindowRecord],
    initial_parameters: Sequence[torch.Tensor],
    initial_probe: np.ndarray,
    baseline: dict[str, Any],
    device: torch.device,
    batch_size: int,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    evaluation = {
        "trainFitProbe": matrix.evaluate_matrix_model(
            name=f"W-D-continuation-seed-{seed}@{pass_count}-train",
            model=model,
            spec=spec,
            songs=matrix.unique_songs(train_probe_records),
            records_by_song=train_probe_records,
            device=device,
            batch_size=batch_size,
        ),
        "trainHoldout": matrix.evaluate_matrix_model(
            name=f"W-D-continuation-seed-{seed}@{pass_count}-holdout",
            model=model,
            spec=spec,
            songs=matrix.unique_songs(holdout_records),
            records_by_song=holdout_records,
            device=device,
            batch_size=batch_size,
        ),
        "calibrationInternal": matrix.evaluate_matrix_model(
            name=f"W-D-continuation-seed-{seed}@{pass_count}-eval",
            model=model,
            spec=spec,
            songs=matrix.unique_songs(eval_records),
            records_by_song=eval_records,
            device=device,
            batch_size=batch_size,
        ),
    }
    current_probe = matrix.probe_predictions(model, spec, probe_records, device, batch_size)
    result = {
        "pass": pass_count,
        "evaluation": evaluation,
        "deltasVsPass0": {
            key: matrix.aggressive.metric_delta(value, baseline[key])
            for key, value in evaluation.items()
        },
        "parameterDriftRelative": sweep.parameter_drift_relative(
            model, initial_parameters
        ),
        "outputChangeDb": matrix.relative_output_change_db(initial_probe, current_probe),
        "batchNorm": matrix.batchnorm_summary(model),
        "checkpoint": checkpoint,
    }
    model.train()
    return result


def _milestone_result(
    *,
    current_update: int,
    updates_per_pass: int,
    model: torch.nn.Module,
    spec: matrix.VariantSpec,
    seed: int,
    train_probe_records: dict[str, list[pilot.WindowRecord]],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    probe_records: list[pilot.WindowRecord],
    initial_parameters: Sequence[torch.Tensor],
    initial_probe: np.ndarray,
    baseline: dict[str, Any],
    device: torch.device,
    batch_size: int,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    pass_count = current_update // updates_per_pass
    result = evaluate_milestone(
        model=model,
        spec=spec,
        seed=seed,
        pass_count=pass_count,
        train_probe_records=train_probe_records,
        eval_records=eval_records,
        holdout_records=holdout_records,
        probe_records=probe_records,
        initial_parameters=initial_parameters,
        initial_probe=initial_probe,
        baseline=baseline,
        device=device,
        batch_size=batch_size,
        checkpoint=checkpoint,
    )
    result["update"] = current_update
    aggregate = result["evaluation"]["calibrationInternal"]["aggregate"]
    print(
        json.dumps(
            {
                "variant": "W-D",
                "seed": seed,
                "pass": pass_count,
                "targetSdrDb": aggregate["aggressiveTargetSdrDb"],
                "teacherResidualSdrDb": aggregate["teacherRemovalResidualSdrDb"],
                "instrumentalSdrDb": aggregate["instrumentalSdrDb"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def run_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    milestones: Sequence[int],
    train_records: list[pilot.WindowRecord],
    train_probe_records: dict[str, list[pilot.WindowRecord]],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    probe_records: list[pilot.WindowRecord],
    window_report: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    spec = matrix.VARIANT_SPECS["W-D"]
    updates_per_pass = len(train_records) // args.batch_size
    if updates_per_pass * args.batch_size != len(train_records):
        raise ValueError("Training window count must be divisible by batch size")
    milestone_updates = tuple(pass_count * updates_per_pass for pass_count in milestones)
    total_updates = args.passes * updates_per_pass
    schedule = matrix.build_sample_schedule(
        len(train_records), args.passes, matrix.make_schedule_seed(seed)
    )
    matrix.validate_schedule(schedule, len(train_records), args.passes)

    parent_path = parent_state_path(args.parent_experiment_root.resolve(), seed)
    if not parent_path.is_file():
        raise FileNotFoundError(f"Missing completed W-D parent state: {parent_path}")
    parent_sha256 = pilot.sha256_file(parent_path)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_contract = validate_parent_payload(
        payload=parent_payload,
        parent_path=parent_path,
        seed=seed,
        args=args,
        schedule=schedule,
        updates_per_pass=updates_per_pass,
    )
    run_contract = make_run_contract(
        args=args,
        seed=seed,
        schedule=schedule,
        parent_path=parent_path,
        parent_sha256=parent_sha256,
        parent_contract=parent_contract,
        window_report=window_report,
        train_window_count=len(train_records),
        device=device,
    )
    parent = continuation_parent_metadata(
        parent_path=parent_path,
        parent_sha256=parent_sha256,
        parent_contract=parent_contract,
    )
    run_root = args.experiment_root.resolve() / "runs" / f"seed-{seed}" / "W-D"
    rolling_path = run_root / "training-state.pt"

    # Recreate the true W-D origin before loading the parent. This preserves
    # drift/output-change diagnostics relative to the original direct-head init.
    model, initialization = matrix.make_matrix_model(
        spec, args.checkpoint.resolve(), seed, device
    )
    initial_parameters = [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]
    initial_probe = matrix.probe_predictions(model, spec, probe_records, device, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    history: list[dict[str, float | int]]
    milestone_results: dict[str, Any]
    baseline: dict[str, Any]
    current_update: int
    continuation_elapsed_before: float
    parent_elapsed_seconds: float
    resume_count: int
    resume_path = find_resume_path(run_root, milestone_updates) if args.resume else None
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_continuation_payload(payload=payload, run_contract=run_contract)
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload["history"])
        milestone_results = dict(payload["milestoneResults"])
        baseline = dict(payload["baseline"])
        current_update = int(payload["update"])
        continuation_elapsed_before = float(payload.get("continuationElapsedSeconds", 0.0))
        parent_elapsed_seconds = float(payload.get("parentElapsedSeconds", 0.0))
        resume_count = int(payload.get("resumeCount", 0)) + 1
        density.restore_rng_state(payload["rngState"])
        print(
            json.dumps(
                {
                    "event": "resume-continuation",
                    "seed": seed,
                    "update": current_update,
                    "pass": current_update / updates_per_pass,
                    "file": str(resume_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        model.load_state_dict(parent_payload["state_dict"], strict=True)
        optimizer.load_state_dict(parent_payload["optimizer_state_dict"])
        history = list(parent_payload["history"])
        milestone_results = copy.deepcopy(parent_payload["milestoneResults"])
        baseline = copy.deepcopy(parent_payload["baseline"])
        current_update = int(parent_payload["update"])
        continuation_elapsed_before = 0.0
        parent_elapsed_seconds = float(parent_payload.get("elapsedSeconds", 0.0))
        resume_count = 0
        density.restore_rng_state(parent_payload["rngState"])
        print(
            json.dumps(
                {
                    "event": "start-continuation",
                    "seed": seed,
                    "parent": str(parent_path),
                    "update": current_update,
                    "pass": current_update / updates_per_pass,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if current_update < args.start_passes * updates_per_pass or current_update > total_updates:
        raise ValueError(f"Unexpected continuation update: {current_update}")
    if len(history) != current_update:
        raise ValueError("Continuation history length does not match its update")
    model.train()
    started = time.perf_counter()

    def elapsed() -> float:
        return continuation_elapsed_before + time.perf_counter() - started

    def save(path: Path, state_format: str, status: str) -> dict[str, Any]:
        return density.atomic_torch_save(
            path,
            checkpoint_payload(
                state_format=state_format,
                status=status,
                run_contract=run_contract,
                parent=parent,
                model=model,
                optimizer=optimizer,
                update=current_update,
                history=history,
                milestones=milestone_results,
                baseline=baseline,
                continuation_elapsed_seconds=elapsed(),
                parent_elapsed_seconds=parent_elapsed_seconds,
                resume_count=resume_count,
            ),
        )

    # A checkpoint is written before its relatively expensive evaluation. If a
    # process stops in that interval, resume from the checkpoint and fill in
    # the omitted milestone before advancing the schedule.
    if (
        current_update in milestone_updates
        and current_update > args.start_passes * updates_per_pass
        and str(current_update // updates_per_pass) not in milestone_results
    ):
        checkpoint_path = run_root / f"pass-{current_update}.pt"
        checkpoint = (
            density.file_metadata(checkpoint_path)
            if checkpoint_path.is_file()
            else save(checkpoint_path, CHECKPOINT_FORMAT, "milestone-checkpoint")
        )
        save(rolling_path, STATE_FORMAT, "evaluating")
        milestone_results[str(current_update // updates_per_pass)] = _milestone_result(
            current_update=current_update,
            updates_per_pass=updates_per_pass,
            model=model,
            spec=spec,
            seed=seed,
            train_probe_records=train_probe_records,
            eval_records=eval_records,
            holdout_records=holdout_records,
            probe_records=probe_records,
            initial_parameters=initial_parameters,
            initial_probe=initial_probe,
            baseline=baseline,
            device=device,
            batch_size=args.batch_size,
            checkpoint=checkpoint,
        )
        save(rolling_path, STATE_FORMAT, "training")

    while current_update < total_updates:
        start = current_update * args.batch_size
        indices = schedule[start : start + args.batch_size]
        if len(indices) != args.batch_size:
            raise AssertionError("Incomplete continuation training batch")
        batch = [train_records[int(index)] for index in indices]
        for record in batch:
            record.materialize()
            if record.input_spec is None or record.teacher_instrumental_spec is None:
                raise ValueError(
                    f"Incomplete continuation record: {record.song.slug}:{record.start}"
                )
        inputs = torch.from_numpy(np.stack([record.input_spec for record in batch])).to(device)
        targets = torch.from_numpy(
            np.stack([record.teacher_instrumental_spec for record in batch])
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        candidate = matrix.predict_instrumental(model, inputs, spec)
        loss = F.l1_loss(candidate, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite W-D continuation loss for seed={seed} update={current_update + 1}"
            )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current_update += 1
        history.append(
            {
                "update": current_update,
                "pass": current_update / updates_per_pass,
                "loss": float(loss.detach().cpu().item()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )

        if current_update in milestone_updates and current_update > args.start_passes * updates_per_pass:
            checkpoint_path = run_root / f"pass-{current_update}.pt"
            checkpoint = save(checkpoint_path, CHECKPOINT_FORMAT, "milestone-checkpoint")
            save(rolling_path, STATE_FORMAT, "evaluating")
            milestone_results[str(current_update // updates_per_pass)] = _milestone_result(
                current_update=current_update,
                updates_per_pass=updates_per_pass,
                model=model,
                spec=spec,
                seed=seed,
                train_probe_records=train_probe_records,
                eval_records=eval_records,
                holdout_records=holdout_records,
                probe_records=probe_records,
                initial_parameters=initial_parameters,
                initial_probe=initial_probe,
                baseline=baseline,
                device=device,
                batch_size=args.batch_size,
                checkpoint=checkpoint,
            )
            save(rolling_path, STATE_FORMAT, "training")
        elif current_update % args.state_interval_updates == 0:
            save(rolling_path, STATE_FORMAT, "training")

        if args.stop_after_updates is not None and current_update >= args.stop_after_updates:
            save(rolling_path, STATE_FORMAT, "paused")
            return run_result(
                status="paused",
                seed=seed,
                current_update=current_update,
                updates_per_pass=updates_per_pass,
                total_updates=total_updates,
                initialization=initialization,
                parent=parent,
                milestones=milestone_results,
                history=history,
                continuation_elapsed_seconds=elapsed(),
                parent_elapsed_seconds=parent_elapsed_seconds,
                resume_count=resume_count,
            )

    save(rolling_path, STATE_FORMAT, "completed")
    return run_result(
        status="completed",
        seed=seed,
        current_update=current_update,
        updates_per_pass=updates_per_pass,
        total_updates=total_updates,
        initialization=initialization,
        parent=parent,
        milestones=milestone_results,
        history=history,
        continuation_elapsed_seconds=elapsed(),
        parent_elapsed_seconds=parent_elapsed_seconds,
        resume_count=resume_count,
    )


def run_result(
    *,
    status: str,
    seed: int,
    current_update: int,
    updates_per_pass: int,
    total_updates: int,
    initialization: dict[str, Any],
    parent: dict[str, Any],
    milestones: dict[str, Any],
    history: list[dict[str, float | int]],
    continuation_elapsed_seconds: float,
    parent_elapsed_seconds: float,
    resume_count: int,
) -> dict[str, Any]:
    result = {
        "status": status,
        "variant": "W-D",
        "seed": seed,
        "update": current_update,
        "pass": current_update / updates_per_pass,
        "totalUpdates": total_updates,
        "updatesPerPass": updates_per_pass,
        "initialization": initialization,
        "parent": parent,
        "milestones": milestones,
        "lossByPass": matrix.aggregate_history_by_pass(history, updates_per_pass),
        "continuationElapsedSeconds": continuation_elapsed_seconds,
        "parentElapsedSeconds": parent_elapsed_seconds,
        "totalElapsedSeconds": parent_elapsed_seconds + continuation_elapsed_seconds,
        "resumeCount": resume_count,
    }
    if status == "completed":
        result["finalCheckpoint"] = milestones[str(current_update // updates_per_pass)][
            "checkpoint"
        ]
    return result


def summarize_runs(
    runs: dict[str, dict[str, Any]],
    seeds: Sequence[int],
    milestones: Sequence[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pass_count in milestones:
        per_seed: dict[str, Any] = {}
        values: dict[str, list[float]] = {key: [] for key in matrix.MATRIX_METRIC_KEYS}
        for seed in seeds:
            run = runs.get(str(seed))
            milestone = run.get("milestones", {}).get(str(pass_count)) if run else None
            if milestone is None:
                continue
            metrics = milestone["evaluation"]["calibrationInternal"]["aggregate"]
            per_seed[str(seed)] = {
                key: float(metrics[key]) for key in matrix.MATRIX_METRIC_KEYS
            }
            for key in matrix.MATRIX_METRIC_KEYS:
                values[key].append(float(metrics[key]))
        summary[str(pass_count)] = {
            "seedCount": len(per_seed),
            "perSeed": per_seed,
            "aggregateAcrossSeeds": {
                key: {
                    "mean": float(np.mean(series)),
                    "sampleStdDev": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
                    "minimum": min(series),
                    "maximum": max(series),
                }
                for key, series in values.items()
                if series
            },
        }
    return summary


def make_seed_decision(
    *, run: dict[str, Any], start_passes: int, final_passes: int
) -> dict[str, Any]:
    milestones = run.get("milestones", {})
    start = milestones.get(str(start_passes))
    final = milestones.get(str(final_passes))
    if start is None or final is None:
        return {
            "status": "pending",
            "reason": "Required start/final milestones are unavailable",
            "proceedToRemainingSeeds": False,
        }
    start_eval = start["evaluation"]["calibrationInternal"]["aggregate"]
    final_eval = final["evaluation"]["calibrationInternal"]["aggregate"]
    start_holdout = start["evaluation"]["trainHoldout"]["aggregate"]
    final_holdout = final["evaluation"]["trainHoldout"]["aggregate"]
    target_delta = float(
        final_eval["aggressiveTargetSdrDb"] - start_eval["aggressiveTargetSdrDb"]
    )
    instrumental_delta = float(
        final_eval["instrumentalSdrDb"] - start_eval["instrumentalSdrDb"]
    )
    holdout_target_delta = float(
        final_holdout["aggressiveTargetSdrDb"]
        - start_holdout["aggressiveTargetSdrDb"]
    )
    target_passes = target_delta >= 0.5
    instrumental_passes = instrumental_delta >= -0.1
    holdout_passes = holdout_target_delta >= -0.1
    passed = target_passes and instrumental_passes and holdout_passes
    return {
        "status": "passed" if passed else "failed",
        "criterion": {
            "evaluationAggressiveTargetGainDbAtLeast": 0.5,
            "evaluationInstrumentalSdrDropDbNoMoreThan": 0.1,
            "trainHoldoutAggressiveTargetDropDbNoMoreThan": 0.1,
        },
        "deltas100ToFinalDb": {
            "evaluationAggressiveTargetSdrDb": target_delta,
            "evaluationInstrumentalSdrDb": instrumental_delta,
            "trainHoldoutAggressiveTargetSdrDb": holdout_target_delta,
        },
        "checks": {
            "targetGain": target_passes,
            "instrumentalSafety": instrumental_passes,
            "holdoutSafety": holdout_passes,
        },
        "proceedToRemainingSeeds": passed,
    }


def build_report(
    *,
    args: argparse.Namespace,
    seeds: Sequence[int],
    milestones: Sequence[int],
    window_report: dict[str, Any],
    device: torch.device,
    runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decisions = {
        seed: make_seed_decision(
            run=run, start_passes=args.start_passes, final_passes=args.passes
        )
        for seed, run in runs.items()
    }
    primary = decisions.get(str(DEFAULT_SEEDS[0]))
    return {
        "schemaVersion": 1,
        "status": "completed" if all(run["status"] == "completed" for run in runs.values()) else "paused",
        "experimentId": EXPERIMENT_ID,
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local teacher-derived checkpoints; do not publish",
        },
        "continuation": {
            "variant": "W-D",
            "parentExperimentId": matrix.EXPERIMENT_ID,
            "startPasses": args.start_passes,
            "passes": args.passes,
            "milestonePasses": list(milestones),
            "seeds": list(seeds),
            "trainWindowCount": 320,
            "batchSize": args.batch_size,
            "updatesPerPass": 320 // args.batch_size,
            "learningRate": args.learning_rate,
            "precision": "float32",
            "batchNormPolicy": "continued in train mode from the parent W-D state",
            "finalTestUsed": False,
            "windowSelectionSha256": density.canonical_sha256(window_report),
        },
        "runs": runs,
        "summary": summarize_runs(runs, seeds, milestones),
        "goNoGo": {
            "primarySeed": DEFAULT_SEEDS[0],
            "primarySeedDecision": primary,
            "nextAction": (
                "continue remaining seeds only when primary seed passes"
                if primary is None or primary.get("proceedToRemainingSeeds")
                else "stop after primary seed; do not continue remaining seeds"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torchVersion": torch.__version__,
            "cudaVersion": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": pilot.sha256_file(Path(__file__).resolve()),
            },
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    seeds, milestones = validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    (
        train_records,
        train_probe_records,
        eval_records,
        holdout_records,
        probe_records,
        window_report,
    ) = prepare_data(args)
    report_path = args.experiment_root.resolve() / "reports" / "inst3-wd-continuation-report.json"
    runs: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        result = run_seed(
            args=args,
            seed=seed,
            milestones=milestones,
            train_records=train_records,
            train_probe_records=train_probe_records,
            eval_records=eval_records,
            holdout_records=holdout_records,
            probe_records=probe_records,
            window_report=window_report,
            device=device,
        )
        runs[str(seed)] = result
        report = build_report(
            args=args,
            seeds=seeds,
            milestones=milestones,
            window_report=window_report,
            device=device,
            runs=runs,
        )
        density.atomic_json_write(report_path, report)
        if result["status"] != "completed":
            print(json.dumps({"status": "paused", "report": str(report_path)}, indent=2))
            return 0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = build_report(
        args=args,
        seeds=seeds,
        milestones=milestones,
        window_report=window_report,
        device=device,
        runs=runs,
    )
    density.atomic_json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "primarySeedDecision": report["goNoGo"]["primarySeedDecision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
