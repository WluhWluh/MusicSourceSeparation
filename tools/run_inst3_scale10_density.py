#!/usr/bin/env python3
"""Run the resumable scale-10 Inst 3 update-density experiment.

The experiment keeps the scale-10 data, 128-frame student, alpha=1 target,
learning rate, seed, and BatchNorm policy fixed.  It changes only update
density and evaluates milestones at 8192, 20480, and 40960 updates.  Training
state includes AdamW and RNG state so an interrupted run can resume exactly.

After training, the runner renders the complete duration of twelve frozen
private listening songs for the initial model, Inst 3, and every milestone.
All generated weights, reports, and audio remain local ignored artifacts.
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
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import render_inst3_objective_listening as listening
import run_inst3_aggressive_scale10 as scale10
import run_inst3_aggressive_target as aggressive
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "musdb18-inst3-scale10"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-scale10-density"
DEFAULT_MANIFEST = (
    ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
)
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
DEFAULT_MILESTONES = (0, 8192, 20480, 40960)
STATE_FORMAT = "local-inst3-scale10-density-state@1"
CHECKPOINT_FORMAT = "local-inst3-scale10-density-checkpoint@1"
ALPHA = 1.0


PRIVATE_LISTENING_SONGS: tuple[dict[str, str], ...] = (
    {
        "name": "already-gone",
        "file": "already_gone.MP3",
        "sha256": "78c253d59f85ab4349ea8e71db00b9e85ae107d2fc7fd148bcb400e556c4e5b1",
    },
    {
        "name": "chasing-the-wind",
        "file": "chasing_the_wind.mp3",
        "sha256": "764bae88c0e02362274f3ec7a3ed5ecfeb597750df8f5ca8abb1700e6bc2274b",
    },
    {
        "name": "coast-town",
        "file": "coast_town.mp3",
        "sha256": "f66be47fc846459f8ac92543b38dab2019765b556cff98539339bbb44cabcae3",
    },
    {
        "name": "coldplay-tove-lo",
        "file": "fun.mp3",
        "sha256": "039b28cdfe220ff273f1f25d52513976160eb080e8b420c7e0b734256e156b75",
    },
    {
        "name": "i-see-fire",
        "file": "i_see_fire.mp3",
        "sha256": "d7f405cfb1579bd7ef8f5ebdc1d1b8f5175c4f9ef36abd465f68eae1742c7be3",
    },
    {
        "name": "imagine",
        "file": "imagine.mp3",
        "sha256": "1c5232d3e75350a473b1726fffcfd11532d0df63ebb84f4c175c97f55de8671a",
    },
    {
        "name": "lugu-lake",
        "file": "lugu_lake.mp3",
        "sha256": "92f73a1357f6a519b28f4e5963f9f203a578b9cc60c73554025776bf34a58a74",
    },
    {
        "name": "north",
        "file": "north.mp3",
        "sha256": "30e6d919265fc6b88349773095ff4a3a983e5fea0fba59fb8ab83f9644346c5f",
    },
    {
        "name": "odd-future",
        "file": "odd_future.mp3",
        "sha256": "d107aeb25e139079ed28f24e5916901f041fd8284cbba1e527346b8576e79a87",
    },
    {
        "name": "traveling-light",
        "file": "traveling_light.mp3",
        "sha256": "477a80828cd4ecd2681a8a9cd31ae39f1b91b4204fadf6e55546b0c3d91dfa11",
    },
    {
        "name": "unseen-sea",
        "file": "unseen_sea.mp3",
        "sha256": "3bcb2c776877a236d96374df6d5d4a52ae97e106a437890a0214a178fba60f97",
    },
    {
        "name": "yoru-ni-kakeru",
        "file": "yoru_ni_kakeru.flac",
        "sha256": "8ff6b5f84deade9ddc9060ff3702dca80cb3aa304bb06eb99e479fa2d92a424b",
    },
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--train-windows-per-song", type=int, default=32)
    parser.add_argument("--holdout-windows-per-song", type=int, default=16)
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40960)
    parser.add_argument("--milestones", default="0,8192,20480,40960")
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--state-interval", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--skip-full-listening", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--require-teacher-cuda", action="store_true")
    return parser.parse_args(argv)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "song"


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": pilot.sha256_file(path),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "numpy": np.random.get_state(),
        "torchCpu": torch.get_rng_state(),
        "torchCuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torchCpu"])
    if torch.cuda.is_available() and state.get("torchCuda"):
        torch.cuda.set_rng_state_all(state["torchCuda"])


def validate_private_songs(samples_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for frozen in PRIVATE_LISTENING_SONGS:
        name = frozen["name"]
        if name in names:
            raise ValueError(f"Duplicate private listening name: {name}")
        names.add(name)
        path = (samples_root / frozen["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = pilot.sha256_file(path)
        if actual_sha256 != frozen["sha256"]:
            raise ValueError(
                f"Private listening source changed: {name}: "
                f"{actual_sha256} != {frozen['sha256']}"
            )
        result.append(
            {
                "name": name,
                "fileName": frozen["file"],
                "file": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    if len(result) != 12:
        raise AssertionError(f"Expected twelve private songs, found {len(result)}")
    return result


def make_run_contract(
    *,
    args: argparse.Namespace,
    device: torch.device,
    manifest_path: Path,
    selection_path: Path,
    window_report: dict[str, Any],
    train_window_count: int,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    contract = {
        "experimentId": "inst3-scale10-density@1",
        "alpha": ALPHA,
        "initialCheckpointSha256": pilot.sha256_file(args.checkpoint.resolve()),
        "manifestSha256": pilot.sha256_file(manifest_path),
        "selectionSha256": pilot.sha256_file(selection_path),
        "windowSelectionSha256": canonical_sha256(window_report),
        "trainWindowCount": train_window_count,
        "trainWindowsPerSong": args.train_windows_per_song,
        "holdoutWindowsPerSong": args.holdout_windows_per_song,
        "evalWindowsPerSong": args.eval_windows_per_song,
        "steps": args.steps,
        "milestones": list(parse_milestones(args.milestones)),
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "torchVersion": torch.__version__,
        "runnerSha256": pilot.sha256_file(runner_path),
    }
    return {"id": canonical_sha256(contract), "contract": contract}


def parse_milestones(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("milestones must not be empty")
    return values


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    milestones = parse_milestones(args.milestones)
    if milestones[0] != 0 or milestones[-1] != args.steps:
        raise ValueError("milestones must start at 0 and end at --steps")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted and unique")
    if args.train_windows_per_song <= 0:
        raise ValueError("train-windows-per-song must be positive")
    if args.holdout_windows_per_song <= 0 or args.eval_windows_per_song <= 0:
        raise ValueError("evaluation window counts must be positive")
    if args.steps <= 0 or args.learning_rate <= 0 or args.state_interval <= 0:
        raise ValueError("steps, learning-rate, and state-interval must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.stop_after_step is not None and not (0 < args.stop_after_step <= args.steps):
        raise ValueError("stop-after-step must be within (0, steps]")
    return milestones


def training_state_payload(
    *,
    state_format: str,
    run_contract: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    order: list[int],
    history: list[dict[str, float | int]],
    milestone_results: dict[str, Any],
    baseline: dict[str, Any],
    elapsed_seconds: float,
    resume_count: int,
    status: str,
) -> dict[str, Any]:
    return {
        "format": state_format,
        "status": status,
        "runContract": run_contract,
        "step": step,
        "alpha": ALPHA,
        "state_dict": cpu_tree(model.state_dict()),
        "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
        "windowOrder": order,
        "history": history,
        "milestoneResults": milestone_results,
        "baseline": baseline,
        "elapsedSeconds": elapsed_seconds,
        "resumeCount": resume_count,
        "rngState": capture_rng_state(),
    }


def validate_resume_payload(payload: dict[str, Any], run_contract: dict[str, Any]) -> None:
    if payload.get("format") not in {STATE_FORMAT, CHECKPOINT_FORMAT}:
        raise ValueError(f"Unexpected training state format: {payload.get('format')}")
    if payload.get("runContract", {}).get("id") != run_contract["id"]:
        raise ValueError("Training state contract does not match this run")
    if payload.get("alpha") != ALPHA:
        raise ValueError("Training state alpha does not match")


def find_resume_path(experiment_root: Path, milestones: tuple[int, ...]) -> Path | None:
    candidates = [experiment_root / "runs" / "training-state.pt"]
    candidates.extend(
        experiment_root / "runs" / f"alpha-1.00-step-{step}.pt"
        for step in milestones
        if step != 0
    )
    best: tuple[int, float, Path] | None = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            step = int(payload["step"])
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        candidate = (step, path.stat().st_mtime, path)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else None


def file_metadata(path: Path) -> dict[str, Any]:
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": pilot.sha256_file(path),
    }


def evaluate_milestone(
    *,
    step: int,
    steps: int,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    eval_songs: list[pilot.SongBundle],
    train_songs: list[pilot.SongBundle],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    selected_records: dict[str, list[pilot.WindowRecord]],
    train_window_count: int,
    baseline_probe: np.ndarray,
    baseline: dict[str, Any],
    initial_parameters: list[torch.Tensor],
    targets: aggressive.AudioDomainTargetCache,
    device: torch.device,
) -> dict[str, Any]:
    calibration_internal = aggressive.evaluate_model(
        f"alpha-1.00@{step}", model, eval_songs, eval_records, targets, device
    )
    train_holdout = aggressive.evaluate_model(
        f"alpha-1.00@{step}-holdout",
        model,
        train_songs,
        holdout_records,
        targets,
        device,
    )
    probe = sweep.predicted_probe_specs(
        model,
        [record for records in eval_records.values() for record in records],
        device,
    )
    result: dict[str, Any] = {
        "step": step,
        "visitsPerTrainingWindow": step / train_window_count,
        "checkpoint": checkpoint,
        "evaluation": {
            "calibrationInternal": calibration_internal,
            "trainHoldout": train_holdout,
        },
        "deltasVsInitial": {
            "calibrationInternal": aggressive.metric_delta(
                calibration_internal, baseline["calibrationInternal"]
            ),
            "trainHoldout": aggressive.metric_delta(
                train_holdout, baseline["trainHoldout"]
            ),
        },
        "parameterDriftRelative": sweep.parameter_drift_relative(
            model, initial_parameters
        ),
        "outputChangeDb": sweep.relative_output_change_db(baseline_probe, probe),
    }
    if step == steps:
        selected = aggressive.evaluate_model(
            f"alpha-1.00@{step}-selected",
            model,
            train_songs + eval_songs,
            selected_records,
            targets,
            device,
        )
        result["evaluation"]["selectedEvaluation"] = selected
        result["deltasVsInitial"]["selectedEvaluation"] = aggressive.metric_delta(
            selected, baseline["selectedEvaluation"]
        )
    print(
        json.dumps(
            {
                "alpha": ALPHA,
                "step": step,
                "targetSdrDb": calibration_internal["aggregate"]["aggressiveTargetSdrDb"],
                "teacherResidualSdrDb": calibration_internal["aggregate"][
                    "teacherRemovalResidualSdrDb"
                ],
                "instrumentalSdrDb": calibration_internal["aggregate"][
                    "instrumentalSdrDb"
                ],
                "outputChangeDb": result["outputChangeDb"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def train_density(
    *,
    args: argparse.Namespace,
    milestones: tuple[int, ...],
    run_contract: dict[str, Any],
    train_records: list[pilot.WindowRecord],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    selected_records: dict[str, list[pilot.WindowRecord]],
    train_songs: list[pilot.SongBundle],
    eval_songs: list[pilot.SongBundle],
    baseline_probe: np.ndarray,
    baseline: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    experiment_root = args.experiment_root.resolve()
    runs_root = experiment_root / "runs"
    rolling_path = runs_root / "training-state.pt"
    model, _ = pilot.make_model(args.checkpoint.resolve(), device)
    sweep.freeze_batchnorm_running_statistics(model)
    initial_parameters = [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    order = np.random.default_rng(args.seed).permutation(len(train_records)).tolist()
    history: list[dict[str, float | int]] = []
    milestone_results: dict[str, Any] = {
        "0": {
            "step": 0,
            "visitsPerTrainingWindow": 0.0,
            "evaluation": baseline,
            "parameterDriftRelative": 0.0,
            "outputChangeDb": 0.0,
            "checkpoint": None,
        }
    }
    current_step = 0
    elapsed_before = 0.0
    resume_count = 0

    resume_path = find_resume_path(experiment_root, milestones) if args.resume else None
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_payload(payload, run_contract)
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload["windowOrder"] != order:
            raise ValueError("Training window order changed")
        history = list(payload["history"])
        milestone_results = dict(payload["milestoneResults"])
        current_step = int(payload["step"])
        elapsed_before = float(payload.get("elapsedSeconds", 0.0))
        resume_count = int(payload.get("resumeCount", 0)) + 1
        restore_rng_state(payload["rngState"])
        print(
            json.dumps(
                {
                    "event": "resume",
                    "file": str(resume_path),
                    "step": current_step,
                    "resumeCount": resume_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    targets = aggressive.AudioDomainTargetCache(ALPHA)
    started = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def save_state(path: Path, state_format: str, status: str) -> dict[str, Any]:
        payload = training_state_payload(
            state_format=state_format,
            run_contract=run_contract,
            model=model,
            optimizer=optimizer,
            step=current_step,
            order=order,
            history=history,
            milestone_results=milestone_results,
            baseline=baseline,
            elapsed_seconds=elapsed(),
            resume_count=resume_count,
            status=status,
        )
        return atomic_torch_save(path, payload)

    def ensure_current_milestone() -> None:
        nonlocal milestone_results
        if current_step == 0 or current_step not in milestones:
            return
        if str(current_step) in milestone_results:
            return
        checkpoint_path = runs_root / f"alpha-1.00-step-{current_step}.pt"
        if checkpoint_path.is_file():
            checkpoint = file_metadata(checkpoint_path)
        else:
            checkpoint = save_state(
                checkpoint_path, CHECKPOINT_FORMAT, "milestone-checkpoint"
            )
        save_state(rolling_path, STATE_FORMAT, "evaluating")
        milestone_results[str(current_step)] = evaluate_milestone(
            step=current_step,
            steps=args.steps,
            model=model,
            checkpoint=checkpoint,
            eval_songs=eval_songs,
            train_songs=train_songs,
            eval_records=eval_records,
            holdout_records=holdout_records,
            selected_records=selected_records,
            train_window_count=len(train_records),
            baseline_probe=baseline_probe,
            baseline=baseline,
            initial_parameters=initial_parameters,
            targets=targets,
            device=device,
        )
        save_state(rolling_path, STATE_FORMAT, "training")

    ensure_current_milestone()
    while current_step < args.steps:
        sweep.freeze_batchnorm_running_statistics(model)
        record = train_records[order[current_step % len(order)]]
        record.materialize()
        if record.input_spec is None:
            raise ValueError(f"Training input is unavailable: {aggressive.record_key(record)}")
        input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
        target_tensor = torch.from_numpy(targets.spectrum(record)[None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_instrumental = input_tensor - model(input_tensor)
        loss = F.l1_loss(predicted_instrumental, target_tensor)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current_step += 1
        history.append(
            {
                "step": current_step,
                "targetLoss": float(loss.detach().cpu().item()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )

        if current_step in milestones:
            checkpoint_path = runs_root / f"alpha-1.00-step-{current_step}.pt"
            checkpoint = save_state(
                checkpoint_path, CHECKPOINT_FORMAT, "milestone-checkpoint"
            )
            save_state(rolling_path, STATE_FORMAT, "evaluating")
            milestone_results[str(current_step)] = evaluate_milestone(
                step=current_step,
                steps=args.steps,
                model=model,
                checkpoint=checkpoint,
                eval_songs=eval_songs,
                train_songs=train_songs,
                eval_records=eval_records,
                holdout_records=holdout_records,
                selected_records=selected_records,
                train_window_count=len(train_records),
                baseline_probe=baseline_probe,
                baseline=baseline,
                initial_parameters=initial_parameters,
                targets=targets,
                device=device,
            )
            save_state(rolling_path, STATE_FORMAT, "training")
        elif current_step % args.state_interval == 0:
            save_state(rolling_path, STATE_FORMAT, "training")

        if args.stop_after_step is not None and current_step >= args.stop_after_step:
            save_state(rolling_path, STATE_FORMAT, "paused")
            return {
                "status": "paused",
                "variant": "alpha-1.00",
                "step": current_step,
                "steps": args.steps,
                "learningRate": args.learning_rate,
                "windowOrder": order,
                "history": history,
                "milestones": milestone_results,
                "elapsedSeconds": elapsed(),
                "resumeCount": resume_count,
            }

    save_state(rolling_path, STATE_FORMAT, "completed")
    return {
        "status": "completed",
        "variant": "alpha-1.00",
        "alpha": ALPHA,
        "step": current_step,
        "steps": args.steps,
        "learningRate": args.learning_rate,
        "batchNormRunningStatistics": "frozen",
        "windowOrder": order,
        "history": history,
        "milestones": milestone_results,
        "elapsedSeconds": elapsed(),
        "resumeCount": resume_count,
        "finalCheckpoint": milestone_results[str(args.steps)]["checkpoint"],
    }


def linear_dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1.0e-12))


def percentile_dbfs(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return -240.0
    return linear_dbfs(float(np.percentile(values, percentile)))


def local_teacher_error_metrics(
    source: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    if source.shape != teacher.shape or source.shape != candidate.shape:
        raise ValueError("Listening diagnostic shapes do not match")
    teacher_removed = source - teacher
    candidate_removed = source - candidate
    error = candidate - teacher
    result: dict[str, Any] = {
        "teacherRemovalResidualSdrDb": pilot.energy_snr_db(
            teacher_removed, candidate_removed
        ),
        "candidateToTeacherInstrumentalSdrDb": pilot.energy_snr_db(
            teacher, candidate
        ),
        "errorRmsDbfs": pilot.rms_dbfs(error),
        "blockMetrics": {},
    }
    active_threshold = 10.0 ** (-60.0 / 20.0)
    for milliseconds in (50, 100, 200):
        block_samples = max(1, round(sample_rate * milliseconds / 1000.0))
        block_count = source.shape[0] // block_samples
        if block_count == 0:
            continue
        usable = block_count * block_samples
        error_blocks = error[:usable].reshape(block_count, block_samples, 2)
        removed_blocks = teacher_removed[:usable].reshape(
            block_count, block_samples, 2
        )
        error_rms = np.sqrt(np.mean(error_blocks.astype(np.float64) ** 2, axis=(1, 2)))
        removed_power = np.sum(
            removed_blocks.astype(np.float64) ** 2, axis=(1, 2)
        )
        removed_rms = np.sqrt(
            removed_power / float(block_samples * removed_blocks.shape[2])
        )
        dots = np.sum(
            error_blocks.astype(np.float64) * removed_blocks.astype(np.float64),
            axis=(1, 2),
        )
        coefficients = np.divide(
            dots,
            removed_power,
            out=np.zeros_like(dots),
            where=removed_power > 1.0e-20,
        )
        retained_rms = np.maximum(coefficients, 0.0) * removed_rms
        active = removed_rms >= active_threshold
        active_retained = retained_rms[active]
        ranking = np.argsort(retained_rms, kind="stable")[::-1]
        hotspots: list[dict[str, Any]] = []
        for index in ranking[:12]:
            hotspots.append(
                {
                    "startSeconds": index * block_samples / sample_rate,
                    "endSeconds": (index + 1) * block_samples / sample_rate,
                    "errorRmsDbfs": linear_dbfs(float(error_rms[index])),
                    "teacherRemovedRmsDbfs": linear_dbfs(float(removed_rms[index])),
                    "retainedTeacherResidualRmsDbfs": linear_dbfs(
                        float(retained_rms[index])
                    ),
                    "projectionCoefficient": float(coefficients[index]),
                }
            )
        result["blockMetrics"][str(milliseconds)] = {
            "blockMilliseconds": milliseconds,
            "blockCount": block_count,
            "activeBlockCount": int(np.count_nonzero(active)),
            "errorRmsP95Dbfs": percentile_dbfs(error_rms, 95.0),
            "errorRmsMaxDbfs": linear_dbfs(float(np.max(error_rms))),
            "retainedTeacherResidualP95Dbfs": percentile_dbfs(
                active_retained, 95.0
            ),
            "retainedTeacherResidualMaxDbfs": linear_dbfs(
                float(np.max(active_retained)) if active_retained.size else 0.0
            ),
            "hotspots": hotspots,
        }
    return result


def render_or_load(
    *,
    output_path: Path,
    render: Callable[[], tuple[np.ndarray, dict[str, Any]]],
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    timing: dict[str, Any] = {}
    if force or not output_path.is_file():
        audio, timing = render()
        listening.write_flac(output_path, audio, pilot.DEFAULT_CONFIG.sample_rate)
    decoded, sample_rate = listening.load_audio(output_path)
    if sample_rate != pilot.DEFAULT_CONFIG.sample_rate:
        raise ValueError(f"Unexpected rendered sample rate: {output_path}")
    metadata = {
        "file": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": pilot.sha256_file(output_path),
        "frames": int(decoded.shape[0]),
        "sampleRate": sample_rate,
        "channels": int(decoded.shape[1]),
        "skippedExisting": not force and not timing,
        **timing,
    }
    return decoded, metadata


def render_full_listening(
    *,
    args: argparse.Namespace,
    trajectory: dict[str, Any],
    device: torch.device,
    frozen_songs: list[dict[str, Any]],
) -> dict[str, Any]:
    experiment_root = args.experiment_root.resolve()
    output_root = experiment_root / "listening-full"
    progress_path = experiment_root / "reports" / "full-listening-report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    contract = pilot.verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    session, providers = pilot.make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    models: dict[str, torch.nn.Module] = {}
    initial, _ = pilot.make_model(args.checkpoint.resolve(), device)
    models["initial"] = initial
    for step in parse_milestones(args.milestones):
        if step == 0:
            continue
        checkpoint = Path(trajectory["milestones"][str(step)]["checkpoint"]["file"])
        models[f"alpha-1.00-step-{step}"] = aggressive.load_checkpoint_model(
            args.checkpoint.resolve(), checkpoint, device
        )

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "songCount": len(frozen_songs),
        "outputCountExpected": len(frozen_songs) * (len(models) + 1),
        "teacherContract": contract,
        "teacherProviders": providers,
        "songs": {},
    }
    try:
        for index, frozen in enumerate(frozen_songs, start=1):
            name = frozen["name"]
            print(f"full listening {index}/{len(frozen_songs)}: {name}", flush=True)
            source_path = Path(frozen["file"])
            source, sample_rate = listening.load_audio(source_path)
            if sample_rate != pilot.DEFAULT_CONFIG.sample_rate:
                raise ValueError(f"Unexpected private source sample rate: {source_path}")
            song_root = output_root / name
            song_root.mkdir(parents=True, exist_ok=True)
            teacher_audio, teacher_metadata = render_or_load(
                output_path=song_root / "teacher-inst3.flac",
                render=lambda source=source: listening.render_teacher(source, session),
                force=args.force_listening,
            )
            candidates: dict[str, Any] = {}
            for candidate_name, model in models.items():
                candidate_audio, candidate_metadata = render_or_load(
                    output_path=song_root / f"{candidate_name}.flac",
                    render=lambda source=source, model=model: listening.render_student(
                        source, model, device
                    ),
                    force=args.force_listening,
                )
                candidate_metadata["diagnosticsVsTeacher"] = local_teacher_error_metrics(
                    source, teacher_audio, candidate_audio, sample_rate
                )
                candidates[candidate_name] = candidate_metadata
                del candidate_audio
            report["songs"][name] = {
                "source": {
                    **frozen,
                    "sampleRate": sample_rate,
                    "frames": int(source.shape[0]),
                    "durationSeconds": source.shape[0] / sample_rate,
                    "decodedFloat32Sha256": pilot.sha256_array(source),
                },
                "teacher": teacher_metadata,
                "candidates": candidates,
            }
            atomic_json_write(progress_path, report)
            del source, teacher_audio
            gc.collect()
    finally:
        del session
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "completed"
    report["outputCount"] = sum(
        1 + len(song["candidates"]) for song in report["songs"].values()
    )
    atomic_json_write(progress_path, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_path = args.manifest.resolve()
    manifest = scale10.validate_manifest(manifest_path)
    data_root = args.data_root.resolve()
    eval_root = args.eval_root.resolve()
    selection_path = data_root / "scale10-selection.json"
    selection = scale10.read_json(selection_path)
    selected_train_entries, eval_entries = scale10.select_entries(
        manifest, {str(item["slug"]) for item in selection["entries"]}
    )
    if selection.get("trainCount") != 10 or len(selected_train_entries) != 10:
        raise ValueError("This run requires exactly ten selected train songs")

    (
        train_songs,
        eval_songs,
        train_records,
        eval_records,
        holdout_records,
        selected_records,
        window_report,
    ) = scale10.prepare_compact_data(
        data_root,
        eval_root,
        selected_train_entries,
        eval_entries,
        train_count=args.train_windows_per_song,
        holdout_count=args.holdout_windows_per_song,
        eval_count=args.eval_windows_per_song,
        seed=args.seed,
    )
    if len(train_records) != 320:
        raise ValueError(f"Expected 320 training windows, found {len(train_records)}")
    if any(song.role == "final-test" for song in train_songs + eval_songs):
        raise AssertionError("final-test data entered the density run")

    run_contract = make_run_contract(
        args=args,
        device=device,
        manifest_path=manifest_path,
        selection_path=selection_path,
        window_report=window_report,
        train_window_count=len(train_records),
    )
    resume_path = (
        find_resume_path(args.experiment_root.resolve(), milestones)
        if args.resume
        else None
    )
    resumed_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resumed_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_payload(resumed_payload, run_contract)

    initial_model, checkpoint_metadata = pilot.make_model(args.checkpoint.resolve(), device)
    baseline_probe = sweep.predicted_probe_specs(
        initial_model,
        [record for records in eval_records.values() for record in records],
        device,
    )
    if resumed_payload is not None:
        baseline = resumed_payload["baseline"]
    else:
        print("baseline evaluation: alpha=1.00", flush=True)
        targets = aggressive.AudioDomainTargetCache(ALPHA)
        baseline = {
            "calibrationInternal": aggressive.evaluate_model(
                "initial-alpha-1.00",
                initial_model,
                eval_songs,
                eval_records,
                targets,
                device,
            ),
            "trainHoldout": aggressive.evaluate_model(
                "initial-alpha-1.00-holdout",
                initial_model,
                train_songs,
                holdout_records,
                targets,
                device,
            ),
            "selectedEvaluation": aggressive.evaluate_model(
                "initial-alpha-1.00-selected",
                initial_model,
                train_songs + eval_songs,
                selected_records,
                targets,
                device,
            ),
        }
        del targets
    del initial_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectory = train_density(
        args=args,
        milestones=milestones,
        run_contract=run_contract,
        train_records=train_records,
        eval_records=eval_records,
        holdout_records=holdout_records,
        selected_records=selected_records,
        train_songs=train_songs,
        eval_songs=eval_songs,
        baseline_probe=baseline_probe,
        baseline=baseline,
        device=device,
    )

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "status": trajectory["status"],
        "experimentId": "inst3-scale10-density@1",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "privateListening": "local user-provided audio; not redistributed",
            "derivedWeights": "local aggressive-target artifacts only; do not publish",
        },
        "runContract": run_contract,
        "manifest": {
            "file": str(manifest_path),
            "sha256": pilot.sha256_file(manifest_path),
            "manifestId": manifest["manifestId"],
            "selectionFile": str(selection_path),
            "selectedTrain": [entry["member"] for entry in selected_train_entries],
            "calibration": [
                entry["member"] for entry in eval_entries if entry["role"] == "calibration"
            ],
            "internalTest": [
                entry["member"]
                for entry in eval_entries
                if entry["role"] == "internal-test"
            ],
            "finalTestUsed": False,
        },
        "student": {
            "checkpoint": checkpoint_metadata["checkpoint"],
            "device": str(device),
            "cuda": {
                "available": torch.cuda.is_available(),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torchVersion": torch.__version__,
                "cudaVersion": torch.version.cuda,
            },
            "batchNormTrainingPolicy": (
                "running-statistics-frozen; model kept in eval mode during updates"
            ),
        },
        "training": {
            "alpha": ALPHA,
            "seed": args.seed,
            "trainSongCount": len(train_songs),
            "calibrationSongCount": sum(song.role == "calibration" for song in eval_songs),
            "internalTestSongCount": sum(
                song.role == "internal-test" for song in eval_songs
            ),
            "officialFinalTestUsed": False,
            "trainWindowCount": len(train_records),
            "steps": args.steps,
            "milestones": milestones,
            "visitsPerTrainingWindow": {
                str(step): step / len(train_records) for step in milestones
            },
            "learningRate": args.learning_rate,
            "optimizer": "AdamW(weight_decay=0), state persisted",
            "gradientClipNorm": 1.0,
            "lossDefinition": (
                "L1 packed student spectrum against direct Inst 3 audio-domain target"
            ),
            "trajectory": trajectory,
        },
        "baseline": baseline,
        "windowSelection": window_report,
        "fullListening": None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": pilot.sha256_file(Path(__file__).resolve()),
            },
        },
    }
    report_path = (
        args.experiment_root.resolve()
        / "reports"
        / "inst3-scale10-density-report.json"
    )
    atomic_json_write(report_path, report)
    if trajectory["status"] != "completed":
        print(
            json.dumps(
                {
                    "status": trajectory["status"],
                    "step": trajectory["step"],
                    "report": str(report_path),
                },
                indent=2,
            )
        )
        return 0

    frozen_songs = validate_private_songs(args.samples_root.resolve())
    if not args.skip_full_listening:
        del train_songs, eval_songs, train_records, eval_records
        del holdout_records, selected_records, baseline_probe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report["fullListening"] = render_full_listening(
            args=args,
            trajectory=trajectory,
            device=device,
            frozen_songs=frozen_songs,
        )
    else:
        report["fullListening"] = {
            "status": "skipped",
            "frozenSongs": frozen_songs,
        }
    report["status"] = "completed"
    atomic_json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "step": trajectory["step"],
                "fullListening": report["fullListening"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
