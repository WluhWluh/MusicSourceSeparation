#!/usr/bin/env python3
"""Run a controlled stability sweep for the local Inst 3 student pilot.

The preceding pilot used a tiny highest-energy window set and updated BatchNorm
running statistics with a batch size of one. This runner keeps the frozen
song/teacher identities, selects windows by vocal and mixture activity, freezes
BatchNorm running statistics during fine-tuning, and evaluates every eligible
window in the complete calibration/internal-test segments.

This is local, non-commercial research. It does not publish audio, teacher
weights, or student checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import run_inst3_distill_pilot as pilot


DEFAULT_SWEEP_ROOT = pilot.ROOT / "data" / "musdb18-inst3-stability-sweep"
DEFAULT_MILESTONES = (0, 1, 4, 8, 16, 32)
DEFAULT_LEARNING_RATES = (1.0e-6, 3.0e-6, 1.0e-5)


def parse_number_list(raw: str, *, integer: bool = False) -> tuple[float | int, ...]:
    values: list[float | int] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        parsed: float | int = int(value) if integer else float(value)
        values.append(parsed)
    if not values:
        raise ValueError(f"Expected a non-empty comma-separated list: {raw!r}")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=pilot.DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--pilot-root", type=Path, default=pilot.DEFAULT_PILOT_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument(
        "--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE
    )
    parser.add_argument("--start-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument(
        "--train-windows-per-song",
        type=int,
        default=8,
        help="Stratified training windows; use 0 to use every eligible window",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--learning-rates", default=','.join(str(value) for value in DEFAULT_LEARNING_RATES)
    )
    parser.add_argument(
        "--milestones", default=','.join(str(value) for value in DEFAULT_MILESTONES)
    )
    parser.add_argument("--teacher-weight", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--force-decode", action="store_true")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Fail instead of silently falling back when CUDA EP is unavailable",
    )
    return parser.parse_args()


def candidate_windows(song: pilot.SongBundle) -> list[pilot.WindowRecord]:
    useful = pilot.DEFAULT_CONFIG.useful_samples
    candidates: list[pilot.WindowRecord] = []
    for start in range(0, song.mixture_gt.shape[0], useful):
        length = min(useful, song.mixture_gt.shape[0] - start)
        mixture = song.mixture_gt[start : start + length]
        vocals = song.vocals[start : start + length]
        rms = float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2)))
        vocal_rms = float(np.sqrt(np.mean(vocals.astype(np.float64) ** 2)))
        candidates.append(pilot.WindowRecord(song, start, length, rms, vocal_rms))
    if not candidates:
        raise ValueError(f"No eligible windows for {song.slug}")
    expected_start = 0
    covered = 0
    for record in candidates:
        if record.start != expected_start:
            raise ValueError(f"Window gap for {song.slug} at {expected_start}")
        expected_start = record.start + record.length
        covered += record.length
    if covered != song.mixture_gt.shape[0]:
        raise ValueError(
            f"Window coverage mismatch for {song.slug}: {covered} != "
            f"{song.mixture_gt.shape[0]}"
        )
    return candidates


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def select_stratified_windows(
    candidates: list[pilot.WindowRecord], count: int
) -> tuple[list[pilot.WindowRecord], list[pilot.WindowRecord]]:
    """Select vocal/mixture activity quantiles and return disjoint holdout data."""
    candidates = [
        item
        for item in candidates
        if item.length >= pilot.DEFAULT_CONFIG.useful_samples // 3
    ]
    if not candidates:
        raise ValueError("No sufficiently long windows remain for training")
    if count == 0 or count >= len(candidates):
        selected = list(candidates)
    elif count < 1:
        raise ValueError("train window count must be zero or positive")
    else:
        vocal_ranks = normalized_ranks(
            np.asarray([item.vocal_rms for item in candidates], dtype=np.float64)
        )
        mixture_ranks = normalized_ranks(
            np.asarray([item.rms for item in candidates], dtype=np.float64)
        )
        selected_indices: set[int] = set()
        for quantile in np.linspace(0.0, 1.0, count):
            scores = np.abs(vocal_ranks - quantile) + np.abs(
                mixture_ranks - quantile
            )
            for index in np.argsort(scores, kind="stable"):
                if int(index) not in selected_indices:
                    selected_indices.add(int(index))
                    break
        selected = [candidates[index] for index in sorted(selected_indices)]
    selected_ids = {id(item) for item in selected}
    holdout = [item for item in candidates if id(item) not in selected_ids]
    return selected, holdout


def window_summary(records: Iterable[pilot.WindowRecord]) -> dict[str, Any]:
    values = list(records)
    return {
        "count": len(values),
        "starts": [item.start for item in values],
        "vocalRms": [item.vocal_rms for item in values],
        "mixtureRms": [item.rms for item in values],
        "vocalRmsRange": [
            min(item.vocal_rms for item in values) if values else None,
            max(item.vocal_rms for item in values) if values else None,
        ],
        "mixtureRmsRange": [
            min(item.rms for item in values) if values else None,
            max(item.rms for item in values) if values else None,
        ],
    }


def freeze_batchnorm_running_statistics(model: torch.nn.Module) -> None:
    """Keep pretrained BN population statistics while retaining gradients."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_drift_relative(
    model: torch.nn.Module, initial_parameters: list[torch.Tensor]
) -> float:
    delta_power = 0.0
    initial_power = 0.0
    for current, initial in zip(model.parameters(), initial_parameters, strict=True):
        current_cpu = current.detach().float().cpu()
        difference = current_cpu - initial
        delta_power += float(torch.sum(difference * difference).item())
        initial_power += float(torch.sum(initial * initial).item())
    return math.sqrt(delta_power / max(initial_power, 1.0e-30))


def predicted_probe_specs(
    model: torch.nn.Module,
    records: list[pilot.WindowRecord],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for record in records:
            record.materialize()
            if record.input_spec is None:
                raise ValueError("Probe record is not materialized")
            input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
            predicted = input_tensor - model(input_tensor)
            outputs.append(predicted.detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs, axis=0)


def relative_output_change_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    difference_rms = float(np.sqrt(np.mean(difference * difference)))
    return 20.0 * math.log10(
        max(difference_rms, 1.0e-30) / max(reference_rms, 1.0e-30)
    )


def train_trajectory(
    *,
    name: str,
    learning_rate: float,
    checkpoint: Path,
    train_records: list[pilot.WindowRecord],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    baseline_probe: np.ndarray,
    baseline_eval: dict[str, Any],
    baseline_holdout: dict[str, Any],
    device: torch.device,
    steps: int,
    milestones: tuple[int, ...],
    teacher_weight: float,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    model, _ = pilot.make_model(checkpoint, device)
    freeze_batchnorm_running_statistics(model)
    initial_parameters = [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    order = np.random.default_rng(seed).permutation(len(train_records)).tolist()
    history: list[dict[str, float | int]] = []
    milestone_results: dict[str, Any] = {
        "0": {
            "step": 0,
            "evaluation": {
                "calibrationInternal": baseline_eval,
                "trainHoldout": baseline_holdout,
            },
            "parameterDriftRelative": 0.0,
            "outputChangeDb": 0.0,
        }
    }
    started = time.perf_counter()
    for step in range(1, steps + 1):
        freeze_batchnorm_running_statistics(model)
        record = train_records[order[(step - 1) % len(order)]]
        record.materialize()
        if (
            record.input_spec is None
            or record.true_instrumental_spec is None
            or record.teacher_instrumental_spec is None
        ):
            raise ValueError(f"Training record is incomplete: {record.song.slug}:{record.start}")
        input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
        true_tensor = torch.from_numpy(record.true_instrumental_spec[None]).to(device)
        teacher_tensor = torch.from_numpy(record.teacher_instrumental_spec[None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_instrumental = input_tensor - model(input_tensor)
        ground_truth_loss = F.l1_loss(predicted_instrumental, true_tensor)
        teacher_loss = F.l1_loss(predicted_instrumental, teacher_tensor)
        total_loss = ground_truth_loss
        if name == "S1-ground-truth-plus-inst3":
            total_loss = total_loss + teacher_weight * teacher_loss
        total_loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "step": step,
                "loss": float(total_loss.detach().cpu().item()),
                "groundTruthLoss": float(ground_truth_loss.detach().cpu().item()),
                "teacherLoss": float(teacher_loss.detach().cpu().item()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )
        if step in milestones:
            eval_songs = []
            seen: set[str] = set()
            for records in eval_records.values():
                if not records:
                    continue
                song = records[0].song
                if song.slug not in seen:
                    seen.add(song.slug)
                    eval_songs.append(song)
            evaluation = pilot.evaluate_model(
                f"{name}@{step}", model, eval_songs, eval_records, device
            )
            holdout_songs = []
            seen.clear()
            for records in holdout_records.values():
                if not records:
                    continue
                song = records[0].song
                if song.slug not in seen:
                    seen.add(song.slug)
                    holdout_songs.append(song)
            holdout_evaluation = pilot.evaluate_model(
                f"{name}@{step}-holdout",
                model,
                holdout_songs,
                holdout_records,
                device,
            )
            probe = predicted_probe_specs(
                model,
                [record for records in eval_records.values() for record in records],
                device,
            )
            milestone_results[str(step)] = {
                "step": step,
                "evaluation": {
                    "calibrationInternal": evaluation,
                    "trainHoldout": holdout_evaluation,
                },
                "parameterDriftRelative": parameter_drift_relative(
                    model, initial_parameters
                ),
                "outputChangeDb": relative_output_change_db(baseline_probe, probe),
            }
            freeze_batchnorm_running_statistics(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "local-inst3-distill-stability-sweep",
            "variant": name,
            "learningRate": learning_rate,
            "steps": steps,
            "teacherWeight": teacher_weight,
            "batchNormRunningStatistics": "frozen",
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        output_path,
    )
    return {
        "variant": name,
        "learningRate": learning_rate,
        "steps": steps,
        "teacherWeight": teacher_weight,
        "batchNormRunningStatistics": "frozen",
        "windowOrder": order,
        "history": history,
        "milestones": milestone_results,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": {
            "file": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": pilot.sha256_file(output_path),
        },
    }


def unique_songs(records_by_song: dict[str, list[pilot.WindowRecord]]) -> list[pilot.SongBundle]:
    songs: list[pilot.SongBundle] = []
    seen: set[str] = set()
    for records in records_by_song.values():
        if records and records[0].song.slug not in seen:
            seen.add(records[0].song.slug)
            songs.append(records[0].song)
    return songs


def main() -> int:
    args = parse_args()
    if args.start_seconds < 0 or args.duration_seconds <= 0:
        raise ValueError("start must be non-negative and duration must be positive")
    if args.train_windows_per_song < 0:
        raise ValueError("train window count must be zero or positive")
    if args.steps <= 0 or args.threads <= 0 or args.teacher_weight < 0:
        raise ValueError("steps, threads, and teacher weight must be valid")
    learning_rates = tuple(float(value) for value in parse_number_list(args.learning_rates))
    milestones = tuple(int(value) for value in parse_number_list(args.milestones, integer=True))
    if milestones[0] != 0 or any(value < 0 or value > args.steps for value in milestones):
        raise ValueError("milestones must include 0 and stay within --steps")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted and unique")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("learning rates must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = args.manifest.resolve()
    frozen_manifest, selected_entries = pilot.load_frozen_pilot_entries(manifest_path)
    archive_path = args.archive.resolve()
    archive_info = pilot.inspect_archive(
        archive_path, [entry["member"] for entry in selected_entries]
    )
    pilot_root = args.pilot_root.resolve()
    raw_root = pilot_root / "raw"
    decoded_root = pilot_root / "decoded"
    teacher_root = pilot_root / "teacher"
    sweep_root = args.sweep_root.resolve()
    reports_root = sweep_root / "reports"
    runs_root = sweep_root / "runs"
    for path in (raw_root, decoded_root, teacher_root, reports_root, runs_root):
        path.mkdir(parents=True, exist_ok=True)

    source_paths: list[tuple[dict[str, Any], Path]] = []
    for entry in selected_entries:
        source_paths.append(
            (entry, pilot.ensure_raw_song(archive_path, raw_root, entry))
        )
    contract_info = pilot.verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    session, teacher_providers = pilot.make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    bundles = [
        pilot.decode_song(
            source,
            entry,
            decoded_root,
            args.start_seconds,
            args.duration_seconds,
            args.force_decode,
        )
        for entry, source in source_paths
    ]
    teacher_reports = {
        song.slug: pilot.render_teacher(
            song, session, teacher_providers, teacher_root, args.force_teacher
        )
        for song in bundles
    }
    train_songs = [song for song in bundles if song.role == "train"]
    eval_songs = [
        song for song in bundles if song.role in {"calibration", "internal-test"}
    ]
    if len(train_songs) != 2 or len(eval_songs) != 2:
        raise ValueError("Expected two train and two evaluation songs")

    train_records: list[pilot.WindowRecord] = []
    holdout_records: dict[str, list[pilot.WindowRecord]] = {}
    selection_report: dict[str, Any] = {}
    for song in train_songs:
        candidates = candidate_windows(song)
        selected, holdout = select_stratified_windows(
            candidates, args.train_windows_per_song
        )
        if not holdout:
            raise ValueError(
                f"No held-out windows for {song.slug}; lower --train-windows-per-song"
            )
        train_records.extend(selected)
        holdout_records[song.slug] = holdout
        selection_report[song.slug] = {
            "all": window_summary(candidates),
            "train": window_summary(selected),
            "holdout": window_summary(holdout),
        }
    eval_records: dict[str, list[pilot.WindowRecord]] = {}
    for song in eval_songs:
        records = candidate_windows(song)
        eval_records[song.slug] = records
        selection_report[song.slug] = {"fullSegment": window_summary(records)}
    for record in train_records:
        record.materialize()
    for records in list(eval_records.values()) + list(holdout_records.values()):
        for record in records:
            record.materialize()

    baseline_model, checkpoint_metadata = pilot.make_model(
        args.checkpoint.resolve(), device
    )
    baseline_eval = pilot.evaluate_model(
        "initial-checkpoint-full-eval", baseline_model, eval_songs, eval_records, device
    )
    baseline_holdout = pilot.evaluate_model(
        "initial-checkpoint-train-holdout",
        baseline_model,
        train_songs,
        holdout_records,
        device,
    )
    probe_records = [record for records in eval_records.values() for record in records]
    baseline_probe = predicted_probe_specs(baseline_model, probe_records, device)
    teacher_eval = pilot.evaluate_teacher(eval_songs, eval_records)
    teacher_holdout = pilot.evaluate_teacher(train_songs, holdout_records)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectories: dict[str, dict[str, Any]] = {}
    for mode in ("S0-ground-truth", "S1-ground-truth-plus-inst3"):
        trajectories[mode] = {}
        for learning_rate in learning_rates:
            tag = f"{mode.lower()}-lr-{learning_rate:.0e}".replace("+", "plus")
            output_path = runs_root / f"{tag}.pt"
            trajectories[mode][f"{learning_rate:.12g}"] = train_trajectory(
                name=mode,
                learning_rate=learning_rate,
                checkpoint=args.checkpoint.resolve(),
                train_records=train_records,
                eval_records=eval_records,
                holdout_records=holdout_records,
                baseline_probe=baseline_probe,
                baseline_eval=baseline_eval,
                baseline_holdout=baseline_holdout,
                device=device,
                steps=args.steps,
                milestones=milestones,
                teacher_weight=args.teacher_weight,
                seed=args.seed,
                output_path=output_path,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "schemaVersion": 1,
        "experimentId": "inst3-distill-stability-sweep@1",
        "status": "completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local sweep artifacts only; do not publish",
        },
        "manifest": {
            "file": str(manifest_path),
            "sha256": pilot.sha256_file(manifest_path),
            "manifestId": frozen_manifest["manifestId"],
            "selectedEntries": [
                {
                    "member": entry["member"],
                    "role": entry["role"],
                    "sourceSha256": entry["sourceSha256"],
                }
                for entry in selected_entries
            ],
            "finalTestUsed": False,
        },
        "dataContract": {
            "inputSemantic": "mixture-gt",
            "studentOutputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
            "mixtureGtDefinition": "vocals + drums + bass + other",
        },
        "archive": archive_info,
        "contract": contract_info,
        "teacher": {
            "contractId": contract_info["contractId"],
            "providers": teacher_providers,
            "onnxruntimeVersion": pilot.ort.__version__,
            "cudaRequired": args.require_teacher_cuda,
            "parameters": {
                "sampleRate": pilot.TEACHER_PARAMS.sample_rate,
                "nFft": pilot.TEACHER_PARAMS.n_fft,
                "hopLength": pilot.TEACHER_PARAMS.hop_length,
                "dimF": pilot.TEACHER_PARAMS.dim_f,
                "modelTimeFrames": pilot.TEACHER_PARAMS.dim_t,
                "trim": pilot.TEACHER_PARAMS.trim,
                "outputScale": pilot.TEACHER_OUTPUT_SCALE,
            },
            "songs": teacher_reports,
            "fullEvaluation": teacher_eval,
            "trainHoldoutEvaluation": teacher_holdout,
        },
        "student": {
            "checkpoint": checkpoint_metadata["checkpoint"],
            "expectedCheckpointSha256": pilot.EXPECTED_CHECKPOINT_SHA256,
            "device": str(device),
            "cuda": {
                "available": torch.cuda.is_available(),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torchVersion": torch.__version__,
                "cudaVersion": torch.version.cuda,
            },
            "batchNormTrainingPolicy": "running-statistics-frozen; model kept in eval mode during gradient updates",
        },
        "split": {
            "train": [song.slug for song in train_songs],
            "calibration": [song.slug for song in bundles if song.role == "calibration"],
            "internalTest": [song.slug for song in bundles if song.role == "internal-test"],
            "startSeconds": args.start_seconds,
            "durationSeconds": args.duration_seconds,
        },
        "windowSelection": selection_report,
        "training": {
            "seed": args.seed,
            "trainWindowCount": len(train_records),
            "steps": args.steps,
            "milestones": milestones,
            "learningRates": learning_rates,
            "teacherWeight": args.teacher_weight,
            "gradientClipNorm": 1.0,
            "optimizer": "AdamW(weight_decay=0)",
            "trajectories": trajectories,
        },
        "baseline": {
            "calibrationInternal": baseline_eval,
            "trainHoldout": baseline_holdout,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": pilot.sha256_file(Path(__file__).resolve()),
            },
        },
    }
    report_path = reports_root / "inst3-distill-stability-sweep-report.json"
    pilot.json_write(report_path, report)
    summary = {
        "status": report["status"],
        "device": str(device),
        "teacherProviders": teacher_providers,
        "trainWindows": len(train_records),
        "evaluationWindows": sum(len(records) for records in eval_records.values()),
        "milestones": milestones,
        "learningRates": learning_rates,
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
