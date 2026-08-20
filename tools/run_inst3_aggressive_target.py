#!/usr/bin/env python3
"""Train a compact TFC-TDF student toward an aggressive Inst 3 target.

This is intentionally separate from the prior anchored objective-alignment
experiment.  The student still predicts a vocal residual, but its direct
instrumental target is constructed in the audio domain before the student's
STFT:

    target(alpha) = (1 - alpha) * MUSDB18 instrumental + alpha * Inst 3

The experiment is local, non-commercial research.  It does not publish
MUSDB18 audio, teacher outputs, or derived checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import render_inst3_objective_listening as extra_listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-aggressive-target"
DEFAULT_MILESTONES = (0, 128, 512, 2048)
DEFAULT_ALPHAS = (0.5, 0.75, 1.0)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=pilot.DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--pilot-root", type=Path, default=pilot.DEFAULT_PILOT_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument(
        "--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE
    )
    parser.add_argument("--start-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--train-windows-per-song", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS)
    )
    parser.add_argument(
        "--milestones", default=",".join(str(value) for value in DEFAULT_MILESTONES)
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-decode", action="store_true")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-extra-listening", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Fail instead of falling back if ONNX Runtime cannot use CUDA.",
    )
    return parser.parse_args(argv)


def parse_numbers(raw: str, *, integer: bool = False) -> tuple[float | int, ...]:
    values: list[float | int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item) if integer else float(item))
    if not values:
        raise ValueError(f"Expected a non-empty comma-separated list: {raw!r}")
    return tuple(values)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "candidate"


def alpha_name(alpha: float) -> str:
    return f"alpha-{alpha:.2f}"


def record_key(record: pilot.WindowRecord) -> tuple[str, int]:
    return record.song.slug, record.start


class AudioDomainTargetCache:
    """Lazily materialize STFT targets from audio-domain target mixtures."""

    def __init__(self, alpha: float) -> None:
        self.alpha = np.float32(alpha)
        self._audio_by_song: dict[str, np.ndarray] = {}
        self._spec_by_record: dict[tuple[str, int], np.ndarray] = {}

    def audio(self, song: pilot.SongBundle) -> np.ndarray:
        target = self._audio_by_song.get(song.slug)
        if target is None:
            if song.teacher_instrumental is None:
                raise ValueError(f"Inst 3 target is unavailable for {song.slug}")
            target = np.ascontiguousarray(
                (np.float32(1.0) - self.alpha) * song.instrumental
                + self.alpha * song.teacher_instrumental,
                dtype=np.float32,
            )
            self._audio_by_song[song.slug] = target
        return target

    def spectrum(self, record: pilot.WindowRecord) -> np.ndarray:
        key = record_key(record)
        target = self._spec_by_record.get(key)
        if target is None:
            target = np.ascontiguousarray(
                pilot.student_window_spec(
                    self.audio(record.song), record.start, record.length
                ),
                dtype=np.float32,
            )
            self._spec_by_record[key] = target
        return target


def aggressive_metrics(
    mixture: np.ndarray,
    true_vocals: np.ndarray,
    true_instrumental: np.ndarray,
    teacher_instrumental: np.ndarray,
    mixed_target: np.ndarray,
    predicted_instrumental: np.ndarray,
) -> dict[str, float | int]:
    """Report both standard-stem fidelity and aggressive-target similarity."""
    metrics = pilot.separation_metrics(
        mixture,
        true_vocals,
        true_instrumental,
        predicted_instrumental,
        teacher_instrumental,
    )
    target_error = predicted_instrumental - mixed_target
    teacher_residual = mixture - teacher_instrumental
    predicted_residual = mixture - predicted_instrumental
    metrics.update(
        {
            "aggressiveTargetSdrDb": pilot.energy_snr_db(
                mixed_target, predicted_instrumental
            ),
            "aggressiveTargetErrorRmsDbfs": pilot.rms_dbfs(target_error),
            "teacherRemovalResidualSdrDb": pilot.energy_snr_db(
                teacher_residual, predicted_residual
            ),
            "teacherRemovalResidualErrorRmsDbfs": pilot.rms_dbfs(
                predicted_residual - teacher_residual
            ),
        }
    )
    return metrics


def evaluate_model(
    name: str,
    model: torch.nn.Module,
    songs: list[pilot.SongBundle],
    records_by_song: dict[str, list[pilot.WindowRecord]],
    targets: AudioDomainTargetCache,
    device: torch.device,
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    aggregate: list[dict[str, np.ndarray]] = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for song in songs:
            predictions: list[np.ndarray] = []
            mixtures: list[np.ndarray] = []
            true_vocals: list[np.ndarray] = []
            true_instrumentals: list[np.ndarray] = []
            teacher_instrumentals: list[np.ndarray] = []
            mixed_targets: list[np.ndarray] = []
            records = records_by_song[song.slug]
            for record in records:
                record.materialize()
                if record.input_spec is None or song.teacher_instrumental is None:
                    raise ValueError(f"Incomplete evaluation record: {record_key(record)}")
                input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
                predicted_spec = input_tensor - model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reconstructed = pilot.student_istft_centered(
                    predicted_spec.detach().cpu().numpy()
                )
                begin = pilot.DEFAULT_CONFIG.trim_samples
                predictions.append(
                    np.ascontiguousarray(
                        reconstructed[begin : begin + record.length], dtype=np.float32
                    )
                )
                start, end = record.start, record.start + record.length
                mixtures.append(song.mixture_gt[start:end])
                true_vocals.append(song.vocals[start:end])
                true_instrumentals.append(song.instrumental[start:end])
                teacher_instrumentals.append(song.teacher_instrumental[start:end])
                mixed_targets.append(targets.audio(song)[start:end])
            mixture = np.concatenate(mixtures, axis=0)
            vocals = np.concatenate(true_vocals, axis=0)
            instrumental = np.concatenate(true_instrumentals, axis=0)
            teacher = np.concatenate(teacher_instrumentals, axis=0)
            target = np.concatenate(mixed_targets, axis=0)
            predicted = np.concatenate(predictions, axis=0)
            per_song[song.slug] = {
                "role": song.role,
                "windowStarts": [record.start for record in records],
                "windowRms": [record.rms for record in records],
                "windowVocalRms": [record.vocal_rms for record in records],
                "metrics": aggressive_metrics(
                    mixture, vocals, instrumental, teacher, target, predicted
                ),
            }
            aggregate.append(
                {
                    "mixture": mixture,
                    "vocals": vocals,
                    "instrumental": instrumental,
                    "teacher": teacher,
                    "target": target,
                    "predicted": predicted,
                }
            )
    merged = {
        key: np.concatenate([item[key] for item in aggregate], axis=0)
        for key in aggregate[0]
    }
    return {
        "variant": name,
        "outputSemantic": "instrumental",
        "perSong": per_song,
        "aggregate": aggressive_metrics(
            merged["mixture"],
            merged["vocals"],
            merged["instrumental"],
            merged["teacher"],
            merged["target"],
            merged["predicted"],
        ),
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def metric_delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    keys = (
        "aggressiveTargetSdrDb",
        "teacherRemovalResidualSdrDb",
        "instrumentalSdrDb",
        "lowVocalInstrumentalSdrDb",
        "accompanimentVocalProjectionDb",
    )
    return {
        key: float(candidate["aggregate"][key] - baseline["aggregate"][key])
        for key in keys
    }


def unique_songs(
    records_by_song: dict[str, list[pilot.WindowRecord]]
) -> list[pilot.SongBundle]:
    result: list[pilot.SongBundle] = []
    seen: set[str] = set()
    for records in records_by_song.values():
        if records and records[0].song.slug not in seen:
            seen.add(records[0].song.slug)
            result.append(records[0].song)
    return result


def checkpoint_path(root: Path, alpha: float, step: int) -> Path:
    return root / "runs" / f"{alpha_name(alpha)}-step-{step}.pt"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    alpha: float,
    step: int,
    learning_rate: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "local-inst3-aggressive-target",
            "variant": alpha_name(alpha),
            "targetSemantic": "audio-domain mixed instrumental target",
            "alpha": alpha,
            "steps": step,
            "learningRate": learning_rate,
            "batchNormRunningStatistics": "frozen",
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        path,
    )
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": pilot.sha256_file(path),
    }


def load_checkpoint_model(
    initial_checkpoint: Path, payload_path: Path, device: torch.device
) -> torch.nn.Module:
    model, _ = pilot.make_model(initial_checkpoint, device)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing state_dict in {payload_path}")
    model.load_state_dict(state_dict, strict=True)
    sweep.freeze_batchnorm_running_statistics(model)
    return model


def train_alpha(
    *,
    alpha: float,
    initial_checkpoint: Path,
    train_records: list[pilot.WindowRecord],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    all_records: dict[str, list[pilot.WindowRecord]],
    eval_songs: list[pilot.SongBundle],
    all_songs: list[pilot.SongBundle],
    baseline_probe: np.ndarray,
    baseline_eval: dict[str, Any],
    baseline_holdout: dict[str, Any],
    baseline_full: dict[str, Any],
    device: torch.device,
    steps: int,
    learning_rate: float,
    milestones: tuple[int, ...],
    seed: int,
    experiment_root: Path,
) -> dict[str, Any]:
    model, _ = pilot.make_model(initial_checkpoint, device)
    sweep.freeze_batchnorm_running_statistics(model)
    initial_parameters = [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    targets = AudioDomainTargetCache(alpha)
    order = np.random.default_rng(seed).permutation(len(train_records)).tolist()
    history: list[dict[str, float | int]] = []
    milestone_results: dict[str, Any] = {
        "0": {
            "step": 0,
            "evaluation": {
                "calibrationInternal": baseline_eval,
                "trainHoldout": baseline_holdout,
                "allFourSongs": baseline_full,
            },
            "parameterDriftRelative": 0.0,
            "outputChangeDb": 0.0,
            "checkpoint": None,
        }
    }
    started = time.perf_counter()
    for step in range(1, steps + 1):
        sweep.freeze_batchnorm_running_statistics(model)
        record = train_records[order[(step - 1) % len(order)]]
        record.materialize()
        if record.input_spec is None:
            raise ValueError(f"Training input is unavailable: {record_key(record)}")
        input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
        target_tensor = torch.from_numpy(targets.spectrum(record)[None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_instrumental = input_tensor - model(input_tensor)
        target_loss = F.l1_loss(predicted_instrumental, target_tensor)
        target_loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "step": step,
                "loss": float(target_loss.detach().cpu().item()),
                "targetLoss": float(target_loss.detach().cpu().item()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )
        if step in milestones:
            checkpoint = save_checkpoint(
                checkpoint_path(experiment_root, alpha, step),
                model,
                alpha,
                step,
                learning_rate,
            )
            evaluation = evaluate_model(
                f"{alpha_name(alpha)}@{step}",
                model,
                eval_songs,
                eval_records,
                targets,
                device,
            )
            holdout_evaluation = evaluate_model(
                f"{alpha_name(alpha)}@{step}-holdout",
                model,
                unique_songs(holdout_records),
                holdout_records,
                targets,
                device,
            )
            probe = sweep.predicted_probe_specs(
                model,
                [record for records in eval_records.values() for record in records],
                device,
            )
            milestone: dict[str, Any] = {
                "step": step,
                "checkpoint": checkpoint,
                "evaluation": {
                    "calibrationInternal": evaluation,
                    "trainHoldout": holdout_evaluation,
                },
                "deltasVsInitial": {
                    "calibrationInternal": metric_delta(evaluation, baseline_eval),
                    "trainHoldout": metric_delta(holdout_evaluation, baseline_holdout),
                },
                "parameterDriftRelative": sweep.parameter_drift_relative(
                    model, initial_parameters
                ),
                "outputChangeDb": sweep.relative_output_change_db(baseline_probe, probe),
            }
            if step == steps:
                full_evaluation = evaluate_model(
                    f"{alpha_name(alpha)}@{step}-all",
                    model,
                    all_songs,
                    all_records,
                    targets,
                    device,
                )
                milestone["evaluation"]["allFourSongs"] = full_evaluation
                milestone["deltasVsInitial"]["allFourSongs"] = metric_delta(
                    full_evaluation, baseline_full
                )
            milestone_results[str(step)] = milestone
            print(
                json.dumps(
                    {
                        "alpha": alpha,
                        "step": step,
                        "targetLoss": history[-1]["targetLoss"],
                        "targetSdrDb": evaluation["aggregate"]["aggressiveTargetSdrDb"],
                        "teacherResidualSdrDb": evaluation["aggregate"]["teacherRemovalResidualSdrDb"],
                        "outputChangeDb": milestone["outputChangeDb"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    final = milestone_results[str(steps)]
    return {
        "variant": alpha_name(alpha),
        "alpha": alpha,
        "targetSemantic": "audio-domain mixed instrumental target",
        "steps": steps,
        "learningRate": learning_rate,
        "batchNormRunningStatistics": "frozen",
        "windowOrder": order,
        "history": history,
        "milestones": milestone_results,
        "elapsedSeconds": time.perf_counter() - started,
        "finalCheckpoint": final["checkpoint"],
    }


def prepare_extra_segments(
    start_seconds: float, duration_seconds: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    segments: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for name, path in extra_listening.DEFAULT_SONGS.items():
        source, sample_rate = extra_listening.load_audio(path)
        segment = extra_listening.slice_listening_segment(
            source, sample_rate, start_seconds, duration_seconds
        )
        segments[name] = segment
        metadata[name] = {
            "file": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": pilot.sha256_file(path),
            "sourceSampleRate": sample_rate,
            "sourceFrames": int(source.shape[0]),
            "listeningStartSeconds": start_seconds,
            "listeningDurationSeconds": duration_seconds,
            "listeningFrames": int(segment.shape[0]),
            "listeningRawFloat32Sha256": pilot.sha256_array(segment),
        }
    return segments, metadata


def render_extra_model(
    *,
    name: str,
    model: torch.nn.Module,
    segments: dict[str, np.ndarray],
    output_root: Path,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for song, segment in segments.items():
        path = output_root / f"{song}-{safe_name(name)}.flac"
        if path.is_file() and not force:
            report[song] = {
                "file": str(path),
                "skippedExisting": True,
                "bytes": path.stat().st_size,
                "sha256": pilot.sha256_file(path),
            }
            continue
        audio, timing = extra_listening.render_student(segment, model, device)
        item = extra_listening.write_flac(path, audio, pilot.DEFAULT_CONFIG.sample_rate)
        item.update({"candidate": name, **timing})
        report[song] = item
    return report


def render_extra_listening(
    *,
    initial_checkpoint: Path,
    trajectories: dict[str, dict[str, Any]],
    start_seconds: float,
    duration_seconds: float,
    output_root: Path,
    teacher_path: Path,
    teacher_contract: Path,
    teacher_tflite: Path,
    threads: int,
    device: torch.device,
    require_teacher_cuda: bool,
    force: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    segments, sources = prepare_extra_segments(start_seconds, duration_seconds)
    result: dict[str, Any] = {
        "sources": sources,
        "initial": {},
        "teacher": {},
        "candidates": {},
    }
    initial, _ = pilot.make_model(initial_checkpoint, device)
    try:
        result["initial"] = render_extra_model(
            name="initial",
            model=initial,
            segments=segments,
            output_root=output_root,
            device=device,
            force=force,
        )
    finally:
        del initial
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["teacherContract"] = pilot.verify_teacher_contract(
        teacher_contract, teacher_path, teacher_tflite
    )
    session, providers = pilot.make_teacher_session(
        teacher_path, threads, require_teacher_cuda
    )
    try:
        for song, segment in segments.items():
            path = output_root / f"{song}-teacher-inst3.flac"
            if path.is_file() and not force:
                result["teacher"][song] = {
                    "file": str(path),
                    "skippedExisting": True,
                    "bytes": path.stat().st_size,
                    "sha256": pilot.sha256_file(path),
                }
                continue
            audio, timing = extra_listening.render_teacher(segment, session)
            item = extra_listening.write_flac(path, audio, pilot.DEFAULT_CONFIG.sample_rate)
            item.update({"teacher": "UVR-MDX-NET-Inst_3", **timing})
            result["teacher"][song] = item
    finally:
        del session
    result["teacherProviders"] = providers

    for variant, trajectory in trajectories.items():
        for raw_step, milestone in trajectory["milestones"].items():
            if raw_step == "0":
                continue
            checkpoint = Path(milestone["checkpoint"]["file"])
            model = load_checkpoint_model(initial_checkpoint, checkpoint, device)
            try:
                name = f"{variant}-step-{raw_step}"
                result["candidates"][name] = render_extra_model(
                    name=name,
                    model=model,
                    segments=segments,
                    output_root=output_root,
                    device=device,
                    force=force,
                )
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start_seconds < 0 or args.duration_seconds <= 0:
        raise ValueError("start must be non-negative and duration must be positive")
    if args.train_windows_per_song <= 0:
        raise ValueError("train window count must be positive")
    if args.steps <= 0 or args.learning_rate <= 0 or args.threads <= 0:
        raise ValueError("steps, learning rate, and threads must be positive")
    alphas = tuple(float(value) for value in parse_numbers(args.alphas))
    milestones = tuple(int(value) for value in parse_numbers(args.milestones, integer=True))
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be within [0, 1]")
    if tuple(sorted(set(alphas))) != alphas:
        raise ValueError("alphas must be sorted and unique")
    if milestones[0] != 0 or milestones[-1] != args.steps:
        raise ValueError("milestones must start at 0 and end at --steps")
    if tuple(sorted(set(milestones))) != milestones or any(
        step < 0 or step > args.steps for step in milestones
    ):
        raise ValueError("milestones must be sorted, unique, and within --steps")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    initial_checkpoint = args.checkpoint.resolve()
    experiment_root = args.experiment_root.resolve()
    reports_root = experiment_root / "reports"
    runs_root = experiment_root / "runs"
    listening_root = experiment_root / "listening"
    extra_listening_root = experiment_root / "listening-extra"
    for path in (reports_root, runs_root, listening_root, extra_listening_root):
        path.mkdir(parents=True, exist_ok=True)

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
    source_paths = [
        (entry, pilot.ensure_raw_song(archive_path, raw_root, entry))
        for entry in selected_entries
    ]
    contract_info = pilot.verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    session, teacher_providers = pilot.make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    try:
        songs = [
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
            for song in songs
        }
    finally:
        del session

    train_songs = [song for song in songs if song.role == "train"]
    eval_songs = [
        song for song in songs if song.role in {"calibration", "internal-test"}
    ]
    if len(train_songs) != 2 or len(eval_songs) != 2:
        raise ValueError("Expected two train songs plus calibration and internal-test")

    train_records: list[pilot.WindowRecord] = []
    holdout_records: dict[str, list[pilot.WindowRecord]] = {}
    all_records: dict[str, list[pilot.WindowRecord]] = {}
    selection_report: dict[str, Any] = {}
    for song in songs:
        candidates = sweep.candidate_windows(song)
        all_records[song.slug] = candidates
        selection_report[song.slug] = {"all": sweep.window_summary(candidates)}
        if song.role == "train":
            selected, holdout = sweep.select_stratified_windows(
                candidates, args.train_windows_per_song
            )
            if not holdout:
                raise ValueError(f"No held-out windows remain for {song.slug}")
            train_records.extend(selected)
            holdout_records[song.slug] = holdout
            selection_report[song.slug].update(
                {
                    "train": sweep.window_summary(selected),
                    "holdout": sweep.window_summary(holdout),
                }
            )
    eval_records = {song.slug: all_records[song.slug] for song in eval_songs}
    for records in all_records.values():
        for record in records:
            record.materialize()

    initial_model, checkpoint_metadata = pilot.make_model(initial_checkpoint, device)
    baseline_probe = sweep.predicted_probe_specs(
        initial_model,
        [record for records in eval_records.values() for record in records],
        device,
    )
    baselines: dict[str, Any] = {}
    for alpha in alphas:
        targets = AudioDomainTargetCache(alpha)
        baselines[alpha_name(alpha)] = {
            "calibrationInternal": evaluate_model(
                f"initial-{alpha_name(alpha)}",
                initial_model,
                eval_songs,
                eval_records,
                targets,
                device,
            ),
            "trainHoldout": evaluate_model(
                f"initial-{alpha_name(alpha)}-holdout",
                initial_model,
                unique_songs(holdout_records),
                holdout_records,
                targets,
                device,
            ),
            "allFourSongs": evaluate_model(
                f"initial-{alpha_name(alpha)}-all",
                initial_model,
                songs,
                all_records,
                targets,
                device,
            ),
        }
    del initial_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectories: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        key = alpha_name(alpha)
        baseline = baselines[key]
        trajectories[key] = train_alpha(
            alpha=alpha,
            initial_checkpoint=initial_checkpoint,
            train_records=train_records,
            eval_records=eval_records,
            holdout_records=holdout_records,
            all_records=all_records,
            eval_songs=eval_songs,
            all_songs=songs,
            baseline_probe=baseline_probe,
            baseline_eval=baseline["calibrationInternal"],
            baseline_holdout=baseline["trainHoldout"],
            baseline_full=baseline["allFourSongs"],
            device=device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            milestones=milestones,
            seed=args.seed,
            experiment_root=experiment_root,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extra_report: dict[str, Any] | None = None
    if not args.skip_extra_listening:
        extra_report = render_extra_listening(
            initial_checkpoint=initial_checkpoint,
            trajectories=trajectories,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
            output_root=extra_listening_root,
            teacher_path=args.teacher.resolve(),
            teacher_contract=args.contract.resolve(),
            teacher_tflite=args.teacher_tflite.resolve(),
            threads=args.threads,
            device=device,
            require_teacher_cuda=args.require_teacher_cuda,
            force=args.force_listening,
        )

    report = {
        "schemaVersion": 1,
        "experimentId": "inst3-aggressive-target@1",
        "status": "completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local aggressive-target artifacts only; do not publish",
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
        "archive": archive_info,
        "contract": contract_info,
        "split": {
            "train": [song.slug for song in train_songs],
            "calibration": [song.slug for song in songs if song.role == "calibration"],
            "internalTest": [song.slug for song in songs if song.role == "internal-test"],
            "startSeconds": args.start_seconds,
            "durationSeconds": args.duration_seconds,
        },
        "dataContract": {
            "inputSemantic": "mixture-gt",
            "mixtureGtDefinition": "vocals + drums + bass + other",
            "studentOutputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
            "targetDefinition": "(1 - alpha) * MUSDB18 instrumental + alpha * Inst 3 instrumental in audio domain before student STFT",
        },
        "teacher": {
            "contractId": contract_info["contractId"],
            "providers": teacher_providers,
            "onnxruntimeVersion": pilot.ort.__version__,
            "cudaRequired": args.require_teacher_cuda,
            "songs": teacher_reports,
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
        "training": {
            "seed": args.seed,
            "alphas": alphas,
            "trainWindowCount": len(train_records),
            "steps": args.steps,
            "milestones": milestones,
            "learningRate": args.learning_rate,
            "optimizer": "AdamW(weight_decay=0)",
            "gradientClipNorm": 1.0,
            "lossDefinition": "L1 packed student spectrum against STFT(audio-domain mixed target)",
            "trajectories": trajectories,
        },
        "baseline": baselines,
        "windowSelection": selection_report,
        "extraListening": extra_report,
        "notes": {
            "qualityIntent": "Aggressive karaoke/vocal removal; standard instrumental fidelity remains a diagnostic rather than the only quality gate.",
            "finalTestIsolation": "Official final-test songs were not extracted or evaluated.",
            "publication": "Do not publish checkpoints, teacher-derived audio, or MUSDB18 audio.",
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
    report_path = reports_root / "inst3-aggressive-target-report.json"
    pilot.json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": str(device),
                "teacherProviders": teacher_providers,
                "variants": list(trajectories),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
