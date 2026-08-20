#!/usr/bin/env python3
"""Run the next controlled Inst 3 objective-alignment experiment.

This runner keeps the frozen four-song pilot split and compares three controls
with three anchored Inst 3 distillation variants:

    S0-ground-truth
    S0-anchor
    S0-vocal-projection
    S1-anchor-inst3-0.01
    S1-anchor-inst3-0.03
    S1-anchor-inst3-0.10

The student predicts a vocal residual and exposes ``mixture - residual`` as
its instrumental output. Anchor and teacher losses are packed-spectrum L1
losses. The vocal-projection control uses a differentiable audio-domain
projection of instrumental error onto the reference vocal stem, and is only
applied to activity-qualified windows.

This is local, non-commercial research. It does not publish audio, teacher
weights, or derived student checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import run_inst3_distill_stability_sweep as sweep


pilot = sweep.pilot
ROOT = pilot.ROOT
DEFAULT_ALIGNMENT_ROOT = ROOT / "data" / "musdb18-inst3-objective-alignment"
DEFAULT_MILESTONES = (0, 1, 4, 8, 16, 32)
DEFAULT_TEACHER_WEIGHTS = (0.01, 0.03, 0.10)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    anchor_weight: float
    vocal_projection_weight: float
    teacher_weight: float


def parse_number_list(raw: str, *, integer: bool = False) -> tuple[float | int, ...]:
    values: list[float | int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item) if integer else float(item))
    if not values:
        raise ValueError(f"Expected a non-empty comma-separated list: {raw!r}")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=pilot.DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--pilot-root", type=Path, default=pilot.DEFAULT_PILOT_ROOT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument(
        "--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE
    )
    parser.add_argument("--start-seconds", type=float, default=15.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--train-windows-per-song", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument(
        "--vocal-projection-weight", type=float, default=0.10
    )
    parser.add_argument(
        "--teacher-weights",
        default=",".join(str(value) for value in DEFAULT_TEACHER_WEIGHTS),
    )
    parser.add_argument(
        "--milestones",
        default=",".join(str(value) for value in DEFAULT_MILESTONES),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--force-decode", action="store_true")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action="store_true",
        help="Fail instead of silently falling back when CUDA EP is unavailable",
    )
    parser.add_argument(
        "--skip-listening",
        action="store_true",
        help="Do not write ignored FLAC listening artifacts",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def record_key(record: pilot.WindowRecord) -> tuple[str, int]:
    return record.song.slug, record.start


def torch_packed_istft(
    packed: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Convert packed [B, 4, F, T] spectra to [B, samples, 2] audio."""
    if packed.ndim != 4 or packed.shape[1] != 4:
        raise ValueError(f"Unexpected packed spectrum shape: {tuple(packed.shape)}")
    left = torch.complex(packed[:, 0], packed[:, 2])
    right = torch.complex(packed[:, 1], packed[:, 3])
    channels = []
    for spectrum in (left, right):
        channels.append(
            torch.istft(
                spectrum,
                n_fft=pilot.DEFAULT_CONFIG.n_fft,
                hop_length=pilot.DEFAULT_CONFIG.hop_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=pilot.DEFAULT_CONFIG.model_input_samples,
                return_complex=False,
            )
        )
    return torch.stack(channels, dim=-1)


def vocal_projection_loss(
    predicted_instrumental: torch.Tensor,
    record: pilot.WindowRecord,
    window: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Penalize the audio-domain component of error aligned with vocals."""
    predicted_wave = torch_packed_istft(predicted_instrumental, window)[0]
    begin = pilot.DEFAULT_CONFIG.trim_samples
    end = begin + record.length
    predicted = predicted_wave[begin:end]
    true_instrumental = torch.from_numpy(
        record.song.instrumental[record.start : record.start + record.length]
    ).to(device)
    vocals = torch.from_numpy(
        record.song.vocals[record.start : record.start + record.length]
    ).to(device)
    error = (predicted - true_instrumental).reshape(-1)
    vocal = vocals.reshape(-1)
    vocal_power = torch.sum(vocal * vocal)
    coefficient = torch.sum(error * vocal) / (vocal_power + 1.0e-12)
    return coefficient * coefficient


def activity_threshold(records: list[pilot.WindowRecord]) -> float:
    values = np.asarray([record.vocal_rms for record in records], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot calculate vocal activity threshold without windows")
    return max(1.0e-5, float(np.percentile(values, 25.0)))


def make_variants(
    anchor_weight: float,
    vocal_weight: float,
    teacher_weights: tuple[float, ...],
) -> list[VariantSpec]:
    variants = [
        VariantSpec("S0-ground-truth", 0.0, 0.0, 0.0),
        VariantSpec("S0-anchor", anchor_weight, 0.0, 0.0),
        VariantSpec("S0-vocal-projection", 0.0, vocal_weight, 0.0),
    ]
    variants.extend(
        VariantSpec(
            f"S1-anchor-inst3-{weight:g}",
            anchor_weight,
            0.0,
            weight,
        )
        for weight in teacher_weights
    )
    return variants


def build_anchor_targets(
    model: torch.nn.Module,
    records: list[pilot.WindowRecord],
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    model.eval()
    targets: dict[tuple[str, int], np.ndarray] = {}
    with torch.inference_mode():
        for record in records:
            record.materialize()
            if record.input_spec is None:
                raise ValueError(f"Missing input spectrum for {record_key(record)}")
            input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
            predicted = input_tensor - model(input_tensor)
            targets[record_key(record)] = predicted.detach().cpu().numpy()[0]
    return targets


def initial_component_losses(
    model: torch.nn.Module,
    records: list[pilot.WindowRecord],
    anchor_targets: dict[tuple[str, int], np.ndarray],
    device: torch.device,
    active_threshold: float,
    teacher_window: torch.Tensor,
) -> dict[str, float | int]:
    model.eval()
    totals = {"groundTruth": 0.0, "anchor": 0.0, "teacher": 0.0, "vocalProjection": 0.0}
    counts = {"all": 0, "activeVocal": 0}
    with torch.inference_mode():
        for record in records:
            record.materialize()
            if (
                record.input_spec is None
                or record.true_instrumental_spec is None
                or record.teacher_instrumental_spec is None
            ):
                raise ValueError(f"Incomplete record for {record_key(record)}")
            input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
            predicted = input_tensor - model(input_tensor)
            true_tensor = torch.from_numpy(record.true_instrumental_spec[None]).to(device)
            anchor_tensor = torch.from_numpy(
                anchor_targets[record_key(record)][None]
            ).to(device)
            teacher_tensor = torch.from_numpy(record.teacher_instrumental_spec[None]).to(device)
            totals["groundTruth"] += float(F.l1_loss(predicted, true_tensor).item())
            totals["anchor"] += float(F.l1_loss(predicted, anchor_tensor).item())
            totals["teacher"] += float(F.l1_loss(predicted, teacher_tensor).item())
            if record.vocal_rms >= active_threshold:
                totals["vocalProjection"] += float(
                    vocal_projection_loss(predicted, record, teacher_window, device).item()
                )
                counts["activeVocal"] += 1
            counts["all"] += 1
    divisor = max(counts["all"], 1)
    active_divisor = max(counts["activeVocal"], 1)
    return {
        "windowCount": counts["all"],
        "activeVocalWindowCount": counts["activeVocal"],
        "groundTruth": totals["groundTruth"] / divisor,
        "anchor": totals["anchor"] / divisor,
        "teacher": totals["teacher"] / divisor,
        "vocalProjection": totals["vocalProjection"] / active_divisor,
    }


def parameter_drift_relative(
    model: torch.nn.Module,
    initial_parameters: list[torch.Tensor],
) -> float:
    delta_power = 0.0
    initial_power = 0.0
    for current, initial in zip(model.parameters(), initial_parameters, strict=True):
        current_cpu = current.detach().float().cpu()
        difference = current_cpu - initial
        delta_power += float(torch.sum(difference * difference).item())
        initial_power += float(torch.sum(initial * initial).item())
    return math.sqrt(delta_power / max(initial_power, 1.0e-30))


def render_model_song(
    model: torch.nn.Module,
    song: pilot.SongBundle,
    records: list[pilot.WindowRecord],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for record in records:
            record.materialize()
            if record.input_spec is None:
                raise ValueError(f"Missing input spectrum for {record_key(record)}")
            input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
            predicted = input_tensor - model(input_tensor)
            wave = pilot.student_istft_centered(predicted.detach().cpu().numpy())
            begin = pilot.DEFAULT_CONFIG.trim_samples
            pieces.append(
                np.ascontiguousarray(
                    wave[begin : begin + record.length], dtype=np.float32
                )
            )
    output = np.concatenate(pieces, axis=0)
    if output.shape[0] != song.mixture_gt.shape[0]:
        raise ValueError(
            f"Rendered length mismatch for {song.slug}: {output.shape[0]} != "
            f"{song.mixture_gt.shape[0]}"
        )
    return output


def write_listening_file(path: Path, audio: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        pilot.DEFAULT_CONFIG.sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": pilot.sha256_file(path),
        "rawFloat32Sha256": pilot.sha256_array(audio),
    }


def metric_delta(
    candidate: dict[str, Any], baseline: dict[str, Any], keys: tuple[str, ...]
) -> dict[str, float]:
    return {
        key: float(candidate["aggregate"][key] - baseline["aggregate"][key])
        for key in keys
    }


def objective_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    min_projection_improvement_db: float = 0.25,
    max_quality_loss_db: float = 0.10,
) -> dict[str, Any]:
    candidate_metrics = candidate["aggregate"]
    baseline_metrics = baseline["aggregate"]
    projection_delta = candidate_metrics["accompanimentVocalProjectionDb"] - baseline_metrics[
        "accompanimentVocalProjectionDb"
    ]
    instrumental_delta = candidate_metrics["instrumentalSdrDb"] - baseline_metrics[
        "instrumentalSdrDb"
    ]
    low_vocal_delta = candidate_metrics["lowVocalInstrumentalSdrDb"] - baseline_metrics[
        "lowVocalInstrumentalSdrDb"
    ]
    passed = (
        projection_delta <= -min_projection_improvement_db
        and instrumental_delta >= -max_quality_loss_db
        and low_vocal_delta >= -max_quality_loss_db
    )
    return {
        "passedObjectiveOnly": passed,
        "thresholds": {
            "minimumVocalProjectionImprovementDb": min_projection_improvement_db,
            "maximumInstrumentalQualityLossDb": max_quality_loss_db,
            "maximumLowVocalQualityLossDb": max_quality_loss_db,
        },
        "deltas": {
            "accompanimentVocalProjectionDb": projection_delta,
            "instrumentalSdrDb": instrumental_delta,
            "lowVocalInstrumentalSdrDb": low_vocal_delta,
        },
        "blindListeningRequired": True,
    }


def train_variant(
    spec: VariantSpec,
    checkpoint: Path,
    records: list[pilot.WindowRecord],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    all_records: dict[str, list[pilot.WindowRecord]],
    eval_songs: list[pilot.SongBundle],
    all_songs: list[pilot.SongBundle],
    anchor_targets: dict[tuple[str, int], np.ndarray],
    baseline_probe: np.ndarray,
    baseline_eval: dict[str, Any],
    baseline_holdout: dict[str, Any],
    baseline_full: dict[str, Any],
    device: torch.device,
    steps: int,
    learning_rate: float,
    milestones: tuple[int, ...],
    active_threshold: float,
    seed: int,
    anchor_window: torch.Tensor,
    output_root: Path,
) -> tuple[dict[str, Any], torch.nn.Module]:
    model, _ = pilot.make_model(checkpoint, device)
    sweep.freeze_batchnorm_running_statistics(model)
    initial_parameters = [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    order = np.random.default_rng(seed).permutation(len(records)).tolist()
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
        }
    }
    started = time.perf_counter()
    for step in range(1, steps + 1):
        sweep.freeze_batchnorm_running_statistics(model)
        record = records[order[(step - 1) % len(records)]]
        record.materialize()
        if (
            record.input_spec is None
            or record.true_instrumental_spec is None
            or record.teacher_instrumental_spec is None
        ):
            raise ValueError(f"Incomplete training record for {record_key(record)}")
        input_tensor = torch.from_numpy(record.input_spec[None]).to(device)
        true_tensor = torch.from_numpy(record.true_instrumental_spec[None]).to(device)
        teacher_tensor = torch.from_numpy(record.teacher_instrumental_spec[None]).to(device)
        anchor_tensor = torch.from_numpy(anchor_targets[record_key(record)][None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_instrumental = input_tensor - model(input_tensor)
        ground_truth_loss = F.l1_loss(predicted_instrumental, true_tensor)
        anchor_loss = F.l1_loss(predicted_instrumental, anchor_tensor)
        teacher_loss = F.l1_loss(predicted_instrumental, teacher_tensor)
        projection_loss = predicted_instrumental.new_zeros(())
        if record.vocal_rms >= active_threshold:
            projection_loss = vocal_projection_loss(
                predicted_instrumental, record, anchor_window, device
            )
        loss = (
            ground_truth_loss
            + spec.anchor_weight * anchor_loss
            + spec.vocal_projection_weight * projection_loss
            + spec.teacher_weight * teacher_loss
        )
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "groundTruthLoss": float(ground_truth_loss.detach().cpu().item()),
                "anchorLoss": float(anchor_loss.detach().cpu().item()),
                "teacherLoss": float(teacher_loss.detach().cpu().item()),
                "vocalProjectionLoss": float(projection_loss.detach().cpu().item()),
                "activeVocal": int(record.vocal_rms >= active_threshold),
                "gradientNormBeforeClip": gradient_norm,
            }
        )
        if step in milestones:
            evaluation = pilot.evaluate_model(
                f"{spec.name}@{step}", model, eval_songs, eval_records, device
            )
            holdout_songs = [
                records_for_song[0].song
                for records_for_song in holdout_records.values()
                if records_for_song
            ]
            holdout_evaluation = pilot.evaluate_model(
                f"{spec.name}@{step}-holdout",
                model,
                holdout_songs,
                holdout_records,
                device,
            )
            probe = sweep.predicted_probe_specs(
                model,
                [record for records_for_song in eval_records.values() for record in records_for_song],
                device,
            )
            result: dict[str, Any] = {
                "step": step,
                "evaluation": {
                    "calibrationInternal": evaluation,
                    "trainHoldout": holdout_evaluation,
                },
                "parameterDriftRelative": parameter_drift_relative(
                    model, initial_parameters
                ),
                "outputChangeDb": sweep.relative_output_change_db(baseline_probe, probe),
            }
            if step == steps:
                result["evaluation"]["allFourSongs"] = pilot.evaluate_model(
                    f"{spec.name}@{step}-all", model, all_songs, all_records, device
                )
            milestone_results[str(step)] = result
            sweep.freeze_batchnorm_running_statistics(model)
    output_path = output_root / "runs" / f"{safe_name(spec.name)}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "local-inst3-objective-alignment",
            "variant": spec.name,
            "anchorWeight": spec.anchor_weight,
            "vocalProjectionWeight": spec.vocal_projection_weight,
            "teacherWeight": spec.teacher_weight,
            "batchNormRunningStatistics": "frozen",
            "steps": steps,
            "learningRate": learning_rate,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        output_path,
    )
    return {
        "variant": spec.name,
        "anchorWeight": spec.anchor_weight,
        "vocalProjectionWeight": spec.vocal_projection_weight,
        "teacherWeight": spec.teacher_weight,
        "steps": steps,
        "learningRate": learning_rate,
        "batchNormRunningStatistics": "frozen",
        "windowOrder": order,
        "history": history,
        "milestones": milestone_results,
        "finalObjectiveGate": objective_gate(
            milestone_results[str(steps)]["evaluation"]["allFourSongs"], baseline_full
        ),
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": {
            "file": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": pilot.sha256_file(output_path),
        },
    }, model


def main() -> int:
    args = parse_args()
    if args.start_seconds < 0 or args.duration_seconds <= 0:
        raise ValueError("start must be non-negative and duration must be positive")
    if args.train_windows_per_song <= 0:
        raise ValueError("train window count must be positive")
    if args.steps <= 0 or args.learning_rate <= 0 or args.threads <= 0:
        raise ValueError("steps, learning rate, and threads must be positive")
    if args.anchor_weight < 0 or args.vocal_projection_weight < 0:
        raise ValueError("loss weights must be non-negative")
    teacher_weights = tuple(float(value) for value in parse_number_list(args.teacher_weights))
    milestones = tuple(int(value) for value in parse_number_list(args.milestones, integer=True))
    if milestones[0] != 0 or any(value < 0 or value > args.steps for value in milestones):
        raise ValueError("milestones must include 0 and stay within --steps")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted and unique")
    if milestones[-1] != args.steps:
        raise ValueError("the final milestone must equal --steps")
    if any(value < 0 for value in teacher_weights):
        raise ValueError("teacher weights must be non-negative")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alignment_root = args.alignment_root.resolve()
    reports_root = alignment_root / "reports"
    listening_root = alignment_root / "listening"
    for path in (reports_root, alignment_root / "runs", listening_root):
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
        (
            entry,
            pilot.ensure_raw_song(archive_path, raw_root, entry),
        )
        for entry in selected_entries
    ]
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
        raise ValueError("Expected two train songs and two evaluation songs")
    if any(song.role == "final-test" for song in bundles):
        raise ValueError("The final-test split must never be used")

    train_records: list[pilot.WindowRecord] = []
    holdout_records: dict[str, list[pilot.WindowRecord]] = {}
    selection_report: dict[str, Any] = {}
    all_records: dict[str, list[pilot.WindowRecord]] = {}
    for song in bundles:
        candidates = sweep.candidate_windows(song)
        all_records[song.slug] = candidates
        selection_report[song.slug] = {
            "all": sweep.window_summary(candidates),
        }
        if song.role == "train":
            selected, holdout = sweep.select_stratified_windows(
                candidates, args.train_windows_per_song
            )
            if not holdout:
                raise ValueError(f"No held-out windows for {song.slug}")
            train_records.extend(selected)
            holdout_records[song.slug] = holdout
            selection_report[song.slug].update(
                {
                    "train": sweep.window_summary(selected),
                    "holdout": sweep.window_summary(holdout),
                }
            )
    eval_records = {
        song.slug: all_records[song.slug]
        for song in eval_songs
    }
    for record in train_records:
        record.materialize()
    for records in holdout_records.values():
        for record in records:
            record.materialize()
    for records in eval_records.values():
        for record in records:
            record.materialize()
    for records in all_records.values():
        for record in records:
            record.materialize()

    baseline_model, checkpoint_metadata = pilot.make_model(
        args.checkpoint.resolve(), device
    )
    active_threshold = activity_threshold(train_records)
    anchor_targets = build_anchor_targets(baseline_model, train_records, device)
    anchor_window = torch.hann_window(
        pilot.DEFAULT_CONFIG.n_fft,
        periodic=True,
        device=device,
    )
    baseline_full = pilot.evaluate_model(
        "initial-checkpoint-all-four", baseline_model, bundles, all_records, device
    )
    baseline_eval = pilot.evaluate_model(
        "initial-checkpoint-calibration-internal",
        baseline_model,
        eval_songs,
        eval_records,
        device,
    )
    holdout_songs = [
        records[0].song for records in holdout_records.values() if records
    ]
    baseline_holdout = pilot.evaluate_model(
        "initial-checkpoint-train-holdout",
        baseline_model,
        holdout_songs,
        holdout_records,
        device,
    )
    teacher_full = pilot.evaluate_teacher(bundles, all_records)
    teacher_eval = pilot.evaluate_teacher(eval_songs, eval_records)
    baseline_probe = sweep.predicted_probe_specs(
        baseline_model,
        [record for records in eval_records.values() for record in records],
        device,
    )
    initial_losses = initial_component_losses(
        baseline_model,
        train_records,
        anchor_targets,
        device,
        active_threshold,
        anchor_window,
    )
    listening: dict[str, Any] = {}
    if not args.skip_listening:
        for song in bundles:
            baseline_audio = render_model_song(
                baseline_model, song, all_records[song.slug], device
            )
            listening.setdefault("initialCheckpoint", {})[song.slug] = write_listening_file(
                listening_root / f"{song.slug}-initial.flac", baseline_audio
            )
            listening.setdefault("teacherInst3", {})[song.slug] = write_listening_file(
                listening_root / f"{song.slug}-teacher-inst3.flac",
                song.teacher_instrumental,
            )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    variants = make_variants(args.anchor_weight, args.vocal_projection_weight, teacher_weights)
    trajectories: dict[str, Any] = {}
    for variant in variants:
        report, model = train_variant(
            variant,
            args.checkpoint.resolve(),
            train_records,
            eval_records,
            holdout_records,
            all_records,
            eval_songs,
            bundles,
            anchor_targets,
            baseline_probe,
            baseline_eval,
            baseline_holdout,
            baseline_full,
            device,
            args.steps,
            args.learning_rate,
            milestones,
            active_threshold,
            args.seed,
            anchor_window,
            alignment_root,
        )
        if not args.skip_listening:
            for song in bundles:
                audio = render_model_song(model, song, all_records[song.slug], device)
                listening.setdefault(variant.name, {})[song.slug] = write_listening_file(
                    listening_root / f"{song.slug}-{safe_name(variant.name)}.flac",
                    audio,
                )
        trajectories[variant.name] = report
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schemaVersion": 1,
        "experimentId": "inst3-objective-alignment@1",
        "status": "completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local alignment artifacts only; do not publish",
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
            "calibration": [song.slug for song in bundles if song.role == "calibration"],
            "internalTest": [song.slug for song in bundles if song.role == "internal-test"],
            "startSeconds": args.start_seconds,
            "durationSeconds": args.duration_seconds,
        },
        "dataContract": {
            "inputSemantic": "mixture-gt",
            "mixtureGtDefinition": "vocals + drums + bass + other",
            "studentOutputSemantic": "instrumental",
            "neuralCoreSemantic": "vocals-residual",
        },
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
            "audit": {
                "allFourSongs": teacher_full,
                "calibrationInternal": teacher_eval,
                "deltaVsInitialCheckpoint": metric_delta(
                    teacher_full,
                    baseline_full,
                    (
                        "instrumentalSdrDb",
                        "vocalSdrDb",
                        "accompanimentVocalProjectionDb",
                        "lowVocalInstrumentalSdrDb",
                    ),
                ),
            },
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
            "trainWindowCount": len(train_records),
            "activeVocalThresholdRms": active_threshold,
            "activeVocalTrainWindowCount": sum(
                record.vocal_rms >= active_threshold for record in train_records
            ),
            "steps": args.steps,
            "milestones": milestones,
            "learningRate": args.learning_rate,
            "anchorWeight": args.anchor_weight,
            "vocalProjectionWeight": args.vocal_projection_weight,
            "teacherWeights": teacher_weights,
            "optimizer": "AdamW(weight_decay=0)",
            "gradientClipNorm": 1.0,
            "lossDefinitions": {
                "groundTruth": "L1 packed instrumental spectrum",
                "anchor": "L1 packed instrumental spectrum against zero-update checkpoint",
                "teacher": "L1 packed instrumental spectrum against Inst 3 output",
                "vocalProjection": "squared audio-domain projection coefficient of instrumental error onto reference vocals",
            },
            "initialComponentLosses": initial_losses,
            "trajectories": trajectories,
        },
        "baseline": {
            "allFourSongs": baseline_full,
            "calibrationInternal": baseline_eval,
            "trainHoldout": baseline_holdout,
        },
        "windowSelection": selection_report,
        "listening": listening,
        "notes": {
            "qualityGate": "Objective-only gate is reported per variant; blind listening remains required.",
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
    report_path = reports_root / "inst3-objective-alignment-report.json"
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
