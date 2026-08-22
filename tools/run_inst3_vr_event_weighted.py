#!/usr/bin/env python3
"""Train an Inst 3 residual student with hard-event loss weighting.

This experiment keeps the corrected V-R-H50 sampling schedule fixed and varies
only the temporal loss weight on the most difficult Inst 3-removal events.
The base H50 pass-25/pass-50 checkpoints are reused as the lambda=0 control;
the weighted arms start independently from the original vocals checkpoint.

The event mask is deliberately explicit and reproducible:

* use the frozen Stage 1 100 ms event arrays;
* keep positive-score events in each song's top 25 percent;
* mark a TFC time frame when its centered FFT support overlaps one of those
  event intervals;
* apply ``1 + lambda`` to every frequency/channel value in marked frames.

This is a spectral time-frame approximation of the audio-domain weighting
proposal.  It does not alter the teacher target, model, sampling schedule,
or evaluation split.  All generated weights, reports, masks, and listening
audio remain local non-commercial research artifacts under ``data/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import render_inst3_objective_listening as listening
import render_inst3_vr_hard_sampling_listening as vr_listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_scale10_density as density
import run_inst3_vr_hard_sampling as hard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-sampling-h50"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-event-weighted"
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_EVAL_ROOT / "musdb18-inst3-oracle-manifest.json"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
EVENT_MILLISECONDS = 100
EVENT_QUANTILE = 0.75
DEFAULT_LAMBDAS = (0.0, 0.25, 0.50)
DEFAULT_MILESTONES = (0, 25, 50)


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique floats, got {raw!r}")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Loss lambdas must be finite and non-negative")
    return values


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integers, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--lambdas",
        default=",".join(f"{value:g}" for value in DEFAULT_LAMBDAS),
        help="Sorted loss-weight values; zero reuses the H50 control",
    )
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument(
        "--milestones", default=",".join(str(value) for value in DEFAULT_MILESTONES)
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--state-interval", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return __import__("hashlib").sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return pilot.sha256_file(path)


def lambda_name(value: float) -> str:
    return f"lambda-{value:.2f}"


def validate_args(args: argparse.Namespace, lambdas: tuple[float, ...], milestones: tuple[int, ...]) -> None:
    if 0.0 not in lambdas:
        raise ValueError("lambdas must include 0.0 for the H50 control")
    if args.passes <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("passes, batch-size, and threads must be positive")
    if args.learning_rate <= 0.0 or args.state_interval <= 0:
        raise ValueError("learning-rate and state-interval must be positive")
    if milestones[0] != 0 or milestones[-1] != args.passes:
        raise ValueError("milestones must start at 0 and end at passes")
    if any(value < 0 or value > args.passes for value in milestones):
        raise ValueError("milestones must be within the training range")
    if args.eval_windows_per_song <= 0:
        raise ValueError("eval-windows-per-song must be positive")
    if args.max_songs is not None and args.max_songs <= 0:
        raise ValueError("max-songs must be positive")
    if args.max_eval_songs is not None and args.max_eval_songs <= 0:
        raise ValueError("max-eval-songs must be positive")


def load_base_report(base_root: Path) -> dict[str, Any]:
    path = base_root / "reports" / "inst3-vr-hard-sampling-report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    contract = report.get("contract", {})
    if report.get("status") != "completed":
        raise ValueError("H50 base report is not completed")
    if contract.get("hardVariant") != "V-R-H50":
        raise ValueError(f"Unexpected base hard variant: {contract.get('hardVariant')}")
    if float(contract.get("hardFraction", -1.0)) != 0.5:
        raise ValueError(f"Unexpected base hard fraction: {contract.get('hardFraction')}")
    return report


def build_frozen_h50_schedule(
    *,
    base_root: Path,
    event_root: Path,
    manifest_path: Path,
    max_songs: int | None,
    seed: int,
    passes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, hard.SongSelection], list[hard.ScheduleItem], dict[str, Any], str]:
    base_report = load_base_report(base_root)
    manifest = hard.load_manifest(manifest_path)
    train_entries = hard.load_train_entries(manifest, max_songs)
    selections, selection_payload, selection_sha = hard.build_selections(
        entries=train_entries,
        event_root=event_root,
        train_windows_per_song=8,
        hard_fraction=0.5,
        seed=seed,
    )
    expected_selection = base_report["contract"]["selectionSha256"]
    if max_songs is None and selection_sha != expected_selection:
        raise ValueError(
            f"H50 selection changed: {selection_sha} != {expected_selection}"
        )
    schedules = hard.build_training_schedule(
        selections, "V-R-H50", passes, seed, hard_variant="V-R-H50"
    )
    schedule_summary = hard.summarize_schedule(
        selections, {"V-R-H50": schedules}, passes, 4
    )["V-R-H50"]
    if max_songs is None:
        expected_schedule = base_report["schedule"]["V-R-H50"]["scheduleSha256"]
        if schedule_summary["scheduleSha256"] != expected_schedule:
            raise ValueError(
                f"H50 schedule changed: {schedule_summary['scheduleSha256']} != {expected_schedule}"
            )
    eval_entries = sorted(
        [
            entry
            for entry in manifest["entries"]
            if entry.get("role") in {"calibration", "internal-test"}
        ],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    return (
        train_entries,
        eval_entries,
        selections,
        schedules,
        {
            "manifest": manifest,
            "baseReport": base_report,
            "selection": selection_payload,
            "selectionSha256": selection_sha,
            "schedule": schedule_summary,
        },
        file_sha256(base_root / "reports" / "inst3-vr-hard-sampling-report.json"),
    )


def load_event_mask_blocks(event_root: Path, slug: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    report_path = event_root / "songs" / f"{slug}.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    array_path = Path(report["eventArrayFile"])
    if not array_path.is_file():
        array_path = event_root / "events" / f"{slug}.npz"
    with np.load(array_path) as values:
        starts = np.ascontiguousarray(values[f"{EVENT_MILLISECONDS}_startSamples"], dtype=np.int64)
        ends = np.ascontiguousarray(values[f"{EVENT_MILLISECONDS}_endSamples"], dtype=np.int64)
        scores = np.ascontiguousarray(values[f"{EVENT_MILLISECONDS}_score"], dtype=np.float32)
        active = np.ascontiguousarray(values[f"{EVENT_MILLISECONDS}_active"], dtype=bool)
    positive = active & np.isfinite(scores) & (scores > 0.0)
    if not np.any(positive):
        raise ValueError(f"No positive {EVENT_MILLISECONDS} ms events for {slug}")
    threshold = float(np.quantile(scores[positive], EVENT_QUANTILE))
    selected = positive & (scores >= threshold)
    if not np.any(selected):
        raise ValueError(f"No selected hard events for {slug}")
    return starts[selected], ends[selected], scores[selected], threshold


def event_frame_mask(
    *,
    candidate_start: int,
    candidate_length: int,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    sample_rate: int = 44_100,
) -> np.ndarray:
    """Map selected song events to centered TFC time frames.

    A frame is selected when its n_fft-wide analysis support overlaps an event
    after the candidate segment is placed at the student's trim offset.
    """
    del sample_rate  # The event arrays are already expressed in samples.
    frame_centers = np.arange(
        pilot.DEFAULT_CONFIG.num_frames, dtype=np.int64
    ) * pilot.DEFAULT_CONFIG.hop_length
    half_fft = pilot.DEFAULT_CONFIG.n_fft // 2
    segment_start = candidate_start
    segment_end = candidate_start + candidate_length
    mask = np.zeros(pilot.DEFAULT_CONFIG.num_frames, dtype=bool)
    for start, end in zip(event_starts, event_ends):
        overlap_start = max(segment_start, int(start))
        overlap_end = min(segment_end, int(end))
        if overlap_end <= overlap_start:
            continue
        local_start = overlap_start - candidate_start + pilot.DEFAULT_CONFIG.trim_samples
        local_end = overlap_end - candidate_start + pilot.DEFAULT_CONFIG.trim_samples
        mask |= (frame_centers + half_fft > local_start) & (
            frame_centers - half_fft < local_end
        )
    return mask


class WeightedCacheStore:
    def __init__(
        self,
        cache_paths: dict[str, Path],
        event_root: Path,
        selections: dict[str, hard.SongSelection],
    ) -> None:
        self.paths = cache_paths
        self.event_root = event_root
        self.selections = selections
        self._open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.mask_stats: dict[str, Any] = {}

    def _load(self, slug: str) -> dict[str, np.ndarray]:
        cached = self._open.get(slug)
        if cached is not None:
            self._open.move_to_end(slug)
            return cached
        path = self.paths[slug]
        selection = self.selections[slug]
        event_starts, event_ends, event_scores, threshold = load_event_mask_blocks(
            self.event_root, slug
        )
        with np.load(path) as values:
            inputs = np.ascontiguousarray(values["inputSpec"], dtype=np.float32)
            targets = np.ascontiguousarray(values["targetResidualSpec"], dtype=np.float32)
            starts = np.ascontiguousarray(values["starts"], dtype=np.int64)
            lengths = np.ascontiguousarray(values["lengths"], dtype=np.int64)
        expected_indices = np.asarray(selection.union_indices, dtype=np.int64)
        if inputs.shape != targets.shape or inputs.shape[0] != len(expected_indices):
            raise ValueError(f"Unexpected H50 cache shape for {slug}: {inputs.shape}")
        if starts.shape[0] != len(expected_indices) or lengths.shape[0] != len(expected_indices):
            raise ValueError(f"Unexpected H50 cache metadata shape for {slug}")
        masks = np.stack(
            [
                event_frame_mask(
                    candidate_start=int(start),
                    candidate_length=int(length),
                    event_starts=event_starts,
                    event_ends=event_ends,
                )
                for start, length in zip(starts, lengths)
            ],
            axis=0,
        )
        cached = {
            "inputSpec": inputs,
            "targetResidualSpec": targets,
            "eventFrameMask": np.ascontiguousarray(masks, dtype=bool),
        }
        self._open[slug] = cached
        self._open.move_to_end(slug)
        self.mask_stats[slug] = {
            "eventThreshold": threshold,
            "selectedEventCount": int(len(event_starts)),
            "candidateCount": int(len(starts)),
            "markedFrameCount": int(masks.sum()),
            "frameCount": int(masks.size),
            "markedFrameFraction": float(masks.mean()),
        }
        return cached

    def batch(
        self, items: Sequence[hard.ScheduleItem]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for item in items:
            arrays = self._load(item.slug)
            inputs.append(arrays["inputSpec"][item.cache_index])
            targets.append(arrays["targetResidualSpec"][item.cache_index])
            masks.append(arrays["eventFrameMask"][item.cache_index])
        return np.stack(inputs), np.stack(targets), np.stack(masks)

    def preload(self) -> None:
        total = len(self.paths)
        for index, slug in enumerate(sorted(self.paths), start=1):
            self._load(slug)
            if index == 1 or index == total or index % 10 == 0:
                print(f"preload weighted cache {index}/{total}", flush=True)


def weighted_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    event_mask: torch.Tensor,
    loss_lambda: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if event_mask.ndim != 2 or event_mask.shape[0] != prediction.shape[0] or event_mask.shape[1] != prediction.shape[-1]:
        raise ValueError(f"Unexpected event mask shape: {event_mask.shape}")
    weights = 1.0 + float(loss_lambda) * event_mask.to(dtype=prediction.dtype)[:, None, None, :]
    error = (prediction - target).abs()
    normalizer = weights.sum() * prediction.shape[1] * prediction.shape[2]
    return (error * weights).sum() / normalizer


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def latest_checkpoint(run_root: Path, contract_id: str) -> tuple[Path, dict[str, Any]] | None:
    best: tuple[int, Path, dict[str, Any]] | None = None
    for path in run_root.glob("step-*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("runContractId") != contract_id:
                continue
            step = int(payload["step"])
        except (OSError, KeyError, RuntimeError, TypeError, ValueError):
            continue
        if best is None or step > best[0]:
            best = (step, path, payload)
    return None if best is None else (best[1], best[2])


def train_weighted_variant(
    *,
    loss_lambda: float,
    checkpoint: Path,
    store: WeightedCacheStore,
    schedule: list[hard.ScheduleItem],
    run_root: Path,
    contract_id: str,
    passes: int,
    milestones: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    state_interval: int,
    resume: bool,
) -> dict[str, Any]:
    records_per_pass = len(schedule) // passes
    updates_per_pass = records_per_pass // batch_size
    total_updates = len(schedule) // batch_size
    milestone_updates = {
        pass_count * updates_per_pass: pass_count for pass_count in milestones
    }
    run_root.mkdir(parents=True, exist_ok=True)
    hard.set_seed(seed)
    model, checkpoint_source = pilot.make_model(checkpoint, device)
    sweep.freeze_batchnorm_running_statistics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    current_update = 0
    history: list[dict[str, float | int]] = []
    resume_count = 0
    if resume:
        found = latest_checkpoint(run_root, contract_id)
        if found is not None:
            path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            current_update = int(payload["step"])
            history = list(payload.get("history", []))
            resume_count = int(payload.get("resumeCount", 0)) + 1
            print(
                json.dumps(
                    {"event": "resume", "lambda": loss_lambda, "step": current_update, "file": str(path)}
                ),
                flush=True,
            )
    store.preload()
    started = time.perf_counter()
    milestone_files: dict[str, dict[str, Any]] = {}
    checkpoint_files: dict[str, dict[str, Any]] = {}

    def discover() -> None:
        for update in milestone_updates:
            path = run_root / f"step-{update}.pt"
            if path.is_file():
                metadata = checkpoint_metadata(path)
                milestone_files[str(update)] = metadata
                checkpoint_files[str(update)] = metadata

    discover()

    def save_checkpoint(step: int, status: str) -> None:
        path = run_root / f"step-{step}.pt"
        payload = {
            "format": "local-inst3-vr-event-weighted-checkpoint@1",
            "status": status,
            "variant": lambda_name(loss_lambda),
            "lossLambda": loss_lambda,
            "runContractId": contract_id,
            "step": step,
            "passes": passes,
            "recordsPerPass": records_per_pass,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "seed": seed,
            "stateDict": hard.cpu_tree(model.state_dict()),
            "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
            "history": history,
            "resumeCount": resume_count,
            "elapsedSeconds": time.perf_counter() - started,
            "checkpointSource": checkpoint_source["checkpoint"],
        }
        metadata = hard.atomic_torch_save(path, payload)
        checkpoint_files[str(step)] = metadata
        if step in milestone_updates:
            milestone_files[str(step)] = metadata

    if current_update == 0 and 0 in milestone_updates:
        save_checkpoint(0, "initial")

    while current_update < total_updates:
        begin = current_update * batch_size
        batch_items = schedule[begin : begin + batch_size]
        if len(batch_items) != batch_size:
            raise AssertionError("Incomplete training batch")
        input_array, target_array, mask_array = store.batch(batch_items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        mask_tensor = torch.from_numpy(mask_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_residual = model(input_tensor)
        loss = weighted_l1_loss(
            predicted_residual, target_tensor, mask_tensor, loss_lambda
        )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            ).item()
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
        if current_update == 1 or current_update % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "lambda": loss_lambda,
                        "update": current_update,
                        "totalUpdates": total_updates,
                        "loss": history[-1]["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if current_update in milestone_updates and current_update != 0:
            save_checkpoint(current_update, "milestone")
            print(
                json.dumps(
                    {
                        "event": "milestone",
                        "lambda": loss_lambda,
                        "pass": milestone_updates[current_update],
                        "update": current_update,
                        "loss": history[-1]["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elif state_interval > 0 and current_update % state_interval == 0:
            save_checkpoint(current_update, "rolling")

    if str(total_updates) not in milestone_files:
        save_checkpoint(total_updates, "completed")
    else:
        discover()
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "variant": lambda_name(loss_lambda),
        "lossLambda": loss_lambda,
        "status": "completed",
        "passes": passes,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": updates_per_pass,
        "updates": total_updates,
        "learningRate": learning_rate,
        "batchSize": batch_size,
        "seed": seed,
        "history": history,
        "milestoneCheckpoints": milestone_files,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
        "resumeCount": resume_count,
    }


def base_control_training(base_report: dict[str, Any]) -> dict[str, Any]:
    training = base_report["training"]["V-R-H50"]
    return {
        "variant": lambda_name(0.0),
        "lossLambda": 0.0,
        "status": "reused-existing-h50",
        "passes": training["passes"],
        "recordsPerPass": training["recordsPerPass"],
        "updatesPerPass": training["updatesPerPass"],
        "updates": training["updates"],
        "learningRate": training["learningRate"],
        "batchSize": training["batchSize"],
        "seed": training["seed"],
        "history": [],
        "milestoneCheckpoints": training["milestoneCheckpoints"],
        "checkpointFiles": training["checkpointFiles"],
        "elapsedSeconds": 0.0,
        "resumeCount": 0,
    }


def render_listening(
    *,
    output_root: Path,
    samples_root: Path,
    checkpoint: Path,
    weighted_results: dict[str, dict[str, Any]],
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    songs = density.validate_private_songs(samples_root.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "local-inst3-vr-event-weighted-listening@1",
        "status": "running",
        "semantic": "residual-vocals-to-instrumental",
        "format": "PCM16 FLAC",
        "studentConfig": dict(listening.DEFAULT_CONFIG.__dict__),
        "controlRoot": str(
            (DEFAULT_BASE_ROOT / "listening-12" / "V-R-H50").resolve()
        ),
        "songs": {},
        "variants": {},
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    models: dict[str, torch.nn.Module] = {}
    for name, result in weighted_results.items():
        if name == lambda_name(0.0):
            continue
        pass_key = str(result["updates"])
        checkpoint_path = Path(result["milestoneCheckpoints"][pass_key]["file"])
        model, _ = pilot.make_model(checkpoint, device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["stateDict"], strict=True)
        sweep.freeze_batchnorm_running_statistics(model)
        models[name] = model
        report["variants"][name] = {
            "lossLambda": result["lossLambda"],
            "checkpoint": checkpoint_metadata(checkpoint_path),
            "outputRoot": str((output_root / name).resolve()),
        }
    try:
        for index, song in enumerate(songs, start=1):
            name = song["name"]
            source, sample_rate = listening.load_audio(Path(song["file"]))
            if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
                raise ValueError(f"Unexpected source sample rate: {song['file']}")
            report["songs"][name] = {
                "source": {
                    **song,
                    "sampleRate": sample_rate,
                    "frames": int(source.shape[0]),
                    "durationSeconds": source.shape[0] / sample_rate,
                    "decodedFloat32Sha256": pilot.sha256_array(source),
                },
                "outputs": {},
            }
            for variant, model in models.items():
                path = output_root / variant / f"{name}.flac"
                skipped = path.is_file() and not force
                if not skipped:
                    audio, timing = vr_listening.render_student(source, model, device)
                    listening.write_flac(path, audio, listening.DEFAULT_CONFIG.sample_rate)
                    del audio
                decoded, decoded_rate = listening.load_audio(path)
                if decoded_rate != sample_rate or decoded.shape != source.shape:
                    raise ValueError(f"Listening output shape mismatch: {path}")
                if not np.isfinite(decoded).all():
                    raise ValueError(f"Non-finite listening output: {path}")
                report["songs"][name]["outputs"][variant] = {
                    "file": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "frames": int(decoded.shape[0]),
                    "sampleRate": int(decoded_rate),
                    "channels": int(decoded.shape[1]),
                    "skippedExisting": skipped,
                    "semantic": "residual-vocals-to-instrumental",
                }
            json_write(output_root / "render-report.json", report)
            print(f"weighted listening {index}/{len(songs)}: {name}", flush=True)
            del source
            gc.collect()
    finally:
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "completed"
    report["outputCount"] = sum(
        len(item["outputs"]) for item in report["songs"].values()
    )
    json_write(output_root / "render-report.json", report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    lambdas = parse_float_list(args.lambdas)
    milestones = parse_int_list(args.milestones)
    validate_args(args, lambdas, milestones)
    base_root = args.base_root.resolve()
    output_root = args.output_root.resolve()
    event_root = args.event_root.resolve()
    eval_root = args.eval_root.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    (
        train_entries,
        eval_entries,
        selections,
        schedules,
        frozen,
        base_report_sha256,
    ) = build_frozen_h50_schedule(
        base_root=base_root,
        event_root=event_root,
        manifest_path=manifest_path,
        max_songs=args.max_songs,
        seed=args.seed,
        passes=args.passes,
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    cache_paths = {
        slug: base_root / "cache" / "train" / f"{slug}.npz"
        for slug in selections
    }
    for slug, path in cache_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

    store = WeightedCacheStore(cache_paths, event_root, selections)
    store.preload()
    output_root.mkdir(parents=True, exist_ok=True)
    json_write(output_root / "event-mask-stats.json", store.mask_stats)

    train_results: dict[str, dict[str, Any]] = {}
    base_report = frozen["baseReport"]
    if 0.0 in lambdas:
        train_results[lambda_name(0.0)] = base_control_training(base_report)
    contract_base = {
        "schema": "local-inst3-vr-event-weighted@1",
        "baseReportSha256": base_report_sha256,
        "baseSelectionSha256": frozen["selectionSha256"],
        "baseScheduleSha256": frozen["schedule"]["scheduleSha256"],
        "eventReportSha256": file_sha256(
            event_root / "reports" / "inst3-vr-hard-events-report.json"
        ),
        "eventMilliseconds": EVENT_MILLISECONDS,
        "eventQuantile": EVENT_QUANTILE,
        "eventMask": "centered n_fft support overlaps selected 100 ms blocks",
        "trainSongCount": len(train_entries),
        "evalSongCount": len(eval_entries),
        "officialFinalTestUsed": False,
        "trainWindowsPerSong": 8,
        "hardVariant": "V-R-H50",
        "hardFraction": 0.5,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "studentCheckpointSha256": file_sha256(checkpoint),
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "loss": "normalized temporal-frame weighted spectral L1",
    }
    for loss_lambda in lambdas:
        if loss_lambda == 0.0:
            continue
        variant = lambda_name(loss_lambda)
        variant_contract = dict(contract_base)
        variant_contract["variant"] = variant
        variant_contract["lossLambda"] = loss_lambda
        contract_id = canonical_sha256(variant_contract)
        train_results[variant] = train_weighted_variant(
            loss_lambda=loss_lambda,
            checkpoint=checkpoint,
            store=store,
            schedule=schedules,
            run_root=output_root / "runs" / variant,
            contract_id=contract_id,
            passes=args.passes,
            milestones=milestones,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
            state_interval=args.state_interval,
            resume=args.resume,
        )
        train_results[variant]["contract"] = {
            "id": contract_id,
            "payload": variant_contract,
        }

    evaluation_inputs: dict[str, dict[str, Any]] = {}
    for variant, result in train_results.items():
        evaluation_inputs[variant] = {
            "milestoneCheckpoints": result["milestoneCheckpoints"],
            "updatesPerPass": result["updatesPerPass"],
        }
    evaluation = hard.evaluate_checkpoints(
        checkpoint=checkpoint,
        experiment_root=output_root,
        train_results=evaluation_inputs,
        eval_entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=args.eval_windows_per_song,
        seed=args.seed,
        device=device,
        milestones=milestones,
    )
    report: dict[str, Any] = {
        "schema": "local-inst3-vr-event-weighted@1",
        "status": "completed",
        "experimentId": "inst3-vr-event-weighted@1",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "teacher-derived targets and checkpoints remain local; not published",
        },
        "contract": contract_base,
        "manifest": {
            "file": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifestId": frozen["manifest"]["manifestId"],
            "trainSongCount": len(train_entries),
            "evaluationSongCount": len(eval_entries),
            "officialFinalTestUsed": False,
        },
        "eventMask": {
            "milliseconds": EVENT_MILLISECONDS,
            "quantile": EVENT_QUANTILE,
            "statsFile": str((output_root / "event-mask-stats.json").resolve()),
            "aggregateMarkedFrameFraction": float(
                np.mean([item["markedFrameFraction"] for item in store.mask_stats.values()])
            ),
            "songCount": len(store.mask_stats),
        },
        "training": train_results,
        "evaluation": evaluation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "cudaDevice": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "notes": {
            "control": "lambda=0 reuses the completed H50 pass-25/pass-50 checkpoints exactly",
            "sampling": "The V-R-H50 schedule is unchanged: four uniform plus four per-song hard-pool draws",
            "loss": "Only selected 100 ms event-overlapping TFC time frames receive the additional lambda weight",
            "evaluation": "Calibration/internal-test only; official final-test songs were not used",
            "publication": "Do not publish checkpoints, teacher-derived audio, or MUSDB18-derived caches",
        },
    }
    json_write(output_root / "reports" / "inst3-vr-event-weighted-report.json", report)

    if not args.skip_listening and args.max_songs is None and args.max_eval_songs is None:
        listening_report = render_listening(
            output_root=output_root / "listening-12",
            samples_root=args.samples_root,
            checkpoint=checkpoint,
            weighted_results=train_results,
            device=device,
            force=args.force_listening,
        )
        report["listening"] = listening_report
        json_write(output_root / "reports" / "inst3-vr-event-weighted-report.json", report)
    else:
        report["listening"] = {"status": "skipped"}
        json_write(output_root / "reports" / "inst3-vr-event-weighted-report.json", report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(
                    (output_root / "reports" / "inst3-vr-event-weighted-report.json").resolve()
                ),
                "variants": list(train_results),
                "evaluation": list(evaluation),
                "listening": report["listening"]["status"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
