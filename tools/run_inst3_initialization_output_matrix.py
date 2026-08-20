#!/usr/bin/env python3
"""Run the four-cell Inst 3 initialization/output-semantics matrix.

The controlled factors are:

    W-R: pretrained vocals checkpoint, residual-vocals neural output
    W-D: pretrained body with reset output convolution, direct instrumental output
    R-R: seeded random initialization, residual-vocals neural output
    R-D: seeded random initialization, direct instrumental output

Every cell sees the same frozen scale-10 windows in the same order, uses the
same optimizer, learning rate, batch size, and FP32 arithmetic, and updates
BatchNorm population statistics.  Milestones are expressed as complete
passes over the 320 training windows.  Three seeds are used by default.

All checkpoints and reports are local non-commercial research artifacts under
the ignored data tree.  The runner persists model, AdamW, RNG, schedule, and
milestone state for exact logical resume after interruption.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import run_inst3_aggressive_scale10 as scale10
import run_inst3_aggressive_target as aggressive
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_scale10_density as density
from tfc_tdf_default_model import (
    TfcTdfNeuralCore,
    TfcTdfNchwWrapper,
    load_default_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "musdb18-inst3-scale10"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-init-output-matrix"
DEFAULT_MANIFEST = (
    ROOT / "data" / "musdb18-inst3-oracle" / "musdb18-inst3-oracle-manifest.json"
)
DEFAULT_VARIANTS = ("W-R", "W-D", "R-R", "R-D")
DEFAULT_SEEDS = (891, 1891, 2891)
DEFAULT_MILESTONE_PASSES = (0, 25, 50, 100)
STATE_FORMAT = "local-inst3-init-output-matrix-state@1"
CHECKPOINT_FORMAT = "local-inst3-init-output-matrix-checkpoint@1"
EXPERIMENT_ID = "inst3-init-output-matrix@1"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    initialization: str
    output_semantic: str
    reset_output_head: bool


VARIANT_SPECS: dict[str, VariantSpec] = {
    "W-R": VariantSpec("W-R", "pretrained-vocals", "residual-vocals", False),
    "W-D": VariantSpec("W-D", "pretrained-body", "direct-instrumental", True),
    "R-R": VariantSpec("R-R", "random", "residual-vocals", False),
    "R-D": VariantSpec("R-D", "random", "direct-instrumental", False),
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--passes", type=int, default=100)
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
        help="Pause each selected run after this many optimizer updates; smoke testing only",
    )
    return parser.parse_args(argv)


def parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated list")
    return values


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_csv_strings(raw))


def validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[VariantSpec, ...], tuple[int, ...], tuple[int, ...]]:
    variant_keys = parse_csv_strings(args.variants)
    unknown = sorted(set(variant_keys) - set(VARIANT_SPECS))
    if unknown:
        raise ValueError(f"Unknown matrix variants: {unknown}")
    if len(set(variant_keys)) != len(variant_keys):
        raise ValueError("Matrix variants must be unique")
    variants = tuple(VARIANT_SPECS[key] for key in variant_keys)
    seeds = parse_csv_ints(args.seeds)
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    milestones = parse_csv_ints(args.milestone_passes)
    if milestones[0] != 0 or milestones[-1] != args.passes:
        raise ValueError("Milestone passes must start at 0 and end at --passes")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("Milestone passes must be sorted and unique")
    positive = (
        args.passes,
        args.batch_size,
        args.train_windows_per_song,
        args.holdout_windows_per_song,
        args.eval_windows_per_song,
        args.train_probe_windows_per_song,
        args.probe_song_count,
        args.state_interval_updates,
        args.threads,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0:
        raise ValueError("Passes, counts, intervals, threads, and learning rate must be positive")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        raise ValueError("--stop-after-updates must be positive")
    return variants, seeds, milestones


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def reset_direct_output_head(core: TfcTdfNeuralCore, seed: int) -> None:
    convolution = core.last_conv[0]
    if not isinstance(convolution, torch.nn.Conv2d):
        raise TypeError("Unexpected TFC-TDF output head")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        convolution.reset_parameters()


def make_matrix_model(
    spec: VariantSpec,
    checkpoint: Path,
    seed: int,
    device: torch.device,
) -> tuple[TfcTdfNchwWrapper, dict[str, Any]]:
    if spec.initialization.startswith("pretrained"):
        core, checkpoint_metadata = load_default_checkpoint(checkpoint)
        if spec.reset_output_head:
            reset_direct_output_head(core, seed)
        initialization_source = checkpoint_metadata["checkpoint"]
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            core = TfcTdfNeuralCore()
        initialization_source = {
            "kind": "torch-default-seeded-random",
            "seed": seed,
            "torchVersion": torch.__version__,
        }
    model = TfcTdfNchwWrapper(core).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    metadata = {
        "variant": asdict(spec),
        "initializationSource": initialization_source,
        "initialStateSha256": state_dict_sha256(model.state_dict()),
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "batchNormPolicy": "train-mode running statistics updated for every matrix cell",
    }
    return model, metadata


def predicted_instrumental_spectrum(
    model_output: torch.Tensor,
    mixture_spectrum: torch.Tensor,
    spec: VariantSpec,
) -> torch.Tensor:
    if model_output.shape != mixture_spectrum.shape:
        raise ValueError(
            f"Model/mixture spectrum mismatch: {model_output.shape} != {mixture_spectrum.shape}"
        )
    if spec.output_semantic == "residual-vocals":
        return mixture_spectrum - model_output
    if spec.output_semantic == "direct-instrumental":
        return model_output
    raise ValueError(f"Unsupported output semantic: {spec.output_semantic}")


def predict_instrumental(
    model: torch.nn.Module,
    mixture_spectrum: torch.Tensor,
    spec: VariantSpec,
) -> torch.Tensor:
    return predicted_instrumental_spectrum(model(mixture_spectrum), mixture_spectrum, spec)


def build_sample_schedule(window_count: int, passes: int, seed: int) -> np.ndarray:
    if window_count <= 0 or passes <= 0:
        raise ValueError("window_count and passes must be positive")
    rng = np.random.default_rng(seed)
    schedule = np.concatenate(
        [rng.permutation(window_count) for _ in range(passes)]
    ).astype(np.int32, copy=False)
    return np.ascontiguousarray(schedule)


def schedule_sha256(schedule: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(schedule.astype("<i4", copy=False)).tobytes()
    ).hexdigest()


def validate_schedule(schedule: np.ndarray, window_count: int, passes: int) -> None:
    if schedule.shape != (window_count * passes,):
        raise ValueError(f"Unexpected schedule shape: {schedule.shape}")
    expected = np.arange(window_count, dtype=np.int32)
    for pass_index in range(passes):
        values = schedule[pass_index * window_count : (pass_index + 1) * window_count]
        if not np.array_equal(np.sort(values), expected):
            raise ValueError(f"Pass {pass_index + 1} is not a complete permutation")


def batchnorm_summary(model: torch.nn.Module) -> dict[str, Any]:
    tracked: list[int] = []
    running_mean_power = 0.0
    running_variance_mean = 0.0
    module_count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        module_count += 1
        if module.num_batches_tracked is not None:
            tracked.append(int(module.num_batches_tracked.detach().cpu().item()))
        if module.running_mean is not None:
            values = module.running_mean.detach().double().cpu()
            running_mean_power += float(torch.sum(values * values).item())
        if module.running_var is not None:
            running_variance_mean += float(module.running_var.detach().double().mean().cpu().item())
    return {
        "moduleCount": module_count,
        "numBatchesTrackedRange": [min(tracked), max(tracked)] if tracked else None,
        "runningMeanL2": math.sqrt(running_mean_power),
        "meanRunningVarianceAcrossModules": (
            running_variance_mean / module_count if module_count else None
        ),
        "stateSha256": state_dict_sha256(
            {
                key: value
                for key, value in model.state_dict().items()
                if "running_" in key or "num_batches_tracked" in key
            }
        ),
    }


def records_by_song_subset(
    records: Sequence[pilot.WindowRecord],
    count_per_song: int,
) -> dict[str, list[pilot.WindowRecord]]:
    grouped: dict[str, list[pilot.WindowRecord]] = {}
    for record in records:
        grouped.setdefault(record.song.slug, []).append(record)
    return {
        slug: values[: min(count_per_song, len(values))]
        for slug, values in grouped.items()
    }


def unique_songs(
    records_by_song: dict[str, list[pilot.WindowRecord]],
) -> list[pilot.SongBundle]:
    songs: list[pilot.SongBundle] = []
    for records in records_by_song.values():
        if records:
            songs.append(records[0].song)
    return songs


def evaluate_matrix_model(
    *,
    name: str,
    model: torch.nn.Module,
    spec: VariantSpec,
    songs: list[pilot.SongBundle],
    records_by_song: dict[str, list[pilot.WindowRecord]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    per_song: dict[str, Any] = {}
    aggregate: list[dict[str, np.ndarray]] = []
    spectral_absolute_error = 0.0
    spectral_value_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for song in songs:
            records = records_by_song[song.slug]
            predictions: list[np.ndarray] = []
            mixtures: list[np.ndarray] = []
            true_vocals: list[np.ndarray] = []
            true_instrumentals: list[np.ndarray] = []
            teacher_instrumentals: list[np.ndarray] = []
            for begin in range(0, len(records), batch_size):
                batch = records[begin : begin + batch_size]
                for record in batch:
                    record.materialize()
                    if (
                        record.input_spec is None
                        or record.teacher_instrumental_spec is None
                        or record.song.teacher_instrumental is None
                    ):
                        raise ValueError(
                            f"Incomplete matrix evaluation record: {record.song.slug}:{record.start}"
                        )
                inputs = np.stack([record.input_spec for record in batch])
                targets = np.stack(
                    [record.teacher_instrumental_spec for record in batch]
                )
                input_tensor = torch.from_numpy(inputs).to(device)
                predicted_tensor = predict_instrumental(model, input_tensor, spec)
                predicted = predicted_tensor.detach().float().cpu().numpy()
                spectral_absolute_error += float(
                    np.sum(np.abs(predicted.astype(np.float64) - targets.astype(np.float64)))
                )
                spectral_value_count += int(predicted.size)
                for offset, record in enumerate(batch):
                    reconstructed = pilot.student_istft_centered(
                        predicted[offset : offset + 1]
                    )
                    trim = pilot.DEFAULT_CONFIG.trim_samples
                    predictions.append(
                        np.ascontiguousarray(
                            reconstructed[trim : trim + record.length],
                            dtype=np.float32,
                        )
                    )
                    start, end = record.start, record.start + record.length
                    mixtures.append(song.mixture_gt[start:end])
                    true_vocals.append(song.vocals[start:end])
                    true_instrumentals.append(song.instrumental[start:end])
                    teacher_instrumentals.append(song.teacher_instrumental[start:end])
            mixture = np.concatenate(mixtures, axis=0)
            vocals = np.concatenate(true_vocals, axis=0)
            instrumental = np.concatenate(true_instrumentals, axis=0)
            teacher = np.concatenate(teacher_instrumentals, axis=0)
            candidate = np.concatenate(predictions, axis=0)
            per_song[song.slug] = {
                "role": song.role,
                "windowStarts": [record.start for record in records],
                "metrics": aggressive.aggressive_metrics(
                    mixture,
                    vocals,
                    instrumental,
                    teacher,
                    teacher,
                    candidate,
                ),
            }
            aggregate.append(
                {
                    "mixture": mixture,
                    "vocals": vocals,
                    "instrumental": instrumental,
                    "teacher": teacher,
                    "candidate": candidate,
                }
            )
    merged = {
        key: np.concatenate([item[key] for item in aggregate], axis=0)
        for key in aggregate[0]
    }
    metrics = aggressive.aggressive_metrics(
        merged["mixture"],
        merged["vocals"],
        merged["instrumental"],
        merged["teacher"],
        merged["teacher"],
        merged["candidate"],
    )
    metrics["teacherSpectrumL1"] = spectral_absolute_error / spectral_value_count
    result = {
        "variant": name,
        "outputSemantic": spec.output_semantic,
        "aggregate": metrics,
        "perSong": per_song,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }
    model.train()
    return result


def probe_predictions(
    model: torch.nn.Module,
    spec: VariantSpec,
    records: Sequence[pilot.WindowRecord],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(records), batch_size):
            batch = records[begin : begin + batch_size]
            for record in batch:
                record.materialize()
                if record.input_spec is None:
                    raise ValueError("Probe input is unavailable")
            inputs = torch.from_numpy(
                np.stack([record.input_spec for record in batch])
            ).to(device)
            outputs.append(
                predict_instrumental(model, inputs, spec)
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )
    model.train()
    return np.concatenate(outputs)


def relative_output_change_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("Probe output shapes differ")
    reference64 = reference.astype(np.float64)
    difference = candidate.astype(np.float64) - reference64
    reference_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    difference_rms = float(np.sqrt(np.mean(difference * difference)))
    return 20.0 * math.log10(
        max(difference_rms, 1.0e-30) / max(reference_rms, 1.0e-30)
    )


def make_schedule_seed(seed: int) -> int:
    return seed ^ 0x5A17C3


def make_run_contract(
    *,
    args: argparse.Namespace,
    spec: VariantSpec,
    seed: int,
    schedule: np.ndarray,
    manifest_path: Path,
    selection_path: Path,
    window_report: dict[str, Any],
    train_window_count: int,
    device: torch.device,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    value = {
        "experimentId": EXPERIMENT_ID,
        "variant": asdict(spec),
        "seed": seed,
        "scheduleSeed": make_schedule_seed(seed),
        "scheduleSha256": schedule_sha256(schedule),
        "passes": args.passes,
        "milestonePasses": list(parse_csv_ints(args.milestone_passes)),
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "optimizer": "AdamW(weight_decay=0)",
        "gradientClipNorm": 1.0,
        "precision": "float32",
        "batchNormPolicy": "updated in train mode for every cell",
        "trainWindowCount": train_window_count,
        "trainWindowsPerSong": args.train_windows_per_song,
        "holdoutWindowsPerSong": args.holdout_windows_per_song,
        "evalWindowsPerSong": args.eval_windows_per_song,
        "manifestSha256": pilot.sha256_file(manifest_path),
        "selectionSha256": pilot.sha256_file(selection_path),
        "windowSelectionSha256": density.canonical_sha256(window_report),
        "initialCheckpointSha256": pilot.sha256_file(args.checkpoint.resolve()),
        "device": str(device),
        "torchVersion": torch.__version__,
        "runnerSha256": pilot.sha256_file(runner_path),
    }
    return {"id": density.canonical_sha256(value), "contract": value}


def validate_resume_payload(payload: dict[str, Any], run_contract: dict[str, Any]) -> None:
    if payload.get("format") not in {STATE_FORMAT, CHECKPOINT_FORMAT}:
        raise ValueError(f"Unexpected matrix state format: {payload.get('format')}")
    if payload.get("runContract", {}).get("id") != run_contract["id"]:
        raise ValueError("Matrix training state contract does not match this run")


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


def checkpoint_payload(
    *,
    state_format: str,
    status: str,
    run_contract: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    history: list[dict[str, float | int]],
    milestone_results: dict[str, Any],
    baseline: dict[str, Any],
    elapsed_seconds: float,
    resume_count: int,
) -> dict[str, Any]:
    return {
        "format": state_format,
        "status": status,
        "runContract": run_contract,
        "update": update,
        "state_dict": density.cpu_tree(model.state_dict()),
        "optimizer_state_dict": density.cpu_tree(optimizer.state_dict()),
        "history": history,
        "milestoneResults": milestone_results,
        "baseline": baseline,
        "elapsedSeconds": elapsed_seconds,
        "resumeCount": resume_count,
        "rngState": density.capture_rng_state(),
    }


def aggregate_history_by_pass(
    history: Sequence[dict[str, float | int]],
    updates_per_pass: int,
) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for begin in range(0, len(history), updates_per_pass):
        values = history[begin : begin + updates_per_pass]
        if not values:
            continue
        losses = [float(item["loss"]) for item in values]
        gradients = [float(item["gradientNormBeforeClip"]) for item in values]
        result.append(
            {
                "pass": begin // updates_per_pass + 1,
                "updates": len(values),
                "meanLoss": float(np.mean(losses)),
                "maxLoss": max(losses),
                "meanGradientNormBeforeClip": float(np.mean(gradients)),
                "maxGradientNormBeforeClip": max(gradients),
            }
        )
    return result


def train_run(
    *,
    args: argparse.Namespace,
    spec: VariantSpec,
    seed: int,
    run_contract: dict[str, Any],
    schedule: np.ndarray,
    train_records: list[pilot.WindowRecord],
    train_probe_records: dict[str, list[pilot.WindowRecord]],
    eval_records: dict[str, list[pilot.WindowRecord]],
    holdout_records: dict[str, list[pilot.WindowRecord]],
    probe_records: list[pilot.WindowRecord],
    device: torch.device,
) -> dict[str, Any]:
    run_root = args.experiment_root.resolve() / "runs" / f"seed-{seed}" / spec.key
    rolling_path = run_root / "training-state.pt"
    window_count = len(train_records)
    if window_count % args.batch_size != 0:
        raise ValueError("Training window count must be divisible by batch size")
    updates_per_pass = window_count // args.batch_size
    milestone_passes = parse_csv_ints(args.milestone_passes)
    milestone_updates = tuple(value * updates_per_pass for value in milestone_passes)
    total_updates = args.passes * updates_per_pass

    model, initialization_metadata = make_matrix_model(
        spec, args.checkpoint.resolve(), seed, device
    )
    initial_parameters = [
        parameter.detach().float().cpu().clone() for parameter in model.parameters()
    ]
    initial_probe = probe_predictions(
        model, spec, probe_records, device, args.batch_size
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    baseline: dict[str, Any]
    milestone_results: dict[str, Any]
    history: list[dict[str, float | int]] = []
    current_update = 0
    elapsed_before = 0.0
    resume_count = 0

    resume_path = (
        find_resume_path(run_root, milestone_updates) if args.resume else None
    )
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_payload(payload, run_contract)
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload["history"])
        milestone_results = dict(payload["milestoneResults"])
        baseline = dict(payload["baseline"])
        current_update = int(payload["update"])
        elapsed_before = float(payload.get("elapsedSeconds", 0.0))
        resume_count = int(payload.get("resumeCount", 0)) + 1
        density.restore_rng_state(payload["rngState"])
        print(
            json.dumps(
                {
                    "event": "resume",
                    "variant": spec.key,
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
        print(f"baseline: seed={seed} variant={spec.key}", flush=True)
        baseline = {
            "trainFitProbe": evaluate_matrix_model(
                name=f"{spec.key}-seed-{seed}@0-train",
                model=model,
                spec=spec,
                songs=unique_songs(train_probe_records),
                records_by_song=train_probe_records,
                device=device,
                batch_size=args.batch_size,
            ),
            "trainHoldout": evaluate_matrix_model(
                name=f"{spec.key}-seed-{seed}@0-holdout",
                model=model,
                spec=spec,
                songs=unique_songs(holdout_records),
                records_by_song=holdout_records,
                device=device,
                batch_size=args.batch_size,
            ),
            "calibrationInternal": evaluate_matrix_model(
                name=f"{spec.key}-seed-{seed}@0-eval",
                model=model,
                spec=spec,
                songs=unique_songs(eval_records),
                records_by_song=eval_records,
                device=device,
                batch_size=args.batch_size,
            ),
        }
        milestone_results = {
            "0": {
                "pass": 0,
                "update": 0,
                "evaluation": baseline,
                "parameterDriftRelative": 0.0,
                "outputChangeDb": 0.0,
                "batchNorm": batchnorm_summary(model),
                "checkpoint": None,
            }
        }

    model.train()
    started = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def save(path: Path, state_format: str, status: str) -> dict[str, Any]:
        return density.atomic_torch_save(
            path,
            checkpoint_payload(
                state_format=state_format,
                status=status,
                run_contract=run_contract,
                model=model,
                optimizer=optimizer,
                update=current_update,
                history=history,
                milestone_results=milestone_results,
                baseline=baseline,
                elapsed_seconds=elapsed(),
                resume_count=resume_count,
            ),
        )

    def evaluate_current_milestone(checkpoint: dict[str, Any]) -> None:
        pass_count = current_update // updates_per_pass
        evaluation = {
            "trainFitProbe": evaluate_matrix_model(
                name=f"{spec.key}-seed-{seed}@{pass_count}-train",
                model=model,
                spec=spec,
                songs=unique_songs(train_probe_records),
                records_by_song=train_probe_records,
                device=device,
                batch_size=args.batch_size,
            ),
            "trainHoldout": evaluate_matrix_model(
                name=f"{spec.key}-seed-{seed}@{pass_count}-holdout",
                model=model,
                spec=spec,
                songs=unique_songs(holdout_records),
                records_by_song=holdout_records,
                device=device,
                batch_size=args.batch_size,
            ),
            "calibrationInternal": evaluate_matrix_model(
                name=f"{spec.key}-seed-{seed}@{pass_count}-eval",
                model=model,
                spec=spec,
                songs=unique_songs(eval_records),
                records_by_song=eval_records,
                device=device,
                batch_size=args.batch_size,
            ),
        }
        current_probe = probe_predictions(
            model, spec, probe_records, device, args.batch_size
        )
        milestone_results[str(pass_count)] = {
            "pass": pass_count,
            "update": current_update,
            "evaluation": evaluation,
            "deltasVsPass0": {
                key: aggressive.metric_delta(value, baseline[key])
                for key, value in evaluation.items()
            },
            "parameterDriftRelative": sweep.parameter_drift_relative(
                model, initial_parameters
            ),
            "outputChangeDb": relative_output_change_db(
                initial_probe, current_probe
            ),
            "batchNorm": batchnorm_summary(model),
            "checkpoint": checkpoint,
        }
        aggregate = evaluation["calibrationInternal"]["aggregate"]
        print(
            json.dumps(
                {
                    "variant": spec.key,
                    "seed": seed,
                    "pass": pass_count,
                    "targetSdrDb": aggregate["aggressiveTargetSdrDb"],
                    "teacherResidualSdrDb": aggregate[
                        "teacherRemovalResidualSdrDb"
                    ],
                    "instrumentalSdrDb": aggregate["instrumentalSdrDb"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model.train()

    if (
        current_update in milestone_updates
        and current_update > 0
        and str(current_update // updates_per_pass) not in milestone_results
    ):
        checkpoint_path = run_root / f"pass-{current_update}.pt"
        checkpoint = (
            density.file_metadata(checkpoint_path)
            if checkpoint_path.is_file()
            else save(checkpoint_path, CHECKPOINT_FORMAT, "milestone-checkpoint")
        )
        save(rolling_path, STATE_FORMAT, "evaluating")
        evaluate_current_milestone(checkpoint)
        save(rolling_path, STATE_FORMAT, "training")

    while current_update < total_updates:
        start = current_update * args.batch_size
        indices = schedule[start : start + args.batch_size]
        if len(indices) != args.batch_size:
            raise AssertionError("Incomplete training batch")
        batch = [train_records[int(index)] for index in indices]
        for record in batch:
            record.materialize()
            if record.input_spec is None or record.teacher_instrumental_spec is None:
                raise ValueError(
                    f"Incomplete matrix training record: {record.song.slug}:{record.start}"
                )
        inputs = torch.from_numpy(
            np.stack([record.input_spec for record in batch])
        ).to(device)
        targets = torch.from_numpy(
            np.stack([record.teacher_instrumental_spec for record in batch])
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        candidate = predict_instrumental(model, inputs, spec)
        loss = F.l1_loss(candidate, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss for {spec.key} seed={seed} update={current_update + 1}"
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

        if current_update in milestone_updates:
            checkpoint_path = run_root / f"pass-{current_update}.pt"
            checkpoint = save(
                checkpoint_path, CHECKPOINT_FORMAT, "milestone-checkpoint"
            )
            save(rolling_path, STATE_FORMAT, "evaluating")
            evaluate_current_milestone(checkpoint)
            save(rolling_path, STATE_FORMAT, "training")
        elif current_update % args.state_interval_updates == 0:
            save(rolling_path, STATE_FORMAT, "training")

        if (
            args.stop_after_updates is not None
            and current_update >= args.stop_after_updates
        ):
            save(rolling_path, STATE_FORMAT, "paused")
            return {
                "status": "paused",
                "variant": asdict(spec),
                "seed": seed,
                "update": current_update,
                "pass": current_update / updates_per_pass,
                "totalUpdates": total_updates,
                "updatesPerPass": updates_per_pass,
                "initialization": initialization_metadata,
                "milestones": milestone_results,
                "lossByPass": aggregate_history_by_pass(history, updates_per_pass),
                "elapsedSeconds": elapsed(),
                "resumeCount": resume_count,
            }

    save(rolling_path, STATE_FORMAT, "completed")
    return {
        "status": "completed",
        "variant": asdict(spec),
        "seed": seed,
        "update": current_update,
        "pass": current_update / updates_per_pass,
        "totalUpdates": total_updates,
        "updatesPerPass": updates_per_pass,
        "initialization": initialization_metadata,
        "milestones": milestone_results,
        "lossByPass": aggregate_history_by_pass(history, updates_per_pass),
        "elapsedSeconds": elapsed(),
        "resumeCount": resume_count,
        "finalCheckpoint": milestone_results[str(args.passes)]["checkpoint"],
    }


MATRIX_METRIC_KEYS = (
    "aggressiveTargetSdrDb",
    "teacherRemovalResidualSdrDb",
    "instrumentalSdrDb",
    "lowVocalInstrumentalSdrDb",
    "accompanimentVocalProjectionDb",
    "teacherSpectrumL1",
)


def summarize_matrix(
    runs: dict[str, dict[str, Any]],
    variants: Sequence[VariantSpec],
    seeds: Sequence[int],
    milestone_passes: Sequence[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for spec in variants:
        variant_summary: dict[str, Any] = {}
        for pass_count in milestone_passes:
            values_by_metric: dict[str, list[float]] = {
                key: [] for key in MATRIX_METRIC_KEYS
            }
            per_seed: dict[str, Any] = {}
            for seed in seeds:
                run = runs.get(f"seed-{seed}/{spec.key}")
                if run is None or str(pass_count) not in run.get("milestones", {}):
                    continue
                metrics = run["milestones"][str(pass_count)]["evaluation"][
                    "calibrationInternal"
                ]["aggregate"]
                per_seed[str(seed)] = {key: float(metrics[key]) for key in MATRIX_METRIC_KEYS}
                for key in MATRIX_METRIC_KEYS:
                    values_by_metric[key].append(float(metrics[key]))
            variant_summary[str(pass_count)] = {
                "seedCount": len(per_seed),
                "perSeed": per_seed,
                "aggregateAcrossSeeds": {
                    key: {
                        "mean": float(np.mean(values)),
                        "sampleStdDev": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        ),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                    for key, values in values_by_metric.items()
                    if values
                },
            }
        summary[spec.key] = variant_summary
    return summary


def make_gate(summary: dict[str, Any], final_pass: int) -> dict[str, Any]:
    rd = summary.get("R-D", {}).get(str(final_pass), {})
    wr = summary.get("W-R", {}).get(str(final_pass), {})
    rd_aggregate = rd.get("aggregateAcrossSeeds", {})
    wr_aggregate = wr.get("aggregateAcrossSeeds", {})
    complete = rd.get("seedCount") == 3 and wr.get("seedCount") == 3
    if not complete:
        return {
            "passed": False,
            "reason": "R-D and W-R do not both have three completed seeds",
            "renderTwelveSongListening": False,
        }
    rd_target = rd_aggregate["aggressiveTargetSdrDb"]["mean"]
    wr_target = wr_aggregate["aggressiveTargetSdrDb"]["mean"]
    rd_residual = rd_aggregate["teacherRemovalResidualSdrDb"]["mean"]
    wr_residual = wr_aggregate["teacherRemovalResidualSdrDb"]["mean"]
    passed = rd_target >= wr_target - 1.0 and rd_residual >= wr_residual - 1.0
    return {
        "passed": passed,
        "criterion": (
            "At final pass, three-seed R-D mean must be within 1 dB of W-R "
            "for both aggressive-target and teacher-removal-residual SDR"
        ),
        "rDMinusWR": {
            "aggressiveTargetSdrDb": rd_target - wr_target,
            "teacherRemovalResidualSdrDb": rd_residual - wr_residual,
        },
        "renderTwelveSongListening": passed,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    variants, seeds, milestone_passes = validate_args(args)
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        raise ValueError("This matrix requires exactly ten selected train songs")

    (
        train_songs,
        eval_songs,
        train_records,
        eval_records,
        holdout_records,
        _selected_records,
        window_report,
    ) = scale10.prepare_compact_data(
        data_root,
        eval_root,
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
        raise AssertionError("Official final-test data entered the matrix")
    train_probe_records = records_by_song_subset(
        train_records, args.train_probe_windows_per_song
    )
    probe_records: list[pilot.WindowRecord] = []
    for records in list(eval_records.values())[: args.probe_song_count]:
        if records:
            probe_records.append(records[0])
    if len(probe_records) != args.probe_song_count:
        raise ValueError("Unable to construct the fixed output probe")

    experiment_root = args.experiment_root.resolve()
    report_path = experiment_root / "reports" / "inst3-init-output-matrix-report.json"
    runs: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "running",
        "experimentId": EXPERIMENT_ID,
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedWeights": "local teacher-derived checkpoints; do not publish",
        },
        "matrix": {
            "variants": [asdict(spec) for spec in variants],
            "seeds": list(seeds),
            "passes": args.passes,
            "milestonePasses": list(milestone_passes),
            "trainWindowCount": len(train_records),
            "batchSize": args.batch_size,
            "updatesPerPass": len(train_records) // args.batch_size,
            "learningRate": args.learning_rate,
            "precision": "float32",
            "batchNormPolicy": "updated in train mode for every matrix cell",
            "controlledFactors": ["initialization", "output semantic"],
        },
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
        "windowSelection": window_report,
        "runs": runs,
        "summary": None,
        "gate": None,
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
    density.atomic_json_write(report_path, report)

    for seed in seeds:
        schedule = build_sample_schedule(
            len(train_records), args.passes, make_schedule_seed(seed)
        )
        validate_schedule(schedule, len(train_records), args.passes)
        for spec in variants:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"matrix run: seed={seed} variant={spec.key}", flush=True)
            run_contract = make_run_contract(
                args=args,
                spec=spec,
                seed=seed,
                schedule=schedule,
                manifest_path=manifest_path,
                selection_path=selection_path,
                window_report=window_report,
                train_window_count=len(train_records),
                device=device,
            )
            result = train_run(
                args=args,
                spec=spec,
                seed=seed,
                run_contract=run_contract,
                schedule=schedule,
                train_records=train_records,
                train_probe_records=train_probe_records,
                eval_records=eval_records,
                holdout_records=holdout_records,
                probe_records=probe_records,
                device=device,
            )
            result["runContract"] = run_contract
            runs[f"seed-{seed}/{spec.key}"] = result
            report["runs"] = runs
            report["status"] = (
                "paused" if result["status"] == "paused" else "running"
            )
            density.atomic_json_write(report_path, report)
            if result["status"] != "completed":
                print(
                    json.dumps(
                        {
                            "status": "paused",
                            "report": str(report_path),
                            "run": f"seed-{seed}/{spec.key}",
                            "update": result["update"],
                        },
                        indent=2,
                    )
                )
                return 0
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize_matrix(runs, variants, seeds, milestone_passes)
    gate = make_gate(summary, args.passes)
    report["summary"] = summary
    report["gate"] = gate
    report["status"] = "completed"
    density.atomic_json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "runCount": len(runs),
                "gate": gate,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
