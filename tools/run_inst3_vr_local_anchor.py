#!/usr/bin/env python3
"""Fine-tune the H50 residual-vocals student only around Inst 3 hard events.

This is a local, non-commercial experiment.  Both arms start from the exact
V-R-H50 pass-50 checkpoint and use the frozen H50 schedule:

* H50-continuation: the existing Inst 3 residual target on every frame;
* H50-local-anchor: the Inst 3 target on selected event frames and the frozen
  H50 output everywhere else.  An optional soft guard can extend the selected
  event region without changing the fixed training-window schedule.

The runner reuses the H50 spectral cache and never regenerates teacher audio.
It evaluates the resulting checkpoints on the song-disjoint holdout and can
render the private twelve-song listening set.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_scale10_density as density
import run_inst3_vr_event_weighted as weighted
import run_inst3_vr_hard_sampling as hard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-sampling-h50"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-local-anchor"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_EVAL_ROOT / "musdb18-inst3-oracle-manifest.json"
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_H50_CHECKPOINT = (
    DEFAULT_BASE_ROOT / "runs" / "V-R-H50" / "step-8000.pt"
)
DEFAULT_LAMBDA_CHECKPOINT = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-event-weighted"
    / "runs"
    / "lambda-0.50"
    / "step-8000.pt"
)
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"

CONTINUATION = "H50-continuation"
LOCAL_ANCHOR = "H50-local-anchor"
VARIANTS = (CONTINUATION, LOCAL_ANCHOR)
SCHEMA = "local-inst3-vr-local-anchor@2"


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
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--lambda-checkpoint", type=Path, default=DEFAULT_LAMBDA_CHECKPOINT)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--milestones", default="0,1,2,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=1.0)
    parser.add_argument(
        "--guard-ms",
        type=float,
        default=0.0,
        help=(
            "Linear soft guard width on each side of the existing event-frame "
            "region; zero preserves the original boolean mask"
        ),
    )
    parser.add_argument("--state-interval", type=int, default=100)
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


def file_sha256(path: Path) -> str:
    return pilot.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return __import__("hashlib").sha256(encoded).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_state_model(
    *,
    architecture_checkpoint: Path,
    state_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing stateDict in {state_path}")
    model, initialization = pilot.make_model(architecture_checkpoint, device)
    model.load_state_dict(state_dict, strict=True)
    sweep.freeze_batchnorm_running_statistics(model)
    model.eval()
    return model, {
        "file": str(state_path.resolve()),
        "sha256": file_sha256(state_path),
        "format": payload.get("format"),
        "variant": payload.get("variant"),
        "step": int(payload.get("step", -1)),
        "initialization": initialization,
    }


def build_frozen_schedule(
    *,
    base_root: Path,
    event_root: Path,
    manifest_path: Path,
    max_songs: int | None,
    seed: int,
    passes: int,
    batch_size: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, hard.SongSelection],
    list[hard.ScheduleItem],
    dict[str, Any],
]:
    base_report = weighted.load_base_report(base_root)
    manifest = hard.load_manifest(manifest_path)
    train_entries = hard.load_train_entries(manifest, max_songs)
    selections, selection_payload, selection_sha = hard.build_selections(
        entries=train_entries,
        event_root=event_root,
        train_windows_per_song=8,
        hard_fraction=0.5,
        seed=seed,
    )
    if max_songs is None:
        expected_selection = base_report["contract"]["selectionSha256"]
        if selection_sha != expected_selection:
            raise ValueError(
                f"H50 selection changed: {selection_sha} != {expected_selection}"
            )
    schedule = hard.build_training_schedule(
        selections,
        "V-R-H50",
        passes,
        seed,
        hard_variant="V-R-H50",
    )
    schedule_summary = hard.summarize_schedule(
        selections, {"V-R-H50": schedule}, passes, batch_size
    )["V-R-H50"]
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
        schedule,
        {
            "manifest": manifest,
            "baseReport": base_report,
            "selection": selection_payload,
            "selectionSha256": selection_sha,
            "schedule": schedule_summary,
            "baseReportSha256": file_sha256(
                base_root / "reports" / "inst3-vr-hard-sampling-report.json"
            ),
        },
    )


def masked_mean_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} != {target.shape}")
    if mask.ndim != 2 or mask.shape[0] != prediction.shape[0] or mask.shape[1] != prediction.shape[-1]:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("Mask weights must be in the [0, 1] range")
    expanded = mask.to(dtype=prediction.dtype)[:, None, None, :]
    denominator = expanded.sum() * prediction.shape[1] * prediction.shape[2]
    return (prediction - target).abs().mul(expanded).sum() / denominator.clamp_min(1.0)


def local_anchor_loss(
    prediction: torch.Tensor,
    teacher_target: torch.Tensor,
    anchor_target: torch.Tensor,
    event_mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != anchor_target.shape:
        raise ValueError(f"Prediction/anchor mismatch: {prediction.shape} != {anchor_target.shape}")
    event_weights = event_mask.to(dtype=prediction.dtype)
    event_loss = masked_mean_l1(prediction, teacher_target, event_weights)
    non_event_loss = masked_mean_l1(
        prediction, anchor_target, 1.0 - event_weights
    )
    return event_loss + float(beta) * non_event_loss, event_loss, non_event_loss


def soft_event_frame_mask(
    *,
    candidate_start: int,
    candidate_length: int,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    guard_ms: float,
    sample_rate: int = 44_100,
) -> np.ndarray:
    """Return the existing event mask with a linear temporal guard.

    The FFT-support-overlap region used by ``weighted.event_frame_mask`` is
    the full-weight core.  Frames outside that core receive a linearly
    decaying weight over ``guard_ms`` on either side.  Using the same core
    makes the zero-width case exactly equivalent to the previous boolean
    mask and avoids weakening frames that were already selected.
    """
    if guard_ms < 0.0:
        raise ValueError("guard_ms must be non-negative")
    base = weighted.event_frame_mask(
        candidate_start=candidate_start,
        candidate_length=candidate_length,
        event_starts=event_starts,
        event_ends=event_ends,
        sample_rate=sample_rate,
    )
    if guard_ms == 0.0:
        return base

    frame_centers = (
        np.arange(pilot.DEFAULT_CONFIG.num_frames, dtype=np.float64)
        * pilot.DEFAULT_CONFIG.hop_length
    )
    half_fft = pilot.DEFAULT_CONFIG.n_fft / 2.0
    segment_start = int(candidate_start)
    segment_end = segment_start + int(candidate_length)
    guard_samples = float(guard_ms) * float(sample_rate) / 1000.0
    if guard_samples <= 0.0:
        return base

    weights = np.zeros(pilot.DEFAULT_CONFIG.num_frames, dtype=np.float32)
    for start, end in zip(event_starts, event_ends):
        overlap_start = max(segment_start, int(start))
        overlap_end = min(segment_end, int(end))
        if overlap_end <= overlap_start:
            continue
        local_start = overlap_start - segment_start + pilot.DEFAULT_CONFIG.trim_samples
        local_end = overlap_end - segment_start + pilot.DEFAULT_CONFIG.trim_samples
        core_start = float(local_start) - half_fft
        core_end = float(local_end) + half_fft
        distance = np.where(
            frame_centers < core_start,
            core_start - frame_centers,
            np.where(frame_centers > core_end, frame_centers - core_end, 0.0),
        )
        contribution = np.clip(1.0 - distance / guard_samples, 0.0, 1.0)
        weights = np.maximum(weights, contribution.astype(np.float32))

    # Preserve the exact selected region even at strict inequality boundaries.
    weights[base] = 1.0
    return np.ascontiguousarray(weights, dtype=np.float32)


class GuardedCacheStore(weighted.WeightedCacheStore):
    """H50 cache store that optionally exposes fractional event weights."""

    def __init__(
        self,
        cache_paths: dict[str, Path],
        event_root: Path,
        selections: dict[str, hard.SongSelection],
        guard_ms: float,
    ) -> None:
        super().__init__(cache_paths, event_root, selections)
        if guard_ms < 0.0:
            raise ValueError("guard_ms must be non-negative")
        self.guard_ms = float(guard_ms)

    def _load(self, slug: str) -> dict[str, np.ndarray]:
        cached = self._open.get(slug)
        if cached is not None:
            self._open.move_to_end(slug)
            return cached
        cached = super()._load(slug)
        if self.guard_ms == 0.0:
            return cached

        selection = self.selections[slug]
        event_starts, event_ends, _event_scores, _threshold = (
            weighted.load_event_mask_blocks(self.event_root, slug)
        )
        with np.load(self.paths[slug]) as values:
            starts = np.ascontiguousarray(values["starts"], dtype=np.int64)
            lengths = np.ascontiguousarray(values["lengths"], dtype=np.int64)
        expected_indices = np.asarray(selection.union_indices, dtype=np.int64)
        if starts.shape[0] != len(expected_indices) or lengths.shape[0] != len(expected_indices):
            raise ValueError(f"Unexpected H50 cache metadata shape for {slug}")
        masks = np.stack(
            [
                soft_event_frame_mask(
                    candidate_start=int(start),
                    candidate_length=int(length),
                    event_starts=event_starts,
                    event_ends=event_ends,
                    guard_ms=self.guard_ms,
                )
                for start, length in zip(starts, lengths)
            ],
            axis=0,
        )
        cached = {
            **cached,
            "eventFrameMask": np.ascontiguousarray(masks, dtype=np.float32),
        }
        self._open[slug] = cached
        self._open.move_to_end(slug)
        self.mask_stats[slug].update(
            {
                "guardMs": self.guard_ms,
                "maskDtype": "float32",
                "weightedFrameSum": float(masks.sum()),
                "weightedFrameFraction": float(masks.mean()),
            }
        )
        return cached


def latest_checkpoint(
    run_root: Path, contract_id: str
) -> tuple[Path, dict[str, Any]] | None:
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


def train_variant(
    *,
    variant: str,
    mode: str,
    h50_checkpoint: Path,
    architecture_checkpoint: Path,
    store: weighted.WeightedCacheStore,
    schedule: list[hard.ScheduleItem],
    run_root: Path,
    contract_id: str,
    passes: int,
    milestones: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    anchor_beta: float,
    guard_ms: float,
    seed: int,
    device: torch.device,
    state_interval: int,
    resume: bool,
    anchor_model: torch.nn.Module | None,
) -> dict[str, Any]:
    if variant not in VARIANTS or mode not in {"continuation", "local-anchor"}:
        raise ValueError(f"Unsupported variant/mode: {variant}/{mode}")
    records_per_pass = len(schedule) // passes
    if records_per_pass % batch_size != 0:
        raise ValueError("Records per pass must be divisible by batch size")
    updates_per_pass = records_per_pass // batch_size
    total_updates = len(schedule) // batch_size
    milestone_updates = {
        pass_count * updates_per_pass: pass_count for pass_count in milestones
    }
    run_root.mkdir(parents=True, exist_ok=True)
    hard.set_seed(seed)
    model, source_metadata = load_state_model(
        architecture_checkpoint=architecture_checkpoint,
        state_path=h50_checkpoint,
        device=device,
    )
    if mode == "local-anchor" and anchor_model is None:
        raise ValueError("Local-anchor training requires an anchor model")
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
                    {"event": "resume", "variant": variant, "step": current_update, "file": str(path)}
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
            "format": "local-inst3-vr-local-anchor-checkpoint@1",
            "status": status,
            "variant": variant,
            "mode": mode,
            "runContractId": contract_id,
            "step": step,
            "passes": passes,
            "recordsPerPass": records_per_pass,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "anchorBeta": anchor_beta,
            "guardMs": guard_ms,
            "seed": seed,
            "stateDict": hard.cpu_tree(model.state_dict()),
            "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
            "history": history,
            "resumeCount": resume_count,
            "elapsedSeconds": time.perf_counter() - started,
            "checkpointSource": source_metadata,
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
        predicted = model(input_tensor)
        if mode == "continuation":
            loss = F.l1_loss(predicted, target_tensor)
            event_loss = loss
            anchor_loss = torch.zeros((), device=device, dtype=loss.dtype)
            event_fraction = 1.0
        else:
            assert anchor_model is not None
            with torch.no_grad():
                anchor_target = anchor_model(input_tensor)
            loss, event_loss, anchor_loss = local_anchor_loss(
                predicted,
                target_tensor,
                anchor_target,
                mask_tensor,
                anchor_beta,
            )
            event_fraction = float(mask_array.mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {current_update + 1}")
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
                "eventLoss": float(event_loss.detach().cpu().item()),
                "anchorLoss": float(anchor_loss.detach().cpu().item()),
                "eventFrameFraction": event_fraction,
                "gradientNormBeforeClip": gradient_norm,
            }
        )
        if current_update == 1 or current_update % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "variant": variant,
                        "update": current_update,
                        "totalUpdates": total_updates,
                        "loss": history[-1]["loss"],
                        "eventLoss": history[-1]["eventLoss"],
                        "anchorLoss": history[-1]["anchorLoss"],
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
                        "variant": variant,
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
        "variant": variant,
        "mode": mode,
        "status": "completed",
        "passes": passes,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": updates_per_pass,
        "updates": total_updates,
        "learningRate": learning_rate,
        "anchorBeta": anchor_beta,
        "batchSize": batch_size,
        "seed": seed,
        "history": history,
        "milestoneCheckpoints": milestone_files,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
        "resumeCount": resume_count,
    }


def evaluate_checkpoint(
    *,
    name: str,
    state_path: Path,
    architecture_checkpoint: Path,
    entries: list[dict[str, Any]],
    eval_root: Path,
    eval_windows_per_song: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = load_state_model(
        architecture_checkpoint=architecture_checkpoint,
        state_path=state_path,
        device=device,
    )
    try:
        result = hard.evaluate_model_state(
            name=name,
            model=model,
            entries=entries,
            eval_root=eval_root,
            eval_windows_per_song=eval_windows_per_song,
            seed=seed,
            device=device,
        )
        result["checkpoint"] = metadata
        return result
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate_initial(
    *,
    architecture_checkpoint: Path,
    entries: list[dict[str, Any]],
    eval_root: Path,
    eval_windows_per_song: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = pilot.make_model(architecture_checkpoint, device)
    sweep.freeze_batchnorm_running_statistics(model)
    model.eval()
    try:
        result = hard.evaluate_model_state(
            name="initial-vocals",
            model=model,
            entries=entries,
            eval_root=eval_root,
            eval_windows_per_song=eval_windows_per_song,
            seed=seed,
            device=device,
        )
        result["checkpoint"] = metadata
        return result
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def anchor_preservation(
    *,
    state_path: Path,
    architecture_checkpoint: Path,
    anchor_model: torch.nn.Module,
    store: weighted.WeightedCacheStore,
    device: torch.device,
) -> dict[str, Any]:
    model, _ = load_state_model(
        architecture_checkpoint=architecture_checkpoint,
        state_path=state_path,
        device=device,
    )
    error_power = 0.0
    anchor_power = 0.0
    event_error_power = 0.0
    event_anchor_power = 0.0
    non_event_frames = 0.0
    event_frames = 0.0
    with torch.inference_mode():
        for slug in sorted(store.paths):
            arrays = store._load(slug)
            inputs = arrays["inputSpec"]
            masks = arrays["eventFrameMask"]
            for begin in range(0, len(inputs), 4):
                input_tensor = torch.from_numpy(inputs[begin : begin + 4]).to(device)
                prediction = model(input_tensor)
                anchor = anchor_model(input_tensor)
                difference = (prediction - anchor).detach().to(dtype=torch.float64)
                anchor64 = anchor.detach().to(dtype=torch.float64)
                mask = torch.from_numpy(masks[begin : begin + 4]).to(
                    device=device, dtype=difference.dtype
                )
                event_weights = mask[:, None, None, :].expand_as(difference)
                non_event_weights = 1.0 - event_weights
                error_power += float(
                    (difference.square() * non_event_weights).sum().cpu()
                )
                anchor_power += float(
                    (anchor64.square() * non_event_weights).sum().cpu()
                )
                event_error_power += float(
                    (difference.square() * event_weights).sum().cpu()
                )
                event_anchor_power += float(
                    (anchor64.square() * event_weights).sum().cpu()
                )
                non_event_frames += float(non_event_weights.sum().cpu())
                event_frames += float(event_weights.sum().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    def db(value: float) -> float:
        return 20.0 * math.log10(max(math.sqrt(value), 1.0e-12))
    return {
        "nonEventCoefficientRmsDbfs": db(error_power / max(non_event_frames, 1)),
        "nonEventAnchorRmsDbfs": db(anchor_power / max(non_event_frames, 1)),
        "eventCoefficientRmsDbfs": db(event_error_power / max(event_frames, 1)),
        "eventAnchorRmsDbfs": db(event_anchor_power / max(event_frames, 1)),
        "nonEventValueCount": non_event_frames,
        "eventValueCount": event_frames,
    }


def render_student(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    useful = listening.DEFAULT_CONFIG.useful_samples
    output = np.empty_like(audio)
    window_count = (audio.shape[0] + useful - 1) // useful
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(window_count):
            start = index * useful
            length = min(useful, audio.shape[0] - start)
            input_spec = listening.student_window_input(audio, start, length)
            input_tensor = torch.from_numpy(input_spec).to(device)
            predicted_residual = model(input_tensor)
            predicted_instrumental = input_tensor - predicted_residual
            reconstructed = listening.student_istft_centered(
                predicted_instrumental.detach().cpu().numpy()
            )
            trim = listening.DEFAULT_CONFIG.trim_samples
            output[start : start + length] = reconstructed[trim : trim + length]
    if not np.isfinite(output).all():
        raise ValueError("Student output contains non-finite values")
    return output, {
        "windowCount": window_count,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def render_listening(
    *,
    output_root: Path,
    samples_root: Path,
    architecture_checkpoint: Path,
    state_paths: dict[str, Path],
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    songs = density.validate_private_songs(samples_root.resolve())
    models: dict[str, torch.nn.Module] = {}
    report: dict[str, Any] = {
        "schema": "local-inst3-vr-local-anchor-listening@1",
        "status": "running",
        "semantic": "residual-vocals-to-instrumental",
        "format": "PCM16 FLAC",
        "outputRoot": str(output_root.resolve()),
        "models": {},
        "songs": {},
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, path in state_paths.items():
        model, metadata = load_state_model(
            architecture_checkpoint=architecture_checkpoint,
            state_path=path,
            device=device,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
        report["models"][name] = {**metadata, "checkpoint": checkpoint_metadata(path)}
    try:
        for index, song in enumerate(songs, start=1):
            source, sample_rate = listening.load_audio(Path(song["file"]))
            if sample_rate != listening.DEFAULT_CONFIG.sample_rate:
                raise ValueError(f"Unexpected sample rate: {song['file']}")
            name = song["name"]
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
                    rendered, timing = render_student(source, model, device)
                    listening.write_flac(path, rendered, sample_rate)
                    del rendered
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
            print(f"listening {index}/{len(songs)}: {name}", flush=True)
            del source
            gc.collect()
    finally:
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "completed"
    report["outputCount"] = sum(
        len(song["outputs"]) for song in report["songs"].values()
    )
    json_write(output_root / "render-report.json", report)
    return report


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.passes <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("passes, batch-size, and threads must be positive")
    if (
        args.learning_rate <= 0.0
        or args.anchor_beta < 0.0
        or args.guard_ms < 0.0
        or args.state_interval <= 0
    ):
        raise ValueError("learning-rate, anchor-beta, and state-interval must be valid")
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


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_int_list(args.milestones)
    validate_args(args, milestones)
    base_root = args.base_root.resolve()
    output_root = args.output_root.resolve()
    event_root = args.event_root.resolve()
    eval_root = args.eval_root.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    h50_checkpoint = args.h50_checkpoint.resolve()
    lambda_checkpoint = args.lambda_checkpoint.resolve()
    for path in (checkpoint, h50_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
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
        schedule,
        frozen,
    ) = build_frozen_schedule(
        base_root=base_root,
        event_root=event_root,
        manifest_path=manifest_path,
        max_songs=args.max_songs,
        seed=args.seed,
        passes=args.passes,
        batch_size=args.batch_size,
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    cache_paths = {
        slug: base_root / "cache" / "train" / f"{slug}.npz"
        for slug in selections
    }
    for path in cache_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    store = GuardedCacheStore(
        cache_paths,
        event_root,
        selections,
        guard_ms=args.guard_ms,
    )
    store.preload()

    anchor_model, anchor_metadata = load_state_model(
        architecture_checkpoint=checkpoint,
        state_path=h50_checkpoint,
        device=device,
    )
    for parameter in anchor_model.parameters():
        parameter.requires_grad_(False)
    anchor_model.eval()

    common_contract = {
        "schema": SCHEMA,
        "baseReportSha256": frozen["baseReportSha256"],
        "baseSelectionSha256": frozen["selectionSha256"],
        "scheduleSha256": frozen["schedule"]["scheduleSha256"],
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "studentCheckpointSha256": file_sha256(checkpoint),
        "trainSongCount": len(train_entries),
        "evalSongCount": len(eval_entries),
        "trainWindowsPerSong": 8,
        "hardVariant": "V-R-H50",
        "hardFraction": 0.5,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "guardMs": args.guard_ms,
        "seed": args.seed,
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "eventMask": (
            "existing H50 selected 100 ms event-frame support region with "
            "linear soft guard"
            if args.guard_ms > 0.0
            else "existing H50 selected 100 ms event frame mask"
        ),
        "eventMaskSemantics": (
            "float32 weights: 1.0 on the existing FFT-support-overlap core, "
            "linear decay to 0 over guardMs on either side"
            if args.guard_ms > 0.0
            else "boolean selected FFT-support-overlap frames"
        ),
        "officialFinalTestUsed": False,
    }
    training_results: dict[str, dict[str, Any]] = {}
    for variant, mode in (
        (CONTINUATION, "continuation"),
        (LOCAL_ANCHOR, "local-anchor"),
    ):
        variant_contract = {
            **common_contract,
            "variant": variant,
            "mode": mode,
        }
        contract_id = canonical_sha256(variant_contract)
        training_results[variant] = train_variant(
            variant=variant,
            mode=mode,
            h50_checkpoint=h50_checkpoint,
            architecture_checkpoint=checkpoint,
            store=store,
            schedule=schedule,
            run_root=output_root / "runs" / variant,
            contract_id=contract_id,
            passes=args.passes,
            milestones=milestones,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            anchor_beta=args.anchor_beta,
            guard_ms=args.guard_ms,
            seed=args.seed,
            device=device,
            state_interval=args.state_interval,
            resume=args.resume,
            anchor_model=anchor_model if mode == "local-anchor" else None,
        )
        training_results[variant]["contract"] = {
            "id": contract_id,
            "payload": variant_contract,
        }

    # Evaluate the original and the two established controls as well as every
    # requested local milestone, using the same holdout window selection.
    evaluation: dict[str, Any] = {}
    evaluation["initial-vocals"] = evaluate_initial(
        architecture_checkpoint=checkpoint,
        entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=args.eval_windows_per_song,
        seed=args.seed,
        device=device,
    )
    evaluation["H50-pass-50"] = evaluate_checkpoint(
        name="H50-pass-50",
        state_path=h50_checkpoint,
        architecture_checkpoint=checkpoint,
        entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=args.eval_windows_per_song,
        seed=args.seed,
        device=device,
    )
    if lambda_checkpoint.is_file():
        evaluation["lambda-0.50-pass-50"] = evaluate_checkpoint(
            name="lambda-0.50-pass-50",
            state_path=lambda_checkpoint,
            architecture_checkpoint=checkpoint,
            entries=eval_entries,
            eval_root=eval_root,
            eval_windows_per_song=args.eval_windows_per_song,
            seed=args.seed,
            device=device,
        )
    for variant, result in training_results.items():
        for pass_count in milestones:
            if pass_count == 0:
                continue
            step = pass_count * result["updatesPerPass"]
            metadata = result["milestoneCheckpoints"].get(str(step))
            if metadata is None:
                raise FileNotFoundError(f"Missing {variant} pass {pass_count}")
            evaluation[f"{variant}@pass-{pass_count}"] = evaluate_checkpoint(
                name=f"{variant}@pass-{pass_count}",
                state_path=Path(metadata["file"]),
                architecture_checkpoint=checkpoint,
                entries=eval_entries,
                eval_root=eval_root,
                eval_windows_per_song=args.eval_windows_per_song,
                seed=args.seed,
                device=device,
            )
    baseline = evaluation["H50-pass-50"]
    for name, result in evaluation.items():
        if name != "H50-pass-50":
            result["deltaVsH50"] = hard.metric_delta(result, baseline)

    final_anchor: dict[str, Any] = {}
    for variant in VARIANTS:
        result = training_results[variant]
        step = args.passes * result["updatesPerPass"]
        metadata = result["milestoneCheckpoints"][str(step)]
        final_anchor[variant] = anchor_preservation(
            state_path=Path(metadata["file"]),
            architecture_checkpoint=checkpoint,
            anchor_model=anchor_model,
            store=store,
            device=device,
        )

    listening_report: dict[str, Any] | None = None
    if not args.skip_listening:
        final_paths = {
            "H50-pass-50": h50_checkpoint,
            "lambda-0.50-pass-50": lambda_checkpoint,
        }
        for variant, result in training_results.items():
            step = args.passes * result["updatesPerPass"]
            final_paths[f"{variant}@pass-{args.passes}"] = Path(
                result["milestoneCheckpoints"][str(step)]["file"]
            )
        final_paths = {name: path for name, path in final_paths.items() if path.is_file()}
        listening_report = render_listening(
            output_root=output_root / "listening-12",
            samples_root=args.samples_root.resolve(),
            architecture_checkpoint=checkpoint,
            state_paths=final_paths,
            device=device,
            force=args.force_listening,
        )

    report = {
        "schema": SCHEMA,
        "status": "completed",
        "experimentId": SCHEMA,
        "contract": common_contract,
        "inputs": {
            "manifest": {
                "file": str(manifest_path),
                "sha256": file_sha256(manifest_path),
            },
            "eventRoot": str(event_root),
            "baseReport": {
                "file": str(
                    (base_root / "reports" / "inst3-vr-hard-sampling-report.json").resolve()
                ),
                "sha256": frozen["baseReportSha256"],
            },
            "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
            "lambdaCheckpoint": (
                checkpoint_metadata(lambda_checkpoint)
                if lambda_checkpoint.is_file()
                else None
            ),
            "architectureCheckpoint": checkpoint_metadata(checkpoint),
        },
        "schedule": frozen["schedule"],
        "maskStats": store.mask_stats,
        "training": training_results,
        "evaluation": evaluation,
        "anchorPreservation": final_anchor,
        "listening": listening_report,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device),
        },
        "notes": {
            "publication": "Local non-commercial research only; do not publish MUSDB18-derived checkpoints or audio.",
            "alignment": "No centered-window change; the preceding alignment diagnostic did not meet its improvement gate.",
            "anchor": "The H50 model is evaluated without gradients and supplies the non-event target.",
            "guard": (
                "The soft guard extends the existing event-frame support region "
                "with linear weights; it does not change candidate selection."
                if args.guard_ms > 0.0
                else "No soft guard; the original boolean event mask is used."
            ),
        },
    }
    json_write(output_root / "reports" / "inst3-vr-local-anchor-report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(
                    (output_root / "reports" / "inst3-vr-local-anchor-report.json").resolve()
                ),
                "trainingVariants": list(training_results),
                "evaluationVariants": list(evaluation),
                "listeningCount": listening_report.get("outputCount", 0)
                if listening_report
                else 0,
            },
            indent=2,
        ),
        flush=True,
    )
    del anchor_model, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
