#!/usr/bin/env python3
"""Train H50 residual students with a short-event audio-domain objective.

This local, non-commercial experiment reuses the completed H50
event-centered spectral cache.  It converts the cached Inst 3 and H50 packed
spectra to PCM once, then trains through a differentiable torch.iSTFT:

* ``audio-event-core``: exact 50 ms event target, H50 anchor outside it;
* ``audio-event-guard25``: exact event target with a 25 ms linear target
  guard, H50 anchor on the complement.

No teacher inference or source-audio decoding is performed here.  All caches,
checkpoints, reports, and listening files remain local research artifacts.
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

import run_inst3_distill_pilot as pilot
import run_inst3_vr_event_centered as centered
import run_inst3_vr_hard_sampling as hard
import validate_tfc_tdf_default_audio as dsp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-event-centered"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-event-audio"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_EVAL_ROOT / "musdb18-inst3-oracle-manifest.json"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_H50_CHECKPOINT = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-hard-sampling-h50"
    / "runs"
    / "V-R-H50"
    / "step-8000.pt"
)
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"

SCHEMA = "local-inst3-vr-event-audio@1"
CACHE_SCHEMA = "local-inst3-vr-event-audio-cache@1"
CORE = "audio-event-core"
GUARD25 = "audio-event-guard25"
VARIANTS = (CORE, GUARD25)
MILESTONES = (0, 1, 2, 5)
SAMPLE_RATE = pilot.DEFAULT_CONFIG.sample_rate
USEFUL_SAMPLES = pilot.DEFAULT_CONFIG.useful_samples


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integers, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--milestones", default="0,1,2,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=1.0)
    parser.add_argument("--guard-ms", type=float, default=25.0)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--skip-listening", action="store_true")
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


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def torch_packed_istft(packed: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Convert packed [B,4,F,T] spectra to [B,model_input_samples,2] PCM."""
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


def charcoal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Weighted audio Charbonnier loss, normalized by active samples/channels."""
    if prediction.shape != target.shape:
        raise ValueError(f"Audio target mismatch: {prediction.shape} != {target.shape}")
    if weights.ndim != 2 or weights.shape != prediction.shape[:2]:
        raise ValueError(f"Audio weight mismatch: {weights.shape} != {prediction.shape[:2]}")
    if torch.any(weights < 0.0) or torch.any(weights > 1.0):
        raise ValueError("Audio weights must be in [0, 1]")
    weight = weights.to(dtype=prediction.dtype).unsqueeze(-1)
    error = torch.sqrt((prediction - target).square() + float(epsilon) ** 2)
    denominator = weight.sum() * prediction.shape[-1]
    return (error * weight).sum() / denominator.clamp_min(1.0)


def build_event_weights(
    *,
    starts: Sequence[int],
    ends: Sequence[int],
    guard_ms: float,
    useful_samples: int = USEFUL_SAMPLES,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact core and fractional target weights in useful PCM samples."""
    if len(starts) != len(ends):
        raise ValueError("Event start/end count mismatch")
    if guard_ms < 0.0:
        raise ValueError("guard_ms must be non-negative")
    index = np.arange(useful_samples, dtype=np.float64)
    core = np.zeros((len(starts), useful_samples), dtype=bool)
    guarded = np.zeros((len(starts), useful_samples), dtype=np.float32)
    guard_samples = float(guard_ms) * sample_rate / 1000.0
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i <= start_i or end_i > useful_samples:
            raise ValueError(f"Event outside useful window: {start_i}:{end_i}")
        core[row, start_i:end_i] = True
        if guard_samples == 0.0:
            guarded[row, start_i:end_i] = 1.0
            continue
        distance = np.maximum(
            np.maximum(float(start_i) - index, index - float(end_i - 1)), 0.0
        )
        guarded[row] = np.clip(1.0 - distance / guard_samples, 0.0, 1.0).astype(
            np.float32
        )
    return core, np.ascontiguousarray(guarded, dtype=np.float32)


def load_centered_records(
    event_root: Path,
    max_songs: int | None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, int]]], str]:
    selection_path = event_root / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "local-inst3-vr-event-centered@1":
        raise ValueError(f"Unexpected event-centered selection: {selection.get('schema')}")
    songs = selection.get("songs")
    if not isinstance(songs, dict) or len(songs) != 80:
        raise ValueError(f"Expected 80 event-centered songs, found {len(songs or {})}")
    ordered = sorted(songs)
    if max_songs is not None:
        if max_songs <= 0:
            raise ValueError("max-songs must be positive")
        ordered = ordered[:max_songs]
    records_by_song: dict[str, list[dict[str, int]]] = {}
    entries: list[dict[str, Any]] = []
    for slug in ordered:
        value = songs[slug]
        centered_rows = value.get("centered")
        if not isinstance(centered_rows, list) or not centered_rows:
            raise ValueError(f"Missing centered rows for {slug}")
        rows: list[dict[str, int]] = []
        for index, row in enumerate(centered_rows):
            rows.append(
                {
                    "index": index,
                    "start": int(row["startSamples"]),
                    "length": int(row["lengthSamples"]),
                    "eventStart": int(row["eventStartSamples"]),
                    "eventEnd": int(row["eventEndSamples"]),
                }
            )
        if any(row["length"] != USEFUL_SAMPLES for row in rows):
            raise ValueError(f"Unexpected centered length for {slug}")
        records_by_song[slug] = rows
        entries.append({"slug": slug, "member": value["member"]})
    h50 = selection.get("h50Checkpoint", {})
    h50_sha = str(h50.get("sha256", ""))
    if not h50_sha:
        raise ValueError("Selection has no H50 checkpoint hash")
    return entries, records_by_song, h50_sha


def event_cache_path(event_root: Path, slug: str) -> tuple[Path, Path]:
    base = event_root / "cache" / "train" / slug
    return base.with_suffix(".npz"), base.with_suffix(".json")


def audio_cache_path(output_root: Path, slug: str) -> tuple[Path, Path]:
    base = output_root / "cache" / "train" / slug
    return base.with_suffix(".npz"), base.with_suffix(".json")


def prepare_audio_cache(
    *,
    event_root: Path,
    output_root: Path,
    slug: str,
    rows: Sequence[dict[str, int]],
    h50_checkpoint: Path,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    source_npz, source_json = event_cache_path(event_root, slug)
    output_npz, output_json = audio_cache_path(output_root, slug)
    if not source_npz.is_file() or not source_json.is_file():
        raise FileNotFoundError(f"Missing event-centered cache for {slug}")
    source_sha = file_sha256(source_npz)
    expected = {
        "schema": CACHE_SCHEMA,
        "sourceEventCacheSha256": source_sha,
        "h50CheckpointSha256": file_sha256(h50_checkpoint),
        "sampleRate": SAMPLE_RATE,
        "usefulSamples": USEFUL_SAMPLES,
        "eventCount": len(rows),
        "starts": [row["start"] for row in rows],
        "lengths": [row["length"] for row in rows],
        "eventStarts": [row["eventStart"] for row in rows],
        "eventEnds": [row["eventEnd"] for row in rows],
    }
    if not force and output_npz.is_file() and output_json.is_file():
        try:
            metadata = json.loads(output_json.read_text(encoding="utf-8"))
            with np.load(output_npz) as values:
                valid = (
                    metadata.get("contract") == expected
                    and values["targetAudio"].shape
                    == values["anchorAudio"].shape
                    == (len(rows), USEFUL_SAMPLES, 2)
                    and values["coreMask"].shape == (len(rows), USEFUL_SAMPLES)
                    and values["guard25Weights"].shape == (len(rows), USEFUL_SAMPLES)
                )
            if valid:
                return output_npz, metadata
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    with np.load(source_npz) as source:
        target_specs = np.ascontiguousarray(source["targetResidualSpec"], dtype=np.float32)
        anchor_specs = np.ascontiguousarray(source["anchorResidualSpec"], dtype=np.float32)
    if target_specs.shape != anchor_specs.shape or target_specs.shape[0] != len(rows):
        raise ValueError(f"Unexpected event cache shape for {slug}: {target_specs.shape}")
    target_audio: list[np.ndarray] = []
    anchor_audio: list[np.ndarray] = []
    for index, row in enumerate(rows):
        target_full = dsp.istft_centered(target_specs[index : index + 1])
        anchor_full = dsp.istft_centered(anchor_specs[index : index + 1])
        trim = pilot.DEFAULT_CONFIG.trim_samples
        target_audio.append(
            np.ascontiguousarray(target_full[trim : trim + USEFUL_SAMPLES], dtype=np.float32)
        )
        anchor_audio.append(
            np.ascontiguousarray(anchor_full[trim : trim + USEFUL_SAMPLES], dtype=np.float32)
        )
    event_starts = [row["eventStart"] - row["start"] for row in rows]
    event_ends = [row["eventEnd"] - row["start"] for row in rows]
    core, guard25 = build_event_weights(
        starts=event_starts,
        ends=event_ends,
        guard_ms=25.0,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_name(output_npz.name + ".tmp")
    np.savez_compressed(
        temporary,
        targetAudio=np.ascontiguousarray(np.stack(target_audio), dtype=np.float32),
        anchorAudio=np.ascontiguousarray(np.stack(anchor_audio), dtype=np.float32),
        coreMask=core,
        guard25Weights=guard25,
    )
    temporary_npz = temporary if temporary.suffix == ".npz" else Path(str(temporary) + ".npz")
    temporary_npz.replace(output_npz)
    metadata = {
        "contract": expected,
        "cache": checkpoint_metadata(output_npz),
        "dsp": {
            "sourceSpectrum": "event-centered targetResidualSpec and anchorResidualSpec",
            "inverse": "validate_tfc_tdf_default_audio.istft_centered",
            "nFft": pilot.DEFAULT_CONFIG.n_fft,
            "hopLength": pilot.DEFAULT_CONFIG.hop_length,
            "trimSamples": pilot.DEFAULT_CONFIG.trim_samples,
        },
        "events": [
            {
                "start": row["start"],
                "eventStart": row["eventStart"],
                "eventEnd": row["eventEnd"],
                "relativeStart": event_starts[index],
                "relativeEnd": event_ends[index],
            }
            for index, row in enumerate(rows)
        ],
    }
    json_write(output_json, metadata)
    return output_npz, metadata


class AudioCacheStore:
    def __init__(self, input_paths: dict[str, Path], audio_paths: dict[str, Path]) -> None:
        self.input_paths = input_paths
        self.audio_paths = audio_paths
        self._open: dict[str, dict[str, np.ndarray]] = {}

    def _load(self, slug: str) -> dict[str, np.ndarray]:
        cached = self._open.get(slug)
        if cached is not None:
            return cached
        with np.load(self.input_paths[slug]) as input_values:
            inputs = np.ascontiguousarray(input_values["inputSpec"], dtype=np.float32)
        with np.load(self.audio_paths[slug]) as audio_values:
            cached = {
                "inputSpec": inputs,
                "targetAudio": np.ascontiguousarray(audio_values["targetAudio"], dtype=np.float32),
                "anchorAudio": np.ascontiguousarray(audio_values["anchorAudio"], dtype=np.float32),
                "coreMask": np.ascontiguousarray(audio_values["coreMask"], dtype=np.float32),
                "guard25Weights": np.ascontiguousarray(audio_values["guard25Weights"], dtype=np.float32),
            }
        expected = cached["inputSpec"].shape[0]
        if any(value.shape[0] != expected for key, value in cached.items() if key != "inputSpec"):
            raise ValueError(f"Audio cache record count mismatch for {slug}")
        self._open[slug] = cached
        return cached

    def batch(
        self, items: Sequence[hard.ScheduleItem], event_weights: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if event_weights not in {"coreMask", "guard25Weights"}:
            raise ValueError(event_weights)
        arrays = [self._load(item.slug) for item in items]
        return (
            np.stack([array["inputSpec"][item.cache_index] for array, item in zip(arrays, items)]),
            np.stack([array["targetAudio"][item.cache_index] for array, item in zip(arrays, items)]),
            np.stack([array["anchorAudio"][item.cache_index] for array, item in zip(arrays, items)]),
            np.stack([array[event_weights][item.cache_index] for array, item in zip(arrays, items)]),
        )

    def preload(self) -> None:
        for index, slug in enumerate(sorted(self.input_paths), start=1):
            self._load(slug)
            if index == 1 or index == len(self.input_paths) or index % 10 == 0:
                print(f"preload audio cache {index}/{len(self.input_paths)}", flush=True)


def build_schedule(
    records_by_song: dict[str, list[dict[str, int]]], passes: int, seed: int
) -> list[hard.ScheduleItem]:
    schedule: list[hard.ScheduleItem] = []
    for pass_index in range(passes):
        items = [
            hard.ScheduleItem(slug, row["index"])
            for slug in sorted(records_by_song)
            for row in records_by_song[slug]
        ]
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        schedule.extend(items[int(index)] for index in rng.permutation(len(items)))
    return schedule


def latest_checkpoint(run_root: Path, contract_id: str) -> tuple[Path, dict[str, Any]] | None:
    best: tuple[int, Path, dict[str, Any]] | None = None
    for path in run_root.glob("step-*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("runContractId") != contract_id:
                continue
            candidate = (int(payload["step"]), path, payload)
        except (OSError, KeyError, RuntimeError, TypeError, ValueError):
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    return None if best is None else (best[1], best[2])


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    variant: str,
    contract_id: str,
    step: int,
    passes: int,
    records_per_pass: int,
    batch_size: int,
    learning_rate: float,
    anchor_beta: float,
    history: list[dict[str, Any]],
    source: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    payload = {
        "format": "local-inst3-vr-event-audio-checkpoint@1",
        "status": status,
        "variant": variant,
        "runContractId": contract_id,
        "step": step,
        "passes": passes,
        "recordsPerPass": records_per_pass,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "anchorBeta": anchor_beta,
        "stateDict": hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
        "history": history,
        "checkpointSource": source,
    }
    return hard.atomic_torch_save(path, payload)


def train_variant(
    *,
    variant: str,
    event_weights: str,
    h50_checkpoint: Path,
    architecture_checkpoint: Path,
    store: AudioCacheStore,
    schedule: list[hard.ScheduleItem],
    run_root: Path,
    contract_id: str,
    passes: int,
    milestones: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    anchor_beta: float,
    seed: int,
    device: torch.device,
    state_interval: int,
    resume: bool,
) -> dict[str, Any]:
    records_per_pass = len(schedule) // passes
    if records_per_pass % batch_size:
        raise ValueError("Audio schedule is not divisible by batch size")
    updates_per_pass = records_per_pass // batch_size
    total_updates = len(schedule) // batch_size
    milestone_updates = {value * updates_per_pass: value for value in milestones}
    run_root.mkdir(parents=True, exist_ok=True)
    hard.set_seed(seed)
    model, source = centered.load_h50_model(architecture_checkpoint, h50_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    current = 0
    history: list[dict[str, Any]] = []
    if resume:
        found = latest_checkpoint(run_root, contract_id)
        if found is not None:
            path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            current = int(payload["step"])
            history = list(payload.get("history", []))
            print(json.dumps({"event": "resume", "variant": variant, "step": current, "file": str(path)}), flush=True)
    store.preload()
    started = time.perf_counter()
    milestone_files: dict[str, dict[str, Any]] = {}
    checkpoint_files: dict[str, dict[str, Any]] = {}

    def discover() -> None:
        for step in milestone_updates:
            path = run_root / f"step-{step}.pt"
            if path.is_file():
                metadata = checkpoint_metadata(path)
                milestone_files[str(step)] = metadata
                checkpoint_files[str(step)] = metadata

    def checkpoint(step: int, status: str) -> None:
        metadata = save_checkpoint(
            run_root / f"step-{step}.pt",
            model=model,
            optimizer=optimizer,
            variant=variant,
            contract_id=contract_id,
            step=step,
            passes=passes,
            records_per_pass=records_per_pass,
            batch_size=batch_size,
            learning_rate=learning_rate,
            anchor_beta=anchor_beta,
            history=history,
            source=source,
            status=status,
        )
        checkpoint_files[str(step)] = metadata
        if step in milestone_updates:
            milestone_files[str(step)] = metadata

    discover()
    if current == 0 and "0" not in milestone_files:
        checkpoint(0, "initial")
    while current < total_updates:
        begin = current * batch_size
        items = schedule[begin : begin + batch_size]
        if len(items) != batch_size:
            raise AssertionError("Incomplete audio training batch")
        input_array, target_array, anchor_array, weight_array = store.batch(items, event_weights)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        anchor_tensor = torch.from_numpy(anchor_array).to(device)
        weight_tensor = torch.from_numpy(weight_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_spec = model(input_tensor)
        predicted_full = torch_packed_istft(predicted_spec, window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted_audio = predicted_full[:, trim : trim + USEFUL_SAMPLES]
        event_loss = charcoal_loss(predicted_audio, target_tensor, weight_tensor)
        anchor_loss = charcoal_loss(predicted_audio, anchor_tensor, 1.0 - weight_tensor)
        loss = event_loss + float(anchor_beta) * anchor_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite audio loss at update {current + 1}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current += 1
        history.append(
            {
                "update": current,
                "pass": current / updates_per_pass,
                "loss": float(loss.detach().cpu()),
                "eventLoss": float(event_loss.detach().cpu()),
                "anchorLoss": float(anchor_loss.detach().cpu()),
                "eventWeightMean": float(weight_array.mean()),
                "gradientNormBeforeClip": gradient_norm,
            }
        )
        if current == 1 or current % 100 == 0:
            print(json.dumps({"event": "progress", "variant": variant, "update": current, "totalUpdates": total_updates, "loss": history[-1]["loss"], "eventLoss": history[-1]["eventLoss"], "anchorLoss": history[-1]["anchorLoss"]}, sort_keys=True), flush=True)
        if current in milestone_updates and current != 0:
            checkpoint(current, "milestone")
        elif state_interval > 0 and current % state_interval == 0:
            checkpoint(current, "rolling")
    discover()
    if str(total_updates) not in milestone_files:
        checkpoint(total_updates, "completed")
    del model, optimizer, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "variant": variant,
        "eventWeights": event_weights,
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
    }


def compare_torch_numpy(
    packed: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    numpy_audio = dsp.istft_centered(packed)
    with torch.inference_mode():
        torch_audio = torch_packed_istft(torch.from_numpy(packed).to(device), window).cpu().numpy()
    error = torch_audio - numpy_audio
    return {
        "maxAbsError": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "numpyRms": float(np.sqrt(np.mean(numpy_audio.astype(np.float64) ** 2))),
        "torchRms": float(np.sqrt(np.mean(torch_audio.astype(np.float64) ** 2))),
    }


def run_smoke(
    *,
    store: AudioCacheStore,
    schedule: list[hard.ScheduleItem],
    h50_checkpoint: Path,
    architecture_checkpoint: Path,
    device: torch.device,
    updates: int,
) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("smoke-updates must be positive")
    store.preload()
    first_slug = sorted(store.input_paths)[0]
    first = store._load(first_slug)
    packed = first["inputSpec"][0:1]
    dsp_compare = compare_torch_numpy(packed, device)
    results: dict[str, Any] = {"dsp": dsp_compare, "variants": {}}
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    for variant, weight_key in ((CORE, "coreMask"), (GUARD25, "guard25Weights")):
        model, _ = centered.load_h50_model(architecture_checkpoint, h50_checkpoint, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, weight_decay=0.0)
        losses: list[float] = []
        gradients: list[float] = []
        for update in range(updates):
            items = [schedule[update % len(schedule)]]
            input_array, target_array, anchor_array, weight_array = store.batch(items, weight_key)
            input_tensor = torch.from_numpy(input_array).to(device)
            target_tensor = torch.from_numpy(target_array).to(device)
            anchor_tensor = torch.from_numpy(anchor_array).to(device)
            weight_tensor = torch.from_numpy(weight_array).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = torch_packed_istft(model(input_tensor), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            predicted = predicted[:, trim : trim + USEFUL_SAMPLES]
            event_loss = charcoal_loss(predicted, target_tensor, weight_tensor)
            anchor_loss = charcoal_loss(predicted, anchor_tensor, 1.0 - weight_tensor)
            loss = event_loss + anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Smoke non-finite loss: {variant}/{update}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            if not torch.isfinite(gradient):
                raise FloatingPointError(f"Smoke non-finite gradient: {variant}/{update}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradients.append(float(gradient.detach().cpu()))
        results["variants"][variant] = {
            "updates": updates,
            "lossFirst": losses[0],
            "lossLast": losses[-1],
            "gradientMax": max(gradients),
        }
        del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.passes <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("passes, batch-size, and threads must be positive")
    if args.learning_rate <= 0.0 or args.anchor_beta < 0.0 or args.state_interval <= 0:
        raise ValueError("learning-rate, anchor-beta, and state-interval must be valid")
    if args.guard_ms <= 0.0:
        raise ValueError("guard-ms must be positive")
    if milestones[0] != 0 or milestones[-1] != args.passes:
        raise ValueError("milestones must start at 0 and end at passes")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted and unique")
    if args.max_eval_songs is not None and args.max_eval_songs <= 0:
        raise ValueError("max-eval-songs must be positive")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_int_list(args.milestones)
    validate_args(args, milestones)
    event_root = args.event_root.resolve()
    output_root = args.output_root.resolve()
    eval_root = args.eval_root.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    h50_checkpoint = args.h50_checkpoint.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not checkpoint.is_file() or not h50_checkpoint.is_file():
        raise FileNotFoundError("Student architecture or H50 checkpoint is missing")
    entries, records_by_song, selection_h50_sha = load_centered_records(
        event_root, args.max_songs
    )
    actual_h50_sha = file_sha256(h50_checkpoint)
    if selection_h50_sha.lower() != actual_h50_sha.lower():
        raise ValueError(f"H50 hash mismatch: selection={selection_h50_sha}, actual={actual_h50_sha}")
    output_root.mkdir(parents=True, exist_ok=True)
    input_paths: dict[str, Path] = {}
    audio_paths: dict[str, Path] = {}
    audio_metadata: dict[str, Any] = {}
    for index, entry in enumerate(entries, start=1):
        slug = entry["slug"]
        event_npz, _event_json = event_cache_path(event_root, slug)
        audio_npz, metadata = prepare_audio_cache(
            event_root=event_root,
            output_root=output_root,
            slug=slug,
            rows=records_by_song[slug],
            h50_checkpoint=h50_checkpoint,
            force=args.force_cache,
        )
        input_paths[slug] = event_npz
        audio_paths[slug] = audio_npz
        audio_metadata[slug] = metadata
        if index == 1 or index == len(entries) or index % 10 == 0:
            print(f"prepared audio cache {index}/{len(entries)}: {slug}", flush=True)
    store = AudioCacheStore(input_paths, audio_paths)
    schedule = build_schedule(records_by_song, args.passes, args.seed)
    if len(schedule) % args.batch_size:
        raise ValueError("Audio schedule is not divisible by batch size")
    schedule_payload = [{"slug": item.slug, "cacheIndex": item.cache_index} for item in schedule]
    schedule_summary = {
        "recordCount": len(schedule),
        "recordsPerPass": len(schedule) // args.passes,
        "updates": len(schedule) // args.batch_size,
        "scheduleSha256": canonical_sha256(schedule_payload),
    }
    if args.smoke_only:
        smoke = run_smoke(
            store=store,
            schedule=schedule,
            h50_checkpoint=h50_checkpoint,
            architecture_checkpoint=checkpoint,
            device=device,
            updates=args.smoke_updates,
        )
        smoke_report = {
            "schema": SCHEMA,
            "status": "smoke-completed",
            "contract": {
                "eventRoot": str(event_root),
                "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
                "trainSongCount": len(entries),
                "device": str(device),
                "guardMs": args.guard_ms,
                "schedule": schedule_summary,
            },
            "smoke": smoke,
        }
        report_path = output_root / "reports" / "inst3-vr-event-audio-smoke.json"
        json_write(report_path, smoke_report)
        print(json.dumps({"status": smoke_report["status"], "report": str(report_path), "smoke": smoke}, indent=2), flush=True)
        return 0

    manifest = hard.load_manifest(manifest_path)
    eval_entries = sorted(
        [entry for entry in manifest["entries"] if entry.get("role") in {"calibration", "internal-test"}],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    common_contract = {
        "schema": SCHEMA,
        "eventCenteredSelectionSha256": file_sha256(event_root / "selection.json"),
        "eventCenteredH50CheckpointSha256": actual_h50_sha,
        "scheduleSha256": schedule_summary["scheduleSha256"],
        "manifestSha256": file_sha256(manifest_path),
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "trainSongCount": len(entries),
        "evalSongCount": len(eval_entries),
        "eventsPerSong": 4,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "guardMs": args.guard_ms,
        "seed": args.seed,
        "device": str(device),
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "objective": "audio-domain Charbonnier through differentiable torch.iSTFT",
        "officialFinalTestUsed": False,
    }
    training: dict[str, Any] = {}
    for variant, weight_key in ((CORE, "coreMask"), (GUARD25, "guard25Weights")):
        variant_contract = {**common_contract, "variant": variant, "eventWeightKey": weight_key}
        training[variant] = train_variant(
            variant=variant,
            event_weights=weight_key,
            h50_checkpoint=h50_checkpoint,
            architecture_checkpoint=checkpoint,
            store=store,
            schedule=schedule,
            run_root=output_root / "runs" / safe_name(variant),
            contract_id=canonical_sha256(variant_contract),
            passes=args.passes,
            milestones=milestones,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            anchor_beta=args.anchor_beta,
            seed=args.seed,
            device=device,
            state_interval=args.state_interval,
            resume=args.resume,
        )
        training[variant]["contract"] = {
            "id": canonical_sha256(variant_contract),
            "payload": variant_contract,
        }
    evaluation: dict[str, Any] = {
        "initial-vocals": centered.evaluate_initial(
            architecture_checkpoint=checkpoint,
            entries=eval_entries,
            eval_root=eval_root,
            eval_windows_per_song=args.eval_windows_per_song,
            seed=args.seed,
            device=device,
        ),
        "H50-pass-50": centered.evaluate_checkpoint(
            name="H50-pass-50",
            state_path=h50_checkpoint,
            architecture_checkpoint=checkpoint,
            entries=eval_entries,
            eval_root=eval_root,
            eval_windows_per_song=args.eval_windows_per_song,
            seed=args.seed,
            device=device,
        ),
    }
    for variant, result in training.items():
        for pass_count in milestones:
            if pass_count == 0:
                continue
            step = pass_count * result["updatesPerPass"]
            metadata = result["milestoneCheckpoints"].get(str(step))
            if metadata is None:
                raise FileNotFoundError(f"Missing {variant} pass {pass_count}")
            evaluation[f"{variant}@pass-{pass_count}"] = centered.evaluate_checkpoint(
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
    for name, value in evaluation.items():
        if name != "H50-pass-50":
            value["deltaVsH50"] = hard.metric_delta(value, baseline)
    listening_report = None
    if not args.skip_listening:
        state_paths = {"H50-pass-50": h50_checkpoint}
        for variant, result in training.items():
            step = args.passes * result["updatesPerPass"]
            state_paths[f"{variant}@pass-{args.passes}"] = Path(
                result["milestoneCheckpoints"][str(step)]["file"]
            )
        listening_report = centered.render_listening(
            output_root=output_root / "listening-12",
            samples_root=args.samples_root.resolve(),
            architecture_checkpoint=checkpoint,
            state_paths=state_paths,
            device=device,
            force=args.force_listening,
        )
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "schedule": schedule_summary,
        "audioCache": {
            "root": str((output_root / "cache" / "train").resolve()),
            "songCount": len(audio_metadata),
            "dsp": "existing NumPy centered iSTFT for fixed PCM targets; differentiable torch.iSTFT for prediction",
        },
        "training": training,
        "evaluation": evaluation,
        "listening": listening_report,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device),
            "runner": {"file": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        },
        "notes": {
            "loss": "event target plus H50 anchor on the complement; guard25 uses a 25 ms linear target guard",
            "publication": "Local non-commercial research only; do not publish MUSDB18-derived checkpoints, caches, or audio.",
            "next_gate": "Inspect 50/100/200 ms positive-projection p95/max, artifact proxies, and 12-song listening before retaining an arm.",
        },
    }
    report_path = output_root / "reports" / "inst3-vr-event-audio-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "trainingVariants": list(training), "evaluationVariants": list(evaluation), "listeningCount": listening_report.get("outputCount", 0) if listening_report else 0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
