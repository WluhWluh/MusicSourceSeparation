#!/usr/bin/env python3
"""Retrain the aggressive Inst 3 residual student for a 24-frame contract.

This is a local, non-commercial MUSDB18 experiment.  The neural core keeps
the TFC-TDF architecture and warm-starts from the completed V-R-H50
checkpoint, but every input and target spectrum is regenerated for the short
window contract:

* 24 frames, 4 hops of context on each side;
* 15-hop (15,360 sample) useful output stride;
* ``F24-Inst3-U``: eight uniform windows per song;
* ``F24-Inst3-H25``: four shared uniform windows plus four event-centered
  windows from the frozen H50 miss manifest.

The runner does not export ONNX/TFLite or change product code.  Decoded source
PCM and teacher-derived artifacts remain local under the MUSDB18 restrictions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_scale10_density as density
import run_inst3_vr_hard_sampling as hard
import tfc_tdf_short_window as short
from tfc_tdf_default_model import (
    DefaultTfcTdfConfig,
    TfcTdfNchwWrapper,
    TfcTdfNeuralCore,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = pilot.DEFAULT_ARCHIVE
DEFAULT_MANIFEST = pilot.DEFAULT_MANIFEST
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-event-centered"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-f24-retrain"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_H50_CHECKPOINT = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-hard-sampling-h50"
    / "runs"
    / "V-R-H50"
    / "step-8000.pt"
)
DEFAULT_TEACHER = pilot.DEFAULT_TEACHER
DEFAULT_CONTRACT = pilot.DEFAULT_CONTRACT
DEFAULT_TEACHER_TFLITE = pilot.DEFAULT_TEACHER_TFLITE
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"

SCHEMA = "local-inst3-tfc-f24-retrain@1"
CACHE_SCHEMA = "local-inst3-tfc-f24-cache@1"
UNIFORM = "F24-Inst3-U"
H25 = "F24-Inst3-H25"
VARIANTS = (UNIFORM, H25)
DEFAULT_FRAMES = 24
DEFAULT_CONTEXT_HOPS = 4
DEFAULT_WINDOWS_PER_SONG = 8
DEFAULT_MILESTONES = (0, 5, 10)
SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class ShortRecord:
    index: int
    start: int
    length: int
    kind: str
    event_start: int = -1
    event_end: int = -1


@dataclass(frozen=True)
class SongPlan:
    slug: str
    member: str
    records: tuple[ShortRecord, ...]
    uniform_indices: tuple[int, ...]
    event_indices: tuple[int, ...]

    @property
    def cache_index_by_start(self) -> dict[int, int]:
        return {record.start: record.index for record in self.records}


@dataclass(frozen=True)
class ScheduleItem:
    slug: str
    cache_index: int


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integers, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--context-hops", type=int, default=DEFAULT_CONTEXT_HOPS)
    parser.add_argument("--train-windows-per-song", type=int, default=DEFAULT_WINDOWS_PER_SONG)
    parser.add_argument("--events-per-song", type=int, default=4)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--milestones", default="0,5,10")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--state-interval", type=int, default=320)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--require-teacher-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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


def make_contract(num_frames: int, context_hops: int) -> short.ShortWindowContract:
    return short.ShortWindowContract(
        num_frames=num_frames,
        left_context_hops=context_hops,
        right_context_hops=context_hops,
    )


def make_short_model(
    state_path: Path,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = DefaultTfcTdfConfig(num_frames=num_frames)
    model = TfcTdfNchwWrapper(TfcTdfNeuralCore(config)).to(device)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    state = payload.get("stateDict")
    if not isinstance(state, dict):
        raise ValueError(f"Missing stateDict in {state_path}")
    model.load_state_dict(state, strict=True)
    sweep.freeze_batchnorm_running_statistics(model)
    model.to(device)
    return model, {
        "statePath": str(state_path.resolve()),
        "stateSha256": file_sha256(state_path),
        "step": int(payload.get("step", -1)),
        "passes": int(payload.get("passes", -1)),
        "format": payload.get("format"),
        "numFrames": num_frames,
    }


def load_event_selection(
    event_root: Path,
    events_per_song: int,
    max_songs: int | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str]:
    path = event_root / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "local-inst3-vr-event-centered@1":
        raise ValueError(f"Unexpected event selection schema: {payload.get('schema')}")
    songs = payload.get("songs")
    if not isinstance(songs, dict) or len(songs) != 80:
        raise ValueError(f"Expected 80 event songs, found {len(songs or {})}")
    ordered = sorted(songs)
    if max_songs is not None:
        if max_songs <= 0:
            raise ValueError("max-songs must be positive")
        ordered = ordered[:max_songs]
    selected: dict[str, list[dict[str, Any]]] = {}
    for slug in ordered:
        rows = songs[slug].get("events")
        centered_rows = songs[slug].get("centered")
        # The event-centered selection report stores the absolute event data
        # in both fields in current runs. Prefer centered rows because they
        # also carry the context placement provenance.
        source_rows = centered_rows or rows
        if not isinstance(source_rows, list) or len(source_rows) < events_per_song:
            raise ValueError(f"Not enough event rows for {slug}")
        selected[slug] = source_rows[:events_per_song]
    h50 = payload.get("h50Checkpoint", {})
    h50_sha = str(h50.get("sha256", ""))
    if not h50_sha:
        raise ValueError("Event selection has no H50 checkpoint hash")
    return payload, selected, h50_sha


def read_decoded_length(oracle_root: Path, slug: str) -> int | None:
    metadata = oracle_root / "decoded" / "train" / slug / "decoded.json"
    if not metadata.is_file():
        return None
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        samples = int(value["samples"])
        return samples if samples > 0 else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def uniform_starts(song_samples: int, contract: short.ShortWindowContract) -> list[int]:
    if song_samples <= 0:
        raise ValueError("song_samples must be positive")
    maximum = max(0, song_samples - contract.stride_samples)
    starts = list(range(0, maximum + 1, contract.stride_samples))
    if not starts:
        starts = [0]
    if starts[-1] != maximum:
        starts.append(maximum)
    return starts


def choose_uniform(
    starts: Sequence[int], count: int, seed: int
) -> list[int]:
    if count <= 0:
        raise ValueError("uniform window count must be positive")
    rng = np.random.default_rng(seed)
    if len(starts) <= count:
        return [int(value) for value in starts]
    selected = rng.choice(np.asarray(starts, dtype=np.int64), size=count, replace=False)
    return [int(value) for value in sorted(selected.tolist())]


def event_output_start(
    row: dict[str, Any], song_samples: int, contract: short.ShortWindowContract
) -> tuple[int, int, int]:
    maximum = max(0, song_samples - contract.stride_samples)
    center = int(row.get("eventCenterSamples", row.get("event_center", 0)))
    start = min(max(int(round(center - contract.stride_samples / 2.0)), 0), maximum)
    event_start = int(row.get("blockStartSamples", row.get("eventStartSamples", 0))) - start
    event_end = int(row.get("blockEndSamples", row.get("eventEndSamples", 0))) - start
    event_start = max(0, min(contract.stride_samples, event_start))
    event_end = max(0, min(contract.stride_samples, event_end))
    if event_end <= event_start:
        raise ValueError(f"Event is outside F24 output region: {row}")
    return start, event_start, event_end


def build_song_plan(
    *,
    slug: str,
    member: str,
    song_samples: int,
    event_rows: Sequence[dict[str, Any]],
    contract: short.ShortWindowContract,
    train_windows_per_song: int,
    seed: int,
    song_index: int,
) -> SongPlan:
    all_starts = uniform_starts(song_samples, contract)
    uniform = choose_uniform(
        all_starts,
        train_windows_per_song,
        seed + song_index * 1_000_003,
    )
    records: list[ShortRecord] = []
    by_start: dict[int, int] = {}

    def add(record: ShortRecord) -> int:
        existing = by_start.get(record.start)
        if existing is not None:
            old = records[existing]
            if record.kind == "event" and old.kind != "event":
                records[existing] = ShortRecord(
                    index=old.index,
                    start=old.start,
                    length=old.length,
                    kind="event",
                    event_start=record.event_start,
                    event_end=record.event_end,
                )
            return existing
        index = len(records)
        normalized = ShortRecord(
            index=index,
            start=record.start,
            length=record.length,
            kind=record.kind,
            event_start=record.event_start,
            event_end=record.event_end,
        )
        records.append(normalized)
        by_start[record.start] = index
        return index

    uniform_indices = tuple(
        add(ShortRecord(-1, start, contract.stride_samples, "uniform"))
        for start in uniform
    )
    event_indices_list: list[int] = []
    for row in event_rows:
        start, event_start, event_end = event_output_start(row, song_samples, contract)
        event_indices_list.append(
            add(
                ShortRecord(
                    -1,
                    start,
                    contract.stride_samples,
                    "event",
                    event_start,
                    event_end,
                )
            )
        )
    return SongPlan(
        slug=slug,
        member=member,
        records=tuple(records),
        uniform_indices=uniform_indices,
        event_indices=tuple(event_indices_list),
    )


def padded_segment(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    result = np.zeros((length, 2), dtype=np.float32)
    source_start = max(0, start)
    source_end = min(audio.shape[0], start + length)
    if source_end > source_start:
        destination = source_start - start
        result[destination : destination + source_end - source_start] = audio[
            source_start:source_end
        ]
    return result


def build_input_and_target_specs(
    mixture: np.ndarray,
    teacher_instrumental: np.ndarray,
    record: ShortRecord,
    contract: short.ShortWindowContract,
) -> tuple[np.ndarray, np.ndarray]:
    input_window = short.assemble_input(
        mixture,
        record.start,
        record.length,
        contract,
        mode="continuous",
    )
    mixture_segment = padded_segment(mixture, record.start, record.length)
    teacher_segment = np.ascontiguousarray(
        teacher_instrumental[: record.length], dtype=np.float32
    )
    residual_segment = mixture_segment - teacher_segment
    target_window = np.zeros_like(input_window)
    left = contract.left_context_samples
    target_window[left : left + record.length] = residual_segment
    return (
        np.ascontiguousarray(short.stft_centered(input_window, contract)[0], dtype=np.float32),
        np.ascontiguousarray(short.stft_centered(target_window, contract)[0], dtype=np.float32),
    )


def cache_paths(root: Path, slug: str) -> tuple[Path, Path]:
    base = root / "cache" / "train" / slug
    return base.with_suffix(".npz"), base.with_suffix(".json")


def prepare_song_cache(
    *,
    output_root: Path,
    song: Any,
    plan: SongPlan,
    entry: dict[str, Any],
    teacher_session: Any,
    contract: short.ShortWindowContract,
    checkpoint: Path,
    h50_checkpoint: Path,
    teacher: Path,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    output, metadata_path = cache_paths(output_root, plan.slug)
    expected = {
        "schema": CACHE_SCHEMA,
        "sourceSha256": entry["sourceSha256"],
        "member": entry["member"],
        "slug": plan.slug,
        "teacherSha256": file_sha256(teacher),
        "teacherContractId": "uvr_mdxnet_inst_3@2",
        "h50CheckpointSha256": file_sha256(h50_checkpoint),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "sampleRate": contract.sample_rate,
        "numFrames": contract.num_frames,
        "inputSamples": contract.input_samples,
        "strideSamples": contract.stride_samples,
        "leftContextHops": contract.left_context_hops,
        "rightContextHops": contract.right_context_hops,
        "records": [
            {
                "start": record.start,
                "length": record.length,
                "kind": record.kind,
                "eventStart": record.event_start,
                "eventEnd": record.event_end,
            }
            for record in plan.records
        ],
    }
    if not force and output.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(output) as values:
                shape = values["inputSpec"].shape
                valid = (
                    metadata.get("contract") == expected
                    and shape == values["targetResidualSpec"].shape
                    and shape[0] == len(plan.records)
                    and tuple(shape[1:]) == (4, contract.frequency_bins, contract.num_frames)
                )
            if valid:
                return output, metadata
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    candidates = [
        hard.Candidate(
            index=record.index,
            start=record.start,
            length=record.length,
            hard_score=0.0,
            rank=record.index + 1,
        )
        for record in plan.records
    ]
    selection = hard.SongSelection(
        slug=plan.slug,
        member=plan.member,
        candidates=tuple(candidates),
        hard_pool_indices=tuple(range(len(candidates))),
        uniform_indices=plan.uniform_indices,
        hard_fraction=0.5,
        hard_indices=plan.uniform_indices[:4] + plan.event_indices,
        union_indices=tuple(range(len(candidates))),
    )
    teacher_segments, teacher_timing = hard.render_selected_teacher_segments(
        song.mixture_gt,
        selection,
        teacher_session,
        progress_label=f"F24/{plan.slug}",
    )
    input_specs: list[np.ndarray] = []
    target_specs: list[np.ndarray] = []
    for index, record in enumerate(plan.records):
        input_spec, target_spec = build_input_and_target_specs(
            song.mixture_gt,
            teacher_segments[index],
            record,
            contract,
        )
        input_specs.append(input_spec)
        target_specs.append(target_spec)
    input_array = np.ascontiguousarray(np.stack(input_specs), dtype=np.float32)
    target_array = np.ascontiguousarray(np.stack(target_specs), dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    np.savez_compressed(
        temporary,
        inputSpec=input_array,
        targetResidualSpec=target_array,
        starts=np.asarray([record.start for record in plan.records], dtype=np.int64),
        lengths=np.asarray([record.length for record in plan.records], dtype=np.int64),
        kinds=np.asarray([record.kind for record in plan.records]),
        eventStarts=np.asarray([record.event_start for record in plan.records], dtype=np.int64),
        eventEnds=np.asarray([record.event_end for record in plan.records], dtype=np.int64),
    )
    temporary_npz = temporary if temporary.suffix == ".npz" else Path(str(temporary) + ".npz")
    temporary_npz.replace(output)
    metadata = {
        "contract": expected,
        "cache": checkpoint_metadata(output),
        "render": {
            "teacher": teacher_timing,
            "mode": "short-window-rebuilt-from-mixtureGt",
            "recordCount": len(plan.records),
        },
        "selection": {
            "uniformIndices": list(plan.uniform_indices),
            "eventIndices": list(plan.event_indices),
        },
    }
    json_write(metadata_path, metadata)
    return output, metadata


class ShortCacheStore:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
        self._open: dict[str, dict[str, np.ndarray]] = {}

    def _load(self, slug: str) -> dict[str, np.ndarray]:
        value = self._open.get(slug)
        if value is not None:
            return value
        with np.load(self.paths[slug]) as arrays:
            value = {
                "inputSpec": np.ascontiguousarray(arrays["inputSpec"], dtype=np.float32),
                "targetResidualSpec": np.ascontiguousarray(
                    arrays["targetResidualSpec"], dtype=np.float32
                ),
            }
        if value["inputSpec"].shape != value["targetResidualSpec"].shape:
            raise ValueError(f"F24 cache shape mismatch for {slug}")
        self._open[slug] = value
        return value

    def batch(self, items: Sequence[ScheduleItem]) -> tuple[np.ndarray, np.ndarray]:
        arrays = [self._load(item.slug) for item in items]
        return (
            np.stack([array["inputSpec"][item.cache_index] for array, item in zip(arrays, items)]),
            np.stack(
                [array["targetResidualSpec"][item.cache_index] for array, item in zip(arrays, items)]
            ),
        )

    def preload(self) -> None:
        for index, slug in enumerate(sorted(self.paths), start=1):
            self._load(slug)
            if index == 1 or index == len(self.paths) or index % 10 == 0:
                print(f"preload F24 cache {index}/{len(self.paths)}", flush=True)


def build_schedule(
    plans: dict[str, SongPlan],
    variant: str,
    passes: int,
    seed: int,
) -> list[ScheduleItem]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    schedule: list[ScheduleItem] = []
    for pass_index in range(passes):
        rows: list[ScheduleItem] = []
        for slug in sorted(plans):
            plan = plans[slug]
            indices = (
                plan.uniform_indices
                if variant == UNIFORM
                else plan.uniform_indices[:4] + plan.event_indices
            )
            rows.extend(ScheduleItem(slug, index) for index in indices)
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        schedule.extend(rows[int(index)] for index in rng.permutation(len(rows)))
    return schedule


def schedule_json(schedule: Sequence[ScheduleItem]) -> list[dict[str, Any]]:
    return [{"slug": item.slug, "cacheIndex": item.cache_index} for item in schedule]


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


def atomic_torch_save(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return checkpoint_metadata(path)


def train_variant(
    *,
    variant: str,
    state_source: Path,
    contract_id: str,
    store: ShortCacheStore,
    schedule: list[ScheduleItem],
    run_root: Path,
    contract: short.ShortWindowContract,
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
    if records_per_pass % batch_size:
        raise ValueError("F24 records per pass must be divisible by batch size")
    updates_per_pass = records_per_pass // batch_size
    total_updates = len(schedule) // batch_size
    milestone_updates = {value * updates_per_pass: value for value in milestones}
    run_root.mkdir(parents=True, exist_ok=True)
    hard.set_seed(seed)
    model, source_metadata = make_short_model(state_source, contract.num_frames, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    current = 0
    history: list[dict[str, Any]] = []
    resume_count = 0
    if resume:
        found = latest_checkpoint(run_root, contract_id)
        if found is not None:
            path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            current = int(payload["step"])
            history = list(payload.get("history", []))
            resume_count = int(payload.get("resumeCount", 0)) + 1
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

    def save(step: int, status: str) -> None:
        metadata = atomic_torch_save(
            run_root / f"step-{step}.pt",
            {
                "format": "local-inst3-tfc-f24-checkpoint@1",
                "status": status,
                "variant": variant,
                "runContractId": contract_id,
                "step": step,
                "passes": passes,
                "recordsPerPass": records_per_pass,
                "batchSize": batch_size,
                "learningRate": learning_rate,
                "seed": seed,
                "numFrames": contract.num_frames,
                "leftContextHops": contract.left_context_hops,
                "rightContextHops": contract.right_context_hops,
                "stateDict": hard.cpu_tree(model.state_dict()),
                "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
                "history": history,
                "resumeCount": resume_count,
                "elapsedSeconds": time.perf_counter() - started,
                "checkpointSource": source_metadata,
            },
        )
        checkpoint_files[str(step)] = metadata
        if step in milestone_updates:
            milestone_files[str(step)] = metadata

    discover()
    if current == 0 and "0" not in milestone_files:
        save(0, "initial")
    while current < total_updates:
        begin = current * batch_size
        items = schedule[begin : begin + batch_size]
        if len(items) != batch_size:
            raise AssertionError("Incomplete F24 batch")
        input_array, target_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(input_tensor)
        loss = F.l1_loss(prediction, target_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite F24 loss at update {current + 1}")
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
                "gradientNormBeforeClip": gradient_norm,
            }
        )
        if current == 1 or current % 100 == 0:
            print(json.dumps({"event": "progress", "variant": variant, "update": current, "totalUpdates": total_updates, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        if current in milestone_updates and current != 0:
            save(current, "milestone")
        elif state_interval > 0 and current % state_interval == 0:
            save(current, "rolling")
    discover()
    if str(total_updates) not in milestone_files:
        save(total_updates, "completed")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "variant": variant,
        "status": "completed",
        "passes": passes,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": updates_per_pass,
        "updates": total_updates,
        "learningRate": learning_rate,
        "batchSize": batch_size,
        "seed": seed,
        "numFrames": contract.num_frames,
        "leftContextHops": contract.left_context_hops,
        "rightContextHops": contract.right_context_hops,
        "history": history,
        "milestoneCheckpoints": milestone_files,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
        "resumeCount": resume_count,
    }


def predict_short(
    model: torch.nn.Module,
    audio: np.ndarray,
    start: int,
    length: int,
    contract: short.ShortWindowContract,
    device: torch.device,
) -> np.ndarray:
    window = short.assemble_input(audio, start, length, contract, mode="continuous")
    spectrum = short.stft_centered(window, contract)
    with torch.inference_mode():
        output = model(torch.from_numpy(spectrum).to(device)).detach().cpu().numpy()
    reconstructed = short.istft_centered(output, contract)
    begin = contract.left_context_samples
    return np.ascontiguousarray(reconstructed[begin : begin + length], dtype=np.float32)


def select_eval_starts(
    song_samples: int,
    contract: short.ShortWindowContract,
    count: int,
) -> list[int]:
    candidates = uniform_starts(song_samples, contract)
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(index)] for index in sorted(set(indices.tolist()))]


def evaluate_model(
    *,
    name: str,
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    eval_root: Path,
    contract: short.ShortWindowContract,
    windows_per_song: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    event_values: dict[int, dict[str, list[np.ndarray]]] = {
        milliseconds: {"missRms": [], "positiveProjectionRms": [], "active": []}
        for milliseconds in (50, 100, 200)
    }
    aggregate = {
        "vocals": hard.PowerAccumulator(),
        "teacherVocals": hard.PowerAccumulator(),
        "instrumental": hard.PowerAccumulator(),
        "teacher": hard.PowerAccumulator(),
        "lowInstrumental": hard.PowerAccumulator(),
    }
    vocal_basis = 0.0
    accompaniment_dot = 0.0
    sample_count = 0
    per_song: dict[str, Any] = {}
    derivative: list[float] = []
    teacher_derivative: list[float] = []
    excess: list[float] = []
    clip_count = 0
    nonfinite_count = 0
    started = time.perf_counter()
    for song_index, entry in enumerate(entries):
        song = hard.load_eval_song(eval_root, entry)
        starts = select_eval_starts(song.mixture_gt.shape[0], contract, windows_per_song)
        song_acc = {
            "vocals": hard.PowerAccumulator(),
            "teacherVocals": hard.PowerAccumulator(),
            "instrumental": hard.PowerAccumulator(),
            "teacher": hard.PowerAccumulator(),
            "lowInstrumental": hard.PowerAccumulator(),
        }
        song_events: dict[int, dict[str, list[np.ndarray]]] = {
            milliseconds: {"missRms": [], "positiveProjectionRms": [], "active": []}
            for milliseconds in (50, 100, 200)
        }
        song_vocal_basis = 0.0
        song_dot = 0.0
        song_samples = 0
        song_derivative: list[float] = []
        song_teacher_derivative: list[float] = []
        song_excess: list[float] = []
        song_clips = 0
        song_nonfinite = 0
        for start in starts:
            length = min(contract.stride_samples, song.mixture_gt.shape[0] - start)
            mixture = song.mixture_gt[start : start + length]
            residual = predict_short(model, song.mixture_gt, start, length, contract, device)
            predicted_instrumental = np.ascontiguousarray(mixture - residual, dtype=np.float32)
            if not np.isfinite(predicted_instrumental).all():
                nonfinite_count += 1
                song_nonfinite += 1
                continue
            vocals = song.vocals[start : start + length]
            true_instrumental = song.instrumental[start : start + length]
            teacher_instrumental = song.teacher_instrumental[start : start + length]
            predicted_vocals = np.ascontiguousarray(mixture - predicted_instrumental, dtype=np.float32)
            for key, reference, candidate in (
                ("vocals", vocals, predicted_vocals),
                ("teacherVocals", vocals, mixture - teacher_instrumental),
                ("instrumental", true_instrumental, predicted_instrumental),
                ("teacher", true_instrumental, teacher_instrumental),
            ):
                song_acc[key].update(reference, candidate)
                aggregate[key].update(reference, candidate)
            low_mask = pilot.low_vocal_mask(vocals)
            song_acc["lowInstrumental"].update(
                true_instrumental[low_mask], predicted_instrumental[low_mask]
            )
            aggregate["lowInstrumental"].update(
                true_instrumental[low_mask], predicted_instrumental[low_mask]
            )
            vocal64 = vocals.astype(np.float64)
            error64 = (predicted_instrumental - true_instrumental).astype(np.float64)
            song_vocal_basis += float(np.sum(vocal64 * vocal64))
            vocal_basis += float(np.sum(vocal64 * vocal64))
            dot = float(np.sum(error64 * vocal64))
            song_dot += dot
            accompaniment_dot += dot
            song_samples += length
            sample_count += length
            teacher_removed = mixture - teacher_instrumental
            miss = predicted_instrumental - teacher_instrumental
            for milliseconds in (50, 100, 200):
                values = hard.block_projection_arrays(
                    teacher_removed, miss, song.sample_rate, milliseconds
                )
                for key in ("missRms", "positiveProjectionRms", "active"):
                    song_events[milliseconds][key].append(values[key])
                    event_values[milliseconds][key].append(values[key])
            predicted_diff = np.diff(predicted_instrumental, axis=0)
            teacher_diff = np.diff(teacher_instrumental, axis=0)
            predicted_db = hard.dbfs(math.sqrt(float(np.mean(predicted_diff.astype(np.float64) ** 2))))
            teacher_db = hard.dbfs(math.sqrt(float(np.mean(teacher_diff.astype(np.float64) ** 2))))
            song_derivative.append(predicted_db)
            song_teacher_derivative.append(teacher_db)
            song_excess.append(predicted_db - teacher_db)
            derivative.append(predicted_db)
            teacher_derivative.append(teacher_db)
            excess.append(predicted_db - teacher_db)
            song_clips += int(np.count_nonzero(np.abs(predicted_instrumental) >= 1.0))
            clip_count += int(np.count_nonzero(np.abs(predicted_instrumental) >= 1.0))
        song_metric = hard.metric_result(
            vocals=song_acc["vocals"],
            teacher_vocals=song_acc["teacherVocals"],
            instrumental=song_acc["instrumental"],
            teacher=song_acc["teacher"],
            low_instrumental=song_acc["lowInstrumental"],
            vocal_basis_power=song_vocal_basis,
            accompaniment_vocal_dot=song_dot,
            sample_count=song_samples,
        )
        song_metric["eventMetrics"] = hard.summarize_event_values(song_events)
        song_metric["mechanicalArtifactProxy"] = hard.mechanical_metrics(
            song_derivative,
            song_teacher_derivative,
            song_excess,
            song_clips,
            song_nonfinite,
        )
        per_song[song.slug] = {
            "role": song.role,
            "windowStarts": starts,
            "metrics": song_metric,
        }
        del song
        gc.collect()
    aggregate_metrics = hard.metric_result(
        vocals=aggregate["vocals"],
        teacher_vocals=aggregate["teacherVocals"],
        instrumental=aggregate["instrumental"],
        teacher=aggregate["teacher"],
        low_instrumental=aggregate["lowInstrumental"],
        vocal_basis_power=vocal_basis,
        accompaniment_vocal_dot=accompaniment_dot,
        sample_count=sample_count,
    )
    aggregate_metrics["eventMetrics"] = hard.summarize_event_values(event_values)
    aggregate_metrics["mechanicalArtifactProxy"] = hard.mechanical_metrics(
        derivative,
        teacher_derivative,
        excess,
        clip_count,
        nonfinite_count,
    )
    return {
        "variant": name,
        "aggregate": aggregate_metrics,
        "perSong": per_song,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
        "contract": contract.as_dict(),
    }


def evaluate_checkpoint(
    *,
    name: str,
    state_path: Path,
    architecture_checkpoint: Path,
    entries: list[dict[str, Any]],
    eval_root: Path,
    contract: short.ShortWindowContract,
    windows_per_song: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = make_short_model(state_path, contract.num_frames, device)
    try:
        result = evaluate_model(
            name=name,
            model=model,
            entries=entries,
            eval_root=eval_root,
            contract=contract,
            windows_per_song=windows_per_song,
            seed=seed,
            device=device,
        )
        result["checkpoint"] = metadata
        return result
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def render_short_song(
    model: torch.nn.Module,
    source: np.ndarray,
    contract: short.ShortWindowContract,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.empty_like(source)
    started = time.perf_counter()
    count = math.ceil(source.shape[0] / contract.stride_samples)
    for index in range(count):
        start = index * contract.stride_samples
        length = min(contract.stride_samples, source.shape[0] - start)
        residual = predict_short(model, source, start, length, contract, device)
        output[start : start + length] = source[start : start + length] - residual
    if not np.isfinite(output).all():
        raise ValueError("Short listening output is non-finite")
    return output, {
        "windowCount": count,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def render_listening(
    *,
    output_root: Path,
    samples_root: Path,
    state_paths: dict[str, Path],
    contract: short.ShortWindowContract,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    songs = density.validate_private_songs(samples_root.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, torch.nn.Module] = {}
    report: dict[str, Any] = {
        "schema": "local-inst3-tfc-f24-listening@1",
        "status": "running",
        "contract": contract.as_dict(),
        "outputRoot": str(output_root.resolve()),
        "models": {},
        "songs": {},
    }
    for name, path in state_paths.items():
        model, metadata = make_short_model(path, contract.num_frames, device)
        models[name] = model
        report["models"][name] = {**metadata, "checkpoint": checkpoint_metadata(path)}
    try:
        for index, song in enumerate(songs, start=1):
            source, sample_rate = listening.load_audio(Path(song["file"]))
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f"Unexpected sample rate for {song['file']}: {sample_rate}")
            report["songs"][song["name"]] = {
                "source": {
                    **song,
                    "frames": int(source.shape[0]),
                    "sampleRate": sample_rate,
                    "decodedFloat32Sha256": pilot.sha256_array(source),
                },
                "outputs": {},
            }
            for name, model in models.items():
                path = output_root / name / f"{song['name']}.flac"
                skipped = path.is_file() and not force
                if not skipped:
                    rendered, timing = render_short_song(model, source, contract, device)
                    listening.write_flac(path, rendered, sample_rate)
                    del rendered
                decoded, decoded_rate = listening.load_audio(path)
                if decoded.shape != source.shape or decoded_rate != sample_rate:
                    raise ValueError(f"Listening output mismatch: {path}")
                report["songs"][song["name"]]["outputs"][name] = {
                    "file": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "frames": int(decoded.shape[0]),
                    "sampleRate": int(decoded_rate),
                    "channels": int(decoded.shape[1]),
                    "skippedExisting": skipped,
                }
            json_write(output_root / "render-report.json", report)
            print(f"listening {index}/{len(songs)}: {song['name']}", flush=True)
            del source
            gc.collect()
    finally:
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "completed"
    report["outputCount"] = sum(
        len(value["outputs"]) for value in report["songs"].values()
    )
    json_write(output_root / "render-report.json", report)
    return report


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return hard.metric_delta(candidate, baseline)


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.num_frames != 24:
        raise ValueError("This first-phase runner is fixed to num-frames=24")
    if args.context_hops != 4:
        raise ValueError("This first-phase runner is fixed to four context hops")
    if args.train_windows_per_song != 8 or args.events_per_song != 4:
        raise ValueError("The first-phase contract requires 8 windows and 4 events per song")
    if args.passes <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("passes, batch-size, and threads must be positive")
    if args.learning_rate <= 0.0 or args.state_interval <= 0:
        raise ValueError("learning-rate and state-interval must be positive")
    if milestones[0] != 0 or milestones[-1] != args.passes:
        raise ValueError("milestones must start at 0 and end at passes")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted and unique")
    if args.max_songs is not None and args.max_songs <= 0:
        raise ValueError("max-songs must be positive")
    if args.max_eval_songs is not None and args.max_eval_songs <= 0:
        raise ValueError("max-eval-songs must be positive")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_int_list(args.milestones)
    validate_args(args, milestones)
    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    oracle_root = args.oracle_root.resolve()
    event_root = args.event_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    h50_checkpoint = args.h50_checkpoint.resolve()
    teacher = args.teacher.resolve()
    contract_path = args.contract.resolve()
    teacher_tflite = args.teacher_tflite.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not checkpoint.is_file() or not h50_checkpoint.is_file():
        raise FileNotFoundError("Student architecture or H50 checkpoint is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    contract = make_contract(args.num_frames, args.context_hops)
    event_selection, event_rows_by_song, event_h50_sha = load_event_selection(
        event_root, args.events_per_song, args.max_songs
    )
    actual_h50_sha = file_sha256(h50_checkpoint)
    if event_h50_sha.lower() != actual_h50_sha.lower():
        raise ValueError(f"H50 checkpoint mismatch: {event_h50_sha} != {actual_h50_sha}")
    manifest = hard.load_manifest(manifest_path)
    train_entries = hard.load_train_entries(manifest, args.max_songs)
    entry_by_slug = {hard.oracle.slugify(entry["fileName"]): entry for entry in train_entries}
    if set(entry_by_slug) != set(event_rows_by_song):
        raise ValueError("Event selection and manifest train song sets differ")
    output_root.mkdir(parents=True, exist_ok=True)
    prep_root = output_root / "prep-work"
    shutil.rmtree(prep_root, ignore_errors=True)
    prep_root.mkdir(parents=True, exist_ok=True)
    contract_info = pilot.verify_teacher_contract(contract_path, teacher, teacher_tflite)
    teacher_session, teacher_providers = pilot.make_teacher_session(
        teacher, args.threads, args.require_teacher_cuda
    )
    plans: dict[str, SongPlan] = {}
    cache_paths_by_song: dict[str, Path] = {}
    cache_metadata: dict[str, Any] = {}
    try:
        for song_index, entry in enumerate(train_entries):
            slug = hard.oracle.slugify(entry["fileName"])
            length = read_decoded_length(oracle_root, slug)
            song = None
            if length is None:
                song = hard.oracle.decode_song(
                    archive,
                    entry,
                    prep_root / "raw",
                    oracle_root / "decoded",
                    force_extract=False,
                    force_decode=False,
                    keep_decoded_wav=False,
                )
                length = int(song.mixture_gt.shape[0])
            plan = build_song_plan(
                slug=slug,
                member=entry["member"],
                song_samples=length,
                event_rows=event_rows_by_song[slug],
                contract=contract,
                train_windows_per_song=args.train_windows_per_song,
                seed=args.seed,
                song_index=song_index,
            )
            expected_cache, expected_metadata = cache_paths(output_root, slug)
            cache_valid = False
            if expected_cache.is_file() and expected_metadata.is_file() and not args.force_cache:
                try:
                    current = json.loads(expected_metadata.read_text(encoding="utf-8"))
                    cache_valid = current.get("contract", {}).get("records") == [
                        {
                            "start": record.start,
                            "length": record.length,
                            "kind": record.kind,
                            "eventStart": record.event_start,
                            "eventEnd": record.event_end,
                        }
                        for record in plan.records
                    ] and current.get("contract", {}).get("h50CheckpointSha256") == actual_h50_sha
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    cache_valid = False
            if not cache_valid:
                if song is None:
                    song = hard.oracle.decode_song(
                        archive,
                        entry,
                        prep_root / "raw",
                        oracle_root / "decoded",
                        force_extract=False,
                        force_decode=False,
                        keep_decoded_wav=False,
                    )
                cache_path, metadata = prepare_song_cache(
                    output_root=output_root,
                    song=song,
                    plan=plan,
                    entry=entry,
                    teacher_session=teacher_session,
                    contract=contract,
                    checkpoint=checkpoint,
                    h50_checkpoint=h50_checkpoint,
                    teacher=teacher,
                    force=args.force_cache,
                )
                cache_metadata[slug] = metadata
            else:
                cache_path = expected_cache
                cache_metadata[slug] = json.loads(expected_metadata.read_text(encoding="utf-8"))
            plans[slug] = plan
            cache_paths_by_song[slug] = cache_path
            print(f"prepared F24 {song_index + 1}/{len(train_entries)}: {slug}", flush=True)
            del song
            shutil.rmtree(prep_root, ignore_errors=True)
            prep_root.mkdir(parents=True, exist_ok=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(prep_root, ignore_errors=True)
        del teacher_session
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection_payload = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": contract.as_dict(),
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "manifestSha256": file_sha256(manifest_path),
        "eventSelectionSha256": file_sha256(event_root / "selection.json"),
        "trainSongCount": len(plans),
        "eventsPerSong": args.events_per_song,
        "songs": {
            slug: {
                "member": plan.member,
                "records": [
                    {
                        "index": record.index,
                        "start": record.start,
                        "length": record.length,
                        "kind": record.kind,
                        "eventStart": record.event_start,
                        "eventEnd": record.event_end,
                    }
                    for record in plan.records
                ],
                "uniformIndices": list(plan.uniform_indices),
                "eventIndices": list(plan.event_indices),
            }
            for slug, plan in sorted(plans.items())
        },
    }
    selection_path = output_root / "selection.json"
    json_write(selection_path, selection_payload)
    schedule_uniform = build_schedule(plans, UNIFORM, args.passes, args.seed)
    schedule_h25 = build_schedule(plans, H25, args.passes, args.seed)
    schedule_summary = {
        variant: {
            "recordCount": len(schedule),
            "recordsPerPass": len(schedule) // args.passes,
            "updates": len(schedule) // args.batch_size,
            "scheduleSha256": canonical_sha256(schedule_json(schedule)),
        }
        for variant, schedule in ((UNIFORM, schedule_uniform), (H25, schedule_h25))
    }
    if any(value["recordsPerPass"] % args.batch_size for value in schedule_summary.values()):
        raise ValueError("F24 schedule is not divisible by batch size")
    store = ShortCacheStore(cache_paths_by_song)
    common_contract = {
        "schema": SCHEMA,
        "shortSelectionSha256": file_sha256(selection_path),
        "eventSelectionSha256": file_sha256(event_root / "selection.json"),
        "manifestSha256": file_sha256(manifest_path),
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "teacher": contract_info,
        "teacherProviders": teacher_providers,
        "contract": contract.as_dict(),
        "trainSongCount": len(train_entries),
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "officialFinalTestUsed": False,
    }
    training: dict[str, Any] = {}
    for variant, schedule in ((UNIFORM, schedule_uniform), (H25, schedule_h25)):
        variant_contract = {
            **common_contract,
            "variant": variant,
            "scheduleSha256": schedule_summary[variant]["scheduleSha256"],
        }
        training[variant] = train_variant(
            variant=variant,
            state_source=h50_checkpoint,
            contract_id=canonical_sha256(variant_contract),
            store=store,
            schedule=schedule,
            run_root=output_root / "runs" / safe_name(variant),
            contract=contract,
            passes=args.passes,
            milestones=milestones,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
            state_interval=args.state_interval,
            resume=args.resume,
        )
        training[variant]["contract"] = {
            "id": canonical_sha256(variant_contract),
            "payload": variant_contract,
        }
    if args.smoke_only:
        report = {
            "schema": SCHEMA,
            "status": "smoke-completed",
            "contract": common_contract,
            "schedule": schedule_summary,
            "training": training,
            "cacheSongCount": len(cache_metadata),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchCuda": torch.version.cuda,
                "cudaAvailable": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "device": str(device),
            },
        }
        path = output_root / "reports" / "inst3-tfc-f24-smoke-report.json"
        json_write(path, report)
        print(json.dumps({"status": report["status"], "report": str(path), "training": list(training)}, indent=2), flush=True)
        return 0
    eval_entries = sorted(
        [entry for entry in manifest["entries"] if entry.get("role") in {"calibration", "internal-test"}],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    evaluation: dict[str, Any] = {
        "F24-static-H50": evaluate_checkpoint(
            name="F24-static-H50",
            state_path=h50_checkpoint,
            architecture_checkpoint=checkpoint,
            entries=eval_entries,
            eval_root=oracle_root,
            contract=contract,
            windows_per_song=args.eval_windows_per_song,
            seed=args.seed,
            device=device,
        )
    }
    for variant, result in training.items():
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
                eval_root=oracle_root,
                contract=contract,
                windows_per_song=args.eval_windows_per_song,
                seed=args.seed,
                device=device,
            )
    baseline = evaluation["F24-static-H50"]
    for name, value in evaluation.items():
        if name != "F24-static-H50":
            value["deltaVsF24StaticH50"] = metric_delta(value, baseline)
    listening_report = None
    if not args.skip_listening:
        state_paths = {"F24-static-H50": h50_checkpoint}
        for variant, result in training.items():
            step = args.passes * result["updatesPerPass"]
            state_paths[f"{variant}@pass-{args.passes}"] = Path(
                result["milestoneCheckpoints"][str(step)]["file"]
            )
        listening_report = render_listening(
            output_root=output_root / "listening-12",
            samples_root=args.samples_root.resolve(),
            state_paths=state_paths,
            contract=contract,
            device=device,
            force=args.force_listening,
        )
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "selection": selection_payload,
        "schedule": schedule_summary,
        "cache": {
            "songCount": len(cache_metadata),
            "root": str((output_root / "cache" / "train").resolve()),
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
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "notes": {
            "purpose": "Dedicated 24-frame retraining from H50 with Inst 3 residual targets",
            "variants": "U uses eight uniform windows; H25 uses four shared uniform plus four event-centered windows",
            "publication": "Local non-commercial research only; do not publish MUSDB18-derived checkpoints, caches, or audio.",
            "nextGate": "Compare against F24-static-H50 and 128-frame H50; no Android export before host quality and listening gates.",
        },
    }
    report_path = output_root / "reports" / "inst3-tfc-f24-retrain-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "trainingVariants": list(training), "evaluationVariants": list(evaluation), "listeningCount": listening_report.get("outputCount", 0) if listening_report else 0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
