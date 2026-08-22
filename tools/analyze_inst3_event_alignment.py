#!/usr/bin/env python3
"""Measure Inst 3-removal event sensitivity to TFC window alignment.

This is a local, non-commercial diagnostic. It does not train or export a
model. For the top 100 ms events in each selected song it compares the
existing useful-stride window with an event-centered window and two windows
shifted by one quarter of the useful span. The same initial, V-R-H50, and
event-weighted student checkpoints are evaluated on every placement.

Train-song teacher output is generated only for the sparse diagnostic windows;
full-song train teacher intermediates are not retained. Evaluation-song
teacher output is read from the frozen song-disjoint oracle cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import analyze_inst3_vr_hard_events as event_analysis
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_teacher_oracle as oracle
import run_inst3_vr_hard_sampling as hard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = pilot.DEFAULT_ARCHIVE
DEFAULT_MANIFEST = pilot.DEFAULT_MANIFEST
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-event-alignment"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_H50_CHECKPOINT = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-hard-sampling-h50"
    / "runs"
    / "V-R-H50"
    / "step-8000.pt"
)
DEFAULT_WEIGHTED_CHECKPOINT = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-event-weighted"
    / "runs"
    / "lambda-0.50"
    / "step-8000.pt"
)
EVENT_MILLISECONDS = (50, 100, 200)
PLACEMENTS = (
    "stride",
    "center",
    "center-minus-quarter",
    "center-plus-quarter",
)
SCHEMA = "local-inst3-event-alignment@1"


@dataclass(frozen=True)
class EventRecord:
    slug: str
    role: str
    member: str
    rank: int
    start: int
    end: int
    score: float
    classification: str
    teacher_rms_dbfs: float
    true_vocal_rms_dbfs: float
    teacher_vocal_cosine: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass(frozen=True)
class Placement:
    event_rank: int
    name: str
    start: int
    length: int
    relative_center: float
    edge_distance: float


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument(
        "--weighted-checkpoint",
        type=Path,
        default=DEFAULT_WEIGHTED_CHECKPOINT,
    )
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument(
        "--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE
    )
    parser.add_argument("--top-events", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-train-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def clamp_start(value: float, song_samples: int, useful_samples: int) -> int:
    maximum = max(0, song_samples - useful_samples)
    return min(max(int(round(value)), 0), maximum)


def event_placements(
    event: EventRecord,
    *,
    stride_start: int,
    song_samples: int,
    useful_samples: int,
) -> list[Placement]:
    if song_samples <= 0:
        raise ValueError("song_samples must be positive")
    length = min(useful_samples, song_samples)
    center_start = clamp_start(
        event.center - useful_samples / 2.0, song_samples, useful_samples
    )
    starts = {
        "stride": clamp_start(stride_start, song_samples, useful_samples),
        "center": center_start,
        "center-minus-quarter": clamp_start(
            center_start - useful_samples / 4.0, song_samples, useful_samples
        ),
        "center-plus-quarter": clamp_start(
            center_start + useful_samples / 4.0, song_samples, useful_samples
        ),
    }
    result: list[Placement] = []
    for name in PLACEMENTS:
        start = starts[name]
        relative = (event.center - start) / float(length)
        result.append(
            Placement(
                event_rank=event.rank,
                name=name,
                start=start,
                length=min(length, song_samples - start),
                relative_center=relative,
                edge_distance=min(relative, 1.0 - relative),
            )
        )
    return result


def event_from_row(
    slug: str,
    role: str,
    member: str,
    row: dict[str, Any],
    *,
    rank: int | None = None,
) -> EventRecord:
    return EventRecord(
        slug=slug,
        role=role,
        member=member,
        rank=int(row.get("rank", 0) if rank is None else rank),
        start=int(row["startSamples"]),
        end=int(row["endSamples"]),
        score=float(row.get("score", 0.0)),
        classification=str(row.get("classification", "unknown")),
        teacher_rms_dbfs=float(row.get("teacherRmsDbfs", -240.0)),
        true_vocal_rms_dbfs=float(row.get("trueVocalRmsDbfs", -240.0)),
        teacher_vocal_cosine=float(row.get("teacherVocalCosine", 0.0)),
    )


def load_train_events(
    event_root: Path, entry: dict[str, Any], top_count: int
) -> tuple[list[EventRecord], dict[str, Any]]:
    slug = oracle.slugify(entry["fileName"])
    path = event_root / "songs" / f"{slug}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in report["topEvents"] if int(row["milliseconds"]) == 100]
    rows.sort(key=lambda row: (-float(row["score"]), int(row["startSamples"])))
    if len(rows) < top_count:
        raise ValueError(f"Only {len(rows)} top 100 ms events for {slug}")
    return [
        event_from_row(
            slug,
            entry["role"],
            entry["member"],
            row,
            rank=index,
        )
        for index, row in enumerate(rows[:top_count])
    ], report


def load_eval_entries(manifest: dict[str, Any], max_songs: int | None) -> list[dict[str, Any]]:
    entries = sorted(
        [
            entry
            for entry in manifest["entries"]
            if entry.get("role") in {"calibration", "internal-test"}
        ],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    if max_songs is not None:
        if max_songs <= 0:
            raise ValueError("max-eval-songs must be positive")
        entries = entries[:max_songs]
    return entries


def load_train_entries(manifest: dict[str, Any], max_songs: int | None) -> list[dict[str, Any]]:
    entries = sorted(
        [entry for entry in manifest["entries"] if entry.get("role") == "train"],
        key=lambda entry: entry["member"],
    )
    if max_songs is not None:
        if max_songs <= 0:
            raise ValueError("max-train-songs must be positive")
        entries = entries[:max_songs]
    return entries


def model_checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def load_student_models(
    checkpoint: Path,
    h50_checkpoint: Path,
    weighted_checkpoint: Path,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    for path in (checkpoint, h50_checkpoint, weighted_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    models: dict[str, torch.nn.Module] = {}
    specs = (
        ("initial", None),
        ("h50-pass-50", h50_checkpoint),
        ("lambda-0.50-pass-50", weighted_checkpoint),
    )
    for name, state_path in specs:
        model, _ = pilot.make_model(checkpoint, device)
        if state_path is not None:
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["stateDict"], strict=True)
            del payload
        sweep.freeze_batchnorm_running_statistics(model)
        model.eval()
        models[name] = model
    return models


def render_residuals(
    audio: np.ndarray,
    placements: Sequence[Placement],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[tuple[int, int], np.ndarray]:
    unique: dict[tuple[int, int], None] = {
        (placement.start, placement.length): None for placement in placements
    }
    keys = list(unique)
    result: dict[tuple[int, int], np.ndarray] = {}
    trim = pilot.DEFAULT_CONFIG.trim_samples
    with torch.inference_mode():
        for begin in range(0, len(keys), batch_size):
            batch_keys = keys[begin : begin + batch_size]
            specs = np.stack(
                [pilot.student_window_spec(audio, start, length) for start, length in batch_keys]
            )
            output = model(torch.from_numpy(specs).to(device))
            output_array = output.detach().cpu().numpy()
            for index, (start, length) in enumerate(batch_keys):
                reconstructed = pilot.student_istft_centered(
                    output_array[index : index + 1]
                )
                residual = np.ascontiguousarray(
                    reconstructed[trim : trim + length], dtype=np.float32
                )
                if residual.shape != (length, 2) or not np.isfinite(residual).all():
                    raise ValueError(f"Invalid residual for window {start}:{start + length}")
                result[(start, length)] = residual
    return result


def render_initial_full_song(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    useful = pilot.DEFAULT_CONFIG.useful_samples
    placements = [
        Placement(
            event_rank=0,
            name="grid",
            start=start,
            length=min(useful, audio.shape[0] - start),
            relative_center=0.0,
            edge_distance=0.0,
        )
        for start in range(0, audio.shape[0], useful)
        if audio.shape[0] - start > 0
    ]
    residuals = render_residuals(audio, placements, model, device, batch_size)
    instrumental = np.empty_like(audio)
    for placement in placements:
        key = (placement.start, placement.length)
        instrumental[placement.start : placement.start + placement.length] = (
            audio[placement.start : placement.start + placement.length] - residuals[key]
        )
    if not np.isfinite(instrumental).all():
        raise ValueError("Initial full-song output contains non-finite values")
    return instrumental


def derive_eval_events(
    song: Any,
    initial_instrumental: np.ndarray,
    top_count: int,
) -> list[EventRecord]:
    vocal_values = event_analysis.rms_blocks(
        song.vocals, round(song.sample_rate * 100 / 1000.0)
    )
    vocal_p20 = float(event_analysis.dbfs(float(np.percentile(vocal_values, 20.0))))
    vocal_p80 = float(event_analysis.dbfs(float(np.percentile(vocal_values, 80.0))))
    rows: list[dict[str, Any]] = []
    for milliseconds in EVENT_MILLISECONDS:
        features = event_analysis.block_features(
            mixture=song.mixture_gt,
            teacher_instrumental=song.teacher_instrumental,
            student_instrumental=initial_instrumental,
            true_vocals=song.vocals,
            sample_rate=song.sample_rate,
            milliseconds=milliseconds,
        )
        rows.extend(
            {"milliseconds": milliseconds, **row}
            for row in event_analysis.top_event_rows(
                features,
                sample_rate=song.sample_rate,
                top_count=top_count,
                vocal_p20_dbfs=vocal_p20,
                vocal_p80_dbfs=vocal_p80,
            )
        )
    primary = [row for row in rows if int(row["milliseconds"]) == 100]
    primary.sort(key=lambda row: (-float(row["score"]), int(row["startSamples"])))
    return [
        event_from_row(
            song.slug,
            song.role,
            song.member,
            row,
            rank=index,
        )
        for index, row in enumerate(primary[:top_count])
    ]


def current_stride_start(
    event: EventRecord,
    *,
    useful_samples: int,
    song_samples: int,
    train_report: dict[str, Any] | None,
) -> int:
    if train_report is not None:
        candidates = [
            row
            for row in train_report["studentWindows"]["ranked"]
            if int(row["startSamples"]) <= event.start < int(row["endSamples"])
        ]
        if candidates:
            selected = max(
                candidates,
                key=lambda row: (
                    float(row.get("hardEventScore", 0.0)),
                    -int(row["startSamples"]),
                ),
            )
            return int(selected["startSamples"])
    return min((event.start // useful_samples) * useful_samples, max(0, song_samples - useful_samples))


def make_teacher_selection(
    slug: str,
    member: str,
    placements: Sequence[Placement],
) -> hard.SongSelection:
    unique: list[tuple[int, int]] = []
    for placement in placements:
        key = (placement.start, placement.length)
        if key not in unique:
            unique.append(key)
    candidates = tuple(
        hard.Candidate(
            index=index,
            start=start,
            length=length,
            hard_score=0.0,
            rank=index + 1,
        )
        for index, (start, length) in enumerate(unique)
    )
    indices = tuple(range(len(candidates)))
    return hard.SongSelection(
        slug=slug,
        member=member,
        candidates=candidates,
        hard_pool_indices=indices,
        uniform_indices=indices,
        hard_fraction=0.0,
        hard_indices=indices,
        union_indices=indices,
    )


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1.0e-12))


def block_metric(
    *,
    mixture: np.ndarray,
    teacher_instrumental_segment: np.ndarray,
    predicted_residual: np.ndarray,
    placement: Placement,
    center: float,
    milliseconds: int,
    sample_rate: int,
) -> dict[str, Any]:
    block_samples = max(1, round(sample_rate * milliseconds / 1000.0))
    block_start = int(round(center - block_samples / 2.0))
    block_end = block_start + block_samples
    segment_start = placement.start
    segment_end = placement.start + placement.length
    overlap_start = max(block_start, segment_start, 0)
    overlap_end = min(block_end, segment_end, mixture.shape[0])
    requested = max(1, block_end - block_start)
    coverage = max(0, overlap_end - overlap_start) / float(requested)
    if overlap_end <= overlap_start:
        return {
            "coverage": coverage,
            "missRmsDbfs": None,
            "positiveProjectionRmsDbfs": None,
            "teacherRmsDbfs": None,
        }
    local_start = overlap_start - segment_start
    local_end = overlap_end - segment_start
    mix = mixture[overlap_start:overlap_end].astype(np.float64, copy=False)
    teacher = teacher_instrumental_segment[
        local_start:local_end
    ].astype(np.float64, copy=False)
    residual = predicted_residual[local_start:local_end].astype(np.float64, copy=False)
    teacher_removed = mix - teacher
    miss = teacher_removed - residual
    teacher_power = float(np.sum(teacher_removed * teacher_removed))
    miss_power = float(np.sum(miss * miss))
    miss_rms = math.sqrt(miss_power / float(max(1, teacher_removed.size)))
    teacher_rms = math.sqrt(teacher_power / float(max(1, teacher_removed.size)))
    coefficient = (
        float(np.sum(miss * teacher_removed)) / teacher_power
        if teacher_power > 1.0e-20
        else 0.0
    )
    positive = max(coefficient, 0.0) * teacher_rms
    return {
        "coverage": coverage,
        "missRmsDbfs": dbfs(miss_rms),
        "positiveProjectionRmsDbfs": dbfs(positive),
        "teacherRmsDbfs": dbfs(teacher_rms),
    }


def render_train_teacher_segments(
    song: Any,
    placements: Sequence[Placement],
    session: Any,
    label: str,
) -> dict[tuple[int, int], np.ndarray]:
    selection = make_teacher_selection(song.slug, song.entry["member"], placements)
    segments, _ = hard.render_selected_teacher_segments(
        song.mixture_gt,
        selection,
        session,
        progress_label=label,
    )
    unique = [
        (candidate.start, candidate.length)
        for candidate in selection.candidates
    ]
    return {key: segment for key, segment in zip(unique, segments)}


def event_rows_for_song(
    *,
    song: Any,
    events: Sequence[EventRecord],
    placements_by_event: dict[int, list[Placement]],
    teacher_segments: dict[tuple[int, int], np.ndarray],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    all_placements = [placement for values in placements_by_event.values() for placement in values]
    unique_placements: list[Placement] = []
    seen: set[tuple[int, int]] = set()
    for placement in all_placements:
        key = (placement.start, placement.length)
        if key not in seen:
            seen.add(key)
            unique_placements.append(placement)
    predictions = {
        name: render_residuals(
            song.mixture_gt,
            unique_placements,
            model,
            device,
            batch_size,
        )
        for name, model in models.items()
    }
    rows: list[dict[str, Any]] = []
    for event in events:
        for placement in placements_by_event[event.rank]:
            key = (placement.start, placement.length)
            teacher_segment = teacher_segments[key]
            for model_name, residuals in predictions.items():
                metrics = {
                    str(milliseconds): block_metric(
                        mixture=song.mixture_gt,
                        teacher_instrumental_segment=np.ascontiguousarray(
                            song.mixture_gt[key[0] : key[0] + key[1]] - teacher_segment,
                            dtype=np.float32,
                        ),
                        predicted_residual=residuals[key],
                        placement=placement,
                        center=event.center,
                        milliseconds=milliseconds,
                        sample_rate=song.sample_rate,
                    )
                    for milliseconds in EVENT_MILLISECONDS
                }
                rows.append(
                    {
                        "slug": event.slug,
                        "role": event.role,
                        "member": event.member,
                        "eventRank": event.rank,
                        "eventStartSamples": event.start,
                        "eventEndSamples": event.end,
                        "eventScore": event.score,
                        "classification": event.classification,
                        "teacherRmsDbfs": event.teacher_rms_dbfs,
                        "trueVocalRmsDbfs": event.true_vocal_rms_dbfs,
                        "teacherVocalCosine": event.teacher_vocal_cosine,
                        "model": model_name,
                        "placement": placement.name,
                        "windowStartSamples": placement.start,
                        "windowLengthSamples": placement.length,
                        "relativeEventCenter": placement.relative_center,
                        "edgeDistance": placement.edge_distance,
                        "metrics": metrics,
                    }
                )
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for milliseconds in EVENT_MILLISECONDS:
            grouped[(row["role"], row["model"], row["placement"], milliseconds)].append(row)
    result: dict[str, Any] = {}
    for (role, model, placement, milliseconds), values in sorted(grouped.items()):
        metrics = [value["metrics"][str(milliseconds)] for value in values]
        full = [value for value in metrics if value["coverage"] >= 0.999]
        positive = [
            10.0 ** (value["positiveProjectionRmsDbfs"] / 20.0)
            for value in full
            if value["positiveProjectionRmsDbfs"] is not None
        ]
        miss = [
            10.0 ** (value["missRmsDbfs"] / 20.0)
            for value in full
            if value["missRmsDbfs"] is not None
        ]
        key = f"{role}/{model}/{placement}/{milliseconds}ms"
        result[key] = {
            "role": role,
            "model": model,
            "placement": placement,
            "milliseconds": milliseconds,
            "eventCount": len(values),
            "fullCoverageCount": len(full),
            "meanCoverage": float(np.mean([value["coverage"] for value in metrics])),
            "positiveProjectionRmsP50Dbfs": dbfs(float(np.percentile(positive, 50))) if positive else None,
            "positiveProjectionRmsP95Dbfs": dbfs(float(np.percentile(positive, 95))) if positive else None,
            "positiveProjectionRmsMaxDbfs": dbfs(float(np.max(positive))) if positive else None,
            "missRmsP50Dbfs": dbfs(float(np.percentile(miss, 50))) if miss else None,
            "missRmsP95Dbfs": dbfs(float(np.percentile(miss, 95))) if miss else None,
            "missRmsMaxDbfs": dbfs(float(np.max(miss))) if miss else None,
        }
    return result


def compare_center_to_stride(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        indexed[(row["role"], row["model"], int(row["eventRank"]), row["slug"])][
            row["placement"]
        ] = row
    result: dict[str, Any] = {}
    for model in sorted({row["model"] for row in rows}):
        for role in sorted({row["role"] for row in rows}):
            for milliseconds in EVENT_MILLISECONDS:
                deltas: list[float] = []
                for (row_role, row_model, _, _), placements in indexed.items():
                    if row_role != role or row_model != model:
                        continue
                    center = placements.get("center", {}).get("metrics", {}).get(str(milliseconds))
                    stride = placements.get("stride", {}).get("metrics", {}).get(str(milliseconds))
                    if not center or not stride:
                        continue
                    if center["coverage"] < 0.999 or stride["coverage"] < 0.999:
                        continue
                    if center["positiveProjectionRmsDbfs"] is None or stride["positiveProjectionRmsDbfs"] is None:
                        continue
                    deltas.append(
                        float(center["positiveProjectionRmsDbfs"])
                        - float(stride["positiveProjectionRmsDbfs"])
                    )
                key = f"{role}/{model}/{milliseconds}ms"
                result[key] = {
                    "role": role,
                    "model": model,
                    "milliseconds": milliseconds,
                    "eventCount": len(deltas),
                    "centerBetterCount": sum(delta < 0.0 for delta in deltas),
                    "centerWorseCount": sum(delta > 0.0 for delta in deltas),
                    "deltaP50Db": float(np.percentile(deltas, 50)) if deltas else None,
                    "deltaP95Db": float(np.percentile(deltas, 95)) if deltas else None,
                    "deltaMinDb": float(np.min(deltas)) if deltas else None,
                    "deltaMaxDb": float(np.max(deltas)) if deltas else None,
                }
    return result


def position_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in sorted({row["role"] for row in rows}):
        # Position is an event property, so do not count it once per model.
        unique: dict[tuple[str, int], float] = {}
        for row in rows:
            if row["role"] == role and row["placement"] == "stride":
                unique[(row["slug"], int(row["eventRank"]))] = float(
                    row["relativeEventCenter"]
                )
        values = list(unique.values())
        result[role] = {
            "eventCount": len(values),
            "relativeCenterP10": float(np.percentile(values, 10)) if values else None,
            "relativeCenterP50": float(np.percentile(values, 50)) if values else None,
            "relativeCenterP90": float(np.percentile(values, 90)) if values else None,
            "outsideWindowCount": sum(value < 0.0 or value > 1.0 for value in values),
            "edgeWithin10PercentCount": sum(
                0.0 <= value <= 1.0 and (value < 0.1 or value > 0.9)
                for value in values
            ),
            "edgeWithin10PercentFraction": (
                float(
                    sum(
                        0.0 <= value <= 1.0 and (value < 0.1 or value > 0.9)
                        for value in values
                    )
                    / len(values)
                )
                if values
                else None
            ),
        }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_events <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("top-events, batch-size, and threads must be positive")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    train_entries = load_train_entries(manifest, args.max_train_songs)
    eval_entries = load_eval_entries(manifest, args.max_eval_songs)
    checkpoint = args.checkpoint.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    models = load_student_models(
        checkpoint,
        args.h50_checkpoint.resolve(),
        args.weighted_checkpoint.resolve(),
        device,
    )
    teacher_session, teacher_providers = pilot.make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    work_root = args.output_root.resolve() / "work"
    rows: list[dict[str, Any]] = []
    event_manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for song_index, entry in enumerate(train_entries, start=1):
            slug = oracle.slugify(entry["fileName"])
            events, train_report = load_train_events(args.event_root.resolve(), entry, args.top_events)
            print(f"train alignment {song_index}/{len(train_entries)}: {slug}", flush=True)
            song = oracle.decode_song(
                args.archive.resolve(),
                entry,
                work_root / "raw",
                work_root / "decoded",
                force_extract=False,
                force_decode=False,
                keep_decoded_wav=False,
            )
            try:
                placements_by_event = {
                    event.rank: event_placements(
                        event,
                        stride_start=current_stride_start(
                            event,
                            useful_samples=pilot.DEFAULT_CONFIG.useful_samples,
                            song_samples=song.mixture_gt.shape[0],
                            train_report=train_report,
                        ),
                        song_samples=song.mixture_gt.shape[0],
                        useful_samples=pilot.DEFAULT_CONFIG.useful_samples,
                    )
                    for event in events
                }
                all_placements = [p for values in placements_by_event.values() for p in values]
                teacher_segments = render_train_teacher_segments(
                    song,
                    all_placements,
                    teacher_session,
                    f"train/{slug}",
                )
                song_rows = event_rows_for_song(
                    song=song,
                    events=events,
                    placements_by_event=placements_by_event,
                    teacher_segments=teacher_segments,
                    models=models,
                    device=device,
                    batch_size=args.batch_size,
                )
                rows.extend(song_rows)
                event_manifest.extend(
                    {
                        "slug": event.slug,
                        "role": event.role,
                        "member": event.member,
                        "rank": event.rank,
                        "startSamples": event.start,
                        "endSamples": event.end,
                        "score": event.score,
                        "classification": event.classification,
                        "placements": [placement.__dict__ for placement in placements_by_event[event.rank]],
                    }
                    for event in events
                )
            finally:
                del song
                if work_root.exists():
                    shutil.rmtree(work_root, ignore_errors=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        initial_model = models["initial"]
        for song_index, entry in enumerate(eval_entries, start=1):
            song = hard.load_eval_song(args.eval_root.resolve(), entry)
            print(f"eval alignment {song_index}/{len(eval_entries)}: {song.slug}", flush=True)
            initial_instrumental = render_initial_full_song(
                song.mixture_gt,
                initial_model,
                device,
                args.batch_size,
            )
            events = derive_eval_events(song, initial_instrumental, args.top_events)
            placements_by_event = {
                event.rank: event_placements(
                    event,
                    stride_start=current_stride_start(
                        event,
                        useful_samples=pilot.DEFAULT_CONFIG.useful_samples,
                        song_samples=song.mixture_gt.shape[0],
                        train_report=None,
                    ),
                    song_samples=song.mixture_gt.shape[0],
                    useful_samples=pilot.DEFAULT_CONFIG.useful_samples,
                )
                for event in events
            }
            all_placements = [p for values in placements_by_event.values() for p in values]
            teacher_segments = {
                (placement.start, placement.length): np.ascontiguousarray(
                    song.teacher_instrumental[
                        placement.start : placement.start + placement.length
                    ],
                    dtype=np.float32,
                )
                for placement in all_placements
            }
            rows.extend(
                event_rows_for_song(
                    song=song,
                    events=events,
                    placements_by_event=placements_by_event,
                    teacher_segments=teacher_segments,
                    models=models,
                    device=device,
                    batch_size=args.batch_size,
                )
            )
            event_manifest.extend(
                {
                    "slug": event.slug,
                    "role": event.role,
                    "member": event.member,
                    "rank": event.rank,
                    "startSamples": event.start,
                    "endSamples": event.end,
                    "score": event.score,
                    "classification": event.classification,
                    "placements": [placement.__dict__ for placement in placements_by_event[event.rank]],
                }
                for event in events
            )
            del song, initial_instrumental
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del teacher_session
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_root = args.output_root.resolve()
    json_write(output_root / "event-manifest.json", {"schema": SCHEMA + ".events", "events": event_manifest})
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": {
            "eventMilliseconds": list(EVENT_MILLISECONDS),
            "topEventsPerSong": args.top_events,
            "trainSongCount": len(train_entries),
            "evaluationSongCount": len(eval_entries),
            "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
            "usefulSeconds": pilot.DEFAULT_CONFIG.useful_samples / pilot.DEFAULT_CONFIG.sample_rate,
            "placements": list(PLACEMENTS),
            "placementFormula": "center=eventCenter-usefulSamples/2; shifts=plus/minus usefulSamples/4; clamp to song",
            "studentSemantic": "residual-vocals",
            "teacherTarget": "mixtureGt - Inst3Instrumental",
            "officialFinalTestUsed": False,
        },
        "inputs": {
            "manifest": {"file": str(args.manifest.resolve()), "sha256": file_sha256(args.manifest.resolve())},
            "eventRoot": str(args.event_root.resolve()),
            "eventReport": file_sha256(args.event_root.resolve() / "reports" / "inst3-vr-hard-events-report.json"),
            "checkpoint": model_checkpoint_metadata(checkpoint),
            "h50Checkpoint": model_checkpoint_metadata(args.h50_checkpoint.resolve()),
            "weightedCheckpoint": model_checkpoint_metadata(args.weighted_checkpoint.resolve()),
            "teacher": model_checkpoint_metadata(args.teacher.resolve()),
            "teacherProviders": teacher_providers,
        },
        "positionSummary": position_summary(rows),
        "aggregate": aggregate_rows(rows),
        "centerVsStride": compare_center_to_stride(rows),
        "rowCount": len(rows),
        "eventCount": len(event_manifest),
        "rowsFile": str((output_root / "alignment-rows.json").resolve()),
        "eventManifestFile": str((output_root / "event-manifest.json").resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "cudaDevice": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device),
            "elapsedSeconds": time.perf_counter() - started,
            "runner": {"file": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        },
        "notes": {
            "stride": "Existing useful-sample grid window; train events use the Stage 1 candidate containing the event.",
            "coverage": "Event metrics are reported with coverage; center-vs-stride comparison requires full coverage.",
            "selection": "Top events are selected only from MUSDB18 train and calibration/internal-test songs; private listening songs are not used.",
            "publication": "All outputs remain local non-commercial research artifacts.",
        },
    }
    json_write(output_root / "alignment-rows.json", {"schema": SCHEMA + ".rows", "rows": rows})
    json_write(output_root / "reports" / "inst3-event-alignment-report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str((output_root / "reports" / "inst3-event-alignment-report.json").resolve()),
                "eventCount": len(event_manifest),
                "rowCount": len(rows),
                "elapsedSeconds": report["environment"]["elapsedSeconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
