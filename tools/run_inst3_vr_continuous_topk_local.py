#!/usr/bin/env python3
"""Train a continuous-context H50 top-k local objective.

The runner recomputes the remaining Inst 3-directed miss from the continuous
H50 checkpoint, excludes song edges, join neighborhoods, and useful-span
edges from the training event pool, then compares:

* ``H50-continuation``: full useful-span audio target ``V_T``;
* ``H50-topk-local``: top-k event losses toward ``V_T`` plus an H50 anchor
  outside each event.

This is local, non-commercial MUSDB18 research.  It does not publish or
integrate checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch

import evaluate_inst3_continuous_baseline as continuous
import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_scale10_density as density
import run_inst3_teacher_oracle as oracle
import run_inst3_vr_hard_sampling as hard
from tfc_tdf_short_window import (
    ShortWindowContract,
    assemble_input,
    istft_centered,
    stft_centered,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_ORACLE_ROOT / "musdb18-inst3-oracle-manifest.json"
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
DEFAULT_BASELINE_REPORT = (
    ROOT
    / "data"
    / "musdb18-inst3-continuous-baseline-evaluation"
    / "continuous-baseline-report.json"
)
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local"

SCHEMA = "local-inst3-vr-continuous-topk-local@1"
CACHE_SCHEMA = "local-inst3-vr-continuous-topk-local-cache@1"
CONTINUATION = "H50-continuation"
TOPK_LOCAL = "H50-topk-local"
VARIANTS = (CONTINUATION, TOPK_LOCAL)
SAMPLE_RATE = 44_100
EVENT_MS = (50, 100)
ALL_EVENT_MS = (50, 100, 200)
JOIN_RADIUS = round(SAMPLE_RATE * 0.100)
EPSILON = 1.0e-4


@dataclass(frozen=True)
class EventRecord:
    index: int
    start: int
    length: int
    event_start: int
    event_end: int
    duration_ms: int
    positive_projection: float
    miss_rms: float
    teacher_rms: float
    stage1_score: float


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integers, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--records-per-song", type=int, default=8)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--milestones", default="0,1,3,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=1.0)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--max-train-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--force-prep", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
    parser.add_argument("--skip-continuous-eval", action="store_true")
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=8)
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
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def dbfs(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def load_train_song(oracle_root: Path, entry: dict[str, Any]) -> pilot.SongBundle:
    slug = oracle.slugify(entry["fileName"])
    decoded_dir = oracle_root / "decoded" / "train" / slug
    decoded_path = decoded_dir / "decoded.npz"
    metadata_path = decoded_dir / "decoded.json"
    if not decoded_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing decoded train cache for {slug}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sourceSha256") != entry["sourceSha256"]:
        raise ValueError(f"Source hash mismatch for {slug}")
    with np.load(decoded_path) as values:
        arrays = {
            name: np.ascontiguousarray(values[name], dtype=np.float32)
            for name in ("mixtureGt", "vocals", "instrumentalGt")
        }
    return pilot.SongBundle(
        role="train",
        member=entry["member"],
        source_sha256=entry["sourceSha256"],
        source_path=Path(metadata["sourceFile"]),
        slug=slug,
        mixture_encoded=arrays["mixtureGt"],
        mixture_gt=arrays["mixtureGt"],
        vocals=arrays["vocals"],
        instrumental=arrays["instrumentalGt"],
        drums=np.empty((0, 2), dtype=np.float32),
        bass=np.empty((0, 2), dtype=np.float32),
        other=np.empty((0, 2), dtype=np.float32),
        sample_rate=int(metadata["sampleRate"]),
    )


def event_array_path(event_root: Path, slug: str) -> tuple[Path, Path]:
    report_path = event_root / "songs" / f"{slug}.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    array_path = Path(report["eventArrayFile"])
    if not array_path.is_file():
        array_path = event_root / "events" / f"{slug}.npz"
    if not array_path.is_file():
        raise FileNotFoundError(array_path)
    return report_path, array_path


def load_stage1_event_candidates(
    event_root: Path,
    song: pilot.SongBundle,
    h50_residual: np.ndarray,
    teacher_instrumental: np.ndarray,
    contract: ShortWindowContract,
) -> tuple[list[EventRecord], dict[str, int]]:
    report_path, array_path = event_array_path(event_root, song.slug)
    stage1_report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates: list[EventRecord] = []
    counts = {
        "sourceEvents": 0,
        "activeEvents": 0,
        "songEdgeExcluded": 0,
        "joinOrWindowEdgeExcluded": 0,
        "shortTailExcluded": 0,
    }
    if h50_residual.shape != song.mixture_gt.shape or teacher_instrumental.shape != song.mixture_gt.shape:
        raise ValueError(f"Continuous source shape mismatch for {song.slug}")
    h50_instrumental = song.mixture_gt - h50_residual
    miss_full = h50_instrumental - teacher_instrumental
    teacher_removed_full = song.mixture_gt - teacher_instrumental
    useful = contract.useful_samples
    with np.load(array_path) as values:
        for milliseconds in EVENT_MS:
            starts = np.asarray(values[f"{milliseconds}_startSamples"], dtype=np.int64)
            ends = np.asarray(values[f"{milliseconds}_endSamples"], dtype=np.int64)
            active = np.asarray(values[f"{milliseconds}_active"], dtype=bool)
            stage1_scores = np.asarray(values[f"{milliseconds}_score"], dtype=np.float32)
            block_samples = max(1, round(SAMPLE_RATE * milliseconds / 1000.0))
            count = int(math.ceil(song.mixture_gt.shape[0] / block_samples))
            padded_teacher = np.pad(
                teacher_removed_full.astype(np.float64, copy=False),
                ((0, count * block_samples - song.mixture_gt.shape[0]), (0, 0)),
            ).reshape(count, block_samples, 2)
            padded_miss = np.pad(
                miss_full.astype(np.float64, copy=False),
                ((0, count * block_samples - song.mixture_gt.shape[0]), (0, 0)),
            ).reshape(count, block_samples, 2)
            teacher_power = np.sum(padded_teacher * padded_teacher, axis=(1, 2))
            miss_power = np.sum(padded_miss * padded_miss, axis=(1, 2))
            teacher_rms = np.sqrt(teacher_power / float(block_samples * 2))
            miss_rms = np.sqrt(miss_power / float(block_samples * 2))
            dots = np.sum(padded_miss * padded_teacher, axis=(1, 2))
            coefficient = np.divide(
                dots,
                teacher_power,
                out=np.zeros_like(dots),
                where=teacher_power > 1.0e-20,
            )
            positive = np.maximum(coefficient, 0.0) * teacher_rms
            for block_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                counts["sourceEvents"] += 1
                index = min(block_index, count - 1)
                if not active[block_index] or teacher_rms[index] < 10.0 ** (-60.0 / 20.0):
                    continue
                counts["activeEvents"] += 1
                center = (int(start) + int(end)) // 2
                window_start = (center // useful) * useful
                if window_start + useful > song.mixture_gt.shape[0]:
                    counts["shortTailExcluded"] += 1
                    continue
                local_start = int(start) - window_start
                local_end = int(end) - window_start
                local_center = (local_start + local_end) // 2
                if center < JOIN_RADIUS or center >= song.mixture_gt.shape[0] - JOIN_RADIUS:
                    counts["songEdgeExcluded"] += 1
                    continue
                if not (
                    int(round(useful * 0.10)) <= local_center < int(round(useful * 0.90))
                    and JOIN_RADIUS <= local_center < useful - JOIN_RADIUS
                    and 0 <= local_start < local_end <= useful
                ):
                    counts["joinOrWindowEdgeExcluded"] += 1
                    continue
                candidates.append(
                    EventRecord(
                        index=len(candidates),
                        start=window_start,
                        length=useful,
                        event_start=local_start,
                        event_end=local_end,
                        duration_ms=milliseconds,
                        positive_projection=float(positive[index]),
                        miss_rms=float(miss_rms[index]),
                        teacher_rms=float(teacher_rms[index]),
                        stage1_score=float(stage1_scores[block_index]),
                    )
                )
    if not candidates:
        raise ValueError(f"No continuous central event candidates for {song.slug}: {counts}")
    return candidates, counts


def load_stage1_top_event_candidates(
    event_root: Path,
    song: pilot.SongBundle,
    contract: ShortWindowContract,
) -> tuple[list[EventRecord], dict[str, int]]:
    """Use Stage 1 top events only as sparse indexes for the continuous rescan."""
    report_path, _array_path = event_array_path(event_root, song.slug)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in report.get("topEvents", [])
        if int(row.get("milliseconds", 0)) in EVENT_MS
    ]
    counts = {
        "sourceEvents": len(rows),
        "activeEvents": 0,
        "songEdgeExcluded": 0,
        "joinOrWindowEdgeExcluded": 0,
        "shortTailExcluded": 0,
    }
    candidates: list[EventRecord] = []
    useful = contract.useful_samples
    for row in rows:
        start = int(row["startSamples"])
        end = int(row["endSamples"])
        center = (start + end) // 2
        if float(row.get("teacherRmsDbfs", -240.0)) < -60.0:
            continue
        counts["activeEvents"] += 1
        window_start = (center // useful) * useful
        if window_start + useful > song.mixture_gt.shape[0]:
            counts["shortTailExcluded"] += 1
            continue
        local_start = start - window_start
        local_end = end - window_start
        local_center = (local_start + local_end) // 2
        if center < JOIN_RADIUS or center >= song.mixture_gt.shape[0] - JOIN_RADIUS:
            counts["songEdgeExcluded"] += 1
            continue
        if not (
            int(round(useful * 0.10)) <= local_center < int(round(useful * 0.90))
            and JOIN_RADIUS <= local_center < useful - JOIN_RADIUS
            and 0 <= local_start < local_end <= useful
        ):
            counts["joinOrWindowEdgeExcluded"] += 1
            continue
        candidates.append(
            EventRecord(
                index=len(candidates),
                start=window_start,
                length=useful,
                event_start=local_start,
                event_end=local_end,
                duration_ms=int(row["milliseconds"]),
                positive_projection=0.0,
                miss_rms=0.0,
                teacher_rms=10.0 ** (float(row.get("teacherRmsDbfs", -240.0)) / 20.0),
                stage1_score=float(row.get("score", 0.0)),
            )
        )
    if not candidates:
        raise ValueError(f"No sparse central candidates for {song.slug}: {counts}")
    return candidates, counts


def render_selected_continuous_windows(
    model: torch.nn.Module,
    audio: np.ndarray,
    starts: Sequence[int],
    contract: ShortWindowContract,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if not starts:
        raise ValueError("No selected continuous starts")
    result: dict[int, np.ndarray] = {}
    unique_starts = tuple(sorted(set(int(value) for value in starts)))
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_begin in range(0, len(unique_starts), batch_size):
            batch_starts = unique_starts[batch_begin : batch_begin + batch_size]
            spectra = np.concatenate(
                [
                    stft_centered(
                        assemble_input(
                            audio,
                            start,
                            contract.useful_samples,
                            contract,
                            mode="continuous",
                        ),
                        contract,
                    )
                    for start in batch_starts
                ],
                axis=0,
            )
            predictions = model(torch.from_numpy(spectra).to(device)).detach().cpu().numpy()
            for offset, start in enumerate(batch_starts):
                reconstructed = istft_centered(predictions[offset : offset + 1], contract)
                begin = contract.left_context_samples
                result[start] = np.ascontiguousarray(
                    reconstructed[begin : begin + contract.useful_samples], dtype=np.float32
                )
    return result, {
        "windowCount": len(unique_starts),
        "inferenceBatchSize": batch_size,
        "elapsedSeconds": time.perf_counter() - started,
        "assembly": "continuous-context-overlap-save",
    }


def score_continuous_event_records(
    song: pilot.SongBundle,
    records: Sequence[EventRecord],
    h50_segments: dict[int, np.ndarray],
    teacher_segments: dict[int, np.ndarray],
) -> list[EventRecord]:
    scored: list[EventRecord] = []
    for index, record in enumerate(records):
        mix = song.mixture_gt[record.start : record.start + record.length]
        h50_residual = h50_segments[record.start]
        teacher_instrumental = teacher_segments[record.start]
        event_slice = slice(record.event_start, record.event_end)
        event_mix = mix[event_slice]
        event_h50_residual = h50_residual[event_slice]
        event_teacher_instrumental = teacher_instrumental[event_slice]
        teacher_removed = event_mix - event_teacher_instrumental
        miss = (event_mix - event_h50_residual) - event_teacher_instrumental
        teacher64 = teacher_removed.astype(np.float64)
        miss64 = miss.astype(np.float64)
        teacher_power = float(np.sum(teacher64 * teacher64))
        miss_power = float(np.sum(miss64 * miss64))
        dot = float(np.sum(miss64 * teacher64))
        teacher_rms = math.sqrt(teacher_power / max(teacher_removed.size, 1))
        positive = max(dot / teacher_power, 0.0) * teacher_rms if teacher_power > 1.0e-30 else 0.0
        scored.append(
            EventRecord(
                index=index,
                start=record.start,
                length=record.length,
                event_start=record.event_start,
                event_end=record.event_end,
                duration_ms=record.duration_ms,
                positive_projection=positive,
                miss_rms=math.sqrt(miss_power / max(miss.size, 1)),
                teacher_rms=teacher_rms,
                stage1_score=record.stage1_score,
            )
        )
    return scored


def event_record_json(record: EventRecord, selected: bool = False) -> dict[str, Any]:
    return {
        "index": record.index,
        "startSamples": record.start,
        "lengthSamples": record.length,
        "eventStartSamples": record.event_start,
        "eventEndSamples": record.event_end,
        "durationMs": record.duration_ms,
        "eventCenterSamples": record.start + (record.event_start + record.event_end) // 2,
        "positiveProjectionRmsDbfs": dbfs(record.positive_projection),
        "missRmsDbfs": dbfs(record.miss_rms),
        "teacherRemovedRmsDbfs": dbfs(record.teacher_rms),
        "stage1Score": record.stage1_score,
        "selected": selected,
    }


def select_top_records(
    candidates: Sequence[EventRecord], records_per_song: int
) -> tuple[list[EventRecord], dict[str, int]]:
    if records_per_song <= 0:
        raise ValueError("records-per-song must be positive")
    ranked = sorted(
        candidates,
        key=lambda item: (-item.positive_projection, -item.miss_rms, item.start, item.event_start),
    )
    selected: list[EventRecord] = []
    used_windows: set[int] = set()
    used_centers: list[int] = []
    min_separation = round(SAMPLE_RATE * 0.100)
    for record in ranked:
        center = record.start + (record.event_start + record.event_end) // 2
        if record.start in used_windows:
            continue
        if any(abs(center - other) < min_separation for other in used_centers):
            continue
        selected.append(record)
        used_windows.add(record.start)
        used_centers.append(center)
        if len(selected) == records_per_song:
            break
    if len(selected) < records_per_song:
        for record in ranked:
            if record.start in used_windows:
                continue
            selected.append(record)
            used_windows.add(record.start)
            if len(selected) == records_per_song:
                break
    repeated = 0
    if len(selected) < records_per_song:
        if not selected:
            raise ValueError("Unable to select any event record")
        original = list(selected)
        index = 0
        while len(selected) < records_per_song:
            selected.append(original[index % len(original)])
            repeated += 1
            index += 1
    return selected, {
        "candidateCount": len(candidates),
        "selectedUniqueCount": len(set(item.start for item in selected)),
        "repeatedRecordCount": repeated,
    }


def cache_paths(root: Path, slug: str) -> tuple[Path, Path]:
    base = root / "cache" / "train" / slug
    return base.with_suffix(".npz"), base.with_suffix(".json")


def selection_path(root: Path, slug: str) -> Path:
    return root / "selection" / "songs" / f"{slug}.json"


def load_cached_records(
    root: Path,
    slug: str,
    expected_contract: dict[str, Any],
) -> tuple[list[EventRecord], dict[str, Any]] | None:
    npz_path, metadata_path = cache_paths(root, slug)
    if not npz_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_contract = metadata.get("contract", {})
        if any(
            actual_contract.get(key) != value
            for key, value in expected_contract.items()
            if key != "runnerSha256"
        ):
            return None
        with np.load(npz_path) as values:
            expected = int(metadata["recordCount"])
            if values["inputSpec"].shape[0] != expected:
                return None
        records = [
            EventRecord(
                index=int(row["index"]),
                start=int(row["startSamples"]),
                length=int(row["lengthSamples"]),
                event_start=int(row["eventStartSamples"]),
                event_end=int(row["eventEndSamples"]),
                duration_ms=int(row["durationMs"]),
                positive_projection=10.0 ** (float(row["positiveProjectionRmsDbfs"]) / 20.0),
                miss_rms=10.0 ** (float(row["missRmsDbfs"]) / 20.0),
                teacher_rms=10.0 ** (float(row["teacherRemovedRmsDbfs"]) / 20.0),
                stage1_score=float(row.get("stage1Score", 0.0)),
            )
            for row in metadata["selectedRecords"]
        ]
        return records, metadata
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def save_song_cache(
    *,
    root: Path,
    song: pilot.SongBundle,
    records: Sequence[EventRecord],
    h50_residual_segments: dict[int, np.ndarray],
    teacher_segments: dict[int, np.ndarray],
    contract: ShortWindowContract,
    contract_payload: dict[str, Any],
    selection_details: dict[str, Any],
) -> dict[str, Any]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in records:
        input_spec = stft_centered(
            assemble_input(
                song.mixture_gt,
                record.start,
                record.length,
                contract,
                mode="continuous",
            ),
            contract,
        )[0]
        begin = record.start
        end = begin + record.length
        targets.append(
            np.ascontiguousarray(
                song.mixture_gt[begin:end] - teacher_segments[record.start],
                dtype=np.float32,
            )
        )
        anchors.append(
            np.ascontiguousarray(h50_residual_segments[record.start], dtype=np.float32)
        )
        mask = np.zeros(record.length, dtype=np.float32)
        mask[record.event_start : record.event_end] = 1.0
        masks.append(mask)
        inputs.append(np.ascontiguousarray(input_spec, dtype=np.float32))
    input_array = np.stack(inputs).astype(np.float32)
    target_array = np.stack(targets).astype(np.float32)
    anchor_array = np.stack(anchors).astype(np.float32)
    mask_array = np.stack(masks).astype(np.float32)
    if not (input_array.ndim == 4 and target_array.shape == anchor_array.shape and target_array.shape[:2] == mask_array.shape):
        raise ValueError(f"Cache shape mismatch for {song.slug}")
    output_path, metadata_path = cache_paths(root, song.slug)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    np.savez_compressed(
        temporary,
        inputSpec=input_array,
        targetAudio=target_array,
        anchorAudio=anchor_array,
        eventMask=mask_array,
        starts=np.asarray([item.start for item in records], dtype=np.int64),
        eventStarts=np.asarray([item.event_start for item in records], dtype=np.int64),
        eventEnds=np.asarray([item.event_end for item in records], dtype=np.int64),
        durationsMs=np.asarray([item.duration_ms for item in records], dtype=np.int64),
    )
    temporary_npz = temporary if temporary.suffix == ".npz" else Path(str(temporary) + ".npz")
    temporary_npz.replace(output_path)
    metadata = {
        "schema": CACHE_SCHEMA,
        "contract": contract_payload,
        "recordCount": len(records),
        "selectedRecords": [event_record_json(item, True) for item in records],
        "selectionDetails": selection_details,
        "cache": {
            "file": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
        },
        "shapes": {
            "inputSpec": list(input_array.shape),
            "targetAudio": list(target_array.shape),
            "eventMask": list(mask_array.shape),
        },
    }
    json_write(metadata_path, metadata)
    json_write(selection_path(root, song.slug), metadata)
    return metadata


def load_cache_metadata(root: Path, slug: str) -> dict[str, Any]:
    _, metadata_path = cache_paths(root, slug)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class CacheStore:
    def __init__(self, paths: dict[str, Path], max_open: int = 8) -> None:
        self.paths = paths
        self.max_open = max_open
        self.open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def load(self, slug: str) -> dict[str, np.ndarray]:
        existing = self.open.get(slug)
        if existing is not None:
            self.open.move_to_end(slug)
            return existing
        with np.load(self.paths[slug]) as values:
            arrays = {
                key: np.ascontiguousarray(values[key], dtype=np.float32)
                for key in ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
            }
        self.open[slug] = arrays
        self.open.move_to_end(slug)
        while len(self.open) > self.max_open:
            self.open.popitem(last=False)
        return arrays

    def batch(self, items: Sequence[tuple[str, int]]) -> tuple[np.ndarray, ...]:
        arrays = [self.load(slug) for slug, _ in items]
        return tuple(
            np.stack([array[key][index] for array, (_, index) in zip(arrays, items)])
            for key in ("inputSpec", "targetAudio", "anchorAudio", "eventMask")
        )

    def preload(self) -> None:
        for index, slug in enumerate(sorted(self.paths), start=1):
            self.load(slug)
            if index == 1 or index == len(self.paths) or index % 10 == 0:
                print(f"preload cache {index}/{len(self.paths)}", flush=True)


def build_schedule(
    slugs: Sequence[str], records_per_song: int, passes: int, seed: int
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for pass_index in range(passes):
        items = [
            (slug, record_index)
            for slug in sorted(slugs)
            for record_index in range(records_per_song)
        ]
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        result.extend(items[int(index)] for index in rng.permutation(len(items)))
    return result


def torch_packed_istft(packed: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    left = torch.complex(packed[:, 0], packed[:, 2])
    right = torch.complex(packed[:, 1], packed[:, 3])
    return torch.stack(
        [
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
            for spectrum in (left, right)
        ],
        dim=-1,
    )


def charbonnier_per_record(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    error = torch.sqrt((prediction - target).square() + EPSILON**2).mean(dim=-1)
    weights = weights.to(dtype=prediction.dtype)
    return (error * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def load_h50_model(
    architecture_checkpoint: Path,
    h50_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, architecture = pilot.make_model(architecture_checkpoint, device)
    payload = torch.load(h50_checkpoint, map_location="cpu", weights_only=False)
    supported_formats = {
        "local-inst3-vr-hard-sampling-checkpoint@1",
        "local-inst3-vr-continuous-topk-local-checkpoint@1",
        "local-inst3-vr-continuation-checkpoint@1",
        "local-inst3-mtg-fma-c1-checkpoint@1",
        "local-inst3-modern-s-pilot-checkpoint@1",
        "local-inst3-modern-s-combined-pilot-checkpoint@1",
        "local-inst3-modern-s-combined-continuation-checkpoint@1",
        "local-inst3-s-only-stress-checkpoint@1",
    }
    if payload.get("format") not in supported_formats:
        raise ValueError(f"Unexpected H50 checkpoint format: {h50_checkpoint}")
    model.load_state_dict(payload["stateDict"], strict=True)
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    return model, {
        "architectureCheckpoint": architecture["checkpoint"],
        "sourceH50": checkpoint_metadata(h50_checkpoint),
        "sourceStep": int(payload.get("step", -1)),
        "sourcePasses": int(payload.get("passes", -1)),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    variant: str,
    contract_id: str,
    step: int,
    args: argparse.Namespace,
    records_per_pass: int,
    history: list[dict[str, Any]],
    source: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    payload = {
        "format": "local-inst3-vr-continuous-topk-local-checkpoint@1",
        "status": status,
        "variant": variant,
        "runContractId": contract_id,
        "step": step,
        "passes": args.passes,
        "recordsPerPass": records_per_pass,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "topK": args.top_k,
        "seed": args.seed,
        "sourceH50": source,
        "stateDict": hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
        "history": history,
    }
    return hard.atomic_torch_save(path, payload)


def train_variant(
    *,
    variant: str,
    cache_paths_map: dict[str, Path],
    schedule: list[tuple[str, int]],
    args: argparse.Namespace,
    device: torch.device,
    contract_id: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    records_per_pass = len(schedule) // args.passes
    if records_per_pass % args.batch_size:
        raise ValueError("records per pass must be divisible by batch size")
    updates_per_pass = records_per_pass // args.batch_size
    total_updates = len(schedule) // args.batch_size
    milestone_updates = {pass_count * updates_per_pass: pass_count for pass_count in parse_int_list(args.milestones)}
    run_root = args.output_root.resolve() / "runs" / variant
    run_root.mkdir(parents=True, exist_ok=True)
    hard.set_seed(args.seed)
    model, model_source = load_h50_model(args.checkpoint.resolve(), args.h50_checkpoint.resolve(), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    current = 0
    history: list[dict[str, Any]] = []
    resume_count = 0
    if args.resume:
        found: tuple[int, Path, dict[str, Any]] | None = None
        for path in run_root.glob("step-*.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                step = int(payload.get("step", -1))
                exact_contract = payload.get("runContractId") == contract_id
                compatible = (
                    payload.get("format") == "local-inst3-vr-continuous-topk-local-checkpoint@1"
                    and payload.get("variant") == variant
                    and int(payload.get("passes", -1)) == args.passes
                    and int(payload.get("recordsPerPass", -1)) == records_per_pass
                    and int(payload.get("batchSize", -1)) == args.batch_size
                    and float(payload.get("learningRate", -1.0)) == float(args.learning_rate)
                    and float(payload.get("anchorBeta", -1.0)) == float(args.anchor_beta)
                    and int(payload.get("topK", -1)) == args.top_k
                    and int(payload.get("seed", -1)) == args.seed
                    and step <= total_updates
                )
                if not (exact_contract or compatible):
                    continue
                candidate = (step, path, payload)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                continue
            if found is None or candidate[0] > found[0]:
                found = candidate
        if found is not None:
            if found[2].get("runContractId") != contract_id:
                print(
                    json.dumps(
                        {
                            "event": "resume-compatible-contract",
                            "variant": variant,
                            "step": found[0],
                            "file": str(found[1]),
                        }
                    ),
                    flush=True,
                )
            _, path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            optimizer.load_state_dict(payload["optimizerStateDict"])
            current = int(payload["step"])
            history = list(payload.get("history", []))
            resume_count = int(payload.get("resumeCount", 0)) + 1
            print(json.dumps({"event": "resume", "variant": variant, "step": current, "file": str(path)}), flush=True)
    store = CacheStore(cache_paths_map)
    store.preload()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    checkpoint_files: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    def checkpoint(step: int, status: str) -> None:
        metadata = save_checkpoint(
            run_root / f"step-{step}.pt",
            model,
            optimizer,
            variant,
            contract_id,
            step,
            args,
            records_per_pass,
            history,
            model_source,
            status,
        )
        checkpoint_files[str(step)] = metadata

    if current == 0 and 0 in milestone_updates:
        checkpoint(0, "initial-from-h50")
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    while current < total_updates:
        items = schedule[current * args.batch_size : (current + 1) * args.batch_size]
        input_array, target_array, anchor_array, mask_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        anchor_tensor = torch.from_numpy(anchor_array).to(device)
        mask_tensor = torch.from_numpy(mask_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_full = torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = predicted_full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
        if variant == CONTINUATION:
            loss = charbonnier_per_record(
                predicted,
                target_tensor,
                torch.ones_like(mask_tensor),
            ).mean()
            event_loss = charbonnier_per_record(predicted, target_tensor, mask_tensor).mean()
            anchor_loss = charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor).mean()
            selected_count = args.batch_size
        else:
            event_losses = charbonnier_per_record(predicted, target_tensor, mask_tensor)
            selected_count = min(args.top_k, event_losses.shape[0])
            event_loss = torch.topk(event_losses, k=selected_count).values.mean()
            anchor_loss = charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor).mean()
            loss = event_loss + args.anchor_beta * anchor_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {variant} update {current + 1}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current += 1
        history.append({
            "update": current,
            "pass": current / updates_per_pass,
            "loss": float(loss.detach().cpu()),
            "eventLoss": float(event_loss.detach().cpu()),
            "anchorLoss": float(anchor_loss.detach().cpu()),
            "gradientNormBeforeClip": gradient_norm,
            "topK": selected_count,
        })
        if current == 1 or current % 100 == 0:
            print(json.dumps({"event": "progress", "variant": variant, "update": current, "totalUpdates": total_updates, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        if current in milestone_updates and current != 0:
            checkpoint(current, "milestone")
            print(json.dumps({"event": "milestone", "variant": variant, "pass": milestone_updates[current], "update": current}, sort_keys=True), flush=True)
        elif args.state_interval > 0 and current % args.state_interval == 0:
            checkpoint(current, "rolling")
    if str(total_updates) not in checkpoint_files:
        checkpoint(total_updates, "completed")
    result = {
        "variant": variant,
        "status": "completed",
        "passes": args.passes,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": updates_per_pass,
        "updates": total_updates,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "topK": args.top_k,
        "seed": args.seed,
        "sourceH50": model_source,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
        "resumeCount": resume_count,
        "history": history,
    }
    del model, optimizer, store, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_smoke(
    cache_paths_map: dict[str, Path],
    schedule: list[tuple[str, int]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.smoke_updates <= 0:
        raise ValueError("smoke-updates must be positive")
    store = CacheStore(cache_paths_map, max_open=len(cache_paths_map))
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    result: dict[str, Any] = {"variants": {}}
    for variant in VARIANTS:
        model, _ = load_h50_model(args.checkpoint.resolve(), args.h50_checkpoint.resolve(), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        losses: list[float] = []
        for update in range(args.smoke_updates):
            items = [schedule[update % len(schedule)]]
            input_array, target_array, anchor_array, mask_array = store.batch(items)
            input_tensor = torch.from_numpy(input_array).to(device)
            target_tensor = torch.from_numpy(target_array).to(device)
            anchor_tensor = torch.from_numpy(anchor_array).to(device)
            mask_tensor = torch.from_numpy(mask_array).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = torch_packed_istft(model(input_tensor), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            predicted = predicted[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
            if variant == CONTINUATION:
                loss = charbonnier_per_record(predicted, target_tensor, torch.ones_like(mask_tensor)).mean()
            else:
                event_losses = charbonnier_per_record(predicted, target_tensor, mask_tensor)
                loss = torch.topk(event_losses, k=1).values.mean() + args.anchor_beta * charbonnier_per_record(predicted, anchor_tensor, 1.0 - mask_tensor).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Smoke non-finite loss: {variant}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            if not torch.isfinite(gradient):
                raise FloatingPointError(f"Smoke non-finite gradient: {variant}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        result["variants"][variant] = {"updates": args.smoke_updates, "lossFirst": losses[0], "lossLast": losses[-1]}
        del model, optimizer
    del store, window
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def prepare_training_data(
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    contract: ShortWindowContract,
    device: torch.device,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    event_manifest_sha = file_sha256(args.event_root.resolve() / "reports" / "inst3-vr-hard-events-report.json")
    runner_sha = file_sha256(Path(__file__).resolve())
    contract_base = {
        "schema": CACHE_SCHEMA,
        "runnerSha256": runner_sha,
        "sourceH50Sha256": file_sha256(args.h50_checkpoint.resolve()),
        "studentArchitectureSha256": file_sha256(args.checkpoint.resolve()),
        "eventReportSha256": event_manifest_sha,
        "sampleRate": SAMPLE_RATE,
        "numFrames": contract.num_frames,
        "inputSamples": contract.input_samples,
        "usefulSamples": contract.useful_samples,
        "leftContextHops": contract.left_context_hops,
        "rightContextHops": contract.right_context_hops,
        "assembly": "continuous-context-overlap-save",
        "candidateDurationsMs": list(EVENT_MS),
        "excludeSongEdgeSamples": JOIN_RADIUS,
        "excludeJoinEdgeSamples": JOIN_RADIUS,
        "excludeUsefulFraction": [0.10, 0.90],
        "recordsPerSong": args.records_per_song,
    }
    paths: dict[str, Path] = {}
    selections: dict[str, Any] = {}
    prep_started = time.perf_counter()
    h50_model, h50_source = load_h50_model(args.checkpoint.resolve(), args.h50_checkpoint.resolve(), device)
    contract_info = pilot.verify_teacher_contract(
        args.contract.resolve(), args.teacher.resolve(), args.teacher_tflite.resolve()
    )
    teacher_session, providers = pilot.make_teacher_session(
        args.teacher.resolve(), args.threads, args.require_teacher_cuda
    )
    try:
        for index, entry in enumerate(entries, start=1):
            slug = oracle.slugify(entry["fileName"])
            expected = {**contract_base, "sourceSha256": entry["sourceSha256"], "member": entry["member"], "slug": slug}
            cached = None if args.force_prep else load_cached_records(output_root, slug, expected)
            if cached is not None:
                _, metadata = cached
                paths[slug] = cache_paths(output_root, slug)[0]
                selections[slug] = metadata
                print(f"reuse continuous cache {index}/{len(entries)}: {slug}", flush=True)
                continue
            song = load_train_song(args.oracle_root.resolve(), entry)
            print(f"prepare continuous train {index}/{len(entries)}: {entry['member']}", flush=True)
            candidates, counts = load_stage1_top_event_candidates(
                args.event_root.resolve(), song, contract
            )
            candidate_starts = sorted(set(item.start for item in candidates))
            h50_segments, h50_timing = render_selected_continuous_windows(
                h50_model,
                song.mixture_gt,
                candidate_starts,
                contract,
                device,
                args.inference_batch_size,
            )
            teacher_candidates = [
                hard.Candidate(
                    index=index,
                    start=start,
                    length=contract.useful_samples,
                    hard_score=0.0,
                    rank=index + 1,
                )
                for index, start in enumerate(candidate_starts)
            ]
            teacher_selection = hard.SongSelection(
                slug=song.slug,
                member=song.member,
                candidates=tuple(teacher_candidates),
                hard_pool_indices=tuple(range(len(teacher_candidates))),
                uniform_indices=tuple(range(len(teacher_candidates))),
                hard_fraction=1.0,
                hard_indices=tuple(range(len(teacher_candidates))),
                union_indices=tuple(range(len(teacher_candidates))),
            )
            teacher_segment_list, teacher_timing = hard.render_selected_teacher_segments(
                song.mixture_gt,
                teacher_selection,
                teacher_session,
                progress_label=f"prepare/{slug}/Inst3",
            )
            teacher_segments = {
                start: segment
                for start, segment in zip(candidate_starts, teacher_segment_list, strict=True)
            }
            scored_candidates = score_continuous_event_records(
                song,
                candidates,
                h50_segments,
                teacher_segments,
            )
            selected, selected_details = select_top_records(scored_candidates, args.records_per_song)
            cache_metadata = save_song_cache(
                root=output_root,
                song=song,
                records=selected,
                h50_residual_segments=h50_segments,
                teacher_segments=teacher_segments,
                contract=contract,
                contract_payload=expected,
                selection_details={
                    **counts,
                    **selected_details,
                    "h50Render": h50_timing,
                    "teacherRender": teacher_timing,
                    "teacherProviders": list(providers),
                    "teacherContractId": contract_info["contractId"],
                    "candidateWindowCount": len(candidate_starts),
                },
            )
            cache_metadata["candidatePool"] = [event_record_json(item) for item in sorted(scored_candidates, key=lambda x: (-x.positive_projection, -x.miss_rms))[: min(32, len(scored_candidates))]]
            _, metadata_path = cache_paths(output_root, slug)
            json_write(metadata_path, cache_metadata)
            paths[slug] = cache_paths(output_root, slug)[0]
            selections[slug] = cache_metadata
            del song, h50_segments, teacher_segments, teacher_segment_list, candidates, scored_candidates, selected
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del teacher_session, h50_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection_payload = {
        "schema": "local-inst3-vr-continuous-topk-local-selection@1",
        "contract": contract_base,
        "songs": selections,
        "selectionSha256": canonical_sha256(selections),
        "elapsedSeconds": time.perf_counter() - prep_started,
    }
    json_write(output_root / "selection.json", selection_payload)
    return paths, selection_payload, {"providers": list(providers), "teacherContract": contract_info}


def summarize_training_evaluation(
    trained_per_song: dict[str, Any],
    baseline_report: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    baseline_songs = baseline_report["musdbEvaluation"]["songs"]
    metric_names = (
        "instrumentalSdrDb",
        "accompanimentErrorRmsDbfs",
        "positiveVocalProjectionDbfs",
        "teacherResidualMissRmsDbfs",
        "teacherRemovedPositiveProjectionDbfs",
    )
    region_names = continuous.REGIONS
    whole = continuous.RegionAccumulator()
    raw_region: dict[str, dict[str, list[float]]] = {
        region: {metric: [] for metric in metric_names} for region in region_names
    }
    per_song_delta: dict[str, Any] = {}
    for slug, value in trained_per_song.items():
        baseline = baseline_songs[slug]["candidates"]["H50-continuous"]
        for region in region_names:
            raw_region[region]["instrumentalSdrDb"].append(value["regions"][region]["instrumentalSdrDb"])
            raw_region[region]["accompanimentErrorRmsDbfs"].append(value["regions"][region]["accompanimentErrorRmsDbfs"])
            raw_region[region]["positiveVocalProjectionDbfs"].append(value["regions"][region]["positiveVocalProjectionDbfs"])
            raw_region[region]["teacherResidualMissRmsDbfs"].append(value["regions"][region]["teacherResidualMissRmsDbfs"])
            raw_region[region]["teacherRemovedPositiveProjectionDbfs"].append(value["regions"][region]["teacherRemovedPositiveProjectionDbfs"])
        per_song_delta[slug] = {
            "regions": {
                region: {
                    metric: float(value["regions"][region][metric] - baseline["regions"][region][metric])
                    for metric in metric_names
                }
                for region in region_names
            },
            "events": {},
        }
        for ms in continuous.EVENT_MS:
            per_song_delta[slug]["events"][str(ms)] = {}
            for region in ("all", *region_names):
                current_event = value["events"][str(ms)][region]
                baseline_event = baseline["events"][str(ms)][region]
                per_song_delta[slug]["events"][str(ms)][region] = {
                    key: (
                        None
                        if key not in current_event or key not in baseline_event
                        else float(current_event[key] - baseline_event[key])
                    )
                    for key in (
                        "missRmsP95Dbfs",
                        "missRmsMaxDbfs",
                        "positiveProjectionP95Dbfs",
                        "positiveProjectionMaxDbfs",
                    )
                }
    region_summary = {
        region: {
            "mean": {metric: float(np.mean(values)) if values else None for metric, values in metrics.items()},
            "median": {metric: float(np.median(values)) if values else None for metric, values in metrics.items()},
        }
        for region, metrics in raw_region.items()
    }
    event_directions: dict[str, Any] = {}
    for ms in continuous.EVENT_MS:
        event_directions[str(ms)] = {}
        for metric in (
            "positiveProjectionP95Dbfs",
            "positiveProjectionMaxDbfs",
            "missRmsP95Dbfs",
            "missRmsMaxDbfs",
        ):
            values = [
                per_song_delta[slug]["events"][str(ms)]["all"][metric]
                for slug in per_song_delta
                if per_song_delta[slug]["events"][str(ms)]["all"].get(metric) is not None
            ]
            event_directions[str(ms)][metric] = {
                "songCount": len(values),
                "improvedSongs": int(sum(value < 0.0 for value in values)),
                "meanDeltaDb": float(np.mean(values)) if values else None,
                "medianDeltaDb": float(np.median(values)) if values else None,
                "worstDeltaDb": float(np.max(values)) if values else None,
            }
    # The whole-song summary is stored by the evaluator and can be compared
    # using the per-song values without retaining PCM arrays.
    whole_metrics = {
        metric: float(np.mean([value["wholeSong"][metric] for value in trained_per_song.values()]))
        for metric in metric_names
    }
    baseline_whole = {
        metric: float(
            np.mean(
                [
                    value["candidates"]["H50-continuous"]["wholeSong"][metric]
                    for value in baseline_songs.values()
                ]
            )
        )
        for metric in metric_names
    }
    delta_whole = {
        metric: float(whole_metrics[metric] - baseline_whole[metric])
        for metric in metric_names
    }
    return {
        "variant": variant,
        "songCount": len(trained_per_song),
        "perSong": trained_per_song,
        "perSongDeltaVsH50": per_song_delta,
        "aggregatePerSongMean": {"wholeSong": whole_metrics, "regions": region_summary},
        "deltaVsH50": {"wholeSong": delta_whole},
        "eventDirectionsVsH50": event_directions,
    }


def evaluate_trained_models(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    checkpoint_paths: dict[str, Path],
    eval_entries: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    baseline_report = json.loads(args.baseline_report.resolve().read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for variant, state_path in checkpoint_paths.items():
        model, source = load_h50_model(args.checkpoint.resolve(), state_path, device)
        per_song: dict[str, Any] = {}
        started = time.perf_counter()
        cache_root = args.output_root.resolve() / "evaluation-cache" / variant
        checkpoint_sha = file_sha256(state_path)
        try:
            for index, entry in enumerate(eval_entries, start=1):
                slug = oracle.slugify(entry["fileName"])
                cached_path = cache_root / f"{slug}.json"
                if cached_path.is_file():
                    try:
                        cached = json.loads(cached_path.read_text(encoding="utf-8"))
                        if (
                            cached.get("checkpointSha256") == checkpoint_sha
                            and cached.get("variant") == variant
                            and cached.get("sourceSha256") == entry["sourceSha256"]
                        ):
                            per_song[slug] = cached["evaluation"]
                            print(f"reuse trained eval {variant} {index}/{len(eval_entries)}: {slug}", flush=True)
                            continue
                    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
                        pass
                song = hard.load_eval_song(args.oracle_root.resolve(), entry)
                print(f"continuous trained eval {variant} {index}/{len(eval_entries)}: {entry['member']}", flush=True)
                residual, timing = continuous.render_student(
                    model,
                    song.mixture_gt,
                    contract,
                    device,
                    args.inference_batch_size,
                )
                candidate = np.ascontiguousarray(song.mixture_gt - residual, dtype=np.float32)
                evaluated = continuous.evaluate_candidate(
                    candidate,
                    song.mixture_gt,
                    song.vocals,
                    song.instrumental,
                    song.teacher_instrumental,
                    contract,
                )
                evaluated["render"] = timing
                evaluated["role"] = song.role
                evaluated["member"] = song.member
                per_song[song.slug] = evaluated
                json_write(
                    cached_path,
                    {
                        "variant": variant,
                        "checkpointSha256": checkpoint_sha,
                        "sourceSha256": entry["sourceSha256"],
                        "evaluation": evaluated,
                    },
                )
                del song, residual, candidate
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[variant] = summarize_training_evaluation(per_song, baseline_report, variant)
        results[variant]["checkpoint"] = checkpoint_metadata(state_path)
        results[variant]["source"] = source
        results[variant]["elapsedSeconds"] = time.perf_counter() - started
    return results


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": int(audio.shape[1]),
    }


def render_listening(
    args: argparse.Namespace,
    contract: ShortWindowContract,
    checkpoint_paths: dict[str, Path],
    device: torch.device,
) -> dict[str, Any]:
    songs = density.validate_private_songs(args.samples_root.resolve())
    output_root = args.output_root.resolve() / "listening-12"
    report: dict[str, Any] = {
        "songCount": len(songs),
        "variants": list(checkpoint_paths),
        "h50BaselineRoot": str(
            (ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation" / "private-listening-12" / "H50-continuous").resolve()
        ),
        "songs": {},
    }
    for variant, state_path in checkpoint_paths.items():
        model, _ = load_h50_model(args.checkpoint.resolve(), state_path, device)
        try:
            for index, song_info in enumerate(songs, start=1):
                source, sample_rate = listening.load_audio(Path(song_info["file"]))
                if sample_rate != SAMPLE_RATE:
                    raise ValueError(f"Unexpected sample rate for {song_info['name']}")
                print(f"listening {variant} {index}/{len(songs)}: {song_info['name']}", flush=True)
                residual, timing = continuous.render_student(
                    model,
                    source,
                    contract,
                    device,
                    args.inference_batch_size,
                )
                instrumental = np.ascontiguousarray(source - residual, dtype=np.float32)
                song_report = report["songs"].setdefault(song_info["name"], {"source": song_info, "variants": {}})
                song_report["variants"][variant] = {
                    "instrumental": write_flac(output_root / variant / f"{song_info['name']}-instrumental.flac", instrumental),
                    "residual": write_flac(output_root / variant / f"{song_info['name']}-residual.flac", residual),
                    "timing": timing,
                }
                json_write(args.output_root.resolve() / "listening-report.json", report)
                del source, residual, instrumental
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report["outputCount"] = sum(
        len(song["variants"]) * 2 for song in report["songs"].values()
    )
    json_write(args.output_root.resolve() / "listening-report.json", report)
    return report


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.records_per_song <= 0 or args.passes <= 0 or args.batch_size <= 0:
        raise ValueError("records, passes, and batch size must be positive")
    if args.top_k <= 0 or args.top_k > args.batch_size:
        raise ValueError("top-k must be within batch size")
    if args.learning_rate <= 0 or args.anchor_beta < 0 or args.threads <= 0:
        raise ValueError("learning-rate, anchor-beta, and threads must be valid")
    if args.state_interval <= 0 or args.inference_batch_size <= 0:
        raise ValueError("state interval and inference batch size must be positive")
    if milestones[0] != 0 or milestones[-1] != args.passes or tuple(sorted(set(milestones))) != milestones:
        raise ValueError("milestones must be sorted, start at 0, and end at passes")
    if args.max_train_songs is not None and args.max_train_songs <= 0:
        raise ValueError("max-train-songs must be positive")
    if args.max_eval_songs is not None and args.max_eval_songs <= 0:
        raise ValueError("max-eval-songs must be positive")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_int_list(args.milestones)
    validate_args(args, milestones)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.device_object = device
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for path in (args.manifest, args.checkpoint, args.h50_checkpoint, args.teacher, args.baseline_report):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    manifest = hard.load_manifest(args.manifest.resolve())
    train_entries = hard.load_train_entries(manifest, args.max_train_songs)
    eval_entries = sorted(
        [entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}],
        key=lambda item: (item["role"], item["member"]),
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    cache_paths_map, selection_payload, prep_metadata = prepare_training_data(
        args, train_entries, contract, device
    )
    schedule = build_schedule(
        list(cache_paths_map), args.records_per_song, args.passes, args.seed
    )
    schedule_payload = [{"slug": slug, "cacheIndex": index} for slug, index in schedule]
    schedule_summary = {
        "recordCount": len(schedule),
        "recordsPerPass": len(schedule) // args.passes,
        "updates": len(schedule) // args.batch_size,
        "scheduleSha256": canonical_sha256(schedule_payload),
    }
    if args.smoke_only:
        smoke = run_smoke(cache_paths_map, schedule, args, device)
        report = {
            "schema": SCHEMA,
            "status": "smoke-completed",
            "contract": selection_payload["contract"],
            "schedule": schedule_summary,
            "smoke": smoke,
            "environment": {"device": str(device), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        }
        report_path = args.output_root / "reports" / "continuous-topk-smoke.json"
        json_write(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path), "smoke": smoke}, indent=2), flush=True)
        return 0
    common_contract = {
        "schema": SCHEMA,
        "selectionSha256": selection_payload["selectionSha256"],
        "scheduleSha256": schedule_summary["scheduleSha256"],
        "manifestSha256": file_sha256(args.manifest.resolve()),
        "h50Checkpoint": checkpoint_metadata(args.h50_checkpoint.resolve()),
        "studentArchitectureSha256": file_sha256(args.checkpoint.resolve()),
        "trainSongCount": len(train_entries),
        "evalSongCount": len(eval_entries),
        "recordsPerSong": args.records_per_song,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "topK": args.top_k,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "assembly": "continuous-context-overlap-save",
        "eventSelection": "actual H50 continuous miss over all Stage 1 50/100 ms blocks, central useful region only",
        "officialFinalTestUsed": False,
        "teacherContract": prep_metadata["teacherContract"],
    }
    training: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_contract = {**common_contract, "variant": variant}
        training[variant] = train_variant(
            variant=variant,
            cache_paths_map=cache_paths_map,
            schedule=schedule,
            args=args,
            device=device,
            contract_id=canonical_sha256(variant_contract),
            source={"h50": checkpoint_metadata(args.h50_checkpoint.resolve()), "variantContract": variant_contract},
        )
        training[variant]["contract"] = {"id": canonical_sha256(variant_contract), "payload": variant_contract}
    checkpoint_paths = {
        variant: Path(
            training[variant]["checkpointFiles"][str(args.passes * training[variant]["updatesPerPass"])]
            ["file"]
        )
        for variant in VARIANTS
    }
    evaluation = None
    if not args.skip_continuous_eval:
        evaluation = evaluate_trained_models(args, contract, checkpoint_paths, eval_entries, device)
    listening_report = None
    if not args.skip_listening:
        listening_paths = {
            "H50-pass-50": args.h50_checkpoint.resolve(),
            **checkpoint_paths,
        }
        listening_report = render_listening(args, contract, listening_paths, device)
    else:
        existing_listening = args.output_root / "listening-report.json"
        if existing_listening.is_file():
            listening_report = json.loads(existing_listening.read_text(encoding="utf-8"))
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "teacher weight and derived outputs remain local; not published",
        },
        "contract": common_contract,
        "selection": selection_payload,
        "schedule": schedule_summary,
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
            "continuation": "Full continuous useful-span audio target V_T on the same selected event windows.",
            "topkLocal": "Top-k event Charbonnier losses toward V_T plus H50 output anchor outside the event.",
            "boundaryExclusion": "Song edges, internal join neighborhoods, and useful-span edges are excluded from the training event pool; they remain evaluation regions only.",
            "publication": "Do not publish checkpoints, teacher-derived audio, or MUSDB18-derived caches.",
        },
    }
    report_path = args.output_root / "reports" / "continuous-topk-local-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "training": list(training), "evaluation": list(evaluation or {}), "listening": 0 if listening_report is None else listening_report.get("outputCount", 0)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
