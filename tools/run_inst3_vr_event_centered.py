#!/usr/bin/env python3
"""Train a residual-vocals student on H50 miss-centered local targets.

This is a local, non-commercial MUSDB18 experiment.  It first scans the
fixed Stage 1 candidate windows with the completed V-R-H50 checkpoint and
Inst 3 teacher.  The worst positive 50/100 ms miss blocks are selected per
song, then each event receives a fresh TFC-TDF window centered on the block.

Two arms use the exact same event-centered records:

* ``event-centered-continuation`` uses the Inst 3 residual target everywhere;
* ``event-centered-local-anchor`` uses that target on event frames and the
  frozen H50 output on all other frames.

All decoded audio, teacher outputs, caches, checkpoints, and listening files
stay under ``data/`` and are not publication artifacts.
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
import run_inst3_vr_local_anchor as local
import run_inst3_vr_miss_driven as miss


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-event-centered"
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_EVAL_ROOT / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ARCHIVE = pilot.DEFAULT_ARCHIVE
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

SCHEMA = "local-inst3-vr-event-centered@1"
CACHE_SCHEMA = "local-inst3-vr-event-centered-cache@1"
CONTINUATION = "event-centered-continuation"
LOCAL_ANCHOR = "event-centered-local-anchor"
VARIANTS = (CONTINUATION, LOCAL_ANCHOR)
EVENT_MILLISECONDS = (50, 100)
MILESTONES = (0, 1, 2, 5)


@dataclass(frozen=True)
class EventRecord:
    candidate_index: int
    candidate_start: int
    block_start: int
    block_end: int
    duration_ms: int
    positive_projection: float
    miss_rms: float
    teacher_rms: float

    @property
    def center(self) -> int:
        return (self.block_start + self.block_end) // 2


@dataclass(frozen=True)
class CenteredRecord:
    index: int
    start: int
    length: int
    event_start: int
    event_end: int
    event_center: int
    duration_ms: int
    score: float
    miss_rms: float


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integers, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--events-per-song", type=int, default=4)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--milestones", default="0,1,2,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--anchor-beta", type=float, default=1.0)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--skip-listening", action="store_true")
    parser.add_argument("--force-listening", action="store_true")
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


def load_h50_model(
    architecture_checkpoint: Path,
    h50_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, initialization = pilot.make_model(architecture_checkpoint, device)
    payload = torch.load(h50_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("stateDict")
    if not isinstance(state, dict):
        raise ValueError(f"Missing H50 stateDict: {h50_checkpoint}")
    model.load_state_dict(state, strict=True)
    sweep.freeze_batchnorm_running_statistics(model)
    model.eval()
    return model, {
        "checkpoint": checkpoint_metadata(h50_checkpoint),
        "step": int(payload.get("step", -1)),
        "passes": int(payload.get("passes", -1)),
        "initialization": initialization,
    }


def load_candidates(event_root: Path, entry: dict[str, Any]) -> tuple[str, list[hard.Candidate]]:
    slug = hard.oracle.slugify(entry["fileName"])
    report_path = event_root / "songs" / f"{slug}.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("song", {}).get("member") != entry["member"]:
        raise ValueError(f"Event report member mismatch for {slug}")
    if report.get("studentWindows", {}).get("usefulSamples") != pilot.DEFAULT_CONFIG.useful_samples:
        raise ValueError(f"Unexpected useful sample count for {slug}")
    candidates: list[hard.Candidate] = []
    for index, row in enumerate(
        sorted(report["studentWindows"]["ranked"], key=lambda item: item["startSamples"])
    ):
        start = int(row["startSamples"])
        end = int(row["endSamples"])
        if end <= start or end - start < pilot.DEFAULT_CONFIG.useful_samples // 3:
            continue
        candidates.append(
            hard.Candidate(
                index=index,
                start=start,
                length=end - start,
                hard_score=float(row.get("hardEventScore", 0.0)),
                rank=int(row.get("rank", 0)),
            )
        )
    if not candidates:
        raise ValueError(f"No candidates in {slug}")
    return slug, candidates


def all_candidate_selection(
    slug: str, member: str, candidates: Sequence[hard.Candidate]
) -> hard.SongSelection:
    indices = tuple(range(len(candidates)))
    return hard.SongSelection(
        slug=slug,
        member=member,
        candidates=tuple(candidates),
        hard_pool_indices=indices,
        uniform_indices=indices,
        hard_fraction=1.0,
        hard_indices=indices,
        union_indices=indices,
    )


def make_block_event_rows(
    *,
    candidate: hard.Candidate,
    mixture: np.ndarray,
    teacher_segment: np.ndarray,
    predicted_residual: np.ndarray,
    sample_rate: int,
) -> list[EventRecord]:
    predicted_instrumental = np.ascontiguousarray(
        mixture - predicted_residual, dtype=np.float32
    )
    miss = np.ascontiguousarray(predicted_instrumental - teacher_segment, dtype=np.float32)
    teacher_removed = np.ascontiguousarray(mixture - teacher_segment, dtype=np.float32)
    rows: list[EventRecord] = []
    for milliseconds in EVENT_MILLISECONDS:
        values = hard.block_projection_arrays(
            teacher_removed, miss, sample_rate, milliseconds
        )
        block_samples = int(values["blockSamples"][0])
        for block_index, active in enumerate(values["active"]):
            if not bool(active):
                continue
            projection = float(values["positiveProjectionRms"][block_index])
            miss_rms = float(values["missRms"][block_index])
            teacher_rms = float(values["teacherRms"][block_index])
            if not all(math.isfinite(value) for value in (projection, miss_rms, teacher_rms)):
                continue
            rows.append(
                EventRecord(
                    candidate_index=candidate.index,
                    candidate_start=candidate.start,
                    block_start=candidate.start + block_index * block_samples,
                    block_end=min(
                        candidate.start + candidate.length,
                        candidate.start + (block_index + 1) * block_samples,
                    ),
                    duration_ms=milliseconds,
                    positive_projection=projection,
                    miss_rms=miss_rms,
                    teacher_rms=teacher_rms,
                )
            )
    return rows


def rank_events(rows: Sequence[EventRecord], limit: int) -> list[EventRecord]:
    if limit <= 0:
        raise ValueError("events-per-song must be positive")
    ranked = sorted(
        rows,
        key=lambda item: (
            -item.positive_projection,
            -item.miss_rms,
            item.block_start,
            item.duration_ms,
        ),
    )
    chosen: list[EventRecord] = []
    for row in ranked:
        if any(row.block_start < other.block_end and other.block_start < row.block_end for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) == limit:
            return chosen
    for row in ranked:
        if row not in chosen:
            chosen.append(row)
            if len(chosen) == limit:
                break
    return chosen


def centered_records(
    events: Sequence[EventRecord], song_samples: int, useful_samples: int
) -> list[CenteredRecord]:
    if song_samples <= 0 or useful_samples <= 0:
        raise ValueError("song and useful sample counts must be positive")
    length = min(song_samples, useful_samples)
    maximum = max(0, song_samples - length)
    records: list[CenteredRecord] = []
    for index, event in enumerate(events):
        start = int(round(event.center - length / 2.0))
        start = min(max(start, 0), maximum)
        if not (start <= event.block_start < event.block_end <= start + length):
            raise AssertionError("Centered context does not contain event")
        records.append(
            CenteredRecord(
                index=index,
                start=start,
                length=length,
                event_start=event.block_start,
                event_end=event.block_end,
                event_center=event.center,
                duration_ms=event.duration_ms,
                score=event.positive_projection,
                miss_rms=event.miss_rms,
            )
        )
    return records


def event_frame_mask(
    *,
    candidate_start: int,
    candidate_length: int,
    event_start: int,
    event_end: int,
) -> np.ndarray:
    frame_centers = np.arange(pilot.DEFAULT_CONFIG.num_frames, dtype=np.int64) * pilot.DEFAULT_CONFIG.hop_length
    half_fft = pilot.DEFAULT_CONFIG.n_fft // 2
    overlap_start = max(candidate_start, event_start)
    overlap_end = min(candidate_start + candidate_length, event_end)
    if overlap_end <= overlap_start:
        raise ValueError("Event is outside centered candidate")
    local_start = overlap_start - candidate_start + pilot.DEFAULT_CONFIG.trim_samples
    local_end = overlap_end - candidate_start + pilot.DEFAULT_CONFIG.trim_samples
    return np.ascontiguousarray(
        (frame_centers + half_fft > local_start)
        & (frame_centers - half_fft < local_end),
        dtype=bool,
    )


def scan_song(
    *,
    song: Any,
    candidates: Sequence[hard.Candidate],
    model: torch.nn.Module,
    teacher_session: Any,
    device: torch.device,
) -> tuple[list[EventRecord], dict[str, Any]]:
    selection = all_candidate_selection(song.slug, song.entry["member"], candidates)
    teacher_segments, teacher_timing = hard.render_selected_teacher_segments(
        song.mixture_gt,
        selection,
        teacher_session,
        progress_label=f"scan/{song.slug}",
    )
    rows: list[EventRecord] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(candidates), 4):
            batch = candidates[begin : begin + 4]
            inputs = np.stack(
                [pilot.student_window_spec(song.mixture_gt, item.start, item.length) for item in batch]
            )
            predicted_specs = model(torch.from_numpy(inputs).to(device)).detach().cpu().numpy()
            for offset, candidate in enumerate(batch):
                reconstructed = pilot.student_istft_centered(predicted_specs[offset : offset + 1])
                trim = pilot.DEFAULT_CONFIG.trim_samples
                predicted_residual = np.ascontiguousarray(
                    reconstructed[trim : trim + candidate.length], dtype=np.float32
                )
                rows.extend(
                    make_block_event_rows(
                        candidate=candidate,
                        mixture=song.mixture_gt[candidate.start : candidate.start + candidate.length],
                        teacher_segment=teacher_segments[begin + offset],
                        predicted_residual=predicted_residual,
                        sample_rate=song.sample_rate,
                    )
                )
    return rows, {
        "candidateCount": len(candidates),
        "teacher": teacher_timing,
        "scanSeconds": time.perf_counter() - started,
    }


def event_json(record: EventRecord) -> dict[str, Any]:
    return {
        "candidateIndex": record.candidate_index,
        "candidateStartSamples": record.candidate_start,
        "blockStartSamples": record.block_start,
        "blockEndSamples": record.block_end,
        "blockLengthSamples": record.block_end - record.block_start,
        "eventCenterSamples": record.center,
        "durationMs": record.duration_ms,
        "positiveProjectionRms": record.positive_projection,
        "positiveProjectionRmsDbfs": hard.dbfs(record.positive_projection),
        "missRms": record.miss_rms,
        "missRmsDbfs": hard.dbfs(record.miss_rms),
        "teacherRms": record.teacher_rms,
        "teacherRmsDbfs": hard.dbfs(record.teacher_rms),
    }


def centered_json(record: CenteredRecord) -> dict[str, Any]:
    return {
        "index": record.index,
        "startSamples": record.start,
        "lengthSamples": record.length,
        "eventStartSamples": record.event_start,
        "eventEndSamples": record.event_end,
        "eventCenterSamples": record.event_center,
        "durationMs": record.duration_ms,
        "positiveProjectionRms": record.score,
        "positiveProjectionRmsDbfs": hard.dbfs(record.score),
        "missRms": record.miss_rms,
        "missRmsDbfs": hard.dbfs(record.miss_rms),
    }


def parse_event_record(row: dict[str, Any]) -> EventRecord:
    """Read either a scan-event row or an older centered-cache row."""
    block_start = int(row.get("blockStartSamples", row.get("eventStartSamples", 0)))
    block_end = int(row.get("blockEndSamples", row.get("eventEndSamples", 0)))
    candidate_start = int(row.get("candidateStartSamples", row.get("startSamples", block_start)))
    if block_end <= block_start:
        raise ValueError("Event interval must have positive length")
    return EventRecord(
        candidate_index=int(row.get("candidateIndex", row.get("index", 0))),
        candidate_start=candidate_start,
        block_start=block_start,
        block_end=block_end,
        duration_ms=int(row["durationMs"]),
        positive_projection=float(row.get("positiveProjectionRms", row.get("score", 0.0))),
        miss_rms=float(row.get("missRms", 0.0)),
        teacher_rms=float(row.get("teacherRms", 0.0)),
    )


def load_existing_events(
    *,
    root: Path,
    slug: str,
    h50_checkpoint_sha256: str,
    expected_count: int = 4,
) -> list[EventRecord] | None:
    """Resume from scan or cache metadata without repeating H50 inference."""
    scan_path, selection_path = scan_paths(root, slug)
    _cache_path, cache_metadata_path = cache_paths(root, slug)
    candidates = (
        (selection_path, "events", True),
        (scan_path, "selected", False),
        (cache_metadata_path, "events", False),
    )
    for path, key, requires_contract in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            contract = payload.get("contract", {})
            contract_hash = contract.get("h50CheckpointSha256")
            if contract_hash is not None and contract_hash != h50_checkpoint_sha256:
                continue
            if requires_contract and contract_hash is None:
                continue
            rows = payload.get(key)
            if not isinstance(rows, list) or len(rows) != expected_count:
                continue
            return [parse_event_record(row) for row in rows]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def scan_paths(root: Path, slug: str) -> tuple[Path, Path]:
    return root / "scan" / "songs" / f"{slug}.json", root / "selection" / f"{slug}.json"


def cache_paths(root: Path, slug: str) -> tuple[Path, Path]:
    return root / "cache" / "train" / f"{slug}.npz", root / "cache" / "train" / f"{slug}.json"


def load_existing_cache(
    *,
    root: Path,
    entry: dict[str, Any],
    slug: str,
    events: Sequence[EventRecord],
    checkpoint: Path,
    h50_checkpoint: Path,
    teacher: Path,
) -> tuple[Path, dict[str, Any], list[CenteredRecord]] | None:
    """Reuse a complete centered cache without decoding the source song."""
    output, metadata_path = cache_paths(root, slug)
    if not output.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contract = metadata["contract"]
        expected = {
            "schema": CACHE_SCHEMA,
            "sourceSha256": entry["sourceSha256"],
            "member": entry["member"],
            "slug": slug,
            "h50CheckpointSha256": file_sha256(h50_checkpoint),
            "studentArchitectureSha256": file_sha256(checkpoint),
            "teacherSha256": file_sha256(teacher),
            "teacherContractId": "uvr_mdxnet_inst_3@2",
            "sampleRate": pilot.DEFAULT_CONFIG.sample_rate,
            "numFrames": pilot.DEFAULT_CONFIG.num_frames,
            "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
            "eventCount": len(events),
            "eventStarts": [item.block_start for item in events],
            "eventEnds": [item.block_end for item in events],
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                return None
        with np.load(output) as values:
            required = {
                "inputSpec",
                "targetResidualSpec",
                "anchorResidualSpec",
                "eventFrameMask",
            }
            if not required.issubset(values.files):
                return None
            shape = values["inputSpec"].shape
            if (
                shape != values["targetResidualSpec"].shape
                or shape != values["anchorResidualSpec"].shape
                or shape[0] != len(events)
            ):
                return None
        rows = metadata.get("events")
        if not isinstance(rows, list) or len(rows) != len(events):
            return None
        centered: list[CenteredRecord] = []
        for index, row in enumerate(rows):
            centered.append(
                CenteredRecord(
                    index=int(row.get("index", index)),
                    start=int(row["startSamples"]),
                    length=int(row["lengthSamples"]),
                    event_start=int(row["eventStartSamples"]),
                    event_end=int(row["eventEndSamples"]),
                    event_center=int(row["eventCenterSamples"]),
                    duration_ms=int(row["durationMs"]),
                    score=float(row["positiveProjectionRms"]),
                    miss_rms=float(row["missRms"]),
                )
            )
        return output, metadata, centered
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prepare_song_cache(
    *,
    root: Path,
    song: Any,
    entry: dict[str, Any],
    records: Sequence[CenteredRecord],
    teacher_session: Any,
    h50_model: torch.nn.Module,
    device: torch.device,
    checkpoint: Path,
    h50_checkpoint: Path,
    teacher: Path,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    output, metadata_path = cache_paths(root, song.slug)
    expected = {
        "schema": CACHE_SCHEMA,
        "sourceSha256": entry["sourceSha256"],
        "member": entry["member"],
        "slug": song.slug,
        "h50CheckpointSha256": file_sha256(h50_checkpoint),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "teacherSha256": file_sha256(teacher),
        "teacherContractId": "uvr_mdxnet_inst_3@2",
        "sampleRate": pilot.DEFAULT_CONFIG.sample_rate,
        "numFrames": pilot.DEFAULT_CONFIG.num_frames,
        "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
        "eventCount": len(records),
        "starts": [item.start for item in records],
        "lengths": [item.length for item in records],
        "eventStarts": [item.event_start for item in records],
        "eventEnds": [item.event_end for item in records],
    }
    if not force and output.is_file() and metadata_path.is_file():
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(output) as values:
                valid = (
                    actual.get("contract") == expected
                    and values["inputSpec"].shape == values["targetResidualSpec"].shape
                    and values["inputSpec"].shape == values["anchorResidualSpec"].shape
                    and values["inputSpec"].shape[0] == len(records)
                )
            if valid:
                return output, actual
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    centered_candidates = [
        hard.Candidate(index=i, start=item.start, length=item.length, hard_score=item.score, rank=i + 1)
        for i, item in enumerate(records)
    ]
    selection = all_candidate_selection(song.slug, entry["member"], centered_candidates)
    teacher_segments, teacher_timing = hard.render_selected_teacher_segments(
        song.mixture_gt,
        selection,
        teacher_session,
        progress_label=f"cache/{song.slug}",
    )
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.inference_mode():
        for index, item in enumerate(records):
            input_spec = pilot.student_window_spec(song.mixture_gt, item.start, item.length)
            anchor_spec = h50_model(torch.from_numpy(input_spec[None]).to(device)).detach().cpu().numpy()[0]
            teacher_residual = song.mixture_gt[item.start : item.start + item.length] - teacher_segments[index]
            inputs.append(input_spec)
            targets.append(hard.student_segment_spec(teacher_residual))
            anchors.append(np.ascontiguousarray(anchor_spec, dtype=np.float32))
            masks.append(
                event_frame_mask(
                    candidate_start=item.start,
                    candidate_length=item.length,
                    event_start=item.event_start,
                    event_end=item.event_end,
                )
            )
    input_array = np.ascontiguousarray(np.stack(inputs), dtype=np.float32)
    target_array = np.ascontiguousarray(np.stack(targets), dtype=np.float32)
    anchor_array = np.ascontiguousarray(np.stack(anchors), dtype=np.float32)
    mask_array = np.ascontiguousarray(np.stack(masks), dtype=bool)
    if not (input_array.shape == target_array.shape == anchor_array.shape):
        raise ValueError(f"Cache shape mismatch for {song.slug}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    np.savez_compressed(
        temporary,
        inputSpec=input_array,
        targetResidualSpec=target_array,
        anchorResidualSpec=anchor_array,
        eventFrameMask=mask_array,
        starts=np.asarray([item.start for item in records], dtype=np.int64),
        lengths=np.asarray([item.length for item in records], dtype=np.int64),
        eventStarts=np.asarray([item.event_start for item in records], dtype=np.int64),
        eventEnds=np.asarray([item.event_end for item in records], dtype=np.int64),
        eventCenters=np.asarray([item.event_center for item in records], dtype=np.int64),
    )
    temporary_npz = temporary if temporary.suffix == ".npz" else Path(str(temporary) + ".npz")
    temporary_npz.replace(output)
    metadata = {
        "contract": expected,
        "cache": checkpoint_metadata(output),
        "render": {
            "mode": "H50-miss-scan-then-event-centered-context",
            "selectedTfcWindowCount": len(records),
            "teacher": teacher_timing,
        },
        "events": [centered_json(item) for item in records],
    }
    json_write(metadata_path, metadata)
    return output, metadata


class EventCenteredCacheStore:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
        self._open: dict[str, dict[str, np.ndarray]] = {}

    def _load(self, slug: str) -> dict[str, np.ndarray]:
        if slug not in self._open:
            with np.load(self.paths[slug]) as values:
                self._open[slug] = {
                    key: np.ascontiguousarray(values[key])
                    for key in (
                        "inputSpec",
                        "targetResidualSpec",
                        "anchorResidualSpec",
                        "eventFrameMask",
                    )
                }
        return self._open[slug]

    def batch(self, items: Sequence[hard.ScheduleItem]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arrays = [self._load(item.slug) for item in items]
        return tuple(
            np.stack([array[key][item.cache_index] for array, item in zip(arrays, items)])
            for key in ("inputSpec", "targetResidualSpec", "anchorResidualSpec", "eventFrameMask")
        )  # type: ignore[return-value]

    def preload(self) -> None:
        for index, slug in enumerate(sorted(self.paths), 1):
            self._load(slug)
            if index == 1 or index == len(self.paths) or index % 10 == 0:
                print(f"preload event-centered cache {index}/{len(self.paths)}", flush=True)


def build_schedule(
    records_by_song: dict[str, list[CenteredRecord]], passes: int, seed: int
) -> list[hard.ScheduleItem]:
    if not records_by_song:
        raise ValueError("No event-centered records")
    schedule: list[hard.ScheduleItem] = []
    slugs = sorted(records_by_song)
    for pass_index in range(passes):
        items = [hard.ScheduleItem(slug, index) for slug in slugs for index in range(len(records_by_song[slug]))]
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
    seed: int,
    history: list[dict[str, Any]],
    source: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    payload = {
        "format": "local-inst3-vr-event-centered-checkpoint@1",
        "status": status,
        "variant": variant,
        "runContractId": contract_id,
        "step": step,
        "passes": passes,
        "recordsPerPass": records_per_pass,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "anchorBeta": anchor_beta,
        "seed": seed,
        "stateDict": hard.cpu_tree(model.state_dict()),
        "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
        "history": history,
        "checkpointSource": source,
    }
    return hard.atomic_torch_save(path, payload)


def train_variant(
    *,
    variant: str,
    mode: str,
    h50_checkpoint: Path,
    architecture_checkpoint: Path,
    store: EventCenteredCacheStore,
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
        raise ValueError("Records per pass must be divisible by batch size")
    updates_per_pass = records_per_pass // batch_size
    total_updates = len(schedule) // batch_size
    milestone_updates = {value * updates_per_pass: value for value in milestones}
    run_root.mkdir(parents=True, exist_ok=True)
    hard.set_seed(seed)
    model, source = load_h50_model(architecture_checkpoint, h50_checkpoint, device)
    anchor_model = None
    if mode == "local-anchor":
        anchor_model, _ = load_h50_model(architecture_checkpoint, h50_checkpoint, device)
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
        anchor_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    current = 0
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
    milestones_out: dict[str, dict[str, Any]] = {}
    checkpoint_files: dict[str, dict[str, Any]] = {}

    def discover() -> None:
        for step in milestone_updates:
            path = run_root / f"step-{step}.pt"
            if path.is_file():
                metadata = checkpoint_metadata(path)
                milestones_out[str(step)] = metadata
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
            seed=seed,
            history=history,
            source=source,
            status=status,
        )
        milestones_out[str(step)] = metadata if step in milestone_updates else milestones_out.get(str(step), metadata)
        checkpoint_files[str(step)] = metadata

    discover()
    if current == 0 and 0 in milestone_updates and "0" not in milestones_out:
        checkpoint(0, "initial")
    while current < total_updates:
        begin = current * batch_size
        items = schedule[begin : begin + batch_size]
        if len(items) != batch_size:
            raise AssertionError("Incomplete batch")
        input_array, target_array, anchor_array, mask_array = store.batch(items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        anchor_tensor = torch.from_numpy(anchor_array).to(device)
        mask_tensor = torch.from_numpy(mask_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(input_tensor)
        if mode == "continuation":
            loss = F.l1_loss(prediction, target_tensor)
            event_loss = loss
            anchor_loss = torch.zeros((), device=device, dtype=loss.dtype)
            event_fraction = 1.0
        else:
            loss, event_loss, anchor_loss = local.local_anchor_loss(
                prediction, target_tensor, anchor_tensor, mask_tensor, anchor_beta
            )
            event_fraction = float(mask_array.mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {current + 1}")
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
            "eventFrameFraction": event_fraction,
            "gradientNormBeforeClip": gradient_norm,
        })
        if current == 1 or current % 100 == 0:
            print(json.dumps({"event": "progress", "variant": variant, "update": current, "totalUpdates": total_updates, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        if current in milestone_updates and current != 0:
            checkpoint(current, "milestone")
        elif state_interval > 0 and current % state_interval == 0:
            checkpoint(current, "rolling")
    discover()
    if str(total_updates) not in milestones_out:
        checkpoint(total_updates, "completed")
    del model, optimizer, anchor_model
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
        "milestoneCheckpoints": milestones_out,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
    }


def anchor_preservation(
    *,
    state_path: Path,
    architecture_checkpoint: Path,
    store: EventCenteredCacheStore,
    device: torch.device,
) -> dict[str, float]:
    model, _ = load_h50_model(architecture_checkpoint, state_path, device)
    error = 0.0
    anchor = 0.0
    count = 0.0
    for slug in sorted(store.paths):
        arrays = store._load(slug)
        with torch.inference_mode():
            for begin in range(0, len(arrays["inputSpec"]), 4):
                inputs = torch.from_numpy(arrays["inputSpec"][begin : begin + 4]).to(device)
                difference = model(inputs).to(dtype=torch.float64)
                reference = torch.from_numpy(arrays["anchorResidualSpec"][begin : begin + 4]).to(device, dtype=torch.float64)
                weights = torch.from_numpy(arrays["eventFrameMask"][begin : begin + 4]).to(device, dtype=torch.float64)[:, None, None, :]
                outside = 1.0 - weights
                error += float((difference - reference).square().mul(outside).sum().cpu())
                anchor += float(reference.square().mul(outside).sum().cpu())
                count += float(outside.sum().cpu() * difference.shape[1] * difference.shape[2])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "outsideAnchorErrorRmsDb": 10.0 * math.log10(max(error / max(count, 1.0), 1.0e-30)),
        "outsideAnchorRmsDb": 10.0 * math.log10(max(anchor / max(count, 1.0), 1.0e-30)),
        "outsideValueCount": count,
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
    model, metadata = load_h50_model(architecture_checkpoint, state_path, device)
    try:
        value = hard.evaluate_model_state(
            name=name,
            model=model,
            entries=entries,
            eval_root=eval_root,
            eval_windows_per_song=eval_windows_per_song,
            seed=seed,
            device=device,
        )
        value["checkpoint"] = metadata
        return value
    finally:
        del model
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
        value = hard.evaluate_model_state(
            name="initial-vocals",
            model=model,
            entries=entries,
            eval_root=eval_root,
            eval_windows_per_song=eval_windows_per_song,
            seed=seed,
            device=device,
        )
        value["checkpoint"] = metadata
        return value
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def render_listening(
    *,
    output_root: Path,
    samples_root: Path,
    architecture_checkpoint: Path,
    state_paths: dict[str, Path],
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    # Reuse the validated renderer used by the prior local-anchor studies.
    return local.render_listening(
        output_root=output_root,
        samples_root=samples_root,
        architecture_checkpoint=architecture_checkpoint,
        state_paths=state_paths,
        device=device,
        force=force,
    )


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return hard.metric_delta(candidate, baseline)


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.events_per_song <= 0 or args.passes <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("events-per-song, passes, batch-size, and threads must be positive")
    if args.learning_rate <= 0.0 or args.anchor_beta < 0.0 or args.state_interval <= 0:
        raise ValueError("learning-rate, anchor-beta, and state-interval must be valid")
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
    event_root = args.event_root.resolve()
    eval_root = args.eval_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    h50_checkpoint = args.h50_checkpoint.resolve()
    teacher = args.teacher.resolve()
    contract = args.contract.resolve()
    teacher_tflite = args.teacher_tflite.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = hard.load_manifest(manifest_path)
    train_entries = hard.load_train_entries(manifest, args.max_songs)
    eval_entries = sorted(
        [entry for entry in manifest["entries"] if entry.get("role") in {"calibration", "internal-test"}],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    if not checkpoint.is_file() or not h50_checkpoint.is_file() or not teacher.is_file():
        raise FileNotFoundError("Student, H50, or teacher checkpoint is missing")
    contract_info = pilot.verify_teacher_contract(contract, teacher, teacher_tflite)
    output_root.mkdir(parents=True, exist_ok=True)
    scan_root = output_root
    teacher_session, teacher_providers = pilot.make_teacher_session(
        teacher, args.threads, args.require_teacher_cuda
    )
    h50_model, h50_metadata = load_h50_model(checkpoint, h50_checkpoint, device)
    records_by_song: dict[str, list[CenteredRecord]] = {}
    scan_metadata: dict[str, Any] = {}
    cache_paths_by_song: dict[str, Path] = {}
    prep_root = output_root / "prep-work"
    shutil.rmtree(prep_root, ignore_errors=True)
    prep_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, entry in enumerate(train_entries, 1):
            slug, candidates = load_candidates(event_root, entry)
            scan_path, selection_path = scan_paths(scan_root, slug)
            events: list[EventRecord] | None = None
            song = None
            if not args.force_scan:
                events = load_existing_events(
                    root=scan_root,
                    slug=slug,
                    h50_checkpoint_sha256=file_sha256(h50_checkpoint),
                    expected_count=args.events_per_song,
                )
            cached = None
            centered: list[CenteredRecord] | None = None
            cache_path: Path | None = None
            cache_meta: dict[str, Any] | None = None
            if events is not None and not args.force_cache and not args.force_scan:
                cached = load_existing_cache(
                    root=output_root,
                    entry=entry,
                    slug=slug,
                    events=events,
                    checkpoint=checkpoint,
                    h50_checkpoint=h50_checkpoint,
                    teacher=teacher,
                )
                if cached is not None:
                    cache_path, cache_meta, centered = cached
            if events is None:
                song = hard.oracle.decode_song(
                    archive,
                    entry,
                    prep_root / "raw",
                    prep_root / "decoded",
                    force_extract=False,
                    force_decode=False,
                    keep_decoded_wav=False,
                )
                rows, timing = scan_song(
                    song=song,
                    candidates=candidates,
                    model=h50_model,
                    teacher_session=teacher_session,
                    device=device,
                )
                selected = rank_events(rows, args.events_per_song)
                if len(selected) < args.events_per_song:
                    raise ValueError(f"Only {len(selected)} events available for {slug}")
                events = selected
                json_write(
                    selection_path,
                    {
                        "contract": {
                            "schema": SCHEMA,
                            "sourceSha256": entry["sourceSha256"],
                            "member": entry["member"],
                            "h50CheckpointSha256": file_sha256(h50_checkpoint),
                            "candidateCount": len(candidates),
                            "eventsPerSong": args.events_per_song,
                            "ranking": "positive projection RMS over non-overlapping 50/100 ms blocks; miss RMS tie-break",
                        },
                        "events": [event_json(item) for item in events],
                    },
                )
                json_write(
                    scan_path,
                    {
                        "schema": SCHEMA,
                        "status": "completed",
                        "slug": slug,
                        "member": entry["member"],
                        "candidateCount": len(candidates),
                        "allBlockCount": len(rows),
                        "selected": [event_json(item) for item in events],
                        "timing": timing,
                    },
                )
            if centered is None or cache_path is None or cache_meta is None:
                # Decode only when the centered cache is absent or invalid.
                # The manifest intentionally does not carry a decoded sample
                # count, so this is also where the exact boundary clamp occurs.
                if song is None:
                    song = hard.oracle.decode_song(
                        archive,
                        entry,
                        prep_root / "raw",
                        prep_root / "decoded",
                        force_extract=False,
                        force_decode=False,
                        keep_decoded_wav=False,
                    )
                centered = centered_records(
                    events,
                    song.mixture_gt.shape[0],
                    pilot.DEFAULT_CONFIG.useful_samples,
                )
                cache_path, cache_meta = prepare_song_cache(
                    root=output_root,
                    song=song,
                    entry=entry,
                    records=centered,
                    teacher_session=teacher_session,
                    h50_model=h50_model,
                    device=device,
                    checkpoint=checkpoint,
                    h50_checkpoint=h50_checkpoint,
                    teacher=teacher,
                    force=args.force_cache,
                )
            assert centered is not None and cache_path is not None and cache_meta is not None
            records_by_song[slug] = centered
            cache_paths_by_song[slug] = cache_path
            scan_metadata[slug] = {
                "member": entry["member"],
                "events": [event_json(item) for item in events],
                "centered": [centered_json(item) for item in centered],
                "cache": cache_meta,
            }
            print(f"prepared {index}/{len(train_entries)}: {slug}", flush=True)
            del song
            # Decoded MUSDB18 PCM is only needed while preparing this song's
            # four centered records.  Keeping all songs in prep-work can
            # consume tens of gigabytes before training begins.
            shutil.rmtree(prep_root, ignore_errors=True)
            prep_root.mkdir(parents=True, exist_ok=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(prep_root, ignore_errors=True)
        del h50_model
        del teacher_session
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection_payload = {
        "schema": SCHEMA,
        "status": "completed",
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "eventsPerSong": args.events_per_song,
        "songs": scan_metadata,
    }
    json_write(output_root / "selection.json", selection_payload)
    selection_sha = file_sha256(output_root / "selection.json")
    schedule = build_schedule(records_by_song, args.passes, args.seed)
    schedule_summary = {
        "recordCount": len(schedule),
        "recordsPerPass": len(schedule) // args.passes,
        "updates": len(schedule) // args.batch_size,
        "scheduleSha256": canonical_sha256([
            {"slug": item.slug, "cacheIndex": item.cache_index} for item in schedule
        ]),
    }
    if schedule_summary["recordsPerPass"] % args.batch_size:
        raise ValueError("Event schedule is not divisible by batch size")
    store = EventCenteredCacheStore(cache_paths_by_song)
    common_contract = {
        "schema": SCHEMA,
        "selectionSha256": selection_sha,
        "scheduleSha256": schedule_summary["scheduleSha256"],
        "manifestSha256": file_sha256(manifest_path),
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "teacher": contract_info,
        "teacherProviders": teacher_providers,
        "trainSongCount": len(train_entries),
        "evalSongCount": len(eval_entries),
        "eventsPerSong": args.events_per_song,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "anchorBeta": args.anchor_beta,
        "seed": args.seed,
        "device": str(device),
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "eventSelection": "H50 actual miss; positive projection RMS over 50/100 ms blocks",
        "windowPlacement": "event center - usefulSamples/2, clamped to song bounds",
        "officialFinalTestUsed": False,
    }
    training: dict[str, Any] = {}
    for variant, mode in ((CONTINUATION, "continuation"), (LOCAL_ANCHOR, "local-anchor")):
        variant_contract = {**common_contract, "variant": variant, "mode": mode}
        training[variant] = train_variant(
            variant=variant,
            mode=mode,
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
    # Use the existing song-disjoint evaluation implementation and fixed
    # coverage windows so this experiment remains comparable to H50.
    evaluation: dict[str, Any] = {
        "initial-vocals": evaluate_initial(
            architecture_checkpoint=checkpoint,
            entries=eval_entries,
            eval_root=eval_root,
            eval_windows_per_song=args.eval_windows_per_song,
            seed=args.seed,
            device=device,
        ),
        "H50-pass-50": evaluate_checkpoint(
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
            result["deltaVsH50"] = metric_delta(result, baseline)
    final_anchor: dict[str, Any] = {}
    for variant, result in training.items():
        step = args.passes * result["updatesPerPass"]
        final_anchor[variant] = anchor_preservation(
            state_path=Path(result["milestoneCheckpoints"][str(step)]["file"]),
            architecture_checkpoint=checkpoint,
            store=store,
            device=device,
        )
    listening_report = None
    if not args.skip_listening:
        paths = {"H50-pass-50": h50_checkpoint}
        for variant, result in training.items():
            step = args.passes * result["updatesPerPass"]
            paths[f"{variant}@pass-{args.passes}"] = Path(result["milestoneCheckpoints"][str(step)]["file"])
        listening_report = render_listening(
            output_root=output_root / "listening-12",
            samples_root=args.samples_root.resolve(),
            architecture_checkpoint=checkpoint,
            state_paths=paths,
            device=device,
            force=args.force_listening,
        )
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "contract": common_contract,
        "selection": selection_payload,
        "schedule": schedule_summary,
        "training": training,
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
            "runner": {"file": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        },
        "notes": {
            "purpose": "H50 actual miss-centered context with Inst 3 event-local target and H50 outside-event anchor",
            "publication": "Local non-commercial research only; do not publish MUSDB18-derived checkpoints, caches, or audio.",
            "evaluation": "Calibration/internal-test only; official final-test was not used.",
        },
    }
    report_path = output_root / "reports" / "inst3-vr-event-centered-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": "completed", "report": str(report_path), "trainingVariants": list(training), "evaluationVariants": list(evaluation), "listeningCount": listening_report.get("outputCount", 0) if listening_report else 0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
