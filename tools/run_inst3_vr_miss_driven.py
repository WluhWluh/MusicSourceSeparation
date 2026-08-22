#!/usr/bin/env python3
"""Continue V-R-H50 using windows ranked by the current model's real miss.

This is a local, non-commercial MUSDB18 experiment.  The H50 checkpoint is
used to scan every Stage 1 candidate window in the 80-song training split.
For each candidate, the runner computes:

    teacher_residual_miss = (mixture - H50_instrumental) - H50_residual
                          = H50_instrumental - Inst3_instrumental

The four miss-driven windows per song are ranked by the largest positive
projection of that miss onto the Inst 3 removed signal across 50, 100, and
200 ms blocks.  Four uniform H50 windows are kept unchanged.  The control and
miss-driven arms then continue the H50 checkpoint for five passes with the
same optimizer state, batch size, learning rate, and schedule budget.

Only selected spectra and compact scan metadata are written under ``data``.
Full-song teacher and decoded audio are removed after each song is prepared.
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

import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_vr_hard_sampling as hard
import run_inst3_vr_local_anchor as local


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-sampling-h50"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-miss-driven"
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_MANIFEST = DEFAULT_EVAL_ROOT / "musdb18-inst3-oracle-manifest.json"
DEFAULT_ARCHIVE = pilot.DEFAULT_ARCHIVE
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_H50_CHECKPOINT = DEFAULT_BASE_ROOT / "runs" / "V-R-H50" / "step-8000.pt"
DEFAULT_TEACHER = pilot.DEFAULT_TEACHER
DEFAULT_CONTRACT = pilot.DEFAULT_CONTRACT
DEFAULT_TEACHER_TFLITE = pilot.DEFAULT_TEACHER_TFLITE
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"

SCHEMA = "local-inst3-vr-miss-driven@1"
CACHE_SCHEMA = "local-inst3-vr-miss-driven-cache@1"
CONTROL = "H50-control"
MISS_DRIVEN = "H50-miss-driven"
VARIANTS = (CONTROL, MISS_DRIVEN)
MILESTONES = (0, 1, 2, 5)


@dataclass(frozen=True)
class CandidateMiss:
    candidate_index: int
    start_samples: int
    length: int
    peak_positive_projection: float
    peak_miss_rms: float
    by_milliseconds: dict[str, dict[str, float]]


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
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--train-windows-per-song", type=int, default=8)
    parser.add_argument("--miss-windows-per-song", type=int, default=4)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--milestones", default="0,1,2,5")
    parser.add_argument("--batch-size", type=int, default=4)
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
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return __import__("hashlib").sha256(payload).hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def cache_path(root: Path, slug: str) -> Path:
    return root / "cache" / "train" / f"{slug}.npz"


def cache_metadata_path(root: Path, slug: str) -> Path:
    return root / "cache" / "train" / f"{slug}.json"


def scan_path(root: Path, slug: str) -> Path:
    return root / "scan" / "songs" / f"{slug}.json"


def load_h50_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("stateDict"), dict):
        raise ValueError(f"H50 checkpoint has no stateDict: {path}")
    if not isinstance(payload.get("optimizerStateDict"), dict):
        raise ValueError(f"H50 checkpoint has no optimizerStateDict: {path}")
    return payload


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
    device: torch.device,
) -> None:
    """Restore a CPU checkpoint and make every tensor usable on ``device``."""
    optimizer.load_state_dict(hard.cpu_tree(state_dict))
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor) and value.device != device:
                state[key] = value.to(device=device)


def assert_optimizer_state_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for value in state.values():
            same_device = (
                isinstance(value, torch.Tensor)
                and value.device.type == device.type
                and (
                    device.index is None
                    or value.device.index == device.index
                )
            )
            if isinstance(value, torch.Tensor) and not same_device:
                raise AssertionError(
                    f"Optimizer state remained on {value.device}; expected {device}"
                )


def build_all_candidate_selection(
    slug: str,
    member: str,
    candidates: list[hard.Candidate],
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


def peak(values: np.ndarray, active: np.ndarray) -> float:
    selected = values[active]
    if selected.size == 0:
        return 0.0
    value = float(np.max(selected))
    return value if math.isfinite(value) else 0.0


def candidate_miss_score(
    *,
    mixture: np.ndarray,
    teacher_instrumental: np.ndarray,
    predicted_residual: np.ndarray,
    sample_rate: int,
    candidate_index: int,
    start_samples: int,
    length: int,
) -> CandidateMiss:
    predicted_instrumental = np.ascontiguousarray(
        mixture - predicted_residual, dtype=np.float32
    )
    miss = np.ascontiguousarray(
        predicted_instrumental - teacher_instrumental, dtype=np.float32
    )
    teacher_removed = np.ascontiguousarray(
        mixture - teacher_instrumental, dtype=np.float32
    )
    by_milliseconds: dict[str, dict[str, float]] = {}
    positive_peak = 0.0
    miss_peak = 0.0
    for milliseconds in (50, 100, 200):
        values = hard.block_projection_arrays(
            teacher_removed, miss, sample_rate, milliseconds
        )
        positive = peak(values["positiveProjectionRms"], values["active"])
        miss_rms = peak(values["missRms"], values["active"])
        positive_peak = max(positive_peak, positive)
        miss_peak = max(miss_peak, miss_rms)
        by_milliseconds[str(milliseconds)] = {
            "peakPositiveProjectionRms": positive,
            "peakPositiveProjectionRmsDbfs": hard.dbfs(positive),
            "peakMissRms": miss_rms,
            "peakMissRmsDbfs": hard.dbfs(miss_rms),
        }
    return CandidateMiss(
        candidate_index=candidate_index,
        start_samples=start_samples,
        length=length,
        peak_positive_projection=positive_peak,
        peak_miss_rms=miss_peak,
        by_milliseconds=by_milliseconds,
    )


def rank_miss_candidates(scores: Sequence[CandidateMiss]) -> list[CandidateMiss]:
    return sorted(
        scores,
        key=lambda item: (
            -item.peak_positive_projection,
            -item.peak_miss_rms,
            item.start_samples,
        ),
    )


def build_miss_selection(
    h50_selection: hard.SongSelection,
    scores: Sequence[CandidateMiss],
    miss_count: int,
) -> tuple[hard.SongSelection, list[CandidateMiss]]:
    if miss_count <= 0:
        raise ValueError("miss_count must be positive")
    if len(h50_selection.uniform_indices) < miss_count:
        raise ValueError("H50 uniform draw count is smaller than miss count")
    uniform_indices = tuple(h50_selection.uniform_indices[:miss_count])
    uniform_set = set(uniform_indices)
    ranked = rank_miss_candidates(scores)
    non_uniform = [item for item in ranked if item.candidate_index not in uniform_set]
    if len(non_uniform) >= miss_count:
        selected_scores = non_uniform[:miss_count]
    else:
        # Very short songs can have fewer than eight valid TFC windows.  Keep
        # the per-song draw budget stable, matching H50's replacement policy:
        # repeat the available non-uniform miss candidates deterministically;
        # only fall back to all ranked candidates when none are available.
        fallback_pool = non_uniform or ranked
        if not fallback_pool:
            raise ValueError(f"No miss candidates available for {h50_selection.slug}")
        selected_scores = [
            fallback_pool[index % len(fallback_pool)] for index in range(miss_count)
        ]
    if len(selected_scores) != miss_count:
        raise ValueError(
            f"Only {len(selected_scores)} miss candidates available; expected {miss_count}"
        )
    miss_indices = tuple(item.candidate_index for item in selected_scores)
    hard_indices = uniform_indices + miss_indices
    union_indices = tuple(sorted(set(hard_indices)))
    selection = hard.SongSelection(
        slug=h50_selection.slug,
        member=h50_selection.member,
        candidates=h50_selection.candidates,
        hard_pool_indices=tuple(item.candidate_index for item in ranked),
        uniform_indices=uniform_indices,
        hard_fraction=0.5,
        hard_indices=hard_indices,
        union_indices=union_indices,
    )
    return selection, selected_scores


def selection_json(
    selection: hard.SongSelection,
    scores: Sequence[CandidateMiss],
    selected_scores: Sequence[CandidateMiss],
) -> dict[str, Any]:
    by_index = {item.candidate_index: item for item in scores}
    selected_indices = {item.candidate_index for item in selected_scores}
    return {
        "slug": selection.slug,
        "member": selection.member,
        "candidateCount": len(selection.candidates),
        "uniformCandidateIndices": list(selection.uniform_indices),
        "missDrivenCandidateIndices": [item.candidate_index for item in selected_scores],
        "trainingCandidateIndices": list(selection.hard_indices),
        "unionCandidateIndices": list(selection.union_indices),
        "candidates": [
            {
                "index": candidate.index,
                "startSamples": candidate.start,
                "length": candidate.length,
                "stage1HardEventScore": candidate.hard_score,
                "stage1Rank": candidate.rank,
                "selected": candidate.index in selected_indices,
                "miss": by_index[candidate.index].by_milliseconds,
                "peakPositiveProjectionRmsDbfs": hard.dbfs(
                    by_index[candidate.index].peak_positive_projection
                ),
                "peakMissRmsDbfs": hard.dbfs(by_index[candidate.index].peak_miss_rms),
            }
            for candidate in selection.candidates
        ],
    }


def make_model_from_h50(
    architecture_checkpoint: Path,
    h50_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, initialization = pilot.make_model(architecture_checkpoint, device)
    payload = load_h50_payload(h50_checkpoint)
    model.load_state_dict(payload["stateDict"], strict=True)
    sweep.freeze_batchnorm_running_statistics(model)
    model.eval()
    return model, {
        "checkpoint": checkpoint_metadata(h50_checkpoint),
        "step": int(payload.get("step", -1)),
        "passes": int(payload.get("passes", -1)),
        "learningRate": float(payload.get("learningRate", 0.0)),
        "initialization": initialization,
        "optimizerStatePreserved": True,
    }


def scan_predictions(
    song: Any,
    candidates: Sequence[hard.Candidate],
    model: torch.nn.Module,
    teacher_session: Any,
    device: torch.device,
    threads: int,
) -> tuple[list[CandidateMiss], list[np.ndarray], dict[str, Any]]:
    del threads
    all_selection = build_all_candidate_selection(song.slug, song.entry["member"], list(candidates))
    teacher_segments, teacher_report = hard.render_selected_teacher_segments(
        song.mixture_gt,
        all_selection,
        teacher_session,
        progress_label=f"scan/{song.slug}",
    )
    scores: list[CandidateMiss] = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(candidates), 4):
            batch_candidates = candidates[begin : begin + 4]
            input_specs = np.stack(
                [
                    pilot.student_window_spec(
                        song.mixture_gt, item.start, item.length
                    )
                    for item in batch_candidates
                ]
            )
            input_tensor = torch.from_numpy(input_specs).to(device)
            predicted_specs = model(input_tensor).detach().cpu().numpy()
            for offset, candidate in enumerate(batch_candidates):
                reconstructed = pilot.student_istft_centered(
                    predicted_specs[offset : offset + 1]
                )
                trim = pilot.DEFAULT_CONFIG.trim_samples
                predicted_residual = np.ascontiguousarray(
                    reconstructed[trim : trim + candidate.length], dtype=np.float32
                )
                scores.append(
                    candidate_miss_score(
                        mixture=song.mixture_gt[
                            candidate.start : candidate.start + candidate.length
                        ],
                        teacher_instrumental=teacher_segments[begin + offset],
                        predicted_residual=predicted_residual,
                        sample_rate=song.sample_rate,
                        candidate_index=candidate.index,
                        start_samples=candidate.start,
                        length=candidate.length,
                    )
                )
    if len(scores) != len(candidates):
        raise AssertionError("Candidate scan count mismatch")
    return scores, teacher_segments, {
        "teacher": teacher_report,
        "h50ScanSeconds": time.perf_counter() - started,
    }


def write_selected_cache(
    *,
    root: Path,
    song: Any,
    selection: hard.SongSelection,
    teacher_segments: Sequence[np.ndarray],
    candidates: Sequence[hard.Candidate],
    scores: Sequence[CandidateMiss],
    selected_scores: Sequence[CandidateMiss],
    entry: dict[str, Any],
    event_root: Path,
    checkpoint: Path,
    h50_checkpoint: Path,
    teacher: Path,
    scan_sha256: str,
    providers: Sequence[str],
    render_report: dict[str, Any],
    force: bool,
) -> Path:
    output = cache_path(root, selection.slug)
    metadata_path = cache_metadata_path(root, selection.slug)
    expected = {
        "schema": CACHE_SCHEMA,
        "sourceSha256": entry["sourceSha256"],
        "member": entry["member"],
        "slug": selection.slug,
        "scanSha256": scan_sha256,
        "h50CheckpointSha256": file_sha256(h50_checkpoint),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "teacherSha256": file_sha256(teacher),
        "teacherContractId": "uvr_mdxnet_inst_3@2",
        "sampleRate": pilot.DEFAULT_CONFIG.sample_rate,
        "numFrames": pilot.DEFAULT_CONFIG.num_frames,
        "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
        "targetAlignment": "segment-origin-0",
        "candidateIndices": list(selection.union_indices),
        "starts": [candidates[index].start for index in selection.union_indices],
        "lengths": [candidates[index].length for index in selection.union_indices],
    }
    if not force and output.is_file() and metadata_path.is_file():
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(output) as values:
                valid = (
                    actual.get("contract") == expected
                    and values["inputSpec"].shape == values["targetResidualSpec"].shape
                    and values["inputSpec"].shape[0] == len(selection.union_indices)
                )
            if valid:
                return output
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    index_to_segment = {
        candidate_index: position
        for position, candidate_index in enumerate(
            candidate.index for candidate in candidates
        )
    }
    input_specs: list[np.ndarray] = []
    target_specs: list[np.ndarray] = []
    miss_scores: list[float] = []
    for candidate_index in selection.union_indices:
        candidate = candidates[candidate_index]
        teacher_instrumental = teacher_segments[index_to_segment[candidate_index]]
        mixture = song.mixture_gt[candidate.start : candidate.start + candidate.length]
        teacher_residual = mixture - teacher_instrumental
        input_specs.append(
            pilot.student_window_spec(song.mixture_gt, candidate.start, candidate.length)
        )
        target_specs.append(hard.student_segment_spec(teacher_residual))
        score = next(
            item.peak_positive_projection
            for item in scores
            if item.candidate_index == candidate_index
        )
        miss_scores.append(score)
    input_array = np.ascontiguousarray(np.stack(input_specs), dtype=np.float32)
    target_array = np.ascontiguousarray(np.stack(target_specs), dtype=np.float32)
    if input_array.shape != target_array.shape:
        raise ValueError(f"Miss-driven cache shape mismatch for {selection.slug}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    np.savez_compressed(
        temporary,
        inputSpec=input_array,
        targetResidualSpec=target_array,
        starts=np.asarray(
            [candidates[index].start for index in selection.union_indices], dtype=np.int64
        ),
        lengths=np.asarray(
            [candidates[index].length for index in selection.union_indices], dtype=np.int64
        ),
        missScores=np.asarray(miss_scores, dtype=np.float32),
    )
    temporary_npz = temporary if temporary.suffix == ".npz" else Path(str(temporary) + ".npz")
    temporary_npz.replace(output)
    metadata = {
        "contract": expected,
        "cache": checkpoint_metadata(output),
        "render": {
            "providers": list(providers),
            "selectedTfcWindowCount": len(selection.union_indices),
            "selectedCandidateCount": len(selection.union_indices),
            "mode": "scan-and-select-sparse-generation-chunks",
            "totalSeconds": render_report.get("totalSeconds"),
        },
        "selected": selection_json(selection, scores, selected_scores),
    }
    json_write(metadata_path, metadata)
    return output


def load_or_scan_song(
    *,
    entry: dict[str, Any],
    h50_selection: hard.SongSelection,
    archive: Path,
    event_root: Path,
    root: Path,
    checkpoint: Path,
    h50_checkpoint: Path,
    teacher: Path,
    model: torch.nn.Module,
    teacher_session: Any,
    teacher_providers: Sequence[str],
    device: torch.device,
    threads: int,
    force_scan: bool,
    force_cache: bool,
    miss_count: int,
) -> tuple[hard.SongSelection, dict[str, Any]]:
    slug = h50_selection.slug
    path = scan_path(root, slug)
    candidates = list(h50_selection.candidates)
    if path.is_file() and not force_scan:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if (
                report.get("contract", {}).get("sourceSha256") == entry["sourceSha256"]
                and report.get("contract", {}).get("h50CheckpointSha256")
                == file_sha256(h50_checkpoint)
                and report.get("contract", {}).get("candidateCount") == len(candidates)
                and report.get("contract", {}).get("missWindowsPerSong")
                == miss_count
            ):
                selected = report["selection"]
                selection = hard.SongSelection(
                    slug=slug,
                    member=entry["member"],
                    candidates=tuple(candidates),
                    hard_pool_indices=tuple(selected["rankedCandidateIndices"]),
                    uniform_indices=tuple(selected["uniformCandidateIndices"]),
                    hard_fraction=0.5,
                    hard_indices=tuple(selected["trainingCandidateIndices"]),
                    union_indices=tuple(selected["unionCandidateIndices"]),
                )
                cache = cache_path(root, slug)
                if cache.is_file() and cache_metadata_path(root, slug).is_file():
                    return selection, report
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    work_root = root / "prep-work"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        song = hard.oracle.decode_song(
            archive,
            entry,
            work_root / "raw",
            work_root / "decoded",
            force_extract=False,
            force_decode=False,
            keep_decoded_wav=False,
        )
        scores, teacher_segments, render_report = scan_predictions(
            song,
            candidates,
            model,
            teacher_session,
            device,
            threads,
        )
        selection, selected_scores = build_miss_selection(
            h50_selection, scores, miss_count=miss_count
        )
        scan_contract = {
            "schema": SCHEMA,
            "sourceSha256": entry["sourceSha256"],
            "member": entry["member"],
            "h50CheckpointSha256": file_sha256(h50_checkpoint),
            "studentArchitectureSha256": file_sha256(checkpoint),
            "teacherSha256": file_sha256(teacher),
            "candidateCount": len(candidates),
            "missWindowsPerSong": miss_count,
            "metricWindowsMs": [50, 100, 200],
            "ranking": "max positive projection of teacherResidualMiss across 50/100/200 ms; peak miss RMS tie-break",
        }
        report = {
            "schema": SCHEMA,
            "status": "completed",
            "contract": scan_contract,
            "selection": {
                "uniformCandidateIndices": list(selection.uniform_indices),
                "missDrivenCandidateIndices": [item.candidate_index for item in selected_scores],
                "trainingCandidateIndices": list(selection.hard_indices),
                "unionCandidateIndices": list(selection.union_indices),
                "rankedCandidateIndices": [item.candidate_index for item in rank_miss_candidates(scores)],
            },
            "candidates": selection_json(selection, scores, selected_scores)["candidates"],
            "timing": render_report,
            "scanDevice": str(device),
            "teacherProviders": list(teacher_providers),
        }
        json_write(path, report)
        scan_sha = file_sha256(path)
        write_selected_cache(
            root=root,
            song=song,
            selection=selection,
            teacher_segments=teacher_segments,
            candidates=candidates,
            scores=scores,
            selected_scores=selected_scores,
            entry=entry,
            event_root=event_root,
            checkpoint=checkpoint,
            h50_checkpoint=h50_checkpoint,
            teacher=teacher,
            scan_sha256=scan_sha,
            providers=teacher_providers,
            render_report=render_report["teacher"],
            force=force_cache,
        )
        return selection, report
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_schedule(
    selections: dict[str, hard.SongSelection],
    passes: int,
    seed: int,
) -> list[hard.ScheduleItem]:
    schedule: list[hard.ScheduleItem] = []
    for pass_index in range(passes):
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        items: list[hard.ScheduleItem] = []
        for slug in sorted(selections):
            selection = selections[slug]
            cache_map = selection.cache_index_by_candidate
            items.extend(
                hard.ScheduleItem(slug, cache_map[index])
                for index in selection.hard_indices
            )
        schedule.extend(items[int(index)] for index in rng.permutation(len(items)))
    return schedule


def schedule_json(schedule: Sequence[hard.ScheduleItem]) -> list[dict[str, Any]]:
    return [{"slug": item.slug, "cacheIndex": item.cache_index} for item in schedule]


def schedule_summary(
    schedule: Sequence[hard.ScheduleItem], passes: int, batch_size: int
) -> dict[str, Any]:
    if len(schedule) % batch_size != 0:
        raise ValueError("Schedule is not divisible by batch size")
    records_per_pass = len(schedule) // passes
    return {
        "recordCount": len(schedule),
        "recordsPerPass": records_per_pass,
        "updates": len(schedule) // batch_size,
        "scheduleSha256": canonical_sha256(schedule_json(schedule)),
    }


def train_arm(
    *,
    name: str,
    architecture_checkpoint: Path,
    h50_checkpoint: Path,
    source_payload: dict[str, Any],
    cache_paths: dict[str, Path],
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
    if records_per_pass % batch_size != 0:
        raise ValueError("Records per pass must be divisible by batch size")
    updates_per_pass = records_per_pass // batch_size
    total_updates = len(schedule) // batch_size
    milestone_updates = {
        pass_count * updates_per_pass: pass_count for pass_count in milestones
    }
    run_root.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, _ = make_model_from_h50(architecture_checkpoint, h50_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    load_optimizer_state(optimizer, source_payload["optimizerStateDict"], device)
    assert_optimizer_state_device(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    current_update = 0
    history: list[dict[str, float | int]] = []
    resume_count = 0
    if resume:
        found = None
        for path in run_root.glob("step-*.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("runContractId") != contract_id:
                    continue
                step = int(payload["step"])
            except (OSError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            if found is None or step > found[0]:
                found = (step, path, payload)
        if found is not None:
            _step, path, payload = found
            model.load_state_dict(payload["stateDict"], strict=True)
            load_optimizer_state(optimizer, payload["optimizerStateDict"], device)
            assert_optimizer_state_device(optimizer, device)
            current_update = int(payload["step"])
            history = list(payload.get("history", []))
            resume_count = int(payload.get("resumeCount", 0)) + 1
            print(json.dumps({"event": "resume", "arm": name, "step": current_update, "file": str(path)}), flush=True)

    store = hard.CacheStore(cache_paths, max_open=len(cache_paths))
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
            "format": "local-inst3-vr-miss-driven-checkpoint@1",
            "status": status,
            "arm": name,
            "runContractId": contract_id,
            "step": step,
            "passes": passes,
            "recordsPerPass": records_per_pass,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "seed": seed,
            "sourceH50Step": int(source_payload.get("step", -1)),
            "sourceH50Checkpoint": checkpoint_metadata(h50_checkpoint),
            "sourceOptimizerPreserved": True,
            "stateDict": hard.cpu_tree(model.state_dict()),
            "optimizerStateDict": hard.cpu_tree(optimizer.state_dict()),
            "history": history,
            "resumeCount": resume_count,
            "elapsedSeconds": time.perf_counter() - started,
        }
        metadata = hard.atomic_torch_save(path, payload)
        checkpoint_files[str(step)] = metadata
        if step in milestone_updates:
            milestone_files[str(step)] = metadata

    if current_update == 0 and 0 in milestone_updates:
        save_checkpoint(0, "initial-from-h50")

    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    while current_update < total_updates:
        batch_items = schedule[current_update * batch_size : (current_update + 1) * batch_size]
        if len(batch_items) != batch_size:
            raise AssertionError("Incomplete training batch")
        input_array, target_array = store.batch(batch_items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(input_tensor)
        loss = F.l1_loss(predicted, target_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {current_update + 1}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item()
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
            print(json.dumps({"event": "progress", "arm": name, "update": current_update, "totalUpdates": total_updates, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        if current_update in milestone_updates and current_update != 0:
            save_checkpoint(current_update, "milestone")
            print(json.dumps({"event": "milestone", "arm": name, "pass": milestone_updates[current_update], "update": current_update, "loss": history[-1]["loss"]}, sort_keys=True), flush=True)
        elif state_interval > 0 and current_update % state_interval == 0:
            save_checkpoint(current_update, "rolling")

    if str(total_updates) not in milestone_files:
        save_checkpoint(total_updates, "completed")
    else:
        discover()
    result = {
        "arm": name,
        "status": "completed",
        "passes": passes,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": updates_per_pass,
        "updates": total_updates,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "seed": seed,
        "sourceH50Step": int(source_payload.get("step", -1)),
        "sourceOptimizerPreserved": True,
        "history": history,
        "milestoneCheckpoints": milestone_files,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
        "resumeCount": resume_count,
    }
    del model, optimizer, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_state(
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
    model, metadata = local.load_state_model(
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


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = parse_int_list(args.milestones)
    if milestones != MILESTONES and args.passes == 5:
        raise ValueError(f"Default five-pass run requires milestones {MILESTONES}")
    if args.train_windows_per_song != 8:
        raise ValueError("This experiment requires exactly 8 total windows per song")
    if args.miss_windows_per_song <= 0 or args.miss_windows_per_song > args.train_windows_per_song // 2:
        raise ValueError(
            "miss-windows-per-song must be between 1 and half the total window count"
        )
    if args.passes <= 0 or args.batch_size <= 0 or args.threads <= 0 or args.state_interval <= 0:
        raise ValueError("passes, batch-size, threads, and state-interval must be positive")
    if args.max_songs is not None and args.max_songs <= 0:
        raise ValueError("max-songs must be positive")
    if args.max_eval_songs is not None and args.max_eval_songs <= 0:
        raise ValueError("max-eval-songs must be positive")
    if args.eval_windows_per_song <= 0:
        raise ValueError("eval-windows-per-song must be positive")

    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    event_root = args.event_root.resolve()
    eval_root = args.eval_root.resolve()
    base_root = args.base_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    h50_checkpoint = args.h50_checkpoint.resolve()
    teacher = args.teacher.resolve()
    contract_path = args.contract.resolve()
    teacher_tflite = args.teacher_tflite.resolve()
    if not archive.is_file() or not checkpoint.is_file() or not h50_checkpoint.is_file():
        raise FileNotFoundError("archive, architecture checkpoint, or H50 checkpoint is missing")
    source_payload = load_h50_payload(h50_checkpoint)
    source_learning_rate = float(source_payload.get("learningRate", 0.0))
    if source_learning_rate <= 0.0:
        raise ValueError("H50 checkpoint has no positive learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output_root.mkdir(parents=True, exist_ok=True)

    teacher_contract = pilot.verify_teacher_contract(
        contract_path, teacher, teacher_tflite
    )

    manifest = hard.load_manifest(manifest_path)
    train_entries = hard.load_train_entries(manifest, args.max_songs)
    eval_entries = sorted(
        [
            entry
            for entry in manifest["entries"]
            if entry.get("role") in {"calibration", "internal-test"}
        ],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    base_report = hard.load_manifest(manifest_path)
    del base_report
    h50_report = json.loads(
        (base_root / "reports" / "inst3-vr-hard-sampling-report.json").read_text(encoding="utf-8")
    )
    h50_selections, _, h50_selection_sha = hard.build_selections(
        entries=train_entries,
        event_root=event_root,
        train_windows_per_song=8,
        hard_fraction=0.5,
        seed=args.seed,
    )
    expected_selection = h50_report["contract"]["selectionSha256"]
    if args.max_songs is None and h50_selection_sha != expected_selection:
        raise ValueError(f"H50 selection changed: {h50_selection_sha} != {expected_selection}")
    if args.max_songs is None and h50_report["status"] != "completed":
        raise ValueError("H50 base report is not completed")

    model, scan_model_metadata = make_model_from_h50(
        checkpoint, h50_checkpoint, device
    )
    teacher_session, teacher_providers = pilot.make_teacher_session(
        teacher, args.threads, args.require_teacher_cuda
    )
    miss_selections: dict[str, hard.SongSelection] = {}
    scan_reports: dict[str, Any] = {}
    miss_cache_paths: dict[str, Path] = {}
    try:
        for index, entry in enumerate(train_entries, start=1):
            slug = hard.oracle.slugify(entry["fileName"])
            h50_selection = h50_selections[slug]
            selection, report = load_or_scan_song(
                entry=entry,
                h50_selection=h50_selection,
                archive=archive,
                event_root=event_root,
                root=output_root,
                checkpoint=checkpoint,
                h50_checkpoint=h50_checkpoint,
                teacher=teacher,
                model=model,
                teacher_session=teacher_session,
                teacher_providers=teacher_providers,
                device=device,
                threads=args.threads,
                force_scan=args.force_scan,
                force_cache=args.force_cache,
                miss_count=args.miss_windows_per_song,
            )
            miss_selections[slug] = selection
            scan_reports[slug] = report
            miss_cache_paths[slug] = cache_path(output_root, slug)
            if not miss_cache_paths[slug].is_file():
                raise FileNotFoundError(miss_cache_paths[slug])
            print(f"scan/cache {index}/{len(train_entries)}: {slug}", flush=True)
    finally:
        del teacher_session, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(miss_selections) != len(train_entries):
        raise AssertionError("Miss-driven selection count mismatch")

    h50_cache_paths = {
        slug: base_root / "cache" / "train" / f"{slug}.npz"
        for slug in h50_selections
    }
    for path in h50_cache_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    schedules = {
        CONTROL: hard.build_training_schedule(
            h50_selections, "V-R-H50", args.passes, args.seed, hard_variant="V-R-H50"
        ),
        MISS_DRIVEN: build_schedule(miss_selections, args.passes, args.seed),
    }
    schedule_info = {
        name: schedule_summary(schedule, args.passes, args.batch_size)
        for name, schedule in schedules.items()
    }
    for name, info in schedule_info.items():
        if info["recordsPerPass"] != len(train_entries) * args.train_windows_per_song:
            raise AssertionError(f"Unexpected record budget for {name}")

    miss_selection_payload = {
        "schema": "local-inst3-vr-miss-driven-selection@1",
        "seed": args.seed,
        "trainWindowsPerSong": args.train_windows_per_song,
        "missWindowsPerSong": args.miss_windows_per_song,
        "ranking": "max positive projection of teacherResidualMiss across 50/100/200 ms; peak miss RMS tie-break",
        "songs": [
            scan_reports[slug]["selection"] | {"slug": slug}
            for slug in sorted(scan_reports)
        ],
    }
    miss_selection_sha = canonical_sha256(miss_selection_payload)
    json_write(output_root / "selection.json", miss_selection_payload)

    common_contract = {
        "schema": SCHEMA,
        "baseH50ReportSha256": file_sha256(
            base_root / "reports" / "inst3-vr-hard-sampling-report.json"
        ),
        "baseH50SelectionSha256": h50_selection_sha,
        "missSelectionSha256": miss_selection_sha,
        "manifestSha256": file_sha256(manifest_path),
        "eventReportSha256": file_sha256(
            event_root / "reports" / "inst3-vr-hard-events-report.json"
        ),
        "studentArchitectureSha256": file_sha256(checkpoint),
        "h50Checkpoint": checkpoint_metadata(h50_checkpoint),
        "sourceH50Step": int(source_payload.get("step", -1)),
        "sourceH50Passes": int(source_payload.get("passes", -1)),
        "sourceOptimizerPreserved": True,
        "trainSongCount": len(train_entries),
        "evalSongCount": len(eval_entries),
        "trainWindowsPerSong": args.train_windows_per_song,
        "missWindowsPerSong": args.miss_windows_per_song,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "learningRate": source_learning_rate,
        "seed": args.seed,
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
        "missDefinition": "teacherResidualMiss = (mixtureGt - H50Instrumental) - H50Residual = H50Instrumental - Inst3Instrumental",
        "ranking": "max positive projection of teacherResidualMiss across 50/100/200 ms; peak miss RMS tie-break",
        "officialFinalTestUsed": False,
        "teacherContract": teacher_contract,
    }

    training: dict[str, Any] = {}
    for name, cache_paths, selections in (
        (CONTROL, h50_cache_paths, h50_selections),
        (MISS_DRIVEN, miss_cache_paths, miss_selections),
    ):
        arm_contract = {
            **common_contract,
            "arm": name,
            "scheduleSha256": schedule_info[name]["scheduleSha256"],
        }
        contract_id = canonical_sha256(arm_contract)
        training[name] = train_arm(
            name=name,
            architecture_checkpoint=checkpoint,
            h50_checkpoint=h50_checkpoint,
            source_payload=source_payload,
            cache_paths=cache_paths,
            schedule=schedules[name],
            run_root=output_root / "runs" / name,
            contract_id=contract_id,
            passes=args.passes,
            milestones=milestones,
            batch_size=args.batch_size,
            learning_rate=source_learning_rate,
            seed=args.seed,
            device=device,
            state_interval=args.state_interval,
            resume=args.resume,
        )
        training[name]["contract"] = {"id": contract_id, "payload": arm_contract}

    evaluation: dict[str, Any] = {}
    evaluation["initial-vocals"] = evaluate_initial(
        architecture_checkpoint=checkpoint,
        entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=args.eval_windows_per_song,
        seed=args.seed,
        device=device,
    )
    evaluation["H50-pass-50"] = evaluate_state(
        name="H50-pass-50",
        state_path=h50_checkpoint,
        architecture_checkpoint=checkpoint,
        entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=args.eval_windows_per_song,
        seed=args.seed,
        device=device,
    )
    for name, result in training.items():
        for pass_count in milestones:
            if pass_count == 0:
                continue
            step = pass_count * result["updatesPerPass"]
            metadata = result["milestoneCheckpoints"].get(str(step))
            if metadata is None:
                raise FileNotFoundError(f"Missing {name} pass {pass_count}")
            evaluation[f"{name}@pass-{pass_count}"] = evaluate_state(
                name=f"{name}@pass-{pass_count}",
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

    listening_report: dict[str, Any] | None = None
    if not args.skip_listening:
        final_paths = {
            "H50-pass-50": h50_checkpoint,
            f"{CONTROL}@pass-{args.passes}": Path(
                training[CONTROL]["milestoneCheckpoints"][str(args.passes * training[CONTROL]["updatesPerPass"])]
                ["file"]
            ),
            f"{MISS_DRIVEN}@pass-{args.passes}": Path(
                training[MISS_DRIVEN]["milestoneCheckpoints"][str(args.passes * training[MISS_DRIVEN]["updatesPerPass"])]
                ["file"]
            ),
        }
        listening_report = local.render_listening(
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
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "teacher": "Inst 3 teacher and derived outputs remain local; not published",
        },
        "contract": common_contract,
        "manifest": {
            "file": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifestId": manifest["manifestId"],
            "trainSongCount": len(train_entries),
            "evaluationSongCount": len(eval_entries),
            "officialFinalTestUsed": False,
        },
        "selection": miss_selection_payload,
        "schedule": schedule_info,
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
            "teacherProviders": list(teacher_providers),
            "teacherContract": teacher_contract,
            "scanModel": scan_model_metadata,
            "runner": {
                "file": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "notes": {
            "control": "H50-control uses the existing four uniform plus four Stage 1 hard windows and continues the H50 checkpoint.",
            "missDriven": "H50-miss-driven uses the same four uniform windows plus four windows ranked from current H50 teacherResidualMiss peaks.",
            "ranking": "Primary score is the maximum positive projection of teacherResidualMiss over 50/100/200 ms; raw miss RMS is a tie-break and is reported separately.",
            "publication": "Do not publish checkpoints, teacher-derived audio, scan artifacts, or MUSDB18-derived caches.",
        },
    }
    report_path = output_root / "reports" / "inst3-vr-miss-driven-report.json"
    json_write(report_path, report)
    print(json.dumps({"status": "completed", "report": str(report_path.resolve()), "training": list(training), "evaluation": list(evaluation), "listeningCount": listening_report.get("outputCount", 0) if listening_report else 0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
