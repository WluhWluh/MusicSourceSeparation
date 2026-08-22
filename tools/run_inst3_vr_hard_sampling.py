#!/usr/bin/env python3
"""Compare uniform and per-song hard-window sampling for Inst 3 distillation.

Both cells start from the frozen ``vocals_epoch=891.ckpt`` checkpoint and keep
its residual-vocals output semantic.  The training target is the residual
removed by the Inst 3 teacher:

    teacher_residual = mixture_gt - teacher_instrumental

The two cells differ only in the fixed eight-window sample set selected for
each of the 80 MUSDB18 train songs:

* ``V-R-U``: eight uniform candidates per song.
* ``V-R-H25``: six of the same uniform candidates plus two candidates from the
  song-local top 25 percent hard-event pool.
* ``V-R-H50``: four uniform candidates plus four candidates from the same
  song-local hard-event pool.  The hard fraction is selected by the CLI so the
  original U/H25 run remains the default.

The Stage 1 hard-event report supplies the candidate starts and scores.  The
runner regenerates teacher output one song at a time, writes only selected
window spectra under the ignored ``data`` tree, and removes full-song decode
and teacher intermediates immediately.  It is a local, non-commercial
research experiment.  It does not export or publish weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_teacher_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_EVAL_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-sampling"
EVENT_SCHEMA = "local-inst3-vr-hard-events@1"
RUN_SCHEMA = "local-inst3-vr-hard-sampling@1"
CACHE_SCHEMA = "local-inst3-vr-hard-sampling-cache@2"
TARGET_ALIGNMENT = "segment-origin-0"
VARIANTS = ("V-R-U", "V-R-H25")
SUPPORTED_VARIANTS = ("V-R-U", "V-R-H25", "V-R-H50")
DEFAULT_MILESTONES = (0, 25, 50)


@dataclass(frozen=True)
class Candidate:
    index: int
    start: int
    length: int
    hard_score: float
    rank: int


@dataclass(frozen=True)
class SongSelection:
    slug: str
    member: str
    candidates: tuple[Candidate, ...]
    hard_pool_indices: tuple[int, ...]
    uniform_indices: tuple[int, ...]
    hard_fraction: float
    hard_indices: tuple[int, ...]
    union_indices: tuple[int, ...]

    @property
    def h25_indices(self) -> tuple[int, ...]:
        """Backward-compatible name for the selected hard-fraction draws."""
        return self.hard_indices

    @property
    def cache_index_by_candidate(self) -> dict[int, int]:
        return {candidate_index: index for index, candidate_index in enumerate(self.union_indices)}


@dataclass(frozen=True)
class ScheduleItem:
    slug: str
    cache_index: int


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"Expected sorted unique integer list, got {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=pilot.DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=pilot.DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=pilot.DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=pilot.DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--train-windows-per-song", type=int, default=8)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument(
        "--hard-variant",
        choices=("V-R-H25", "V-R-H50"),
        default="V-R-H25",
        help="Name of the hard-sampling arm; U is always trained as the control",
    )
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument(
        "--milestones",
        default=",".join(str(value) for value in DEFAULT_MILESTONES),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--state-interval",
        type=int,
        default=500,
        help="Write a resumable operational checkpoint at this many updates",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--eval-windows-per-song", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--force-prep", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--require-teacher-cuda", action=argparse.BooleanOptionalAction, default=True)
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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return pilot.sha256_file(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "song"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifestId") != "musdb18-inst3-oracle-split@1":
        raise ValueError(f"Unexpected manifest: {value.get('manifestId')}")
    return value


def load_train_entries(manifest: dict[str, Any], max_songs: int | None) -> list[dict[str, Any]]:
    entries = sorted(
        [entry for entry in manifest["entries"] if entry.get("role") == "train"],
        key=lambda entry: entry["member"],
    )
    if len(entries) != 80:
        raise ValueError(f"Expected 80 train entries, found {len(entries)}")
    if max_songs is not None:
        if max_songs <= 0:
            raise ValueError("max-songs must be positive")
        entries = entries[:max_songs]
    return entries


def load_stage1_candidates(
    event_root: Path,
    entry: dict[str, Any],
) -> tuple[str, dict[str, Any], list[Candidate]]:
    slug = oracle.slugify(entry["fileName"])
    report_path = event_root / "songs" / f"{slug}.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing Stage 1 report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("song", {}).get("member") != entry["member"]:
        raise ValueError(f"Stage 1 report member mismatch for {slug}")
    if report.get("studentWindows", {}).get("usefulSamples") != pilot.DEFAULT_CONFIG.useful_samples:
        raise ValueError(f"Unexpected student window contract for {slug}")
    ranked = report["studentWindows"].get("ranked", [])
    candidates: list[Candidate] = []
    for index, row in enumerate(sorted(ranked, key=lambda item: item["startSamples"])):
        start = int(row["startSamples"])
        end = int(row["endSamples"])
        if end <= start or end - start < pilot.DEFAULT_CONFIG.useful_samples // 3:
            continue
        score = float(row.get("hardEventScore", 0.0))
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"Invalid hard score at {slug}:{start}: {score}")
        candidates.append(
            Candidate(
                index=index,
                start=start,
                length=end - start,
                hard_score=score,
                rank=int(row.get("rank", 0)),
            )
        )
    if len(candidates) < 1:
        raise ValueError(f"No valid Stage 1 windows for {slug}")
    return slug, report, candidates


def choose_draws(
    rng: np.random.Generator,
    values: Sequence[int],
    count: int,
    *,
    replace_if_needed: bool = False,
) -> list[int]:
    if count <= 0:
        return []
    if count > len(values) and not replace_if_needed:
        raise ValueError(f"Cannot choose {count} values from {len(values)} candidates")
    replace = count > len(values)
    return [int(value) for value in rng.choice(np.asarray(values), size=count, replace=replace)]


def build_song_selection(
    *,
    slug: str,
    member: str,
    candidates: list[Candidate],
    train_windows_per_song: int,
    hard_fraction: float,
    seed: int,
    song_index: int,
) -> SongSelection:
    count = train_windows_per_song
    if count <= 0:
        raise ValueError("train-windows-per-song must be positive")
    if len(candidates) < 1:
        raise ValueError(f"{slug} has no valid windows")
    if not (0.0 < hard_fraction < 1.0):
        raise ValueError("hard-fraction must be between 0 and 1")

    uniform_rng = np.random.default_rng(seed + song_index * 1_000_003)
    uniform_indices = choose_draws(
        uniform_rng,
        list(range(len(candidates))),
        count,
        replace_if_needed=True,
    )
    hard_count = max(1, round(count * hard_fraction))
    uniform_count = count - hard_count
    hard_pool_size = max(1, math.ceil(len(candidates) * 0.25))
    hard_pool_indices = tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: (-candidates[index].hard_score, candidates[index].start),
        )[:hard_pool_size]
    )
    hard_rng = np.random.default_rng(seed + song_index * 1_000_003 + 17)
    # Prefer hard candidates absent from the complete uniform draw so the
    # normal case has exactly six shared draws and two replacements.  Short
    # songs can have fewer than eight valid windows; those draws deliberately
    # use replacement and the report exposes the reduced unique cache set.
    preferred = [index for index in hard_pool_indices if index not in uniform_indices]
    fallback = [index for index in hard_pool_indices if index not in preferred]
    hard_source = preferred + fallback
    if not hard_source:
        raise ValueError(f"Hard pool is empty for {slug}")
    hard_indices = choose_draws(
        hard_rng,
        hard_source,
        hard_count,
        replace_if_needed=True,
    )
    hard_indices = list(uniform_indices[:uniform_count]) + hard_indices
    if len(hard_indices) != count:
        raise AssertionError(
            f"Hard selection has wrong draw count for {slug}: {hard_indices}"
        )
    union_indices = tuple(sorted(set(uniform_indices) | set(hard_indices)))
    return SongSelection(
        slug=slug,
        member=member,
        candidates=tuple(candidates),
        hard_pool_indices=hard_pool_indices,
        uniform_indices=tuple(uniform_indices),
        hard_fraction=hard_fraction,
        hard_indices=tuple(hard_indices),
        union_indices=union_indices,
    )


def selection_json(selection: SongSelection) -> dict[str, Any]:
    return {
        "slug": selection.slug,
        "member": selection.member,
        "candidateCount": len(selection.candidates),
        "hardPoolCount": len(selection.hard_pool_indices),
        "hardPoolCandidateIndices": list(selection.hard_pool_indices),
        "uniformCandidateIndices": list(selection.uniform_indices),
        "hardFraction": selection.hard_fraction,
        "hardCandidateIndices": list(selection.hard_indices),
        # Keep the original key for consumers of the H25 report format.
        "h25CandidateIndices": list(selection.h25_indices),
        "unionCandidateIndices": list(selection.union_indices),
        "candidates": [
            {
                "index": candidate.index,
                "startSamples": candidate.start,
                "length": candidate.length,
                "hardEventScore": candidate.hard_score,
                "stage1Rank": candidate.rank,
            }
            for candidate in selection.candidates
        ],
    }


def build_selections(
    *,
    entries: list[dict[str, Any]],
    event_root: Path,
    train_windows_per_song: int,
    hard_fraction: float,
    seed: int,
) -> tuple[dict[str, SongSelection], dict[str, Any], str]:
    result: dict[str, SongSelection] = {}
    serialized: list[dict[str, Any]] = []
    for song_index, entry in enumerate(entries):
        slug, _, candidates = load_stage1_candidates(event_root, entry)
        selection = build_song_selection(
            slug=slug,
            member=entry["member"],
            candidates=candidates,
            train_windows_per_song=train_windows_per_song,
            hard_fraction=hard_fraction,
            seed=seed,
            song_index=song_index,
        )
        result[slug] = selection
        serialized.append(selection_json(selection))
    payload = {
        "schema": "local-inst3-vr-hard-sampling-selection@1",
        "seed": seed,
        "trainWindowsPerSong": train_windows_per_song,
        "hardFraction": hard_fraction,
        "songs": serialized,
    }
    return result, payload, canonical_sha256(payload)


def cache_file(experiment_root: Path, slug: str) -> Path:
    return experiment_root / "cache" / "train" / f"{slug}.npz"


def cache_metadata_file(experiment_root: Path, slug: str) -> Path:
    return experiment_root / "cache" / "train" / f"{slug}.json"


def student_segment_spec(segment: np.ndarray) -> np.ndarray:
    """Build a student target spectrum from a locally sliced PCM segment."""
    if segment.ndim != 2 or segment.shape[1] != 2:
        raise ValueError(f"Expected stereo segment, got {segment.shape}")
    if segment.shape[0] <= 0:
        raise ValueError("Expected a non-empty segment")
    return pilot.student_window_spec(segment, 0, segment.shape[0])


def expected_cache_metadata(
    *,
    selection: SongSelection,
    entry: dict[str, Any],
    stage1_report_sha256: str,
    checkpoint: Path,
    teacher: Path,
) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "sourceSha256": entry["sourceSha256"],
        "member": entry["member"],
        "slug": selection.slug,
        "stage1ReportSha256": stage1_report_sha256,
        "teacherSha256": file_sha256(teacher),
        "teacherContractId": "uvr_mdxnet_inst_3@2",
        "studentCheckpointSha256": file_sha256(checkpoint),
        "sampleRate": pilot.DEFAULT_CONFIG.sample_rate,
        "numFrames": pilot.DEFAULT_CONFIG.num_frames,
        "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
        "targetAlignment": TARGET_ALIGNMENT,
        "candidateIndices": list(selection.union_indices),
        "starts": [selection.candidates[index].start for index in selection.union_indices],
        "lengths": [selection.candidates[index].length for index in selection.union_indices],
    }


def cache_is_valid(path: Path, metadata_path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if actual.get("contract") != expected:
        return False
    try:
        with np.load(path) as values:
            if values["inputSpec"].shape != values["targetResidualSpec"].shape:
                return False
            if values["inputSpec"].shape[0] != len(expected["candidateIndices"]):
                return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def render_selected_teacher_segments(
    audio: np.ndarray,
    selections: SongSelection,
    session: Any,
    *,
    progress_label: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Render only MDX generation chunks intersecting selected TFC windows.

    MDX chunks are non-overlapping after the model's trim region.  Building a
    chunk directly from zero-extended song PCM is therefore equivalent to
    ``mdx_reference.build_windows`` for the requested indices, while avoiding
    inference over the rest of a multi-minute song.  The selected segments are
    returned in ``selection.union_indices`` order.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected [samples, 2] audio, got {audio.shape}")
    params = pilot.TEACHER_PARAMS
    generation_size = params.generation_size
    chunk_size = params.chunk_size
    required: set[int] = set()
    for candidate_index in selections.union_indices:
        candidate = selections.candidates[candidate_index]
        required.update(
            range(candidate.start // generation_size, (candidate.start + candidate.length - 1) // generation_size + 1)
        )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    blocks: dict[int, np.ndarray] = {}
    started = time.perf_counter()
    for ordinal, chunk_index in enumerate(sorted(required), start=1):
        generation_start = chunk_index * generation_size
        window_start = generation_start - params.trim
        window_end = window_start + chunk_size
        window = np.zeros((2, chunk_size), dtype=np.float32)
        source_start = max(0, window_start)
        source_end = min(audio.shape[0], window_end)
        if source_end > source_start:
            destination_start = source_start - window_start
            window[:, destination_start : destination_start + source_end - source_start] = audio[
                source_start:source_end
            ].T
        model_input = oracle.stft_centered(window[None, ...], params)
        model_output = session.run(
            [output_name],
            {input_name: np.ascontiguousarray(model_input, dtype=np.float32)},
        )[0]
        expected_shape = (1, 4, params.dim_f, params.dim_t)
        if tuple(model_output.shape) != expected_shape:
            raise ValueError(f"Unexpected Inst 3 output shape: {model_output.shape}")
        reconstructed = oracle.istft_centered(model_output.astype(np.float32), params)[0]
        useful = np.ascontiguousarray(
            reconstructed[:, params.trim : -params.trim].T * np.float32(pilot.TEACHER_OUTPUT_SCALE),
            dtype=np.float32,
        )
        if useful.shape != (generation_size, 2):
            raise ValueError(f"Unexpected selected teacher block shape: {useful.shape}")
        blocks[chunk_index] = useful
        if ordinal == 1 or ordinal == len(required) or ordinal % 10 == 0:
            print(f"{progress_label}: selected MDX chunks {ordinal}/{len(required)}", flush=True)

    segments: list[np.ndarray] = []
    for candidate_index in selections.union_indices:
        candidate = selections.candidates[candidate_index]
        segment = np.zeros((candidate.length, 2), dtype=np.float32)
        cursor = 0
        while cursor < candidate.length:
            absolute = candidate.start + cursor
            chunk_index = absolute // generation_size
            chunk_offset = absolute - chunk_index * generation_size
            take = min(candidate.length - cursor, generation_size - chunk_offset)
            segment[cursor : cursor + take] = blocks[chunk_index][chunk_offset : chunk_offset + take]
            cursor += take
        segments.append(segment)
    return segments, {
        "selectedMdxChunkCount": len(required),
        "selectedTfcWindowCount": len(selections.union_indices),
        "totalSeconds": time.perf_counter() - started,
        "mode": "sparse-generation-chunks",
    }


def prepare_train_caches(
    *,
    archive: Path,
    entries: list[dict[str, Any]],
    selections: dict[str, SongSelection],
    event_root: Path,
    experiment_root: Path,
    checkpoint: Path,
    teacher: Path,
    session: Any,
    force: bool,
    threads: int,
) -> dict[str, Path]:
    cache_paths: dict[str, Path] = {}
    work_root = experiment_root / "prep-work"
    for song_index, entry in enumerate(entries, start=1):
        slug = oracle.slugify(entry["fileName"])
        selection = selections[slug]
        report_path = event_root / "songs" / f"{slug}.json"
        expected = expected_cache_metadata(
            selection=selection,
            entry=entry,
            stage1_report_sha256=file_sha256(report_path),
            checkpoint=checkpoint,
            teacher=teacher,
        )
        output_path = cache_file(experiment_root, slug)
        metadata_path = cache_metadata_file(experiment_root, slug)
        cache_paths[slug] = output_path
        if not force and cache_is_valid(output_path, metadata_path, expected):
            print(f"cache {song_index}/{len(entries)} reuse {slug}", flush=True)
            continue

        print(f"cache {song_index}/{len(entries)} render {entry['member']}", flush=True)
        shutil.rmtree(work_root, ignore_errors=True)
        work_root.mkdir(parents=True, exist_ok=True)
        try:
            song = oracle.decode_song(
                archive,
                entry,
                work_root / "raw",
                work_root / "decoded",
                force_extract=False,
                force_decode=False,
                keep_decoded_wav=False,
            )
            teacher_segments, render_report = render_selected_teacher_segments(
                song.mixture_gt,
                selection,
                session,
                progress_label=f"train/{slug}",
            )
            input_specs: list[np.ndarray] = []
            target_specs: list[np.ndarray] = []
            starts: list[int] = []
            lengths: list[int] = []
            scores: list[float] = []
            for segment_index, candidate_index in enumerate(selection.union_indices):
                candidate = selection.candidates[candidate_index]
                teacher_instrumental = teacher_segments[segment_index]
                teacher_residual = song.mixture_gt[
                    candidate.start : candidate.start + candidate.length
                ] - teacher_instrumental
                input_specs.append(
                    pilot.student_window_spec(song.mixture_gt, candidate.start, candidate.length)
                )
                target_specs.append(student_segment_spec(teacher_residual))
                starts.append(candidate.start)
                lengths.append(candidate.length)
                scores.append(candidate.hard_score)
            input_array = np.ascontiguousarray(np.stack(input_specs), dtype=np.float32)
            target_array = np.ascontiguousarray(np.stack(target_specs), dtype=np.float32)
            if input_array.shape != target_array.shape:
                raise ValueError(f"Training cache shape mismatch for {slug}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(output_path.name + ".tmp")
            np.savez_compressed(
                temporary,
                inputSpec=input_array,
                targetResidualSpec=target_array,
                starts=np.asarray(starts, dtype=np.int64),
                lengths=np.asarray(lengths, dtype=np.int64),
                hardScores=np.asarray(scores, dtype=np.float32),
            )
            # numpy appends .npz when the temporary name does not end in .npz.
            temporary_npz = temporary if temporary.suffix == ".npz" else Path(str(temporary) + ".npz")
            temporary_npz.replace(output_path)
            metadata = {
                "contract": expected,
                "render": {
                    "providers": session.get_providers(),
                    "windowCount": render_report.get("selectedMdxChunkCount"),
                    "selectedTfcWindowCount": render_report.get("selectedTfcWindowCount"),
                    "mode": render_report.get("mode"),
                    "timing": render_report.get("timing"),
                    "totalSeconds": render_report.get("totalSeconds"),
                },
                "cache": {
                    "file": str(output_path.resolve()),
                    "bytes": output_path.stat().st_size,
                    "sha256": file_sha256(output_path),
                },
                "selected": {
                    "uniformCandidateIndices": list(selection.uniform_indices),
                    "hardFraction": selection.hard_fraction,
                    "hardCandidateIndices": list(selection.hard_indices),
                    "h25CandidateIndices": list(selection.h25_indices),
                    "unionCandidateIndices": list(selection.union_indices),
                },
            }
            json_write(metadata_path, metadata)
            del teacher_segments, input_specs, target_specs
            del input_array, target_array, song
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
    return cache_paths


def prepare_eval_caches(
    *,
    archive: Path,
    entries: list[dict[str, Any]],
    eval_root: Path,
    session: Any,
    contract_info: dict[str, Any],
    runner_revision: str,
    force: bool,
) -> None:
    """Bring the song-disjoint holdout cache in line with the frozen manifest."""
    for index, entry in enumerate(entries, start=1):
        slug = oracle.slugify(entry["fileName"])
        decoded_dir = eval_root / "decoded" / entry["role"] / slug
        teacher_dir = eval_root / "teacher" / entry["role"] / slug
        decoded_metadata = decoded_dir / "decoded.json"
        decoded_cache = decoded_dir / "decoded.npz"
        cache_valid = False
        if decoded_metadata.is_file() and decoded_cache.is_file():
            try:
                decoded_value = json.loads(decoded_metadata.read_text(encoding="utf-8"))
                cache_valid = decoded_value.get("sourceSha256") == entry["sourceSha256"]
            except (OSError, json.JSONDecodeError):
                cache_valid = False
        teacher_valid = False
        if cache_valid:
            for metadata_path in sorted(teacher_dir.glob("mixture-gt-*.json")):
                try:
                    value = json.loads(metadata_path.read_text(encoding="utf-8"))
                    teacher_valid = (
                        value.get("song", {}).get("sourceSha256") == entry["sourceSha256"]
                        and value.get("contract", {}).get("contractId") == contract_info["contractId"]
                        and Path(value["output"]["npz"]["file"]).is_file()
                    )
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    teacher_valid = False
                if teacher_valid:
                    break
        if cache_valid and teacher_valid and not force:
            print(f"eval cache {index}/{len(entries)} reuse {slug}", flush=True)
            continue

        if decoded_dir.exists() and not cache_valid:
            shutil.rmtree(decoded_dir, ignore_errors=True)
        if teacher_dir.exists() and not teacher_valid:
            shutil.rmtree(teacher_dir, ignore_errors=True)
        print(f"eval cache {index}/{len(entries)} render {entry['member']}", flush=True)
        song = oracle.decode_song(
            archive,
            entry,
            eval_root / "raw",
            eval_root / "decoded",
            force_extract=False,
            force_decode=False,
            keep_decoded_wav=False,
        )
        oracle.render_or_load_cached(
            eval_root,
            song,
            "mixture-gt",
            song.mixture_gt,
            session,
            contract_info,
            runner_revision,
            force=True,
        )
        del song
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CacheStore:
    def __init__(self, paths: dict[str, Path], max_open: int = 4) -> None:
        self.paths = paths
        self.max_open = max_open
        self._open: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _load(self, slug: str) -> dict[str, np.ndarray]:
        arrays = self._open.get(slug)
        if arrays is not None:
            self._open.move_to_end(slug)
            return arrays
        path = self.paths[slug]
        with np.load(path) as values:
            arrays = {
                "inputSpec": np.ascontiguousarray(values["inputSpec"], dtype=np.float32),
                "targetResidualSpec": np.ascontiguousarray(
                    values["targetResidualSpec"], dtype=np.float32
                ),
            }
        self._open[slug] = arrays
        self._open.move_to_end(slug)
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return arrays

    def batch(self, items: Sequence[ScheduleItem]) -> tuple[np.ndarray, np.ndarray]:
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for item in items:
            arrays = self._load(item.slug)
            inputs.append(arrays["inputSpec"][item.cache_index])
            targets.append(arrays["targetResidualSpec"][item.cache_index])
        return np.stack(inputs), np.stack(targets)

    def preload(self) -> None:
        """Decompress each compact cache once before the long training loop."""
        total = len(self.paths)
        for index, slug in enumerate(sorted(self.paths), start=1):
            self._load(slug)
            if index == 1 or index == total or index % 10 == 0:
                print(f"preload training cache {index}/{total}", flush=True)


def build_training_schedule(
    selections: dict[str, SongSelection],
    variant: str,
    passes: int,
    seed: int,
    hard_variant: str = "V-R-H25",
) -> list[ScheduleItem]:
    if variant not in ("V-R-U", hard_variant) or hard_variant not in (
        "V-R-H25",
        "V-R-H50",
    ):
        raise ValueError(f"Unknown variant: {variant}")
    schedule: list[ScheduleItem] = []
    slugs = sorted(selections)
    for pass_index in range(passes):
        # Use the same permutation for both cells.  Only the selected
        # candidate changes; batch/order effects must not become a second
        # variable.
        rng = np.random.default_rng(seed + pass_index * 1_000_003)
        pass_items: list[ScheduleItem] = []
        for song_index, slug in enumerate(slugs):
            selection = selections[slug]
            candidate_indices = (
                selection.uniform_indices
                if variant == "V-R-U"
                else selection.hard_indices
            )
            cache_map = selection.cache_index_by_candidate
            pass_items.extend(
                ScheduleItem(slug, cache_map[candidate_index])
                for candidate_index in candidate_indices
            )
        schedule.extend(pass_items[int(index)] for index in rng.permutation(len(pass_items)))
    return schedule


def schedule_json(schedule: Sequence[ScheduleItem]) -> list[dict[str, Any]]:
    return [{"slug": item.slug, "cacheIndex": item.cache_index} for item in schedule]


def summarize_schedule(
    selections: dict[str, SongSelection],
    schedules: dict[str, list[ScheduleItem]],
    passes: int,
    batch_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant, schedule in schedules.items():
        if len(schedule) % batch_size != 0:
            raise ValueError(f"{variant} schedule is not divisible by batch size")
        per_song: dict[str, int] = {}
        for item in schedule[: len(schedule) // passes]:
            per_song[item.slug] = per_song.get(item.slug, 0) + 1
        result[variant] = {
            "recordCount": len(schedule),
            "recordsPerPass": len(schedule) // passes,
            "updates": len(schedule) // batch_size,
            "recordsPerSongPerPass": per_song,
            "scheduleSha256": canonical_sha256(schedule_json(schedule)),
        }
    return result


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
        candidate = (step, path, payload)
        if best is None or step > best[0]:
            best = candidate
    return None if best is None else (best[1], best[2])


def train_variant(
    *,
    variant: str,
    checkpoint: Path,
    cache_paths: dict[str, Path],
    selections: dict[str, SongSelection],
    schedule: list[ScheduleItem],
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
    milestone_updates = {pass_count * updates_per_pass: pass_count for pass_count in milestones}
    run_root.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model, checkpoint_metadata = pilot.make_model(checkpoint, device)
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
            print(json.dumps({"event": "resume", "variant": variant, "step": current_update, "file": str(path)}), flush=True)

    # A global shuffle makes a small LRU repeatedly evict songs.  The selected
    # spectra are bounded (roughly a few hundred MiB for all 80 songs), so
    # retain every cache in RAM and pay decompression only once.
    store = CacheStore(cache_paths, max_open=len(cache_paths))
    store.preload()
    started = time.perf_counter()
    milestone_files: dict[str, dict[str, Any]] = {}
    checkpoint_files: dict[str, dict[str, Any]] = {}

    def discover_milestone_files() -> None:
        for update in milestone_updates:
            path = run_root / f"step-{update}.pt"
            if path.is_file():
                metadata = {
                    "file": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                milestone_files[str(update)] = metadata
                checkpoint_files[str(update)] = metadata

    discover_milestone_files()

    def save_checkpoint(step: int, status: str) -> dict[str, Any]:
        path = run_root / f"step-{step}.pt"
        payload = {
            "format": "local-inst3-vr-hard-sampling-checkpoint@1",
            "status": status,
            "variant": variant,
            "runContractId": contract_id,
            "step": step,
            "passes": passes,
            "recordsPerPass": records_per_pass,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "seed": seed,
            "stateDict": cpu_tree(model.state_dict()),
            "optimizerStateDict": cpu_tree(optimizer.state_dict()),
            "history": history,
            "resumeCount": resume_count,
            "elapsedSeconds": time.perf_counter() - started,
            "checkpointSource": checkpoint_metadata["checkpoint"],
        }
        metadata = atomic_torch_save(path, payload)
        checkpoint_files[str(step)] = metadata
        if step in milestone_updates:
            milestone_files[str(step)] = metadata
        return metadata

    if current_update == 0 and 0 in milestone_updates:
        save_checkpoint(0, "initial")

    while current_update < total_updates:
        begin = current_update * batch_size
        batch_items = schedule[begin : begin + batch_size]
        if len(batch_items) != batch_size:
            raise AssertionError("Incomplete training batch")
        input_array, target_array = store.batch(batch_items)
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted_residual = model(input_tensor)
        loss = F.l1_loss(predicted_residual, target_tensor)
        loss.backward()
        # Avoid the fused multi-tensor CUDA path here.  It is not needed for
        # this small model and has produced delayed illegal-memory errors on
        # long Windows CUDA runs even when the preceding backward is valid.
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
                        "variant": variant,
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
        # A resumed run may have loaded the final state without recreating the
        # in-memory milestone map.  Re-scan after training so evaluation sees
        # every requested checkpoint, including pass 25.
        discover_milestone_files()
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
        "history": history,
        "milestoneCheckpoints": milestone_files,
        "checkpointFiles": checkpoint_files,
        "elapsedSeconds": time.perf_counter() - started,
        "resumeCount": resume_count,
    }


def load_eval_song(eval_root: Path, entry: dict[str, Any]) -> pilot.SongBundle:
    role = entry["role"]
    slug = oracle.slugify(entry["fileName"])
    decoded_dir = eval_root / "decoded" / role / slug
    decoded_path = decoded_dir / "decoded.npz"
    metadata_path = decoded_dir / "decoded.json"
    if not decoded_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing evaluation decode cache for {role}/{slug}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(decoded_path) as values:
        arrays = {
            name: np.ascontiguousarray(values[name], dtype=np.float32)
            for name in ("mixtureGt", "vocals", "instrumentalGt")
        }
    teacher_dir = eval_root / "teacher" / role / slug
    metadata_paths = sorted(teacher_dir.glob("mixture-gt-*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"Missing evaluation Inst 3 cache for {slug}")
    teacher_metadata_path = metadata_paths[-1]
    teacher_metadata = json.loads(teacher_metadata_path.read_text(encoding="utf-8"))
    teacher_npz = Path(teacher_metadata["output"]["npz"]["file"])
    if not teacher_npz.is_file():
        teacher_npz = teacher_metadata_path.with_suffix(".npz")
    with np.load(teacher_npz) as values:
        teacher_instrumental = np.ascontiguousarray(values["instrumental"], dtype=np.float32)
    if teacher_instrumental.shape != arrays["mixtureGt"].shape:
        raise ValueError(f"Evaluation teacher length mismatch for {slug}")
    return pilot.SongBundle(
        role=role,
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
        teacher_instrumental=teacher_instrumental,
        teacher_vocals=np.ascontiguousarray(arrays["mixtureGt"] - teacher_instrumental, dtype=np.float32),
        teacher_metadata=teacher_metadata,
    )


def select_eval_records(song: pilot.SongBundle, count: int, seed: int) -> list[pilot.WindowRecord]:
    if count <= 0:
        raise ValueError("eval-windows-per-song must be positive")
    useful = pilot.DEFAULT_CONFIG.useful_samples
    candidates: list[pilot.WindowRecord] = []
    for start in range(0, song.mixture_gt.shape[0], useful):
        length = min(useful, song.mixture_gt.shape[0] - start)
        if length < useful // 3:
            continue
        mixture = song.mixture_gt[start : start + length]
        vocals = song.vocals[start : start + length]
        candidates.append(
            pilot.WindowRecord(
                song,
                start,
                length,
                float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2))),
                float(np.sqrt(np.mean(vocals.astype(np.float64) ** 2))),
            )
        )
    if len(candidates) <= count:
        return candidates
    selected: set[int] = set(int(value) for value in np.linspace(0, len(candidates) - 1, count, dtype=int))
    if len(selected) < count:
        rng = np.random.default_rng(seed)
        available = [index for index in range(len(candidates)) if index not in selected]
        selected.update(int(value) for value in rng.choice(available, size=count - len(selected), replace=False))
    return [candidates[index] for index in sorted(selected)]


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1.0e-12))


def block_projection_arrays(
    teacher_removed: np.ndarray,
    miss: np.ndarray,
    sample_rate: int,
    milliseconds: int,
) -> dict[str, np.ndarray]:
    block_samples = max(1, round(sample_rate * milliseconds / 1000.0))
    count = max(1, math.ceil(teacher_removed.shape[0] / block_samples))
    padded_teacher = np.pad(
        teacher_removed.astype(np.float64, copy=False),
        ((0, count * block_samples - teacher_removed.shape[0]), (0, 0)),
    ).reshape(count, block_samples, 2)
    padded_miss = np.pad(
        miss.astype(np.float64, copy=False),
        ((0, count * block_samples - miss.shape[0]), (0, 0)),
    ).reshape(count, block_samples, 2)
    teacher_power = np.sum(padded_teacher * padded_teacher, axis=(1, 2))
    miss_power = np.sum(padded_miss * padded_miss, axis=(1, 2))
    teacher_rms = np.sqrt(teacher_power / float(block_samples * 2))
    miss_rms = np.sqrt(miss_power / float(block_samples * 2))
    dots = np.sum(padded_miss * padded_teacher, axis=(1, 2))
    coefficient = np.divide(dots, teacher_power, out=np.zeros_like(dots), where=teacher_power > 1.0e-20)
    positive = np.maximum(coefficient, 0.0) * teacher_rms
    active = teacher_rms >= 10.0 ** (-60.0 / 20.0)
    return {
        "teacherRms": teacher_rms,
        "missRms": miss_rms,
        "positiveProjectionRms": positive,
        "active": active,
        "blockSamples": np.asarray([block_samples], dtype=np.int64),
    }


class PowerAccumulator:
    def __init__(self) -> None:
        self.reference_power = 0.0
        self.error_power = 0.0
        self.dot = 0.0
        self.candidate_power = 0.0
        self.count = 0

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        reference64 = reference.astype(np.float64, copy=False)
        candidate64 = candidate.astype(np.float64, copy=False)
        error = candidate64 - reference64
        self.reference_power += float(np.sum(reference64 * reference64))
        self.error_power += float(np.sum(error * error))
        self.dot += float(np.sum(reference64 * candidate64))
        self.candidate_power += float(np.sum(candidate64 * candidate64))
        self.count += int(reference.size)

    def snr_db(self) -> float:
        return 10.0 * math.log10(max(self.reference_power, 1.0e-30) / max(self.error_power, 1.0e-30))

    def projection_db(self, basis_power: float | None = None) -> float:
        denominator = self.reference_power if basis_power is None else basis_power
        return 20.0 * math.log10(max(abs(self.dot), 1.0e-30) / max(denominator, 1.0e-30))


def metric_result(
    *,
    vocals: PowerAccumulator,
    teacher_vocals: PowerAccumulator,
    instrumental: PowerAccumulator,
    teacher: PowerAccumulator,
    low_instrumental: PowerAccumulator,
    vocal_basis_power: float,
    accompaniment_vocal_dot: float,
    sample_count: int,
) -> dict[str, Any]:
    return {
        "samples": sample_count,
        "instrumentalSdrDb": instrumental.snr_db(),
        "teacherInstrumentalSdrDb": teacher.snr_db(),
        "vocalSdrDb": vocals.snr_db(),
        "teacherVocalSdrDb": teacher_vocals.snr_db(),
        "accompanimentErrorRmsDbfs": dbfs(math.sqrt(instrumental.error_power / max(instrumental.count, 1))),
        "accompanimentVocalProjectionDb": 20.0 * math.log10(
            max(abs(accompaniment_vocal_dot) / max(vocal_basis_power, 1.0e-30), 1.0e-12)
        ),
        "lowVocalInstrumentalSdrDb": low_instrumental.snr_db(),
        "reconstructionMaxAbsError": 0.0,
    }


def evaluate_model_state(
    *,
    name: str,
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    eval_root: Path,
    eval_windows_per_song: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_event_values: dict[int, dict[str, list[np.ndarray]]] = {
        milliseconds: {
            "missRms": [],
            "positiveProjectionRms": [],
            "active": [],
        }
        for milliseconds in (50, 100, 200)
    }
    per_song: dict[str, Any] = {}
    aggregate_accumulators = {
        "vocals": PowerAccumulator(),
        "teacherVocals": PowerAccumulator(),
        "instrumental": PowerAccumulator(),
        "teacher": PowerAccumulator(),
        "lowInstrumental": PowerAccumulator(),
    }
    vocal_basis_power = 0.0
    sample_count = 0
    derivative_values: list[float] = []
    teacher_derivative_values: list[float] = []
    derivative_excess_values: list[float] = []
    clip_count = 0
    nonfinite_count = 0
    accompaniment_vocal_dot = 0.0
    started = time.perf_counter()

    with torch.inference_mode():
        for song_index, entry in enumerate(entries):
            song = load_eval_song(eval_root, entry)
            records = select_eval_records(song, eval_windows_per_song, seed + song_index)
            song_events: dict[int, dict[str, list[np.ndarray]]] = {
                milliseconds: {"missRms": [], "positiveProjectionRms": [], "active": []}
                for milliseconds in (50, 100, 200)
            }
            song_accumulators = {
                "vocals": PowerAccumulator(),
                "teacherVocals": PowerAccumulator(),
                "instrumental": PowerAccumulator(),
                "teacher": PowerAccumulator(),
                "lowInstrumental": PowerAccumulator(),
            }
            song_vocal_basis_power = 0.0
            song_samples = 0
            song_derivatives: list[float] = []
            song_teacher_derivatives: list[float] = []
            song_excess: list[float] = []
            song_clip_count = 0
            song_nonfinite_count = 0
            song_accompaniment_vocal_dot = 0.0
            for begin in range(0, len(records), 4):
                batch = records[begin : begin + 4]
                input_specs = np.stack(
                    [pilot.student_window_spec(song.mixture_gt, record.start, record.length) for record in batch]
                )
                input_tensor = torch.from_numpy(input_specs).to(device)
                predicted_residual_specs = model(input_tensor).detach().cpu().numpy()
                for offset, record in enumerate(batch):
                    reconstructed_residual = pilot.student_istft_centered(
                        predicted_residual_specs[offset : offset + 1]
                    )
                    trim = pilot.DEFAULT_CONFIG.trim_samples
                    residual = np.ascontiguousarray(
                        reconstructed_residual[trim : trim + record.length], dtype=np.float32
                    )
                    start = record.start
                    end = start + record.length
                    mixture = song.mixture_gt[start:end]
                    vocals = song.vocals[start:end]
                    true_instrumental = song.instrumental[start:end]
                    teacher_instrumental = song.teacher_instrumental[start:end]
                    predicted_instrumental = np.ascontiguousarray(mixture - residual, dtype=np.float32)
                    predicted_vocals = np.ascontiguousarray(mixture - predicted_instrumental, dtype=np.float32)
                    if not np.isfinite(predicted_instrumental).all():
                        song_nonfinite_count += 1
                        continue
                    song_clip_count += int(np.count_nonzero(np.abs(predicted_instrumental) >= 1.0))
                    for key, reference, candidate in (
                        ("vocals", vocals, predicted_vocals),
                        ("teacherVocals", vocals, mixture - teacher_instrumental),
                        ("instrumental", true_instrumental, predicted_instrumental),
                        ("teacher", true_instrumental, teacher_instrumental),
                    ):
                        song_accumulators[key].update(reference, candidate)
                        aggregate_accumulators[key].update(reference, candidate)
                    low_mask = pilot.low_vocal_mask(vocals)
                    song_accumulators["lowInstrumental"].update(
                        true_instrumental[low_mask], predicted_instrumental[low_mask]
                    )
                    aggregate_accumulators["lowInstrumental"].update(
                        true_instrumental[low_mask], predicted_instrumental[low_mask]
                    )
                    song_vocal_basis_power += float(np.sum(vocals.astype(np.float64) ** 2))
                    vocal_basis_power += float(np.sum(vocals.astype(np.float64) ** 2))
                    segment_error = predicted_instrumental.astype(np.float64) - true_instrumental.astype(np.float64)
                    segment_vocals = vocals.astype(np.float64)
                    segment_vocal_dot = float(np.sum(segment_error * segment_vocals))
                    song_accompaniment_vocal_dot += segment_vocal_dot
                    accompaniment_vocal_dot += segment_vocal_dot
                    song_samples += record.length
                    sample_count += record.length
                    teacher_removed = mixture - teacher_instrumental
                    miss = predicted_instrumental - teacher_instrumental
                    for milliseconds in (50, 100, 200):
                        values = block_projection_arrays(teacher_removed, miss, song.sample_rate, milliseconds)
                        for key in ("missRms", "positiveProjectionRms", "active"):
                            song_events[milliseconds][key].append(values[key])
                            all_event_values[milliseconds][key].append(values[key])
                    predicted_diff = np.diff(predicted_instrumental, axis=0)
                    teacher_diff = np.diff(teacher_instrumental, axis=0)
                    predicted_derivative = math.sqrt(float(np.mean(predicted_diff.astype(np.float64) ** 2)))
                    teacher_derivative = math.sqrt(float(np.mean(teacher_diff.astype(np.float64) ** 2)))
                    song_derivatives.append(dbfs(predicted_derivative))
                    song_teacher_derivatives.append(dbfs(teacher_derivative))
                    song_excess.append(dbfs(predicted_derivative) - dbfs(teacher_derivative))
                    derivative_values.append(dbfs(predicted_derivative))
                    teacher_derivative_values.append(dbfs(teacher_derivative))
                    derivative_excess_values.append(dbfs(predicted_derivative) - dbfs(teacher_derivative))
            song_metric = metric_result(
                vocals=song_accumulators["vocals"],
                teacher_vocals=song_accumulators["teacherVocals"],
                instrumental=song_accumulators["instrumental"],
                teacher=song_accumulators["teacher"],
                low_instrumental=song_accumulators["lowInstrumental"],
                vocal_basis_power=song_vocal_basis_power,
                accompaniment_vocal_dot=song_accompaniment_vocal_dot,
                sample_count=song_samples,
            )
            song_metric["eventMetrics"] = summarize_event_values(song_events)
            song_metric["mechanicalArtifactProxy"] = mechanical_metrics(
                song_derivatives, song_teacher_derivatives, song_excess, song_clip_count, song_nonfinite_count
            )
            per_song[song.slug] = {
                "role": song.role,
                "windowStarts": [record.start for record in records],
                "metrics": song_metric,
            }
            del song
            gc.collect()

    aggregate_metrics = metric_result(
        vocals=aggregate_accumulators["vocals"],
        teacher_vocals=aggregate_accumulators["teacherVocals"],
        instrumental=aggregate_accumulators["instrumental"],
        teacher=aggregate_accumulators["teacher"],
        low_instrumental=aggregate_accumulators["lowInstrumental"],
        vocal_basis_power=vocal_basis_power,
        accompaniment_vocal_dot=accompaniment_vocal_dot,
        sample_count=sample_count,
    )
    aggregate_metrics["eventMetrics"] = summarize_event_values(all_event_values)
    aggregate_metrics["mechanicalArtifactProxy"] = mechanical_metrics(
        derivative_values,
        teacher_derivative_values,
        derivative_excess_values,
        clip_count,
        nonfinite_count,
    )
    return {
        "variant": name,
        "aggregate": aggregate_metrics,
        "perSong": per_song,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def summarize_event_values(values: dict[int, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for milliseconds, arrays in values.items():
        miss = np.concatenate(arrays["missRms"]) if arrays["missRms"] else np.zeros(0, dtype=np.float64)
        positive = (
            np.concatenate(arrays["positiveProjectionRms"])
            if arrays["positiveProjectionRms"]
            else np.zeros(0, dtype=np.float64)
        )
        active = np.concatenate(arrays["active"]) if arrays["active"] else np.zeros(0, dtype=bool)
        active_positive = positive[active]
        active_miss = miss[active]
        result[str(milliseconds)] = {
            "blockCount": int(miss.size),
            "activeBlockCount": int(np.count_nonzero(active)),
            "missRmsP95Dbfs": dbfs(float(np.percentile(active_miss, 95.0))) if active_miss.size else -240.0,
            "missRmsMaxDbfs": dbfs(float(np.max(active_miss))) if active_miss.size else -240.0,
            "positiveProjectionRmsP95Dbfs": dbfs(float(np.percentile(active_positive, 95.0))) if active_positive.size else -240.0,
            "positiveProjectionRmsMaxDbfs": dbfs(float(np.max(active_positive))) if active_positive.size else -240.0,
            "activePositiveAboveMinus35Dbfs": int(np.count_nonzero(active_positive >= 10.0 ** (-35.0 / 20.0))),
            "activePositiveAboveMinus30Dbfs": int(np.count_nonzero(active_positive >= 10.0 ** (-30.0 / 20.0))),
            "hotspots": [
                {"rank": rank + 1, "positiveProjectionRmsDbfs": dbfs(float(value))}
                for rank, value in enumerate(np.sort(active_positive)[::-1][:12])
            ],
        }
    return result


def mechanical_metrics(
    derivative: Sequence[float],
    teacher_derivative: Sequence[float],
    excess: Sequence[float],
    clip_count: int,
    nonfinite_count: int,
) -> dict[str, Any]:
    return {
        "windowCount": len(derivative),
        "predictedDerivativeP95Dbfs": float(np.percentile(derivative, 95.0)) if derivative else -240.0,
        "teacherDerivativeP95Dbfs": float(np.percentile(teacher_derivative, 95.0)) if teacher_derivative else -240.0,
        "derivativeExcessP95Db": float(np.percentile(excess, 95.0)) if excess else 0.0,
        "derivativeExcessMaxDb": float(np.max(excess)) if excess else 0.0,
        "clipSampleCount": int(clip_count),
        "nonfiniteWindowCount": int(nonfinite_count),
        "definition": "within-window first-difference energy; diagnostic artifact proxy, not a perceptual score",
    }


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "instrumentalSdrDb",
        "teacherInstrumentalSdrDb",
        "vocalSdrDb",
        "accompanimentVocalProjectionDb",
    )
    result = {
        key: float(candidate["aggregate"][key] - baseline["aggregate"][key])
        for key in keys
    }
    for milliseconds in (50, 100, 200):
        candidate_events = candidate["aggregate"]["eventMetrics"][str(milliseconds)]
        baseline_events = baseline["aggregate"]["eventMetrics"][str(milliseconds)]
        for key in ("missRmsP95Dbfs", "missRmsMaxDbfs", "positiveProjectionRmsP95Dbfs", "positiveProjectionRmsMaxDbfs"):
            result[f"{milliseconds}ms.{key}"] = float(candidate_events[key] - baseline_events[key])
    return result


def evaluate_checkpoints(
    *,
    checkpoint: Path,
    experiment_root: Path,
    train_results: dict[str, Any],
    eval_entries: list[dict[str, Any]],
    eval_root: Path,
    eval_windows_per_song: int,
    seed: int,
    device: torch.device,
    milestones: tuple[int, ...],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    state_paths: list[tuple[str, Path | None]] = [("initial", None)]
    for variant, training in train_results.items():
        for pass_count in milestones:
            if pass_count == 0:
                continue
            metadata = training["milestoneCheckpoints"].get(str(pass_count * training["updatesPerPass"]))
            if metadata is None:
                raise FileNotFoundError(f"Missing {variant} milestone {pass_count}")
            state_paths.append((f"{variant}@pass-{pass_count}", Path(metadata["file"])))

    baseline_model, _ = pilot.make_model(checkpoint, device)
    sweep.freeze_batchnorm_running_statistics(baseline_model)
    results["initial"] = evaluate_model_state(
        name="initial",
        model=baseline_model,
        entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=eval_windows_per_song,
        seed=seed,
        device=device,
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, state_path in state_paths[1:]:
        model, _ = pilot.make_model(checkpoint, device)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["stateDict"], strict=True)
        sweep.freeze_batchnorm_running_statistics(model)
        results[name] = evaluate_model_state(
            name=name,
            model=model,
            entries=eval_entries,
            eval_root=eval_root,
            eval_windows_per_song=eval_windows_per_song,
            seed=seed,
            device=device,
        )
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline = results["initial"]
    for name, value in results.items():
        if name != "initial":
            value["deltaVsInitial"] = metric_delta(value, baseline)
    return results


def validate_args(args: argparse.Namespace, milestones: tuple[int, ...]) -> None:
    if args.train_windows_per_song <= 0 or args.batch_size <= 0:
        raise ValueError("window count and batch size must be positive")
    if not (0.0 < args.hard_fraction < 1.0):
        raise ValueError("hard-fraction must be between 0 and 1")
    if args.passes <= 0 or args.learning_rate <= 0 or args.threads <= 0 or args.state_interval <= 0:
        raise ValueError("passes, learning rate, and threads must be positive")
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
    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    event_root = args.event_root.resolve()
    eval_root = args.eval_root.resolve()
    experiment_root = args.experiment_root.resolve()
    checkpoint = args.checkpoint.resolve()
    teacher = args.teacher.resolve()
    contract = args.contract.resolve()
    teacher_tflite = args.teacher_tflite.resolve()
    variants = ("V-R-U", args.hard_variant)
    manifest = load_manifest(manifest_path)
    train_entries = load_train_entries(manifest, args.max_songs)
    eval_entries = sorted(
        [entry for entry in manifest["entries"] if entry.get("role") in {"calibration", "internal-test"}],
        key=lambda entry: (entry["role"], entry["member"]),
    )
    if args.max_eval_songs is not None:
        eval_entries = eval_entries[: args.max_eval_songs]
    if not checkpoint.is_file() or not teacher.is_file():
        raise FileNotFoundError("Student checkpoint or Inst 3 teacher is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    experiment_root.mkdir(parents=True, exist_ok=True)
    selections, selection_payload, selection_sha256 = build_selections(
        entries=train_entries,
        event_root=event_root,
        train_windows_per_song=args.train_windows_per_song,
        hard_fraction=args.hard_fraction,
        seed=args.seed,
    )
    json_write(experiment_root / "selection.json", selection_payload)
    contract_info = pilot.verify_teacher_contract(contract, teacher, teacher_tflite)
    runner_revision = file_sha256(Path(__file__).resolve())
    teacher_session, teacher_providers = pilot.make_teacher_session(
        teacher,
        args.threads,
        args.require_teacher_cuda,
    )
    try:
        cache_paths = prepare_train_caches(
            archive=archive,
            entries=train_entries,
            selections=selections,
            event_root=event_root,
            experiment_root=experiment_root,
            checkpoint=checkpoint,
            teacher=teacher,
            session=teacher_session,
            force=args.force_prep,
            threads=args.threads,
        )
        prepare_eval_caches(
            archive=archive,
            entries=eval_entries,
            eval_root=eval_root,
            session=teacher_session,
            contract_info=contract_info,
            runner_revision=runner_revision,
            force=args.force_prep,
        )
    finally:
        del teacher_session
        if device.type == "cuda":
            torch.cuda.empty_cache()

    schedules = {
        variant: build_training_schedule(
            selections,
            variant,
            args.passes,
            args.seed,
            hard_variant=args.hard_variant,
        )
        for variant in variants
    }
    schedule_summary = summarize_schedule(selections, schedules, args.passes, args.batch_size)
    if any(value["recordsPerPass"] != len(train_entries) * args.train_windows_per_song for value in schedule_summary.values()):
        raise AssertionError("Unexpected records per pass")
    run_contract_base = {
        "schema": RUN_SCHEMA,
        "selectionSha256": selection_sha256,
        "manifestSha256": file_sha256(manifest_path),
        "eventReportSha256": file_sha256(event_root / "reports" / "inst3-vr-hard-events-report.json"),
        "checkpointSha256": file_sha256(checkpoint),
        "teacherSha256": file_sha256(teacher),
        "teacherContractId": contract_info["contractId"],
        "trainSongCount": len(train_entries),
        "trainWindowsPerSong": args.train_windows_per_song,
        "hardFraction": args.hard_fraction,
        "hardVariant": args.hard_variant,
        "passes": args.passes,
        "milestones": list(milestones),
        "batchSize": args.batch_size,
        "stateInterval": args.state_interval,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "cacheSchema": CACHE_SCHEMA,
        "targetAlignment": TARGET_ALIGNMENT,
        "studentSemantic": "residual-vocals",
        "targetSemantic": "mixtureGt - Inst3Instrumental",
    }
    train_results: dict[str, Any] = {}
    for variant in variants:
        variant_contract = dict(run_contract_base)
        variant_contract["variant"] = variant
        variant_contract["scheduleSha256"] = schedule_summary[variant]["scheduleSha256"]
        contract_id = canonical_sha256(variant_contract)
        train_results[variant] = train_variant(
            variant=variant,
            checkpoint=checkpoint,
            cache_paths=cache_paths,
            selections=selections,
            schedule=schedules[variant],
            run_root=experiment_root / "runs" / safe_name(variant),
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

    evaluation = evaluate_checkpoints(
        checkpoint=checkpoint,
        experiment_root=experiment_root,
        train_results=train_results,
        eval_entries=eval_entries,
        eval_root=eval_root,
        eval_windows_per_song=args.eval_windows_per_song,
        seed=args.seed,
        device=device,
        milestones=milestones,
    )
    report = {
        "schema": RUN_SCHEMA,
        "status": "completed",
        "experimentId": (
            "inst3-vr-hard-sampling@1"
            if args.hard_variant == "V-R-H25"
            else "inst3-vr-hard-sampling-h50@1"
        ),
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "teacher weight and derived outputs remain local; not published",
        },
        "manifest": {
            "file": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifestId": manifest["manifestId"],
            "trainSongCount": len(train_entries),
            "evaluationSongCount": len(eval_entries),
            "officialFinalTestUsed": False,
        },
        "contract": {
            **run_contract_base,
            "selectionSha256": selection_sha256,
            "teacher": contract_info,
            "teacherProviders": teacher_providers,
        },
        "selection": selection_payload,
        "schedule": schedule_summary,
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
            "sampling": (
                "Fixed deterministic per-song eight-window set repeated each pass; "
                f"{args.hard_variant} uses {args.hard_fraction:.2f} hard-event draws "
                "from the song-local top-25-percent pool and the remainder from "
                "the shared uniform draw."
            ),
            "hardScore": "Stage 1 positive projection score; diagnostic ranking aid, not a vocal ground-truth label.",
            "evaluation": "Calibration/internal-test only; event metrics are computed on deterministic coverage windows.",
            "mechanicalArtifactProxy": "First-difference energy and clipping/nonfinite counts are diagnostic only; they are not perceptual judgments.",
            "publication": "Do not publish checkpoints, teacher-derived audio, or MUSDB18-derived caches.",
        },
    }
    report_path = experiment_root / "reports" / "inst3-vr-hard-sampling-report.json"
    json_write(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "device": str(device),
                "variants": list(train_results),
                "evaluation": list(evaluation),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
