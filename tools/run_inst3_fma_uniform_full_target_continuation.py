#!/usr/bin/env python3
"""Continue step 7408 on broad uniformly sampled FMA Inst 3 targets.

The 71-song pool is reconstructed from the two frozen category-listening
selections.  Each pass uses one randomly jittered full useful-span window for
every non-overlapping-duration slot in every song.  Longer songs therefore
contribute proportionally more windows; there is no per-song balancing.

Unlike the preceding event experiments, the complete useful span is trained
toward the Inst 3 residual target and no step-7408 anchor is used.  A fixed,
broad set of production-aligned windows and the 301 manually marked leakage
events are evaluated after every pass.  These are descriptive in-pool metrics,
not a song-disjoint generalization estimate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

import render_inst3_mtg_fma_event_listening as external
import run_inst3_distill_pilot as pilot
import run_inst3_distill_stability_sweep as sweep
import run_inst3_fma_human_leakage_continuation as human
import run_inst3_fma_survey_continuation as prior
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract, assemble_input, periodic_hann


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "modern-song-fma-s-leakage-survey-continuation"
POOL_DIRECTORIES = (
    DATA_ROOT / "listening-fma-training-category-sample-20" / "step-7408",
    DATA_ROOT / "listening-fma-training-category-remainder-51" / "step-7408",
)
DEFAULT_SOURCE = DATA_ROOT / "runs" / "step-7408.pt"
DEFAULT_ARCHITECTURE = pilot.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = ROOT / "data" / "modern-song-fma-uniform-full-target-continuation"
DEFAULT_HUMAN_EVENTS = (
    ROOT / "data" / "modern-song-fma-human-leakage-continuation" / "event-manifest.json"
)

SAMPLE_RATE = 44_100
SOURCE_STEP = 7_408
SOURCE_PASS = 23
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
EVAL_WINDOWS_PER_SONG = 16
DEFAULT_MAX_PASSES = 8
DEFAULT_MIN_PASSES = 3
DEFAULT_PATIENCE = 2
DEFAULT_MIN_IMPROVEMENT_DB = 0.10
CONTRACT = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
CHECKPOINT_FORMAT = "local-inst3-fma-uniform-full-target-continuation-checkpoint@1"
SCHEMA = "local-inst3-fma-uniform-full-target-continuation@1"
EPSILON = local.EPSILON


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--human-events", type=Path, default=DEFAULT_HUMAN_EVENTS)
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    parser.add_argument("--min-passes", type=int, default=DEFAULT_MIN_PASSES)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-improvement-db", type=float, default=DEFAULT_MIN_IMPROVEMENT_DB)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--eval-windows-per-song", type=int, default=EVAL_WINDOWS_PER_SONG)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--preflight-updates", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def load_pool_metadata() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher_index = human.load_teacher_index()
    songs: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for directory in POOL_DIRECTORIES:
        selection_path = directory / "selection.json"
        render_path = directory / "render-report.json"
        selection = prior.read_json(selection_path)
        render = prior.read_json(render_path)
        if render.get("status") != "completed":
            raise ValueError(f"Incomplete render report: {render_path}")
        rendered = render.get("songs") or {}
        for selected in selection.get("songs", []):
            slug = str(selected.get("slug", ""))
            output = rendered.get(slug)
            if not slug or output is None or slug in songs:
                raise ValueError(f"Missing or duplicate pool song: {slug}")
            source_info = output.get("source") or {}
            source_path = Path(str(source_info.get("file", ""))).resolve()
            source_sha256 = str(source_info.get("sha256", "")).lower()
            if source_path != Path(str(selected.get("sourcePath", ""))).resolve():
                raise ValueError(f"Source path mismatch for {slug}")
            if source_sha256 != str(selected.get("sourceSha256", "")).lower():
                raise ValueError(f"Source SHA mismatch for {slug}")
            teacher = human.resolve_teacher(teacher_index, source_sha256, slug)
            teacher_path = Path(teacher["teacherPath"]).resolve()
            frame_count = int(source_info.get("frames", 0))
            if not source_path.is_file() or not teacher_path.is_file() or frame_count < CONTRACT.useful_samples:
                raise FileNotFoundError(f"Invalid source/teacher contract for {slug}")
            songs[slug] = {
                "slug": slug,
                "artistName": str(selected.get("artistName", "")),
                "trackName": str(selected.get("trackName", "")),
                "category": str(selected.get("category", "")),
                "languageCode": str(selected.get("languageCode", "")),
                "sourcePath": str(source_path),
                "sourceSha256": source_sha256,
                "teacherPath": str(teacher_path),
                "teacherProvenance": teacher["provenance"],
                "frameCount": frame_count,
                "durationSeconds": frame_count / SAMPLE_RATE,
                "fullSlotsPerPass": frame_count // CONTRACT.useful_samples,
                "trainingStages": list(selected.get("trainingStages") or []),
            }
        inputs.append(
            {
                "directory": str(directory.resolve()),
                "selection": prior.checkpoint_metadata(selection_path),
                "renderReport": prior.checkpoint_metadata(render_path),
            }
        )
    ordered = [songs[key] for key in sorted(songs)]
    if len(ordered) != 71:
        raise ValueError(f"Expected 71 FMA training songs, got {len(ordered)}")
    full_slots = sum(int(song["fullSlotsPerPass"]) for song in ordered)
    selection: dict[str, Any] = {
        "schema": SCHEMA + ".selection",
        "poolInputs": inputs,
        "songCount": len(ordered),
        "durationHours": sum(int(song["frameCount"]) for song in ordered) / SAMPLE_RATE / 3600.0,
        "fullSlotsPerPassBeforeBatchPadding": full_slots,
        "categorySongCounts": dict(sorted(Counter(song["category"] for song in ordered).items())),
        "categorySlotCounts": dict(
            sorted(
                {
                    category: sum(int(song["fullSlotsPerPass"]) for song in ordered if song["category"] == category)
                    for category in {song["category"] for song in ordered}
                }.items()
            )
        ),
        "songs": ordered,
        "sampling": {
            "unit": "one random full useful-span start per equal start-domain bin per song per pass",
            "songWeighting": "proportional to decoded duration; no per-song balancing",
            "usefulSamples": CONTRACT.useful_samples,
            "randomStartResolution": "one sample",
            "batchPadding": "repeat uniformly selected records only as needed for divisibility",
        },
        "officialFinalTestUsed": False,
    }
    selection["selectionSha256"] = prior.canonical_sha256(selection)
    return ordered, selection


def random_uniform_starts(frame_count: int, pass_index: int, seed: int, song_index: int) -> list[int]:
    count = frame_count // CONTRACT.useful_samples
    maximum = frame_count - CONTRACT.useful_samples
    if count <= 0 or maximum < 0:
        raise ValueError("Song is too short for a full useful-span window")
    rng = np.random.default_rng(seed + pass_index * 1_000_003 + song_index * 10_007 + 31)
    domain = maximum + 1
    starts: list[int] = []
    for slot in range(count):
        low = domain * slot // count
        high_exclusive = domain * (slot + 1) // count
        high_exclusive = max(high_exclusive, low + 1)
        starts.append(int(rng.integers(low, high_exclusive)))
    if len(set(starts)) != len(starts) or any(start < 0 or start > maximum for start in starts):
        raise AssertionError("Invalid random uniform starts")
    return starts


def build_training_schedule(
    songs: Sequence[dict[str, Any]], passes: int, seed: int, batch_size: int
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    schedule: list[tuple[str, int]] = []
    rows: list[dict[str, Any]] = []
    expected = sum(int(song["fullSlotsPerPass"]) for song in songs)
    padding = (-expected) % batch_size
    for pass_index in range(passes):
        records: list[tuple[str, int]] = []
        for song_index, song in enumerate(songs):
            records.extend(
                (str(song["slug"]), start)
                for start in random_uniform_starts(int(song["frameCount"]), pass_index, seed, song_index)
            )
        rng = np.random.default_rng(seed + pass_index * 1_000_003 + 911)
        if padding:
            extra = rng.choice(len(records), size=padding, replace=False)
            records.extend(records[int(index)] for index in extra)
        records = [records[int(index)] for index in rng.permutation(len(records))]
        schedule.extend(records)
        counts = Counter(slug for slug, _ in records)
        rows.append(
            {
                "pass": pass_index + 1,
                "recordCount": len(records),
                "uniqueRecordCount": len(set(records)),
                "batchPaddingRecords": padding,
                "songRecordCountMin": min(counts.values()),
                "songRecordCountMax": max(counts.values()),
                "songCounts": dict(sorted(counts.items())),
            }
        )
    summary = {
        "passes": rows,
        "recordsPerPass": expected + padding,
        "recordsBeforeBatchPadding": expected,
        "batchPaddingRecordsPerPass": padding,
        "updatesPerPass": (expected + padding) // batch_size,
        "maxPasses": passes,
        "recordCount": len(schedule),
        "scheduleSha256": prior.canonical_sha256(schedule),
    }
    return schedule, summary


def build_evaluation_schedule(
    songs: Sequence[dict[str, Any]], windows_per_song: int
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    records: list[tuple[str, int]] = []
    per_song: dict[str, list[int]] = {}
    for song in songs:
        count = int(song["fullSlotsPerPass"])
        if count < windows_per_song:
            raise ValueError(f"{song['slug']} has only {count} full windows")
        indices = np.rint(np.linspace(0, count - 1, windows_per_song)).astype(np.int64)
        if len(set(indices.tolist())) != windows_per_song:
            raise ValueError(f"Duplicate evaluation index for {song['slug']}")
        starts = [int(index) * CONTRACT.useful_samples for index in indices]
        per_song[str(song["slug"])] = starts
        records.extend((str(song["slug"]), start) for start in starts)
    summary = {
        "role": "fixed broad in-pool descriptive evaluation",
        "windowsPerSong": windows_per_song,
        "songCount": len(songs),
        "recordCount": len(records),
        "productionAligned": True,
        "perSongStarts": per_song,
        "scheduleSha256": prior.canonical_sha256(records),
    }
    return records, summary


def batch_stft(waves: np.ndarray) -> np.ndarray:
    expected = (CONTRACT.input_samples, 2)
    if waves.ndim != 3 or waves.shape[1:] != expected:
        raise ValueError(f"Expected (batch, {expected[0]}, 2), got {waves.shape}")
    channels = waves.transpose(0, 2, 1)
    pad = CONTRACT.n_fft // 2
    padded = np.pad(channels, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, CONTRACT.n_fft, axis=2)[
        :, :, :: CONTRACT.hop_length, :
    ]
    if frames.shape[2] != CONTRACT.num_frames:
        raise ValueError(f"Unexpected STFT frame count: {frames.shape}")
    windowed = frames * periodic_hann(CONTRACT.n_fft)[None, None, None, :]
    spectrum = np.fft.rfft(windowed, n=CONTRACT.n_fft, axis=-1).astype(np.complex64)
    spectrum = spectrum.transpose(0, 1, 3, 2)
    packed = np.stack(
        [
            spectrum[:, 0].real,
            spectrum[:, 1].real,
            spectrum[:, 0].imag,
            spectrum[:, 1].imag,
        ],
        axis=1,
    )
    return np.ascontiguousarray(packed, dtype=np.float32)


class AudioPool:
    def __init__(self, songs: Sequence[dict[str, Any]]) -> None:
        self.metadata = {str(song["slug"]): song for song in songs}
        self.source: dict[str, np.ndarray] = {}
        self.teacher: dict[str, np.ndarray] = {}
        self.bytes = 0

    def load(self) -> dict[str, Any]:
        started = time.perf_counter()
        for index, slug in enumerate(sorted(self.metadata), start=1):
            song = self.metadata[slug]
            source = external.load_audio(Path(song["sourcePath"]))
            teacher = external.load_audio(Path(song["teacherPath"]))
            expected = int(song["frameCount"])
            if source.shape != teacher.shape or source.shape != (expected, 2):
                raise ValueError(f"Decoded source/teacher mismatch for {slug}: {source.shape}, {teacher.shape}, expected {expected}")
            if not np.isfinite(source).all() or not np.isfinite(teacher).all():
                raise ValueError(f"Non-finite decoded audio for {slug}")
            self.source[slug] = source
            self.teacher[slug] = teacher
            self.bytes += source.nbytes + teacher.nbytes
            print(
                json.dumps(
                    {
                        "event": "audio-loaded",
                        "index": index,
                        "total": len(self.metadata),
                        "slug": slug,
                        "residentGiB": self.bytes / 2**30,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return {
            "songCount": len(self.source),
            "bytes": self.bytes,
            "giB": self.bytes / 2**30,
            "elapsedSeconds": time.perf_counter() - started,
        }

    def prepare_batch(
        self, records: Sequence[tuple[str, int]], *, include_audio: bool = False
    ) -> tuple[np.ndarray, ...]:
        waves: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        source_rows: list[np.ndarray] = []
        teacher_rows: list[np.ndarray] = []
        for slug, start in records:
            source = self.source[slug]
            teacher = self.teacher[slug]
            end = start + CONTRACT.useful_samples
            if start < 0 or end > source.shape[0]:
                raise ValueError(f"Window outside {slug}: {start}:{end}")
            waves.append(
                assemble_input(source, start, CONTRACT.useful_samples, CONTRACT, mode="continuous")
            )
            source_row = source[start:end]
            teacher_row = teacher[start:end]
            targets.append(np.ascontiguousarray(source_row - teacher_row, dtype=np.float32))
            if include_audio:
                source_rows.append(source_row)
                teacher_rows.append(teacher_row)
        output: list[np.ndarray] = [batch_stft(np.stack(waves)), np.stack(targets).astype(np.float32)]
        if include_audio:
            output.extend(
                [
                    np.ascontiguousarray(np.stack(source_rows), dtype=np.float32),
                    np.ascontiguousarray(np.stack(teacher_rows), dtype=np.float32),
                ]
            )
        return tuple(output)


def db_array(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(values, 1.0e-12))


def residual_metrics(target: np.ndarray, miss: np.ndarray) -> dict[str, np.ndarray]:
    if target.shape != miss.shape or target.ndim < 3 or target.shape[-1] != 2:
        raise ValueError(f"Invalid residual metric arrays: {target.shape}, {miss.shape}")
    axes = (-2, -1)
    target64 = target.astype(np.float64, copy=False)
    miss64 = miss.astype(np.float64, copy=False)
    target_power = np.sum(target64 * target64, axis=axes)
    miss_power = np.sum(miss64 * miss64, axis=axes)
    dot = np.sum(target64 * miss64, axis=axes)
    sample_count = int(target.shape[-2] * target.shape[-1])
    removed_rms = np.sqrt(target_power / sample_count)
    miss_rms = np.sqrt(miss_power / sample_count)
    projection = np.maximum(dot / np.maximum(target_power, 1.0e-30), 0.0) * removed_rms
    return {
        "removedRmsDbfs": db_array(removed_rms),
        "missRmsDbfs": db_array(miss_rms),
        "positiveProjectionDbfs": db_array(projection),
    }


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def summarize_metric_rows(rows: dict[str, list[float]]) -> dict[str, Any]:
    return {name: summarize_values(values) for name, values in sorted(rows.items())}


def _prefetched_batches(
    pool: AudioPool,
    records: Sequence[tuple[str, int]],
    batch_size: int,
    *,
    include_audio: bool,
):
    if len(records) % batch_size:
        raise ValueError("Record count must be divisible by batch size")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fma-window-prefetch") as executor:
        future: Future[tuple[np.ndarray, ...]] | None = None
        for begin in range(0, len(records), batch_size):
            batch_records = records[begin : begin + batch_size]
            if future is None:
                future = executor.submit(pool.prepare_batch, batch_records, include_audio=include_audio)
            prepared = future.result()
            next_begin = begin + batch_size
            if next_begin < len(records):
                future = executor.submit(
                    pool.prepare_batch,
                    records[next_begin : next_begin + batch_size],
                    include_audio=include_audio,
                )
            else:
                future = None
            yield batch_records, prepared


def evaluate_broad_windows(
    model: torch.nn.Module,
    pool: AudioPool,
    records: Sequence[tuple[str, int]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    started = time.perf_counter()
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    whole: dict[str, list[float]] = defaultdict(list)
    short: dict[int, dict[str, list[float]]] = {
        milliseconds: defaultdict(list) for milliseconds in (50, 100, 200)
    }
    per_song: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_sums = Counter()
    with torch.inference_mode():
        for batch_records, prepared in _prefetched_batches(
            pool, records, batch_size, include_audio=True
        ):
            input_array, target, source, teacher = prepared
            prediction_full = local.torch_packed_istft(
                model(torch.from_numpy(input_array).to(device)), window
            )
            trim = pilot.DEFAULT_CONFIG.trim_samples
            prediction = (
                prediction_full[:, trim : trim + CONTRACT.useful_samples]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            miss = np.ascontiguousarray(target - prediction, dtype=np.float32)
            candidate = np.ascontiguousarray(source - prediction, dtype=np.float32)
            whole_batch = residual_metrics(target, miss)
            teacher_power = np.sum(teacher.astype(np.float64) ** 2, axis=(1, 2))
            error_power = np.sum(miss.astype(np.float64) ** 2, axis=(1, 2))
            target_power = np.sum(target.astype(np.float64) ** 2, axis=(1, 2))
            first_difference = np.diff(miss.astype(np.float64), axis=1)
            first_difference_rms = np.sqrt(np.mean(first_difference * first_difference, axis=(1, 2)))
            for index, (slug, _start) in enumerate(batch_records):
                category = str(pool.metadata[slug]["category"])
                values = {
                    "removedRmsDbfs": float(whole_batch["removedRmsDbfs"][index]),
                    "targetErrorRmsDbfs": float(whole_batch["missRmsDbfs"][index]),
                    "positiveProjectionDbfs": float(whole_batch["positiveProjectionDbfs"][index]),
                    "instrumentalMatchSnrDb": float(
                        10.0 * math.log10(max(float(teacher_power[index]), 1.0e-30) / max(float(error_power[index]), 1.0e-30))
                    ),
                    "residualTargetSnrDb": float(
                        10.0 * math.log10(max(float(target_power[index]), 1.0e-30) / max(float(error_power[index]), 1.0e-30))
                    ),
                    "firstDifferenceErrorRmsDbfs": float(db_array(np.asarray([first_difference_rms[index]]))[0]),
                    "candidatePeak": float(np.max(np.abs(candidate[index]))),
                }
                for name, value in values.items():
                    whole[name].append(value)
                    per_song[slug][name].append(value)
                    per_category[category][name].append(value)
                global_sums["teacherPower"] += float(teacher_power[index])
                global_sums["targetPower"] += float(target_power[index])
                global_sums["errorPower"] += float(error_power[index])
                global_sums["sampleValues"] += int(target[index].size)
                global_sums["candidateClipCount"] += int(np.count_nonzero(np.abs(candidate[index]) > 1.0))
            for milliseconds in (50, 100, 200):
                length = round(SAMPLE_RATE * milliseconds / 1000.0)
                block_count = CONTRACT.useful_samples // length
                offset = (CONTRACT.useful_samples - block_count * length) // 2
                target_blocks = target[:, offset : offset + block_count * length].reshape(
                    target.shape[0], block_count, length, 2
                )
                miss_blocks = miss[:, offset : offset + block_count * length].reshape(
                    miss.shape[0], block_count, length, 2
                )
                metrics = residual_metrics(target_blocks, miss_blocks)
                for name, values in metrics.items():
                    short[milliseconds][name].extend(values.reshape(-1).tolist())
                if milliseconds == 100:
                    for index, (slug, _start) in enumerate(batch_records):
                        category = str(pool.metadata[slug]["category"])
                        values = metrics["positiveProjectionDbfs"][index].reshape(-1).tolist()
                        per_song[slug]["short100PositiveProjectionDbfs"].extend(values)
                        per_category[category]["short100PositiveProjectionDbfs"].extend(values)
    sample_values = max(int(global_sums["sampleValues"]), 1)
    error_power = max(float(global_sums["errorPower"]), 1.0e-30)
    result = {
        "role": "fixed broad in-pool descriptive evaluation",
        "elapsedSeconds": time.perf_counter() - started,
        "recordCount": len(records),
        "wholeWindow": summarize_metric_rows(whole),
        "shortBlocks": {
            str(milliseconds): summarize_metric_rows(rows)
            for milliseconds, rows in short.items()
        },
        "global": {
            "targetErrorRmsDbfs": float(20.0 * math.log10(math.sqrt(error_power / sample_values))),
            "instrumentalMatchSnrDb": float(
                10.0 * math.log10(max(float(global_sums["teacherPower"]), 1.0e-30) / error_power)
            ),
            "residualTargetSnrDb": float(
                10.0 * math.log10(max(float(global_sums["targetPower"]), 1.0e-30) / error_power)
            ),
            "candidateClipCount": int(global_sums["candidateClipCount"]),
        },
        "byCategory": {
            category: summarize_metric_rows(rows) for category, rows in sorted(per_category.items())
        },
        "bySong": {
            slug: summarize_metric_rows(rows) for slug, rows in sorted(per_song.items())
        },
    }
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    del window
    return result


def load_human_event_groups(path: Path, pool: AudioPool) -> dict[str, dict[str, Any]]:
    manifest = prior.read_json(path.resolve())
    groups = manifest.get("groups") or {}
    output: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        slug = str(group.get("slug", ""))
        if slug not in pool.source:
            raise ValueError(f"Human-event song is absent from the 71-song pool: {slug}")
        output[str(key)] = group
    return output


def evaluate_human_events(
    model: torch.nn.Module,
    pool: AudioPool,
    groups: dict[str, dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    all_rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()
    inferred_windows = 0
    for group in groups.values():
        slug = str(group["slug"])
        source = pool.source[slug]
        teacher = pool.teacher[slug]
        indices = human.required_production_windows(group["events"], source.shape[0])
        segments = human.infer_production_windows(model, source, indices, device, batch_size)
        inferred_windows += len(indices)
        for event in group["events"]:
            metrics: dict[str, dict[str, float]] = {}
            center = int(event["centerSamples"])
            for milliseconds in (50, 100, 200):
                length = round(SAMPLE_RATE * milliseconds / 1000.0)
                start = center - length // 2
                end = start + length
                residual = human.extract_production_block(segments, start, end)
                source_block = source[start:end]
                teacher_block = teacher[start:end]
                candidate = np.ascontiguousarray(source_block - residual, dtype=np.float32)
                metrics[str(milliseconds)] = external.block_metric(
                    source_block, teacher_block, candidate, length // 2, milliseconds
                )
            row = {"eventId": event["eventId"], "metrics": metrics}
            all_rows.append(row)
            by_category[str(group["category"])].append(row)
    result = {
        "role": "301 manually marked in-pool leakage events",
        "elapsedSeconds": time.perf_counter() - started,
        "eventCount": len(all_rows),
        "inferredProductionWindowCount": inferred_windows,
        "all": prior.summarize_rows(all_rows),
        "byCategory": {
            category: prior.summarize_rows(rows) for category, rows in sorted(by_category.items())
        },
    }
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    return result


def train_updates(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pool: AudioPool,
    records: Sequence[tuple[str, int]],
    args: argparse.Namespace,
    device: torch.device,
    *,
    pass_index: int,
    global_update_offset: int,
    max_updates: int | None = None,
) -> list[dict[str, Any]]:
    model.train()
    sweep.freeze_batchnorm_running_statistics(model)
    updates = len(records) // args.batch_size
    if max_updates is not None:
        updates = min(updates, max_updates)
        records = records[: updates * args.batch_size]
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, (_batch_records, prepared) in enumerate(
        _prefetched_batches(pool, records, args.batch_size, include_audio=False), start=1
    ):
        batch_started = time.perf_counter()
        input_array, target_array = prepared
        input_tensor = torch.from_numpy(input_array).to(device)
        target_tensor = torch.from_numpy(target_array).to(device)
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        predicted_full = local.torch_packed_istft(model(input_tensor), window)
        trim = pilot.DEFAULT_CONFIG.trim_samples
        predicted = predicted_full[:, trim : trim + CONTRACT.useful_samples]
        forward_seconds = time.perf_counter() - forward_started
        loss = torch.sqrt((predicted - target_tensor).square() + EPSILON**2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at pass {pass_index + 1}, update {update}")
        backward_started = time.perf_counter()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False).item())
        if not math.isfinite(gradient):
            raise FloatingPointError(f"Non-finite gradient at pass {pass_index + 1}, update {update}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "pass": pass_index + 1,
            "update": update,
            "globalStep": SOURCE_STEP + global_update_offset + update,
            "loss": float(loss.detach().cpu()),
            "gradientNormBeforeClip": gradient,
            "forwardSeconds": forward_seconds,
            "backwardSeconds": time.perf_counter() - backward_started,
            "batchSeconds": time.perf_counter() - batch_started,
            "cudaAllocatedBytes": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None,
            "cudaReservedBytes": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None,
        }
        rows.append(row)
        if update == 1 or update % args.log_every == 0 or update == updates:
            print(json.dumps({"event": "progress", **row, "updates": updates}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {"event": "pass-complete", "pass": pass_index + 1, "updates": updates, "elapsedSeconds": time.perf_counter() - started},
            sort_keys=True,
        ),
        flush=True,
    )
    del window
    return rows


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    source: dict[str, Any],
    contract_id: str,
    selection_hash: str,
    schedule_hash: str,
    history: list[dict[str, Any]],
    local_pass: int,
    global_step: int,
    records_per_pass: int,
) -> dict[str, Any]:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "milestone",
        "variant": "FMA-uniform-full-target",
        "runContractId": contract_id,
        "sourceCheckpoint": source["checkpoint"]["file"],
        "sourceStep": SOURCE_STEP,
        "sourcePass": source["sourcePass"],
        "localPass": local_pass,
        "step": global_step,
        "globalStep": global_step,
        "continuationPass": int(source["sourcePass"]) + local_pass,
        "recordsPerPass": records_per_pass,
        "updatesPerPass": records_per_pass // args.batch_size,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "selectionSha256": selection_hash,
        "scheduleSha256": schedule_hash,
        "lossContract": "full useful-window Charbonnier toward mixture - Inst3; no source anchor",
        "assembly": CONTRACT.as_dict(assembly="continuous-context-overlap-save"),
        "stateDict": prior.cpu_tree(model.state_dict()),
        "optimizerStateDict": prior.cpu_tree(optimizer.state_dict()),
        "history": history,
    }
    prior.atomic_torch_save(path, payload)
    return prior.checkpoint_metadata(path)


def load_latest_checkpoint(
    run_root: Path, contract_id: str, selection_hash: str, schedule_hash: str
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in run_root.glob("step-*.pt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("format") == CHECKPOINT_FORMAT
                and payload.get("runContractId") == contract_id
                and payload.get("selectionSha256") == selection_hash
                and payload.get("scheduleSha256") == schedule_hash
            ):
                candidates.append((int(payload.get("localPass", -1)), path, payload))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def make_report(
    args: argparse.Namespace,
    selection: dict[str, Any],
    schedule_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    source: dict[str, Any],
    contract_id: str,
    contract_payload: dict[str, Any],
    audio_load: dict[str, Any],
    preflight: dict[str, Any] | None,
    history: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    updates_per_pass = int(schedule_summary["updatesPerPass"])
    return {
        "schema": SCHEMA,
        "status": status,
        "selection": selection,
        "schedule": schedule_summary,
        "evaluationSchedule": eval_summary,
        "source": source,
        "contract": {"id": contract_id, "payload": contract_payload},
        "audioLoad": audio_load,
        "preflight": preflight,
        "training": {
            "maxPasses": args.max_passes,
            "completedPasses": len(history) // updates_per_pass,
            "history": history,
        },
        "evaluations": evaluations,
        "stop": {"patience": args.patience, "minImprovementDb": args.min_improvement_db},
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def run(
    args: argparse.Namespace,
    songs: list[dict[str, Any]],
    selection: dict[str, Any],
    pool: AudioPool,
    audio_load: dict[str, Any],
    schedule: list[tuple[str, int]],
    schedule_summary: dict[str, Any],
    eval_records: list[tuple[str, int]],
    eval_summary: dict[str, Any],
    human_groups: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    contract_payload = {
        "schema": SCHEMA,
        "sourceCheckpointSha256": selection["sourceCheckpoint"]["sha256"],
        "selectionSha256": selection["selectionSha256"],
        "scheduleSha256": schedule_summary["scheduleSha256"],
        "evaluationScheduleSha256": eval_summary["scheduleSha256"],
        "recordsPerPass": schedule_summary["recordsPerPass"],
        "updatesPerPass": schedule_summary["updatesPerPass"],
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "assembly": "continuous-context-overlap-save",
        "target": "full useful-window mixture - Inst3 residual target",
        "anchor": None,
        "batchNormRunningStatistics": "frozen",
        "officialFinalTestUsed": False,
    }
    contract_id = prior.canonical_sha256(contract_payload)
    run_root = args.output_root / "runs"
    report_path = args.output_root / "reports" / "training-report.json"
    run_root.mkdir(parents=True, exist_ok=True)
    prior.set_seed(args.seed)
    model, optimizer, source, _ = human.load_source_model(args, device)
    completed_passes = 0
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    found = None
    if args.resume:
        found = load_latest_checkpoint(
            run_root, contract_id, selection["selectionSha256"], schedule_summary["scheduleSha256"]
        )
    if found is not None:
        path, payload = found
        model.load_state_dict(payload["stateDict"], strict=True)
        optimizer.load_state_dict(payload["optimizerStateDict"])
        prior.move_optimizer_state(optimizer, device)
        for optimizer_group in optimizer.param_groups:
            optimizer_group["lr"] = args.learning_rate
        completed_passes = int(payload.get("localPass", 0))
        history = list(payload.get("history", []))
        if report_path.is_file():
            old = prior.read_json(report_path)
            evaluations = list(old.get("evaluations", []))
            preflight = old.get("preflight")
        print(json.dumps({"event": "resume", "pass": completed_passes, "checkpoint": str(path)}, sort_keys=True), flush=True)
    elif args.preflight_updates:
        monitor = prior.GpuMonitor(device)
        monitor.start()
        rows = train_updates(
            model,
            optimizer,
            pool,
            schedule[: args.preflight_updates * args.batch_size],
            args,
            device,
            pass_index=0,
            global_update_offset=0,
            max_updates=args.preflight_updates,
        )
        preflight = {
            "status": "completed",
            "updates": len(rows),
            "finite": all(math.isfinite(float(row["loss"])) for row in rows),
            "trainingGpuMonitor": monitor.stop(),
            "history": rows,
        }
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, optimizer, source, _ = human.load_source_model(args, device)

    if not evaluations:
        broad = evaluate_broad_windows(model, pool, eval_records, device, args.eval_batch_size)
        marked = evaluate_human_events(model, pool, human_groups, device, args.eval_batch_size)
        evaluations.append(
            {
                "pass": 0,
                "globalStep": SOURCE_STEP,
                "role": "step-7408-source-baseline",
                "broad": broad,
                "humanLeakageEvents": marked,
                "improvementFromReferenceDb": None,
                "meaningfulImprovement": None,
            }
        )
        print(
            json.dumps(
                {
                    "event": "baseline",
                    "broadP95_100ms": broad["shortBlocks"]["100"]["positiveProjectionDbfs"]["p95"],
                    "broadTargetErrorDbfs": broad["global"]["targetErrorRmsDbfs"],
                    "humanP95_100ms": marked["all"]["100"]["positiveProjectionP95Dbfs"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        prior.json_write(
            report_path,
            make_report(
                args,
                selection,
                schedule_summary,
                eval_summary,
                source,
                contract_id,
                contract_payload,
                audio_load,
                preflight,
                history,
                evaluations,
                "running",
            ),
        )

    reference_p95 = float(
        evaluations[0]["broad"]["shortBlocks"]["100"]["positiveProjectionDbfs"]["p95"]
    )
    best_p95 = min(
        float(row["broad"]["shortBlocks"]["100"]["positiveProjectionDbfs"]["p95"])
        for row in evaluations
    )
    no_improvement = 0
    for row in evaluations[1:]:
        if bool(row.get("meaningfulImprovement")):
            reference_p95 = float(
                row["broad"]["shortBlocks"]["100"]["positiveProjectionDbfs"]["p95"]
            )
            no_improvement = 0
        else:
            no_improvement += 1

    records_per_pass = int(schedule_summary["recordsPerPass"])
    updates_per_pass = int(schedule_summary["updatesPerPass"])
    try:
        for pass_index in range(completed_passes, args.max_passes):
            pass_records = schedule[
                pass_index * records_per_pass : (pass_index + 1) * records_per_pass
            ]
            monitor = prior.GpuMonitor(device)
            monitor.start()
            rows = train_updates(
                model,
                optimizer,
                pool,
                pass_records,
                args,
                device,
                pass_index=pass_index,
                global_update_offset=pass_index * updates_per_pass,
            )
            training_gpu = monitor.stop()
            history.extend(rows)
            local_pass = pass_index + 1
            global_step = SOURCE_STEP + local_pass * updates_per_pass
            checkpoint = save_checkpoint(
                run_root / f"step-{global_step}.pt",
                model,
                optimizer,
                args,
                source,
                contract_id,
                selection["selectionSha256"],
                schedule_summary["scheduleSha256"],
                history,
                local_pass,
                global_step,
                records_per_pass,
            )
            broad = evaluate_broad_windows(model, pool, eval_records, device, args.eval_batch_size)
            marked = evaluate_human_events(model, pool, human_groups, device, args.eval_batch_size)
            current_p95 = float(
                broad["shortBlocks"]["100"]["positiveProjectionDbfs"]["p95"]
            )
            improvement = reference_p95 - current_p95
            meaningful = improvement >= args.min_improvement_db
            best_p95 = min(best_p95, current_p95)
            evaluations.append(
                {
                    "pass": local_pass,
                    "globalStep": global_step,
                    "role": "uniform-full-target-continuation",
                    "checkpoint": checkpoint,
                    "broad": broad,
                    "humanLeakageEvents": marked,
                    "improvementFromReferenceDb": improvement,
                    "meaningfulImprovement": meaningful,
                    "bestP95Dbfs": best_p95,
                    "trainingGpuMonitor": training_gpu,
                }
            )
            if meaningful:
                reference_p95 = current_p95
                no_improvement = 0
            else:
                no_improvement += 1
            report = make_report(
                args,
                selection,
                schedule_summary,
                eval_summary,
                source,
                contract_id,
                contract_payload,
                audio_load,
                preflight,
                history,
                evaluations,
                "running",
            )
            prior.json_write(report_path, report)
            gpu_p50 = (training_gpu.get("gpuUtilizationPercent") or {}).get("p50")
            print(
                json.dumps(
                    {
                        "event": "evaluation",
                        "pass": local_pass,
                        "globalStep": global_step,
                        "broadP95_100ms": current_p95,
                        "improvementFromReferenceDb": improvement,
                        "meaningfulImprovement": meaningful,
                        "broadTargetErrorDbfs": broad["global"]["targetErrorRmsDbfs"],
                        "broadInstrumentalMatchSnrDb": broad["global"]["instrumentalMatchSnrDb"],
                        "humanP95_100ms": marked["all"]["100"]["positiveProjectionP95Dbfs"],
                        "trainingGpuP50": gpu_p50,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if local_pass >= args.min_passes and no_improvement >= args.patience:
                report["status"] = "completed"
                report["stop"] = {
                    "reason": "broad in-pool 100ms projection p95 plateau",
                    "stoppedAfterPass": local_pass,
                    "consecutiveNonMeaningfulPasses": no_improvement,
                    "minImprovementDb": args.min_improvement_db,
                    "patience": args.patience,
                    "bestP95Dbfs": best_p95,
                }
                prior.json_write(report_path, report)
                break
        else:
            report = prior.read_json(report_path)
            report["status"] = "completed"
            report["stop"] = {
                "reason": "max-passes guard",
                "maxPasses": args.max_passes,
                "minImprovementDb": args.min_improvement_db,
                "patience": args.patience,
                "bestP95Dbfs": best_p95,
            }
            prior.json_write(report_path, report)
    finally:
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prior.read_json(report_path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_passes <= 0 or args.min_passes <= 0 or args.min_passes > args.max_passes:
        raise ValueError("Invalid pass limits")
    if args.patience <= 0 or args.min_improvement_db < 0:
        raise ValueError("Invalid plateau settings")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.eval_windows_per_song <= 0:
        raise ValueError("Batch sizes and evaluation window count must be positive")
    if args.preflight_updates < 0 or args.threads <= 0:
        raise ValueError("Invalid preflight or thread count")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    songs, selection = load_pool_metadata()
    selection["sourceCheckpoint"] = prior.checkpoint_metadata(args.source_checkpoint)
    selection["sourceStep"] = SOURCE_STEP
    selection["selectionSha256"] = prior.canonical_sha256(
        {key: value for key, value in selection.items() if key != "selectionSha256"}
    )
    schedule, schedule_summary = build_training_schedule(
        songs, args.max_passes, args.seed, args.batch_size
    )
    eval_records, eval_summary = build_evaluation_schedule(
        songs, args.eval_windows_per_song
    )
    prior.json_write(args.output_root / "pool-selection.json", selection)
    prior.json_write(
        args.output_root / "training-schedule.json",
        {"summary": schedule_summary, "schedule": schedule},
    )
    prior.json_write(args.output_root / "evaluation-schedule.json", eval_summary)
    print(
        json.dumps(
            {
                "event": "selection-complete",
                "songs": len(songs),
                "durationHours": selection["durationHours"],
                "recordsPerPass": schedule_summary["recordsPerPass"],
                "updatesPerPass": schedule_summary["updatesPerPass"],
                "evalRecords": len(eval_records),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.selection_only:
        return 0

    pool = AudioPool(songs)
    audio_load = pool.load()
    human_groups = load_human_event_groups(args.human_events, pool)
    result = run(
        args,
        songs,
        selection,
        pool,
        audio_load,
        schedule,
        schedule_summary,
        eval_records,
        eval_summary,
        human_groups,
        device,
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "report": str(args.output_root / "reports" / "training-report.json"),
                "completedPasses": result.get("training", {}).get("completedPasses"),
                "stop": result.get("stop"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    del pool
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
