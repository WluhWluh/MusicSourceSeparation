#!/usr/bin/env python3
"""Build a full-MUSDB18 train hard-event manifest for the vocal-residual model.

This is a diagnostic-only stage.  It keeps the original
``vocals_epoch=891.ckpt`` output semantic:

    teacher_removed = mixture - Inst3_instrumental
    student_removed = mixture - initial_instrumental
    miss = teacher_removed - student_removed

The runner does not train, export, or publish a model.  It processes one
song at a time, writes compact event arrays and summaries under the ignored
``data`` tree, and removes decoded/teacher intermediates unless explicitly
asked to retain them.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import run_inst3_distill_pilot as pilot
import run_inst3_teacher_oracle as oracle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = pilot.DEFAULT_ARCHIVE
DEFAULT_MANIFEST = pilot.DEFAULT_MANIFEST
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_TEACHER = pilot.DEFAULT_TEACHER
DEFAULT_CONTRACT = pilot.DEFAULT_CONTRACT
DEFAULT_TEACHER_TFLITE = pilot.DEFAULT_TEACHER_TFLITE
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-events"
DEFAULT_EVENT_MILLISECONDS = (50, 100, 200)
EVENT_SCHEMA = "local-inst3-vr-hard-events@1"
RUNNER_ID = "inst3-vr-hard-events@1"
ACTIVE_FLOOR_DBFS = -60.0
TOP_EVENTS_PER_SONG = 24


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list")
    if tuple(sorted(set(values))) != values or any(value <= 0 for value in values):
        raise ValueError("Values must be sorted, unique, and positive")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--teacher-tflite", type=Path, default=DEFAULT_TEACHER_TFLITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--event-milliseconds",
        default=",".join(str(value) for value in DEFAULT_EVENT_MILLISECONDS),
    )
    parser.add_argument("--top-events-per-song", type=int, default=TOP_EVENTS_PER_SONG)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--require-teacher-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=891)
    return parser.parse_args(argv)


def dbfs(value: float | np.ndarray, floor: float = -240.0) -> float | np.ndarray:
    result = 20.0 * np.log10(np.maximum(np.asarray(value), 1.0e-12))
    if result.ndim == 0:
        return max(float(result), floor)
    return np.maximum(result, floor)


def rms_blocks(audio: np.ndarray, block_samples: int) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected [samples, 2] audio, got {audio.shape}")
    if block_samples <= 0:
        raise ValueError("block_samples must be positive")
    count = max(1, math.ceil(audio.shape[0] / block_samples))
    padded = np.zeros((count * block_samples, 2), dtype=np.float64)
    padded[: audio.shape[0]] = audio.astype(np.float64, copy=False)
    blocks = padded.reshape(count, block_samples, 2)
    return np.sqrt(np.mean(blocks * blocks, axis=(1, 2)))


def block_features(
    *,
    mixture: np.ndarray,
    teacher_instrumental: np.ndarray,
    student_instrumental: np.ndarray,
    true_vocals: np.ndarray,
    sample_rate: int,
    milliseconds: int,
    active_floor_dbfs: float = ACTIVE_FLOOR_DBFS,
) -> dict[str, np.ndarray | int | float]:
    for name, value in (
        ("mixture", mixture),
        ("teacher_instrumental", teacher_instrumental),
        ("student_instrumental", student_instrumental),
        ("true_vocals", true_vocals),
    ):
        if value.shape != mixture.shape:
            raise ValueError(f"{name} shape {value.shape} != {mixture.shape}")
    block_samples = max(1, round(sample_rate * milliseconds / 1000.0))
    count = max(1, math.ceil(mixture.shape[0] / block_samples))
    padded = lambda value: np.pad(  # noqa: E731 - compact block alignment helper
        value.astype(np.float64, copy=False),
        ((0, count * block_samples - value.shape[0]), (0, 0)),
    ).reshape(count, block_samples, 2)
    mixture_blocks = padded(mixture)
    teacher_blocks = padded(teacher_instrumental)
    student_blocks = padded(student_instrumental)
    vocal_blocks = padded(true_vocals)

    teacher_removed = mixture_blocks - teacher_blocks
    student_removed = mixture_blocks - student_blocks
    miss = teacher_removed - student_removed
    teacher_power = np.sum(teacher_removed * teacher_removed, axis=(1, 2))
    miss_power = np.sum(miss * miss, axis=(1, 2))
    vocal_power = np.sum(vocal_blocks * vocal_blocks, axis=(1, 2))
    teacher_rms = np.sqrt(teacher_power / float(block_samples * 2))
    miss_rms = np.sqrt(miss_power / float(block_samples * 2))
    vocal_rms = np.sqrt(vocal_power / float(block_samples * 2))
    dots = np.sum(miss * teacher_removed, axis=(1, 2))
    coefficients = np.divide(
        dots,
        teacher_power,
        out=np.zeros_like(dots),
        where=teacher_power > 1.0e-20,
    )
    positive_projection_rms = np.maximum(coefficients, 0.0) * teacher_rms
    vocal_dots = np.sum(teacher_removed * vocal_blocks, axis=(1, 2))
    vocal_cosine = np.divide(
        vocal_dots,
        np.sqrt(teacher_power * vocal_power),
        out=np.zeros_like(vocal_dots),
        where=(teacher_power > 1.0e-20) & (vocal_power > 1.0e-20),
    )
    teacher_floor = 10.0 ** (active_floor_dbfs / 20.0)
    active = teacher_rms >= teacher_floor
    # The square-root activity factor stops a high-energy broad error from
    # completely dominating a shorter event with similar positive projection.
    score = positive_projection_rms * np.sqrt(np.maximum(teacher_rms, teacher_floor))
    score = np.where(active, score, 0.0)
    starts = np.arange(count, dtype=np.int64) * block_samples
    ends = np.minimum(starts + block_samples, mixture.shape[0])
    return {
        "blockSamples": block_samples,
        "blockCount": count,
        "startSamples": starts,
        "endSamples": ends,
        "teacherRms": teacher_rms.astype(np.float32),
        "missRms": miss_rms.astype(np.float32),
        "trueVocalRms": vocal_rms.astype(np.float32),
        "projectionCoefficient": coefficients.astype(np.float32),
        "positiveProjectionRms": positive_projection_rms.astype(np.float32),
        "teacherVocalCosine": vocal_cosine.astype(np.float32),
        "active": active,
        "score": score.astype(np.float32),
    }


def classify_event(
    *,
    teacher_rms_dbfs: float,
    true_vocal_rms_dbfs: float,
    teacher_vocal_cosine: float,
    vocal_p20_dbfs: float,
    vocal_p80_dbfs: float,
) -> str:
    """Return a deliberately conservative diagnostic label, not ground truth."""
    if (
        true_vocal_rms_dbfs >= vocal_p80_dbfs
        and teacher_vocal_cosine >= 0.25
    ):
        return "vocal-aligned"
    if (
        teacher_rms_dbfs >= -45.0
        and true_vocal_rms_dbfs <= vocal_p20_dbfs
    ):
        return "vocal-like-or-non-vocal"
    return "mixed-or-uncertain"


def top_event_rows(
    features: dict[str, np.ndarray | int | float],
    *,
    sample_rate: int,
    top_count: int,
    vocal_p20_dbfs: float,
    vocal_p80_dbfs: float,
) -> list[dict[str, Any]]:
    active = np.asarray(features["active"], dtype=bool)
    score = np.asarray(features["score"], dtype=np.float32)
    order = np.lexsort(
        (
            np.asarray(features["startSamples"], dtype=np.int64),
            -score,
        )
    )
    rows: list[dict[str, Any]] = []
    for index in order:
        index = int(index)
        if not active[index]:
            continue
        start = int(np.asarray(features["startSamples"])[index])
        end = int(np.asarray(features["endSamples"])[index])
        teacher_dbfs = float(dbfs(float(np.asarray(features["teacherRms"])[index])))
        vocal_dbfs = float(dbfs(float(np.asarray(features["trueVocalRms"])[index])))
        cosine = float(np.asarray(features["teacherVocalCosine"])[index])
        rows.append(
            {
                "blockIndex": index,
                "startSamples": start,
                "endSamples": end,
                "startSeconds": start / sample_rate,
                "endSeconds": end / sample_rate,
                "teacherRmsDbfs": teacher_dbfs,
                "missRmsDbfs": float(dbfs(float(np.asarray(features["missRms"])[index]))),
                "positiveProjectionRmsDbfs": float(
                    dbfs(float(np.asarray(features["positiveProjectionRms"])[index]))
                ),
                "projectionCoefficient": float(
                    np.asarray(features["projectionCoefficient"])[index]
                ),
                "trueVocalRmsDbfs": vocal_dbfs,
                "teacherVocalCosine": cosine,
                "score": float(np.asarray(features["score"])[index]),
                "classification": classify_event(
                    teacher_rms_dbfs=teacher_dbfs,
                    true_vocal_rms_dbfs=vocal_dbfs,
                    teacher_vocal_cosine=cosine,
                    vocal_p20_dbfs=vocal_p20_dbfs,
                    vocal_p80_dbfs=vocal_p80_dbfs,
                ),
            }
        )
        if len(rows) >= top_count:
            break
    return rows


def aggregate_student_windows(
    features: dict[str, np.ndarray | int | float],
    *,
    useful_samples: int,
    song_samples: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    starts = np.asarray(features["startSamples"], dtype=np.int64)
    scores = np.asarray(features["score"], dtype=np.float32)
    active = np.asarray(features["active"], dtype=bool)
    positive = np.asarray(features["positiveProjectionRms"], dtype=np.float32)
    teacher = np.asarray(features["teacherRms"], dtype=np.float32)
    window_count = max(1, math.ceil(song_samples / useful_samples))
    result: list[dict[str, Any]] = []
    for window_index in range(window_count):
        start = window_index * useful_samples
        end = min(song_samples, start + useful_samples)
        if end - start < useful_samples // 3:
            continue
        selected = (starts >= start) & (starts < end) & active
        if np.any(selected):
            selected_scores = scores[selected]
            selected_positive = positive[selected]
            selected_teacher = teacher[selected]
            score = float(np.max(selected_scores))
            event_count = int(np.count_nonzero(selected))
            sum_positive = float(np.sum(selected_positive))
            max_teacher = float(np.max(selected_teacher))
        else:
            score = 0.0
            event_count = 0
            sum_positive = 0.0
            max_teacher = 0.0
        result.append(
            {
                "windowIndex": window_index,
                "startSamples": start,
                "endSamples": end,
                "startSeconds": start / sample_rate,
                "endSeconds": end / sample_rate,
                "activeEventCount": event_count,
                "hardEventScore": score,
                "sumPositiveProjectionRms": sum_positive,
                "maxTeacherRemovedRmsDbfs": float(dbfs(max_teacher)),
            }
        )
    result.sort(key=lambda item: (-item["hardEventScore"], item["startSamples"]))
    for rank, item in enumerate(result, start=1):
        item["rank"] = rank
    return result


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def npz_write(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def cleanup_song_intermediates(output_root: Path, song: oracle.DecodedSong) -> None:
    raw_path = output_root / "raw" / song.entry["archiveSplit"] / song.entry["fileName"]
    decoded_path = output_root / "decoded" / song.entry["role"] / song.slug
    teacher_path = output_root / "teacher" / song.entry["role"] / song.slug
    raw_path.unlink(missing_ok=True)
    for path in (decoded_path, teacher_path):
        if path.is_dir():
            shutil.rmtree(path)
    for parent in (raw_path.parent, decoded_path.parent, teacher_path.parent):
        try:
            parent.rmdir()
        except OSError:
            pass


def render_initial_instrumental(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    useful = pilot.DEFAULT_CONFIG.useful_samples
    output = np.empty_like(audio)
    window_count = math.ceil(audio.shape[0] / useful)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(window_count):
            start = index * useful
            length = min(useful, audio.shape[0] - start)
            input_spec = pilot.student_window_spec(audio, start, length)
            input_tensor = torch.from_numpy(input_spec[None]).to(device)
            predicted_instrumental = input_tensor - model(input_tensor)
            reconstructed = pilot.student_istft_centered(
                predicted_instrumental.detach().cpu().numpy()
            )
            trim = pilot.DEFAULT_CONFIG.trim_samples
            output[start : start + length] = reconstructed[trim : trim + length]
    if not np.isfinite(output).all():
        raise ValueError("Initial model output contains non-finite values")
    return output, {
        "windowCount": window_count,
        "windowUsefulSamples": useful,
        "elapsedSeconds": time.perf_counter() - started,
        "device": str(device),
    }


def song_analysis(
    *,
    song: oracle.DecodedSong,
    teacher_instrumental: np.ndarray,
    initial_instrumental: np.ndarray,
    event_milliseconds: tuple[int, ...],
    top_events_per_song: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if teacher_instrumental.shape != song.mixture_gt.shape:
        raise ValueError("Teacher output does not match decoded mixture shape")
    if initial_instrumental.shape != song.mixture_gt.shape:
        raise ValueError("Initial output does not match decoded mixture shape")
    teacher_removed = song.mixture_gt - teacher_instrumental
    student_removed = song.mixture_gt - initial_instrumental
    if not np.isfinite(teacher_removed).all() or not np.isfinite(student_removed).all():
        raise ValueError("Removed-content arrays contain non-finite values")

    vocal_rms_by_ms: dict[int, np.ndarray] = {}
    for milliseconds in event_milliseconds:
        block_samples = max(1, round(song.sample_rate * milliseconds / 1000.0))
        vocal_rms_by_ms[milliseconds] = rms_blocks(song.vocals, block_samples)
    vocal_values = vocal_rms_by_ms[100]
    vocal_p20_dbfs = float(dbfs(float(np.percentile(vocal_values, 20.0))))
    vocal_p80_dbfs = float(dbfs(float(np.percentile(vocal_values, 80.0))))

    event_report: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    all_top_rows: list[dict[str, Any]] = []
    window_features: dict[str, np.ndarray] = {}
    for milliseconds in event_milliseconds:
        features = block_features(
            mixture=song.mixture_gt,
            teacher_instrumental=teacher_instrumental,
            student_instrumental=initial_instrumental,
            true_vocals=song.vocals,
            sample_rate=song.sample_rate,
            milliseconds=milliseconds,
        )
        key = str(milliseconds)
        arrays[f"{key}_startSamples"] = np.asarray(features["startSamples"], dtype=np.int64)
        arrays[f"{key}_endSamples"] = np.asarray(features["endSamples"], dtype=np.int64)
        for name in (
            "teacherRms",
            "missRms",
            "trueVocalRms",
            "projectionCoefficient",
            "positiveProjectionRms",
            "teacherVocalCosine",
            "score",
        ):
            arrays[f"{key}_{name}"] = np.asarray(features[name], dtype=np.float32)
        arrays[f"{key}_active"] = np.asarray(features["active"], dtype=np.bool_)
        top_rows = top_event_rows(
            features,
            sample_rate=song.sample_rate,
            top_count=top_events_per_song,
            vocal_p20_dbfs=vocal_p20_dbfs,
            vocal_p80_dbfs=vocal_p80_dbfs,
        )
        for row in top_rows:
            all_top_rows.append({"milliseconds": milliseconds, **row})
        active = np.asarray(features["active"], dtype=bool)
        positive = np.asarray(features["positiveProjectionRms"], dtype=np.float32)
        miss = np.asarray(features["missRms"], dtype=np.float32)
        event_report[key] = {
            "milliseconds": milliseconds,
            "blockSamples": int(features["blockSamples"]),
            "blockCount": int(features["blockCount"]),
            "activeBlockCount": int(np.count_nonzero(active)),
            "activeFraction": float(np.mean(active)),
            "teacherRemovedRmsP50Dbfs": float(dbfs(float(np.percentile(np.asarray(features["teacherRms"]), 50.0)))),
            "teacherRemovedRmsP95Dbfs": float(dbfs(float(np.percentile(np.asarray(features["teacherRms"]), 95.0)))),
            "missRmsP50Dbfs": float(dbfs(float(np.percentile(miss, 50.0)))),
            "missRmsP95Dbfs": float(dbfs(float(np.percentile(miss, 95.0)))),
            "positiveProjectionRmsP95Dbfs": float(
                dbfs(float(np.percentile(positive[active], 95.0))) if np.any(active) else -240.0
            ),
            "positiveProjectionRmsMaxDbfs": float(
                dbfs(float(np.max(positive[active]))) if np.any(active) else -240.0
            ),
            "topEvents": top_rows,
        }

    primary = block_features(
        mixture=song.mixture_gt,
        teacher_instrumental=teacher_instrumental,
        student_instrumental=initial_instrumental,
        true_vocals=song.vocals,
        sample_rate=song.sample_rate,
        milliseconds=100,
    )
    windows = aggregate_student_windows(
        primary,
        useful_samples=pilot.DEFAULT_CONFIG.useful_samples,
        song_samples=song.mixture_gt.shape[0],
        sample_rate=song.sample_rate,
    )
    categories: dict[str, int] = {}
    for row in all_top_rows:
        category = row["classification"]
        categories[category] = categories.get(category, 0) + 1
    all_top_rows.sort(key=lambda row: (-row["score"], row["milliseconds"], row["startSamples"]))
    report = {
        "song": {
            "member": song.entry["member"],
            "role": song.entry["role"],
            "slug": song.slug,
            "sourceSha256": song.entry["sourceSha256"],
            "samples": int(song.mixture_gt.shape[0]),
            "durationSeconds": song.mixture_gt.shape[0] / song.sample_rate,
            "sampleRate": song.sample_rate,
        },
        "diagnosticContract": {
            "teacherRemoved": "mixtureGt - inst3Instrumental",
            "studentRemoved": "mixtureGt - initialInstrumental",
            "miss": "teacherRemoved - studentRemoved",
            "positiveProjection": "max(dot(miss, teacherRemoved) / power(teacherRemoved), 0) * teacherRemovedRms",
            "score": "positiveProjectionRms * sqrt(max(teacherRemovedRms, activeFloor))",
            "activeFloorDbfs": ACTIVE_FLOOR_DBFS,
            "classification": "heuristic; 100 ms true-vocal p20/p80 plus teacher-vocal cosine",
        },
        "vocalRmsPercentiles100msDbfs": {
            "p20": vocal_p20_dbfs,
            "p50": float(dbfs(float(np.percentile(vocal_values, 50.0)))),
            "p80": vocal_p80_dbfs,
        },
        "events": event_report,
        "topEvents": all_top_rows,
        "topEventClassificationCounts": categories,
        "studentWindows": {
            "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
            "count": len(windows),
            "ranked": windows,
        },
    }
    return report, arrays


def aggregate_report(
    song_reports: list[dict[str, Any]], event_milliseconds: tuple[int, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "songCount": len(song_reports),
        "durationSeconds": float(sum(item["song"]["durationSeconds"] for item in song_reports)),
        "byMilliseconds": {},
        "classificationCountsTopEvents": {},
        "songSummaries": [],
    }
    for milliseconds in event_milliseconds:
        rows = [item["events"][str(milliseconds)] for item in song_reports]
        result["byMilliseconds"][str(milliseconds)] = {
            "blockCount": int(sum(row["blockCount"] for row in rows)),
            "activeBlockCount": int(sum(row["activeBlockCount"] for row in rows)),
            "activeFractionAcrossBlocks": float(
                sum(row["activeBlockCount"] for row in rows)
                / max(1, sum(row["blockCount"] for row in rows))
            ),
            "songP95PositiveProjectionRmsDbfs": float(
                np.percentile([row["positiveProjectionRmsP95Dbfs"] for row in rows], 95.0)
            ),
            "songMaxPositiveProjectionRmsDbfs": float(
                np.max([row["positiveProjectionRmsMaxDbfs"] for row in rows])
            ),
        }
    for item in song_reports:
        for category, count in item["topEventClassificationCounts"].items():
            result["classificationCountsTopEvents"][category] = (
                result["classificationCountsTopEvents"].get(category, 0) + count
            )
        result["songSummaries"].append(
            {
                "slug": item["song"]["slug"],
                "durationSeconds": item["song"]["durationSeconds"],
                "top100msScore": max(
                    (
                        row["score"]
                        for row in item["topEvents"]
                        if row["milliseconds"] == 100
                    ),
                    default=0.0,
                ),
                "top100msEvents": sum(
                    1 for row in item["topEvents"] if row["milliseconds"] == 100
                ),
                "classificationCounts": item["topEventClassificationCounts"],
            }
        )
    result["songSummaries"].sort(key=lambda item: (-item["top100msScore"], item["slug"]))
    return result


def load_completed_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("schema") != EVENT_SCHEMA or value.get("status") != "completed":
        return None
    return value


def load_completed_song_report(path: Path, event_path: Path) -> dict[str, Any] | None:
    if not path.is_file() or not event_path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        with np.load(event_path) as arrays:
            if "100_score" not in arrays:
                return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if value.get("song", {}).get("member") is None:
        return None
    if "events" not in value or "studentWindows" not in value:
        return None
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    event_milliseconds = parse_int_list(args.event_milliseconds)
    if 100 not in event_milliseconds:
        raise ValueError("The frozen stage-1 contract requires a 100 ms event resolution")
    if args.threads <= 0 or args.top_events_per_song <= 0:
        raise ValueError("threads and top-events-per-song must be positive")
    if args.max_songs is not None and args.max_songs <= 0:
        raise ValueError("max-songs must be positive")
    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    teacher = args.teacher.resolve()
    contract = args.contract.resolve()
    teacher_tflite = args.teacher_tflite.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root = output_root / "reports"
    songs_root = output_root / "songs"
    event_root = output_root / "events"
    reports_root.mkdir(parents=True, exist_ok=True)
    songs_root.mkdir(parents=True, exist_ok=True)
    event_root.mkdir(parents=True, exist_ok=True)

    manifest = oracle.build_manifest(
        archive,
        manifest_path,
        args.seed,
        calibration_count=10,
        internal_test_count=10,
        force=False,
    )
    entries = sorted(
        [entry for entry in manifest["entries"] if entry["role"] == "train"],
        key=lambda item: item["member"],
    )
    if len(entries) != 80:
        raise ValueError(f"Expected exactly 80 train entries, found {len(entries)}")
    if args.max_songs is not None:
        entries = entries[: args.max_songs]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    contract_info = pilot.verify_teacher_contract(contract, teacher, teacher_tflite)
    teacher_session, teacher_providers = pilot.make_teacher_session(
        teacher, args.threads, args.require_teacher_cuda
    )
    model, checkpoint_metadata = pilot.make_model(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    runner_revision = sha256_file(Path(__file__).resolve())
    report_path = reports_root / "inst3-vr-hard-events-report.json"
    existing = load_completed_report(report_path) if not args.force else None
    completed_by_member = {
        item["song"]["member"]: item for item in (existing or {}).get("songs", [])
    }
    song_reports: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    try:
        for index, entry in enumerate(entries, start=1):
            print(f"analyze {index}/{len(entries)}: {entry['member']}", flush=True)
            if entry["member"] in completed_by_member and not args.force:
                song_reports.append(completed_by_member[entry["member"]])
                print("  reused completed event report", flush=True)
                continue
            slug = oracle.slugify(entry["fileName"])
            cached_song = load_completed_song_report(
                songs_root / f"{slug}.json",
                event_root / f"{slug}.npz",
            )
            if cached_song is not None and cached_song["song"]["member"] == entry["member"] and not args.force:
                song_reports.append(cached_song)
                print("  reused completed per-song event report", flush=True)
                continue
            song = oracle.decode_song(
                archive,
                entry,
                output_root / "raw",
                output_root / "decoded",
                force_extract=args.force,
                force_decode=args.force,
                keep_decoded_wav=False,
            )
            try:
                teacher_instrumental, _, teacher_metadata = oracle.render_or_load_cached(
                    output_root,
                    song,
                    "mixture-gt",
                    song.mixture_gt,
                    teacher_session,
                    contract_info,
                    runner_revision,
                    args.force,
                )
                initial_instrumental, student_timing = render_initial_instrumental(
                    song.mixture_gt, model, device
                )
                report, arrays = song_analysis(
                    song=song,
                    teacher_instrumental=teacher_instrumental,
                    initial_instrumental=initial_instrumental,
                    event_milliseconds=event_milliseconds,
                    top_events_per_song=args.top_events_per_song,
                )
                report["teacher"] = {
                    "providers": teacher_metadata.get("providers", teacher_providers),
                    "cacheKey": teacher_metadata.get("cacheKey"),
                    "outputSha256": teacher_metadata["output"]["instrumentalRawFloat32Sha256"],
                    "renderSeconds": teacher_metadata.get("render", {}).get("timing", {}).get("totalSeconds"),
                }
                report["student"] = {
                    "semantic": "residual-vocals",
                    "checkpoint": str(checkpoint),
                    "checkpointSha256": checkpoint_metadata["checkpoint"]["sha256"],
                    "timing": student_timing,
                }
                report["eventArrayFile"] = str((event_root / f"{song.slug}.npz").resolve())
                npz_write(event_root / f"{song.slug}.npz", arrays)
                json_write(songs_root / f"{song.slug}.json", report)
                song_reports.append(report)
                del teacher_instrumental, initial_instrumental, arrays, report
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            finally:
                if not args.keep_intermediates:
                    cleanup_song_intermediates(output_root, song)
    finally:
        del model
        del teacher_session
        if device.type == "cuda":
            torch.cuda.empty_cache()

    song_reports.sort(key=lambda item: item["song"]["member"])
    aggregate = aggregate_report(song_reports, event_milliseconds)
    report = {
        "schema": EVENT_SCHEMA,
        "status": "completed" if len(song_reports) == len(entries) else "partial",
        "experimentId": RUNNER_ID,
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "source weight redistribution permission not established; not redistributed",
            "derivedOutputs": "ignored local event arrays and reports only; do not publish",
        },
        "manifest": {
            "file": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifestId": manifest["manifestId"],
            "role": "train",
            "songCount": len(entries),
            "finalTestUsed": False,
        },
        "contract": {
            "teacher": contract_info,
            "student": {
                "checkpoint": str(checkpoint),
                "checkpointSha256": checkpoint_metadata["checkpoint"]["sha256"],
                "semantic": "residual-vocals",
                "instrumentalReconstruction": "mixtureGt - studentResidual",
                "sampleRate": pilot.DEFAULT_CONFIG.sample_rate,
                "numFrames": pilot.DEFAULT_CONFIG.num_frames,
                "usefulSamples": pilot.DEFAULT_CONFIG.useful_samples,
            },
            "eventMilliseconds": list(event_milliseconds),
            "activeFloorDbfs": ACTIVE_FLOOR_DBFS,
        },
        "runner": {
            "file": str(Path(__file__).resolve()),
            "sha256": runner_revision,
            "torchVersion": torch.__version__,
            "device": str(device),
            "cuda": {
                "available": torch.cuda.is_available(),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "version": torch.version.cuda,
            },
            "teacherProviders": teacher_providers,
            "elapsedSeconds": time.perf_counter() - started_all,
        },
        "aggregation": aggregate,
        "songs": song_reports,
    }
    json_write(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "songs": len(song_reports),
        "teacherProviders": teacher_providers,
        "aggregation": aggregate,
        "report": str(report_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
