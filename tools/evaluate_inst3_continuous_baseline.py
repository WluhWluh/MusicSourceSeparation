#!/usr/bin/env python3
"""Evaluate the original and H50 TFC-TDF models with continuous context.

This report deliberately separates model behavior from the old isolated
zero-padded listening path.  The original checkpoint and H50 checkpoint use
the same 128-frame continuous overlap-save contract.  Inst 3 is loaded from
the native full-song MDX oracle cache for MUSDB18 evaluation songs and is
rendered with its native MDX contract for the private listening set.

All generated audio and reports are local non-commercial research artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import render_inst3_objective_listening as listening
import run_inst3_distill_pilot as pilot
import run_inst3_scale10_density as density
import run_inst3_teacher_oracle as teacher_oracle
import run_inst3_vr_hard_sampling as hard
from tfc_tdf_short_window import (
    ShortWindowContract,
    assemble_input,
    istft_centered,
    seam_metrics,
    stft_centered,
)


ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-continuous-baseline-evaluation"
DEFAULT_H50_PRIVATE_INSTRUMENTAL_ROOT = (
    ROOT
    / "data"
    / "musdb18-inst3-f24-h50-comparison"
    / "h50-128-continuous-instrumental"
)
SAMPLE_RATE = 44_100
JOIN_RADIUS_SAMPLES = round(SAMPLE_RATE * 0.100)
REGIONS = (
    "song-edge-100ms",
    "join-neighborhood-100ms",
    "window-center-80pct",
    "window-edge-10pct",
)
MODELS = ("initial-continuous", "H50-continuous", "Inst3-native")
EVENT_MS = (50, 100, 200)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument(
        "--h50-private-instrumental-root",
        type=Path,
        default=DEFAULT_H50_PRIVATE_INSTRUMENTAL_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-eval-songs", type=int)
    parser.add_argument("--skip-private", action="store_true")
    parser.add_argument(
        "--private-only",
        action="store_true",
        help="Update an existing report with private renders without rerunning MUSDB evaluation",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-teacher-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA for native Inst 3 private renders",
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


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def dbfs(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def rms(value: np.ndarray) -> float:
    return math.sqrt(float(np.mean(value.astype(np.float64) ** 2)))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64).reshape(-1)
    right64 = right.astype(np.float64).reshape(-1)
    denominator = math.sqrt(float(np.dot(left64, left64) * np.dot(right64, right64)))
    return float(np.dot(left64, right64) / max(denominator, 1.0e-30))


def snr(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference64 = reference.astype(np.float64)
    error64 = candidate.astype(np.float64) - reference64
    return 10.0 * math.log10(
        max(float(np.sum(reference64 * reference64)), 1.0e-30)
        / max(float(np.sum(error64 * error64)), 1.0e-30)
    )


def positive_projection_db(value: float, basis: np.ndarray) -> float:
    basis64 = basis.astype(np.float64)
    basis_power = float(np.sum(basis64 * basis64))
    if basis_power <= 1.0e-30:
        return -240.0
    coefficient = max(float(value), 0.0) / basis_power
    return dbfs(coefficient * rms(basis))


class RegionAccumulator:
    """Sufficient statistics for one sample region."""

    def __init__(self) -> None:
        self.count = 0
        self.reference_power = 0.0
        self.candidate_power = 0.0
        self.error_power = 0.0
        self.vocal_power = 0.0
        self.vocal_dot = 0.0
        self.teacher_removed_power = 0.0
        self.teacher_miss_power = 0.0
        self.teacher_miss_dot = 0.0

    def update(
        self,
        reference_instrumental: np.ndarray,
        candidate: np.ndarray,
        vocals: np.ndarray,
        teacher_removed: np.ndarray,
        teacher_miss: np.ndarray,
    ) -> None:
        ref64 = reference_instrumental.astype(np.float64)
        candidate64 = candidate.astype(np.float64)
        vocals64 = vocals.astype(np.float64)
        teacher64 = teacher_removed.astype(np.float64)
        miss64 = teacher_miss.astype(np.float64)
        error64 = candidate64 - ref64
        self.count += int(ref64.size)
        self.reference_power += float(np.sum(ref64 * ref64))
        self.candidate_power += float(np.sum(candidate64 * candidate64))
        self.error_power += float(np.sum(error64 * error64))
        self.vocal_power += float(np.sum(vocals64 * vocals64))
        self.vocal_dot += float(np.sum(error64 * vocals64))
        self.teacher_removed_power += float(np.sum(teacher64 * teacher64))
        self.teacher_miss_power += float(np.sum(miss64 * miss64))
        self.teacher_miss_dot += float(np.sum(miss64 * teacher64))

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {"samples": 0}
        return {
            "samples": self.count,
            "instrumentalSdrDb": 10.0
            * math.log10(max(self.reference_power, 1.0e-30) / max(self.error_power, 1.0e-30)),
            "accompanimentErrorRmsDbfs": dbfs(
                math.sqrt(self.error_power / float(self.count))
            ),
            "candidateRmsDbfs": dbfs(
                math.sqrt(self.candidate_power / float(self.count))
            ),
            "positiveVocalProjectionDbfs": (
                dbfs(
                    max(self.vocal_dot, 0.0)
                    / max(self.vocal_power, 1.0e-30)
                    * math.sqrt(self.vocal_power / float(self.count))
                )
                if self.vocal_power > 1.0e-30
                else -240.0
            ),
            "teacherResidualMissRmsDbfs": dbfs(
                math.sqrt(self.teacher_miss_power / float(self.count))
            ),
            "teacherRemovedRmsDbfs": dbfs(
                math.sqrt(self.teacher_removed_power / float(self.count))
            ),
            "teacherRemovedPositiveProjectionDbfs": (
                dbfs(
                    max(self.teacher_miss_dot, 0.0)
                    / max(self.teacher_removed_power, 1.0e-30)
                    * math.sqrt(self.teacher_removed_power / float(self.count))
                )
                if self.teacher_removed_power > 1.0e-30
                else -240.0
            ),
        }


def region_masks(sample_count: int, contract: ShortWindowContract) -> dict[str, np.ndarray]:
    """Partition samples into song edge, joins, window center, and window edge."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    indices = np.arange(sample_count, dtype=np.int64)
    window_count = math.ceil(sample_count / contract.stride_samples)
    owner = np.minimum(indices // contract.stride_samples, window_count - 1)
    local = indices - owner * contract.stride_samples
    song_edge = (indices < JOIN_RADIUS_SAMPLES) | (
        indices >= max(0, sample_count - JOIN_RADIUS_SAMPLES)
    )
    join = np.zeros(sample_count, dtype=bool)
    for boundary in range(contract.stride_samples, sample_count, contract.stride_samples):
        join |= np.abs(indices - boundary) < JOIN_RADIUS_SAMPLES
    center = (
        (local >= int(round(contract.stride_samples * 0.10)))
        & (local < int(round(contract.stride_samples * 0.90)))
    )
    masks = {
        "song-edge-100ms": song_edge,
        "join-neighborhood-100ms": join & ~song_edge,
        "window-center-80pct": center & ~song_edge & ~join,
        "window-edge-10pct": ~(song_edge | join | center),
    }
    if any(int(mask.sum()) == 0 for mask in masks.values()):
        raise ValueError(
            f"Region partition has an empty region for {sample_count} samples: "
            f"{ {key: int(value.sum()) for key, value in masks.items()} }"
        )
    if not np.array_equal(
        np.sum(np.stack([masks[key] for key in REGIONS]), axis=0),
        np.ones(sample_count, dtype=np.int64),
    ):
        raise AssertionError("Region masks overlap or leave gaps")
    return masks


def region_for_sample(sample: int, sample_count: int, contract: ShortWindowContract) -> str:
    sample = min(max(int(sample), 0), sample_count - 1)
    song_edge = sample < JOIN_RADIUS_SAMPLES or sample >= sample_count - JOIN_RADIUS_SAMPLES
    if song_edge:
        return "song-edge-100ms"
    remainder = sample % contract.stride_samples
    near_join = remainder < JOIN_RADIUS_SAMPLES or (
        contract.stride_samples - remainder < JOIN_RADIUS_SAMPLES
    )
    if near_join:
        return "join-neighborhood-100ms"
    local = remainder
    if int(round(contract.stride_samples * 0.10)) <= local < int(
        round(contract.stride_samples * 0.90)
    ):
        return "window-center-80pct"
    return "window-edge-10pct"


def event_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "activeCount": 0}
    active = [row for row in rows if row["active"] > 0.5]
    if not active:
        return {"count": len(rows), "activeCount": 0}

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in active], dtype=np.float64)

    miss = values("missRms")
    projection = values("positiveProjection")
    return {
        "count": len(rows),
        "activeCount": len(active),
        "missRmsP95Dbfs": dbfs(float(np.percentile(miss, 95.0))),
        "missRmsMaxDbfs": dbfs(float(np.max(miss))),
        "positiveProjectionP95Dbfs": dbfs(float(np.percentile(projection, 95.0))),
        "positiveProjectionMaxDbfs": dbfs(float(np.max(projection))),
        "positiveAboveMinus35Dbfs": int(
            np.count_nonzero(projection >= 10.0 ** (-35.0 / 20.0))
        ),
        "positiveAboveMinus30Dbfs": int(
            np.count_nonzero(projection >= 10.0 ** (-30.0 / 20.0))
        ),
    }


def event_rows(
    candidate: np.ndarray,
    mixture: np.ndarray,
    teacher_instrumental: np.ndarray,
    contract: ShortWindowContract,
    milliseconds: int,
) -> dict[str, list[dict[str, float]]]:
    block_samples = max(1, round(SAMPLE_RATE * milliseconds / 1000.0))
    result = {"all": []}
    result.update({name: [] for name in REGIONS})
    for start in range(0, mixture.shape[0], block_samples):
        end = min(mixture.shape[0], start + block_samples)
        teacher_removed = mixture[start:end] - teacher_instrumental[start:end]
        miss = candidate[start:end] - teacher_instrumental[start:end]
        teacher_power = float(np.sum(teacher_removed.astype(np.float64) ** 2))
        miss_power = float(np.sum(miss.astype(np.float64) ** 2))
        teacher_rms = math.sqrt(teacher_power / max(teacher_removed.size, 1))
        coefficient = (
            float(np.sum(miss.astype(np.float64) * teacher_removed.astype(np.float64)))
            / teacher_power
            if teacher_power > 1.0e-30
            else 0.0
        )
        projection = max(coefficient, 0.0) * teacher_rms
        row = {
            "startSamples": float(start),
            "endSamples": float(end),
            "centerSamples": float((start + end) / 2.0),
            "missRms": math.sqrt(miss_power / max(miss.size, 1)),
            "positiveProjection": projection,
            "active": float(teacher_rms >= 10.0 ** (-60.0 / 20.0)),
        }
        result["all"].append(row)
        result[region_for_sample((start + end) // 2, mixture.shape[0], contract)].append(row)
    return result


def evaluate_candidate(
    candidate: np.ndarray,
    mixture: np.ndarray,
    vocals: np.ndarray,
    true_instrumental: np.ndarray,
    teacher_instrumental: np.ndarray,
    contract: ShortWindowContract,
) -> dict[str, Any]:
    if candidate.shape != mixture.shape:
        raise ValueError(f"Candidate shape mismatch: {candidate.shape} != {mixture.shape}")
    if not np.isfinite(candidate).all():
        raise ValueError("Candidate contains non-finite values")
    masks = region_masks(mixture.shape[0], contract)
    teacher_removed = mixture - teacher_instrumental
    teacher_miss = candidate - teacher_instrumental
    accumulators = {name: RegionAccumulator() for name in REGIONS}
    for name, mask in masks.items():
        accumulators[name].update(
            true_instrumental[mask],
            candidate[mask],
            vocals[mask],
            teacher_removed[mask],
            teacher_miss[mask],
        )
    events = {}
    for milliseconds in EVENT_MS:
        grouped = event_rows(
            candidate,
            mixture,
            teacher_instrumental,
            contract,
            milliseconds,
        )
        events[str(milliseconds)] = {
            name: event_summary(rows) for name, rows in grouped.items()
        }
    boundaries = tuple(
        range(contract.stride_samples, mixture.shape[0], contract.stride_samples)
    )
    return {
        "regions": {name: accumulator.result() for name, accumulator in accumulators.items()},
        "events": events,
        "wholeSong": {
            **RegionAccumulatorResult(
                true_instrumental,
                candidate,
                vocals,
                teacher_removed,
                teacher_miss,
            ),
            "seamMetrics": seam_metrics(candidate, boundaries),
        },
    }


def RegionAccumulatorResult(
    reference_instrumental: np.ndarray,
    candidate: np.ndarray,
    vocals: np.ndarray,
    teacher_removed: np.ndarray,
    teacher_miss: np.ndarray,
) -> dict[str, Any]:
    accumulator = RegionAccumulator()
    accumulator.update(
        reference_instrumental,
        candidate,
        vocals,
        teacher_removed,
        teacher_miss,
    )
    return accumulator.result()


def merge_region_results(values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in REGIONS:
        accumulator = RegionAccumulator()
        # Region reports contain sufficient statistics only indirectly, so
        # merge from raw per-song arrays in main instead of this helper.
        del accumulator
        result[name] = {
            "songCount": len(values),
            "reports": [value["regions"][name] for value in values],
        }
    return result


def load_h50_model(
    architecture_checkpoint: Path,
    state_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, architecture = pilot.make_model(architecture_checkpoint, device)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "local-inst3-vr-hard-sampling-checkpoint@1":
        raise ValueError(f"Unexpected H50 checkpoint format: {state_path}")
    state = payload.get("stateDict")
    if not isinstance(state, dict):
        raise ValueError(f"Missing H50 stateDict: {state_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        "architectureCheckpoint": architecture["checkpoint"],
        "statePath": str(state_path.resolve()),
        "stateSha256": sha256_file(state_path),
        "step": int(payload.get("step", -1)),
        "variant": payload.get("variant"),
        "runContractId": payload.get("runContractId"),
    }


def render_student(
    model: torch.nn.Module,
    audio: np.ndarray,
    contract: ShortWindowContract,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = np.empty_like(audio)
    window_count = math.ceil(audio.shape[0] / contract.stride_samples)
    boundaries = tuple(
        index * contract.stride_samples
        for index in range(1, window_count)
        if index * contract.stride_samples < audio.shape[0]
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, window_count, batch_size):
            starts = [
                index * contract.stride_samples
                for index in range(batch_start, min(window_count, batch_start + batch_size))
            ]
            lengths = [min(contract.stride_samples, audio.shape[0] - start) for start in starts]
            spectra = np.concatenate(
                [
                    stft_centered(
                        assemble_input(
                            audio,
                            start,
                            length,
                            contract,
                            mode="continuous",
                        ),
                        contract,
                    )
                    for start, length in zip(starts, lengths)
                ],
                axis=0,
            )
            predictions = model(torch.from_numpy(spectra).to(device)).detach().cpu().numpy()
            for offset, (start, length) in enumerate(zip(starts, lengths)):
                reconstructed = istft_centered(predictions[offset : offset + 1], contract)
                begin = contract.left_context_samples
                output[start : start + length] = reconstructed[begin : begin + length]
    if not np.isfinite(output).all():
        raise ValueError("Student residual contains non-finite values")
    return output, {
        "windowCount": window_count,
        "boundaries": len(boundaries),
        "elapsedSeconds": time.perf_counter() - started,
        "inferenceBatchSize": batch_size,
    }


def merge_event_summaries(
    per_song: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for milliseconds in EVENT_MS:
        result[str(milliseconds)] = {}
        for region in ("all", *REGIONS):
            rows: list[dict[str, float]] = []
            for song in per_song:
                summary = song["events"][str(milliseconds)][region]
                # Reconstruct summary aggregates is not possible from a p95;
                # per-song event rows are retained separately by the runner.
                del summary
            result[str(milliseconds)][region] = {"songCount": len(per_song)}
    return result


def comparison_summary(aggregates: dict[str, Any]) -> dict[str, Any]:
    """Return H50-minus-initial deltas for the matched continuous paths."""
    initial = aggregates["initial-continuous"]
    h50 = aggregates["H50-continuous"]
    metric_names = (
        "instrumentalSdrDb",
        "accompanimentErrorRmsDbfs",
        "positiveVocalProjectionDbfs",
        "teacherResidualMissRmsDbfs",
        "teacherRemovedPositiveProjectionDbfs",
    )
    result = {
        "H50-minus-initial": {
            "wholeSong": {
                metric: float(h50["wholeSong"][metric] - initial["wholeSong"][metric])
                for metric in metric_names
            },
            "regions": {},
            "events": {},
        }
    }
    for region in REGIONS:
        result["H50-minus-initial"]["regions"][region] = {
            "mean": {
                metric: (
                    None
                    if h50["regions"][region]["mean"][metric] is None
                    or initial["regions"][region]["mean"][metric] is None
                    else float(
                        h50["regions"][region]["mean"][metric]
                        - initial["regions"][region]["mean"][metric]
                    )
                )
                for metric in metric_names
            },
            "median": {
                metric: (
                    None
                    if h50["regions"][region]["median"][metric] is None
                    or initial["regions"][region]["median"][metric] is None
                    else float(
                        h50["regions"][region]["median"][metric]
                        - initial["regions"][region]["median"][metric]
                    )
                )
                for metric in metric_names
            },
        }
    for milliseconds in EVENT_MS:
        result["H50-minus-initial"]["events"][str(milliseconds)] = {}
        for region in ("all", *REGIONS):
            initial_event = initial["events"][str(milliseconds)][region]
            h50_event = h50["events"][str(milliseconds)][region]
            result["H50-minus-initial"]["events"][str(milliseconds)][region] = {
                metric: (
                    None
                    if metric not in initial_event or metric not in h50_event
                    else float(h50_event[metric] - initial_event[metric])
                )
                for metric in (
                    "missRmsP95Dbfs",
                    "missRmsMaxDbfs",
                    "positiveProjectionP95Dbfs",
                    "positiveProjectionMaxDbfs",
                    "positiveAboveMinus35Dbfs",
                    "positiveAboveMinus30Dbfs",
                )
            }
    return result


def write_audio(path: Path, audio: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frames": int(audio.shape[0]),
        "channels": int(audio.shape[1]),
        "sampleRate": SAMPLE_RATE,
    }


def evaluate_musdb(
    *,
    args: argparse.Namespace,
    contract: ShortWindowContract,
    initial_model: torch.nn.Module,
    h50_model: torch.nn.Module,
    entries: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    per_song: dict[str, Any] = {}
    aggregate_raw: dict[str, dict[str, dict[str, list[float]]]] = {
        model: {region: {metric: [] for metric in (
            "instrumentalSdrDb",
            "accompanimentErrorRmsDbfs",
            "positiveVocalProjectionDbfs",
            "teacherResidualMissRmsDbfs",
            "teacherRemovedPositiveProjectionDbfs",
        )} for region in REGIONS}
        for model in MODELS
    }
    aggregate_events: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        model: {str(ms): {region: [] for region in ("all", *REGIONS)} for ms in EVENT_MS}
        for model in MODELS
    }
    whole_accumulators = {
        model: RegionAccumulator() for model in MODELS
    }
    started = time.perf_counter()
    for index, entry in enumerate(entries, start=1):
        song = hard.load_eval_song(args.eval_root.resolve(), entry)
        print(f"continuous eval {index}/{len(entries)}: {entry['member']}", flush=True)
        initial_residual, initial_render = render_student(
            initial_model,
            song.mixture_gt,
            contract,
            args.device_object,
            args.inference_batch_size,
        )
        h50_residual, h50_render = render_student(
            h50_model,
            song.mixture_gt,
            contract,
            args.device_object,
            args.inference_batch_size,
        )
        candidates = {
            "initial-continuous": song.mixture_gt - initial_residual,
            "H50-continuous": song.mixture_gt - h50_residual,
            "Inst3-native": song.teacher_instrumental,
        }
        song_report: dict[str, Any] = {
            "role": song.role,
            "member": song.member,
            "slug": song.slug,
            "samples": int(song.mixture_gt.shape[0]),
            "durationSeconds": song.mixture_gt.shape[0] / SAMPLE_RATE,
            "renders": {
                "initial-continuous": initial_render,
                "H50-continuous": h50_render,
            },
            "candidates": {},
        }
        for model_name, candidate in candidates.items():
            evaluated = evaluate_candidate(
                candidate,
                song.mixture_gt,
                song.vocals,
                song.instrumental,
                song.teacher_instrumental,
                contract,
            )
            song_report["candidates"][model_name] = evaluated
            whole_accumulators[model_name].update(
                song.instrumental,
                candidate,
                song.vocals,
                song.mixture_gt - song.teacher_instrumental,
                candidate - song.teacher_instrumental,
            )
            for region in REGIONS:
                for metric in aggregate_raw[model_name][region]:
                    value = evaluated["regions"][region].get(metric)
                    if value is not None and math.isfinite(float(value)):
                        aggregate_raw[model_name][region][metric].append(float(value))
            for milliseconds in EVENT_MS:
                # Store block rows transiently in the report-free accumulator.
                grouped = event_rows(
                    candidate,
                    song.mixture_gt,
                    song.teacher_instrumental,
                    contract,
                    milliseconds,
                )
                for region, rows in grouped.items():
                    aggregate_events[model_name][str(milliseconds)][region].extend(rows)
        song_report["h50VsInitial"] = {
            region: {
                "snrDb": snr(
                    candidates["initial-continuous"][region_masks(song.mixture_gt.shape[0], contract)[region]],
                    candidates["H50-continuous"][region_masks(song.mixture_gt.shape[0], contract)[region]],
                ),
                "correlation": correlation(
                    candidates["initial-continuous"][region_masks(song.mixture_gt.shape[0], contract)[region]],
                    candidates["H50-continuous"][region_masks(song.mixture_gt.shape[0], contract)[region]],
                ),
            }
            for region in REGIONS
        }
        per_song[song.slug] = song_report
        del song, initial_residual, h50_residual, candidates
        gc.collect()
        if args.device_object.type == "cuda":
            torch.cuda.empty_cache()

    aggregates: dict[str, Any] = {}
    for model_name in MODELS:
        aggregates[model_name] = {
            "wholeSong": whole_accumulators[model_name].result(),
            "regions": {
                region: {
                    "songCount": len(entries),
                    "mean": {
                        metric: float(np.mean(values)) if values else None
                        for metric, values in aggregate_raw[model_name][region].items()
                    },
                    "median": {
                        metric: float(np.median(values)) if values else None
                        for metric, values in aggregate_raw[model_name][region].items()
                    },
                }
                for region in REGIONS
            },
            "events": {
                str(milliseconds): {
                    region: event_summary(rows)
                    for region, rows in aggregate_events[model_name][str(milliseconds)].items()
                }
                for milliseconds in EVENT_MS
            },
        }
    return {
        "songCount": len(entries),
        "songs": per_song,
        "aggregates": aggregates,
        "comparisons": comparison_summary(aggregates),
        "elapsedSeconds": time.perf_counter() - started,
    }


def render_private(
    *,
    args: argparse.Namespace,
    contract: ShortWindowContract,
    initial_model: torch.nn.Module,
    h50_model: torch.nn.Module,
    output_root: Path,
) -> dict[str, Any]:
    songs = density.validate_private_songs(args.samples_root.resolve())
    session, providers = pilot.make_teacher_session(
        pilot.DEFAULT_TEACHER,
        args.threads,
        args.require_teacher_cuda,
    )
    report: dict[str, Any] = {
        "songCount": len(songs),
        "providers": providers,
        "models": {},
        "songs": {},
    }
    try:
        for index, song_info in enumerate(songs, start=1):
            name = song_info["name"]
            source, sample_rate = listening.load_audio(Path(song_info["file"]))
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f"Unexpected sample rate for {name}: {sample_rate}")
            print(f"private render {index}/{len(songs)}: {name}", flush=True)
            initial_residual, initial_timing = render_student(
                initial_model,
                source,
                contract,
                args.device_object,
                args.inference_batch_size,
            )
            h50_private_path = args.h50_private_instrumental_root.resolve() / f"{name}.flac"
            if h50_private_path.is_file() and not args.force:
                h50_instrumental, h50_rate = listening.load_audio(h50_private_path)
                if h50_rate != SAMPLE_RATE or h50_instrumental.shape != source.shape:
                    raise ValueError(f"Invalid retained H50 continuous file: {h50_private_path}")
                h50_residual = np.ascontiguousarray(source - h50_instrumental, dtype=np.float32)
                h50_timing = {
                    "reusedContinuousRender": str(h50_private_path.resolve()),
                    "assembly": "continuous-context-overlap-save",
                }
            else:
                h50_residual, h50_timing = render_student(
                    h50_model,
                    source,
                    contract,
                    args.device_object,
                    args.inference_batch_size,
                )
            teacher_instrumental, teacher_vocals, teacher_timing = teacher_oracle.render_teacher_audio(
                session,
                source,
                progress_label=f"private/{name}/Inst3",
            )
            candidates = {
                "initial-continuous": {
                    "residual": initial_residual,
                    "instrumental": source - initial_residual,
                    "timing": initial_timing,
                },
                "H50-continuous": {
                    "residual": h50_residual,
                    "instrumental": source - h50_residual,
                    "timing": h50_timing,
                },
                "Inst3-native": {
                    "residual": teacher_vocals,
                    "instrumental": teacher_instrumental,
                    "timing": teacher_timing,
                },
            }
            song_report = {
                "source": {
                    **song_info,
                    "sampleRate": sample_rate,
                    "frames": int(source.shape[0]),
                    "durationSeconds": source.shape[0] / SAMPLE_RATE,
                },
                "candidates": {},
            }
            for model_name, values in candidates.items():
                instrumental_path = output_root / "private-listening-12" / model_name / f"{name}-instrumental.flac"
                residual_path = output_root / "private-listening-12" / model_name / f"{name}-residual.flac"
                song_report["candidates"][model_name] = {
                    "instrumental": write_audio(instrumental_path, values["instrumental"]),
                    "residual": write_audio(residual_path, values["residual"]),
                    "timing": values["timing"],
                }
            report["songs"][name] = song_report
            json_write(output_root / "private-render-report.json", report)
            del source, initial_residual, h50_residual, teacher_instrumental, teacher_vocals, candidates
            gc.collect()
            if args.device_object.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del session
    return report


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.inference_batch_size <= 0:
        raise ValueError("inference-batch-size must be positive")
    if args.max_eval_songs is not None and args.max_eval_songs <= 0:
        raise ValueError("max-eval-songs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    args.device_object = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for path in (args.checkpoint, args.h50_checkpoint, args.manifest):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    entries = sorted(
        [entry for entry in manifest["entries"] if entry["role"] in {"calibration", "internal-test"}],
        key=lambda item: (item["role"], item["member"]),
    )
    if args.max_eval_songs is not None:
        entries = entries[: args.max_eval_songs]
    if not entries:
        raise ValueError("No evaluation entries")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.private_only:
        report_path = output_root / "continuous-baseline-report.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"--private-only requires an existing report: {report_path}"
            )
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        initial_model, initial_meta = pilot.make_model(args.checkpoint.resolve(), args.device_object)
        h50_model, h50_meta = load_h50_model(
            args.checkpoint.resolve(), args.h50_checkpoint.resolve(), args.device_object
        )
        try:
            existing_report["privateListening"] = render_private(
                args=args,
                contract=ShortWindowContract(
                    num_frames=128,
                    left_context_hops=5,
                    right_context_hops=5,
                ),
                initial_model=initial_model,
                h50_model=h50_model,
                output_root=output_root,
            )
            existing_report["privateListeningUpdatedAt"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            json_write(report_path, existing_report)
        finally:
            del initial_model, h50_model, initial_meta, h50_meta
            if args.device_object.type == "cuda":
                torch.cuda.empty_cache()
        print(json.dumps({"status": "completed", "report": str(report_path.resolve())}, indent=2))
        return 0
    contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    initial_model, initial_meta = pilot.make_model(args.checkpoint.resolve(), args.device_object)
    initial_model.eval()
    h50_model, h50_meta = load_h50_model(
        args.checkpoint.resolve(), args.h50_checkpoint.resolve(), args.device_object
    )
    started = time.perf_counter()
    try:
        musdb_report = evaluate_musdb(
            args=args,
            contract=contract,
            initial_model=initial_model,
            h50_model=h50_model,
            entries=entries,
            output_root=output_root,
        )
        private_report = None
        if not args.skip_private:
            private_report = render_private(
                args=args,
                contract=contract,
                initial_model=initial_model,
                h50_model=h50_model,
                output_root=output_root,
            )
    finally:
        del initial_model, h50_model
        if args.device_object.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "local-inst3-continuous-baseline-evaluation@1",
        "status": "completed",
        "licenseDisposition": {
            "scope": "local non-commercial research only",
            "musdb18": "educational/non-commercial source audio; not redistributed",
            "inst3": "teacher weight and derived outputs remain local; not published",
        },
        "contract": contract.as_dict(assembly="continuous-context-overlap-save"),
        "regionContract": {
            "regions": list(REGIONS),
            "joinRadiusSamples": JOIN_RADIUS_SAMPLES,
            "joinRadiusMilliseconds": 100,
            "centerFraction": [0.10, 0.90],
            "eventBlockMilliseconds": list(EVENT_MS),
            "eventAssignment": "event center; song edge takes precedence over join, then center, then window edge",
        },
        "inputs": {
            "manifest": {
                "file": str(args.manifest.resolve()),
                "sha256": sha256_file(args.manifest.resolve()),
                "roles": sorted({entry["role"] for entry in entries}),
                "songCount": len(entries),
            },
            "originalCheckpoint": {
                "file": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint.resolve()),
            },
            "h50Checkpoint": h50_meta,
            "teacherContract": "uvr_mdxnet_inst_3@2 native oracle cache/private renderer",
            "retainedH50PrivateRoot": str(args.h50_private_instrumental_root.resolve()),
        },
        "models": {
            "initial-continuous": initial_meta["checkpoint"],
            "H50-continuous": h50_meta,
            "Inst3-native": {
                "semantic": "native MDX instrumental",
                "contract": "UVR-MDX-NET Inst 3 7680 FFT / 1024 hop / native trim",
            },
        },
        "runtime": {
            "device": str(args.device_object),
            "threads": args.threads,
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "musdbEvaluation": musdb_report,
        "privateListening": private_report,
        "elapsedSeconds": time.perf_counter() - started,
        "notes": {
            "assembly": "Initial and H50 use identical continuous context. Inst 3 uses its native full-song MDX assembly.",
            "metrics": "Instrumental SDR is waveform energy-SNR; teacher-residual metrics are aggressive-removal diagnostics, not perceptual scores.",
            "boundary": "The report separates song edges, join neighborhoods, useful-window center, and useful-window edges; no claim is made that a region is pure vocal or pure accompaniment.",
        },
    }
    json_write(output_root / "continuous-baseline-report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str((output_root / "continuous-baseline-report.json").resolve()),
                "songs": len(entries),
                "privateSongs": 0 if private_report is None else private_report["songCount"],
                "device": str(args.device_object),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
