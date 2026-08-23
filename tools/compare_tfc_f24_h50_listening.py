#!/usr/bin/env python3
"""Compare F24-static-H50 and 128-frame H50 private listening outputs.

The retained 128-frame H50 files were rendered with isolated, zero-padded
windows.  F24-static-H50 uses continuous overlap-save context.  This tool
therefore also renders a matched 128-frame continuous control from the same
H50 checkpoint, then reports whole-song, short-block, residual-projection,
and boundary diagnostics for all three paths.

The private songs have no isolated vocal ground truth.  Projection and
residual metrics in this report are relative diagnostics, not SDR claims.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import render_inst3_objective_listening as listening
import run_inst3_scale10_density as density
import run_inst3_tfc_short_window_retrain as f24
import run_inst3_vr_hard_sampling as hard
from tfc_tdf_short_window import ShortWindowContract, render_with_backend, seam_metrics
from tfc_tdf_short_window import sha256_array, sha256_file, write_flac


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_ROOT = ROOT / "data" / "samples"
DEFAULT_F24_ROOT = ROOT / "data" / "musdb18-inst3-f24-retrain" / "listening-12"
DEFAULT_H50_ROOT = ROOT / "data" / "musdb18-inst3-vr-hard-sampling-h50" / "listening-12"
DEFAULT_H50_CHECKPOINT = (
    ROOT
    / "data"
    / "musdb18-inst3-vr-hard-sampling-h50"
    / "runs"
    / "V-R-H50"
    / "step-8000.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "musdb18-inst3-f24-h50-comparison"

F24_NAME = "F24-static-H50"
H128_ISOLATED = "H50-128-isolated"
H128_CONTINUOUS = "H50-128-continuous"
BLOCK_MS = (50, 100, 200)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--f24-root", type=Path, default=DEFAULT_F24_ROOT)
    parser.add_argument("--h50-root", type=Path, default=DEFAULT_H50_ROOT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render the matched 128-frame continuous control and stop before comparison",
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


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def dbfs(value: float, floor: float = -240.0) -> float:
    if not math.isfinite(value) or value <= 10.0 ** (floor / 20.0):
        return floor
    return 20.0 * math.log10(value)


def rms(value: np.ndarray) -> float:
    return math.sqrt(float(np.mean(value.astype(np.float64) ** 2)))


def rms_db(value: np.ndarray) -> float:
    return dbfs(rms(value))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64).reshape(-1)
    b = right.astype(np.float64).reshape(-1)
    denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / max(denominator, 1.0e-30))


def snr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    error = candidate.astype(np.float64) - ref
    return 10.0 * math.log10(
        max(float(np.sum(ref * ref)), 1.0e-30)
        / max(float(np.sum(error * error)), 1.0e-30)
    )


def projection_rms(candidate: np.ndarray, basis: np.ndarray) -> float:
    """Positive coherent projection of candidate onto a reference residual."""
    cand = candidate.astype(np.float64)
    ref = basis.astype(np.float64)
    basis_power = float(np.sum(ref * ref))
    if basis_power <= 1.0e-20:
        return 0.0
    coefficient = float(np.sum(cand * ref) / basis_power)
    return max(coefficient, 0.0) * math.sqrt(basis_power / ref.size)


def projection_db(candidate: np.ndarray, basis: np.ndarray) -> float:
    value = projection_rms(candidate, basis)
    return dbfs(value)


def block_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    h128_instrumental: np.ndarray,
    milliseconds: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    block_samples = max(1, round(sample_rate * milliseconds / 1000.0))
    rows: list[dict[str, Any]] = []
    for start in range(0, source.shape[0], block_samples):
        end = min(source.shape[0], start + block_samples)
        source_block = source[start:end]
        candidate_block = candidate[start:end]
        basis_block = h128_instrumental[start:end]
        basis_rms = rms(basis_block)
        if basis_rms < 10.0 ** (-60.0 / 20.0):
            continue
        candidate_residual = source_block - candidate_block
        h128_residual = source_block - basis_block
        rows.append(
            {
                "startSamples": start,
                "endSamples": end,
                "startSeconds": start / sample_rate,
                "basisRmsDbfs": rms_db(basis_block),
                "candidateResidualRmsDbfs": rms_db(candidate_residual),
                "h128ResidualRmsDbfs": rms_db(h128_residual),
                "residualEnergyDeltaDb": rms_db(candidate_residual)
                - rms_db(h128_residual),
                "candidateInstrumentalProjectionDb": projection_db(
                    candidate_block, h128_residual
                ),
                "h128InstrumentalProjectionDb": projection_db(
                    basis_block, h128_residual
                ),
                "projectionDeltaDb": projection_db(candidate_block, h128_residual)
                - projection_db(basis_block, h128_residual),
                "candidateVsH128DeltaRmsDbfs": rms_db(candidate_block - basis_block),
            }
        )
    return rows


def summarize_blocks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    projection = values("projectionDeltaDb")
    residual = values("residualEnergyDeltaDb")
    difference = values("candidateVsH128DeltaRmsDbfs")
    top_projection = sorted(rows, key=lambda row: row["projectionDeltaDb"])[:12]
    top_residual = sorted(rows, key=lambda row: row["residualEnergyDeltaDb"])[:12]
    return {
        "count": len(rows),
        "projectionDeltaDb": {
            "mean": float(np.mean(projection)),
            "median": float(np.median(projection)),
            "p05": float(np.percentile(projection, 5)),
            "p95": float(np.percentile(projection, 95)),
            "lowerThanZeroCount": int(np.count_nonzero(projection < 0.0)),
            "higherThanZeroCount": int(np.count_nonzero(projection > 0.0)),
            "minimum": float(np.min(projection)),
            "maximum": float(np.max(projection)),
        },
        "residualEnergyDeltaDb": {
            "mean": float(np.mean(residual)),
            "median": float(np.median(residual)),
            "p05": float(np.percentile(residual, 5)),
            "p95": float(np.percentile(residual, 95)),
            "lowerThanZeroCount": int(np.count_nonzero(residual < 0.0)),
            "higherThanZeroCount": int(np.count_nonzero(residual > 0.0)),
        },
        "differenceRmsDbfs": {
            "mean": float(np.mean(difference)),
            "median": float(np.median(difference)),
            "p95": float(np.percentile(difference, 95)),
        },
        "topProjectionReductions": top_projection,
        "topResidualReductions": top_residual,
    }


def output_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    h128_instrumental: np.ndarray,
    contract: ShortWindowContract,
) -> dict[str, Any]:
    residual = source - candidate
    h128_residual = source - h128_instrumental
    derivative = np.diff(candidate, axis=0)
    source_derivative = np.diff(source, axis=0)
    return {
        "frames": int(candidate.shape[0]),
        "rmsDbfs": rms_db(candidate),
        "residualRmsDbfs": rms_db(residual),
        "h128ResidualRmsDbfs": rms_db(h128_residual),
        "residualRmsDeltaVsH128Db": rms_db(residual) - rms_db(h128_residual),
        "candidateVsH128SnrDb": snr(h128_instrumental, candidate),
        "candidateVsH128Correlation": correlation(candidate, h128_instrumental),
        "candidateVsH128DeltaRmsDbfs": rms_db(candidate - h128_instrumental),
        "candidateInstrumentalProjectionOnH128ResidualDb": projection_db(
            candidate, h128_residual
        ),
        "h128InstrumentalProjectionOnOwnResidualDb": projection_db(
            h128_instrumental, h128_residual
        ),
        "derivativeRmsDbfs": rms_db(derivative),
        "sourceDerivativeRmsDbfs": rms_db(source_derivative),
        "derivativeDeltaVsSourceDb": rms_db(derivative) - rms_db(source_derivative),
        "seams": seam_metrics(candidate, tuple(range(contract.stride_samples, candidate.shape[0], contract.stride_samples))),
        "contract": contract.as_dict(),
    }


def pair_metrics(left: np.ndarray, right: np.ndarray, source: np.ndarray) -> dict[str, float]:
    """Compare two accompaniment tracks and their derived residuals."""
    left_residual = source - left
    right_residual = source - right
    return {
        "leftVsRightSnrDb": snr(right, left),
        "leftVsRightCorrelation": correlation(left, right),
        "leftMinusRightDeltaRmsDbfs": rms_db(left - right),
        "leftResidualMinusRightResidualDeltaRmsDbfs": rms_db(
            left_residual - right_residual
        ),
        "leftResidualRmsDeltaDb": rms_db(left_residual) - rms_db(right_residual),
    }


def make_render_backend(model: torch.nn.Module, device: torch.device):
    def backend(spectrum: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(spectrum).to(device)).detach().cpu().numpy()

    return backend


def derive_instrumental(source: np.ndarray, vocals: np.ndarray) -> np.ndarray:
    if source.shape != vocals.shape:
        raise ValueError(f"Source/vocals shape mismatch: {source.shape} != {vocals.shape}")
    instrumental = np.ascontiguousarray(source - vocals, dtype=np.float32)
    if not np.isfinite(instrumental).all():
        raise ValueError("Derived instrumental contains non-finite values")
    return instrumental


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    samples_root = args.samples_root.resolve()
    f24_root = args.f24_root.resolve()
    h50_root = args.h50_root.resolve()
    checkpoint = args.h50_checkpoint.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    songs = density.validate_private_songs(samples_root)
    f24_contract = ShortWindowContract(num_frames=24, left_context_hops=4, right_context_hops=4)
    h128_contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
    model, model_metadata = f24.make_short_model(checkpoint, 128, device)
    model.eval()
    backend = make_render_backend(model, device)
    continuous_root = output_root / "h50-128-continuous"
    continuous_instrumental_root = output_root / "h50-128-continuous-instrumental"
    listening_root = output_root / "listening-12"
    report: dict[str, Any] = {
        "schema": "local-inst3-f24-h50-comparison@1",
        "status": "running",
        "contracts": {
            "f24": f24_contract.as_dict(),
            "h128": h128_contract.as_dict(assembly="continuous-context-overlap-save"),
            "h128Existing": "isolated-zero-padded windows from retained H50 listening renderer",
        },
        "inputs": {
            "h50Checkpoint": checkpoint_metadata(checkpoint),
            "f24Root": str(f24_root),
            "h50Root": str(h50_root),
            "sourceRoot": str(samples_root),
        },
        "model": model_metadata,
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "threads": args.threads,
        },
        "songs": {},
    }
    try:
        for index, song in enumerate(songs, start=1):
            name = song["name"]
            source, sample_rate = listening.load_audio(Path(song["file"]))
            if sample_rate != f24_contract.sample_rate:
                raise ValueError(f"Unexpected source rate for {name}")
            if args.render_only:
                started = time.perf_counter()
                rendered_continuous = render_with_backend(
                    source,
                    h128_contract,
                    backend,
                    assembly_mode="continuous",
                )
                continuous_path = continuous_root / f"{name}.flac"
                continuous_instrumental = derive_instrumental(
                    source, rendered_continuous.audio
                )
                continuous_instrumental_path = (
                    continuous_instrumental_root / f"{name}.flac"
                )
                if not continuous_path.is_file() or args.force:
                    write_flac(
                        continuous_path,
                        rendered_continuous.audio,
                        sample_rate,
                    )
                if not continuous_instrumental_path.is_file() or args.force:
                    write_flac(
                        continuous_instrumental_path,
                        continuous_instrumental,
                        sample_rate,
                    )
                report["songs"][name] = {
                    "source": {
                        **song,
                        "sampleRate": sample_rate,
                        "frames": int(source.shape[0]),
                        "decodedFloat32Sha256": sha256_array(source),
                    },
                    "continuousRender": {
                        "vocalsFile": str(continuous_path.resolve()),
                        "vocalsSha256": sha256_file(continuous_path),
                        "instrumentalFile": str(continuous_instrumental_path.resolve()),
                        "instrumentalSha256": sha256_file(continuous_instrumental_path),
                        "windowCount": rendered_continuous.window_count,
                        "elapsedSeconds": rendered_continuous.elapsed_seconds,
                        "wallSeconds": time.perf_counter() - started,
                    },
                }
                json_write(output_root / "comparison-report.json", report)
                print(f"rendered {index}/{len(songs)}: {name}", flush=True)
                del source, continuous_instrumental, rendered_continuous
                continue
            f24_path = f24_root / F24_NAME / f"{name}.flac"
            h128_path = h50_root / "V-R-H50" / f"{name}.flac"
            if not f24_path.is_file() or not h128_path.is_file():
                raise FileNotFoundError(f"Missing retained output for {name}")
            f24_audio, f24_rate = listening.load_audio(f24_path)
            h128_isolated, h128_rate = listening.load_audio(h128_path)
            if f24_rate != sample_rate or h128_rate != sample_rate:
                raise ValueError(f"Sample rate mismatch for {name}")
            if f24_audio.shape != source.shape or h128_isolated.shape != source.shape:
                raise ValueError(f"Frame shape mismatch for {name}")
            started = time.perf_counter()
            rendered_continuous = render_with_backend(
                source,
                h128_contract,
                backend,
                assembly_mode="continuous",
            )
            h128_continuous = np.ascontiguousarray(rendered_continuous.audio, dtype=np.float32)
            h128_continuous_instrumental = derive_instrumental(source, h128_continuous)
            continuous_path = continuous_root / f"{name}.flac"
            continuous_instrumental_path = continuous_instrumental_root / f"{name}.flac"
            if not continuous_path.is_file() or args.force:
                write_flac(continuous_path, h128_continuous, sample_rate)
            if not continuous_instrumental_path.is_file() or args.force:
                write_flac(
                    continuous_instrumental_path,
                    h128_continuous_instrumental,
                    sample_rate,
                )
            # Render the isolated control once to verify that the retained
            # 128-frame files were produced by the expected assembly path.
            rendered_isolated = render_with_backend(
                source,
                h128_contract,
                backend,
                assembly_mode="isolated",
            )
            h128_isolated_reference_vocals = np.ascontiguousarray(
                rendered_isolated.audio, dtype=np.float32
            )
            h128_isolated_reference = derive_instrumental(
                source, h128_isolated_reference_vocals
            )
            verification = {
                "renderedIsolatedVsRetainedSnrDb": snr(h128_isolated, h128_isolated_reference),
                "renderedIsolatedVsRetainedDeltaRmsDbfs": rms_db(h128_isolated_reference - h128_isolated),
                "renderedIsolatedVsRetainedCorrelation": correlation(h128_isolated_reference, h128_isolated),
            }
            variants = {
                F24_NAME: f24_audio,
                H128_ISOLATED: h128_isolated,
                H128_CONTINUOUS: h128_continuous_instrumental,
            }
            h128_basis = h128_isolated
            song_report: dict[str, Any] = {
                "source": {
                    **song,
                    "sampleRate": sample_rate,
                    "frames": int(source.shape[0]),
                    "decodedFloat32Sha256": sha256_array(source),
                },
                "retainedOutputs": {
                    "f24": {"file": str(f24_path), "sha256": sha256_file(f24_path)},
                    "h128Isolated": {"file": str(h128_path), "sha256": sha256_file(h128_path)},
                },
                "continuousRender": {
                    "vocalsFile": str(continuous_path.resolve()),
                    "vocalsSha256": sha256_file(continuous_path),
                    "instrumentalFile": str(continuous_instrumental_path.resolve()),
                    "instrumentalSha256": sha256_file(continuous_instrumental_path),
                    "windowCount": rendered_continuous.window_count,
                    "elapsedSeconds": rendered_continuous.elapsed_seconds,
                    "wallSeconds": time.perf_counter() - started,
                },
                "assemblyVerification": verification,
                "directComparisons": {
                    "f24VsH128Continuous": pair_metrics(
                        f24_audio, h128_continuous_instrumental, source
                    ),
                    "h128ContinuousVsH128Isolated": pair_metrics(
                        h128_continuous_instrumental, h128_isolated, source
                    ),
                    "f24VsH128Isolated": pair_metrics(
                        f24_audio, h128_isolated, source
                    ),
                },
                "variants": {},
                "blocks": {},
            }
            for variant, audio in variants.items():
                song_report["variants"][variant] = output_metrics(
                    source,
                    audio,
                    h128_basis,
                    f24_contract if variant == F24_NAME else h128_contract,
                )
                for milliseconds in BLOCK_MS:
                    rows = block_metrics(
                        source,
                        audio,
                        h128_basis,
                        milliseconds,
                        sample_rate,
                    )
                    song_report["blocks"].setdefault(str(milliseconds), {})[variant] = {
                        "summary": summarize_blocks(rows),
                        "rows": rows,
                    }
            report["songs"][name] = song_report
            json_write(output_root / "comparison-report.json", report)
            print(f"compared {index}/{len(songs)}: {name}", flush=True)
            del source, f24_audio, h128_isolated, h128_continuous
            del h128_continuous_instrumental
            del h128_isolated_reference, h128_isolated_reference_vocals
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.render_only:
        report["status"] = "render-completed"
        report["outputCount"] = len(report["songs"])
        report["notes"] = {
            "purpose": "Matched 128-frame H50 continuous-context listening control",
            "nextStep": "Run without --render-only to perform F24/128-frame comparison analysis.",
        }
        json_write(output_root / "comparison-report.json", report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str((output_root / "comparison-report.json").resolve()),
                    "songs": len(report["songs"]),
                    "continuousRoot": str(continuous_root.resolve()),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    # Aggregate per-song summary only; keep the detailed rows in each song.
    aggregate: dict[str, Any] = {"variants": {}, "blocks": {}}
    aggregate["directComparisons"] = {}
    for comparison in (
        "f24VsH128Continuous",
        "h128ContinuousVsH128Isolated",
        "f24VsH128Isolated",
    ):
        values = [
            value["directComparisons"][comparison]
            for value in report["songs"].values()
        ]
        aggregate["directComparisons"][comparison] = {
            "meanLeftVsRightSnrDb": float(
                np.mean([item["leftVsRightSnrDb"] for item in values])
            ),
            "medianLeftVsRightSnrDb": float(
                np.median([item["leftVsRightSnrDb"] for item in values])
            ),
            "meanLeftVsRightCorrelation": float(
                np.mean([item["leftVsRightCorrelation"] for item in values])
            ),
            "meanLeftMinusRightDeltaRmsDbfs": float(
                np.mean([item["leftMinusRightDeltaRmsDbfs"] for item in values])
            ),
            "meanLeftResidualRmsDeltaDb": float(
                np.mean([item["leftResidualRmsDeltaDb"] for item in values])
            ),
        }
    for variant in (F24_NAME, H128_CONTINUOUS):
        metrics = [value["variants"][variant] for value in report["songs"].values()]
        aggregate["variants"][variant] = {
            "meanCandidateVsH128SnrDb": float(np.mean([m["candidateVsH128SnrDb"] for m in metrics])),
            "medianCandidateVsH128SnrDb": float(np.median([m["candidateVsH128SnrDb"] for m in metrics])),
            "meanCandidateVsH128DeltaRmsDbfs": float(np.mean([m["candidateVsH128DeltaRmsDbfs"] for m in metrics])),
            "meanResidualRmsDeltaVsH128Db": float(np.mean([m["residualRmsDeltaVsH128Db"] for m in metrics])),
            "meanDerivativeDeltaVsSourceDb": float(np.mean([m["derivativeDeltaVsSourceDb"] for m in metrics])),
            "seamP95Ratios": [m["seams"].get("p95Ratio") for m in metrics],
        }
    for milliseconds in BLOCK_MS:
        aggregate["blocks"][str(milliseconds)] = {}
        for variant in (F24_NAME, H128_CONTINUOUS):
            summaries = [
                value["blocks"][str(milliseconds)][variant]["summary"]
                for value in report["songs"].values()
            ]
            aggregate["blocks"][str(milliseconds)][variant] = {
                "meanProjectionDeltaDb": float(np.mean([s["projectionDeltaDb"]["mean"] for s in summaries])),
                "medianProjectionDeltaDb": float(np.median([s["projectionDeltaDb"]["median"] for s in summaries])),
                "songsWithLowerProjectionMean": int(sum(s["projectionDeltaDb"]["mean"] < 0 for s in summaries)),
                "meanResidualEnergyDeltaDb": float(np.mean([s["residualEnergyDeltaDb"]["mean"] for s in summaries])),
                "songsWithLowerResidualEnergyMean": int(sum(s["residualEnergyDeltaDb"]["mean"] < 0 for s in summaries)),
            }
    report["aggregate"] = aggregate
    report["status"] = "completed"
    report["outputCount"] = len(report["songs"])
    report["notes"] = {
        "interpretation": "F24/H128 comparisons use H128 isolated residual as a relative vocal-like basis; no private-song stem ground truth is available.",
        "assemblyControl": "H128-continuous separates context assembly effects from the F24 time-dimension effect; retained H128 files are isolated-window outputs.",
        "subjectiveClaims": "Lower candidate instrumental projection onto H128 residual can be consistent with fewer retained vocal-like leaks, but may also reflect accompaniment damage.",
    }
    json_write(output_root / "comparison-report.json", report)
    print(json.dumps({"status": report["status"], "report": str((output_root / "comparison-report.json").resolve()), "songs": len(report["songs"]), "continuousRoot": str(continuous_root.resolve()), "continuousInstrumentalRoot": str(continuous_instrumental_root.resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
