#!/usr/bin/env python3
"""Render the most salient continuous H50 training events for listening.

The event records come from the continuous H50 training selection.  For each
selected event this tool writes identical PCM snippets for the mixture, the
Inst 3 target, and the original/continuation checkpoints.  The snippets are
local non-commercial research artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

import run_inst3_distill_pilot as pilot
import run_inst3_vr_continuous_topk_local as local
from tfc_tdf_short_window import ShortWindowContract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local" / "selection.json"
DEFAULT_CACHE_ROOT = ROOT / "data" / "musdb18-inst3-vr-continuous-topk-local"
DEFAULT_ORACLE_ROOT = ROOT / "data" / "musdb18-inst3-oracle"
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_H50 = (
    ROOT / "data" / "musdb18-inst3-vr-hard-sampling-h50" / "runs" / "V-R-H50" / "step-8000.pt"
)
DEFAULT_SOURCE_CONTINUATION = (
    DEFAULT_CACHE_ROOT / "runs" / "H50-continuation" / "step-800.pt"
)
DEFAULT_CONTINUATION_ROOT = ROOT / "data" / "musdb18-inst3-vr-continuation"
DEFAULT_OUTPUT = ROOT / "data" / "musdb18-inst3-training-event-listening"
SAMPLE_RATE = 44_100
CONTEXT_SECONDS = 1.0
MODEL_VARIANTS = (
    "H50-pass-50",
    "H50-continuation@step-800",
    "H50-continuation+1",
    "H50-continuation+3",
    "H50-continuation+5",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h50-checkpoint", type=Path, default=DEFAULT_H50)
    parser.add_argument("--source-continuation", type=Path, default=DEFAULT_SOURCE_CONTINUATION)
    parser.add_argument("--continuation-root", type=Path, default=DEFAULT_CONTINUATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-events", type=int, default=40)
    parser.add_argument("--context-seconds", type=float, default=CONTEXT_SECONDS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return pilot.sha256_file(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "event"


def load_source_audio(oracle_root: Path, slug: str) -> np.ndarray:
    path = oracle_root / "decoded" / "train" / slug / "decoded.npz"
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["sampleRate"]) != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate for {slug}")
    with np.load(path) as values:
        return np.ascontiguousarray(values["mixtureGt"], dtype=np.float32)


def select_events(selection: dict[str, Any], top_count: int) -> list[dict[str, Any]]:
    if top_count <= 0:
        raise ValueError("top-events must be positive")
    rows: list[dict[str, Any]] = []
    for slug, song in selection["songs"].items():
        for record_index, record in enumerate(song["selectedRecords"]):
            rows.append(
                {
                    "slug": slug,
                    "recordIndex": record_index,
                    **record,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["positiveProjectionRmsDbfs"]),
            -float(row["missRmsDbfs"]),
            row["slug"],
            int(row["eventCenterSamples"]),
        )
    )
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int, int, int]] = set()
    for row in rows:
        key = (
            row["slug"],
            int(row["startSamples"]),
            int(row["eventStartSamples"]),
            int(row["eventEndSamples"]),
        )
        if key in used:
            continue
        used.add(key)
        selected.append(row)
        if len(selected) >= top_count:
            break
    for index, row in enumerate(selected, start=1):
        row["globalRank"] = index
    if not selected:
        raise ValueError("No selected events")
    return selected


def snippet_bounds(
    event: dict[str, Any], source_samples: int, context_samples: int
) -> tuple[int, int, int, int]:
    window_start = int(event["startSamples"])
    window_length = int(event["lengthSamples"])
    event_center = window_start + int(event["eventCenterSamples"]) - window_start
    # eventCenterSamples is absolute in the selection metadata.
    event_center = int(event["eventCenterSamples"])
    start = max(window_start, event_center - context_samples)
    end = min(window_start + window_length, event_center + context_samples)
    start = max(0, start)
    end = min(source_samples, end)
    if end <= start:
        raise ValueError(f"Invalid snippet bounds for {event['slug']}: {start}:{end}")
    event_start = max(start, int(event["startSamples"]) + int(event["eventStartSamples"]))
    event_end = min(end, int(event["startSamples"]) + int(event["eventEndSamples"]))
    return start, end, event_start - start, event_end - start


def write_flac(path: Path, audio: np.ndarray) -> dict[str, Any]:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected stereo audio, got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return {
        "file": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "frames": int(audio.shape[0]),
        "sampleRate": SAMPLE_RATE,
        "channels": 2,
        "peak": float(np.max(np.abs(audio))),
    }


def checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.continuation_root.resolve() / "runs" / "H50-continuation-plus5"
    return {
        "H50-pass-50": args.h50_checkpoint.resolve(),
        "H50-continuation@step-800": args.source_continuation.resolve(),
        "H50-continuation+1": root / "step-960.pt",
        "H50-continuation+3": root / "step-1280.pt",
        "H50-continuation+5": root / "step-1600.pt",
    }


def load_model(
    checkpoint: Path,
    architecture_checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    model, _ = local.load_h50_model(architecture_checkpoint, checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predict_residual(
    model: torch.nn.Module,
    input_specs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True, device=device)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, input_specs.shape[0], 4):
            batch = torch.from_numpy(input_specs[begin : begin + 4]).to(device)
            full = local.torch_packed_istft(model(batch), window)
            trim = pilot.DEFAULT_CONFIG.trim_samples
            outputs.append(
                full[:, trim : trim + pilot.DEFAULT_CONFIG.useful_samples]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(outputs, axis=0)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_events <= 0 or args.context_seconds <= 0 or args.threads <= 0:
        raise ValueError("top-events, context-seconds, and threads must be positive")
    if args.inference_batch_size <= 0:
        raise ValueError("inference-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    events = select_events(selection, args.top_events)
    paths = checkpoint_paths(args)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    context_samples = round(args.context_seconds * SAMPLE_RATE)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_store = local.CacheStore(
        {
            slug: args.cache_root.resolve() / "cache" / "train" / f"{slug}.npz"
            for slug in selection["songs"]
        },
        max_open=4,
    )
    models: dict[str, torch.nn.Module] = {}
    for variant, path in paths.items():
        print(f"load model {variant}: {path.name}", flush=True)
        models[variant] = load_model(path, args.checkpoint.resolve(), device)
    report: dict[str, Any] = {
        "schema": "local-inst3-training-event-listening@1",
        "status": "running",
        "selection": {
            "file": str(args.selection.resolve()),
            "sha256": sha256_file(args.selection.resolve()),
            "topEvents": args.top_events,
            "contextSeconds": args.context_seconds,
            "eventRanking": "continuous H50 positiveProjectionRmsDbfs descending, missRmsDbfs tie-break",
        },
        "models": {variant: {"file": str(path.resolve()), "sha256": sha256_file(path)} for variant, path in paths.items()},
        "events": [],
        "runtime": {"device": str(device), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    for event in events:
        slug = event["slug"]
        source = load_source_audio(args.oracle_root.resolve(), slug)
        cache = cache_store.load(slug)
        record_index = int(event["recordIndex"])
        input_spec = cache["inputSpec"][record_index : record_index + 1]
        target_residual = cache["targetAudio"][record_index]
        window_start = int(event["startSamples"])
        window_source = source[window_start : window_start + pilot.DEFAULT_CONFIG.useful_samples]
        if window_source.shape[0] != pilot.DEFAULT_CONFIG.useful_samples:
            raise ValueError(f"Short source window for {slug}")
        target_instrumental = np.ascontiguousarray(window_source - target_residual, dtype=np.float32)
        start, end, event_local_start, event_local_end = snippet_bounds(
            event, source.shape[0], context_samples
        )
        local_start = start - window_start
        local_end = end - window_start
        prefix = f"event-{int(event['globalRank']):03d}-{safe_name(slug)}"
        event_report: dict[str, Any] = {
            "rank": int(event["globalRank"]),
            "slug": slug,
            "recordIndex": record_index,
            "durationMs": int(event["durationMs"]),
            "eventScoreDbfs": float(event["positiveProjectionRmsDbfs"]),
            "eventMissDbfs": float(event["missRmsDbfs"]),
            "windowStartSamples": window_start,
            "snippetStartSamples": start,
            "snippetEndSamples": end,
            "snippetDurationSeconds": (end - start) / SAMPLE_RATE,
            "eventStartInSnippet": event_local_start,
            "eventEndInSnippet": event_local_end,
            "outputs": {},
        }
        event_report["outputs"]["mixture"] = write_flac(
            output_root / "mixture" / f"{prefix}-mixture.flac", source[start:end]
        )
        event_report["outputs"]["target-instrumental"] = write_flac(
            output_root / "target-instrumental" / f"{prefix}-target-instrumental.flac",
            target_instrumental[local_start:local_end],
        )
        event_report["outputs"]["target-residual"] = write_flac(
            output_root / "target-residual" / f"{prefix}-target-residual.flac",
            target_residual[local_start:local_end],
        )
        for variant, model in models.items():
            residual_window = predict_residual(model, input_spec, device)[0]
            instrumental_window = np.ascontiguousarray(window_source - residual_window, dtype=np.float32)
            event_report["outputs"][variant] = {
                "instrumental": write_flac(
                    output_root / variant / f"{prefix}-instrumental.flac",
                    instrumental_window[local_start:local_end],
                ),
                "residual": write_flac(
                    output_root / variant / f"{prefix}-residual.flac",
                    residual_window[local_start:local_end],
                ),
            }
        report["events"].append(event_report)
        json_write(output_root / "event-listening-report.json", report)
        print(f"render event {event['globalRank']}/{len(events)}: {slug}", flush=True)
        del source, cache, input_spec, target_residual, target_instrumental, window_source
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for model in models.values():
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["status"] = "completed"
    report["outputCount"] = sum(
        1
        for event in report["events"]
        for value in event["outputs"].values()
        for _ in (value if isinstance(value, dict) and "file" not in value else (value,))
    )
    json_write(output_root / "event-listening-report.json", report)
    print(json.dumps({"status": report["status"], "events": len(events), "report": str((output_root / "event-listening-report.json").resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
