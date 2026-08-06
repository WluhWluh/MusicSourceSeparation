#!/usr/bin/env python3
"""Validate canonical HTDemucs E2E WAV outputs and playback timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import wave
from pathlib import Path


STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
SAMPLE_RATE = 44_100
CHANNELS = 2
SAMPLE_WIDTH = 2
SEEK_FRAMES = 4_096
SEAM_RADIUS = 2_048


def parse_stems(value: str) -> tuple[str, ...]:
    stems = tuple(part.strip() for part in value.split(",") if part.strip())
    if not stems:
        raise argparse.ArgumentTypeError("Provide at least one stem")
    if len(stems) != len(set(stems)):
        raise argparse.ArgumentTypeError("Stem names must be unique")
    unknown = sorted(set(stems) - set(STEMS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stems: {', '.join(unknown)}")
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument(
        "--stems",
        type=parse_stems,
        default=STEMS,
        help="Comma-separated stem order (default: six-stem HTDemucs order)",
    )
    return parser.parse_args()


def percentile_nearest_rank(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def pcm_frames(payload: bytes) -> list[tuple[int, int]]:
    if len(payload) % (CHANNELS * SAMPLE_WIDTH):
        raise ValueError("PCM payload is not aligned to stereo PCM16 frames")
    return list(struct.iter_unpack("<hh", payload))


def seek_evidence(reader: wave.Wave_read, target: int, frame_count: int) -> dict:
    reader.setpos(target)
    payload = reader.readframes(min(SEEK_FRAMES, frame_count - target))
    frames = pcm_frames(payload)
    return {
        "targetFrame": target,
        "framesRead": len(frames),
        "byteSize": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nonZeroSamples": sum(sample != 0 for frame in frames for sample in frame),
    }


def seam_evidence(reader: wave.Wave_read, boundary: int, frame_count: int) -> dict:
    start = max(0, boundary - SEAM_RADIUS)
    end = min(frame_count, boundary + SEAM_RADIUS + 1)
    reader.setpos(start)
    frames = pcm_frames(reader.readframes(end - start))
    seam_index = boundary - start
    channels = []
    for channel in range(CHANNELS):
        differences = [
            abs(frames[index][channel] - frames[index - 1][channel])
            for index in range(1, len(frames))
        ]
        boundary_delta = abs(
            frames[seam_index][channel] - frames[seam_index - 1][channel]
        )
        local_without_boundary = differences[: seam_index - 1] + differences[seam_index:]
        local_p95 = percentile_nearest_rank(local_without_boundary, 0.95)
        channels.append(
            {
                "channel": channel,
                "boundaryDeltaPcm16": boundary_delta,
                "localDerivativeP95Pcm16": local_p95,
                "deltaToLocalP95Ratio": boundary_delta / max(1, local_p95),
            }
        )
    return {"boundaryFrame": boundary, "channels": channels}


def validate_wav(path: Path, expected_frames: int, stride: int) -> dict:
    with wave.open(str(path), "rb") as reader:
        metadata = {
            "channels": reader.getnchannels(),
            "sampleWidthBytes": reader.getsampwidth(),
            "sampleRate": reader.getframerate(),
            "frameCount": reader.getnframes(),
            "compression": reader.getcomptype(),
        }
        expected = {
            "channels": CHANNELS,
            "sampleWidthBytes": SAMPLE_WIDTH,
            "sampleRate": SAMPLE_RATE,
            "frameCount": expected_frames,
            "compression": "NONE",
        }
        if metadata != expected:
            raise ValueError(f"Unexpected WAV contract for {path}: {metadata}")
        targets = sorted({0, expected_frames // 2, max(0, expected_frames - SEEK_FRAMES)})
        seeks = [seek_evidence(reader, target, expected_frames) for target in targets]
        seams = [
            seam_evidence(reader, boundary, expected_frames)
            for boundary in range(stride, expected_frames, stride)
        ]
    ratios = [
        channel["deltaToLocalP95Ratio"]
        for seam in seams
        for channel in seam["channels"]
    ]
    return {
        "path": str(path.resolve()),
        "metadata": metadata,
        "seekChecks": seeks,
        "seams": seams,
        "seamSummary": {
            "count": len(seams),
            "maximumDeltaToLocalP95Ratio": max(ratios, default=0.0),
            "meanDeltaToLocalP95Ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        },
    }


def completed_attempt(report: dict) -> dict:
    attempts = [attempt for attempt in report["attempts"] if attempt["status"] == "complete"]
    if len(attempts) != 1:
        raise ValueError(f"Expected one completed attempt, found {len(attempts)}")
    return attempts[0]


def playback_projection(attempt: dict, ready_window_count: int) -> dict:
    windows = attempt["windows"]
    if len(windows) < ready_window_count:
        return {
            "readyWindowCount": ready_window_count,
            "supported": False,
            "reason": "track-has-fewer-windows-than-ready-threshold",
        }
    completion_ms = []
    elapsed = 0.0
    for window in windows:
        elapsed += window["total"]["wallMs"]
        completion_ms.append(elapsed)
    start_index = ready_window_count - 1
    buffered_frames = sum(
        window["finalizedFrames"] for window in windows[:ready_window_count]
    )
    minimum_before_refill = buffered_frames / SAMPLE_RATE
    underruns = []
    previous_completion = completion_ms[start_index]
    for index in range(ready_window_count, len(windows)):
        interval_seconds = (completion_ms[index] - previous_completion) / 1000.0
        buffered_frames -= interval_seconds * SAMPLE_RATE
        margin_seconds = buffered_frames / SAMPLE_RATE
        minimum_before_refill = min(minimum_before_refill, margin_seconds)
        if buffered_frames < 0:
            underruns.append(
                {
                    "beforeWindowIndex": index,
                    "deficitSeconds": -margin_seconds,
                }
            )
            buffered_frames = 0.0
        buffered_frames += windows[index]["finalizedFrames"]
        previous_completion = completion_ms[index]
    prepare_ms = attempt["prepare"]["total"]["wallMs"]
    return {
        "readyWindowCount": ready_window_count,
        "supported": True,
        "notAudioTrackObservation": True,
        "model": "discrete-window-producer-versus-wall-clock-consumer",
        "producerWallMsToPlaybackStart": prepare_ms + completion_ms[start_index],
        "bufferedAudioSecondsAtStart": sum(
            window["finalizedFrames"] for window in windows[:ready_window_count]
        )
        / SAMPLE_RATE,
        "minimumBufferBeforeRefillSeconds": minimum_before_refill,
        "projectedUnderrunCount": len(underruns),
        "projectedUnderruns": underruns,
    }


def sampled_system_evidence(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    patterns = {
        "nativeHeapPssKb": re.compile(r"Native Heap:\s+(\d+)"),
        "graphicsPssKb": re.compile(r"Graphics:\s+(\d+)"),
        "totalPssKb": re.compile(r"TOTAL PSS:\s+(\d+)"),
        "totalRssKb": re.compile(r"TOTAL RSS:\s+(\d+)"),
        "memAvailableKb": re.compile(r"MemAvailable:\s+(\d+)"),
        "swapFreeKb": re.compile(r"SwapFree:\s+(\d+)"),
    }
    series = {name: [] for name in patterns}
    thermal_statuses = []
    thermal_payloads = []
    for row in rows:
        payload = f"{row.get('memory', '')} | {row.get('system', '')}"
        for name, pattern in patterns.items():
            match = pattern.search(payload)
            if match:
                series[name].append(int(match.group(1)))
        thermal = row.get("thermal", "")
        status = re.search(r"Thermal Status:\s+(\d+)", thermal)
        if status:
            thermal_statuses.append(int(status.group(1)))
        if thermal:
            thermal_payloads.append(thermal)
    return {
        "sampleCount": len(rows),
        "intervalWarning": "ADB sampling perturbs the measured run; use clean runs for latency.",
        "peak": {
            name: max(values) if values else None
            for name, values in series.items()
            if name not in {"memAvailableKb", "swapFreeKb"}
        },
        "minimum": {
            name: min(series[name]) if series[name] else None
            for name in ("memAvailableKb", "swapFreeKb")
        },
        "thermalStatusValues": sorted(set(thermal_statuses)),
        "thermalPayloadDistinctCount": len(set(thermal_payloads)),
        "thermalSensorLimitation": (
            "thermalservice exposes duplicate AP/BAT/SKIN names and cached values; "
            "only thermal severity is treated as reliable"
        ),
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    attempt = completed_attempt(report)
    expected_frames = report["source"]["selectedFrames"]
    stride = report["contract"]["strideSamples"]
    outputs = {
        stem: validate_wav(args.output_dir / f"{stem}.wav", expected_frames, stride)
        for stem in args.stems
    }
    result = {
        "schemaVersion": 1,
        "sourceReport": str(args.report.resolve()),
        "sourceRunId": report["runId"],
        "status": "passed",
        "scope": {
            "seek": "random-access-file-read",
            "seams": "pcm16-boundary-delta-versus-local-derivative-p95",
            "playback": "timing-projection-not-audiotrack-observation",
        },
        "stemOrder": list(args.stems),
        "outputs": outputs,
        "playbackBufferProjections": [
            playback_projection(attempt, ready_window_count=1),
            playback_projection(attempt, ready_window_count=2),
        ],
        "sampledSystem": sampled_system_evidence(args.samples) if args.samples else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
