#!/usr/bin/env python3
"""Summarize Android gfxinfo framestats captured after a reset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


PROFILE_MARKER = "---PROFILEDATA---"


def read_android_aggregate(lines: list[str]) -> dict[str, object] | None:
    patterns = {
        "totalFrames": re.compile(r"^Total frames rendered:\s*(\d+)$"),
        "modernJankyFrames": re.compile(r"^Janky frames:\s*(\d+)\s*\(([\d.]+)%\)$"),
        "p50Ms": re.compile(r"^50th percentile:\s*(\d+)ms$"),
        "p90Ms": re.compile(r"^90th percentile:\s*(\d+)ms$"),
        "p95Ms": re.compile(r"^95th percentile:\s*(\d+)ms$"),
        "p99Ms": re.compile(r"^99th percentile:\s*(\d+)ms$"),
        "missedVsync": re.compile(r"^Number Missed Vsync:\s*(\d+)$"),
        "highInputLatency": re.compile(r"^Number High input latency:\s*(\d+)$"),
        "slowUiThread": re.compile(r"^Number Slow UI thread:\s*(\d+)$"),
        "slowIssueDrawCommands": re.compile(r"^Number Slow issue draw commands:\s*(\d+)$"),
    }
    required = set(patterns) | {"modernJankPercent"}
    for start, line in enumerate(lines):
        if not patterns["totalFrames"].match(line.strip()):
            continue
        values: dict[str, object] = {}
        for candidate in lines[start:]:
            stripped = candidate.strip()
            if stripped.startswith("HISTOGRAM:"):
                break
            for key, pattern in patterns.items():
                if key in values:
                    continue
                match = pattern.match(stripped)
                if not match:
                    continue
                if key == "modernJankyFrames":
                    values[key] = int(match.group(1))
                    values["modernJankPercent"] = float(match.group(2))
                else:
                    values[key] = int(match.group(1))
        if required.issubset(values):
            return values
    return None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = math.ceil(fraction * len(values)) - 1
    return sorted(values)[max(0, min(index, len(values) - 1))]


def read_frames(path: Path) -> list[dict[str, int]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    frames: list[dict[str, int]] = []
    inside_profile = False
    header: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if stripped == PROFILE_MARKER:
            inside_profile = not inside_profile
            header = None
            continue
        if not inside_profile or not stripped:
            continue
        row = next(csv.reader([stripped]))
        if row[0] == "Flags":
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        try:
            frame = {
                name: int(value)
                for name, value in zip(header, row)
                if name
            }
        except ValueError:
            continue
        duration_ns = frame["FrameCompleted"] - frame["IntendedVsync"]
        if frame["Flags"] == 0 and 0 < duration_ns < 10_000_000_000:
            frames.append(frame)
    return frames


def summarize(frames: list[dict[str, int]]) -> dict[str, object]:
    durations_ms = [
        (frame["FrameCompleted"] - frame["IntendedVsync"]) / 1_000_000
        for frame in frames
    ]
    if frames:
        interval_ms = (
            max(frame["FrameCompleted"] for frame in frames)
            - min(frame["IntendedVsync"] for frame in frames)
        ) / 1_000_000
    else:
        interval_ms = 0.0
    return {
        "frameCount": len(frames),
        "intervalMs": interval_ms,
        "framesPerSecond": len(frames) * 1000 / interval_ms if interval_ms > 0 else 0.0,
        "durationMs": {
            "mean": sum(durations_ms) / len(durations_ms) if durations_ms else 0.0,
            "p50": percentile(durations_ms, 0.50),
            "p90": percentile(durations_ms, 0.90),
            "p95": percentile(durations_ms, 0.95),
            "p99": percentile(durations_ms, 0.99),
            "maximum": max(durations_ms, default=0.0),
        },
        "thresholdCounts": {
            "over20Ms": sum(duration > 20 for duration in durations_ms),
            "over50Ms": sum(duration > 50 for duration in durations_ms),
            "over100Ms": sum(duration > 100 for duration in durations_ms),
            "over200Ms": sum(duration > 200 for duration in durations_ms),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("framestats", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lines = args.framestats.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    android_aggregate = read_android_aggregate(lines)
    report = {
        "androidAggregate": android_aggregate,
        "framestatsTail": summarize(read_frames(args.framestats)),
        "note": (
            "Android aggregate metrics come from the first complete contiguous "
            "gfxinfo aggregate block and cover its full Stats-since interval. "
            "The framestats tail may be truncated by Android's ring buffer and "
            "must not be treated as full-sweep FPS."
            if android_aggregate is not None
            else "No complete Android aggregate block was found. The framestats "
            "tail may be truncated by Android's ring buffer and must not be "
            "treated as full-sweep FPS."
        ),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
