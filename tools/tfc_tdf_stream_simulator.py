#!/usr/bin/env python3
"""Host-only dry/wet stream scheduler for the compact TFC-TDF experiment.

This module deliberately does not run LiteRT. It models the scheduling and
timeline contract around the current fixed-window TFC-TDF producer so that
dry fallback, late windows, seek epochs, and bounded memory can be tested
without an Android device or a real-time audio clock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TfcTdfWindowContract:
    """The fixed-window and valid-output contract of the current candidate."""

    sample_rate: int = 44_100
    input_samples: int = 130_048
    trim_samples: int = 5_120
    useful_samples: int = 119_808
    playback_block_samples: int = 1_024

    @property
    def input_channels(self) -> int:
        return 2

    def window_count_for(self, track_samples: int) -> int:
        if track_samples <= 0:
            raise ValueError("track_samples must be positive")
        return math.ceil(track_samples / self.useful_samples)

    def plan(self, window_index: int, track_samples: int, epoch: int) -> "WindowPlan":
        if window_index < 0:
            raise ValueError("window_index must be non-negative")
        start = window_index * self.useful_samples
        if start >= track_samples:
            raise ValueError("window_index is past the track")
        actual = min(self.useful_samples, track_samples - start)
        return WindowPlan(
            window_index=window_index,
            epoch=epoch,
            start_sample=start,
            actual_samples=actual,
        )


@dataclass(frozen=True)
class WindowPlan:
    window_index: int
    epoch: int
    start_sample: int
    actual_samples: int

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.actual_samples


@dataclass(frozen=True)
class SimulationCommand:
    """A command applied before the output block at ``at_wall_ms``."""

    at_wall_ms: float
    kind: str
    value: Any = None


@dataclass(frozen=True)
class OutputBlock:
    wall_ms: float
    segment: int
    position_sample: int
    samples: np.ndarray
    mode: str
    reason: str
    epoch: int
    model_id: str


@dataclass(frozen=True)
class TraceEvent:
    wall_ms: float
    event: str
    segment: int
    position_sample: int
    epoch: int
    mode: str | None = None
    reason: str | None = None
    window_index: int | None = None
    model_id: str | None = None


@dataclass
class _WindowTask:
    plan: WindowPlan
    model_id: str
    ready_wall_ms: float
    wet_samples: np.ndarray


class BoundedSampleRing:
    """A small non-overlapping block ring with explicit eviction accounting."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self._blocks: OrderedDict[int, np.ndarray] = OrderedDict()
        self.evicted_blocks = 0
        self.max_sample_count = 0
        self.max_block_count = 0

    @property
    def sample_count(self) -> int:
        return sum(int(value.shape[0]) for value in self._blocks.values())

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()

    def discard_before(self, position_sample: int) -> None:
        for start in list(self._blocks):
            value = self._blocks[start]
            end = start + value.shape[0]
            if end <= position_sample:
                del self._blocks[start]
                continue
            if start < position_sample:
                self._blocks[position_sample] = np.ascontiguousarray(
                    value[position_sample - start :]
                )
                del self._blocks[start]
        self._blocks = OrderedDict(sorted(self._blocks.items()))

    def write(self, start_sample: int, samples: np.ndarray) -> None:
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError("ring samples must have shape [samples, channels]")
        value = np.ascontiguousarray(samples, dtype=np.float32)
        end_sample = start_sample + value.shape[0]
        for block_start, existing in self._blocks.items():
            existing_end = block_start + existing.shape[0]
            if start_sample < existing_end and block_start < end_sample:
                raise ValueError(
                    f"overlapping ring blocks at samples {start_sample}:{end_sample} "
                    f"and {block_start}:{existing_end}"
                )
        self._blocks[start_sample] = value
        self._blocks = OrderedDict(sorted(self._blocks.items()))
        while self.sample_count > self.capacity_samples and self._blocks:
            self._blocks.popitem(last=False)
            self.evicted_blocks += 1
        self.max_sample_count = max(self.max_sample_count, self.sample_count)
        self.max_block_count = max(self.max_block_count, self.block_count)

    def read(self, start_sample: int, sample_count: int) -> np.ndarray | None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        position = start_sample
        remaining = sample_count
        chunks: list[np.ndarray] = []
        for block_start, value in self._blocks.items():
            block_end = block_start + value.shape[0]
            if block_end <= position:
                continue
            if block_start > position:
                break
            offset = position - block_start
            length = min(remaining, value.shape[0] - offset)
            chunks.append(value[offset : offset + length])
            position += length
            remaining -= length
            if remaining == 0:
                if len(chunks) == 1:
                    return np.ascontiguousarray(chunks[0])
                return np.ascontiguousarray(np.concatenate(chunks, axis=0))
        return None


@dataclass
class SimulationResult:
    contract: TfcTdfWindowContract
    output_blocks: list[OutputBlock]
    events: list[TraceEvent]
    stats: dict[str, Any]

    def validate(self) -> None:
        previous_by_segment: dict[int, int] = {}
        for block in self.output_blocks:
            if block.samples.ndim != 2 or block.samples.shape[1] != self.contract.input_channels:
                raise AssertionError("output block has an invalid channel shape")
            if block.samples.shape[0] <= 0:
                raise AssertionError("output block must not be empty")
            if block.samples.shape[0] > self.contract.playback_block_samples:
                raise AssertionError("output block exceeds playback block size")
            if block.position_sample < 0:
                raise AssertionError("output block starts before the track")
            if block.position_sample + block.samples.shape[0] > self.stats["trackSamples"]:
                raise AssertionError("output block extends beyond the track")
            if block.mode not in {"dry", "wet"}:
                raise AssertionError(f"unknown output mode: {block.mode}")
            expected = previous_by_segment.get(block.segment)
            if expected is not None and expected != block.position_sample:
                raise AssertionError(
                    f"gap or duplicate in segment {block.segment}: "
                    f"expected {expected}, got {block.position_sample}"
                )
            if not np.isfinite(block.samples).all():
                raise AssertionError("output contains non-finite samples")
            previous_by_segment[block.segment] = (
                block.position_sample + block.samples.shape[0]
            )

    def mode_ranges(self) -> list[dict[str, Any]]:
        ranges: list[dict[str, Any]] = []
        for block in self.output_blocks:
            end = block.position_sample + block.samples.shape[0]
            if (
                ranges
                and ranges[-1]["segment"] == block.segment
                and ranges[-1]["mode"] == block.mode
                and ranges[-1]["epoch"] == block.epoch
                and ranges[-1]["modelId"] == block.model_id
                and ranges[-1]["endSample"] == block.position_sample
            ):
                ranges[-1]["endSample"] = end
                ranges[-1]["blockCount"] += 1
            else:
                ranges.append(
                    {
                        "segment": block.segment,
                        "mode": block.mode,
                        "startSample": block.position_sample,
                        "endSample": end,
                        "blockCount": 1,
                        "epoch": block.epoch,
                        "modelId": block.model_id,
                    }
                )
        return ranges

    def trace_text(self) -> str:
        lines: list[str] = []
        for item in self.mode_ranges():
            start = item["startSample"] / self.contract.sample_rate
            end = item["endSample"] / self.contract.sample_rate
            lines.append(
                f"{start:8.3f}s-{end:8.3f}s "
                f"segment={item['segment']} mode={item['mode']} "
                f"blocks={item['blockCount']} epoch={item['epoch']}"
            )
        return "\n".join(lines)

    def rendered_sha256(self) -> str:
        digest = hashlib.sha256()
        for block in self.output_blocks:
            digest.update(np.ascontiguousarray(block.samples, dtype="<f4").tobytes())
        return digest.hexdigest()


class TfcTdfStreamSimulator:
    """Deterministic playback-clock simulator with dry fallback."""

    def __init__(
        self,
        source: np.ndarray,
        *,
        contract: TfcTdfWindowContract | None = None,
        read_ahead_rate: float = 8.0,
        inference_latency_ms: float = 100.0,
        worker_count: int = 1,
        ring_capacity_windows: int = 3,
        model_scales: dict[str, float] | None = None,
        initial_model_id: str = "tfc-tdf-current",
    ) -> None:
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("source must have shape [samples, 2]")
        if source.shape[0] == 0:
            raise ValueError("source must not be empty")
        if read_ahead_rate <= 0:
            raise ValueError("read_ahead_rate must be positive")
        if inference_latency_ms < 0:
            raise ValueError("inference_latency_ms must be non-negative")
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if ring_capacity_windows <= 0:
            raise ValueError("ring_capacity_windows must be positive")

        self.source = np.ascontiguousarray(source, dtype=np.float32)
        self.contract = contract or TfcTdfWindowContract()
        self.read_ahead_rate = read_ahead_rate
        self.inference_latency_ms = inference_latency_ms
        self.worker_count = worker_count
        self.ring_capacity_samples = (
            ring_capacity_windows * self.contract.useful_samples
        )
        self.model_scales = model_scales or {
            "tfc-tdf-current": 0.25,
            "tfc-tdf-next": 0.20,
        }
        self.initial_model_id = initial_model_id
        if initial_model_id not in self.model_scales:
            raise ValueError(f"missing scale for model {initial_model_id}")

    def _make_wet_samples(self, plan: WindowPlan, model_id: str) -> np.ndarray:
        scale = self.model_scales[model_id]
        return np.ascontiguousarray(
            self.source[plan.start_sample : plan.end_sample] * scale,
            dtype=np.float32,
        )

    def run(
        self,
        commands: Iterable[SimulationCommand] = (),
        *,
        max_wall_ms: float | None = None,
    ) -> SimulationResult:
        ordered_commands = sorted(commands, key=lambda item: item.at_wall_ms)
        if any(item.at_wall_ms < 0 for item in ordered_commands):
            raise ValueError("commands cannot be scheduled before time zero")

        dry_ring = BoundedSampleRing(self.ring_capacity_samples)
        wet_ring = BoundedSampleRing(self.ring_capacity_samples)
        events: list[TraceEvent] = []
        output_blocks: list[OutputBlock] = []
        pending_tasks: list[_WindowTask] = []
        command_index = 0
        epoch = 0
        segment = 0
        playhead = 0
        wall_ms = 0.0
        playing = True
        enabled = True
        model_id = self.initial_model_id
        last_mode: str | None = None
        dropped_epoch_tasks = 0
        late_windows = 0
        ready_windows = 0
        max_pending_tasks = 0
        next_window_index = 0
        window_count = self.contract.window_count_for(self.source.shape[0])
        worker_available: list[float] = []
        session_start_wall_ms = 0.0
        session_analysis_start = 0

        def schedule_session(start_sample: int, now_ms: float) -> None:
            nonlocal epoch, pending_tasks, playhead, model_id
            nonlocal dropped_epoch_tasks, max_pending_tasks
            nonlocal next_window_index, worker_available
            nonlocal session_start_wall_ms, session_analysis_start, last_mode
            epoch += 1
            dropped_epoch_tasks += len(pending_tasks)
            pending_tasks = []
            wet_ring.clear()
            playhead = start_sample
            first_index = start_sample // self.contract.useful_samples
            session_start_wall_ms = now_ms
            session_analysis_start = first_index * self.contract.useful_samples
            next_window_index = first_index
            worker_available = [now_ms] * self.worker_count
            last_mode = None
            enqueue_until_horizon(now_ms)
            events.append(
                TraceEvent(
                    wall_ms=now_ms,
                    event="session_started",
                    segment=segment,
                    position_sample=start_sample,
                    epoch=epoch,
                    model_id=model_id,
                )
            )

        def enqueue_until_horizon(now_ms: float) -> None:
            nonlocal next_window_index, max_pending_tasks
            horizon_end = min(
                self.source.shape[0], playhead + self.ring_capacity_samples
            )
            while (
                next_window_index < window_count
                and next_window_index * self.contract.useful_samples < horizon_end
            ):
                plan = self.contract.plan(
                    next_window_index, self.source.shape[0], epoch
                )
                input_ready = session_start_wall_ms + (
                    (plan.end_sample - session_analysis_start)
                    / self.contract.sample_rate
                    / self.read_ahead_rate
                    * 1000.0
                )
                worker_index = min(
                    range(self.worker_count), key=lambda index: worker_available[index]
                )
                task_start = max(now_ms, input_ready, worker_available[worker_index])
                ready_ms = task_start + self.inference_latency_ms
                worker_available[worker_index] = ready_ms
                pending_tasks.append(
                    _WindowTask(
                        plan=plan,
                        model_id=model_id,
                        ready_wall_ms=ready_ms,
                        wet_samples=self._make_wet_samples(plan, model_id),
                    )
                )
                next_window_index += 1
            max_pending_tasks = max(max_pending_tasks, len(pending_tasks))

        def publish_ready(now_ms: float) -> None:
            nonlocal pending_tasks, ready_windows, late_windows, max_pending_tasks
            remaining: list[_WindowTask] = []
            for task in pending_tasks:
                if task.ready_wall_ms > now_ms:
                    remaining.append(task)
                    continue
                plan = task.plan
                if plan.epoch != epoch:
                    continue
                if plan.end_sample <= playhead:
                    late_windows += 1
                    events.append(
                        TraceEvent(
                            wall_ms=now_ms,
                            event="window_late",
                            segment=segment,
                            position_sample=playhead,
                            epoch=epoch,
                            window_index=plan.window_index,
                            model_id=task.model_id,
                            reason="valid-output-already-played",
                        )
                    )
                    continue
                if plan.end_sample > playhead + self.ring_capacity_samples:
                    remaining.append(task)
                    continue
                start = max(plan.start_sample, playhead)
                samples = task.wet_samples[start - plan.start_sample :]
                if samples.shape[0] > 0:
                    wet_ring.write(start, samples)
                    ready_windows += 1
                    events.append(
                        TraceEvent(
                            wall_ms=now_ms,
                            event="window_ready",
                            segment=segment,
                            position_sample=start,
                            epoch=epoch,
                            window_index=plan.window_index,
                            model_id=task.model_id,
                        )
                    )
            pending_tasks = remaining
            max_pending_tasks = max(max_pending_tasks, len(pending_tasks))

        def apply_command(command: SimulationCommand) -> None:
            nonlocal command_index, playing, enabled, model_id, segment, playhead
            nonlocal epoch, dropped_epoch_tasks, last_mode
            kind = command.kind
            if kind == "seek":
                target = int(command.value)
                if target < 0 or target >= self.source.shape[0]:
                    raise ValueError(f"seek target outside track: {target}")
                segment += 1
                playhead = target
                schedule_session(target, wall_ms)
                event_name = "seek"
            elif kind == "pause":
                playing = False
                event_name = "pause"
            elif kind == "resume":
                playing = True
                event_name = "resume"
            elif kind == "enable":
                enabled = True
                schedule_session(playhead, wall_ms)
                event_name = "enable"
            elif kind == "disable":
                enabled = False
                epoch += 1
                dropped_epoch_tasks += len(pending_tasks)
                pending_tasks.clear()
                wet_ring.clear()
                last_mode = None
                event_name = "disable"
            elif kind == "model":
                model_id = str(command.value)
                if model_id not in self.model_scales:
                    raise ValueError(f"missing scale for model {model_id}")
                enabled = True
                schedule_session(playhead, wall_ms)
                event_name = "model_switch"
            else:
                raise ValueError(f"unknown command: {kind}")
            events.append(
                TraceEvent(
                    wall_ms=wall_ms,
                    event=event_name,
                    segment=segment,
                    position_sample=playhead,
                    epoch=epoch,
                    model_id=model_id,
                )
            )
            command_index += 1

        schedule_session(0, 0.0)
        tick_ms = (
            self.contract.playback_block_samples
            / self.contract.sample_rate
            * 1000.0
        )
        default_limit = max(
            (item.at_wall_ms for item in ordered_commands), default=0.0
        ) + max(1_000.0, self.source.shape[0] / self.contract.sample_rate * 1000.0 * 2)
        limit = max_wall_ms if max_wall_ms is not None else default_limit

        while wall_ms <= limit and (playhead < self.source.shape[0] or command_index < len(ordered_commands)):
            while command_index < len(ordered_commands) and ordered_commands[command_index].at_wall_ms <= wall_ms + 1e-6:
                apply_command(ordered_commands[command_index])
            enqueue_until_horizon(wall_ms)
            publish_ready(wall_ms)

            if playing and playhead < self.source.shape[0]:
                end = min(
                    playhead + self.contract.playback_block_samples,
                    self.source.shape[0],
                )
                dry_ring.write(playhead, self.source[playhead:end])
                dry_samples = dry_ring.read(playhead, end - playhead)
                if dry_samples is None:
                    raise AssertionError("dry ring failed to provide current block")
                wet_samples = wet_ring.read(playhead, end - playhead)
                if enabled and wet_samples is not None:
                    mode = "wet"
                    reason = "window-ready"
                    selected = wet_samples
                else:
                    mode = "dry"
                    reason = "disabled" if not enabled else "wet-not-ready"
                    selected = dry_samples
                if mode != last_mode:
                    events.append(
                        TraceEvent(
                            wall_ms=wall_ms,
                            event="mode_switch",
                            segment=segment,
                            position_sample=playhead,
                            epoch=epoch,
                            mode=mode,
                            reason=reason,
                            model_id=model_id,
                        )
                    )
                    last_mode = mode
                output_blocks.append(
                    OutputBlock(
                        wall_ms=wall_ms,
                        segment=segment,
                        position_sample=playhead,
                        samples=selected,
                        mode=mode,
                        reason=reason,
                        epoch=epoch,
                        model_id=model_id,
                    )
                )
                playhead = end
                dry_ring.discard_before(playhead)
                wet_ring.discard_before(playhead)
            elif not playing:
                next_command = (
                    ordered_commands[command_index].at_wall_ms
                    if command_index < len(ordered_commands)
                    else limit
                )
                if next_command > wall_ms:
                    wall_ms = min(next_command, wall_ms + tick_ms)
                    continue
            wall_ms += tick_ms

        result = SimulationResult(
            contract=self.contract,
            output_blocks=output_blocks,
            events=events,
            stats={
                "trackSamples": int(self.source.shape[0]),
                "trackSeconds": self.source.shape[0] / self.contract.sample_rate,
                "outputBlockCount": len(output_blocks),
                "dryBlockCount": sum(block.mode == "dry" for block in output_blocks),
                "wetBlockCount": sum(block.mode == "wet" for block in output_blocks),
                "modeSwitchCount": sum(item.event == "mode_switch" for item in events),
                "readyWindowCount": ready_windows,
                "lateWindowCount": late_windows,
                "droppedEpochTaskCount": dropped_epoch_tasks,
                "maxPendingTaskCount": max_pending_tasks,
                "dryRing": {
                    "capacitySamples": dry_ring.capacity_samples,
                    "maxSampleCount": dry_ring.max_sample_count,
                    "maxBlockCount": dry_ring.max_block_count,
                    "evictedBlocks": dry_ring.evicted_blocks,
                },
                "wetRing": {
                    "capacitySamples": wet_ring.capacity_samples,
                    "maxSampleCount": wet_ring.max_sample_count,
                    "maxBlockCount": wet_ring.max_block_count,
                    "evictedBlocks": wet_ring.evicted_blocks,
                },
            },
        )
        result.stats["modeRanges"] = result.mode_ranges()
        result.stats["renderedFloat32Sha256"] = result.rendered_sha256()
        result.validate()
        return result


def demo_source(sample_rate: int, seconds: float) -> np.ndarray:
    samples = max(1, int(round(sample_rate * seconds)))
    time = np.arange(samples, dtype=np.float32) / sample_rate
    left = 0.55 * np.sin(2.0 * np.pi * 220.0 * time)
    right = 0.45 * np.sin(2.0 * np.pi * 330.0 * time + 0.2)
    return np.stack((left, right), axis=1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--read-ahead-rate", type=float, default=8.0)
    parser.add_argument("--inference-latency-ms", type=float, default=100.0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--ring-capacity-windows", type=int, default=3)
    parser.add_argument("--seek-at-ms", type=float)
    parser.add_argument("--seek-to-seconds", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = TfcTdfWindowContract()
    source = demo_source(contract.sample_rate, args.duration_seconds)
    commands: list[SimulationCommand] = []
    if args.seek_at_ms is not None or args.seek_to_seconds is not None:
        if args.seek_at_ms is None or args.seek_to_seconds is None:
            raise SystemExit("--seek-at-ms and --seek-to-seconds must be supplied together")
        commands.append(
            SimulationCommand(
                at_wall_ms=args.seek_at_ms,
                kind="seek",
                value=int(round(args.seek_to_seconds * contract.sample_rate)),
            )
        )
    result = TfcTdfStreamSimulator(
        source,
        contract=contract,
        read_ahead_rate=args.read_ahead_rate,
        inference_latency_ms=args.inference_latency_ms,
        worker_count=args.worker_count,
        ring_capacity_windows=args.ring_capacity_windows,
    ).run(commands)
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "simulation": {
            "readAheadRate": args.read_ahead_rate,
            "inferenceLatencyMs": args.inference_latency_ms,
            "workerCount": args.worker_count,
            "ringCapacityWindows": args.ring_capacity_windows,
        },
        "contract": asdict(contract),
        "stats": result.stats,
        "events": [asdict(item) for item in result.events],
        "trace": result.trace_text(),
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
