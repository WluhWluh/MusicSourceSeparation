#!/usr/bin/env python3
"""Bounded local-audio read-ahead for the host TFC-TDF experiment.

The reader is deliberately independent from playback decoding. It reads audio
by absolute frame position, keeps only a bounded input ring, and emits padded
model windows when their valid source range is complete. It does not write
full-song PCM or retain samples after a submitted window is no longer needed.
"""

from __future__ import annotations

import math
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class AudioFrameSource(Protocol):
    sample_rate: int
    channel_count: int
    frame_count: int

    def read(self, start_frame: int, frame_count: int) -> np.ndarray:
        """Return float32 interleaved frames with shape [frames, channels]."""


class ArrayAudioFrameSource:
    """Small deterministic source adapter used by host tests and simulation."""

    def __init__(self, samples: np.ndarray, sample_rate: int = 44_100) -> None:
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples must have shape [frames, 2]")
        if samples.shape[0] == 0:
            raise ValueError("samples must not be empty")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.samples = np.ascontiguousarray(samples, dtype=np.float32)
        self.sample_rate = sample_rate
        self.channel_count = 2
        self.frame_count = int(self.samples.shape[0])

    def read(self, start_frame: int, frame_count: int) -> np.ndarray:
        _validate_read_range(start_frame, frame_count, self.frame_count)
        return np.ascontiguousarray(
            self.samples[start_frame : start_frame + frame_count],
            dtype=np.float32,
        )


class WavPcm16FrameSource:
    """Independent PCM16 WAV reader used to exercise the local-file path.

    Android compressed-file decoding is intentionally a separate adapter in a
    later phase. A MediaExtractor/MediaCodec instance can implement the same
    absolute-frame interface without changing the read-ahead scheduler.
    """

    def __init__(self, path: Path, expected_sample_rate: int = 44_100) -> None:
        self.path = Path(path)
        self._wave = wave.open(str(self.path), "rb")
        self.sample_rate = self._wave.getframerate()
        self.channel_count = self._wave.getnchannels()
        self.frame_count = self._wave.getnframes()
        if self._wave.getcomptype() != "NONE":
            self.close()
            raise ValueError("Only uncompressed WAV is supported")
        if self._wave.getsampwidth() != 2:
            self.close()
            raise ValueError("Only PCM16 WAV is supported")
        if self.sample_rate != expected_sample_rate:
            self.close()
            raise ValueError(
                f"Expected {expected_sample_rate} Hz WAV, got {self.sample_rate} Hz"
            )
        if self.channel_count not in (1, 2):
            self.close()
            raise ValueError("WAV must be mono or stereo")

    def read(self, start_frame: int, frame_count: int) -> np.ndarray:
        _validate_read_range(start_frame, frame_count, self.frame_count)
        self._wave.setpos(start_frame)
        raw = self._wave.readframes(frame_count)
        expected_bytes = frame_count * self.channel_count * 2
        if len(raw) != expected_bytes:
            raise IOError(
                f"Short WAV read at frame {start_frame}: "
                f"expected {expected_bytes} bytes, got {len(raw)}"
            )
        values = np.frombuffer(raw, dtype="<i2").reshape(-1, self.channel_count)
        stereo = values if self.channel_count == 2 else np.repeat(values, 2, axis=1)
        return np.ascontiguousarray(stereo.astype(np.float32) / 32768.0)

    def close(self) -> None:
        self._wave.close()

    def __enter__(self) -> "WavPcm16FrameSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_read_range(start_frame: int, frame_count: int, total_frames: int) -> None:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if start_frame + frame_count > total_frames:
        raise ValueError("read range exceeds source")


class _BoundedFrameRing:
    def __init__(self, capacity_frames: int) -> None:
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        self.capacity_frames = capacity_frames
        self._blocks: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_frame_count = 0
        self.max_block_count = 0
        self.evicted_blocks = 0

    @property
    def frame_count(self) -> int:
        return sum(int(value.shape[0]) for value in self._blocks.values())

    @property
    def first_frame(self) -> int | None:
        return next(iter(self._blocks), None)

    def clear(self) -> None:
        self._blocks.clear()

    def write(self, start_frame: int, samples: np.ndarray) -> None:
        if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] == 0:
            raise ValueError("ring samples must have shape [frames, 2]")
        value = np.ascontiguousarray(samples, dtype=np.float32)
        end_frame = start_frame + value.shape[0]
        for block_start, existing in self._blocks.items():
            existing_end = block_start + existing.shape[0]
            if start_frame < existing_end and block_start < end_frame:
                raise ValueError("overlapping input ring blocks")
        self._blocks[start_frame] = value
        self._blocks = OrderedDict(sorted(self._blocks.items()))
        while self.frame_count > self.capacity_frames and self._blocks:
            self._blocks.popitem(last=False)
            self.evicted_blocks += 1
        self.max_frame_count = max(self.max_frame_count, self.frame_count)
        self.max_block_count = max(self.max_block_count, len(self._blocks))

    def read(self, start_frame: int, frame_count: int) -> np.ndarray | None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        position = start_frame
        remaining = frame_count
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

    def discard_before(self, position_frame: int) -> None:
        for start in list(self._blocks):
            value = self._blocks[start]
            end = start + value.shape[0]
            if end <= position_frame:
                del self._blocks[start]
            elif start < position_frame:
                self._blocks[position_frame] = np.ascontiguousarray(
                    value[position_frame - start :]
                )
                del self._blocks[start]
        self._blocks = OrderedDict(sorted(self._blocks.items()))


@dataclass(frozen=True)
class ReadAheadWindow:
    window_index: int
    epoch: int
    start_sample: int
    actual_samples: int
    input_samples: np.ndarray
    valid_samples: np.ndarray

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.actual_samples


class LocalAudioReadAhead:
    """Bounded absolute-position reader for one analysis session."""

    def __init__(
        self,
        source: AudioFrameSource,
        *,
        input_samples: int = 130_048,
        trim_samples: int = 5_120,
        useful_samples: int = 119_808,
        capacity_windows: int = 3,
        read_chunk_samples: int = 16_384,
    ) -> None:
        if source.sample_rate != 44_100:
            raise ValueError(f"Expected 44,100 Hz source, got {source.sample_rate}")
        if source.channel_count != 2:
            raise ValueError("Read-ahead requires stereo source")
        if input_samples <= 0 or trim_samples < 0 or useful_samples <= 0:
            raise ValueError("Invalid window contract")
        if trim_samples * 2 + useful_samples != input_samples:
            raise ValueError("input_samples must equal trim*2 + useful_samples")
        if capacity_windows <= 0 or read_chunk_samples <= 0:
            raise ValueError("capacity_windows and read_chunk_samples must be positive")

        self.source = source
        self.input_samples = input_samples
        self.trim_samples = trim_samples
        self.useful_samples = useful_samples
        self.capacity_samples = capacity_windows * useful_samples
        self.read_chunk_samples = read_chunk_samples
        self._ring = _BoundedFrameRing(self.capacity_samples)
        self._epoch = -1
        self._active = False
        self._analysis_start = 0
        self._read_cursor = 0
        self._next_window_index = 0
        self._read_calls = 0
        self._frames_read = 0
        self._submitted_windows = 0
        self._late_windows = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset(self, start_sample: int, epoch: int) -> None:
        if start_sample < 0 or start_sample >= self.source.frame_count:
            raise ValueError("start_sample is outside the source")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._ring.clear()
        self._epoch = epoch
        self._active = True
        self._analysis_start = (start_sample // self.useful_samples) * self.useful_samples
        self._read_cursor = self._analysis_start
        self._next_window_index = self._analysis_start // self.useful_samples

    def cancel(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._ring.clear()
        self._epoch = epoch
        self._active = False
        self._read_cursor = 0
        self._next_window_index = 0

    def pump(self, playback_sample: int, max_read_samples: int) -> int:
        if playback_sample < 0 or playback_sample >= self.source.frame_count:
            raise ValueError("playback_sample is outside the source")
        if max_read_samples <= 0:
            raise ValueError("max_read_samples must be positive")
        if not self._active:
            return 0

        ring_floor = self._ring.first_frame
        if ring_floor is None:
            ring_floor = self._analysis_start
        horizon_end = min(
            self.source.frame_count,
            max(ring_floor + self.capacity_samples, playback_sample),
        )
        remaining = max_read_samples
        frames_read = 0
        while remaining > 0 and self._read_cursor < horizon_end:
            count = min(
                remaining,
                self.read_chunk_samples,
                horizon_end - self._read_cursor,
            )
            chunk = self.source.read(self._read_cursor, count)
            if chunk.shape != (count, 2):
                raise ValueError(f"Source returned unexpected chunk shape: {chunk.shape}")
            self._ring.write(self._read_cursor, chunk)
            self._read_cursor += count
            remaining -= count
            frames_read += count
            self._read_calls += 1
            self._frames_read += count
        return frames_read

    def pop_ready_windows(self, playback_sample: int) -> list[ReadAheadWindow]:
        if playback_sample < 0 or playback_sample >= self.source.frame_count:
            raise ValueError("playback_sample is outside the source")
        if not self._active:
            return []

        result: list[ReadAheadWindow] = []
        window_count = math.ceil(self.source.frame_count / self.useful_samples)
        while self._next_window_index < window_count:
            start = self._next_window_index * self.useful_samples
            actual = min(self.useful_samples, self.source.frame_count - start)
            end = start + actual
            if end <= playback_sample:
                self._late_windows += 1
                self._next_window_index += 1
                self._ring.discard_before(end)
                continue
            valid = self._ring.read(start, actual)
            if valid is None:
                break
            padded = np.zeros((self.input_samples, 2), dtype=np.float32)
            padded[self.trim_samples : self.trim_samples + actual] = valid
            result.append(
                ReadAheadWindow(
                    window_index=self._next_window_index,
                    epoch=self._epoch,
                    start_sample=start,
                    actual_samples=actual,
                    input_samples=padded,
                    valid_samples=valid,
                )
            )
            self._submitted_windows += 1
            self._next_window_index += 1
            self._ring.discard_before(end)
        return result

    def stats(self) -> dict[str, int | bool]:
        return {
            "epoch": self._epoch,
            "active": self._active,
            "readCalls": self._read_calls,
            "framesRead": self._frames_read,
            "submittedWindowCount": self._submitted_windows,
            "lateWindowCount": self._late_windows,
            "inputRingCapacitySamples": self._ring.capacity_frames,
            "inputRingMaxSamples": self._ring.max_frame_count,
            "inputRingMaxBlocks": self._ring.max_block_count,
            "inputRingEvictedBlocks": self._ring.evicted_blocks,
        }
