import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
import sys

sys.path.insert(0, str(TOOLS))

from tfc_tdf_read_ahead import (  # noqa: E402
    ArrayAudioFrameSource,
    LocalAudioReadAhead,
    WavPcm16FrameSource,
)


class TrackingSource(ArrayAudioFrameSource):
    def __init__(self, samples: np.ndarray) -> None:
        super().__init__(samples)
        self.read_ranges: list[tuple[int, int]] = []

    def read(self, start_frame: int, frame_count: int) -> np.ndarray:
        self.read_ranges.append((start_frame, frame_count))
        return super().read(start_frame, frame_count)


class TfcTdfReadAheadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.useful = 119_808
        frames = self.useful * 2 + 321
        values = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2)
        self.source = values / values.max()

    def make_reader(self, source=None, **kwargs) -> LocalAudioReadAhead:
        return LocalAudioReadAhead(
            source or TrackingSource(self.source),
            input_samples=130_048,
            trim_samples=5_120,
            useful_samples=self.useful,
            **kwargs,
        )

    def test_reads_in_bounded_chunks_and_submits_padded_window(self) -> None:
        source = TrackingSource(self.source)
        reader = self.make_reader(source, capacity_windows=2, read_chunk_samples=4_096)
        reader.reset(start_sample=123, epoch=7)

        reader.pump(playback_sample=123, max_read_samples=self.useful)
        windows = reader.pop_ready_windows(playback_sample=123)

        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window.window_index, 0)
        self.assertEqual(window.epoch, 7)
        np.testing.assert_array_equal(window.valid_samples, self.source[: self.useful])
        np.testing.assert_array_equal(window.input_samples[:5_120], 0.0)
        np.testing.assert_array_equal(
            window.input_samples[5_120 : 5_120 + self.useful],
            self.source[: self.useful],
        )
        np.testing.assert_array_equal(window.input_samples[-5_120:], 0.0)
        self.assertTrue(source.read_ranges)
        self.assertLessEqual(max(count for _, count in source.read_ranges), 4_096)
        self.assertLessEqual(reader.stats()["inputRingMaxSamples"], 2 * self.useful)

    def test_seek_resets_input_ring_and_epoch(self) -> None:
        reader = self.make_reader(capacity_windows=2)
        reader.reset(start_sample=0, epoch=1)
        reader.pump(playback_sample=0, max_read_samples=self.useful)
        self.assertEqual(reader.pop_ready_windows(0)[0].epoch, 1)

        target = self.useful + 777
        reader.reset(start_sample=target, epoch=2)
        self.assertEqual(reader.pop_ready_windows(target), [])
        reader.pump(playback_sample=target, max_read_samples=self.useful)
        windows = reader.pop_ready_windows(target)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].window_index, 1)
        self.assertEqual(windows[0].epoch, 2)
        np.testing.assert_array_equal(
            windows[0].valid_samples,
            self.source[self.useful : 2 * self.useful],
        )

    def test_slow_pump_does_not_submit_incomplete_window(self) -> None:
        reader = self.make_reader(capacity_windows=2, read_chunk_samples=1_024)
        reader.reset(start_sample=0, epoch=1)
        reader.pump(playback_sample=0, max_read_samples=1_023)
        self.assertEqual(reader.pop_ready_windows(0), [])
        self.assertEqual(reader.stats()["submittedWindowCount"], 0)

    def test_cancel_stops_future_reads_and_results(self) -> None:
        source = TrackingSource(self.source)
        reader = self.make_reader(source)
        reader.reset(start_sample=0, epoch=3)
        reader.pump(playback_sample=0, max_read_samples=2_048)
        reads_before_cancel = len(source.read_ranges)

        reader.cancel(epoch=4)

        self.assertEqual(reader.pump(0, 2_048), 0)
        self.assertEqual(reader.pop_ready_windows(0), [])
        self.assertEqual(len(source.read_ranges), reads_before_cancel)
        self.assertFalse(reader.stats()["active"])
        self.assertEqual(reader.stats()["epoch"], 4)

    def test_wav_pcm16_source_reads_stereo_and_mono(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stereo_path = root / "stereo.wav"
            stereo_int16 = np.array(
                [[-32768, 32767], [0, -16384], [16384, 8192]],
                dtype="<i2",
            )
            self.write_wav(stereo_path, stereo_int16)
            with WavPcm16FrameSource(stereo_path) as source:
                self.assertEqual(source.frame_count, 3)
                np.testing.assert_allclose(
                    source.read(0, 3), stereo_int16.astype(np.float32) / 32768.0
                )

            mono_path = root / "mono.wav"
            mono_int16 = np.array([[-32768], [16384]], dtype="<i2")
            self.write_wav(mono_path, mono_int16)
            with WavPcm16FrameSource(mono_path) as source:
                self.assertEqual(source.channel_count, 1)
                decoded = source.read(0, 2)
                self.assertEqual(decoded.shape, (2, 2))
                np.testing.assert_array_equal(decoded[:, 0], decoded[:, 1])

    @staticmethod
    def write_wav(path: Path, samples: np.ndarray) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(samples.shape[1])
            output.setsampwidth(2)
            output.setframerate(44_100)
            output.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())


if __name__ == "__main__":
    unittest.main()
