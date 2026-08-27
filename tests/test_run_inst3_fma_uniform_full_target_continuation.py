import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_inst3_fma_uniform_full_target_continuation as runner  # noqa: E402
from tfc_tdf_short_window import stft_centered  # noqa: E402


class UniformScheduleTest(unittest.TestCase):
    def songs(self):
        useful = runner.CONTRACT.useful_samples
        return [
            {"slug": "short", "frameCount": useful * 3 + 17, "fullSlotsPerPass": 3},
            {"slug": "long", "frameCount": useful * 5 + 31, "fullSlotsPerPass": 5},
        ]

    def test_random_starts_cover_every_slot_and_are_deterministic(self) -> None:
        frames = self.songs()[1]["frameCount"]
        first = runner.random_uniform_starts(frames, 0, 891, 1)
        second = runner.random_uniform_starts(frames, 0, 891, 1)
        later = runner.random_uniform_starts(frames, 1, 891, 1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, later)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(0 <= start <= frames - runner.CONTRACT.useful_samples for start in first))

    def test_schedule_uses_all_slots_and_only_batch_padding_repeats(self) -> None:
        schedule, summary = runner.build_training_schedule(self.songs(), 2, 891, 4)
        self.assertEqual(summary["recordsBeforeBatchPadding"], 8)
        self.assertEqual(summary["recordsPerPass"], 8)
        self.assertEqual(len(schedule), 16)
        self.assertEqual(summary["passes"][0]["uniqueRecordCount"], 8)

    def test_schedule_adds_minimum_batch_padding(self) -> None:
        songs = self.songs()[:1]
        schedule, summary = runner.build_training_schedule(songs, 1, 891, 4)
        self.assertEqual(summary["recordsBeforeBatchPadding"], 3)
        self.assertEqual(summary["batchPaddingRecordsPerPass"], 1)
        self.assertEqual(len(schedule), 4)
        self.assertEqual(len(set(schedule)), 3)


class BatchStftTest(unittest.TestCase):
    def test_batch_stft_matches_single_window_contract(self) -> None:
        rng = np.random.default_rng(891)
        waves = rng.normal(0, 0.1, size=(2, runner.CONTRACT.input_samples, 2)).astype(np.float32)
        actual = runner.batch_stft(waves)
        expected = np.concatenate(
            [stft_centered(wave, runner.CONTRACT) for wave in waves], axis=0
        )
        np.testing.assert_array_equal(actual, expected)


class ResidualMetricTest(unittest.TestCase):
    def test_exact_target_has_floor_projection_and_miss(self) -> None:
        target = np.ones((2, 100, 2), dtype=np.float32) * 0.25
        metrics = runner.residual_metrics(target, np.zeros_like(target))
        np.testing.assert_array_equal(metrics["missRmsDbfs"], np.full(2, -240.0))
        np.testing.assert_array_equal(metrics["positiveProjectionDbfs"], np.full(2, -240.0))


if __name__ == "__main__":
    unittest.main()
