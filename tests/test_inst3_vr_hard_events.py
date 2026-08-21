from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_inst3_vr_hard_events import (  # noqa: E402
    aggregate_student_windows,
    block_features,
    classify_event,
    parse_int_list,
)


class Inst3VrHardEventsTest(unittest.TestCase):
    def test_parse_int_list_requires_sorted_unique_positive_values(self) -> None:
        self.assertEqual(parse_int_list("50,100,200"), (50, 100, 200))
        with self.assertRaises(ValueError):
            parse_int_list("100,50")
        with self.assertRaises(ValueError):
            parse_int_list("50,50")

    def test_block_features_detect_positive_teacher_residual_miss(self) -> None:
        mixture = np.ones((10, 2), dtype=np.float32) * 0.5
        teacher = np.zeros_like(mixture)
        initial = np.ones_like(mixture) * 0.25
        vocals = np.ones_like(mixture) * 0.5
        features = block_features(
            mixture=mixture,
            teacher_instrumental=teacher,
            student_instrumental=initial,
            true_vocals=vocals,
            sample_rate=100,
            milliseconds=100,
            active_floor_dbfs=-60.0,
        )
        self.assertEqual(int(features["blockCount"]), 1)
        self.assertTrue(bool(np.asarray(features["active"])[0]))
        self.assertGreater(float(np.asarray(features["projectionCoefficient"])[0]), 0.0)
        self.assertGreater(float(np.asarray(features["score"])[0]), 0.0)

    def test_block_features_zero_when_student_matches_teacher(self) -> None:
        mixture = np.ones((12, 2), dtype=np.float32) * 0.4
        teacher = np.ones_like(mixture) * 0.1
        features = block_features(
            mixture=mixture,
            teacher_instrumental=teacher,
            student_instrumental=teacher,
            true_vocals=mixture - teacher,
            sample_rate=100,
            milliseconds=100,
        )
        np.testing.assert_allclose(features["missRms"], 0.0)
        np.testing.assert_allclose(features["score"], 0.0)

    def test_classification_is_explicitly_heuristic(self) -> None:
        self.assertEqual(
            classify_event(
                teacher_rms_dbfs=-20.0,
                true_vocal_rms_dbfs=-10.0,
                teacher_vocal_cosine=0.8,
                vocal_p20_dbfs=-40.0,
                vocal_p80_dbfs=-15.0,
            ),
            "vocal-aligned",
        )
        self.assertEqual(
            classify_event(
                teacher_rms_dbfs=-30.0,
                true_vocal_rms_dbfs=-50.0,
                teacher_vocal_cosine=0.0,
                vocal_p20_dbfs=-40.0,
                vocal_p80_dbfs=-15.0,
            ),
            "vocal-like-or-non-vocal",
        )

    def test_student_window_aggregation_ranks_active_windows(self) -> None:
        features = {
            "startSamples": np.asarray([0, 10, 100, 110], dtype=np.int64),
            "score": np.asarray([0.1, 0.7, 0.2, 0.0], dtype=np.float32),
            "active": np.asarray([True, True, True, False]),
            "positiveProjectionRms": np.asarray([0.1, 0.4, 0.2, 0.0], dtype=np.float32),
            "teacherRms": np.asarray([0.2, 0.4, 0.3, 0.0], dtype=np.float32),
        }
        windows = aggregate_student_windows(
            features,
            useful_samples=100,
            song_samples=200,
            sample_rate=100,
        )
        self.assertEqual(windows[0]["windowIndex"], 0)
        self.assertAlmostEqual(windows[0]["hardEventScore"], 0.7)
        self.assertEqual(windows[0]["activeEventCount"], 2)
        self.assertEqual(windows[1]["activeEventCount"], 1)


if __name__ == "__main__":
    unittest.main()
