import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_aggressive_target import (  # noqa: E402
    AudioDomainTargetCache,
    aggressive_metrics,
    alpha_name,
)


class Inst3AggressiveTargetTest(unittest.TestCase):
    def test_alpha_name_is_stable_for_output_paths(self) -> None:
        self.assertEqual(alpha_name(0.5), "alpha-0.50")
        self.assertEqual(alpha_name(0.75), "alpha-0.75")
        self.assertEqual(alpha_name(1.0), "alpha-1.00")

    def test_audio_target_is_a_direct_weighted_mix(self) -> None:
        instrumental = np.asarray([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        teacher = np.asarray([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32)
        song = SimpleNamespace(
            slug="fixture",
            instrumental=instrumental,
            teacher_instrumental=teacher,
        )
        cache = AudioDomainTargetCache(0.75)

        target = cache.audio(song)

        np.testing.assert_allclose(target, 0.25 * instrumental + 0.75 * teacher)
        self.assertIs(target, cache.audio(song))

    def test_aggressive_metrics_report_exact_target_and_teacher_match(self) -> None:
        mixture = np.asarray([[0.8, -0.8], [0.4, -0.4]], dtype=np.float32)
        vocals = np.asarray([[0.2, -0.2], [0.1, -0.1]], dtype=np.float32)
        instrumental = mixture - vocals
        teacher = np.asarray([[0.5, -0.5], [0.25, -0.25]], dtype=np.float32)

        metrics = aggressive_metrics(
            mixture,
            vocals,
            instrumental,
            teacher,
            teacher,
            teacher,
        )

        self.assertGreater(metrics["aggressiveTargetSdrDb"], 100.0)
        self.assertGreater(metrics["teacherRemovalResidualSdrDb"], 100.0)
        self.assertLess(metrics["aggressiveTargetErrorRmsDbfs"], -150.0)


if __name__ == "__main__":
    unittest.main()
