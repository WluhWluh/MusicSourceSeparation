import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_inst3_private_language_quality import metric_row, short_event_row  # noqa: E402


class PrivateLanguageQualityTest(unittest.TestCase):
    def test_exact_inst3_candidate_has_zero_relative_error(self) -> None:
        source = np.zeros((100, 2), dtype=np.float32)
        reference = np.ones_like(source) * 0.5
        candidate = reference.copy()
        row = metric_row(source, reference, candidate)
        self.assertLess(row["relativeErrorToRemovedDb"], -200.0)
        self.assertLess(row["coherentRetainedRelativeDb"], -200.0)
        self.assertGreater(row["teacherMatchSnrDb"], 200.0)

    def test_residual_in_candidate_is_detected(self) -> None:
        source = np.ones((100, 2), dtype=np.float32)
        reference = np.zeros_like(source)
        candidate = np.ones_like(source) * 0.5
        row = metric_row(source, reference, candidate)
        self.assertAlmostEqual(row["relativeErrorToRemovedDb"], -6.0206, places=3)
        self.assertGreater(row["coherentRetainedRelativeDb"], -7.0)

    def test_short_event_metrics_detect_local_retained_content(self) -> None:
        source = np.ones((4_410, 2), dtype=np.float32)
        reference = np.zeros_like(source)
        candidate = np.zeros_like(source)
        candidate[:2_205] = 0.5
        row = short_event_row(source, reference, candidate, 50)
        self.assertGreater(row["positiveProjectionMaxDbfs"], -7.0)
        self.assertGreater(row["positiveAboveMinus30Dbfs"], 0)


if __name__ == "__main__":
    unittest.main()
