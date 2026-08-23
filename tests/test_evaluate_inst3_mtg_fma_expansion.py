import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_inst3_mtg_fma_expansion import severity_score  # noqa: E402


class ExpansionEvaluationTest(unittest.TestCase):
    def test_severity_score_prefers_larger_local_projection(self) -> None:
        low = {str(ms): {"positiveProjectionMaxDbfs": -40.0, "positiveProjectionP95Dbfs": -45.0} for ms in (50, 100, 200)}
        high = {str(ms): {"positiveProjectionMaxDbfs": -20.0, "positiveProjectionP95Dbfs": -30.0} for ms in (50, 100, 200)}
        self.assertGreater(severity_score(high), severity_score(low))


if __name__ == "__main__":
    unittest.main()
