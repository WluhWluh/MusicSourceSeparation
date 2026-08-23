import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_mtg_fma_c1 import (  # noqa: E402
    ARMS,
    EXTRA_RECORDS_PER_PASS,
    RECORDS_PER_PASS,
    build_schedules,
    event_mask,
)


class Inst3MtgFmaC1Test(unittest.TestCase):
    def test_event_mask_has_core_and_soft_guard(self) -> None:
        center = 5_000
        mask = event_mask(10_000, center)
        self.assertEqual(mask.shape, (10_000,))
        self.assertEqual(float(mask[center]), 1.0)
        self.assertGreater(float(mask[center - 2_205 - 500]), 0.0)
        self.assertEqual(float(mask[0]), 0.0)

    def test_equal_budget_schedules(self) -> None:
        musdb = [f"song-{index:02d}" for index in range(80)]
        external = [f"external::{index}" for index in range(4)]
        schedules = build_schedules(musdb, 8, external, 2, 891)
        self.assertEqual(set(schedules), set(ARMS))
        self.assertTrue(all(len(value) == RECORDS_PER_PASS * 2 for value in schedules.values()))
        self.assertEqual(sum(key.startswith("external::") for key, _ in schedules[ARMS[1]]), EXTRA_RECORDS_PER_PASS * 2)
        self.assertEqual(sum(key.startswith("external::") for key, _ in schedules[ARMS[0]]), 0)


if __name__ == "__main__":
    unittest.main()
