import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_inst3_continuous_baseline import (  # noqa: E402
    REGIONS,
    ShortWindowContract,
    event_rows,
    region_masks,
)


class EvaluateInst3ContinuousBaselineTest(unittest.TestCase):
    def test_region_masks_partition_samples(self) -> None:
        contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
        masks = region_masks(contract.stride_samples * 3 + 1_000, contract)
        self.assertEqual(set(masks), set(REGIONS))
        stacked = np.stack([masks[name] for name in REGIONS])
        np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(stacked.shape[1], dtype=np.int64))
        self.assertGreater(int(masks["song-edge-100ms"].sum()), 0)
        self.assertGreater(int(masks["join-neighborhood-100ms"].sum()), 0)
        self.assertGreater(int(masks["window-center-80pct"].sum()), 0)
        self.assertGreater(int(masks["window-edge-10pct"].sum()), 0)

    def test_event_rows_assign_the_join_region_by_center(self) -> None:
        contract = ShortWindowContract(num_frames=128, left_context_hops=5, right_context_hops=5)
        samples = contract.stride_samples * 2
        mixture = np.ones((samples, 2), dtype=np.float32)
        teacher = np.zeros_like(mixture)
        candidate = np.zeros_like(mixture)
        rows = event_rows(candidate, mixture, teacher, contract, 50)
        self.assertEqual(len(rows["all"]), int(np.ceil(samples / 2_205)))
        join_rows = rows["join-neighborhood-100ms"]
        self.assertGreater(len(join_rows), 0)


if __name__ == "__main__":
    unittest.main()
