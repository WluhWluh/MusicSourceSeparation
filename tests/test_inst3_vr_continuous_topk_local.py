import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_vr_continuous_topk_local import (  # noqa: E402
    EventRecord,
    ShortWindowContract,
    charbonnier_per_record,
    select_top_records,
)


class ContinuousTopkLocalTest(unittest.TestCase):
    def test_selects_unique_central_records_before_repetition(self) -> None:
        records = [
            EventRecord(i, i * 10_000, 100, 5_000, 5_500, 50, 1.0 - i * 0.01, 0.5, 0.6, 0.0)
            for i in range(10)
        ]
        selected, details = select_top_records(records, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(details["repeatedRecordCount"], 0)
        self.assertEqual(len({item.start for item in selected}), 4)

    def test_repeats_when_the_song_has_too_few_unique_records(self) -> None:
        record = EventRecord(0, 0, 100, 5_000, 5_500, 50, 1.0, 0.5, 0.6, 0.0)
        selected, details = select_top_records([record], 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(details["repeatedRecordCount"], 2)

    def test_charbonnier_mask_keeps_batch_dimension(self) -> None:
        prediction = torch.zeros((2, 6, 2), dtype=torch.float32, requires_grad=True)
        target = torch.ones_like(prediction)
        mask = torch.zeros((2, 6), dtype=torch.float32)
        mask[0, :2] = 1.0
        mask[1, 2:4] = 1.0
        value = charbonnier_per_record(prediction, target, mask)
        self.assertEqual(tuple(value.shape), (2,))
        value.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
