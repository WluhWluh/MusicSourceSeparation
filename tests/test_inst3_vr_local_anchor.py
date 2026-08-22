import sys
import unittest
from pathlib import Path

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_vr_local_anchor import (  # noqa: E402
    LOCAL_ANCHOR,
    local_anchor_loss,
    masked_mean_l1,
    soft_event_frame_mask,
)
from run_inst3_distill_pilot import DEFAULT_CONFIG  # noqa: E402


class Inst3VrLocalAnchorTest(unittest.TestCase):
    def test_masked_mean_l1_uses_only_marked_frames(self) -> None:
        prediction = torch.tensor([[[[0.0, 10.0], [0.0, 10.0]]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[True, False]])

        result = masked_mean_l1(prediction, target, mask)

        self.assertEqual(float(result), 0.0)

    def test_local_anchor_loss_separates_event_and_non_event_targets(self) -> None:
        prediction = torch.zeros((1, 1, 1, 2))
        teacher = torch.tensor([[[[2.0, 10.0]]]])
        anchor = torch.tensor([[[[7.0, 3.0]]]])
        mask = torch.tensor([[True, False]])

        total, event_loss, anchor_loss = local_anchor_loss(
            prediction, teacher, anchor, mask, beta=1.0
        )

        self.assertEqual(float(event_loss), 2.0)
        self.assertEqual(float(anchor_loss), 3.0)
        self.assertEqual(float(total), 5.0)

    def test_local_anchor_variant_name_is_stable(self) -> None:
        self.assertEqual(LOCAL_ANCHOR, "H50-local-anchor")

    def test_zero_guard_matches_the_original_boolean_event_mask(self) -> None:
        kwargs = {
            "candidate_start": 0,
            "candidate_length": DEFAULT_CONFIG.useful_samples,
            "event_starts": np.asarray([50_000], dtype=np.int64),
            "event_ends": np.asarray([54_410], dtype=np.int64),
        }

        zero = soft_event_frame_mask(**kwargs, guard_ms=0.0)
        guarded = soft_event_frame_mask(**kwargs, guard_ms=25.0)

        self.assertEqual(zero.dtype, np.bool_)
        self.assertTrue(np.array_equal(zero, guarded >= 1.0))
        self.assertTrue(np.all(guarded >= 0.0))
        self.assertTrue(np.all(guarded <= 1.0))

    def test_soft_guard_adds_fractional_context_without_weakening_core(self) -> None:
        guarded = soft_event_frame_mask(
            candidate_start=0,
            candidate_length=DEFAULT_CONFIG.useful_samples,
            event_starts=np.asarray([50_000], dtype=np.int64),
            event_ends=np.asarray([54_410], dtype=np.int64),
            guard_ms=25.0,
        )

        self.assertGreater(np.count_nonzero((guarded > 0.0) & (guarded < 1.0)), 0)
        self.assertGreater(float(guarded.sum()), float(np.count_nonzero(guarded >= 1.0)))

    def test_local_anchor_loss_accepts_fractional_guard_weights(self) -> None:
        prediction = torch.zeros((1, 1, 1, 3))
        teacher = torch.tensor([[[[2.0, 4.0, 100.0]]]])
        anchor = torch.tensor([[[[7.0, 3.0, 1.0]]]])
        mask = torch.tensor([[1.0, 0.5, 0.0]])

        total, event_loss, anchor_loss = local_anchor_loss(
            prediction, teacher, anchor, mask, beta=1.0
        )

        self.assertAlmostEqual(float(event_loss), 8.0 / 3.0, places=6)
        self.assertAlmostEqual(float(anchor_loss), 5.0 / 3.0, places=6)
        self.assertAlmostEqual(float(total), 13.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
