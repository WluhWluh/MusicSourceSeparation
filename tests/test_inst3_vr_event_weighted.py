import sys
import unittest
from pathlib import Path

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_vr_event_weighted import (  # noqa: E402
    event_frame_mask,
    weighted_l1_loss,
)
from tfc_tdf_default_model import DEFAULT_CONFIG  # noqa: E402


class Inst3VrEventWeightedTest(unittest.TestCase):
    def test_event_block_maps_to_centered_tfc_frame_after_trim(self) -> None:
        candidate_start = 100_000
        event_starts = np.asarray([candidate_start], dtype=np.int64)
        event_ends = np.asarray([candidate_start + 100], dtype=np.int64)

        mask = event_frame_mask(
            candidate_start=candidate_start,
            candidate_length=DEFAULT_CONFIG.useful_samples,
            event_starts=event_starts,
            event_ends=event_ends,
        )

        # The first local event begins at the five-hop trim boundary.  The
        # centered FFT frame at that boundary must be selected.
        self.assertTrue(mask[5])
        self.assertFalse(mask[:4].any())

    def test_event_outside_candidate_does_not_mark_frames(self) -> None:
        candidate_start = 100_000
        event_starts = np.asarray([candidate_start + DEFAULT_CONFIG.useful_samples + 1], dtype=np.int64)
        event_ends = event_starts + 100

        mask = event_frame_mask(
            candidate_start=candidate_start,
            candidate_length=DEFAULT_CONFIG.useful_samples,
            event_starts=event_starts,
            event_ends=event_ends,
        )

        self.assertFalse(mask.any())

    def test_zero_lambda_matches_normalized_l1(self) -> None:
        prediction = torch.tensor([[[[0.0, 2.0], [3.0, -1.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        event_mask = torch.tensor([[True, False]])

        weighted = weighted_l1_loss(prediction, target, event_mask, 0.0)
        ordinary = (prediction - target).abs().mean()

        torch.testing.assert_close(weighted, ordinary)

    def test_positive_lambda_changes_marked_frame_weight(self) -> None:
        prediction = torch.zeros((1, 1, 1, 2))
        target = torch.tensor([[[[1.0, 3.0]]]])
        event_mask = torch.tensor([[True, False]])

        weighted = weighted_l1_loss(prediction, target, event_mask, 0.5)
        expected = torch.tensor((1.5 * 1.0 + 3.0) / (1.5 + 1.0))

        torch.testing.assert_close(weighted, expected)
        self.assertNotEqual(float(weighted), float((prediction - target).abs().mean()))


if __name__ == "__main__":
    unittest.main()
