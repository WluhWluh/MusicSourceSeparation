import sys
import unittest
from pathlib import Path

import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_vr_local_anchor import (  # noqa: E402
    LOCAL_ANCHOR,
    local_anchor_loss,
    masked_mean_l1,
)


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


if __name__ == "__main__":
    unittest.main()
