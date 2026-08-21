from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import render_inst3_wd_listening as listening  # noqa: E402


class Inst3WdListeningTest(unittest.TestCase):
    def test_pass_parser_requires_sorted_positive_values(self) -> None:
        self.assertEqual(listening.parse_csv_ints("100,125,200"), (100, 125, 200))
        with self.assertRaises(ValueError):
            listening.validate_args(
                type("Args", (), {"passes": "125,100", "seed": 891, "threads": 8})()
            )

    def test_direct_and_residual_semantics_are_distinct(self) -> None:
        mixture = torch.tensor([[[[3.0]]]])
        output = torch.tensor([[[[1.25]]]])
        torch.testing.assert_close(
            listening.select_instrumental_spectrum(
                output, mixture, "direct-instrumental"
            ),
            output,
        )
        torch.testing.assert_close(
            listening.select_instrumental_spectrum(output, mixture, "residual-vocals"),
            torch.tensor([[[[1.75]]]]),
        )

    def test_checkpoint_path_uses_eighty_updates_per_pass(self) -> None:
        path = listening.checkpoint_path(Path("root"), 891, 125)
        self.assertEqual(
            path.as_posix(), "root/runs/seed-891/W-D/pass-10000.pt"
        )

    def test_parent_pass_uses_the_matrix_root(self) -> None:
        path = listening.source_checkpoint_path(
            Path("continuation"), Path("matrix"), 891, 100
        )
        self.assertEqual(
            path.as_posix(), "matrix/runs/seed-891/W-D/pass-8000.pt"
        )


if __name__ == "__main__":
    unittest.main()
