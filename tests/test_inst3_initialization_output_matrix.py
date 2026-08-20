from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_inst3_initialization_output_matrix as matrix  # noqa: E402


class Inst3InitializationOutputMatrixTest(unittest.TestCase):
    def test_default_contract_is_four_cells_three_seeds_and_three_milestones(self) -> None:
        args = matrix.parse_args([])

        variants, seeds, milestones = matrix.validate_args(args)

        self.assertEqual(tuple(item.key for item in variants), matrix.DEFAULT_VARIANTS)
        self.assertEqual(seeds, matrix.DEFAULT_SEEDS)
        self.assertEqual(milestones, (0, 25, 50, 100))
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.learning_rate, 1.0e-4)

    def test_each_pass_is_a_complete_reproducible_permutation(self) -> None:
        first = matrix.build_sample_schedule(8, 3, 891)
        second = matrix.build_sample_schedule(8, 3, 891)

        matrix.validate_schedule(first, 8, 3)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (24,))
        self.assertEqual(matrix.schedule_sha256(first), matrix.schedule_sha256(second))

    def test_output_semantics_are_distinct(self) -> None:
        mixture = torch.tensor([1.0, 2.0])
        raw = torch.tensor([0.25, 0.75])

        residual = matrix.predicted_instrumental_spectrum(
            raw, mixture, matrix.VARIANT_SPECS["R-R"]
        )
        direct = matrix.predicted_instrumental_spectrum(
            raw, mixture, matrix.VARIANT_SPECS["R-D"]
        )

        torch.testing.assert_close(residual, torch.tensor([0.75, 1.25]))
        torch.testing.assert_close(direct, raw)

    def test_random_cells_share_the_same_initial_neural_state(self) -> None:
        device = torch.device("cpu")
        residual, residual_metadata = matrix.make_matrix_model(
            matrix.VARIANT_SPECS["R-R"], Path("unused"), 891, device
        )
        direct, direct_metadata = matrix.make_matrix_model(
            matrix.VARIANT_SPECS["R-D"], Path("unused"), 891, device
        )

        self.assertEqual(
            residual_metadata["initialStateSha256"],
            direct_metadata["initialStateSha256"],
        )
        for key, value in residual.state_dict().items():
            torch.testing.assert_close(value, direct.state_dict()[key])

    def test_reset_direct_head_does_not_change_the_pretrained_body(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            core = matrix.TfcTdfNeuralCore()
        before = {key: value.clone() for key, value in core.state_dict().items()}

        matrix.reset_direct_output_head(core, 2)

        changed = [
            key
            for key, value in core.state_dict().items()
            if not torch.equal(value, before[key])
        ]
        self.assertTrue(changed)
        self.assertTrue(all(key.startswith("last_conv.0.") for key in changed))

    def test_three_seed_gate_requires_random_direct_to_track_warm_residual(self) -> None:
        def variant(mean: float) -> dict[str, object]:
            metrics = {
                key: {"mean": mean, "sampleStdDev": 0.0, "minimum": mean, "maximum": mean}
                for key in matrix.MATRIX_METRIC_KEYS
            }
            return {"100": {"seedCount": 3, "aggregateAcrossSeeds": metrics}}

        passed = matrix.make_gate({"R-D": variant(9.5), "W-R": variant(10.0)}, 100)
        failed = matrix.make_gate({"R-D": variant(8.5), "W-R": variant(10.0)}, 100)

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
