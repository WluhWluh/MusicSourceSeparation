import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from tfc_tdf_short_window import (  # noqa: E402
    ShortWindowContract,
    assemble_input,
    istft_centered,
    seam_metrics,
    stft_centered,
)


class TfcTdfShortWindowTest(unittest.TestCase):
    def test_default_24_frame_contract_has_expected_stride_options(self) -> None:
        contract = ShortWindowContract(
            num_frames=24,
            left_context_hops=5,
            right_context_hops=5,
        )
        self.assertEqual(contract.input_samples, 23 * 1024)
        self.assertEqual(contract.stride_hops, 13)
        self.assertEqual(contract.stride_samples, 13 * 1024)
        self.assertAlmostEqual(contract.output_efficiency, 13 / 23)

    def test_context_must_leave_an_output_hop(self) -> None:
        with self.assertRaises(ValueError):
            ShortWindowContract(
                num_frames=24,
                left_context_hops=12,
                right_context_hops=11,
            )

    def test_continuous_assembly_keeps_context_across_boundaries(self) -> None:
        contract = ShortWindowContract(
            num_frames=24,
            left_context_hops=2,
            right_context_hops=2,
        )
        source = np.arange(100_000 * 2, dtype=np.float32).reshape(100_000, 2)
        start = contract.stride_samples
        window = assemble_input(
            source,
            start,
            contract.stride_samples,
            contract,
            mode="continuous",
        )
        expected_start = start - contract.left_context_samples
        np.testing.assert_array_equal(
            window,
            source[expected_start : expected_start + contract.input_samples],
        )

    def test_short_dsp_round_trip_is_finite_and_shape_preserving(self) -> None:
        contract = ShortWindowContract(
            num_frames=24,
            left_context_hops=2,
            right_context_hops=2,
        )
        source = np.sin(
            np.arange(contract.input_samples * 2, dtype=np.float32).reshape(
                contract.input_samples,
                2,
            )
            * 0.001
        ).astype(np.float32)
        reconstructed = istft_centered(stft_centered(source, contract), contract)
        self.assertEqual(reconstructed.shape, source.shape)
        self.assertTrue(np.isfinite(reconstructed).all())
        np.testing.assert_allclose(reconstructed, source, atol=2e-6, rtol=2e-6)

    def test_seam_metrics_reports_every_internal_boundary(self) -> None:
        source = np.zeros((10_000, 2), dtype=np.float32)
        source[5_000:] = 1.0
        report = seam_metrics(source, (2_000, 5_000, 8_000))
        self.assertEqual(report["count"], 3)
        self.assertGreaterEqual(report["countRatioAtLeast2"], 1)
        self.assertIn("observations", report)


if __name__ == "__main__":
    unittest.main()
