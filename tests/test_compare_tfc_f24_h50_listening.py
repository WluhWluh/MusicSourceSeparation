import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_tfc_f24_h50_listening import (  # noqa: E402
    block_metrics,
    derive_instrumental,
    pair_metrics,
    projection_rms,
    summarize_blocks,
)


class CompareTfcF24H50ListeningTest(unittest.TestCase):
    def test_projection_is_zero_for_orthogonal_candidate(self) -> None:
        basis = np.zeros((100, 2), dtype=np.float32)
        basis[:, 0] = 1.0
        candidate = np.zeros_like(basis)
        candidate[:, 1] = 1.0
        self.assertAlmostEqual(projection_rms(candidate, basis), 0.0, places=6)

    def test_block_metrics_preserve_short_boundaries(self) -> None:
        source = np.ones((4_410, 2), dtype=np.float32)
        candidate = source.copy()
        basis = np.ones_like(source) * 0.5
        rows = block_metrics(source, candidate, basis, 50, 44_100)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["startSamples"], 0)
        self.assertEqual(rows[0]["endSamples"], 2_205)

    def test_summary_counts_projection_reductions(self) -> None:
        rows = [
            {"projectionDeltaDb": -1.0, "residualEnergyDeltaDb": -0.5, "candidateVsH128DeltaRmsDbfs": -20.0},
            {"projectionDeltaDb": 0.5, "residualEnergyDeltaDb": 0.2, "candidateVsH128DeltaRmsDbfs": -18.0},
        ]
        summary = summarize_blocks(rows)
        self.assertEqual(summary["projectionDeltaDb"]["lowerThanZeroCount"], 1)
        self.assertEqual(summary["residualEnergyDeltaDb"]["higherThanZeroCount"], 1)

    def test_derived_instrumental_is_source_minus_vocals(self) -> None:
        source = np.asarray([[0.75, -0.25], [0.5, 0.25]], dtype=np.float32)
        vocals = np.asarray([[0.25, -0.5], [0.1, 0.1]], dtype=np.float32)
        np.testing.assert_allclose(
            derive_instrumental(source, vocals),
            np.asarray([[0.5, 0.25], [0.4, 0.15]], dtype=np.float32),
        )

    def test_pair_metrics_use_accompaniment_and_residual_consistently(self) -> None:
        source = np.ones((100, 2), dtype=np.float32)
        left = np.full_like(source, 0.8)
        right = np.full_like(source, 0.7)
        result = pair_metrics(left, right, source)
        self.assertAlmostEqual(result["leftMinusRightDeltaRmsDbfs"], -20.0, places=3)
        self.assertAlmostEqual(
            result["leftResidualMinusRightResidualDeltaRmsDbfs"], -20.0, places=3
        )


if __name__ == "__main__":
    unittest.main()
