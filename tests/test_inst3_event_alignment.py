import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_inst3_event_alignment import (  # noqa: E402
    EventRecord,
    block_metric,
    event_from_row,
    event_placements,
    position_summary,
)
from tfc_tdf_default_model import DEFAULT_CONFIG  # noqa: E402


class Inst3EventAlignmentTest(unittest.TestCase):
    def test_event_rank_can_be_assigned_by_selection_order(self) -> None:
        row = {
            "startSamples": 100,
            "endSamples": 4_510,
            "score": 1.0,
            "classification": "vocal-aligned",
        }
        first = event_from_row("fixture", "train", "fixture", row, rank=0)
        second = event_from_row("fixture", "train", "fixture", row, rank=1)

        self.assertEqual(first.rank, 0)
        self.assertEqual(second.rank, 1)

    def test_position_summary_deduplicates_models(self) -> None:
        rows = []
        for model in ("initial", "h50-pass-50", "lambda-0.50-pass-50"):
            rows.append(
                {
                    "role": "internal-test",
                    "slug": "song",
                    "eventRank": 0,
                    "placement": "stride",
                    "relativeEventCenter": 0.05,
                    "model": model,
                }
            )

        summary = position_summary(rows)["internal-test"]

        self.assertEqual(summary["eventCount"], 1)
        self.assertEqual(summary["edgeWithin10PercentCount"], 1)

    def test_center_and_quarter_placements_have_expected_relative_positions(self) -> None:
        event = EventRecord(
            slug="fixture",
            role="train",
            member="fixture",
            rank=1,
            start=100_000,
            end=104_410,
            score=1.0,
            classification="vocal-aligned",
            teacher_rms_dbfs=-20.0,
            true_vocal_rms_dbfs=-20.0,
            teacher_vocal_cosine=0.8,
        )
        placements = event_placements(
            event,
            stride_start=0,
            song_samples=1_000_000,
            useful_samples=DEFAULT_CONFIG.useful_samples,
        )
        by_name = {placement.name: placement for placement in placements}

        self.assertAlmostEqual(by_name["center"].relative_center, 0.5, places=3)
        self.assertAlmostEqual(
            by_name["center-minus-quarter"].relative_center, 0.75, places=3
        )
        self.assertAlmostEqual(
            by_name["center-plus-quarter"].relative_center, 0.25, places=3
        )

    def test_placement_is_clamped_at_song_end(self) -> None:
        event = EventRecord(
            slug="fixture",
            role="train",
            member="fixture",
            rank=1,
            start=990_000,
            end=994_410,
            score=1.0,
            classification="mixed-or-uncertain",
            teacher_rms_dbfs=-20.0,
            true_vocal_rms_dbfs=-30.0,
            teacher_vocal_cosine=0.2,
        )
        placements = event_placements(
            event,
            stride_start=900_000,
            song_samples=1_000_000,
            useful_samples=DEFAULT_CONFIG.useful_samples,
        )

        self.assertTrue(all(placement.start + placement.length <= 1_000_000 for placement in placements))
        self.assertTrue(any(placement.relative_center > 1.0 for placement in placements))
        self.assertTrue(0.0 <= placements[1].relative_center <= 1.0)

    def test_block_metric_is_zero_when_student_matches_teacher_removed(self) -> None:
        length = 10_000
        mixture = np.zeros((length, 2), dtype=np.float32)
        teacher_instrumental = np.zeros((length, 2), dtype=np.float32)
        teacher_instrumental[:, 0] = 0.25
        teacher_instrumental[:, 1] = -0.25
        predicted_residual = mixture - teacher_instrumental
        placement = event_placements(
            EventRecord(
                slug="fixture",
                role="train",
                member="fixture",
                rank=1,
                start=4_000,
                end=8_410,
                score=1.0,
                classification="vocal-aligned",
                teacher_rms_dbfs=-12.0,
                true_vocal_rms_dbfs=-12.0,
                teacher_vocal_cosine=1.0,
            ),
            stride_start=0,
            song_samples=length,
            useful_samples=6_000,
        )[1]

        result = block_metric(
            mixture=mixture,
            teacher_instrumental_segment=teacher_instrumental[placement.start : placement.start + placement.length],
            predicted_residual=predicted_residual[placement.start : placement.start + placement.length],
            placement=placement,
            center=6_205,
            milliseconds=50,
            sample_rate=44_100,
        )

        self.assertLess(result["missRmsDbfs"], -140.0)
        self.assertLess(result["positiveProjectionRmsDbfs"], -140.0)
        self.assertAlmostEqual(result["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
