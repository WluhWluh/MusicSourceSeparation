from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_inst3_vr_hard_sampling import (  # noqa: E402
    Candidate,
    build_selections,
    build_training_schedule,
    build_song_selection,
    student_segment_spec,
    summarize_schedule,
)
import run_inst3_distill_pilot as pilot  # noqa: E402


class Inst3VrHardSamplingTest(unittest.TestCase):
    def make_candidates(self, count: int = 20) -> list[Candidate]:
        return [
            Candidate(
                index=index,
                start=index * 1000,
                length=1000,
                hard_score=float(count - index),
                rank=index + 1,
            )
            for index in range(count)
        ]

    def test_h25_keeps_eight_draws_and_uses_song_local_top_quarter(self) -> None:
        selection = build_song_selection(
            slug="song",
            member="train/song.stem.mp4",
            candidates=self.make_candidates(),
            train_windows_per_song=8,
            hard_fraction=0.25,
            seed=891,
            song_index=0,
        )
        self.assertEqual(len(selection.uniform_indices), 8)
        self.assertEqual(len(selection.h25_indices), 8)
        self.assertEqual(len(selection.hard_pool_indices), 5)
        self.assertEqual(
            set(selection.h25_indices[6:]),
            set(selection.h25_indices[6:]).intersection(selection.hard_pool_indices),
        )
        self.assertEqual(
            set(selection.h25_indices[:6]),
            set(selection.uniform_indices[:6]),
        )

    def test_short_song_preserves_draw_budget_with_replacement(self) -> None:
        selection = build_song_selection(
            slug="short",
            member="train/short.stem.mp4",
            candidates=self.make_candidates(5),
            train_windows_per_song=8,
            hard_fraction=0.25,
            seed=891,
            song_index=0,
        )
        self.assertEqual(len(selection.uniform_indices), 8)
        self.assertEqual(len(selection.h25_indices), 8)
        self.assertLessEqual(len(selection.union_indices), 5)

    def test_h50_preserves_eight_draws_and_uses_four_hard_draws(self) -> None:
        selection = build_song_selection(
            slug="song",
            member="train/song.stem.mp4",
            candidates=self.make_candidates(),
            train_windows_per_song=8,
            hard_fraction=0.50,
            seed=891,
            song_index=0,
        )
        self.assertEqual(len(selection.uniform_indices), 8)
        self.assertEqual(len(selection.hard_indices), 8)
        self.assertEqual(selection.hard_fraction, 0.50)
        self.assertEqual(
            set(selection.hard_indices[:4]),
            set(selection.uniform_indices[:4]),
        )
        self.assertTrue(
            set(selection.hard_indices[4:]).issubset(
                set(selection.hard_pool_indices)
            )
        )

    def test_global_schedule_has_same_pass_shape_for_both_cells(self) -> None:
        candidates = self.make_candidates()
        selections = {
            f"song-{index}": build_song_selection(
                slug=f"song-{index}",
                member=f"train/song-{index}.stem.mp4",
                candidates=candidates,
                train_windows_per_song=8,
                hard_fraction=0.25,
                seed=891,
                song_index=index,
            )
            for index in range(3)
        }
        uniform = build_training_schedule(selections, "V-R-U", 2, 891)
        hard = build_training_schedule(selections, "V-R-H25", 2, 891)
        self.assertEqual(len(uniform), 48)
        self.assertEqual(len(hard), len(uniform))
        self.assertEqual(
            summarize_schedule(selections, {"U": uniform, "H": hard}, 2, 4)["U"]["recordsPerPass"],
            24,
        )

    def test_h50_schedule_is_supported(self) -> None:
        candidates = self.make_candidates()
        selections = {
            f"song-{index}": build_song_selection(
                slug=f"song-{index}",
                member=f"train/song-{index}.stem.mp4",
                candidates=candidates,
                train_windows_per_song=8,
                hard_fraction=0.50,
                seed=891,
                song_index=index,
            )
            for index in range(3)
        }
        hard = build_training_schedule(
            selections, "V-R-H50", 2, 891, hard_variant="V-R-H50"
        )
        self.assertEqual(len(hard), 48)

    def test_segment_target_uses_local_origin(self) -> None:
        segment = np.zeros((pilot.DEFAULT_CONFIG.useful_samples, 2), dtype=np.float32)
        segment[12_000:18_000, 0] = 0.25
        segment[18_000:24_000, 1] = -0.125

        actual = student_segment_spec(segment)
        expected = pilot.student_window_spec(segment, 0, segment.shape[0])
        misplaced = pilot.student_window_spec(
            segment, pilot.DEFAULT_CONFIG.useful_samples, segment.shape[0]
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertGreater(float(np.linalg.norm(actual)), 0.0)
        self.assertLess(float(np.linalg.norm(misplaced)), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
