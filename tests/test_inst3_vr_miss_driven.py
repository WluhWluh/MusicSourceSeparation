import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_inst3_vr_miss_driven import (  # noqa: E402
    CandidateMiss,
    build_miss_selection,
    candidate_miss_score,
    canonical_sha256,
    rank_miss_candidates,
    selection_json,
)
from run_inst3_vr_hard_sampling import Candidate, SongSelection  # noqa: E402


class Inst3VrMissDrivenTest(unittest.TestCase):
    def make_selection(self, count: int = 20) -> SongSelection:
        candidates = tuple(
            Candidate(
                index=index,
                start=index * 1000,
                length=1000,
                hard_score=float(count - index),
                rank=index + 1,
            )
            for index in range(count)
        )
        return SongSelection(
            slug="song",
            member="train/song.stem.mp4",
            candidates=candidates,
            hard_pool_indices=tuple(range(count)),
            uniform_indices=(0, 1, 2, 3, 4, 5, 6, 7),
            hard_fraction=0.5,
            hard_indices=(0, 1, 2, 3, 4, 5, 6, 7),
            union_indices=tuple(range(8)),
        )

    def make_scores(self) -> list[CandidateMiss]:
        scores = [CandidateMiss(index, index * 1000, 1000, 0.0, 0.0, {}) for index in range(20)]
        for index, score in ((8, 0.10), (9, 0.90), (10, 0.40), (11, 0.90)):
            scores[index] = CandidateMiss(index, index * 1000, 1000, score, score / 2.0, {})
        return scores

    def test_rank_uses_positive_projection_then_miss_rms_then_start(self) -> None:
        scores = [
            CandidateMiss(1, 200, 1000, 0.5, 0.2, {}),
            CandidateMiss(2, 100, 1000, 0.5, 0.3, {}),
            CandidateMiss(3, 50, 1000, 0.6, 0.1, {}),
        ]
        self.assertEqual([item.candidate_index for item in rank_miss_candidates(scores)], [3, 2, 1])

    def test_selection_has_four_uniform_and_four_non_overlapping_miss_windows(self) -> None:
        selection, selected = build_miss_selection(
            self.make_selection(), self.make_scores(), miss_count=4
        )
        self.assertEqual(len(selection.hard_indices), 8)
        self.assertEqual(len(selected), 4)
        self.assertEqual(tuple(selection.hard_indices[:4]), (0, 1, 2, 3))
        self.assertEqual(
            set(selection.hard_indices[:4]).intersection(selection.hard_indices[4:]),
            set(),
        )
        self.assertEqual(
            [item.candidate_index for item in selected], [9, 11, 10, 8]
        )

    def test_selection_rejects_insufficient_non_uniform_candidates(self) -> None:
        candidates = tuple(
            Candidate(index, index * 1000, 1000, 0.0, index + 1)
            for index in range(4)
        )
        short_selection = SongSelection(
            slug="short",
            member="train/short.stem.mp4",
            candidates=candidates,
            hard_pool_indices=(0, 1, 2, 3),
            uniform_indices=(0, 1, 2, 3),
            hard_fraction=0.5,
            hard_indices=(0, 1, 2, 3),
            union_indices=(0, 1, 2, 3),
        )
        scores = [CandidateMiss(index, index, 1000, float(index), float(index), {}) for index in range(4)]
        selection, selected = build_miss_selection(short_selection, scores, miss_count=4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(selection.hard_indices), 8)
        self.assertEqual(
            [item.candidate_index for item in selected], [3, 2, 1, 0]
        )

    def test_candidate_score_peak_is_maximum_of_all_three_block_sizes(self) -> None:
        sample_rate = 1000
        teacher_removed = np.ones((1000, 2), dtype=np.float32)
        miss = np.zeros((1000, 2), dtype=np.float32)
        miss[0:50] = 0.5
        miss[100:200] = 0.25
        score = candidate_miss_score(
            mixture=teacher_removed,
            teacher_instrumental=np.zeros_like(teacher_removed),
            predicted_residual=teacher_removed - miss,
            sample_rate=sample_rate,
            candidate_index=0,
            start_samples=0,
            length=1000,
        )
        values = [
            score.by_milliseconds[str(milliseconds)]["peakPositiveProjectionRms"]
            for milliseconds in (50, 100, 200)
        ]
        self.assertAlmostEqual(score.peak_positive_projection, max(values), places=7)
        self.assertAlmostEqual(score.peak_miss_rms, max(
            score.by_milliseconds[str(milliseconds)]["peakMissRms"]
            for milliseconds in (50, 100, 200)
        ), places=7)

    def test_selection_hash_is_reproducible(self) -> None:
        selection, selected = build_miss_selection(
            self.make_selection(), self.make_scores(), miss_count=4
        )
        payload = selection_json(selection, self.make_scores(), selected)
        selection_again, selected_again = build_miss_selection(
            self.make_selection(), self.make_scores(), miss_count=4
        )
        payload_again = selection_json(
            selection_again, self.make_scores(), selected_again
        )
        self.assertEqual(canonical_sha256(payload), canonical_sha256(payload_again))


if __name__ == "__main__":
    unittest.main()
