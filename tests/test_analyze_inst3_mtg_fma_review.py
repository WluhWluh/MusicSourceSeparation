import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_inst3_mtg_fma_review import conservative_song_policy  # noqa: E402


class HumanReviewAnalysisTest(unittest.TestCase):
    def test_high_i_song_is_not_a_training_candidate(self) -> None:
        summary = {"marks": {"R": 1, "K": 3, "I": 12}, "fractions": {"R": 0.0625, "I": 0.75}}
        self.assertEqual(conservative_song_policy(summary), "restoration-or-validation-only")

    def test_clean_r_song_is_candidate(self) -> None:
        summary = {"marks": {"R": 6, "K": 10, "I": 0}, "fractions": {"R": 0.375, "I": 0.0}}
        self.assertEqual(conservative_song_policy(summary), "reviewed-r-candidate")

    def test_mixed_song_can_feed_separate_correction_arms(self) -> None:
        summary = {"marks": {"R": 5, "K": 6, "I": 5}, "fractions": {"R": 0.3125, "I": 0.3125}}
        self.assertEqual(conservative_song_policy(summary), "ambiguous-review-only")


if __name__ == "__main__":
    unittest.main()
