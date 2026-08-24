import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_modern_song_batch2_rounds import classify_pool, song_comparison_summary  # noqa: E402


def row(mark: str, center: float = 1.0) -> dict:
    return {
        "mark": mark,
        "sourceOrderInt": 7,
        "centerSecondsFloat": center,
        "category": "pop-synth",
        "splitRole": "train",
        "artistName": "Artist",
        "trackName": "Track",
    }


class ModernSongBatch2RoundsTest(unittest.TestCase):
    def test_pool_classification_preserves_i_as_safety(self) -> None:
        self.assertEqual(classify_pool({"first": row("S"), "second": row("S")}), "stable-S")
        self.assertEqual(classify_pool({"first": row("R"), "second": row("S")}), "stable-aggressive")
        self.assertEqual(classify_pool({"first": row("I"), "second": row("K")}), "safety-I")
        self.assertEqual(classify_pool({"first": row("K"), "second": row("R")}), "disagreement-aggressive")
        self.assertEqual(classify_pool({"first": row("K"), "second": row("K")}), "stable-K")

    def test_song_summary_counts_marks_and_pools(self) -> None:
        matches = [
            {"sourceOrder": 7, "first": row("S"), "second": row("S"), "centerDistanceSeconds": 0.0},
            {"sourceOrder": 7, "first": row("R", 2.0), "second": row("K", 2.0), "centerDistanceSeconds": 0.0},
            {"sourceOrder": 7, "first": row("I", 3.0), "second": row("I", 3.0), "centerDistanceSeconds": 0.0},
        ]
        summary = song_comparison_summary(matches)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["matchedCount"], 3)
        self.assertEqual(summary[0]["firstMarks"], {"S": 1, "R": 1, "K": 0, "I": 1})
        self.assertEqual(summary[0]["pools"]["stableS"], 1)
        self.assertEqual(summary[0]["pools"]["disagreementAggressive"], 1)
        self.assertEqual(summary[0]["pools"]["safetyI"], 1)


if __name__ == "__main__":
    unittest.main()
