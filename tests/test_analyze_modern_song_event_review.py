import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_modern_song_event_review import MARKS, clusters  # noqa: E402


class ModernSongReviewAnalysisTest(unittest.TestCase):
    def test_accepts_s_as_distinct_mark(self) -> None:
        self.assertEqual(set(MARKS), {"S", "R", "K", "I"})

    def test_clusters_nearby_s_events(self) -> None:
        rows = [
            {"centerSeconds": 10.0},
            {"centerSeconds": 11.5},
            {"centerSeconds": 15.0},
        ]
        result = clusters(rows)
        self.assertEqual([len(item) for item in result], [2, 1])


if __name__ == "__main__":
    unittest.main()
