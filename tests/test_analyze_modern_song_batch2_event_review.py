import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_modern_song_batch2_event_review import choose_full_annotation_songs  # noqa: E402


class ModernSongBatch2ReviewAnalysisTest(unittest.TestCase):
    def test_selection_includes_all_songs_and_one_r_only_per_category(self) -> None:
        categories = ("pop-synth", "hiphop-rnb", "dance-electronic", "latin-modern", "pop-rock", "singer-songwriter")
        songs = []
        order = 1
        for category in categories:
            for index in range(9):
                songs.append({
                    "sourceOrder": order,
                    "sourceId": str(order),
                    "artistName": f"s-{category}-{index}",
                    "trackName": "s",
                    "category": category,
                    "counts": {"S": 1, "R": 0, "K": 4, "I": 0},
                    "aggressiveCount": 1,
                    "eventScoreDbfs": {"mean": -20.0},
                })
                order += 1
            songs.append({
                "sourceOrder": order,
                "sourceId": str(order),
                "artistName": f"r-{category}",
                "trackName": "r",
                "category": category,
                "counts": {"S": 0, "R": 3, "K": 2, "I": 0},
                "aggressiveCount": 3,
                "eventScoreDbfs": {"mean": -15.0},
            })
            order += 1
        selected = choose_full_annotation_songs(songs)
        self.assertEqual(len(selected), 60)
        self.assertEqual(sum(song["counts"]["S"] > 0 for song in selected), 54)
        self.assertEqual(sum("R-only" in song["selectionReason"] for song in selected), 6)


if __name__ == "__main__":
    unittest.main()
