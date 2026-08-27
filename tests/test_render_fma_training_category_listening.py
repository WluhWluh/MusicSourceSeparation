import sys
import unittest
from collections import Counter
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import render_fma_training_category_listening as renderer  # noqa: E402


class RenderFmaTrainingCategoryListeningTest(unittest.TestCase):
    def test_category_selection_is_stable_and_rotates_stages(self) -> None:
        songs = []
        for category in ("alpha", "beta"):
            for stage in renderer.STAGES:
                for index in range(2):
                    songs.append(
                        {
                            "slug": f"{category}-{stage}-{index}",
                            "category": category,
                            "artistName": f"artist-{category}-{stage}-{index}",
                            "trainingStages": [stage],
                        }
                    )
        quotas = {"alpha": 3, "beta": 2}
        first = renderer.select_category_sample(songs, quotas, "fixture")
        second = renderer.select_category_sample(songs, quotas, "fixture")
        self.assertEqual(
            [song["slug"] for song in first],
            [song["slug"] for song in second],
        )
        self.assertEqual(Counter(song["category"] for song in first), Counter(quotas))
        self.assertEqual(
            {song["selectedForStage"] for song in first},
            set(renderer.STAGES),
        )

    def test_source_order_parsing(self) -> None:
        self.assertEqual(renderer.source_order_from_path("021-track.mp3"), 21)
        self.assertIsNone(renderer.source_order_from_path("track.mp3"))

    def test_complement_excludes_exact_slugs_and_reindexes(self) -> None:
        songs = [
            {
                "slug": f"song-{index}",
                "category": "hiphop-rnb",
                "artistName": f"Artist {index}",
                "trackName": f"Track {index}",
                "trainingStages": ["S86"],
            }
            for index in range(5)
        ]
        selected = renderer.select_complement(
            songs,
            {"songs": [{"slug": "song-1"}, {"slug": "song-3"}]},
        )
        self.assertEqual([song["slug"] for song in selected], ["song-0", "song-2", "song-4"])
        self.assertEqual([song["index"] for song in selected], [1, 2, 3])

    def test_complement_rejects_song_outside_pool(self) -> None:
        songs = [
            {
                "slug": "song-0",
                "category": "hiphop-rnb",
                "artistName": "Artist",
                "trackName": "Track",
                "trainingStages": ["S86"],
            }
        ]
        with self.assertRaises(ValueError):
            renderer.select_complement(songs, {"songs": [{"slug": "unknown"}]})

    def test_output_name_is_short_and_stable(self) -> None:
        song = {"index": 1, "slug": "fma-" + "very-long-name-" * 20}
        first = renderer.compact_output_name(song)
        second = renderer.compact_output_name(song)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 53)
        self.assertTrue(first.startswith("01-fma-very-long-name-"))
        self.assertTrue(first.endswith(".flac"))


if __name__ == "__main__":
    unittest.main()
