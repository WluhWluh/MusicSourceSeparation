import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_modern_s_pilot import (  # noqa: E402
    HOLDOUT_QUOTAS,
    TRAIN_QUOTAS,
    build_song_split,
    choose_s_events,
    s_clusters,
)


class ModernSPilotTest(unittest.TestCase):
    def test_cluster_selection_is_capped_at_two(self) -> None:
        events = [
            {"mark": "S", "centerSeconds": 10.0, "eventScoreDbfs": -20.0},
            {"mark": "S", "centerSeconds": 11.0, "eventScoreDbfs": -10.0},
            {"mark": "S", "centerSeconds": 20.0, "eventScoreDbfs": -15.0},
            {"mark": "R", "centerSeconds": 30.0, "eventScoreDbfs": -5.0},
        ]
        self.assertEqual([len(group) for group in s_clusters(events)], [2, 1])
        selected = choose_s_events("song", {"song": events})
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["centerSeconds"], 11.0)

    def test_split_uses_frozen_quotas(self) -> None:
        records = {}
        events = {}
        counts = {"hiphop-rnb": 11, "pop-synth": 9, "latin-modern": 5, "singer-songwriter": 6, "dance-electronic": 2, "pop-rock": 2}
        index = 0
        for category, count in counts.items():
            for _ in range(count):
                slug = f"{category}-{index}"
                records[slug] = {"slug": slug, "category": category, "artistName": slug, "trackName": slug}
                events[slug] = [{"mark": "S", "centerSeconds": float(j * 10), "eventScoreDbfs": -20.0} for j in range(2)]
                index += 1
        train, holdout, _ = build_song_split(records, events)
        self.assertEqual(len(train), sum(TRAIN_QUOTAS.values()))
        self.assertEqual(len(holdout), sum(HOLDOUT_QUOTAS.values()))
        self.assertFalse(set(train) & set(holdout))


if __name__ == "__main__":
    unittest.main()
