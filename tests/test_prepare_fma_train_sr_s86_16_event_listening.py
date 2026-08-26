import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import prepare_fma_train_sr_s86_16_event_listening as preparation  # noqa: E402


class PrepareFmaTrainSrS86Test(unittest.TestCase):
    def test_selection_keeps_song_with_i_and_requires_train_sr(self) -> None:
        report = {
            "songs": [
                {"sourceBatch": "batch2", "sourceOrder": 1, "priorSplitRole": "train"},
                {"sourceBatch": "batch2", "sourceOrder": 2, "priorSplitRole": "train"},
                {"sourceBatch": "batch2", "sourceOrder": 3, "priorSplitRole": "holdout"},
            ]
        }
        rows = [
            {"sourceBatch": "batch2", "sourceOrder": "1", "mark": "S", "eventId": "a"},
            {"sourceBatch": "batch2", "sourceOrder": "1", "mark": "I", "eventId": "b"},
            {"sourceBatch": "batch2", "sourceOrder": "2", "mark": "K", "eventId": "c"},
            {"sourceBatch": "batch2", "sourceOrder": "3", "mark": "R", "eventId": "d"},
        ]
        selected = preparation.select_songs(report, rows)
        self.assertEqual([int(song["sourceOrder"]) for song in selected], [1])

    def test_pair_layout(self) -> None:
        h50 = np.full((preparation.SNIPPET_SAMPLES, 2), 0.25, dtype=np.float32)
        inst3 = np.full((preparation.SNIPPET_SAMPLES, 2), -0.25, dtype=np.float32)
        pair = preparation.make_pair(h50, inst3)
        self.assertEqual(pair.shape, (preparation.PAIR_SAMPLES, 2))
        self.assertTrue(np.all(pair[preparation.SNIPPET_SAMPLES:preparation.SNIPPET_SAMPLES + preparation.GAP_SAMPLES] == 0))
        second_start = preparation.SNIPPET_SAMPLES + preparation.GAP_SAMPLES + preparation.SNIPPET_SAMPLES
        self.assertTrue(np.all(pair[second_start:] == 0))

    def test_csv_mark_is_first_and_blank(self) -> None:
        event = {
            "eventId": "e1", "serial": 1, "sourceBatch": "batch2", "sourceOrder": 1,
            "priorSplitRole": "train", "category": "pop", "languageCode": "",
            "artistName": "A", "trackName": "T", "centerSeconds": 1.0,
            "eventScoreDbfs": -20.0,
            "metrics": {"50": {"positiveProjectionDbfs": -21}, "100": {"positiveProjectionDbfs": -22}, "200": {"positiveProjectionDbfs": -23}},
            "outputs": {"pairedListening": {"file": "pair"}, "h50ContinuationPlus5": {"file": "h50"}, "inst3Instrumental": {"file": "i3"}, "mixture": {"file": "mix"}},
            "license": "CC", "licenseUrl": "url",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            preparation.write_review_csv(path, [event])
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0][:2], ["mark", "eventId"])
            self.assertEqual(rows[1][0], "")


if __name__ == "__main__":
    unittest.main()
