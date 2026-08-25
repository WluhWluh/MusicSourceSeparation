import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import prepare_all_fma_s86_event_listening as preparation  # noqa: E402


class PrepareAllFmaS86EventListeningTest(unittest.TestCase):
    def test_y_selection_excludes_n(self) -> None:
        records = [{"order": 1}, {"order": 2}, {"order": 3}]
        reviews = {
            1: {"keep": "N"},
            2: {"keep": "Y"},
            3: {"keep": "N"},
        }
        self.assertEqual(
            [row["order"] for row in preparation.select_y_records(records, reviews)],
            [2],
        )

    def test_scan_contract_returns_five_candidates(self) -> None:
        samples = 44_100 * 12
        time_axis = np.arange(samples, dtype=np.float32)[:, None]
        source = np.concatenate(
            (0.1 * np.sin(time_axis / 1000.0), 0.1 * np.cos(time_axis / 900.0)),
            axis=1,
        ).astype(np.float32)
        teacher = np.zeros_like(source)
        candidate = source.copy()
        rows = preparation.event_tools.scan_candidates(
            source, teacher, candidate, "fixture", 50, 5, 5
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual([row["songCandidateIndex"] for row in rows], list(range(1, 6)))

    def test_pair_has_exact_silence_layout(self) -> None:
        h50 = np.full((preparation.SNIPPET_SAMPLES, 2), 0.25, dtype=np.float32)
        inst3 = np.full((preparation.SNIPPET_SAMPLES, 2), -0.25, dtype=np.float32)
        pair = preparation.make_pair(h50, inst3)
        self.assertEqual(pair.shape, (preparation.PAIR_SAMPLES, 2))
        first_gap = pair[preparation.SNIPPET_SAMPLES : preparation.SNIPPET_SAMPLES + preparation.GAP_SAMPLES]
        second_gap = pair[preparation.SNIPPET_SAMPLES + preparation.GAP_SAMPLES + preparation.SNIPPET_SAMPLES :]
        self.assertTrue(np.array_equal(first_gap, np.zeros_like(first_gap)))
        self.assertTrue(np.array_equal(second_gap, np.zeros_like(second_gap)))
        self.assertTrue(np.all(pair[: preparation.SNIPPET_SAMPLES] == 0.25))
        self.assertTrue(
            np.all(
                pair[
                    preparation.SNIPPET_SAMPLES
                    + preparation.GAP_SAMPLES : preparation.SNIPPET_SAMPLES
                    + preparation.GAP_SAMPLES
                    + preparation.SNIPPET_SAMPLES
                ]
                == -0.25
            )
        )

    def test_csv_starts_with_mark_and_event_id(self) -> None:
        event = {
            "eventId": "fixture-event-01",
            "serial": 1,
            "sourceBatch": "batch1",
            "sourceOrder": 2,
            "priorSplitRole": "train",
            "category": "pop-synth",
            "languageCode": "en",
            "artistName": "Artist",
            "trackName": "Track",
            "centerSeconds": 1.0,
            "eventScoreDbfs": -20.0,
            "metrics": {
                "50": {"positiveProjectionDbfs": -21.0},
                "100": {"positiveProjectionDbfs": -22.0},
                "200": {"positiveProjectionDbfs": -23.0},
            },
            "outputs": {
                "pairedListening": {"file": "pair.flac"},
                "h50ContinuationPlus5": {"file": "h50.flac"},
                "inst3Instrumental": {"file": "inst3.flac"},
                "mixture": {"file": "mix.flac"},
            },
            "license": "CC",
            "licenseUrl": "https://example.invalid",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            preparation.write_review_csv(path, [event])
            with path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header[:2], ["mark", "eventId"])


if __name__ == "__main__":
    unittest.main()
