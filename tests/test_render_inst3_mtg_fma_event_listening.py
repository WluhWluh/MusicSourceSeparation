import sys
import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from render_inst3_mtg_fma_event_listening import (  # noqa: E402
    PAIR_GAP_SAMPLES,
    PAIR_SAMPLES,
    SNIPPET_SAMPLES,
    event_score,
    make_listening_pair,
    scan_candidates,
    write_review_csv,
)


class MtfFmaEventListeningTest(unittest.TestCase):
    def test_scan_returns_full_two_second_events_with_temporal_coverage(self) -> None:
        sample_rate = 44_100
        length = sample_rate * 20
        source = np.zeros((length, 2), dtype=np.float32)
        teacher = np.zeros_like(source)
        candidate = np.zeros_like(source)
        # Four removed-content events in separate temporal regions.
        for second in (2, 7, 12, 17):
            teacher[second * sample_rate : second * sample_rate + 2_000] = 0.4
            candidate[second * sample_rate : second * sample_rate + 2_000] = 0.2
        rows = scan_candidates(source, teacher, candidate, "song", 100, 4, 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["snippetEndSamples"] - row["snippetStartSamples"] == SNIPPET_SAMPLES for row in rows))
        centers = [row["centerSamples"] for row in rows]
        self.assertGreater(max(centers) - min(centers), sample_rate * 4)

    def test_event_score_prefers_coherent_projection(self) -> None:
        weak = {ms: {"positiveProjectionDbfs": -40.0, "missRmsDbfs": -30.0} for ms in (50, 100, 200)}
        strong = {ms: {"positiveProjectionDbfs": -20.0, "missRmsDbfs": -30.0} for ms in (50, 100, 200)}
        self.assertGreater(event_score(strong), event_score(weak))

    def test_listening_pair_has_requested_order_and_gap(self) -> None:
        h50 = np.ones((SNIPPET_SAMPLES, 2), dtype=np.float32) * 0.25
        inst3 = np.ones((SNIPPET_SAMPLES, 2), dtype=np.float32) * -0.5
        pair = make_listening_pair(h50, inst3)
        self.assertEqual(pair.shape, (PAIR_SAMPLES, 2))
        np.testing.assert_allclose(pair[:SNIPPET_SAMPLES], h50)
        np.testing.assert_allclose(pair[SNIPPET_SAMPLES : SNIPPET_SAMPLES + PAIR_GAP_SAMPLES], 0.0)
        np.testing.assert_allclose(pair[-SNIPPET_SAMPLES:], inst3)

    def test_review_csv_preserves_existing_human_marks(self) -> None:
        event = {
            "eventId": "event-1",
            "sourceSet": "expansion",
            "role": "candidate-pool",
            "source": "fma",
            "sourceId": "1",
            "artistName": "Artist",
            "trackName": "Track",
            "license": "CC",
            "centerSeconds": 1.0,
            "eventScoreDbfs": -20.0,
            "metrics": {str(ms): {"positiveProjectionDbfs": -20.0} for ms in (50, 100, 200)},
            "outputs": {
                "h50ContinuationPlus5": {"file": "h50.flac"},
                "inst3Instrumental": {"file": "inst3.flac"},
                "mixture": {"file": "mixture.flac"},
                "pairedListening": {"file": "pair.flac"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_review_csv(path, [event])
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["mark"] = "R"
            rows[0]["notes"] = "keep this review"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            write_review_csv(path, [event])
            with path.open(encoding="utf-8-sig", newline="") as handle:
                preserved = list(csv.DictReader(handle))[0]
            self.assertEqual(preserved["mark"], "R")
            self.assertEqual(preserved["notes"], "keep this review")


if __name__ == "__main__":
    unittest.main()
