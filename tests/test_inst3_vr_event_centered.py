import sys
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_inst3_vr_event_centered import (  # noqa: E402
    EventRecord,
    centered_records,
    event_frame_mask,
    make_block_event_rows,
    load_existing_events,
    rank_events,
)


class Inst3VrEventCenteredTest(unittest.TestCase):
    def test_block_rows_keep_absolute_offsets(self) -> None:
        mixture = np.ones((8_820, 2), dtype=np.float32)
        teacher = np.zeros_like(mixture)
        predicted = np.zeros_like(mixture)
        predicted[2_205:4_410] = -0.5
        rows = make_block_event_rows(
            candidate=type("Candidate", (), {"index": 3, "start": 10_000, "length": 8_820})(),
            mixture=mixture,
            teacher_segment=teacher,
            predicted_residual=predicted,
            sample_rate=44_100,
        )
        self.assertTrue(rows)
        self.assertTrue(all(item.block_start >= 10_000 for item in rows))
        self.assertTrue(any(item.duration_ms == 50 for item in rows))

    def test_rank_prefers_projection_and_rejects_overlap_first(self) -> None:
        rows = [
            EventRecord(0, 0, 100, 200, 100, 0.9, 0.1, 1.0),
            EventRecord(0, 0, 110, 160, 50, 0.8, 0.1, 1.0),
            EventRecord(0, 0, 500, 600, 100, 0.7, 0.1, 1.0),
        ]
        selected = rank_events(rows, 2)
        self.assertEqual([item.block_start for item in selected], [100, 500])

    def test_centered_window_clamps_and_contains_event(self) -> None:
        event = EventRecord(0, 0, 9_900, 10_000, 100, 0.5, 0.2, 1.0)
        result = centered_records([event], song_samples=10_000, useful_samples=6_000)[0]
        self.assertEqual(result.start, 4_000)
        self.assertLessEqual(result.start, result.event_start)
        self.assertLessEqual(result.event_end, result.start + result.length)

    def test_event_frame_mask_requires_event_inside_window(self) -> None:
        mask = event_frame_mask(
            candidate_start=0,
            candidate_length=119_808,
            event_start=50_000,
            event_end=54_410,
        )
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertGreater(int(mask.sum()), 0)
        with self.assertRaises(ValueError):
            event_frame_mask(
                candidate_start=0,
                candidate_length=100,
                event_start=200,
                event_end=300,
            )

    def test_resume_reads_scan_manifest_and_centered_cache_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scan = root / "scan" / "songs"
            selection = root / "selection"
            scan.mkdir(parents=True)
            selection.mkdir(parents=True)
            rows = [
                {
                    "candidateIndex": index,
                    "candidateStartSamples": index * 10_000,
                    "blockStartSamples": index * 10_000 + 100,
                    "blockEndSamples": index * 10_000 + 2_305,
                    "durationMs": 50,
                    "positiveProjectionRms": 0.1,
                    "missRms": 0.2,
                    "teacherRms": 0.3,
                }
                for index in range(3)
            ]
            (scan / "song.json").write_text(
                '{"selected": ' + json.dumps(rows) + "}",
                encoding="utf-8",
            )
            events = load_existing_events(
                root=root,
                slug="song",
                h50_checkpoint_sha256="sha",
                expected_count=3,
            )
            self.assertIsNotNone(events)
            self.assertEqual([item.block_start for item in events or []], [100, 10_100, 20_100])


if __name__ == "__main__":
    unittest.main()
