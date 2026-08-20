from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import prepare_inst3_scale10_data as prepare  # noqa: E402
import run_inst3_aggressive_scale10 as scale10  # noqa: E402


class Inst3Scale10Test(unittest.TestCase):
    def test_selection_is_rank_ordered_and_train_only(self) -> None:
        manifest = {
            "manifestId": "musdb18-inst3-oracle-split@1",
            "entries": [
                {"role": "train", "rank": "b", "member": "train/b", "fileName": "b", "sourceSha256": "2"},
                {"role": "train", "rank": "a", "member": "train/a", "fileName": "a", "sourceSha256": "1"},
                {"role": "calibration", "rank": "0", "member": "train/c", "fileName": "c", "sourceSha256": "3"},
            ],
        }
        selected = prepare.select_train_entries(manifest, 2)
        self.assertEqual([entry["member"] for entry in selected], ["train/a", "train/b"])

    def test_coverage_selection_spans_candidates(self) -> None:
        class Item:
            def __init__(self, start: int) -> None:
                self.start = start
                self.length = scale10.pilot.DEFAULT_CONFIG.useful_samples
                self.rms = float(start + 1)
                self.vocal_rms = float(100 - start)

        candidates = [Item(index) for index in range(20)]
        selected = scale10.select_coverage_windows(candidates, 6, 891)
        starts = [item.start for item in selected]
        self.assertEqual(len(starts), 6)
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 19)

    def test_audio_target_is_linear(self) -> None:
        instrumental = np.ones((4, 2), dtype=np.float32)
        teacher = np.zeros((4, 2), dtype=np.float32)
        self.assertTrue(np.allclose((1.0 - 0.25) * instrumental + 0.25 * teacher, 0.75))

    def test_compaction_preserves_selected_samples(self) -> None:
        samples = np.arange(24, dtype=np.float32).reshape(12, 2)
        song = scale10.pilot.SongBundle(
            role="train",
            member="train/fixture.stem.mp4",
            source_sha256="fixture",
            source_path=Path("fixture.stem.mp4"),
            slug="fixture",
            mixture_encoded=samples,
            mixture_gt=samples,
            vocals=samples + 100,
            instrumental=samples + 200,
            drums=np.empty((0, 2), dtype=np.float32),
            bass=np.empty((0, 2), dtype=np.float32),
            other=np.empty((0, 2), dtype=np.float32),
            sample_rate=44_100,
            teacher_instrumental=samples + 300,
        )
        records = [
            scale10.pilot.WindowRecord(song, 2, 3, 1.0, 2.0),
            scale10.pilot.WindowRecord(song, 8, 2, 3.0, 4.0),
        ]
        compact, groups = scale10.compact_song(song, [("evaluation", records)])
        expected = np.concatenate([samples[2:5], samples[8:10]], axis=0)
        np.testing.assert_array_equal(compact.mixture_gt, expected)
        self.assertIs(compact.mixture_gt, compact.mixture_encoded)
        self.assertEqual([record.start for record in groups["evaluation"]], [0, 3])
        self.assertEqual([record.length for record in groups["evaluation"]], [3, 2])


if __name__ == "__main__":
    unittest.main()
