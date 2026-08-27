import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_inst3_fma_human_leakage_continuation as runner  # noqa: E402


class HumanLeakageHotspotTest(unittest.TestCase):
    def test_search_finds_strongest_projection_inside_mark_interval(self) -> None:
        frames = runner.CONTRACT.useful_samples * 2
        source = np.zeros((frames, 2), dtype=np.float32)
        teacher = np.zeros_like(source)
        candidate = np.zeros_like(source)
        expected = runner.CONTRACT.useful_samples
        half = round(runner.SAMPLE_RATE * 0.100 / 2)
        source[expected - half : expected + half] = 0.5
        candidate[expected - half : expected + half] = 0.5
        result = runner.search_mark_hotspot(
            source,
            teacher,
            candidate,
            expected / runner.SAMPLE_RATE + 0.75,
        )
        self.assertLessEqual(abs(result["centerSamples"] - expected), round(runner.SAMPLE_RATE * 0.010))
        self.assertGreater(result["metrics100"]["positiveProjectionDbfs"], -7.0)

    def test_nearby_selected_centers_are_merged_with_provenance(self) -> None:
        def row(index: int, center: int, projection: float):
            return {
                "sourceMarkIndex": index,
                "markSeconds": center / runner.SAMPLE_RATE,
                "markSamples": center,
                "centerSamples": center,
                "centerSeconds": center / runner.SAMPLE_RATE,
                "metrics100": {
                    "positiveProjectionDbfs": projection,
                    "missRmsDbfs": projection - 1,
                    "removedRmsDbfs": projection - 2,
                },
            }

        rows = [row(1, 100_000, -10.0), row(2, 101_000, -12.0), row(3, 110_000, -11.0)]
        selected = runner.deduplicate_hotspots(rows)
        self.assertEqual(len(selected), 2)
        first = min(selected, key=lambda item: item["centerSamples"])
        self.assertEqual(first["centerSamples"], 100_000)
        self.assertEqual(first["mergedMarkCount"], 2)
        self.assertEqual([item["sourceMarkIndex"] for item in first["sourceMarks"]], [1, 2])


class HumanLeakageScheduleTest(unittest.TestCase):
    def records(self):
        return [
            {"key": f"song-{index % 3}", "index": index // 3, "eventId": f"event-{index}"}
            for index in range(11)
        ]

    def test_schedule_is_equal_over_time_and_not_song_balanced(self) -> None:
        schedule, summary = runner.build_equal_schedule(self.records(), 11, 891, 24)
        self.assertEqual(len(schedule), 11 * 24)
        counts = Counter(schedule)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(summary["passes"][0]["eventCountMin"], 2)
        self.assertEqual(summary["passes"][0]["eventCountMax"], 3)
        self.assertEqual(summary["passes"][-1]["cumulativeEventCountMin"], 24)
        self.assertEqual(summary["passes"][-1]["cumulativeEventCountMax"], 24)

    def test_schedule_is_deterministic_and_changes_each_pass(self) -> None:
        first, _ = runner.build_equal_schedule(self.records(), 2, 891, 24)
        second, _ = runner.build_equal_schedule(self.records(), 2, 891, 24)
        self.assertEqual(first, second)
        self.assertNotEqual(first[:24], first[24:])


if __name__ == "__main__":
    unittest.main()
