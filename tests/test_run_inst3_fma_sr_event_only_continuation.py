import sys
import unittest
from collections import Counter
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_inst3_fma_sr_event_only_continuation as runner  # noqa: E402


class FmaSrEventOnlyScheduleTest(unittest.TestCase):
    def _records(self):
        records = []
        for index in range(39):
            records.append({"key": "song-s", "index": index, "mark": "S", "eventId": f"s-{index}"})
        for index in range(62):
            records.append({"key": "song-r", "index": index, "mark": "R", "eventId": f"r-{index}"})
        return records

    def test_event_level_s_frequency_is_about_twice_r(self) -> None:
        schedule, summary = runner.build_schedule(self._records(), 1, 891)
        self.assertEqual(len(schedule), runner.RECORDS_PER_PASS)
        counts = Counter(schedule)
        s_counts = [counts[("song-s", index)] for index in range(39)]
        r_counts = [counts[("song-r", index)] for index in range(62)]
        self.assertEqual(set(s_counts), {10, 11})
        self.assertEqual(set(r_counts), {5})
        self.assertEqual(summary["passes"][0]["sRecords"], 394)
        self.assertEqual(summary["passes"][0]["rRecords"], 310)

    def test_schedule_is_deterministic_and_changes_by_pass(self) -> None:
        records = self._records()
        first, _ = runner.build_schedule(records, 3, 891)
        second, _ = runner.build_schedule(records, 3, 891)
        self.assertEqual(first, second)
        self.assertNotEqual(first[:runner.RECORDS_PER_PASS], first[runner.RECORDS_PER_PASS:2 * runner.RECORDS_PER_PASS])


if __name__ == "__main__":
    unittest.main()
