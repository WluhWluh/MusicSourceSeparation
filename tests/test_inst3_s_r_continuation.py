import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_s_r_continuation import (  # noqa: E402
    ARMS,
    RECORDS_PER_PASS,
    build_schedules,
)


class Inst3SrContinuationTest(unittest.TestCase):
    def _pools(self):
        s_records = [
            {
                "key": f"external::S::song-{song:02d}",
                "index": event,
                "songKey": f"external::S::song-{song:02d}",
                "eventId": f"s-{song:02d}-{event}",
            }
            for song in range(38)
            for event in range(3 if song < 10 else 2)
        ]
        r_records = [
            {
                "key": f"external::R::song-{song:02d}",
                "index": event,
                "songKey": f"external::R::song-{song:02d}",
                "eventId": f"r-{song:02d}-{event}",
            }
            for song in range(20)
            for event in range(3 if song < 6 else 2)
        ]
        paths = {
            f"musdb::{song:02d}": Path(f"{song:02d}.npz")
            for song in range(80)
        }
        self.assertEqual(len(s_records), 86)
        self.assertEqual(len(r_records), 46)
        return paths, s_records, r_records

    def test_four_arms_have_equal_budget_and_expected_pool_counts(self) -> None:
        paths, s_records, r_records = self._pools()
        schedules, summary = build_schedules(paths, s_records, r_records, 2, 891)
        self.assertEqual(tuple(schedules), ARMS)
        for arm in ARMS:
            self.assertEqual(len(schedules[arm]), 2 * RECORDS_PER_PASS)
            self.assertEqual(summary["schedules"][arm]["recordCount"], 2 * RECORDS_PER_PASS)
        for pass_index in range(2):
            begin = pass_index * RECORDS_PER_PASS
            end = (pass_index + 1) * RECORDS_PER_PASS
            for arm in ARMS:
                current = schedules[arm][begin:end]
                counts = {"MUSDB": 0, "S": 0, "R": 0}
                for key, _ in current:
                    if key.startswith("musdb::"):
                        counts["MUSDB"] += 1
                    elif "::S::" in key:
                        counts["S"] += 1
                    else:
                        counts["R"] += 1
                if arm == ARMS[0]:
                    self.assertEqual(counts, {"MUSDB": 640, "S": 64, "R": 0})
                elif arm == ARMS[1]:
                    self.assertEqual(counts, {"MUSDB": 640, "S": 32, "R": 32})
                elif arm == ARMS[2]:
                    self.assertEqual(counts, {"MUSDB": 0, "S": 704, "R": 0})
                else:
                    self.assertEqual(counts, {"MUSDB": 0, "S": 452, "R": 252})

    def test_schedule_is_deterministic_and_event_only_uses_every_record(self) -> None:
        paths, s_records, r_records = self._pools()
        first, first_summary = build_schedules(paths, s_records, r_records, 5, 891)
        second, second_summary = build_schedules(paths, s_records, r_records, 5, 891)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        s_used = {
            item
            for item in first[ARMS[2]]
            if "::S::" in item[0]
        }
        sr_used = {
            item
            for item in first[ARMS[3]]
            if "::S::" in item[0] or "::R::" in item[0]
        }
        self.assertEqual(len(s_used), 86)
        self.assertEqual(len(sr_used), 132)


if __name__ == "__main__":
    unittest.main()
