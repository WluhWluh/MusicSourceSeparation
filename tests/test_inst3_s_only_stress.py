import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_s_only_stress import (  # noqa: E402
    BALANCED_EXTRA,
    RECORDS_PER_PASS,
    REPEAT_FACTOR,
    S_RECORDS,
    build_schedule,
)


class Inst3SOnlyStressTest(unittest.TestCase):
    def _records(self) -> list[dict[str, object]]:
        return [
            {
                "key": f"external::song-{song:02d}",
                "index": event,
                "songKey": f"external::song-{song:02d}",
                "eventId": f"event-{song:02d}-{event}",
            }
            for song in range(38)
            for event in range(3 if song < 10 else 2)
        ]

    def test_schedule_has_fixed_all_s_budget_and_is_deterministic(self) -> None:
        records = self._records()
        self.assertEqual(len(records), S_RECORDS)
        first, first_summary = build_schedule(records, 5, 891)
        second, second_summary = build_schedule(records, 5, 891)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary["recordsPerPass"], RECORDS_PER_PASS)
        self.assertEqual(first_summary["updates"], 5 * RECORDS_PER_PASS // 4)
        for pass_index in range(5):
            current = first[pass_index * RECORDS_PER_PASS : (pass_index + 1) * RECORDS_PER_PASS]
            self.assertEqual(len(current), RECORDS_PER_PASS)
            counts = {item: current.count(item) for item in set(current)}
            self.assertEqual(set(counts.values()), {REPEAT_FACTOR, REPEAT_FACTOR + 1})
            self.assertEqual(sum(value == REPEAT_FACTOR + 1 for value in counts.values()), BALANCED_EXTRA)

    def test_schedule_changes_with_seed(self) -> None:
        records = self._records()
        first, _ = build_schedule(records, 1, 891)
        second, _ = build_schedule(records, 1, 892)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
