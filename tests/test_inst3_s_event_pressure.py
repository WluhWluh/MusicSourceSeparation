import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_s_event_pressure import (  # noqa: E402
    BASE_PRESSURE_PASS,
    BASE_PRESSURE_STEP,
    RECORDS_PER_PASS,
    UPDATES_PER_PASS,
    build_future_schedule,
    pressure_pass_from_step,
)


class Inst3SEventPressureTest(unittest.TestCase):
    def _records(self, prefix: str, count: int, songs: int) -> list[dict[str, object]]:
        return [
            {
                "key": f"external::{prefix}::song-{index % songs:02d}",
                "index": index,
                "songKey": f"external::{prefix}::song-{index % songs:02d}",
                "eventId": f"{prefix}-{index}",
            }
            for index in range(count)
        ]

    def test_pass_mapping(self) -> None:
        self.assertEqual(pressure_pass_from_step(BASE_PRESSURE_STEP), BASE_PRESSURE_PASS)
        self.assertEqual(pressure_pass_from_step(BASE_PRESSURE_STEP + UPDATES_PER_PASS), BASE_PRESSURE_PASS + 1)
        with self.assertRaises(ValueError):
            pressure_pass_from_step(BASE_PRESSURE_STEP + 1)

    def test_future_schedule_is_s_only_and_budgeted(self) -> None:
        s_records = self._records("S", 86, 38)
        r_records = self._records("R", 46, 20)
        schedule, info = build_future_schedule(s_records, r_records, BASE_PRESSURE_PASS, 2, 891)
        self.assertEqual(len(schedule), 2 * RECORDS_PER_PASS)
        self.assertEqual(info["poolCounts"], {"S": 2 * RECORDS_PER_PASS})
        self.assertEqual(info["uniqueRecords"], 86)


if __name__ == "__main__":
    unittest.main()
