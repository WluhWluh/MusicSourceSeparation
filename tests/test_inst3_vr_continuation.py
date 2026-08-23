import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_inst3_vr_continuation import build_offset_schedule  # noqa: E402


class ContinuationTest(unittest.TestCase):
    def test_offset_schedule_is_deterministic_and_uses_same_records(self) -> None:
        first = build_offset_schedule(["b", "a"], 3, 2, 891, 5)
        second = build_offset_schedule(["b", "a"], 3, 2, 891, 5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual({item for item in first}, {("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)})

    def test_offset_changes_pass_permutation(self) -> None:
        first = build_offset_schedule(["a", "b"], 4, 1, 891, 0)
        continued = build_offset_schedule(["a", "b"], 4, 1, 891, 5)
        self.assertNotEqual(first, continued)


if __name__ == "__main__":
    unittest.main()
