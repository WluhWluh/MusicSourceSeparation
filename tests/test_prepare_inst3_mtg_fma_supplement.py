import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_inst3_mtg_fma_supplement import duration_seconds, safe_name  # noqa: E402


class PrepareSupplementTest(unittest.TestCase):
    def test_duration_parser(self) -> None:
        self.assertAlmostEqual(duration_seconds("03:27"), 207.0)
        self.assertAlmostEqual(duration_seconds("01:02:03"), 3723.0)

    def test_safe_name(self) -> None:
        self.assertEqual(safe_name("Artist / Song"), "artist-song")


if __name__ == "__main__":
    unittest.main()
