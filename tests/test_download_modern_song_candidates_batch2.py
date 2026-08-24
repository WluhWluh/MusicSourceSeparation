import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from download_modern_song_candidates_batch2 import BATCH2_COUNT, BATCH2_QUOTAS  # noqa: E402


class ModernSongBatch2Test(unittest.TestCase):
    def test_surplus_quota_is_above_minimum_target(self) -> None:
        self.assertEqual(BATCH2_COUNT, 118)
        self.assertEqual(sum(BATCH2_QUOTAS.values()), BATCH2_COUNT)
        self.assertEqual(BATCH2_QUOTAS["dance-electronic"], 12)
        self.assertEqual(BATCH2_QUOTAS["latin-modern"], 18)


if __name__ == "__main__":
    unittest.main()
