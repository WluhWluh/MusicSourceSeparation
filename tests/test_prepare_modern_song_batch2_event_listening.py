import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_modern_song_batch2_event_listening import (  # noqa: E402
    CANDIDATES_PER_SONG,
    SPLIT_WEIGHTS,
    allocate_counts,
)


class ModernSongBatch2PreparationTest(unittest.TestCase):
    def test_candidate_budget_is_small_first_pass(self) -> None:
        self.assertEqual(CANDIDATES_PER_SONG, 5)

    def test_split_target_uses_normalized_70_20_30_weights(self) -> None:
        self.assertEqual(
            allocate_counts(101, SPLIT_WEIGHTS),
            {"train": 59, "calibration": 17, "holdout": 25},
        )


if __name__ == "__main__":
    unittest.main()
