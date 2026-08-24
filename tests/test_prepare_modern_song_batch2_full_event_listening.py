import unittest

import numpy as np

from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_modern_song_event_listening import PAIR_SAMPLES, SNIPPET_SAMPLES, make_pair  # noqa: E402


class ModernSongBatch2FullListeningTest(unittest.TestCase):
    def test_pair_contract_is_4_6_seconds(self) -> None:
        h50 = np.ones((SNIPPET_SAMPLES, 2), dtype=np.float32)
        inst3 = np.ones((SNIPPET_SAMPLES, 2), dtype=np.float32) * -1.0
        pair = make_pair(h50, inst3)
        self.assertEqual(pair.shape, (PAIR_SAMPLES, 2))
        self.assertEqual(PAIR_SAMPLES, 202_860)
        np.testing.assert_allclose(pair[-13_230:], 0.0)


if __name__ == "__main__":
    unittest.main()
