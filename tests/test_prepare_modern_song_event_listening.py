import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_modern_song_event_listening import (  # noqa: E402
    GAP_SAMPLES,
    PAIR_SAMPLES,
    SNIPPET_SAMPLES,
    make_pair,
)


class PrepareModernSongEventListeningTest(unittest.TestCase):
    def test_pair_has_trailing_silence(self) -> None:
        h50 = np.ones((SNIPPET_SAMPLES, 2), dtype=np.float32) * 0.25
        inst3 = np.ones((SNIPPET_SAMPLES, 2), dtype=np.float32) * -0.5
        pair = make_pair(h50, inst3)
        self.assertEqual(pair.shape, (PAIR_SAMPLES, 2))
        np.testing.assert_allclose(pair[:SNIPPET_SAMPLES], h50)
        np.testing.assert_allclose(pair[SNIPPET_SAMPLES : SNIPPET_SAMPLES + GAP_SAMPLES], 0.0)
        second = SNIPPET_SAMPLES + GAP_SAMPLES
        np.testing.assert_allclose(pair[second : second + SNIPPET_SAMPLES], inst3)
        np.testing.assert_allclose(pair[-GAP_SAMPLES:], 0.0)


if __name__ == "__main__":
    unittest.main()
