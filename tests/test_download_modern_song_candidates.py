import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from download_modern_song_candidates import (  # noqa: E402
    accepted_license,
    category_for,
    duration_seconds,
    likely_vocal_track,
)


class ModernSongCandidateTest(unittest.TestCase):
    def test_license_rejects_nd(self) -> None:
        self.assertTrue(accepted_license("Attribution-NonCommercial-ShareAlike 3.0"))
        self.assertTrue(accepted_license("Creative Commons Attribution-Share-Alike"))
        self.assertFalse(accepted_license("Attribution-NonCommercial-NoDerivatives 4.0"))

    def test_category_excludes_experimental(self) -> None:
        self.assertEqual(category_for(["Pop", "Synth Pop"], "en"), "pop-synth")
        self.assertIsNone(category_for(["Experimental Pop", "Pop"], "en"))

    def test_duration_parser(self) -> None:
        self.assertEqual(duration_seconds("04:10"), 250.0)

    def test_vocal_filter_rejects_explicit_instrumental(self) -> None:
        row = {
            "track_title": "Utopia (instrumental)",
            "album_title": "Album",
            "track_language_code": "en",
            "track_lyricist": "",
            "track_information": "",
            "tags": "[]",
        }
        self.assertFalse(likely_vocal_track(row, ["Pop", "Dance"], "pop-synth"))


if __name__ == "__main__":
    unittest.main()
