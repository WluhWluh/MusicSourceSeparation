import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from render_inst3_training_event_listening import safe_name, snippet_bounds  # noqa: E402


class TrainingEventListeningTest(unittest.TestCase):
    def test_snippet_stays_inside_useful_window_and_marks_event(self) -> None:
        event = {
            "startSamples": 100_000,
            "lengthSamples": 119_808,
            "eventStartSamples": 50_000,
            "eventEndSamples": 52_205,
            "eventCenterSamples": 151_102,
        }
        start, end, event_start, event_end = snippet_bounds(event, 400_000, 44_100)
        self.assertGreaterEqual(start, 100_000)
        self.assertLessEqual(end, 219_808)
        self.assertGreaterEqual(event_start, 0)
        self.assertLess(event_start, event_end)

    def test_safe_name_is_ascii_path_component(self) -> None:
        self.assertEqual(safe_name("a song/with spaces"), "a-song-with-spaces")


if __name__ == "__main__":
    unittest.main()
