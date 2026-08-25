import csv
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_modern_song_fma_s86_review as analysis  # noqa: E402


def row(mark: str, batch: str, order: int, center: float) -> dict[str, str]:
    return {
        "mark": mark,
        "eventId": f"{batch}-{order}-{center}",
        "sourceBatch": batch,
        "sourceOrder": str(order),
        "centerSeconds": str(center),
        "category": "pop-synth",
        "priorSplitRole": "train",
        "positiveProjection100Dbfs": "-20",
        "batch": batch,
        "order": order,
        "center": center,
        "role": "train",
        "categoryValue": "pop-synth",
    }


class AnalyzeModernSongFmaS86ReviewTest(unittest.TestCase):
    def test_load_rows_normalizes_current_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "mark", "eventId", "sourceBatch", "sourceOrder", "centerSeconds",
                        "category", "priorSplitRole", "positiveProjection100Dbfs",
                    ),
                )
                writer.writeheader()
                values = row("S", "batch2", 7, 1.5)
                writer.writerow({key: values[key] for key in writer.fieldnames})
            loaded = analysis.load_rows(path)
        self.assertEqual(loaded[0]["batch"], "batch2")
        self.assertEqual(loaded[0]["order"], 7)
        self.assertEqual(loaded[0]["center"], 1.5)

    def test_mark_summary_counts_aggressive_subset(self) -> None:
        values = [row(mark, "batch1", 1, index) for index, mark in enumerate(("S", "R", "K", "I"))]
        summary = analysis.mark_summary(values)
        self.assertEqual(summary["marks"], {"S": 1, "R": 1, "K": 1, "I": 1})
        self.assertEqual(summary["aggressiveCountSPlusR"], 2)
        self.assertEqual(summary["aggressiveFractionSPlusR"], 0.5)

    def test_nearest_matching_is_one_to_one_and_respects_tolerance(self) -> None:
        current = analysis.group_by(
            [row("S", "batch1", 1, 1.0), row("K", "batch1", 1, 2.0)],
            analysis.song_key,
        )
        prior = analysis.group_by(
            [row("R", "batch1", 1, 1.1), row("I", "batch1", 1, 2.8)],
            analysis.song_key,
        )
        matches = analysis.nearest_matches(current, prior, [("batch1", 1)], 0.2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["current"]["mark"], "S")
        self.assertEqual(matches[0]["prior"]["mark"], "R")

    def test_song_pool_excludes_i_only_songs(self) -> None:
        values = [
            row("S", "batch2", 1, 1.0),
            row("I", "batch2", 1, 2.0),
            row("K", "batch2", 1, 3.0),
            row("I", "batch2", 1, 4.0),
            row("I", "batch2", 1, 5.0),
            row("R", "batch2", 2, 1.0),
            row("K", "batch2", 2, 2.0),
            row("K", "batch2", 2, 3.0),
            row("K", "batch2", 2, 4.0),
            row("K", "batch2", 2, 5.0),
        ]
        pool = analysis.current_pool_summary(values)
        self.assertEqual(pool["strictNoI"]["songCount"], 1)
        self.assertEqual(pool["strictNoI"]["aggressiveEventCount"], 1)


if __name__ == "__main__":
    unittest.main()
