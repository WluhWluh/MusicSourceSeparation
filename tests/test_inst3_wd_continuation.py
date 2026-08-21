from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_inst3_initialization_output_matrix as matrix  # noqa: E402
import run_inst3_wd_continuation as continuation  # noqa: E402


class Inst3WdContinuationTest(unittest.TestCase):
    def test_default_contract_is_seed_891_from_100_to_200_passes(self) -> None:
        args = continuation.parse_args([])

        seeds, milestones = continuation.validate_args(args)

        self.assertEqual(seeds, (891,))
        self.assertEqual(milestones, (100, 125, 150, 200))
        self.assertEqual(args.start_passes, 100)
        self.assertEqual(args.passes, 200)

    def test_continuation_schedule_preserves_the_matrix_prefix(self) -> None:
        continuation_schedule = matrix.build_sample_schedule(
            320, 200, matrix.make_schedule_seed(891)
        )
        parent_schedule = continuation.expected_parent_schedule(
            window_count=320, start_passes=100, seed=891
        )

        np.testing.assert_array_equal(
            continuation_schedule[: len(parent_schedule)], parent_schedule
        )

    def test_parent_validation_rejects_a_changed_prefix_contract(self) -> None:
        args = SimpleNamespace(
            start_passes=100,
            batch_size=4,
            learning_rate=1.0e-4,
            train_windows_per_song=32,
            holdout_windows_per_song=16,
            eval_windows_per_song=16,
        )
        schedule = matrix.build_sample_schedule(320, 200, matrix.make_schedule_seed(891))
        parent_schedule = continuation.expected_parent_schedule(
            window_count=320, start_passes=100, seed=891
        )
        payload = {
            "format": matrix.STATE_FORMAT,
            "status": "completed",
            "update": 8000,
            "history": [{}] * 8000,
            "baseline": {},
            "rngState": {},
            "milestoneResults": {"100": {}},
            "runContract": {
                "id": "parent",
                "contract": {
                    "experimentId": matrix.EXPERIMENT_ID,
                    "variant": {
                        "key": "W-D",
                        "initialization": "pretrained-body",
                        "output_semantic": "direct-instrumental",
                        "reset_output_head": True,
                    },
                    "seed": 891,
                    "passes": 100,
                    "batchSize": 4,
                    "learningRate": 1.0e-4,
                    "trainWindowsPerSong": 32,
                    "holdoutWindowsPerSong": 16,
                    "evalWindowsPerSong": 16,
                    "trainWindowCount": 320,
                    "scheduleSeed": matrix.make_schedule_seed(891),
                    "scheduleSha256": matrix.schedule_sha256(parent_schedule),
                },
            },
        }
        changed = schedule.copy()
        changed[0], changed[1] = changed[1], changed[0]

        with self.assertRaisesRegex(ValueError, "preserve the parent prefix"):
            continuation.validate_parent_payload(
                payload=payload,
                parent_path=Path("parent.pt"),
                seed=891,
                args=args,
                schedule=changed,
                updates_per_pass=80,
            )

    def test_seed_decision_requires_gain_and_safety(self) -> None:
        def milestone(target: float, instrumental: float, holdout_target: float) -> dict:
            return {
                "evaluation": {
                    "calibrationInternal": {
                        "aggregate": {
                            "aggressiveTargetSdrDb": target,
                            "instrumentalSdrDb": instrumental,
                        }
                    },
                    "trainHoldout": {
                        "aggregate": {"aggressiveTargetSdrDb": holdout_target}
                    },
                }
            }

        passed = continuation.make_seed_decision(
            run={"milestones": {"100": milestone(10.0, 9.0, 10.0), "200": milestone(10.5, 8.9, 9.9)}},
            start_passes=100,
            final_passes=200,
        )
        failed = continuation.make_seed_decision(
            run={"milestones": {"100": milestone(10.0, 9.0, 10.0), "200": milestone(10.49, 8.9, 9.9)}},
            start_passes=100,
            final_passes=200,
        )

        self.assertTrue(passed["proceedToRemainingSeeds"])
        self.assertFalse(failed["proceedToRemainingSeeds"])


if __name__ == "__main__":
    unittest.main()
