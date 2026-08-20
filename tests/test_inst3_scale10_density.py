from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import run_inst3_scale10_density as density  # noqa: E402


class Inst3Scale10DensityTest(unittest.TestCase):
    def test_default_milestones_match_update_density_contract(self) -> None:
        args = density.parse_args([])

        milestones = density.validate_args(args)

        self.assertEqual(milestones, (0, 8192, 20480, 40960))
        self.assertEqual(args.steps, 40960)

    def test_resume_selection_prefers_highest_persisted_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir(parents=True)
            rolling = runs / "training-state.pt"
            milestone = runs / "alpha-1.00-step-8192.pt"
            torch.save({"step": 4096}, rolling)
            torch.save({"step": 8192}, milestone)

            selected = density.find_resume_path(root, (0, 8192, 20480, 40960))

            self.assertEqual(selected, milestone)

    def test_resume_contract_rejects_a_different_run(self) -> None:
        payload = {
            "format": density.STATE_FORMAT,
            "runContract": {"id": "old"},
            "alpha": density.ALPHA,
        }

        with self.assertRaisesRegex(ValueError, "contract does not match"):
            density.validate_resume_payload(payload, {"id": "new"})

    def test_local_metrics_capture_short_retained_teacher_residual(self) -> None:
        source = np.ones((1000, 2), dtype=np.float32)
        teacher = np.zeros_like(source)
        candidate = np.full_like(source, 0.5)

        metrics = density.local_teacher_error_metrics(
            source,
            teacher,
            candidate,
            sample_rate=1000,
        )

        expected_db = -6.020599913279624
        self.assertAlmostEqual(metrics["teacherRemovalResidualSdrDb"], -expected_db)
        self.assertAlmostEqual(metrics["errorRmsDbfs"], expected_db)
        blocks = metrics["blockMetrics"]["100"]
        self.assertEqual(blocks["blockCount"], 10)
        self.assertEqual(blocks["activeBlockCount"], 10)
        self.assertAlmostEqual(blocks["retainedTeacherResidualP95Dbfs"], expected_db)
        self.assertAlmostEqual(blocks["retainedTeacherResidualMaxDbfs"], expected_db)

    def test_local_metrics_require_aligned_audio(self) -> None:
        source = np.zeros((100, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "shapes do not match"):
            density.local_teacher_error_metrics(
                source,
                source[:-1],
                source,
                sample_rate=1000,
            )


if __name__ == "__main__":
    unittest.main()
