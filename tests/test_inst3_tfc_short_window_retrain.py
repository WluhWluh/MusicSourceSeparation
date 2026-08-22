import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_inst3_tfc_short_window_retrain import (  # noqa: E402
    H25,
    UNIFORM,
    build_input_and_target_specs,
    build_song_plan,
    build_schedule,
    make_contract,
    make_short_model,
    event_output_start,
    uniform_starts,
)


class TfcShortWindowRetrainTest(unittest.TestCase):
    def test_contract_dimensions_match_f24_plan(self) -> None:
        contract = make_contract(24, 4)
        self.assertEqual(contract.input_samples, 23_552)
        self.assertEqual(contract.stride_samples, 15_360)
        self.assertEqual(contract.useful_samples, 15_360)
        self.assertAlmostEqual(contract.output_efficiency, 15 / 23, places=6)

    def test_uniform_starts_cover_tail(self) -> None:
        contract = make_contract(24, 4)
        starts = uniform_starts(contract.stride_samples * 3 + 100, contract)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], contract.stride_samples * 3 + 100 - contract.stride_samples)
        self.assertEqual(len(starts), len(set(starts)))

    def test_event_center_is_clamped_and_relative_interval_valid(self) -> None:
        contract = make_contract(24, 4)
        row = {
            "eventCenterSamples": 10,
            "eventStartSamples": 0,
            "eventEndSamples": 2_205,
        }
        start, event_start, event_end = event_output_start(row, 100_000, contract)
        self.assertEqual(start, 0)
        self.assertGreaterEqual(event_start, 0)
        self.assertGreater(event_end, event_start)

    def test_h25_plan_has_shared_uniform_and_event_records(self) -> None:
        contract = make_contract(24, 4)
        rows = [
            {
                "eventCenterSamples": 30_000,
                "eventStartSamples": 28_898,
                "eventEndSamples": 31_103,
            },
            {
                "eventCenterSamples": 60_000,
                "eventStartSamples": 58_898,
                "eventEndSamples": 61_103,
            },
            {
                "eventCenterSamples": 90_000,
                "eventStartSamples": 88_898,
                "eventEndSamples": 91_103,
            },
            {
                "eventCenterSamples": 120_000,
                "eventStartSamples": 118_898,
                "eventEndSamples": 121_103,
            },
        ]
        plan = build_song_plan(
            slug="song",
            member="train/song.stem.mp4",
            song_samples=200_000,
            event_rows=rows,
            contract=contract,
            train_windows_per_song=8,
            seed=891,
            song_index=0,
        )
        self.assertEqual(len(plan.uniform_indices), 8)
        self.assertEqual(len(plan.event_indices), 4)
        self.assertEqual(len(plan.records), len(set(record.start for record in plan.records)))
        schedule_u = build_schedule({"song": plan}, UNIFORM, 1, 891)
        schedule_h = build_schedule({"song": plan}, H25, 1, 891)
        self.assertEqual(len(schedule_u), 8)
        self.assertEqual(len(schedule_h), 8)

    def test_short_input_and_target_contract_shapes(self) -> None:
        contract = make_contract(24, 4)
        mixture = np.zeros((50_000, 2), dtype=np.float32)
        teacher = np.zeros((contract.stride_samples, 2), dtype=np.float32)
        teacher[:, 0] = 0.25
        record = build_song_plan(
            slug="song",
            member="train/song.stem.mp4",
            song_samples=50_000,
            event_rows=[
                {
                    "eventCenterSamples": 20_000,
                    "eventStartSamples": 18_898,
                    "eventEndSamples": 21_103,
                }
            ] * 4,
            contract=contract,
            train_windows_per_song=8,
            seed=891,
            song_index=0,
        ).records[0]
        input_spec, target_spec = build_input_and_target_specs(
            mixture, teacher, record, contract
        )
        self.assertEqual(input_spec.shape, (4, 1_025, 24))
        self.assertEqual(target_spec.shape, input_spec.shape)

    def test_h50_state_loads_into_f24_model(self) -> None:
        checkpoint = (
            ROOT
            / "data"
            / "musdb18-inst3-vr-hard-sampling-h50"
            / "runs"
            / "V-R-H50"
            / "step-8000.pt"
        )
        if not checkpoint.is_file():
            self.skipTest("local H50 checkpoint unavailable")
        model, _ = make_short_model(checkpoint, 24, torch.device("cpu"))
        output = model(torch.zeros((1, 4, 1_025, 24)))
        self.assertEqual(tuple(output.shape), (1, 4, 1_025, 24))


if __name__ == "__main__":
    unittest.main()
