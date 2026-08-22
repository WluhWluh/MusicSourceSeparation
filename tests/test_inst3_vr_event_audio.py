import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_inst3_vr_event_audio import (  # noqa: E402
    charcoal_loss,
    build_event_weights,
    torch_packed_istft,
)
import run_inst3_vr_event_audio as audio_runner  # noqa: E402
import validate_tfc_tdf_default_audio as dsp  # noqa: E402
import run_inst3_distill_pilot as pilot  # noqa: E402


class Inst3VrEventAudioTest(unittest.TestCase):
    def test_core_and_guard_weights_have_expected_support(self) -> None:
        core, guard = build_event_weights(
            starts=[100], ends=[200], guard_ms=25.0, useful_samples=1_000, sample_rate=1_000
        )
        self.assertEqual(core.dtype, np.dtype(bool))
        self.assertTrue(np.all(core[0, 100:200]))
        self.assertTrue(np.all(guard[0, 100:200] == 1.0))
        self.assertAlmostEqual(float(guard[0, 75]), 0.0, places=6)
        self.assertGreater(float(guard[0, 99]), 0.9)
        self.assertGreater(float(guard[0, 200]), 0.9)
        self.assertAlmostEqual(float(guard[0, 225]), 0.0, places=6)

    def test_event_weights_reject_out_of_range_interval(self) -> None:
        with self.assertRaises(ValueError):
            build_event_weights(
                starts=[0], ends=[1_001], guard_ms=25.0, useful_samples=1_000
            )

    def test_charbonnier_zero_and_weighted_values(self) -> None:
        prediction = torch.zeros((1, 4, 2), dtype=torch.float32)
        target = torch.ones_like(prediction)
        weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        value = charcoal_loss(prediction, target, weights, epsilon=1.0e-6)
        self.assertAlmostEqual(float(value), 1.0, places=4)
        self.assertGreater(float(charcoal_loss(prediction, target, torch.ones_like(weights))), 0.9)

    def test_charbonnier_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            charcoal_loss(
                torch.zeros((1, 4, 2)),
                torch.zeros((1, 3, 2)),
                torch.ones((1, 4)),
            )

    def test_torch_istft_matches_existing_numpy_contract(self) -> None:
        rng = np.random.default_rng(891)
        packed = rng.standard_normal(
            (1, 4, pilot.DEFAULT_CONFIG.frequency_bins, pilot.DEFAULT_CONFIG.num_frames),
            dtype=np.float32,
        )
        device = torch.device("cpu")
        window = torch.hann_window(pilot.DEFAULT_CONFIG.n_fft, periodic=True)
        with torch.inference_mode():
            torch_audio = torch_packed_istft(torch.from_numpy(packed), window).numpy()
        numpy_audio = dsp.istft_centered(packed)
        self.assertLess(float(np.max(np.abs(torch_audio - numpy_audio))), 1.0e-5)

    def test_audio_loss_backward_is_finite(self) -> None:
        prediction = torch.zeros((1, 32, 2), dtype=torch.float32, requires_grad=True)
        target = torch.ones_like(prediction)
        weights = torch.ones((1, 32), dtype=torch.float32)
        loss = charcoal_loss(prediction, target, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
