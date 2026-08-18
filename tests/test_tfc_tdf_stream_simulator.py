import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from tfc_tdf_stream_simulator import (  # noqa: E402
    SimulationCommand,
    TfcTdfStreamSimulator,
    TfcTdfWindowContract,
)


class TfcTdfStreamSimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = TfcTdfWindowContract(playback_block_samples=1024)
        sample_count = self.contract.useful_samples * 4 + 4096
        values = np.arange(sample_count * 2, dtype=np.float32).reshape(sample_count, 2)
        self.source = values / values.max()

    def run_simulator(self, **kwargs):
        commands = kwargs.pop("commands", ())
        simulator = TfcTdfStreamSimulator(
            self.source,
            contract=self.contract,
            ring_capacity_windows=kwargs.pop("ring_capacity_windows", 3),
            **kwargs,
        )
        result = simulator.run(commands)
        result.validate()
        return result

    def test_fast_producer_starts_dry_then_switches_to_wet(self) -> None:
        result = self.run_simulator(read_ahead_rate=20.0, inference_latency_ms=50.0)

        self.assertGreater(result.stats["dryBlockCount"], 0)
        self.assertGreater(result.stats["wetBlockCount"], 0)
        self.assertGreaterEqual(result.stats["modeSwitchCount"], 2)
        self.assertIn("mode=wet", result.trace_text())
        self.assertEqual(result.stats["outputBlockCount"], len(result.output_blocks))

    def test_slow_producer_falls_back_to_dry_without_gaps(self) -> None:
        result = self.run_simulator(read_ahead_rate=1.0, inference_latency_ms=500.0)

        self.assertEqual(result.stats["wetBlockCount"], 0)
        self.assertEqual(result.stats["dryBlockCount"], result.stats["outputBlockCount"])
        self.assertEqual(result.stats["outputBlockCount"], self.expected_block_count())
        rendered = np.concatenate([block.samples for block in result.output_blocks], axis=0)
        np.testing.assert_array_equal(rendered, self.source)

    def test_late_window_is_not_allowed_to_replace_emitted_dry(self) -> None:
        result = self.run_simulator(read_ahead_rate=1.0, inference_latency_ms=0.0)

        self.assertGreater(result.stats["lateWindowCount"], 0)
        self.assertEqual(result.stats["wetBlockCount"], 0)
        rendered = np.concatenate([block.samples for block in result.output_blocks], axis=0)
        np.testing.assert_array_equal(rendered, self.source)

    def test_seek_discards_old_epoch_and_rebuilds_from_new_position(self) -> None:
        target = self.contract.useful_samples * 2 + 2048
        commands = [SimulationCommand(at_wall_ms=100.0, kind="seek", value=target)]
        result = self.run_simulator(
            read_ahead_rate=20.0,
            inference_latency_ms=1_000.0,
            commands=commands,
        )

        self.assertGreaterEqual(result.stats["droppedEpochTaskCount"], 1)
        self.assertGreaterEqual(result.stats["wetBlockCount"], 1)
        self.assertEqual(result.output_blocks[0].segment, 0)
        self.assertTrue(any(block.segment == 1 for block in result.output_blocks))
        seek_events = [event for event in result.events if event.event == "seek"]
        self.assertEqual(len(seek_events), 1)
        self.assertEqual(seek_events[0].position_sample, target)

    def test_model_switch_drops_old_results_and_uses_new_model(self) -> None:
        commands = [SimulationCommand(at_wall_ms=100.0, kind="model", value="tfc-tdf-next")]
        result = self.run_simulator(
            read_ahead_rate=20.0,
            inference_latency_ms=1_000.0,
            commands=commands,
        )

        self.assertGreaterEqual(result.stats["droppedEpochTaskCount"], 1)
        next_model_wet = [
            block
            for block in result.output_blocks
            if block.mode == "wet" and block.model_id == "tfc-tdf-next"
        ]
        self.assertGreater(len(next_model_wet), 0)
        for block in next_model_wet[:3]:
            expected = self.source[block.position_sample : block.position_sample + block.samples.shape[0]] * 0.20
            np.testing.assert_allclose(block.samples, expected)

    def test_wet_read_spans_adjacent_model_windows(self) -> None:
        boundary = self.contract.useful_samples
        contract = TfcTdfWindowContract(playback_block_samples=1_000)
        result = TfcTdfStreamSimulator(
            self.source,
            contract=contract,
            read_ahead_rate=100.0,
            inference_latency_ms=0.0,
        ).run()
        result.validate()

        crossing = [
            block
            for block in result.output_blocks
            if block.position_sample < boundary
            < block.position_sample + block.samples.shape[0]
        ]
        self.assertEqual(len(crossing), 1)
        self.assertEqual(crossing[0].mode, "wet")

    def test_pause_and_resume_do_not_create_output_gap(self) -> None:
        commands = [
            SimulationCommand(at_wall_ms=500.0, kind="pause"),
            SimulationCommand(at_wall_ms=2000.0, kind="resume"),
        ]
        result = self.run_simulator(
            read_ahead_rate=20.0,
            inference_latency_ms=50.0,
            commands=commands,
        )

        self.assertTrue(any(event.event == "pause" for event in result.events))
        self.assertTrue(any(event.event == "resume" for event in result.events))
        self.assertEqual(result.stats["outputBlockCount"], self.expected_block_count())
        result.validate()

    def test_disable_and_enable_rebuilds_the_stream_epoch(self) -> None:
        commands = [
            SimulationCommand(at_wall_ms=500.0, kind="disable"),
            SimulationCommand(at_wall_ms=1500.0, kind="enable"),
        ]
        result = self.run_simulator(
            read_ahead_rate=20.0,
            inference_latency_ms=50.0,
            commands=commands,
        )

        self.assertTrue(any(event.event == "disable" for event in result.events))
        self.assertTrue(any(event.event == "enable" for event in result.events))
        self.assertGreaterEqual(result.stats["droppedEpochTaskCount"], 1)
        self.assertTrue(
            any(
                block.mode == "dry" and block.reason == "disabled"
                for block in result.output_blocks
            )
        )
        self.assertTrue(
            any(
                block.mode == "wet" and block.epoch >= 2
                for block in result.output_blocks
            )
        )

    def test_ring_memory_is_bounded_and_played_blocks_are_released(self) -> None:
        result = self.run_simulator(
            read_ahead_rate=100.0,
            inference_latency_ms=0.0,
            ring_capacity_windows=2,
        )

        capacity = self.contract.useful_samples * 2
        self.assertLessEqual(result.stats["dryRing"]["maxSampleCount"], capacity)
        self.assertLessEqual(result.stats["wetRing"]["maxSampleCount"], capacity)
        self.assertLessEqual(result.stats["dryRing"]["maxBlockCount"], 2)
        self.assertLessEqual(result.stats["wetRing"]["maxBlockCount"], 3)
        self.assertLessEqual(result.stats["maxPendingTaskCount"], 3)

    def expected_block_count(self) -> int:
        return int(np.ceil(self.source.shape[0] / self.contract.playback_block_samples))


if __name__ == "__main__":
    unittest.main()
