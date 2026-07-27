"""Tests for SITLInjector and the SITL integration path."""
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.sitl_injector import DEFAULT_SCENARIO, SITLInjector
from src.core.types import VisionEvent


# ──────────────────────────────────────────────────────────────────────────────
# SITLInjector unit tests
# ──────────────────────────────────────────────────────────────────────────────

class SITLInjectorTests(unittest.IsolatedAsyncioTestCase):

    def _fast_scenario(self) -> list:
        """A very fast scenario (1ms delays) for unit testing."""
        return [
            (0.001, "human_survivor", 0.91),
            (0.002, "fire",           0.88),
            (0.003, None,             0.0),   # end sentinel
        ]

    async def test_injects_correct_number_of_events(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, drone_id="test-drone", scenario=self._fast_scenario())
        await inj.run()
        self.assertEqual(inj.injected_count, 2)

    async def test_events_have_correct_anomaly_types(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, scenario=self._fast_scenario())
        await inj.run()
        types = [q.get_nowait().anomaly_type for _ in range(2)]
        self.assertEqual(types[0], "human_survivor")
        self.assertEqual(types[1], "fire")

    async def test_events_are_vision_event_instances(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, scenario=self._fast_scenario())
        await inj.run()
        while not q.empty():
            event = q.get_nowait()
            self.assertIsInstance(event, VisionEvent)

    async def test_drone_id_is_propagated(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, drone_id="sitl-42", scenario=self._fast_scenario())
        await inj.run()
        event = q.get_nowait()
        self.assertEqual(event.drone_id, "sitl-42")

    async def test_confidence_is_correct(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, scenario=self._fast_scenario())
        await inj.run()
        event = q.get_nowait()
        self.assertAlmostEqual(event.confidence, 0.91, places=2)

    async def test_event_timestamp_is_recent(self):
        before = time.time()
        q = asyncio.Queue()
        inj = SITLInjector(q, scenario=self._fast_scenario())
        await inj.run()
        after = time.time()
        event = q.get_nowait()
        self.assertGreaterEqual(event.timestamp, before)
        self.assertLessEqual(event.timestamp, after + 0.1)

    async def test_injected_count_starts_at_zero(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, scenario=self._fast_scenario())
        self.assertEqual(inj.injected_count, 0)

    async def test_empty_scenario_injects_nothing(self):
        q = asyncio.Queue()
        inj = SITLInjector(q, scenario=[(0.001, None, 0.0)])  # only sentinel
        await inj.run()
        self.assertEqual(inj.injected_count, 0)
        self.assertTrue(q.empty())

    async def test_default_scenario_has_four_events(self):
        """DEFAULT_SCENARIO must have exactly 4 real events + 1 sentinel."""
        real_events = [s for s in DEFAULT_SCENARIO if s[1] is not None]
        self.assertEqual(len(real_events), 4)

    async def test_run_loop_repeats(self):
        """run_loop(repeat=True) should re-inject after completing one pass."""
        q = asyncio.Queue()
        scenario = [(0.001, "fire", 0.9), (0.002, None, 0.0)]
        inj = SITLInjector(q, scenario=scenario)

        # Run for 2 iterations then cancel
        async def run_two():
            count = 0
            await inj.run()
            count += 1
            inj._injected.clear()
            await inj.run()
            count += 1
            return count

        count = await run_two()
        self.assertEqual(count, 2)
        # Should have injected 1 event per run × 2 runs = 2 total in queue
        self.assertEqual(q.qsize(), 2)


# ──────────────────────────────────────────────────────────────────────────────
# SITL telemetry decoder integration
# ──────────────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self, msg_type, **fields):
        self._type = msg_type
        self.__dict__.update(fields)

    def get_type(self):
        return self._type

    def get_srcSystem(self):
        return getattr(self, "_src", 1)


class FakeParser:
    def __init__(self, messages):
        self._messages = messages

    def parse_buffer(self, _):
        return self._messages


def _make_sitl_decoder_with_messages(messages):
    from src.core.mavlink_adapter import MAVLinkTelemetryDecoder
    decoder = object.__new__(MAVLinkTelemetryDecoder)
    decoder.values = {}
    decoder.parser = FakeParser(messages)
    return decoder


class SITLMAVLinkIntegrationTests(unittest.TestCase):

    def test_heartbeat_plus_position_produces_snapshot(self):
        hb = FakeMessage("HEARTBEAT", base_mode=0x80, custom_mode=4)
        hb._src = 5
        pos = FakeMessage("GLOBAL_POSITION_INT",
                           lat=-353632610, lon=1491652300,
                           relative_alt=25000, hdg=9000)
        decoder = _make_sitl_decoder_with_messages([hb, pos])
        snap = decoder.feed(b"x", received_at=1000.0)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.drone_id, "mav-5")
        self.assertEqual(snap.flight_mode, "GUIDED")
        self.assertTrue(snap.is_armed)
        self.assertAlmostEqual(snap.altitude_m, 25.0, places=1)

    def test_full_sitl_message_burst(self):
        """Simulate a typical SITL telemetry burst."""
        hb = FakeMessage("HEARTBEAT", base_mode=0x80, custom_mode=3)
        hb._src = 1
        msgs = [
            hb,
            FakeMessage("GLOBAL_POSITION_INT",
                        lat=-353632610, lon=1491652300,
                        relative_alt=30000, hdg=27000),
            FakeMessage("SYS_STATUS", battery_remaining=75),
            FakeMessage("VFR_HUD", groundspeed=10.0, climb=0.5),
            FakeMessage("GPS_RAW_INT", fix_type=3, satellites_visible=12),
        ]
        decoder = _make_sitl_decoder_with_messages(msgs)
        snap = decoder.feed(b"x", received_at=5000.0)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.battery_percent, 75.0)
        self.assertAlmostEqual(snap.ground_speed_mps, 10.0)
        self.assertEqual(snap.gps_fix_type, 3)
        self.assertEqual(snap.satellites_visible, 12)
        self.assertEqual(snap.flight_mode, "AUTO")


# ──────────────────────────────────────────────────────────────────────────────
# main.py SITL mode smoke test
# ──────────────────────────────────────────────────────────────────────────────

class MainSITLModeTests(unittest.IsolatedAsyncioTestCase):

    async def test_sitl_pipeline_runs_and_shuts_down(self):
        """run_sitl_pipeline must start, run the injector, and exit cleanly."""
        import importlib
        import sys
        import argparse
        from unittest.mock import MagicMock, AsyncMock

        # Ensure a clean import of src.main with all heavy deps mocked
        mock_llama = MagicMock()
        mock_llama.Llama.return_value = MagicMock()
        mock_llama.LlamaGrammar.from_string.return_value = MagicMock()

        # Clear cached modules so patches take effect
        for key in list(sys.modules.keys()):
            if "nodes.commander" in key or "src.nodes.commander" in key:
                del sys.modules[key]

        mock_commander = MagicMock()
        mock_commander.generate_mavlink_command = AsyncMock(return_value=None)

        mock_inj = MagicMock()
        mock_inj.run = AsyncMock(return_value=None)

        mock_wp = MagicMock()
        mock_wp.get_weather = AsyncMock(return_value=None)

        with patch.dict(sys.modules, {"llama_cpp": mock_llama}):
            import src.main as main_mod
            importlib.reload(main_mod)

            with (
                patch("src.nodes.commander.CommanderNode", return_value=mock_commander),
                patch("src.core.sitl_injector.SITLInjector", return_value=mock_inj),
                patch("src.main.WeatherProvider", return_value=mock_wp),
                patch("asyncio.sleep", AsyncMock(return_value=None)),
            ):
                args = argparse.Namespace(mode="sitl")
                try:
                    await main_mod.run_sitl_pipeline(args)
                except Exception as exc:
                    self.fail(f"run_sitl_pipeline raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
