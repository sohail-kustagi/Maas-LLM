import asyncio
import unittest

from src.core.pipeline import PipelineCoordinator, parse_sandbox_telemetry
from src.core.types import TelemetrySnapshot, VisionEvent


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_flows_through_analyst_and_commander(self):
        telemetry = TelemetrySnapshot(
            drone_id="drone-1",
            timestamp=100.0,
            latitude=-35.36,
            longitude=149.16,
            altitude_m=20.0,
            heading_deg=90.0,
            battery_percent=80.0,
        )
        event = VisionEvent(
            drone_id="drone-1",
            timestamp=101.0,
            anomaly_type="human_survivor",
            confidence=0.92,
        )
        observed = {}

        def analyst(event_value, telemetry_value):
            observed["event"] = event_value
            observed["telemetry"] = telemetry_value
            return "investigate survivor"

        async def commander(context, telemetry_value):
            observed["context"] = context
            return {"command": "SET_POSITION_TARGET_LOCAL_NED"}

        coordinator = PipelineCoordinator(
            analyst=analyst,
            commander=commander,
            telemetry=lambda: telemetry,
        )
        await coordinator.publish(event)
        result = await coordinator.run_once()

        self.assertEqual(result.context, "investigate survivor")
        self.assertEqual(result.command["command"], "SET_POSITION_TARGET_LOCAL_NED")
        self.assertIs(observed["event"], event)
        self.assertIs(observed["telemetry"], telemetry)

    async def test_missing_telemetry_drops_event(self):
        coordinator = PipelineCoordinator(
            analyst=lambda _event, _telemetry: "unused",
            commander=lambda _context, _telemetry: asyncio.sleep(0),
            telemetry=lambda: None,
        )
        await coordinator.publish(
            VisionEvent("drone-1", 100.0, "person", 0.9)
        )
        self.assertIsNone(await coordinator.run_once())

    def test_parses_explicit_sandbox_telemetry(self):
        telemetry = parse_sandbox_telemetry(
            b'{"drone_id":"drone-1","timestamp":100,"lat":1,"lon":2,"alt":3,"heading":4,"battery_percent":75}'
        )
        self.assertEqual(telemetry.drone_id, "drone-1")
        self.assertEqual(telemetry.battery_percent, 75.0)

    def test_rejects_incomplete_sandbox_telemetry(self):
        with self.assertRaises(ValueError):
            parse_sandbox_telemetry(b'{"lat":1}')


if __name__ == "__main__":
    unittest.main()
