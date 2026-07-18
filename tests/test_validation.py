import unittest

from src.core.types import TelemetrySnapshot
from src.core.validation import CommandValidationError, validate_commander_output


class CommanderValidationTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = TelemetrySnapshot(
            drone_id="drone-1",
            timestamp=100.0,
            latitude=-35.36,
            longitude=149.16,
            altitude_m=20.0,
            heading_deg=90.0,
            battery_percent=80.0,
        )
        self.command = {
            "command": "SET_POSITION_TARGET_LOCAL_NED",
            "target_system": 1,
            "target_component": 1,
            "x": 10.0,
            "y": 5.0,
            "z": 0.0,
            "reasoning": "Investigate the detected survivor.",
        }

    def test_accepts_safe_command(self):
        validated = validate_commander_output(self.command, self.telemetry, now=103.0)
        self.assertEqual(validated.command, "SET_POSITION_TARGET_LOCAL_NED")
        self.assertEqual(validated.x, 10.0)

    def test_rejects_missing_field(self):
        invalid_command = dict(self.command)
        del invalid_command["reasoning"]
        with self.assertRaises(CommandValidationError):
            validate_commander_output(invalid_command, self.telemetry, now=103.0)

    def test_rejects_unsafe_offset(self):
        invalid_command = dict(self.command, x=101.0)
        with self.assertRaises(CommandValidationError):
            validate_commander_output(invalid_command, self.telemetry, now=103.0)

    def test_rejects_stale_telemetry(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(self.command, self.telemetry, now=106.0)

    def test_rejects_low_battery(self):
        low_battery = TelemetrySnapshot(
            drone_id=self.telemetry.drone_id,
            timestamp=self.telemetry.timestamp,
            latitude=self.telemetry.latitude,
            longitude=self.telemetry.longitude,
            altitude_m=self.telemetry.altitude_m,
            heading_deg=self.telemetry.heading_deg,
            battery_percent=10.0,
        )
        with self.assertRaises(CommandValidationError):
            validate_commander_output(self.command, low_battery, now=103.0)


if __name__ == "__main__":
    unittest.main()
