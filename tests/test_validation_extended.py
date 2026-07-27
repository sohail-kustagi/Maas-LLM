"""Extended validation tests — edge cases not covered by the original test_validation.py."""
import math
import unittest

from src.core.types import TelemetrySnapshot
from src.core.validation import CommandValidationError, validate_commander_output


def _base_command(**overrides) -> dict:
    cmd = {
        "command": "SET_POSITION_TARGET_LOCAL_NED",
        "target_system": 1,
        "target_component": 1,
        "x": 10.0,
        "y": 5.0,
        "z": -2.0,
        "reasoning": "Investigate the fire.",
    }
    cmd.update(overrides)
    return cmd


def _snap(**overrides) -> TelemetrySnapshot:
    base = dict(
        drone_id="drone-1",
        timestamp=100.0,
        latitude=-35.36,
        longitude=149.16,
        altitude_m=20.0,
        heading_deg=90.0,
        battery_percent=80.0,
    )
    base.update(overrides)
    return TelemetrySnapshot(**base)


class ValidationEdgeCaseTests(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # NaN and Inf offsets — must be rejected
    # ------------------------------------------------------------------ #
    def test_rejects_nan_x(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(x=float("nan")), _snap(), now=103.0)

    def test_rejects_inf_y(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(y=float("inf")), _snap(), now=103.0)

    def test_rejects_neg_inf_z(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(z=float("-inf")), _snap(), now=103.0)

    # ------------------------------------------------------------------ #
    # Boundary values for x/y (±100.0 allowed, ±100.01 rejected)
    # ------------------------------------------------------------------ #
    def test_accepts_x_at_max(self):
        cmd = validate_commander_output(_base_command(x=100.0), _snap(), now=103.0)
        self.assertEqual(cmd.x, 100.0)

    def test_accepts_x_at_min(self):
        cmd = validate_commander_output(_base_command(x=-100.0), _snap(), now=103.0)
        self.assertEqual(cmd.x, -100.0)

    def test_rejects_x_just_over_max(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(x=100.01), _snap(), now=103.0)

    def test_rejects_y_just_over_max(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(y=-100.01), _snap(), now=103.0)

    # ------------------------------------------------------------------ #
    # Boundary values for z (min=-50, max=20)
    # ------------------------------------------------------------------ #
    def test_accepts_z_at_min(self):
        cmd = validate_commander_output(_base_command(z=-50.0), _snap(), now=103.0)
        self.assertEqual(cmd.z, -50.0)

    def test_accepts_z_at_max(self):
        cmd = validate_commander_output(_base_command(z=20.0), _snap(), now=103.0)
        self.assertEqual(cmd.z, 20.0)

    def test_rejects_z_below_min(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(z=-50.01), _snap(), now=103.0)

    def test_rejects_z_above_max(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(z=20.01), _snap(), now=103.0)

    # ------------------------------------------------------------------ #
    # target_system / target_component boundaries
    # ------------------------------------------------------------------ #
    def test_rejects_target_system_zero(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(target_system=0), _snap(), now=103.0)

    def test_rejects_target_system_256(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(target_system=256), _snap(), now=103.0)

    def test_accepts_target_system_255(self):
        cmd = validate_commander_output(_base_command(target_system=255), _snap(), now=103.0)
        self.assertEqual(cmd.target_system, 255)

    def test_rejects_target_component_zero(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(target_component=0), _snap(), now=103.0)

    # ------------------------------------------------------------------ #
    # Wrong command name
    # ------------------------------------------------------------------ #
    def test_rejects_wrong_command_name(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(command="DANGEROUS_CMD"), _snap(), now=103.0)

    # ------------------------------------------------------------------ #
    # reasoning field edge cases
    # ------------------------------------------------------------------ #
    def test_rejects_empty_reasoning(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(reasoning=""), _snap(), now=103.0)

    def test_rejects_whitespace_only_reasoning(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(reasoning="   "), _snap(), now=103.0)

    def test_accepts_long_reasoning(self):
        cmd = validate_commander_output(
            _base_command(reasoning="A" * 500), _snap(), now=103.0
        )
        self.assertEqual(len(cmd.reasoning), 500)

    # ------------------------------------------------------------------ #
    # Type coercion / type errors
    # ------------------------------------------------------------------ #
    def test_rejects_boolean_x(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(x=True), _snap(), now=103.0)

    def test_rejects_string_y(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(y="far"), _snap(), now=103.0)

    def test_rejects_target_system_as_float(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(target_system=1.0), _snap(), now=103.0)

    # ------------------------------------------------------------------ #
    # Telemetry age boundary: freshness at exactly 5.0s is OK, 5.001s is not
    # ------------------------------------------------------------------ #
    def test_accepts_telemetry_at_exactly_5s_old(self):
        snap = _snap(timestamp=100.0)
        cmd = validate_commander_output(_base_command(), snap, now=105.0)
        self.assertIsNotNone(cmd)

    def test_rejects_telemetry_just_over_5s_old(self):
        snap = _snap(timestamp=100.0)
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(), snap, now=105.001)

    # ------------------------------------------------------------------ #
    # Battery boundary: exactly 15% is OK, 14.9% is not
    # ------------------------------------------------------------------ #
    def test_accepts_battery_at_15_percent(self):
        snap = _snap(battery_percent=15.0)
        cmd = validate_commander_output(_base_command(), snap, now=103.0)
        self.assertIsNotNone(cmd)

    def test_rejects_battery_just_below_15(self):
        snap = _snap(battery_percent=14.9)
        with self.assertRaises(CommandValidationError):
            validate_commander_output(_base_command(), snap, now=103.0)

    # ------------------------------------------------------------------ #
    # No telemetry — must still validate the command itself
    # ------------------------------------------------------------------ #
    def test_no_telemetry_still_validates_command(self):
        cmd = validate_commander_output(_base_command(), telemetry=None)
        self.assertEqual(cmd.command, "SET_POSITION_TARGET_LOCAL_NED")

    # ------------------------------------------------------------------ #
    # Not a dict / wrong type input
    # ------------------------------------------------------------------ #
    def test_rejects_list_input(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output(["command", "bad"], _snap(), now=103.0)

    def test_rejects_string_input(self):
        with self.assertRaises(CommandValidationError):
            validate_commander_output("not a dict", _snap(), now=103.0)


if __name__ == "__main__":
    unittest.main()
