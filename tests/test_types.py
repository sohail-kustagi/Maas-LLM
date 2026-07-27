"""Tests for TelemetrySnapshot, VisionEvent, and CommanderCommand in core/types.py."""
import time
import unittest

from src.core.types import CommanderCommand, TelemetrySnapshot, VisionEvent


def _make_snapshot(**overrides) -> TelemetrySnapshot:
    base = dict(
        drone_id="drone-1",
        timestamp=1_000_000.0,
        latitude=-35.363261,
        longitude=149.165230,
        altitude_m=25.0,
        heading_deg=270.0,
    )
    base.update(overrides)
    return TelemetrySnapshot(**base)


class TelemetrySnapshotTests(unittest.TestCase):
    def test_required_fields_only(self):
        snap = _make_snapshot()
        self.assertEqual(snap.drone_id, "drone-1")
        self.assertIsNone(snap.battery_percent)
        self.assertIsNone(snap.ground_speed_mps)
        self.assertIsNone(snap.climb_rate_mps)
        self.assertIsNone(snap.gps_fix_type)
        self.assertIsNone(snap.satellites_visible)
        self.assertIsNone(snap.flight_mode)
        self.assertIsNone(snap.is_armed)
        self.assertIsNone(snap.failsafe_state)
        self.assertIsNone(snap.home_latitude)
        self.assertIsNone(snap.home_longitude)
        self.assertIsNone(snap.home_altitude_m)

    def test_optional_fields_set(self):
        snap = _make_snapshot(
            battery_percent=85.0,
            ground_speed_mps=10.5,
            climb_rate_mps=2.0,
            gps_fix_type=3,
            satellites_visible=12,
            flight_mode="GUIDED",
            is_armed=True,
            failsafe_state="NONE",
            home_latitude=-35.36,
            home_longitude=149.16,
            home_altitude_m=0.0,
        )
        self.assertEqual(snap.battery_percent, 85.0)
        self.assertEqual(snap.ground_speed_mps, 10.5)
        self.assertEqual(snap.climb_rate_mps, 2.0)
        self.assertEqual(snap.gps_fix_type, 3)
        self.assertEqual(snap.satellites_visible, 12)
        self.assertEqual(snap.flight_mode, "GUIDED")
        self.assertTrue(snap.is_armed)
        self.assertEqual(snap.failsafe_state, "NONE")
        self.assertAlmostEqual(snap.home_latitude, -35.36)

    def test_age_seconds_with_explicit_now(self):
        snap = _make_snapshot(timestamp=1_000_000.0)
        self.assertAlmostEqual(snap.age_seconds(now=1_000_005.0), 5.0)

    def test_age_seconds_cannot_be_negative(self):
        snap = _make_snapshot(timestamp=1_000_010.0)
        # now is earlier than timestamp — should clamp to 0
        self.assertEqual(snap.age_seconds(now=1_000_000.0), 0.0)

    def test_age_seconds_default_uses_wall_clock(self):
        snap = _make_snapshot(timestamp=time.time())
        self.assertGreaterEqual(snap.age_seconds(), 0.0)

    def test_snapshot_is_frozen(self):
        snap = _make_snapshot()
        with self.assertRaises((AttributeError, TypeError)):
            snap.drone_id = "hacked"  # type: ignore[misc]


class VisionEventTests(unittest.TestCase):
    def test_vision_event_fields(self):
        event = VisionEvent(
            drone_id="drone-2",
            timestamp=1_000_001.0,
            anomaly_type="fire",
            confidence=0.95,
        )
        self.assertEqual(event.drone_id, "drone-2")
        self.assertEqual(event.anomaly_type, "fire")
        self.assertAlmostEqual(event.confidence, 0.95)
        self.assertIsNone(event.frame_ref)

    def test_vision_event_with_frame_ref(self):
        event = VisionEvent("d", 0.0, "flood", 0.8, frame_ref="frame_042.jpg")
        self.assertEqual(event.frame_ref, "frame_042.jpg")


class CommanderCommandTests(unittest.TestCase):
    def test_as_dict_round_trip(self):
        cmd = CommanderCommand(
            command="SET_POSITION_TARGET_LOCAL_NED",
            target_system=1,
            target_component=1,
            x=10.0,
            y=-5.0,
            z=-2.0,
            reasoning="Investigate fire to the east.",
        )
        d = cmd.as_dict()
        self.assertEqual(d["command"], "SET_POSITION_TARGET_LOCAL_NED")
        self.assertEqual(d["x"], 10.0)
        self.assertEqual(d["y"], -5.0)
        self.assertIn("reasoning", d)


if __name__ == "__main__":
    unittest.main()
