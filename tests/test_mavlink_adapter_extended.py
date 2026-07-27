"""Extended MAVLink adapter tests — covers all new message types and new snapshot fields."""
import unittest

from src.core.mavlink_adapter import MAVLinkTelemetryDecoder


class FakeMessage:
    def __init__(self, message_type, **fields):
        self.message_type = message_type
        self.__dict__.update(fields)

    def get_type(self):
        return self.message_type

    def get_srcSystem(self):
        return getattr(self, "_src_system", 1)


class FakeParser:
    def __init__(self, messages):
        self.messages = messages

    def parse_buffer(self, _data):
        return self.messages


def _decoder(messages) -> MAVLinkTelemetryDecoder:
    decoder = object.__new__(MAVLinkTelemetryDecoder)
    decoder.values = {}
    decoder.parser = FakeParser(messages)
    return decoder


def _base_messages(src_system=7, base_mode=0, custom_mode=4) -> list:
    """Minimum messages required to produce a snapshot."""
    hb = FakeMessage("HEARTBEAT", base_mode=base_mode, custom_mode=custom_mode)
    hb._src_system = src_system
    return [
        hb,
        FakeMessage(
            "GLOBAL_POSITION_INT",
            lat=-353632610, lon=1491652300,
            relative_alt=20500, hdg=9000,
        ),
    ]


class MAVLinkExtendedTests(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # HEARTBEAT: armed state
    # ------------------------------------------------------------------ #
    def test_unarmed_drone_base_mode_zero(self):
        decoder = _decoder(_base_messages(base_mode=0))
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertIsNotNone(snap)
        self.assertFalse(snap.is_armed)

    def test_armed_drone_bit7_set(self):
        # MAV_MODE_FLAG_SAFETY_ARMED = 0x80
        decoder = _decoder(_base_messages(base_mode=0x80))
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertTrue(snap.is_armed)

    def test_armed_bit7_set_plus_other_bits(self):
        decoder = _decoder(_base_messages(base_mode=0x81))
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertTrue(snap.is_armed)

    # ------------------------------------------------------------------ #
    # HEARTBEAT: flight mode
    # ------------------------------------------------------------------ #
    def test_guided_flight_mode_decoded(self):
        decoder = _decoder(_base_messages(custom_mode=4))
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertEqual(snap.flight_mode, "GUIDED")

    def test_rtl_flight_mode_decoded(self):
        decoder = _decoder(_base_messages(custom_mode=6))
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertEqual(snap.flight_mode, "RTL")

    def test_unknown_flight_mode_gets_mode_prefix(self):
        decoder = _decoder(_base_messages(custom_mode=99))
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertEqual(snap.flight_mode, "MODE_99")

    # ------------------------------------------------------------------ #
    # VFR_HUD: ground speed and climb rate
    # ------------------------------------------------------------------ #
    def test_vfr_hud_ground_speed(self):
        msgs = _base_messages() + [FakeMessage("VFR_HUD", groundspeed=12.5, climb=0.0)]
        decoder = _decoder(msgs)
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertAlmostEqual(snap.ground_speed_mps, 12.5)

    def test_vfr_hud_climb_rate(self):
        msgs = _base_messages() + [FakeMessage("VFR_HUD", groundspeed=0.0, climb=3.2)]
        decoder = _decoder(msgs)
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertAlmostEqual(snap.climb_rate_mps, 3.2)

    def test_snapshot_without_vfr_hud_has_none_speed(self):
        decoder = _decoder(_base_messages())
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertIsNone(snap.ground_speed_mps)
        self.assertIsNone(snap.climb_rate_mps)

    # ------------------------------------------------------------------ #
    # GPS_RAW_INT: fix type and satellites
    # ------------------------------------------------------------------ #
    def test_gps_raw_int_fix_type(self):
        msgs = _base_messages() + [FakeMessage("GPS_RAW_INT", fix_type=3, satellites_visible=14)]
        decoder = _decoder(msgs)
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertEqual(snap.gps_fix_type, 3)
        self.assertEqual(snap.satellites_visible, 14)

    def test_snapshot_without_gps_raw_has_none(self):
        decoder = _decoder(_base_messages())
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertIsNone(snap.gps_fix_type)
        self.assertIsNone(snap.satellites_visible)

    # ------------------------------------------------------------------ #
    # HOME_POSITION
    # ------------------------------------------------------------------ #
    def test_home_position_decoded(self):
        msgs = _base_messages() + [
            FakeMessage(
                "HOME_POSITION",
                latitude=-353632610,   # -35.363261 in 1e7 units
                longitude=1491652300,  # 149.16523 in 1e7 units
                altitude=50000,        # 50.0 m in mm
            )
        ]
        decoder = _decoder(msgs)
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertAlmostEqual(snap.home_latitude, -35.363261, places=5)
        self.assertAlmostEqual(snap.home_longitude, 149.16523, places=4)
        self.assertAlmostEqual(snap.home_altitude_m, 50.0, places=1)

    def test_snapshot_without_home_has_none(self):
        decoder = _decoder(_base_messages())
        snap = decoder.feed(b"x", received_at=100.0)
        self.assertIsNone(snap.home_latitude)
        self.assertIsNone(snap.home_longitude)
        self.assertIsNone(snap.home_altitude_m)

    # ------------------------------------------------------------------ #
    # All-messages integration test
    # ------------------------------------------------------------------ #
    def test_full_message_set_produces_complete_snapshot(self):
        hb = FakeMessage("HEARTBEAT", base_mode=0x80, custom_mode=4)
        hb._src_system = 3
        msgs = [
            hb,
            FakeMessage("GLOBAL_POSITION_INT", lat=-353632610, lon=1491652300, relative_alt=25000, hdg=27000),
            FakeMessage("SYS_STATUS", battery_remaining=72),
            FakeMessage("VFR_HUD", groundspeed=8.5, climb=1.5),
            FakeMessage("GPS_RAW_INT", fix_type=3, satellites_visible=10),
            FakeMessage("HOME_POSITION", latitude=-353640000, longitude=1491600000, altitude=0),
        ]
        decoder = _decoder(msgs)
        snap = decoder.feed(b"x", received_at=200.0)
        self.assertEqual(snap.drone_id, "mav-3")
        self.assertEqual(snap.battery_percent, 72.0)
        self.assertTrue(snap.is_armed)
        self.assertEqual(snap.flight_mode, "GUIDED")
        self.assertAlmostEqual(snap.ground_speed_mps, 8.5)
        self.assertAlmostEqual(snap.climb_rate_mps, 1.5)
        self.assertEqual(snap.gps_fix_type, 3)
        self.assertEqual(snap.satellites_visible, 10)
        self.assertIsNotNone(snap.home_latitude)


if __name__ == "__main__":
    unittest.main()
