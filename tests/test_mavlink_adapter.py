import unittest

from src.core.mavlink_adapter import MAVLinkTelemetryDecoder


class FakeMessage:
    def __init__(self, message_type, **fields):
        self.message_type = message_type
        self.__dict__.update(fields)

    def get_type(self):
        return self.message_type

    def get_srcSystem(self):
        return 7


class FakeParser:
    def __init__(self, messages):
        self.messages = messages

    def parse_buffer(self, _data):
        return self.messages


class MAVLinkAdapterTests(unittest.TestCase):
    def test_common_messages_form_snapshot(self):
        decoder = object.__new__(MAVLinkTelemetryDecoder)
        decoder.values = {}
        decoder.parser = FakeParser(
            [
                FakeMessage("HEARTBEAT"),
                FakeMessage(
                    "GLOBAL_POSITION_INT",
                    lat=-353632610,
                    lon=1491652300,
                    relative_alt=20500,
                    hdg=9000,
                ),
                FakeMessage("SYS_STATUS", battery_remaining=82),
            ]
        )
        snapshot = decoder.feed(b"packet", received_at=100.0)
        self.assertEqual(snapshot.drone_id, "mav-7")
        self.assertAlmostEqual(snapshot.latitude, -35.363261)
        self.assertAlmostEqual(snapshot.longitude, 149.16523)
        self.assertEqual(snapshot.altitude_m, 20.5)
        self.assertEqual(snapshot.heading_deg, 90.0)
        self.assertEqual(snapshot.battery_percent, 82.0)

    def test_snapshot_waits_for_position_and_heartbeat(self):
        decoder = object.__new__(MAVLinkTelemetryDecoder)
        decoder.values = {}
        decoder.parser = FakeParser([FakeMessage("HEARTBEAT")])
        self.assertIsNone(decoder.feed(b"packet", received_at=100.0))


if __name__ == "__main__":
    unittest.main()
