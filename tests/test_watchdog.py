"""Tests for WatchdogNode — without a real camera or YOLO model.

Uses sys.modules patching to mock cv2 and ultralytics before import.
"""
import asyncio
import sys
import time
import unittest
from unittest.mock import MagicMock, patch


def _setup_mocks():
    """Inject fake cv2 and ultralytics into sys.modules."""
    mock_cv2 = MagicMock()
    mock_ultralytics = MagicMock()

    mods = {
        "cv2": mock_cv2,
        "ultralytics": mock_ultralytics,
        "ultralytics.YOLO": mock_ultralytics.YOLO,
    }
    return mock_cv2, mock_ultralytics, mods


def _import_watchdog(mock_cv2, mock_ultralytics_mod):
    """Import WatchdogNode with cv2 and ultralytics mocked."""
    # Ensure a fresh import
    for key in list(sys.modules.keys()):
        if "watchdog" in key:
            del sys.modules[key]

    with patch.dict(sys.modules, {
        "cv2": mock_cv2,
        "ultralytics": mock_ultralytics_mod,
    }):
        from src.nodes.watchdog import WatchdogNode
        return WatchdogNode


def _make_mock_cap(is_open=True):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = is_open
    return mock_cap


class WatchdogInitTests(unittest.TestCase):

    def _get_class(self):
        mock_cv2, mock_ultra, _ = _setup_mocks()
        return _import_watchdog(mock_cv2, mock_ultra), mock_ultra

    def test_configurable_parameters(self):
        WatchdogNode, mock_ultra = self._get_class()
        with patch.dict(sys.modules, {"cv2": MagicMock(), "ultralytics": mock_ultra}):
            node = WatchdogNode(
                model_path="custom.pt",
                event_queue=None,
                sample_interval=0.5,
                confidence_threshold=0.75,
            )
        self.assertEqual(node.sample_interval, 0.5)
        self.assertEqual(node.confidence_threshold, 0.75)

    def test_cap_is_none_at_init(self):
        WatchdogNode, mock_ultra = self._get_class()
        with patch.dict(sys.modules, {"cv2": MagicMock(), "ultralytics": mock_ultra}):
            node = WatchdogNode()
        self.assertIsNone(node.cap)

    def test_default_confidence_threshold(self):
        WatchdogNode, mock_ultra = self._get_class()
        with patch.dict(sys.modules, {"cv2": MagicMock(), "ultralytics": mock_ultra}):
            node = WatchdogNode()
        self.assertEqual(node.confidence_threshold, 0.6)


class WatchdogCameraTests(unittest.TestCase):

    def _node(self):
        mock_cv2, mock_ultra, _ = _setup_mocks()
        WatchdogNode = _import_watchdog(mock_cv2, mock_ultra)
        node = WatchdogNode()
        node._mock_cv2 = mock_cv2
        return node, mock_cv2

    def test_start_camera_success(self):
        node, mock_cv2 = self._node()
        mock_cv2.VideoCapture.return_value = _make_mock_cap(is_open=True)
        result = node.start_camera(0)
        self.assertTrue(result)

    def test_start_camera_failure_returns_false(self):
        node, mock_cv2 = self._node()
        mock_cv2.VideoCapture.return_value = _make_mock_cap(is_open=False)
        result = node.start_camera(0)
        self.assertFalse(result)

    def test_stop_releases_cap(self):
        node, _ = self._node()
        mock_cap = _make_mock_cap()
        node.cap = mock_cap
        node.stop()
        mock_cap.release.assert_called_once()

    def test_stop_without_cap_does_not_raise(self):
        node, _ = self._node()
        node.cap = None
        node.stop()  # must not raise


class WatchdogVisionLoopTests(unittest.IsolatedAsyncioTestCase):

    def _node(self, event_queue=None, confidence_threshold=0.6, sample_interval=0.0):
        mock_cv2, mock_ultra, _ = _setup_mocks()
        WatchdogNode = _import_watchdog(mock_cv2, mock_ultra)
        node = WatchdogNode(
            event_queue=event_queue,
            confidence_threshold=confidence_threshold,
            sample_interval=sample_interval,
        )
        node._mock_cv2 = mock_cv2
        return node

    async def test_vision_loop_exits_if_camera_not_initialized(self):
        node = self._node()
        node.cap = None
        await asyncio.wait_for(node.run_vision_loop(), timeout=1.0)

    async def test_high_confidence_person_emits_event(self):
        event_queue = asyncio.Queue()
        node = self._node(event_queue=event_queue, sample_interval=0.0)
        node.last_event_at = 0.0

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        call_count = [0]
        def read_side():
            call_count[0] += 1
            if call_count[0] == 1:
                return True, MagicMock()
            return False, None
        mock_cap.read.side_effect = read_side
        node.cap = mock_cap

        box = MagicMock()
        box.cls = [0]
        box.conf = [0.92]
        r = MagicMock()
        r.boxes = [box]
        node.model = MagicMock(return_value=[r])

        await node.run_vision_loop()

        self.assertFalse(event_queue.empty())
        event = event_queue.get_nowait()
        from src.core.types import VisionEvent
        self.assertIsInstance(event, VisionEvent)
        self.assertEqual(event.anomaly_type, "human_survivor")

    async def test_low_confidence_does_not_emit_event(self):
        event_queue = asyncio.Queue()
        node = self._node(event_queue=event_queue, confidence_threshold=0.6, sample_interval=0.0)
        node.last_event_at = 0.0

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        call_count = [0]
        def read_side():
            call_count[0] += 1
            if call_count[0] == 1:
                return True, MagicMock()
            return False, None
        mock_cap.read.side_effect = read_side
        node.cap = mock_cap

        box = MagicMock()
        box.cls = [0]
        box.conf = [0.3]  # below threshold
        r = MagicMock()
        r.boxes = [box]
        node.model = MagicMock(return_value=[r])

        await node.run_vision_loop()
        self.assertTrue(event_queue.empty())

    async def test_non_person_class_does_not_emit_event(self):
        event_queue = asyncio.Queue()
        node = self._node(event_queue=event_queue, sample_interval=0.0)
        node.last_event_at = 0.0

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        call_count = [0]
        def read_side():
            call_count[0] += 1
            if call_count[0] == 1:
                return True, MagicMock()
            return False, None
        mock_cap.read.side_effect = read_side
        node.cap = mock_cap

        box = MagicMock()
        box.cls = [2]  # car — not person
        box.conf = [0.99]
        r = MagicMock()
        r.boxes = [box]
        node.model = MagicMock(return_value=[r])

        await node.run_vision_loop()
        self.assertTrue(event_queue.empty())

    async def test_rate_limiting_prevents_duplicate_events(self):
        event_queue = asyncio.Queue()
        node = self._node(event_queue=event_queue, sample_interval=60.0)
        node.last_event_at = time.time()  # just emitted

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        call_count = [0]
        def read_side():
            call_count[0] += 1
            if call_count[0] == 1:
                return True, MagicMock()
            return False, None
        mock_cap.read.side_effect = read_side
        node.cap = mock_cap

        box = MagicMock()
        box.cls = [0]
        box.conf = [0.95]
        r = MagicMock()
        r.boxes = [box]
        node.model = MagicMock(return_value=[r])

        await node.run_vision_loop()
        self.assertTrue(event_queue.empty())


if __name__ == "__main__":
    unittest.main()
