"""Tests for CommanderNode — without loading a real LLM model.

The LLM is mocked at the sys.modules level before import so we never need
a real llama_cpp installation for testing.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from src.core.types import TelemetrySnapshot


def _make_snap(**overrides) -> TelemetrySnapshot:
    base = dict(
        drone_id="drone-1",
        timestamp=time.time(),
        latitude=-35.363261,
        longitude=149.165230,
        altitude_m=25.0,
        heading_deg=90.0,
        battery_percent=80.0,
    )
    base.update(overrides)
    return TelemetrySnapshot(**base)


def _valid_llm_response() -> dict:
    return {
        "choices": [{"text": json.dumps({
            "command": "SET_POSITION_TARGET_LOCAL_NED",
            "target_system": 1,
            "target_component": 1,
            "x": 10.0,
            "y": 5.0,
            "z": -2.0,
            "reasoning": "Move toward fire."
        })}]
    }


def _make_node_with_mock_llm(llm_return_value):
    """Return a CommanderNode with a mocked Llama that returns the given value."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.return_value = llm_return_value

    mock_llama_cpp = MagicMock()
    mock_llama_cpp.Llama.return_value = mock_llm_instance
    mock_llama_cpp.LlamaGrammar.from_string.return_value = MagicMock()

    # Patch llama_cpp in sys.modules to avoid ImportError
    with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
        # Force re-import with the mock in place
        if "src.nodes.commander" in sys.modules:
            del sys.modules["src.nodes.commander"]
        if "nodes.commander" in sys.modules:
            del sys.modules["nodes.commander"]

        from src.nodes.commander import CommanderNode

        with patch.object(os.path, "exists", return_value=True):
            node = object.__new__(CommanderNode)
            node.model_path = "/fake/model.gguf"
            node.llm = mock_llm_instance
            node.grammar = mock_llama_cpp.LlamaGrammar.from_string("stub")

    return node


class CommanderPromptTests(unittest.TestCase):
    """Test that the source code has the right prompt structure."""

    def _get_source(self):
        import inspect
        mock_llama_cpp = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            if "src.nodes.commander" in sys.modules:
                del sys.modules["src.nodes.commander"]
            from src.nodes.commander import CommanderNode
            return inspect.getsource(CommanderNode.generate_mavlink_command)

    def test_system_prompt_contains_command_name(self):
        source = self._get_source()
        self.assertIn("SET_POSITION_TARGET_LOCAL_NED", source)

    def test_prompt_format_includes_phi3_tokens(self):
        source = self._get_source()
        self.assertIn("<|system|>", source)
        self.assertIn("<|user|>", source)
        self.assertIn("<|assistant|>", source)

    def test_grammar_uses_correct_json_separator(self):
        """Grammar must use single ':' not '::' as JSON key-value separator."""
        mock_llama_cpp = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            if "src.nodes.commander" in sys.modules:
                del sys.modules["src.nodes.commander"]
            import inspect
            from src.nodes.commander import CommanderNode
            source = inspect.getsource(CommanderNode.__init__)
        # Should have ws ":" ws (correct JSON), not ws "::" ws (wrong)
        self.assertNotIn("'::'", source)


class CommanderGenerateTests(unittest.IsolatedAsyncioTestCase):

    async def test_valid_llm_output_returns_command_dict(self):
        node = _make_node_with_mock_llm(_valid_llm_response())
        snap = _make_snap()
        result = await node.generate_mavlink_command("Investigate fire.", snap)
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], "SET_POSITION_TARGET_LOCAL_NED")
        self.assertEqual(result["x"], 10.0)

    async def test_invalid_json_returns_none(self):
        node = _make_node_with_mock_llm({"choices": [{"text": "this is not json at all"}]})
        result = await node.generate_mavlink_command("Investigate.", _make_snap())
        self.assertIsNone(result)

    async def test_validation_failure_returns_none(self):
        """x=999 violates MAX_HORIZONTAL_OFFSET_M=100 → None."""
        bad = json.dumps({
            "command": "SET_POSITION_TARGET_LOCAL_NED",
            "target_system": 1, "target_component": 1,
            "x": 999.0, "y": 0.0, "z": 0.0,
            "reasoning": "Too far."
        })
        node = _make_node_with_mock_llm({"choices": [{"text": bad}]})
        result = await node.generate_mavlink_command("Investigate.", _make_snap())
        self.assertIsNone(result)

    async def test_stale_telemetry_causes_validation_failure(self):
        node = _make_node_with_mock_llm(_valid_llm_response())
        stale = _make_snap(timestamp=time.time() - 60.0)
        result = await node.generate_mavlink_command("Investigate.", stale)
        self.assertIsNone(result)

    async def test_low_battery_causes_validation_failure(self):
        node = _make_node_with_mock_llm(_valid_llm_response())
        low_battery = _make_snap(battery_percent=5.0)
        result = await node.generate_mavlink_command("Investigate.", low_battery)
        self.assertIsNone(result)

    async def test_works_without_telemetry(self):
        node = _make_node_with_mock_llm(_valid_llm_response())
        result = await node.generate_mavlink_command("Investigate.", telemetry=None)
        self.assertIsNotNone(result)
        self.assertIn("command", result)


class CommanderDownloadModelTests(unittest.TestCase):

    def _get_node_class(self):
        mock_llama_cpp = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            if "src.nodes.commander" in sys.modules:
                del sys.modules["src.nodes.commander"]
            from src.nodes.commander import CommanderNode
        return CommanderNode

    def test_skips_download_if_file_exists(self):
        CommanderNode = self._get_node_class()
        with tempfile.NamedTemporaryFile() as f:
            node = object.__new__(CommanderNode)
            node.model_path = f.name
            with patch("urllib.request.urlretrieve") as mock_dl:
                node.download_model("some/repo", "model.gguf")
                mock_dl.assert_not_called()

    def test_report_callback_handles_zero_total_size(self):
        CommanderNode = self._get_node_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "nonexistent.gguf")
            node = object.__new__(CommanderNode)
            node.model_path = model_path

            captured_callback = []

            def fake_urlretrieve(url, path, reporthook):
                captured_callback.append(reporthook)

            with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                node.download_model("some/repo", "nonexistent.gguf")

            self.assertEqual(len(captured_callback), 1)
            callback = captured_callback[0]
            try:
                callback(0, 8192, 0)  # total_size=0 → must NOT raise ZeroDivisionError
            except ZeroDivisionError:
                self.fail("report callback raised ZeroDivisionError for total_size=0")


if __name__ == "__main__":
    unittest.main()
