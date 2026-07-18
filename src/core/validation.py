import math
from typing import Any, Mapping, Optional

from .types import CommanderCommand, TelemetrySnapshot


COMMAND_NAME = "SET_POSITION_TARGET_LOCAL_NED"
MAX_HORIZONTAL_OFFSET_M = 100.0
MIN_DOWN_OFFSET_M = -50.0
MAX_DOWN_OFFSET_M = 20.0
MAX_TELEMETRY_AGE_SECONDS = 5.0


class CommandValidationError(ValueError):
    """Raised when an LLM command cannot be safely forwarded."""


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandValidationError(f"{field_name} must be a number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise CommandValidationError(f"{field_name} must be finite")
    return numeric_value


def validate_commander_output(
    output: Mapping[str, Any],
    telemetry: Optional[TelemetrySnapshot] = None,
    now: Optional[float] = None,
) -> CommanderCommand:
    if not isinstance(output, Mapping):
        raise CommandValidationError("commander output must be an object")

    required_fields = {
        "command",
        "target_system",
        "target_component",
        "x",
        "y",
        "z",
        "reasoning",
    }
    missing_fields = required_fields.difference(output)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise CommandValidationError(f"missing fields: {missing}")

    if output["command"] != COMMAND_NAME:
        raise CommandValidationError("unsupported command")

    target_system = output["target_system"]
    target_component = output["target_component"]
    if isinstance(target_system, bool) or not isinstance(target_system, int):
        raise CommandValidationError("target_system must be an integer")
    if isinstance(target_component, bool) or not isinstance(target_component, int):
        raise CommandValidationError("target_component must be an integer")
    if not 1 <= target_system <= 255:
        raise CommandValidationError("target_system is outside MAVLink range")
    if not 1 <= target_component <= 255:
        raise CommandValidationError("target_component is outside MAVLink range")

    x = _number(output["x"], "x")
    y = _number(output["y"], "y")
    z = _number(output["z"], "z")
    if abs(x) > MAX_HORIZONTAL_OFFSET_M or abs(y) > MAX_HORIZONTAL_OFFSET_M:
        raise CommandValidationError("horizontal offset exceeds safety bound")
    if not MIN_DOWN_OFFSET_M <= z <= MAX_DOWN_OFFSET_M:
        raise CommandValidationError("vertical offset exceeds safety bound")

    reasoning = output["reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise CommandValidationError("reasoning must be a non-empty string")

    if telemetry is not None:
        if telemetry.age_seconds(now) > MAX_TELEMETRY_AGE_SECONDS:
            raise CommandValidationError("telemetry is stale")
        if telemetry.battery_percent is not None and telemetry.battery_percent < 15:
            raise CommandValidationError("battery is too low for reroute")

    return CommanderCommand(
        command=COMMAND_NAME,
        target_system=target_system,
        target_component=target_component,
        x=x,
        y=y,
        z=z,
        reasoning=reasoning.strip(),
    )
