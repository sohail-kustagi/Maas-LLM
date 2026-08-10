"""MAAS-LLM — Main Entry Point

Modes:
  sandbox  (default) — Webcam → YOLO → Analyst → Commander
  sitl               — MAVProxy UDP → Analyst → Commander (scripted events, no camera)

Environment variables:
  CAMERA_INDEX          int   (default: 0)
  VISION_SAMPLE_INTERVAL float (default: 0.2)
  VISION_CONFIDENCE     float (default: 0.6)
  YOLO_MODEL_PATH       str   (default: weights/best.pt)
  DRY_RUN               1     Print command, wait for [y/N] before sending (default: 1 in sitl, 0 in sandbox)
  SITL_UDP_IN           str   (default: 127.0.0.1:14550) — where we receive raw MAVLink from MAVProxy
  TELEMETRY_IN_PORT     int   (default: 9000) — UDP port for sandbox telemetry
  COMMAND_OUT_PORT      int   (default: 9001) — UDP port for outbound commands
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from src.core.pipeline import parse_sandbox_telemetry
from src.core.mavlink_adapter import MAVLinkDependencyError, MAVLinkTelemetryDecoder
from src.core.types import VisionEvent
from src.core.weather_provider import WeatherProvider
from src.core.weather_policy import validate_weather_for_flight, WeatherPolicyError
from src.core.flight_planner import evaluate_feasibility
from src.core.route_types import DroneCapabilities, FlightPlan, Waypoint
from src.core.mission_profiles import PROFILES, MissionProfile
from src.core.mission_profiles import PROFILES, MissionProfile
from src.nodes.analyst import AnalystNode
from src.core.evaluator import HackathonEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────

current_telemetry: dict = {}
current_snapshot = None
mavlink_source_addr = None   # set when first MAVLink datagram arrives


# ──────────────────────────────────────────────────────────────────────────────
# UDP Telemetry Server
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryUDPServer:
    """Accepts both JSON sandbox telemetry and raw MAVLink bytes."""

    def __init__(self):
        self.mavlink_decoder = None

    def connection_made(self, transport):
        self.transport = transport
        logger.info("[UDP Server] Listening for telemetry on bound port.")

    def connection_lost(self, exc):
        pass

    def datagram_received(self, data, addr):
        global current_telemetry, current_snapshot, mavlink_source_addr
        self.last_addr = addr

        # Try JSON sandbox telemetry first (e.g. from a test harness)
        try:
            payload = json.loads(data.decode("utf-8"))
            current_telemetry = payload
            current_snapshot = parse_sandbox_telemetry(data)
            return
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

        # Fall back to raw MAVLink bytes (from MAVProxy / SITL)
        try:
            if self.mavlink_decoder is None:
                self.mavlink_decoder = MAVLinkTelemetryDecoder()
            decoded = self.mavlink_decoder.feed(data)
            if decoded is not None:
                mavlink_source_addr = addr   # remember who's sending MAVLink
                current_snapshot = decoded
                current_telemetry = {
                    "drone_id":        decoded.drone_id,
                    "timestamp":       decoded.timestamp,
                    "lat":             decoded.latitude,
                    "lon":             decoded.longitude,
                    "alt":             decoded.altitude_m,
                    "heading":         decoded.heading_deg,
                    "battery_percent": decoded.battery_percent,
                    "flight_mode":     decoded.flight_mode,
                    "is_armed":        decoded.is_armed,
                }
                logger.debug("[Telemetry] MAVLink snapshot: %s", current_telemetry)
        except MAVLinkDependencyError as err:
            logger.error("Raw MAVLink received but decoder unavailable: %s", err)
        except (ValueError, AttributeError) as err:
            logger.warning("Ignoring invalid datagram from %s: %s", addr, err)


# ──────────────────────────────────────────────────────────────────────────────
# Event Processing
# ──────────────────────────────────────────────────────────────────────────────

async def _dry_run_gate(command: dict, dry_run: bool) -> bool:
    """If dry_run is True, print the command and wait for user confirmation.
    Returns True if the command should be sent."""
    if not dry_run:
        return True

    print("\n" + "═" * 60)
    print("  DRY-RUN: Commander output (NOT YET SENT)")
    print("═" * 60)
    print(json.dumps(command, indent=2))
    print("═" * 60)
    try:
        answer = await asyncio.get_running_loop().run_in_executor(
            None, lambda: input("  Send this command? [y/N]: ").strip().lower()
        )
        return answer in ("y", "yes")
    except EOFError:
        # Non-interactive environment — auto-deny
        logger.info("[DryRun] Non-interactive mode — command NOT sent.")
        return False


async def process_watchdog_events(
    event_queue: asyncio.Queue,
    analyst: AnalystNode,
    commander,
    transport,
    weather_provider: WeatherProvider,
    mission_profile: MissionProfile,
    dry_run: bool = False,
    command_out_addr: tuple[str, int] = ("127.0.0.1", 9001),
) -> None:
    while True:
        event = await event_queue.get()
        try:
            if current_snapshot is None:
                logger.warning("[Pipeline] Skipping event — no telemetry yet. "
                               "Ensure MAVProxy / SITL is sending to the telemetry port.")
                continue

            # Fetch weather and evaluate feasibility
            weather = None
            feasibility = None
            try:
                weather = await weather_provider.get_weather(
                    current_snapshot.latitude, current_snapshot.longitude
                )
                if weather:
                    validate_weather_for_flight(weather)
                    single_wp = Waypoint(
                        latitude=current_snapshot.latitude,
                        longitude=current_snapshot.longitude,
                        altitude_m=current_snapshot.altitude_m,
                    )
                    plan = FlightPlan(
                        plan_id="event-plan",
                        drone_id=event.drone_id,
                        waypoints=[single_wp],
                        created_timestamp=event.timestamp,
                    )
                    capabilities = DroneCapabilities(
                        max_speed_mps=15.0, max_climb_rate_mps=5.0,
                        max_descent_rate_mps=3.0, min_altitude_m=5.0,
                        max_altitude_m=120.0, max_mission_distance_m=5000.0,
                        max_waypoints=20, battery_reserve_percent=20.0,
                        inspection_dwell_time_s=30.0,
                    )
                    feasibility = evaluate_feasibility(plan, current_snapshot, capabilities, weather)
            except WeatherPolicyError as err:
                logger.warning("[Weather] Policy blocked event: %s", err)
                continue
            except Exception as err:
                logger.warning("[Weather] Fetch failed (proceeding without weather): %s", err)

            # Build analyst context and run Commander
            context = analyst.generate_event_context(event, current_snapshot, weather, feasibility, mission_profile)
            command = await commander.generate_mavlink_command(context, current_snapshot, mission_profile, event.anomaly_type)

            if command:
                approved = await _dry_run_gate(command, dry_run)
                if approved:
                    logger.info("[Pipeline] Sending command to %s:%d — %s", *command_out_addr, command)
                    
                    # Direct SITL Injection via pymavlink to the MAVProxy source address
                    if mavlink_source_addr is not None:
                        try:
                            from pymavlink import mavutil
                            mav = mavutil.mavlink.MAVLink(None, srcSystem=255, srcComponent=0)
                            dest = mavlink_source_addr

                            logger.info("[SITL] → GUIDED mode")
                            msg_mode = mavutil.mavlink.MAVLink_command_long_message(
                                1, 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                1, 4, 0, 0, 0, 0, 0
                            )
                            transport.sendto(msg_mode.pack(mav), dest)
                            await asyncio.sleep(0.5)

                            logger.info("[SITL] → ARM")
                            msg_arm = mavutil.mavlink.MAVLink_command_long_message(
                                1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                1, 0, 0, 0, 0, 0, 0
                            )
                            transport.sendto(msg_arm.pack(mav), dest)
                            await asyncio.sleep(1.5)

                            logger.info("[SITL] → TAKEOFF 15m")
                            msg_takeoff = mavutil.mavlink.MAVLink_command_long_message(
                                1, 1, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                0, 0, 0, 0, 0, 0, 15
                            )
                            transport.sendto(msg_takeoff.pack(mav), dest)
                            await asyncio.sleep(8.0)

                            x = float(command.get("x", 0))
                            y = float(command.get("y", 0))
                            z = -15.0   # always target 15 m altitude
                            logger.info(f"[SITL] → SET_POSITION_TARGET_LOCAL_NED x={x} y={y} z={z} → {dest}")
                            msg_pos = mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                                0, 1, 1,
                                mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
                                0b0000111111111000,   # mask: pos only
                                x, y, z,
                                0, 0, 0, 0, 0, 0, 0, 0
                            )
                            transport.sendto(msg_pos.pack(mav), dest)
                            logger.info("[SITL] MAVLink sequence complete.")
                        except Exception as e:
                            logger.error(f"[SITL] Direct injection failed: {e}", exc_info=True)
                    else:
                        logger.warning("[SITL] No MAVLink source address yet — is MAVProxy connected?")
                    transport.sendto(json.dumps(command).encode("utf-8"), command_out_addr)
                else:
                    logger.info("[Pipeline] Command NOT sent (dry-run denied).")
            else:
                logger.warning("[Pipeline] Commander returned no valid command for event: %s", event.anomaly_type)
        finally:
            event_queue.task_done()


# ──────────────────────────────────────────────────────────────────────────────
# Startup banner
# ──────────────────────────────────────────────────────────────────────────────

def _print_banner(mode: str, telemetry_port: int, command_out_port: int, dry_run: bool, profile: str = "sandbox", show_ui: bool = False) -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║           MAAS-LLM  ·  Autonomous Aerial AI         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Mode        : {mode.upper():<38}║")
    print(f"║  Profile     : {profile.upper():<38}║")
    print(f"║  Live UI     : {'ENABLED' if show_ui else 'DISABLED':<38}║")
    print(f"║  Telemetry   : UDP 0.0.0.0:{telemetry_port:<26}║")
    print(f"║  Command Out : UDP 127.0.0.1:{command_out_port:<24}║")
    print(f"║  Dry-Run     : {'ENABLED — awaiting approval' if dry_run else 'DISABLED — live commands':<38}║")
    if mode == "sitl":
        print("║  SITL        : Connect MAVProxy → UDP 127.0.0.1:" + f"{telemetry_port:<5}║")
        print("║  Example     : mavproxy.py --master tcp:127.0.0.1:5762 \\   ║")
        print("║                  --out udp:127.0.0.1:" + f"{telemetry_port:<18}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry points
# ──────────────────────────────────────────────────────────────────────────────

async def run_sandbox_pipeline(args: argparse.Namespace) -> None:
    """Webcam → YOLO → Analyst → Commander."""
    from src.nodes.watchdog import WatchdogNode
    from src.nodes.commander import CommanderNode

    dry_run = bool(int(os.getenv("DRY_RUN", "0")))
    telemetry_port = int(os.getenv("TELEMETRY_IN_PORT", "9000"))
    command_out_port = int(os.getenv("COMMAND_OUT_PORT", "9001"))

    _print_banner("sandbox", telemetry_port, command_out_port, dry_run, getattr(args, "profile", "sandbox"), getattr(args, "show_ui", False))

    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        TelemetryUDPServer,
        local_addr=("0.0.0.0", telemetry_port),
    )

    analyst = AnalystNode()
    commander = CommanderNode()
    weather_provider = WeatherProvider(use_fake=False)
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    profile_name = getattr(args, "profile", "sandbox")
    mission_profile = PROFILES.get(profile_name, PROFILES["free"])
    
    class_map = {cls_id: "anomaly" for cls_id in mission_profile.yolo_class_watchlist}
    watchdog = WatchdogNode(
        model_path=os.getenv("YOLO_MODEL_PATH", "weights/best.pt"),
        event_queue=event_queue,
        sample_interval=float(os.getenv("VISION_SAMPLE_INTERVAL", "0.2")),
        confidence_threshold=float(os.getenv("VISION_CONFIDENCE", "0.6")),
        class_map=class_map,
        show_ui=getattr(args, "show_ui", False),
        mission_profile_name=profile_name,
    )

    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    if not watchdog.start_camera(camera_index):
        logger.error("[Sandbox] Failed to start webcam index %d. Exiting.", camera_index)
        transport.close()
        return

    vision_task = asyncio.create_task(watchdog.run_vision_loop())
    event_task = asyncio.create_task(
        process_watchdog_events(
            event_queue, analyst, commander, transport,
            weather_provider, mission_profile, dry_run,
            ("127.0.0.1", command_out_port),
        )
    )

    try:
        logger.info("[Sandbox] Pipeline running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    finally:
        vision_task.cancel()
        event_task.cancel()
        watchdog.stop()
        transport.close()


async def run_sitl_pipeline(args: argparse.Namespace) -> None:
    """MAVProxy UDP → MAVLinkDecoder → Analyst → Commander (scripted events)."""
    from src.nodes.commander import CommanderNode
    from src.nodes.watchdog import WatchdogNode
    from src.core.sitl_injector import SITLInjector

    dry_run = bool(int(os.getenv("DRY_RUN", "1")))  # default: dry-run ON for SITL
    telemetry_port = int(os.getenv("TELEMETRY_IN_PORT", "9000"))
    command_out_port = int(os.getenv("COMMAND_OUT_PORT", "9001"))

    _print_banner("sitl", telemetry_port, command_out_port, dry_run, getattr(args, "profile", "sandbox"), getattr(args, "show_ui", False))

    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        TelemetryUDPServer,
        local_addr=("0.0.0.0", telemetry_port),
    )

    analyst = AnalystNode()
    commander = CommanderNode()
    weather_provider = WeatherProvider(use_fake=True)  # use fake in SITL to avoid API calls
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    profile_name = getattr(args, "profile", "sandbox")
    mission_profile = PROFILES.get(profile_name, PROFILES["free"])
    
    injector = SITLInjector(event_queue, drone_id="sitl-drone-1", mission_profile_name=profile_name)

    # Allow webcam YOLO in SITL if requested
    show_ui = getattr(args, "show_ui", False)
    class_map = {cls_id: "anomaly" for cls_id in mission_profile.yolo_class_watchlist}
    watchdog = WatchdogNode(
        model_path=os.getenv("YOLO_MODEL_PATH", "weights/best.pt"),
        event_queue=event_queue,
        sample_interval=float(os.getenv("VISION_SAMPLE_INTERVAL", "0.2")),
        confidence_threshold=float(os.getenv("VISION_CONFIDENCE", "0.6")),
        class_map=class_map,
        show_ui=show_ui,
        mission_profile_name=profile_name,
    )
    
    evaluator = None
    if getattr(args, "eval_video", None):
        evaluator = HackathonEvaluator(args.eval_video, profile_name)
        watchdog.set_evaluator(evaluator)
        commander.set_evaluator(evaluator)
        
    vision_task = None
    if show_ui or getattr(args, "eval_video", None):
        video_src = getattr(args, "eval_video", None) or int(os.getenv("CAMERA_INDEX", "0"))
        if watchdog.start_camera(video_src):
            vision_task = asyncio.create_task(watchdog.run_vision_loop())
        else:
            logger.error("[SITL] Failed to start webcam index %d for UI. Continuing without it.", camera_index)

    logger.info("[SITL] Waiting 5 s for telemetry to arrive before injecting events...")
    await asyncio.sleep(5.0)

    inject_task = asyncio.create_task(injector.run())
    event_task = asyncio.create_task(
        process_watchdog_events(
            event_queue, analyst, commander, transport,
            weather_provider, mission_profile, dry_run,
            ("127.0.0.1", command_out_port),
        )
    )

    try:
        if getattr(args, "eval_video", None):
            logger.info("[SITL] Video Eval Mode: Waiting for video to finish...")
            if vision_task:
                await vision_task
            logger.info("[SITL] Video finished. Giving pipeline 5s to flush events.")
            await asyncio.sleep(5.0)
        else:
            logger.info("[SITL] Pipeline running. Waiting for scenario to complete...")
            await inject_task  # finishes after ~120 s
            logger.info("[SITL] Scenario complete. Giving pipeline 10 s to flush events.")
            await asyncio.sleep(10.0)
    finally:
        inject_task.cancel()
        event_task.cancel()
        if vision_task:
            vision_task.cancel()
            watchdog.stop()
        if evaluator:
            evaluator.generate_report()
        transport.close()
        logger.info("[SITL] Pipeline shut down cleanly.")

async def run_free_pipeline(args: argparse.Namespace) -> None:
    """Live telemetry → Validates Waypoint → ARMs & Takes Off → Flies to Lat/Lon → YOLO Watchdog active."""
    from src.nodes.watchdog import WatchdogNode
    from src.nodes.commander import CommanderNode
    from src.core.mavlink_adapter import MAVLinkTelemetryDecoder

    if args.lat is None or args.lon is None or args.alt is None:
        logger.error("[FreeMode] --lat, --lon, and --alt must be provided.")
        return

    dry_run = bool(int(os.getenv("DRY_RUN", "1")))
    telemetry_port = int(os.getenv("TELEMETRY_IN_PORT", "9000"))
    command_out_port = int(os.getenv("COMMAND_OUT_PORT", "9001"))

    profile_name = getattr(args, "profile", "free")
    mission_profile = PROFILES.get(profile_name, PROFILES["free"])

    _print_banner("free", telemetry_port, command_out_port, dry_run, profile_name, getattr(args, "show_ui", False))

    loop = asyncio.get_running_loop()
    transport, server = await loop.create_datagram_endpoint(
        TelemetryUDPServer,
        local_addr=("0.0.0.0", telemetry_port),
    )

    analyst = AnalystNode()
    commander = CommanderNode()
    weather_provider = WeatherProvider(use_fake=False)
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    class_map = {cls_id: "anomaly" for cls_id in mission_profile.yolo_class_watchlist}
    watchdog = WatchdogNode(
        model_path=os.getenv("YOLO_MODEL_PATH", "weights/best.pt"),
        event_queue=event_queue,
        sample_interval=float(os.getenv("VISION_SAMPLE_INTERVAL", "0.2")),
        confidence_threshold=float(os.getenv("VISION_CONFIDENCE", "0.6")),
        class_map=class_map,
        show_ui=getattr(args, "show_ui", False),
        mission_profile_name=profile_name,
    )

    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    if not watchdog.start_camera(camera_index):
        logger.error("[FreeMode] Failed to start webcam index %d. Exiting.", camera_index)
        transport.close()
        return

    logger.info("[FreeMode] Waiting for valid telemetry connection...")
    while current_snapshot is None:
        await asyncio.sleep(1.0)
    
    logger.info(f"[FreeMode] Connected! Current Pos: {current_snapshot.latitude}, {current_snapshot.longitude}, Alt: {current_snapshot.altitude_m}m")
    
    logger.info(f"[FreeMode] Target Pos: {args.lat}, {args.lon}, Alt: {args.alt}m")
    command = {
        "command": "SET_POSITION_TARGET_GLOBAL_INT",
        "target_system": 1,
        "target_component": 1,
        "lat": args.lat,
        "lon": args.lon,
        "alt": args.alt,
        "reasoning": "Free flight mode to user coordinates."
    }

    approved = await _dry_run_gate(command, dry_run)
    if approved:
        if hasattr(transport, "_protocol") and hasattr(transport._protocol, "last_addr"):
            try:
                import time
                from pymavlink import mavutil
                mav = mavutil.mavlink.MAVLink(None, srcSystem=255, srcComponent=0)
                
                if not current_snapshot.is_armed or current_snapshot.altitude_m < 2.0:
                    logger.info("[FreeMode] Drone is grounded. Forcing GUIDED, ARMING, and TAKING OFF...")
                    msg_mode = mavutil.mavlink.MAVLink_command_long_message(1, 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 4, 0, 0, 0, 0, 0)
                    transport.sendto(msg_mode.pack(mav), transport._protocol.last_addr)
                    await asyncio.sleep(0.5)
                    
                    msg_arm = mavutil.mavlink.MAVLink_command_long_message(1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                    transport.sendto(msg_arm.pack(mav), transport._protocol.last_addr)
                    await asyncio.sleep(1.0)
                    
                    msg_takeoff = mavutil.mavlink.MAVLink_command_long_message(1, 1, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, args.alt)
                    transport.sendto(msg_takeoff.pack(mav), transport._protocol.last_addr)
                    await asyncio.sleep(8.0)
                
                msg_pos = MAVLinkTelemetryDecoder.build_set_position_target_global_int(args.lat, args.lon, args.alt, mav)
                logger.info(f"[FreeMode] Injecting MAVLink position target globally.")
                transport.sendto(msg_pos.pack(mav), transport._protocol.last_addr)
            except Exception as e:
                logger.error(f"[FreeMode] MAVLink dispatch failed: {e}")
    else:
        logger.info("[FreeMode] Command denied.")

    vision_task = asyncio.create_task(watchdog.run_vision_loop())
    event_task = asyncio.create_task(
        process_watchdog_events(
            event_queue, analyst, commander, transport,
            weather_provider, mission_profile, dry_run,
            ("127.0.0.1", command_out_port),
        )
    )

    try:
        logger.info("[FreeMode] Pipeline running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    finally:
        vision_task.cancel()
        event_task.cancel()
        watchdog.stop()
        transport.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAAS-LLM Autonomous Aerial AI Pipeline")
    parser.add_argument(
        "--mode",
        choices=["sandbox", "sitl", "free"],
        default="sandbox",
        help="sandbox = webcam + real-time; sitl = MAVProxy UDP + scripted events; free = GPS Waypoint (default: sandbox)",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="sandbox",
        help="Mission profile (flood, fire, search_and_rescue, etc.)"
    )
    parser.add_argument("--lat", type=float, help="Target latitude for free mode")
    parser.add_argument("--lon", type=float, help="Target longitude for free mode")
    parser.add_argument("--alt", type=float, help="Target altitude for free mode")
    parser.add_argument("--show-ui", action="store_true", help="Enable Live YOLO UI window")
    parser.add_argument("--eval-video", type=str, help="Path to pre-recorded video for Hackathon Evaluator mode")
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    try:
        if _args.mode == "sitl":
            asyncio.run(run_sitl_pipeline(_args))
        elif _args.mode == "free":
            asyncio.run(run_free_pipeline(_args))
        else:
            asyncio.run(run_sandbox_pipeline(_args))
    except KeyboardInterrupt:
        logger.info("Pipeline terminated by user.")
