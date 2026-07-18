import asyncio
import json
import logging
import os
from core.pipeline import parse_sandbox_telemetry
from core.mavlink_adapter import MAVLinkDependencyError, MAVLinkTelemetryDecoder
from core.types import VisionEvent
from nodes.watchdog import WatchdogNode
from nodes.analyst import AnalystNode
from nodes.commander import CommanderNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global state for telemetry
current_telemetry = {}
current_snapshot = None

class TelemetryUDPServer:
    def __init__(self):
        self.mavlink_decoder = None

    def connection_made(self, transport):
        self.transport = transport
        logging.info("[UDP Server] Listening for Edge Middleware Telemetry on port 9000...")

    def datagram_received(self, data, addr):
        global current_telemetry, current_snapshot
        try:
            payload = json.loads(data.decode('utf-8'))
            current_telemetry = payload
            current_snapshot = parse_sandbox_telemetry(data)
            return
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

        try:
            if self.mavlink_decoder is None:
                self.mavlink_decoder = MAVLinkTelemetryDecoder()
            decoded_snapshot = self.mavlink_decoder.feed(data)
            if decoded_snapshot is not None:
                current_snapshot = decoded_snapshot
                current_telemetry = {
                    "drone_id": decoded_snapshot.drone_id,
                    "timestamp": decoded_snapshot.timestamp,
                    "lat": decoded_snapshot.latitude,
                    "lon": decoded_snapshot.longitude,
                    "alt": decoded_snapshot.altitude_m,
                    "heading": decoded_snapshot.heading_deg,
                    "battery_percent": decoded_snapshot.battery_percent,
                }
        except MAVLinkDependencyError as error:
            logging.error("Raw MAVLink received but decoder is unavailable: %s", error)
        except (ValueError, AttributeError) as error:
            logging.warning("Ignoring invalid telemetry datagram from %s: %s", addr, error)


async def process_watchdog_events(event_queue, analyst, commander, transport):
    while True:
        event = await event_queue.get()
        try:
            if current_snapshot is None:
                logging.warning("Skipping vision event without telemetry")
                continue
            context = analyst.generate_event_context(event, current_snapshot)
            command = await commander.generate_mavlink_command(context, current_snapshot)
            if command:
                logging.info("Sending validated command to Edge Middleware: %s", command)
                transport.sendto(json.dumps(command).encode('utf-8'), ('127.0.0.1', 9001))
        finally:
            event_queue.task_done()

async def sandbox_pipeline():
    logging.info("Starting Maas-LLM Sandbox Pipeline...")
    
    # 1. Start UDP Server for Telemetry
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: TelemetryUDPServer(),
        local_addr=('127.0.0.1', 9000)
    )

    # 2. Initialize Nodes
    analyst = AnalystNode()
    commander = CommanderNode()
    event_queue = asyncio.Queue(maxsize=32)
    watchdog = WatchdogNode(
        model_path=os.getenv("YOLO_MODEL_PATH", "yolov10n.pt"),
        event_queue=event_queue,
        sample_interval=float(os.getenv("VISION_SAMPLE_INTERVAL", "0.2")),
        confidence_threshold=float(os.getenv("VISION_CONFIDENCE", "0.6")),
    )

    # Attempt to download YOLO model dynamically if missing
    if not os.path.exists(watchdog.model_path):
        logging.info("YOLO model not found at %s; Ultralytics may download it.", watchdog.model_path)

    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    if not watchdog.start_camera(camera_index):
        logging.error("Failed to start webcam. Exiting sandbox.")
        return

    # 3. Vision Loop Task
    vision_task = asyncio.create_task(watchdog.run_vision_loop())
    event_task = asyncio.create_task(
        process_watchdog_events(event_queue, analyst, commander, transport)
    )
    
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        vision_task.cancel()
        event_task.cancel()
        watchdog.stop()
        transport.close()

if __name__ == "__main__":
    try:
        asyncio.run(sandbox_pipeline())
    except KeyboardInterrupt:
        logging.info("Sandbox Terminated.")
