import asyncio
import json
import logging
from nodes.watchdog import WatchdogNode
from nodes.analyst import AnalystNode
from nodes.commander import CommanderNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global state for telemetry
current_telemetry = {}

class TelemetryUDPServer:
    def connection_made(self, transport):
        self.transport = transport
        logging.info("[UDP Server] Listening for Edge Middleware Telemetry on port 9000...")

    def datagram_received(self, data, addr):
        global current_telemetry
        try:
            payload = json.loads(data.decode('utf-8'))
            # Expecting something like: {"alt": 10.5, "lat": 47.3, "lon": 8.5, "heading": 180}
            current_telemetry = payload
        except Exception as e:
            pass # Ignore malformed packets silently

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
    watchdog = WatchdogNode(model_path="yolov10n.pt") # Ensure this model downloads/exists or use yolov8n.pt if v10 isn't cached

    # Attempt to download YOLO model dynamically if missing
    import os
    if not os.path.exists("yolov10n.pt"):
        logging.info("YOLOv10n not found locally, falling back to ultralytics auto-download...")

    if not watchdog.start_camera(0): # Try index 0
        logging.error("Failed to start webcam. Exiting sandbox.")
        return

    # 3. Vision Loop Task
    vision_task = asyncio.create_task(watchdog.run_vision_loop())
    
    # 4. Mock the "Trigger" linkage
    # In a real async architecture, Watchdog would push to a Queue.
    # For this sandbox, we will poll a mock trigger or intercept the watchdog loop.
    # To avoid modifying the watchdog loop heavily, we'll just simulate a trigger every 30 seconds for testing
    # if the watchdog doesn't trigger it natively.
    
    while True:
        await asyncio.sleep(30)
        # We simulate a trigger manually here just for integration testing
        logging.info("[Sandbox] Simulating an Anomaly Trigger for Pipeline Test...")
        
        if not current_telemetry:
            # Fake some telemetry if none received
            current_telemetry = {"lat": -35.363261, "lon": 149.165230, "alt": 20.0, "heading": 90}
            
        context = analyst.generate_context("Human Casualty", current_telemetry)
        
        command = await commander.generate_mavlink_command(context)
        
        if command:
            # Send the command back to the Edge Middleware over UDP port 9001
            logging.info(f"[Sandbox] Sending Reroute Command to Edge Middleware: {command}")
            transport.sendto(json.dumps(command).encode('utf-8'), ('127.0.0.1', 9001))

if __name__ == "__main__":
    try:
        asyncio.run(sandbox_pipeline())
    except KeyboardInterrupt:
        logging.info("Sandbox Terminated.")
