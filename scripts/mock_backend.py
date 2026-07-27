#!/usr/bin/env python3
import socket
import json
import time
from pymavlink import mavutil

def main():
    print("🚀 [Mock Backend] Starting up...")
    
    # 1. Connect to SITL via MAVProxy (UDP 14551)
    master = mavutil.mavlink_connection('udpin:127.0.0.1:14551', source_system=254)
    print("⏳ [Mock Backend] Waiting for heartbeat from SITL...")
    master.wait_heartbeat()
    print(f"💓 [Mock Backend] Heartbeat received from system {master.target_system} component {master.target_component}")

    # 2. Put drone in GUIDED mode, arm, and takeoff so it can accept position commands
    print("🛫 [Mock Backend] Arming and taking off to 20m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        1, 4, 0, 0, 0, 0, 0) # Mode 4 = GUIDED
    time.sleep(1)
    
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0)
    time.sleep(1)
    
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, 20)
    
    print("✅ [Mock Backend] Takeoff commanded. Listening for MAAS-LLM JSON on UDP 9001...")

    # 3. Listen for JSON from MAAS-LLM on UDP 9001
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9001))

    while True:
        data, addr = sock.recvfrom(4096)
        try:
            command = json.loads(data.decode('utf-8'))
            if command.get("command") == "SET_POSITION_TARGET_LOCAL_NED":
                print(f"📥 [Mock Backend] Received JSON command: {command}")
                
                # Convert JSON to MAVLink binary and send
                # Coordinate frame 1 = MAV_FRAME_LOCAL_NED
                # Type mask 0b0000111111111000 = 0x0DF8 (ignore velocity/accel/yaw, only use position)
                master.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    master.target_system,
                    master.target_component,
                    1,  # coordinate_frame (LOCAL_NED)
                    0x0DF8, # type_mask (position only)
                    command["x"], command["y"], command["z"], # x, y, z
                    0, 0, 0, # vx, vy, vz
                    0, 0, 0, # afx, afy, afz
                    0, 0     # yaw, yaw_rate
                )
                print(f"🚁 [Mock Backend] Forwarded MAVLink SET_POSITION_TARGET to {master.target_system}:{master.target_component}! Moving to x={command['x']}, y={command['y']}, z={command['z']}")
                
        except Exception as e:
            print(f"❌ [Mock Backend] Error processing command: {e}")

if __name__ == "__main__":
    main()
