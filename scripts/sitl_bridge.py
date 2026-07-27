#!/usr/bin/env python3
"""
sitl_bridge.py — One-command SITL development launcher.

Starts ArduPilot SITL and bridges it to MAAS-LLM via MAVProxy.

Usage:
    python scripts/sitl_bridge.py                    # launch everything
    python scripts/sitl_bridge.py --no-sitl          # MAVProxy only (SITL already running)
    python scripts/sitl_bridge.py --print-manual     # print manual start commands
    python scripts/sitl_bridge.py --check            # check dependencies only

Architecture:
    SITL (TCP 5762) → MAVProxy → UDP 127.0.0.1:14550 → MAAS-LLM UDP:9000

Dependencies:
    - ArduPilot at ~/ardupilot (already installed)
    - mavproxy.py (install: pip install MAVProxy)
    - MAAS-LLM main.py

This script does NOT start MAAS-LLM itself. Start it separately:
    cd /home/cipher/MAAS/Maas-LLM && python -m src.main --mode sitl
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ARDUPILOT_ROOT = os.path.expanduser("~/ardupilot")
SIM_VEHICLE = os.path.join(ARDUPILOT_ROOT, "Tools", "autotest", "sim_vehicle.py")

SITL_TCP_HOST = "127.0.0.1"
SITL_TCP_PORT = 5762

MAVPROXY_OUT_HOST = "127.0.0.1"
MAVPROXY_OUT_PORT = 14550  # → MAAS-LLM listens here

MAAS_LLM_TELEMETRY_PORT = 9000  # MAAS-LLM main.py listens on this port

# ──────────────────────────────────────────────────────────────────────────────
# Dependency checks
# ──────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    ok = True

    # ArduPilot
    if os.path.isfile(SIM_VEHICLE):
        print(f"  ✅  ArduPilot sim_vehicle.py found at {SIM_VEHICLE}")
    else:
        print(f"  ❌  sim_vehicle.py NOT found at {SIM_VEHICLE}")
        print("       Install ArduPilot: git clone https://github.com/ArduPilot/ardupilot ~/ardupilot")
        ok = False

    # MAVProxy
    mavproxy = shutil.which("mavproxy.py")
    if mavproxy:
        print(f"  ✅  mavproxy.py found at {mavproxy}")
    else:
        print("  ❌  mavproxy.py NOT found in PATH")
        print("       Install: pip install MAVProxy")
        ok = False

    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Process launchers
# ──────────────────────────────────────────────────────────────────────────────

def launch_sitl() -> subprocess.Popen:
    """Start ArduPilot SITL in a subprocess."""
    cmd = [
        sys.executable, SIM_VEHICLE,
        "--vehicle", "ArduCopter",
        "--no-rebuild",          # skip compilation check for speed
        "--speedup", "1",
        "-L", "KSFO",            # San Francisco airport (sensible default)
        "--console",             # open the HUD console
        "--map",                 # open map window
        "--out", f"tcpin:{SITL_TCP_HOST}:{SITL_TCP_PORT}",
    ]
    print(f"\n[SITL] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=ARDUPILOT_ROOT)
    return proc


def launch_mavproxy() -> subprocess.Popen:
    """Start MAVProxy bridging SITL → MAAS-LLM UDP port."""
    # Primary output → MAAS-LLM
    # Optional: add --out udp:127.0.0.1:14551 for a GCS like QGroundControl
    cmd = [
        "mavproxy.py",
        f"--master=tcp:{SITL_TCP_HOST}:{SITL_TCP_PORT}",
        f"--out=udp:{MAVPROXY_OUT_HOST}:{MAVPROXY_OUT_PORT}",
        "--out=udp:127.0.0.1:14551",
        "--daemon",              # run headless (no console)
        "--non-interactive",
        "--baudrate=115200",
    ]
    print(f"\n[MAVProxy] Starting bridge: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    return proc


# ──────────────────────────────────────────────────────────────────────────────
# Manual instructions
# ──────────────────────────────────────────────────────────────────────────────

def print_manual_commands() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           MAAS-LLM × ArduPilot SITL — Manual Start Guide           ║
╠══════════════════════════════════════════════════════════════════════╣

Step 1 — Start ArduPilot SITL (Terminal 1)
──────────────────────────────────────────
  cd ~/ardupilot
  python Tools/autotest/sim_vehicle.py \\
      --vehicle ArduCopter \\
      --no-rebuild \\
      --map \\
      --console

  Wait for: "APM: EKF2 IMU0 is using GPS" (takes ~30 s first boot)

Step 2 — Start MAVProxy bridge (Terminal 2)
───────────────────────────────────────────
  mavproxy.py \\
      --master tcp:127.0.0.1:5762 \\
      --out udp:127.0.0.1:14550

  This forwards raw MAVLink bytes to MAAS-LLM on UDP 14550.

  NOTE: MAAS-LLM listens on TELEMETRY_IN_PORT (default 9000).
  If you change that env var, update --out udp:127.0.0.1:PORT accordingly.

Step 3 — Start MAAS-LLM in SITL mode (Terminal 3)
──────────────────────────────────────────────────
  cd /home/cipher/MAAS/Maas-LLM
  TELEMETRY_IN_PORT=14550 python -m src.main --mode sitl

  Or to receive on port 9000 and bridge from port 14550 via loopback:
  TELEMETRY_IN_PORT=9000 python -m src.main --mode sitl
  (and set MAVProxy --out udp:127.0.0.1:9000)

Step 4 — Verify telemetry is flowing
──────────────────────────────────────────────────
  You should see MAAS-LLM print telemetry snapshots in the logs.
  After ~10 s the SITLInjector starts firing synthetic anomaly events.
  In DRY_RUN=1 mode (default for SITL) each command requires [y/N] approval.

To shift to PRODUCTION (real Golang backend):
──────────────────────────────────────────────────
  Replace 127.0.0.1 in MAVProxy --out with the Golang backend's ZeroTier IP.
  MAAS-LLM code requires zero changes.

╚══════════════════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MAAS-LLM SITL Bridge Launcher")
    parser.add_argument("--no-sitl",    action="store_true", help="Skip SITL launch (already running)")
    parser.add_argument("--print-manual", action="store_true", help="Print manual start commands")
    parser.add_argument("--check",      action="store_true", help="Check dependencies only")
    args = parser.parse_args()

    if args.print_manual:
        print_manual_commands()
        return

    print("\n🛸 MAAS-LLM SITL Bridge — Dependency Check\n")
    deps_ok = check_dependencies()

    if args.check or not deps_ok:
        sys.exit(0 if deps_ok else 1)

    processes: list[subprocess.Popen] = []

    try:
        if not args.no_sitl:
            sitl = launch_sitl()
            processes.append(sitl)
            print("\n[Bridge] Waiting 15 s for SITL to initialize...")
            time.sleep(15)

        mavproxy = launch_mavproxy()
        processes.append(mavproxy)

        print(f"""
✅ Bridge running!

MAVProxy is forwarding SITL → UDP {MAVPROXY_OUT_HOST}:{MAVPROXY_OUT_PORT}

Now start MAAS-LLM in SITL mode (in a separate terminal):

  cd /home/cipher/MAAS/Maas-LLM
  TELEMETRY_IN_PORT={MAVPROXY_OUT_PORT} python -m src.main --mode sitl

Press Ctrl+C to stop the bridge.
""")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Bridge] Shutting down...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("[Bridge] All processes stopped.")


if __name__ == "__main__":
    main()
