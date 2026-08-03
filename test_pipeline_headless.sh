#!/bin/bash
pkill -f sim_vehicle.py
pkill -f mavproxy
pkill -f mock_backend.py
pkill -f arducopter

echo "Starting Headless SITL..."
# Run WITHOUT --map and --console to avoid X11 crashes in the agent terminal
python3 ~/ardupilot/Tools/autotest/sim_vehicle.py --vehicle ArduCopter --no-rebuild --out 127.0.0.1:14550 --out 127.0.0.1:14551 > sitl.log 2>&1 &
sleep 25

echo "Starting Mock Backend..."
./venv/bin/python -u scripts/mock_backend.py > mock.log 2>&1 &
sleep 5

echo "Starting Pipeline..."
DRY_RUN=0 TELEMETRY_IN_PORT=14550 ./venv/bin/python -u -m src.main --mode sitl > pipeline.log 2>&1 &
sleep 45

echo "Done."
