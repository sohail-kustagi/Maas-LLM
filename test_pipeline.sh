#!/bin/bash
pkill -f sim_vehicle.py
pkill -f mavproxy
pkill -f mock_backend.py
pkill -f arducopter

echo "Starting SITL..."
python3 ~/ardupilot/Tools/autotest/sim_vehicle.py --vehicle ArduCopter --no-rebuild --out 127.0.0.1:14550 --out 127.0.0.1:14551 > sitl.log 2>&1 &
sleep 20

echo "Starting Mock Backend..."
./venv/bin/python scripts/mock_backend.py > mock.log 2>&1 &
sleep 5

echo "Starting Pipeline..."
DRY_RUN=0 TELEMETRY_IN_PORT=14550 ./venv/bin/python -m src.main --mode sitl > pipeline.log 2>&1 &
sleep 60

echo "Done."
