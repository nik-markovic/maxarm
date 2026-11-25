#!/usr/bin/env bash
# The tiles demo, on the board, in the venv install.sh made: the calibration pages on http://<board>:8080, and
# /IOTCONNECT -- commands, telemetry, the stream -- once the camera is calibrated and the device's files are in
# config/. Ctrl-C to stop.
#
#   ./run.sh                # the pages, /IOTCONNECT, the console
#   ./run.sh --no-iotc      # without the cloud
#   ./run.sh --no-webapp    # without the pages
set -e
cd "$(dirname "$(readlink -f "$0")")"

# What the image brings, not pip (src/requirements.txt): stop here rather than part-way through.
venv/bin/python -c "import numpy, cv2, tflite_runtime, gi"

# The CPUs at full speed, so the camera, the tiles and the stream keep up; 'ondemand' again after a reboot.
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > $governor
done

export PYTHONUNBUFFERED=1
exec venv/bin/python src/main.py --host :: "$@"
