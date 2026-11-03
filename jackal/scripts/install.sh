#!/usr/bin/env bash
# Install what both apps need, here, from this source tree: a venv with src/requirements.txt in it, and the
# device's certificate and key in config/ if it has none yet. Run it from anywhere, once:
#
#   scripts/install.sh                      # the venv at .venv, beside src/
#   VENV=../.venv scripts/install.sh        # another venv, made if it is not there
#
# The TFLite interpreter is not installed: on the i.MX95 it is the BSP's tflite_runtime, which the venv sees; on a
# PC, `pip install ai-edge-litert` into the venv.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  # --system-site-packages: the BSP's own numpy, cv2 and tflite_runtime stay visible on the board.
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install -r src/requirements.txt

if ! "$VENV/bin/python" -c "import tflite_runtime" 2>/dev/null \
    && ! "$VENV/bin/python" -c "import ai_edge_litert" 2>/dev/null \
    && ! "$VENV/bin/python" -c "import tensorflow" 2>/dev/null; then
  echo "NOTE: no TFLite interpreter, so no letters can be read. On a PC: $VENV/bin/pip install ai-edge-litert"
fi

if [ ! -f config/device-pkey.pem ]; then
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 3650 \
    -keyout config/device-pkey.pem -out config/device-cert.pem -subj "/CN=maxarm"
  echo "Made config/device-cert.pem: paste it as the device's certificate when you create the device in /IOTCONNECT."
fi
if [ ! -f config/iotcDeviceConfig.json ]; then
  echo "Next: download iotcDeviceConfig.json from the device's page in /IOTCONNECT into config/."
fi
echo "Done. Calibrate: $VENV/bin/python src/calibration/calibration.py; then run: $VENV/bin/python src/main.py"
