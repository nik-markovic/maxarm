#!/usr/bin/env bash
# Install the tiles demo on the FRDM-IMX95, as root, where its package was unpacked (README.md):
#
#   ./install.sh
#
# Again after unpacking a newer package over it: config/ -- the calibration, the device's identity -- is not in the
# package and stays. Then ./run.sh. With an Ara240 in the board, its language model too (connector/), started at
# every boot; without one, that is left out, and connector/install.sh adds it once the card is in.
set -e
cd "$(dirname "$(readlink -f "$0")")"

scripts/install-ch341.sh

if [ ! -d venv ]; then
  echo "Creating the Python Virtual environment in venv/. This may take some time..."
  # --system-site-packages: the image's numpy, cv2, tflite_runtime and gi, never pip's (src/requirements.txt).
  python3 -m venv --system-site-packages venv
fi
. venv/bin/activate
scripts/install.sh
deactivate

# The Ara240 is on PCIe, vendor 0x1e58.
if grep -qx 0x1e58 /sys/bus/pci/devices/*/vendor; then
  echo
  echo "An Ara240 is in the board: installing its language model (connector/)."
  connector/install.sh
  connector/install-service.sh
else
  echo
  echo "No Ara240 in the board: the agent command and the Chat page stay off."
  echo "With one later: connector/install.sh, then connector/install-service.sh."
fi

echo
echo "Use this certificate to register the device with /IOTCONNECT (config/device-cert.pem):"
cat config/device-cert.pem
echo "Then download the device's iotcDeviceConfig.json into config/."
echo
echo "Done. Start the demo: ./run.sh -- the calibration pages on http://<board>:8080, and /IOTCONNECT once the"
echo "camera is calibrated and the device's files are in config/ (start it again after either)."
echo "At every boot: scripts/install-service.sh"
