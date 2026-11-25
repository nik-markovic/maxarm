#!/usr/bin/env bash
# Start the demo (../run.sh) at every boot, as the systemd service tiles, from where ../install.sh installed it.
# As root:
#
#   scripts/install-service.sh            install it, and start it now
#   scripts/install-service.sh --remove   stop it, and take it out of /etc/systemd/system
#
# Logs: journalctl -u tiles -f. Again after moving this directory: the service names its path.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

UNIT=/etc/systemd/system/tiles.service

# The two apps' own services, from before they were one.
for old in maxarm-calibration maxarm-app; do
  if [ -f "/etc/systemd/system/$old.service" ]; then
    systemctl disable --now "$old.service"
    rm "/etc/systemd/system/$old.service"
  fi
done

if [ "$1" = "--remove" ]; then
  systemctl disable --now tiles.service 2> /dev/null || true
  rm -f "$UNIT"
  systemctl daemon-reload
  echo "Removed the service tiles. Start the demo by hand: ./run.sh"
  exit 0
fi

sed "s|@DIR@|$PWD|g" scripts/systemd/tiles.service > "$UNIT"
systemctl daemon-reload
systemctl enable tiles.service
systemctl restart tiles.service
echo "The service tiles is starting, and will at every boot. Logs: journalctl -u tiles -f"
