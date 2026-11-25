#!/usr/bin/env bash
# The language model on the Ara240, for the app's `agent` command: NXP's eIQ AAF Connector installed in this
# directory, and Qwen2.5-7B-Instruct for it. On the board, as root, once:
#
#   connector/install.sh
#
# Optional: the app runs without it. ../install.sh runs this, and install-service.sh after it, when it finds an
# Ara240 in the board; with a card put in later, run both by hand. Gigabytes, and many minutes. Delete venv/ to
# install again from scratch.
#
# What the connector needs comes with the image (LF_v6.18.37-2.1.0): the Ara240's runtime and its proxy
# (ara2-proxy.service), and the connector's and optimum-ara's wheels in /usr/share/python-wheels. This adds venv/
# here -- torch, transformers and the rest from PyPI -- and the model, 11 GB from Hugging Face, where the image's own
# ara2-fetch-models puts it: /usr/share/llm/Qwen2.5-7B-Instruct, the path the model's config.json names. The image's
# /usr/bin/aaf-connector is not used: it installs into /usr/share/eiq and writes a service into /etc/systemd.
set -e
cd "$(dirname "$(readlink -f "$0")")"

ls /usr/share/python-wheels/eiq_aaf_connector-*.whl /usr/share/python-wheels/optimum_ara-*.whl > /dev/null

if [ ! -d venv ]; then
  # Without --system-site-packages: the connector brings its own torch, numpy and opencv, which must not mix with the
  # image's that the app runs on.
  python3 -m venv venv
fi

echo "Installing the connector and its packages into connector/venv. This takes several minutes..."
# optimum-ara first: the connector requires it, and it is not on PyPI.
venv/bin/pip install --disable-pip-version-check /usr/share/python-wheels/optimum_ara-*.whl
venv/bin/pip install --disable-pip-version-check /usr/share/python-wheels/eiq_aaf_connector-*.whl

./apply-patch.sh

echo "Fetching Qwen2.5-7B-Instruct for the Ara240 (11 GB) into /usr/share/llm; a re-run checks what is there..."
ara2-fetch-models --repo-id nxp/Qwen2.5-7B-Instruct-Ara240

echo
echo "Done. Start it at every boot: connector/install-service.sh; or by hand, after each boot: connector/run.sh"
