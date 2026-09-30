#!/usr/bin/env bash
# Rebuild files/reader.tflite, the tile reader, from scratch: this directory and the network.
#
#   ./training/tiles/make-reader.sh
#
# 1. Fonts: google/fonts at the commit in fonts.txt, only the families listed
#    there, into fonts/ here, where render.py looks. About 64 MB, not committed.
# 2. Reader: train.py -- 240k synthetic tiles from fixed seeds, 14 epochs, the
#    best one exported as int8 -- about 15 minutes on 16 cores. It also writes
#    files/reader.json: the fonts' hash, the versions, the accuracy, the model's hash.
# 3. Check: check-reader.py on the scenes in baseline/, with the calibration
#    they were taken with. No camera, and no calibration of your own, needed.
#
# The Python it runs is the repository's venv, ../.venv, or $PYTHON; its
# packages are requirements.txt here:
#
#   python3.12 -m venv ../.venv && ../.venv/bin/pip install -r training/tiles/requirements.txt
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
python="${PYTHON:-$root/../.venv/bin/python}"
fonts="$here/fonts"
commit="$(awk '$1 == "commit" { print $2 }' "$here/fonts.txt")"
mapfile -t families < <(grep -v -e '^#' -e '^commit ' -e '^$' "$here/fonts.txt")

if [ ! -d "$fonts/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/google/fonts "$fonts"
fi
if ! git -C "$fonts" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$fonts" fetch --filter=blob:none origin "$commit"
fi
git -C "$fonts" sparse-checkout set "${families[@]}"
git -C "$fonts" -c advice.detachedHead=false checkout --detach "$commit"

"$python" -u "$here/train.py"
"$python" "$here/check-reader.py"
