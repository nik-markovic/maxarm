#!/usr/bin/env python3
"""What the reader sees: each tile's face in its four quarter turns, the chosen one boxed green.

    ./agenttools/dump-crops.py training/tiles/baseline/tiles-1.png  # -> files/tiles-1-crops.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import glyphs                           # noqa: E402
import pose                             # noqa: E402
import read                             # noqa: E402
import scene                            # noqa: E402
import topdown                          # noqa: E402
from mapping import DeskMapping         # noqa: E402

ZOOM = 2


def main() -> int:
    path = Path(sys.argv[1])
    frame = cv2.imread(str(path))
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    reader = read.Reader()
    view = topdown.render(frame, mapping, scene.TILE_HEIGHT_MM)
    field = pose.EdgeField(view, mapping.camera_position())
    rows = []
    for glyph in glyphs.find_glyphs(view):
        fit = pose.fit_tile(field, view.to_ground(glyph.centre))
        faces = scene._faces(frame, mapping, fit)
        reading = reader.read(faces)
        cells = []
        for turn, face in enumerate(faces):
            cell = cv2.cvtColor(cv2.resize(read.normalise(face), None, fx=ZOOM, fy=ZOOM,
                                           interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            colour = (0, 200, 0) if turn == reading.turn else (60, 60, 60)
            cells.append(cv2.copyMakeBorder(cell, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=colour))
        label = np.zeros((cells[0].shape[0], 90, 3), np.uint8)
        cv2.putText(label, f"{reading.letter} {reading.confidence:.2f}", (6, cells[0].shape[0] // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(cells + [label]))
    out = ROOT / "files" / (path.stem + "-crops.png")
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"{len(rows)} tiles -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
