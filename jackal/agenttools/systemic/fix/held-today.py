#!/usr/bin/env python3
"""No arm, one frame: the lens's numbers (focal length, middle) from today's grid sheets alone, held; where the
camera stands and which way it looks from today's three cubes alone; then the sheets read through that camera.
Against the cubes alone with everything free (the app's way).

    held-today.py
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tj", HERE / "today-joint.py")
tj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj)

state = pickle.load(open("/tmp/today-joint.pkl", "rb"))
tj.FLIP[:] = state["flip"]
sheets_alone = state["sheets_alone"]
start = state["both"].copy()
for name, held in (("cubes alone, all free (the app's way)", None), ("cubes alone, lens held at the sheets' numbers",
                                                                     sheets_alone[:3])):
    x = start.copy()
    free = list(range(tj.CAM)) + list(range(tj.CUBES, tj.SHEET0))
    if held is not None:
        x[:3] = held
        free = list(range(3, tj.CAM)) + list(range(tj.CUBES, tj.SHEET0))
    x = tj.solve(lambda z: np.concatenate([tj.edge_res(z), tj.held(z)]), x, free)
    x = tj.sheet_poses(x)
    tj.report(name, x)
