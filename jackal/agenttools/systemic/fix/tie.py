"""The turn and shift the app laid the arm on the camera with until 4 Nov (fit.link_arm, since removed): kept for these
research scripts' comparisons."""
import numpy as np

HANDEDNESS = np.diag([-1.0, 1.0])


def link_arm(desk, true):
    desk = np.asarray(desk, float) @ HANDEDNESS.T
    true = np.asarray(true, float)
    left, _, right = np.linalg.svd((desk - desk.mean(0)).T @ (true - true.mean(0)))
    turn = (left @ np.diag([1.0, np.sign(np.linalg.det(left @ right))]) @ right).T
    return turn, true.mean(0) - turn @ desk.mean(0), HANDEDNESS.copy()
