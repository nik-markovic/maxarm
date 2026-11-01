#!/usr/bin/env python3
"""Does the C920's lens bend straight lines? A sheet of paper's four borders, found to a fraction of a pixel in a
frame: each should be straight. Fitted: the one radial bend k1 (round the frame's middle, at the cubes' focal length,
OpenCV's sign: negative is barrel) that straightens all four best. No arm; camera.json is not used.

    .venv/bin/python impala/agenttools/lens/straight.py FRAME.png [focal-px] [OUT.jpg]

Prints each border: its length, how far its points lie from the straight line through them (rms, worst), and its bow
(the middle's distance from the chord through its ends, + away from the frame's middle), as found and with k1 undone.
"""

import math
import sys

import cv2
import numpy as np

ACROSS_PX = 14                       # how far either side of the border the profile looks
STEP_PX = 6                          # one border point every this many pixels
END_SHARE = 0.04                     # left out at each end: the corners


def paper_corners(gray):
    """The sheet: the largest bright patch, its outline as four corners, in order round it."""
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (0, 0), 2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    sheet = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contour = max(cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    corners = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)[:, 0, :].astype(float)
    if len(corners) != 4:
        raise SystemExit(f"the sheet's outline has {len(corners)} corners, not 4")
    return corners


def border_points(gray, start, end):
    """Points on the border from `start` to `end`: at each, the strongest step across it, to a fraction of a pixel."""
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    along = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-along[1], along[0]])
    length = np.linalg.norm(end - start)
    offsets = np.arange(-ACROSS_PX, ACROSS_PX + 0.01, 0.25)
    points = []
    for distance in np.arange(END_SHARE * length, (1 - END_SHARE) * length, STEP_PX):
        centre = start + distance * along
        line = centre + offsets[:, None] * normal
        profile = cv2.remap(smooth, line[:, :1].astype(np.float32), line[:, 1:].astype(np.float32),
                            cv2.INTER_LINEAR)[:, 0]
        slope = np.abs(np.gradient(profile))
        peak = int(np.argmax(slope))
        if not 2 <= peak < len(slope) - 2:
            continue
        before, here, after = slope[peak - 1:peak + 2]
        shift = 0.5 * (before - after) / (before - 2 * here + after) if before - 2 * here + after else 0.0
        points.append(centre + (offsets[peak] + shift * 0.25) * normal)
    return np.array(points)


def straightness(points, middle):
    """rms and worst distance from the straight line through `points`, and the bow: + away from `middle`."""
    mean = points.mean(axis=0)
    along = np.linalg.svd(points - mean)[2][0]
    normal = np.array([-along[1], along[0]])
    if np.dot(mean - middle, normal) < 0:
        normal = -normal                                  # + points away from the frame's middle
    across = (points - mean) @ normal
    position = (points - mean) @ along
    ends = np.array([position.min(), position.max()])
    curve = np.polyfit(position, across, 2)
    chord = np.polyval(curve, ends)
    bow = np.polyval(curve, ends.mean()) - chord.mean()
    return math.sqrt(np.mean(across ** 2)), np.abs(across).max(), bow


def undone(points, lens, k1):
    """The points as a lens without the bend k1 would have put them."""
    straight = cv2.undistortPoints(points.reshape(-1, 1, 2), lens, np.array([k1, 0, 0, 0]))[:, 0, :]
    return straight * lens[0, 0] + lens[:2, 2]


def main():
    frame = cv2.imread(sys.argv[1])
    focal = float(sys.argv[2]) if len(sys.argv) > 2 else 1670.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    middle = np.array([(width - 1) / 2, (height - 1) / 2])
    lens = np.array([[focal, 0, middle[0]], [0, focal, middle[1]], [0, 0, 1]])
    corners = paper_corners(gray)
    borders = [border_points(gray, corners[index], corners[(index + 1) % 4]) for index in range(4)]

    def cost(k1):
        return sum(straightness(undone(points, lens, k1), middle)[0] ** 2 * len(points) for points in borders)

    trials = np.linspace(-0.4, 0.4, 161)
    best = trials[int(np.argmin([cost(k1) for k1 in trials]))]
    for _ in range(20):                                  # golden-section, round the best of the grid
        low, high = best - 0.005, best + 0.005
        while high - low > 1e-5:
            first, second = high - 0.618 * (high - low), low + 0.618 * (high - low)
            low, high = (low, second) if cost(first) < cost(second) else (first, high)
        best = (low + high) / 2
        break
    print(f"{width} x {height}, focal {focal:.0f} px; corners {np.round(corners).astype(int).tolist()}")
    print(f"the bend that straightens the four best: k1 {best:+.4f}")
    print("  border       points  length px   as found: rms  worst  bow     k1 undone: rms  worst  bow")
    for index, points in enumerate(borders):
        found = straightness(points, middle)
        fixed = straightness(undone(points, lens, best), middle)
        length = np.linalg.norm(corners[(index + 1) % 4] - corners[index])
        print(f"  {index}: {np.round(corners[index]).astype(int)}  {len(points):4d}  {length:7.0f}       "
              f"{found[0]:5.2f} {found[1]:5.2f} {found[2]:+6.2f}          {fixed[0]:5.2f} {fixed[1]:5.2f} {fixed[2]:+6.2f}")
    for k1 in (-0.08, -0.04, 0.0, 0.04):
        print(f"  k1 {k1:+.2f}: bows " + ", ".join(f"{straightness(undone(points, lens, k1), middle)[2]:+6.2f}"
                                               for points in borders))
    if len(sys.argv) > 3:
        drawn = frame.copy()
        for points in borders:
            for point in points:
                cv2.circle(drawn, (int(round(point[0])), int(round(point[1]))), 2, (255, 0, 255), -1)
        cv2.imwrite(sys.argv[3], drawn, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
