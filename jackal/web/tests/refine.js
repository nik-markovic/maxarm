// The points which refine the camera calibration (tests/api.py) -- CALIBRATE's three cubes, 1 to 3 until forgotten,
// then the red cube's: their table, what SAVE CALIBRATION would change, and the place where the next point would
// add most.
//
// Each point is off by how far the camera puts the cube's centre from where the arm says the cup was, across the
// desk (XY), and by how far from 40 mm tall it sees the cube (Z): through the three cubes' calibration, CALIBRATE's
// ("calibrate"), and through the one SAVE would write, the camera fitted to the points. Z is none for a point kept
// before the cube's edges were.

import { fmt } from '/common/page.js';
import { cubeIcon } from '/common/height.js';
import { toView } from '/common/shares.js';

// The places looked at for the next point, desk mm apart; and how close to the view's edge a cube may stand there,
// as a share of it.
const NEXT_STEP_MM = 20;
const NEXT_VIEW_MARGIN = 0.04;

// 'ok', 'warn' or 'bad', by `limits.tests.refine.off`, as on the desk page; '' for nothing.
export function rank(off, [warn, bad]) {
  if (off === null || off === undefined) return '';
  return off <= warn ? 'ok' : off <= bad ? 'warn' : 'bad';
}

// What a point is coloured by: how far off it is after SAVE, or through CALIBRATE's while SAVE cannot be done.
export const offShown = (row) => (row.off.after === null ? row.off.calibrate : row.off.after);

export function tableRows(refine, offLimits) {
  const cell = (off) => `<td class="num dev ${rank(off, offLimits)}">${fmt(off, 2)}</td>`;
  const heightCell = (off) => `<td class="num dev ${rank(off === null ? null : Math.abs(off), offLimits)}">${
    fmt(off, 2, true)}</td>`;
  const row = (point, tail = '') => `
    <td class="num">${fmt(point.x)}</td><td class="num">${fmt(point.y)}</td>
    <td class="num faint">${fmt(point.seen[0])}</td><td class="num faint">${fmt(point.seen[1])}</td>
    ${cell(point.off.calibrate)}${heightCell(point.height.calibrate)}
    ${cell(point.off.after)}${heightCell(point.height.after)}<td class="actions">${tail}</td>`;
  const points = refine.points.map((point) => `<tr class="${point.cube ? 'cube' : ''}" title="${point.cube
    ? `The ${point.cube} cube, recorded on the camera page and seen in CALIBRATE's frame, ${point.taken}`
    : `Recorded ${point.taken}`}">
    <td><span class="badge ${rank(offShown(point), offLimits)}">${point.id}</span>${point.cube ? cubeIcon(point.cube)
      : ''}</td>
    ${row(point, `<button class="btn small ghost" data-id="${point.id}" title="Forget it">×</button>`)}</tr>`);
  const empty = refine.points.length ? [] : ['<tr class="empty"><td colspan="10">No points: FIND RED CUBE, '
    + 'centre the cube under the cup, RECORD POINT.</td></tr>'];
  return [...points, ...empty].join('');
}

// What SAVE CALIBRATION would do, or why it cannot.
export function fitLine(refine) {
  const { fit, problem } = refine;
  if (!fit) return `<span class="note">${problem}</span>`;
  const change = fit.change_mm
    ? `It puts the desk <b>${fmt(fit.change_mm[0])}</b> mm from the saved calibration, <b>${fmt(fit.change_mm[1])}</b>
      at worst, in reach and in view.` : '';
  if (fit.is_cubes_alone) {
    return `No points: SAVE puts back the calibration of the three cubes alone. ${change}`;
  }
  const height = fit.height_mm ? ` Their cubes' tops are <b>${fmt(fit.height_mm[0])}</b> mm from 40 typically,
    <b>${fmt(fit.height_mm[1])}</b> at worst.` : '';
  return `After SAVE the arm's readings are <b>${fmt(fit.off_mm[0])}</b> mm typically, <b>${fmt(fit.off_mm[1])}</b>
    at worst, from where the other points put each (XY).${height} ${change} ${armLine(fit.arm)}`;
}

// How the arm measures against the camera: information, not counted against any point.
function armLine(arm) {
  if (!arm) return '';
  const [low, high] = arm.stretch.map((scale) => 100 * (scale - 1));
  const apart = low >= 0 ? `${fmt(low, 1)} to ${fmt(high, 1)}% longer`
    : high <= 0 ? `${fmt(-high, 1)} to ${fmt(-low, 1)}% shorter`
      : `${fmt(-low, 1)}% shorter to ${fmt(high, 1)}% longer`;
  return `<span class="note">The camera measures the arm's distances ${apart}, its x and y ${fmt(arm.skew_deg, 1)}
    deg off square.</span>`;
}

export function savedLine(refine) {
  const { refined } = refine;
  return refined ? `camera.json's camera is fitted to ${refined.points} points.`
    : 'camera.json is the three cubes\' calibration alone.';
}

// Where the red cube would add most next: the place in reach, whole in view, furthest from every point.
// Null once every such place is within two cubes' width of one.
export function nextSpot(rows, limits, matrix) {
  if (!matrix) return null;
  const { reach, reach_min: reachMin, fan } = limits;
  const size = limits.tests.find.cube;
  let best = null;
  for (let x = -reach; x <= reach; x += NEXT_STEP_MM) {
    for (let y = -reach; y <= reach; y += NEXT_STEP_MM) {
      const radius = Math.hypot(x, y);
      const bearing = (Math.atan2(x, -y) * 180) / Math.PI;
      if (radius < reachMin + size || radius > reach - size / 2 || Math.abs(bearing) > fan - 10) continue;
      if (!isWholeInView(matrix, x, y, size)) continue;
      const gap = Math.min(...rows.map((row) => Math.hypot(row.x - x, row.y - y)));
      if (!best || gap > best.gap) best = { x, y, gap };
    }
  }
  return best && best.gap >= 2 * size ? best : null;
}

function isWholeInView(matrix, x, y, size) {
  const half = size / 2;
  for (const z of [0, size]) {
    for (const [dx, dy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const share = toView(matrix, [x + dx * half, y + dy * half, z]);
      if (!share || share.some((value) => value < NEXT_VIEW_MARGIN || value > 1 - NEXT_VIEW_MARGIN)) return false;
    }
  }
  return true;
}
