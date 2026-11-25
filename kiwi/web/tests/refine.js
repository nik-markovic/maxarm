// The points which refine the camera calibration (tests/api.py) -- CALIBRATE's three cubes, 1 to 3 until forgotten,
// then the red cube's: their table, what SAVE CALIBRATION would change, and the place where the next point would
// add most.
//
// Each point is off by how far the camera puts the cube's center from where the arm says the cup was, across the
// desk (XY), and by how far from its real height it sees the cube (Z), through the saved calibration ("now"); and,
// without it, by how far the correction of the other points puts its reading from its cube ("without"). A point that
// does not fit its neighbors is left out of the correction at SAVE.

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

// What a point is colored by: how far off it is now, through the saved calibration; red if it does not fit the others.
export const offShown = (row) => (row.fits === false ? Infinity : row.off.now);

export function tableRows(refine, offLimits) {
  const cell = (off) => `<td class="num dev ${rank(off, offLimits)}">${fmt(off, 2)}</td>`;
  const heightCell = (off) => `<td class="num dev ${rank(off === null ? null : Math.abs(off), offLimits)}">${
    fmt(off, 2, true)}</td>`;
  const row = (point, tail = '') => `
    <td class="num">${fmt(point.x)}</td><td class="num">${fmt(point.y)}</td>
    <td class="num faint">${fmt(point.seen[0])}</td><td class="num faint">${fmt(point.seen[1])}</td>
    ${cell(point.off.now)}${heightCell(point.height.now)}
    ${cell(point.off.without)}<td class="actions">${tail}</td>`;
  const points = refine.points.map((point) => `<tr class="${point.cube ? 'cube' : ''}" title="${point.cube
    ? `The ${point.cube} cube, recorded on the camera page, ${point.taken}`
    : `Recorded ${point.taken}`}${point.fits === false ? '. Does not agree with the other points: SAVE leaves it '
      + 'out; remove it and record it again' : ''}">
    <td><span class="badge ${rank(offShown(point), offLimits)}">${point.id}</span>${point.cube ? cubeIcon(point.cube)
      : ''}</td>
    ${row(point, `<button class="btn small ghost" data-id="${point.id}" title="Remove it">×</button>`)}</tr>`);
  const empty = refine.points.length ? [] : ['<tr class="empty"><td colspan="9">No points: bring the cup over the '
    + 'red cube, center the cube under it, RECORD POINT.</td></tr>'];
  return [...points, ...empty].join('');
}

// What SAVE CALIBRATION would do, or why it cannot.
export function fitLine(refine) {
  const { fit, problem } = refine;
  if (!fit) return `<span class="note">${problem}</span>`;
  if (fit.is_cubes_alone) return 'No points: SAVE goes back to the calibration of the three cubes alone.';
  const misfits = fit.does_not_fit.map(({ id, mm }) => `<b>Point ${id} does not agree</b> with the others, by
    ${fmt(mm)} mm: SAVE leaves it out. Remove it and record it again.`).join(' ');
  const without = fit.off_mm ? `Between points the calibration is off by <b>${fmt(fit.off_mm[0])}</b> mm
    typically, <b>${fmt(fit.off_mm[1])}</b> at worst.` : '';
  const change = fit.change_mm ? `SAVE moves it by <b>${fmt(fit.change_mm[0])}</b> mm typically,
    <b>${fmt(fit.change_mm[1])}</b> at worst.` : '';
  return [misfits, without, change].filter(Boolean).join(' ');
}

// How the arm measures against the camera: information, not counted against any point.
export function armLine(arm) {
  if (!arm) return '';
  const [low, high] = arm.stretch.map((scale) => 100 * (scale - 1));
  const apart = low >= 0 ? `${fmt(low, 1)} to ${fmt(high, 1)}% longer`
    : high <= 0 ? `${fmt(-high, 1)} to ${fmt(-low, 1)}% shorter`
      : `${fmt(-low, 1)}% shorter to ${fmt(high, 1)}% longer`;
  return `The camera measures the arm's distances ${apart}, its x and y ${fmt(arm.skew_deg, 1)} deg off square.`;
}

// Which calibration is in use, unless the points changed since it was saved (SAVE says so).
export function savedLine(refine) {
  if (refine.saved === false) return '';
  const { refined } = refine;
  return refined ? `In use: the calibration of ${refined.points} points.`
    : 'In use: the calibration of the three cubes alone.';
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
