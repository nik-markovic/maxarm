// The red cube's points, which refine the camera calibration (tests/api.py): their table, what SAVE CALIBRATION
// would change, and the place where the next point would add most.
//
// Each point, and each of the camera page's three cubes, is off by how far the camera puts the cube's centre from
// where the arm says the cup was: through the saved calibration ("now"), and through the one SAVE would write.

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

// What a point is coloured by: how far off it is after SAVE, or now while SAVE cannot be done.
export const offShown = (row) => (row.off.after === null ? row.off.now : row.off.after);

export function tableRows(refine, offLimits) {
  const cell = (off) => `<td class="num dev ${rank(off, offLimits)}">${fmt(off, 2)}</td>`;
  const row = (point, tail = '') => `
    <td class="num">${fmt(point.x)}</td><td class="num">${fmt(point.y)}</td>
    <td class="num faint">${fmt(point.seen[0])}</td><td class="num faint">${fmt(point.seen[1])}</td>
    ${cell(point.off.now)}${cell(point.off.after)}<td class="actions">${tail}</td>`;
  const cubes = refine.cubes.map((cube) => `<tr class="cube" title="The ${cube.colour} cube, from the camera page">
    <td>${cubeIcon(cube.colour)}</td>${row(cube)}</tr>`);
  const points = refine.points.map((point) => `<tr title="Recorded ${point.taken}">
    <td><span class="badge ${rank(offShown(point), offLimits)}">${point.id}</span></td>
    ${row(point, `<button class="btn small ghost" data-id="${point.id}" title="Forget it">×</button>`)}</tr>`);
  const empty = refine.points.length ? [] : ['<tr class="empty"><td colspan="8">No red cube points yet: FIND RED CUBE, '
    + 'centre the cube under the cup, RECORD POINT.</td></tr>'];
  return [...cubes, ...points, ...empty].join('');
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
  return `After SAVE the points are off by <b>${fmt(fit.off_mm[0])}</b> mm typically, <b>${fmt(fit.off_mm[1])}</b>
    at worst. ${change} ${fit.arm[0].toUpperCase()}${fit.arm.slice(1)}.`;
}

export function savedLine(refine) {
  const { refined } = refine;
  return refined ? `camera.json is refined by ${refined.points} points.`
    : 'camera.json is the three cubes\' calibration alone.';
}

// Where the red cube would add most next: the place in reach, whole in view, furthest from every point and cube.
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
