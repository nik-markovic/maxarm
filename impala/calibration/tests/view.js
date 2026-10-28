// The live view's marks, drawn over the camera's picture through the calibration (common/shares.js):
// - guides: the reach on the desk, 95 and 270 mm, orange and red as on the grid; the arm's target, the cup's rim
//   there, faint green, with a line down to the desk under it. Not where the servos say the cup is: drawn through the
//   camera, that mixes the camera's error with the arm's (the owner's, PILOT-impala);
// - the red cube as FIND RED CUBE found it: its edges in pink, fainter where hidden, an X under its centre, and its
//   desk x, y over it, clear of the cup's end;
// - ARM TO DESK's point: a crosshair where the view was clicked, half see-through, the cup's rim round the point on
//   the desk, and its desk x, y;
// - with the guides, the red cube's points: each a small ring, numbered, where the camera saw the cube's centre,
//   coloured by how far off it is (refine.js); and a dashed circle on the desk, a place for the next point.
// The cup's rim is drawn its true size (tests/api.py). The drawing itself, and the reach and the arm, are the shared
// ViewMarks' (common/marks.js).

import { INK as SHARED_INK, MONO, ViewMarks } from '/common/marks.js';
import { circle, toDesk } from '/common/shares.js';
import { offShown, rank } from './refine.js';

// The ring under the pointer is blue, to see against a pale desk (the owner's).
const INK = { ...SHARED_INK, cube: '#ff4df8', red: '#ff4d5e', pointer: '#3d9bff' };
const RANK_INK = { ok: INK.green, warn: '#e6b84a', bad: INK.reach, '': '#8d9aa9' };
// The found cube's coordinates are written this far over the cup's end, which comes down to 4 over the cube's top,
// and at least so many pixels: the cup's end stays in sight.
const LABEL_OVER_CUP_MM = 16;
const LABEL_OVER_CUP_PX = 18;
export class LiveMarks extends ViewMarks {
  // found: the red cube; touch: { u, v, desk }; hover: [u, v]; showGuides; refine: the red cube's points and the
  // cubes, as the tests' state has them; next: { x, y }, desk mm.
  constructor(canvas) {
    super(canvas, { found: null, touch: null, hover: null, showGuides: true, refine: null, next: null });
  }

  drawMarks() {
    if (this.marks.showGuides) {
      this.drawReach();
      this.drawNext();
      this.drawPoints();
      this.drawArm();
    }
    this.drawFound();
    this.drawHover();
    this.drawTouch();
  }

  drawNext() {
    const { next, limits, matrix, deskToTrue } = this.marks;
    if (!next) return;
    const rim = circle(matrix, [next.x, next.y], limits.tests.find.cube / 2, 0, deskToTrue)
      .map((share) => share && [share[0] * this.width, share[1] * this.height]);
    this.ctx.setLineDash([5, 4]);
    this.polyline(rim, INK.red, 1.5, 0.8);
    this.ctx.setLineDash([]);
  }

  drawPoints() {
    const { refine, limits } = this.marks;
    if (!refine) return;
    const { ctx } = this;
    for (const row of refine.points) {
      const [x, y] = [row.view[0] * this.width, row.view[1] * this.height];
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = RANK_INK[rank(offShown(row), limits.tests.refine.off)];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.font = `600 10px ${MONO}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = INK.shadow;
      ctx.fillText(String(row.id), x + 8, y + 1);
      ctx.fillStyle = INK.white;
      ctx.fillText(String(row.id), x + 7, y);
    }
  }

  drawFound() {
    const { found } = this.marks;
    if (!found) return;
    const corners = found.corners.map((corner) => this.at(corner));
    if (corners.some((corner) => !corner)) return;
    for (const [start, end] of [...found.hidden, [0, 2], [1, 3]]) {
      this.polyline([corners[start], corners[end]], INK.cube, 1, 0.25);
    }
    for (const [start, end, isOutline] of found.seen) {
      this.polyline([corners[start], corners[end]], INK.cube, isOutline ? 1.2 : 1, 0.6);
    }
    const { down } = this.marks.limits.tests.find;
    const end = this.at([found.x, found.y, down]);
    const over = this.at([found.x, found.y, down + LABEL_OVER_CUP_MM]);
    if (end && over) {
      const bottom = Math.min(over[1], end[1] - LABEL_OVER_CUP_PX);
      this.label(`${found.x.toFixed(1)}, ${found.y.toFixed(1)}`, [end[0], bottom - 11], INK.cube, 'center');
    }
  }

  drawHover() {
    const { hover, matrix } = this.marks;
    const desk = hover && toDesk(matrix, hover);
    if (!desk) return;
    this.polyline(this.ring(desk), INK.pointer, 2, 0.9);
    this.label(`${desk[0].toFixed(1)}, ${desk[1].toFixed(1)}`, [hover[0] * this.width + 14,
      hover[1] * this.height + 22], INK.pointer, 'left');
  }

  drawTouch() {
    const { touch } = this.marks;
    if (!touch) return;
    const { ctx } = this;
    const x = touch.u * this.width;
    const y = touch.v * this.height;
    // At least a twentieth of the view across (the owner's): a twelfth.
    const half = Math.max(this.width, this.height) / 24;
    const gap = 7;
    ctx.globalAlpha = 0.5;
    for (const [ink, width] of [[INK.shadow, 4], [INK.green, 2]]) {
      ctx.strokeStyle = ink;
      ctx.lineWidth = width;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        ctx.beginPath();
        ctx.moveTo(x + dx * gap, y + dy * gap);
        ctx.lineTo(x + dx * half, y + dy * half);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    if (!touch.desk) return;
    this.polyline(this.ring(touch.desk), INK.green, 2, 0.95);
    this.dot([x, y], INK.green);
    this.label(`${touch.desk[0].toFixed(1)}, ${touch.desk[1].toFixed(1)}`, [x + half * 0.55, y + half * 0.55],
      INK.green, 'left');
  }

  // The cup's rim round a desk point, `height` over the desk, its true size, in canvas pixels.
  ring(point, height = 0) {
    return super.ring(point, height, this.marks.limits.tests.touch.cup_radius);
  }
}
