// The live view's marks, drawn over the camera's picture through the calibration (common/shares.js):
// - guides: the reach on the desk, 95 and 270 mm, orange and red as on the grid; the arm's target, the cup's rim
//   there, faint green, with a line down to the desk under it: where it was sent, not where its servos say it is
//   (the owner's, PILOT-impala). Not once the arm is down at ARM TO DESK's point: that point's marks say it;
// - ARM TO DESK's point, blue as the ring under the pointer: a crosshair where the view was clicked, half
//   see-through, the cup's rim round the point on the desk, a dot and its desk x, y; once the arm is down there, the
//   dot alone. Once its servos are read there, the same in green where they say the cup is: the cup's rim at the
//   height they say, the crosshair and the dot on the desk under it. The test is whether that rim sits on the cup in
//   the picture, not whether the arm met the point (the owner's, 30 Oct);
// - with the guides, the red cube's points: each a small ring, numbered, where the camera saw the cube's centre,
//   coloured by how far off it is (refine.js); and a dashed circle on the desk, a place for the next point;
// - with the desk grid (off unless asked), desk space every 10 mm on the desk, thin and pale, under everything else:
//   every 50 mm a little stronger, and the two lines through the arm's turning axis stronger still.
// The cup's rim is drawn its true size (tests/api.py). The drawing itself, and the reach and the arm, are the shared
// ViewMarks' (common/marks.js). The red cube FIND RED CUBE found is drawn in Python (applib/overlay/cube.py), as the
// app's stream shows it, a layer under these (tests.js).

import { INK as SHARED_INK, MONO, ViewMarks } from '/common/marks.js';
import { circle, toDesk } from '/common/shares.js';
import { offShown, rank } from './refine.js';

// The ring under the pointer is blue, to see against a pale desk (the owner's); so is ARM TO DESK's point, clicked.
export const INK = { ...SHARED_INK, cube: '#ff4df8', red: '#ff4d5e', pointer: '#3d9bff', desk: '#c9a4ff' };
// The desk grid: a line every so many mm, a stronger one every so many, out to the reach and a little past it; each
// line drawn through points this far apart, so that a part behind the camera is left out.
const DESK_GRID_MM = 10;
const DESK_GRID_STRONG_MM = 50;
const DESK_GRID_PAST_REACH_MM = 30;
const RANK_INK = { ok: INK.green, warn: '#e6b84a', bad: INK.reach, '': '#8d9aa9' };
// ARM TO DESK's dots: where the servos say the cup is, over where it was clicked, smaller, so that the clicked one
// still shows round it when the two meet.
const CLICKED_DOT_PX = 3;
const SAYS_DOT_PX = 2;
export class LiveMarks extends ViewMarks {
  // touch: { u, v, desk, phase, says }, as tests.js has it; hover: [u, v]; showGuides; refine: the red cube's points
  // and the cubes, as the tests' state has them; next: { x, y }, desk mm; showDeskGrid.
  constructor(canvas) {
    super(canvas, { touch: null, hover: null, showGuides: true, refine: null, next: null, showDeskGrid: false });
  }

  drawMarks() {
    const { touch } = this.marks;
    if (this.marks.showDeskGrid) this.drawDeskGrid();
    if (this.marks.showGuides) {
      this.drawReach();
      this.drawNext();
      this.drawPoints();
      if (!touch || !['landed', 'read'].includes(touch.phase)) this.drawArm();
    }
    this.drawHover();
    this.drawTouch();
  }

  // Desk space on the desk, x and y every 10 mm: lay grid paper along it, and its lines run with these all over the
  // sheet where the camera is right; the lines x = 0 and y = 0 cross over the arm's turning axis.
  drawDeskGrid() {
    const end = Math.ceil((this.marks.limits.reach + DESK_GRID_PAST_REACH_MM) / DESK_GRID_STRONG_MM)
      * DESK_GRID_STRONG_MM;
    const steps = Array.from({ length: (2 * end) / DESK_GRID_MM + 1 }, (_, index) => -end + index * DESK_GRID_MM);
    for (const k of steps) {
      const [width, alpha] = k === 0 ? [1.2, 0.85] : k % DESK_GRID_STRONG_MM === 0 ? [1, 0.6] : [0.75, 0.4];
      this.polyline(steps.map((along) => this.at([k, along, 0])), INK.desk, width, alpha);
      this.polyline(steps.map((along) => this.at([along, k, 0])), INK.desk, width, alpha);
    }
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
    const clicked = [touch.u * this.width, touch.v * this.height];
    if (touch.phase === 'sent') {
      this.crosshair(clicked, INK.pointer);
      if (touch.desk) {
        this.polyline(this.ring(touch.desk), INK.pointer, 2, 0.95);
        const half = this.crosshairHalf();
        this.label(`${touch.desk[0].toFixed(1)}, ${touch.desk[1].toFixed(1)}`,
          [clicked[0] + half * 0.55, clicked[1] + half * 0.55], INK.pointer, 'left');
      }
    }
    if (touch.desk) this.dot(clicked, INK.pointer, CLICKED_DOT_PX);
    const { says } = touch;
    const under = says && this.at([says[0], says[1], 0]);
    if (!under) return;
    this.crosshair(under, INK.green);
    this.polyline(this.ring(says, says[2]), INK.green, 2, 0.95);
    this.dot(under, INK.green, SAYS_DOT_PX);
  }

  // A crosshair round a canvas point, half see-through, its middle left open.
  crosshair([x, y], ink) {
    const { ctx } = this;
    const half = this.crosshairHalf();
    const gap = 7;
    ctx.globalAlpha = 0.5;
    for (const [shade, width] of [[INK.shadow, 4], [ink, 2]]) {
      ctx.strokeStyle = shade;
      ctx.lineWidth = width;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        ctx.beginPath();
        ctx.moveTo(x + dx * gap, y + dy * gap);
        ctx.lineTo(x + dx * half, y + dy * half);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  // Half a crosshair across: at least a twentieth of the view across (the owner's), a twelfth.
  crosshairHalf() {
    return Math.max(this.width, this.height) / 24;
  }

  // The cup's rim round a desk point, `height` over the desk, its true size, in canvas pixels.
  ring(point, height = 0) {
    return super.ring(point, height, this.marks.limits.tests.touch.cup_radius);
  }
}
