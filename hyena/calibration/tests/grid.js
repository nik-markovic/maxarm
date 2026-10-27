// The tests page's view from above: the shared grid in desk space, with where the camera found the red cube -- its
// footprint, turned as it stands, an X under its centre -- and the point of desk last clicked in the live view.
// And the red cube's points that refine the calibration, as on the desk page: each a numbered circle where the arm
// said the cup was, coloured by how far off it is; and a dashed circle, a place for the next point.

import { CUBE_INK, GridView, INK } from '/common/grid.js';
import { offShown, rank } from './refine.js';

const RANK_INK = { ok: INK.green, warn: INK.warn, bad: INK.bad, '': INK.neutral };

export class TestsGrid extends GridView {
  constructor(canvas, limits, onPick) {
    super(canvas, limits, onPick);
    this.view.found = null;        // the red cube: corners, desk mm, the first four on the desk
    this.view.touch = null;        // [x, y], desk mm
    this.view.refine = null;       // the points and the cubes, as the tests' state has them
    this.view.next = null;         // { x, y }: a place for the next point
  }

  drawMarks() {
    const { found, touch, refine, next } = this.view;
    if (next) this.drawNext(next);
    if (refine) this.drawPoints(refine);
    if (found) this.drawFound(found);
    if (touch) this.drawTouch(touch);
  }

  drawFound(found) {
    const { ctx } = this;
    const base = found.corners.slice(0, 4).map(([x, y]) => this.toScreen(x, y));
    ctx.beginPath();
    this.trace(base);
    ctx.closePath();
    ctx.fillStyle = `${CUBE_INK.red}33`;
    ctx.fill();
    ctx.strokeStyle = CUBE_INK.red;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
    this.line(base[0], base[2]);
    this.line(base[1], base[3]);
  }

  drawTouch(touch) {
    const { ctx } = this;
    const [x, y] = this.toScreen(touch[0], touch[1]);
    ctx.strokeStyle = INK.green;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, this.limits.tests.touch.cup_radius * this.scale, 0, 2 * Math.PI);
    ctx.stroke();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      this.line([x + dx * 3, y + dy * 3], [x + dx * 9, y + dy * 9]);
    }
  }

  drawNext({ x, y }) {
    const { ctx } = this;
    const [cx, cy] = this.toScreen(x, y);
    ctx.beginPath();
    ctx.arc(cx, cy, (this.limits.tests.find.cube / 2) * this.scale, 0, 2 * Math.PI);
    ctx.fillStyle = `${CUBE_INK.red}14`;
    ctx.fill();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = CUBE_INK.red;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = '600 10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = CUBE_INK.red;
    ctx.fillText('next', cx, cy);
  }

  drawPoints(refine) {
    const { ctx } = this;
    const offLimits = this.limits.tests.refine.off;
    ctx.font = '600 10.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const point of refine.points) {
      const [x, y] = this.toScreen(point.x, point.y);
      ctx.fillStyle = RANK_INK[rank(offShown(point), offLimits)];
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#070a0d';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#071009';
      ctx.fillText(String(point.id), x, y + 0.5);
    }
  }
}
