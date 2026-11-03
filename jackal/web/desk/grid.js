// The desk page's view from above: the shared grid, with the touch points on it, coloured by how far each one
// sits off the desk.

import { GridView, INK } from '/common/grid.js';

export class DeskGrid extends GridView {
  constructor(canvas, limits, onPick) {
    super(canvas, limits, onPick);
    this.view.points = [];
  }

  drawMarks() {
    const { ctx } = this;
    const [warn, bad] = this.limits.desk.deviation;
    ctx.font = '600 10.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const point of this.view.points) {
      const [x, y] = this.toScreen(point.x, point.y);
      const size = point.deviation === null ? null : Math.abs(point.deviation);
      ctx.fillStyle = size === null ? INK.neutral : size <= warn ? INK.green : size <= bad ? INK.warn : INK.bad;
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
