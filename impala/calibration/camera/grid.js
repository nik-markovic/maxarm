// The camera page's view from above: the shared grid in desk space, with where the cubes are suggested to go,
// dashed -- fainter once that cube is recorded -- and where each one was recorded, solid, the cross at its centre
// where the cup was.

import { CUBE_INK, GridView } from '/common/grid.js';

export class CubeGrid extends GridView {
  constructor(canvas, limits, onPick) {
    super(canvas, limits, onPick);
    this.view.suggested = [];      // { colour, x, y }, desk mm
    this.view.cubes = {};          // colour: { x, y, z } as recorded, or null
  }

  drawMarks() {
    const { ctx } = this;
    const half = this.limits.camera.cube / 2;
    ctx.font = '600 10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const spot of this.view.suggested) {
      const ink = CUBE_INK[spot.colour];
      ctx.globalAlpha = this.view.cubes[spot.colour] ? 0.45 : 1;
      this.square(spot.x, spot.y, half);
      ctx.fillStyle = `${ink}1f`;
      ctx.fill();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ink;
      ctx.fillText(spot.colour, ...this.toScreen(spot.x, spot.y));
      ctx.globalAlpha = 1;
    }
    for (const [colour, cube] of Object.entries(this.view.cubes)) {
      if (!cube) continue;
      const ink = CUBE_INK[colour];
      this.square(cube.x, cube.y, half);
      ctx.fillStyle = `${ink}8c`;
      ctx.shadowColor = ink;
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 2;
      ctx.stroke();
      const [x, y] = this.toScreen(cube.x, cube.y);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      this.line([x - 6, y], [x + 6, y]);
      this.line([x, y - 6], [x, y + 6]);
    }
  }

  square(x, y, half) {
    const corners = [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]];
    this.ctx.beginPath();
    this.trace(corners.map(([cx, cy]) => this.toScreen(cx, cy)));
    this.ctx.closePath();
  }
}
