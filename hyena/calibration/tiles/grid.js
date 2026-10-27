// The Tiles page's view from above: the shared grid in desk space, with the work areas -- pickup green, dropoff
// blue, the line between them and the arm's centre line dashed -- and each tile LOCATE TILES found, its face in its
// status's colour, numbered. The pointer picks out the tile under it; a click is that tile, for onPickTile(n).

import { GridView } from '/common/grid.js';
import { AREA_INK, STATUS_INK, bearings, tileAt } from './view.js';

export class TilesGrid extends GridView {
  constructor(canvas, limits, onPickTile) {
    super(canvas, limits, (x, y) => {
      const tile = this.view.found ? tileAt(this.view.found.tiles, [x, y]) : null;
      if (tile) onPickTile(tile.n);
    });
    this.view.found = null;        // LOCATE TILES' result, as the state has it
    this.view.picked = null;       // a tile's number, drawn picked out
  }

  // Where the pointer is, not snapped to the grid: tiles lie anywhere.
  snapped(event) {
    const box = this.canvas.getBoundingClientRect();
    return this.toArm(event.clientX - box.left, event.clientY - box.top);
  }

  drawMarks() {
    this.drawAreas();
    if (this.view.found) this.drawTiles(this.view.found.tiles);
  }

  drawAreas() {
    const { ctx } = this;
    const { fan, tiles } = this.limits;
    for (const area of tiles.areas) {
      const [from, to] = bearings(area.side, fan);
      ctx.beginPath();
      this.trace(this.arc(area.outer, from, to).concat(this.arc(area.inner, to, from)));
      ctx.closePath();
      ctx.fillStyle = `${AREA_INK[area.name]}26`;
      ctx.fill();
      ctx.strokeStyle = `${AREA_INK[area.name]}99`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    const pickup = tiles.areas.find((area) => area.name === 'pickup');
    const dropoff = tiles.areas.find((area) => area.name === 'dropoff');
    if (!pickup || !dropoff) return;
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    this.trace(this.arc(pickup.inner, -fan, fan));
    ctx.stroke();
    this.line(this.atBearing(dropoff.inner, 0), this.atBearing(pickup.outer, 0));
    ctx.setLineDash([]);
  }

  drawTiles(tiles) {
    const { ctx } = this;
    const hovered = this.hover ? tileAt(tiles, this.hover) : null;
    ctx.font = '700 10.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const tile of tiles) {
      const ink = STATUS_INK[tile.status];
      const isPicked = tile.n === this.view.picked || (hovered && hovered.n === tile.n);
      ctx.beginPath();
      this.trace(tile.corners.map(([x, y]) => this.toScreen(x, y)));
      ctx.closePath();
      ctx.fillStyle = `${ink}${isPicked ? '88' : '40'}`;
      ctx.fill();
      ctx.strokeStyle = ink;
      ctx.lineWidth = isPicked ? 2.5 : 1.5;
      ctx.stroke();
      const [x, y] = this.toScreen(tile.x, tile.y);
      ctx.fillStyle = '#e9eef4';
      ctx.fillText(String(tile.n), x, y + 0.5);
    }
  }

  // Over a tile: its number and desk x, y. Elsewhere: nothing to pick.
  drawHover() {
    const tile = this.hover && this.view.found ? tileAt(this.view.found.tiles, this.hover) : null;
    if (!tile) return;
    const { ctx } = this;
    const [x, y] = this.toScreen(tile.x, tile.y);
    const text = `${tile.n}: ${tile.x.toFixed(1)}, ${tile.y.toFixed(1)}`;
    ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace';
    const width = ctx.measureText(text).width + 12;
    const left = Math.min(x + 14, this.size - width - 4);
    const top = Math.max(y - 32, 4);
    ctx.fillStyle = 'rgba(18, 23, 30, 0.92)';
    ctx.strokeStyle = STATUS_INK[tile.status];
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(left, top, width, 20, 5);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#e4eaf0';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, left + 6, top + 10.5);
  }
}
