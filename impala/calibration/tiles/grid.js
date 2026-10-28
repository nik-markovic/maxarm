// The Tiles page's view from above: the shared grid in desk space, with the work areas -- pickup green, dropoff
// blue, the line between them and the arm's centre line dashed, the arc words go along dotted -- and each tile
// LOCATE TILES found, its face in its status's colour, its letter at the letter's bottom right corner, as over the
// picture; a tile PLACE has carried off dashed and faint, and where they were, grey, once found again. Where PLACE
// sets each letter of a word down: dashed until the tile is there, then filled. The pointer picks out the tile under it; a click is that tile, for onPickTile(n).

import { GridView } from '/common/grid.js';
import { AREA_INK, CARRIED_INK, STATUS_INK, bearings, letterBadge, tileAt, tileName, wordInk } from './view.js';

export class TilesGrid extends GridView {
  constructor(canvas, limits, onPickTile) {
    super(canvas, limits, (x, y) => {
      const tile = this.view.found ? tileAt(this.view.found.tiles, [x, y]) : null;
      if (tile) onPickTile(tile.n);
    });
    this.view.found = null;        // LOCATE TILES' result, as the state has it
    this.view.word = null;         // PLACE's, the same
    this.view.moved = null;        // the numbers of the tiles PLACE has carried off
    this.view.picked = null;       // a tile's number, drawn picked out
  }

  // Where the pointer is, not snapped to the grid: tiles lie anywhere.
  snapped(event) {
    const box = this.canvas.getBoundingClientRect();
    return this.toArm(event.clientX - box.left, event.clientY - box.top);
  }

  drawMarks() {
    this.drawAreas();
    if (this.view.word) this.drawWord(this.view.word);
    if (this.view.found) this.drawCarried(this.view.found.carried || []);
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
    ctx.setLineDash([2, 5]);
    ctx.strokeStyle = `${AREA_INK.dropoff}cc`;
    ctx.beginPath();
    this.trace(this.arc(tiles.word.arc, -fan, fan));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawWord(word) {
    const { ctx } = this;
    for (const step of word.steps) {
      const ink = wordInk(step.status);
      ctx.beginPath();
      this.trace(step.corners.map(([x, y]) => this.toScreen(x, y)));
      ctx.closePath();
      if (step.status === 'placed') {
        ctx.fillStyle = `${ink}55`;
        ctx.fill();
      } else {
        ctx.setLineDash([4, 3]);
      }
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);
      this.drawBadge({ ...step, is_read: true, upright: step.facing }, ink);
    }
  }

  // Where the tiles PLACE carried were, before the tiles were found again.
  drawCarried(tiles) {
    const { ctx } = this;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = `${CARRIED_INK}99`;
    ctx.lineWidth = 1;
    for (const tile of tiles) {
      ctx.beginPath();
      this.trace(tile.corners.map(([x, y]) => this.toScreen(x, y)));
      ctx.closePath();
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  drawTiles(tiles) {
    const { ctx } = this;
    const hovered = this.hover ? tileAt(tiles, this.hover) : null;
    for (const tile of tiles) {
      const ink = STATUS_INK[tile.status];
      if (this.view.moved && this.view.moved.has(tile.n)) {
        ctx.beginPath();
        this.trace(tile.corners.map(([x, y]) => this.toScreen(x, y)));
        ctx.closePath();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = `${ink}77`;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        continue;
      }
      const isPicked = tile.n === this.view.picked || (hovered && hovered.n === tile.n);
      ctx.beginPath();
      this.trace(tile.corners.map(([x, y]) => this.toScreen(x, y)));
      ctx.closePath();
      ctx.fillStyle = `${ink}${isPicked ? '88' : '40'}`;
      ctx.fill();
      ctx.strokeStyle = ink;
      ctx.lineWidth = isPicked ? 2.5 : 1.5;
      ctx.stroke();
      this.drawBadge(tile, ink);
    }
  }

  // The letter on a disc at its bottom right corner; '?' or '–', ringed, at the tile's centre.
  drawBadge(tile, ink) {
    const { ctx } = this;
    const { text, corner, isMarked } = letterBadge(tile);
    const [x, y] = corner ? this.toScreen(corner[0], corner[1]) : this.toScreen(tile.x, tile.y);
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(x, y, 7.5, 0, 2 * Math.PI);
    ctx.fill();
    if (isMarked) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.font = '700 10.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#0b0f14';
    ctx.fillText(text, x, y + 0.5);
  }

  // Over a tile: its number, letter and desk x, y. Elsewhere: nothing to pick.
  drawHover() {
    const tile = this.hover && this.view.found ? tileAt(this.view.found.tiles, this.hover) : null;
    if (!tile) return;
    const { ctx } = this;
    const [x, y] = this.toScreen(tile.x, tile.y);
    const text = `${tileName(tile)}: ${tile.x.toFixed(1)}, ${tile.y.toFixed(1)}`;
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
