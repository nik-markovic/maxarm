// The Tiles page's marks over the camera's picture, live or the snapshot, through the calibration:
// - with the areas: the pickup area, green, and the dropoff area, blue, on the desk; the line between them and the
//   arm's centre line, dashed; and the arm's reach, faint;
// - each tile as LOCATE TILES found it: its top face outlined in its status's colour, thin and see-through so the
//   tile's own edge shows beside it, its number over it;
// - the arm: where it was sent, and where its servos say the cup is;
// - in the full-size view, the tile under the pointer, picked out.
// The tiles are drawn at their top faces' height: what the camera sees of them.

import { INK, ViewMarks } from '/common/marks.js';
import { toDesk } from '/common/shares.js';

export const STATUS_INK = { ready: '#41c363', 'too close': '#ffd23f', elsewhere: '#9db0c8', 'out of reach': '#f0524a' };
export const AREA_INK = { pickup: '#41c363', dropoff: '#4c8dff' };

export class TilesMarks extends ViewMarks {
  // found: LOCATE TILES' result, as the state has it; showAreas; hover: [u, v], shares of the view; picked: a tile's
  // number, drawn picked out.
  constructor(canvas) {
    super(canvas, { found: null, showAreas: true, hover: null, picked: null });
  }

  drawMarks() {
    if (this.marks.showAreas) {
      this.drawAreas();
      this.drawReach(0.35);
    }
    this.drawTiles();
    this.drawArm();
  }

  drawAreas() {
    const { tiles, fan } = this.marks.limits;
    for (const area of tiles.areas) {
      const [from, to] = bearings(area.side, fan);
      const outline = [...this.arc(area.outer, from, to), ...this.arc(area.inner, to, from)];
      this.fill(outline, AREA_INK[area.name], 0.13);
      this.polyline([...outline, outline[0]], AREA_INK[area.name], 1.2, 0.55);
    }
    const pickup = tiles.areas.find((area) => area.name === 'pickup');
    const dropoff = tiles.areas.find((area) => area.name === 'dropoff');
    if (!pickup || !dropoff) return;
    this.ctx.setLineDash([7, 5]);
    this.polyline(this.arc(pickup.inner, -fan, fan), INK.white, 1.5, 0.7);
    this.polyline([this.at([0, -dropoff.inner, 0]), this.at([0, -pickup.outer, 0])], INK.white, 1.5, 0.7);
    this.ctx.setLineDash([]);
  }

  drawTiles() {
    const { found, limits, picked } = this.marks;
    if (!found) return;
    const hovered = this.hovered();
    for (const tile of found.tiles) {
      const face = tile.corners.map(([x, y]) => this.at([x, y, limits.tiles.height]));
      if (face.some((corner) => !corner)) continue;
      const ink = STATUS_INK[tile.status];
      const isPicked = tile.n === picked || (hovered && tile.n === hovered.n);
      if (isPicked) this.fill(face, ink, 0.12);
      this.polyline([...face, face[0]], ink, isPicked ? 1.5 : 1, isPicked ? 0.9 : 0.6);
      const top = face.reduce((best, corner) => (corner[1] < best[1] ? corner : best));
      this.badge(String(tile.n), [top[0], top[1] - 12], ink);
    }
  }

  // The tile under the pointer, in the full-size view: its top face holds the point seen there.
  hovered() {
    const { hover, found, matrix, limits } = this.marks;
    if (!hover || !found) return null;
    const desk = toDesk(matrix, hover, limits.tiles.height);
    return desk ? tileAt(found.tiles, desk) : null;
  }

  // A tile's number on a small disc of its colour.
  badge(text, [x, y], ink) {
    const { ctx } = this;
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, 2 * Math.PI);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.font = '700 11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#0b0f14';
    ctx.fillText(text, x, y + 0.5);
  }
}

// The bearings an area spans, degrees from straight out: x < 0 is the arm's left, at negative bearings.
export function bearings(side, fan) {
  return side < 0 ? [-fan, 0] : side > 0 ? [0, fan] : [-fan, fan];
}

// The tile whose face holds a desk point, or null.
export function tileAt(tiles, [x, y]) {
  return tiles.find((tile) => isInside(tile.corners, [x, y])) || null;
}

function isInside(corners, [x, y]) {
  let sign = 0;
  for (let index = 0; index < corners.length; index += 1) {
    const [ax, ay] = corners[index];
    const [bx, by] = corners[(index + 1) % corners.length];
    const cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    if (cross !== 0) {
      if (sign && Math.sign(cross) !== sign) return false;
      sign = Math.sign(cross);
    }
  }
  return true;
}
