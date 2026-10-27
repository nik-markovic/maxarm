// The Tiles page's marks over the camera's picture, live or the snapshot, through the calibration:
// - with the areas: the pickup area, green, and the dropoff area, blue, on the desk; the line between them and the
//   arm's centre line, dashed; and the arm's reach, faint;
// - each tile as LOCATE TILES found it: its top face outlined in its status's colour, thin and see-through so the
//   tile's own edge shows beside it, and its letter on a badge at the letter's bottom right corner, so the badge
//   says which way up it is; '?' for a letter not read, '–' for a blank tile, both ringed in white, over the top;
//   a tile PLACE has carried off, dashed and faint, with no badge;
// - the arc PLACE sets words down along, dotted, with the areas; and where each letter of the word goes: the tile's
//   face, dashed until the tile is there, then filled, its letter at the letter's bottom right corner;
// - the arm: where it was sent, and where its servos say the cup is;
// - in the full-size view, the tile under the pointer, picked out.
// The tiles are drawn at their top faces' height: what the camera sees of them.

import { INK, ViewMarks } from '/common/marks.js';
import { toDesk } from '/common/shares.js';

export const STATUS_INK = { ready: '#41c363', 'too close': '#ffd23f', elsewhere: '#9db0c8', 'out of reach': '#f0524a' };
export const AREA_INK = { pickup: '#41c363', dropoff: '#4c8dff' };
// A letter of the word by how its tile is doing: to come, being carried, set down; stopped, refused or failed.
const WORD_INK = { waiting: '#ffffff', carrying: '#ffd23f', placed: '#4c8dff' };
export const wordInk = (status) => WORD_INK[status] || '#f0524a';

export class TilesMarks extends ViewMarks {
  // found: LOCATE TILES' result, as the state has it; word: PLACE's, the same; moved: the numbers of the tiles PLACE
  // has carried off; showAreas; hover: [u, v], shares of the view; picked: a tile's number, drawn picked out.
  constructor(canvas) {
    super(canvas, { found: null, word: null, moved: null, showAreas: true, hover: null, picked: null });
  }

  drawMarks() {
    if (this.marks.showAreas) {
      this.drawAreas();
      this.drawReach(0.35);
    }
    this.drawWord();
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
    this.ctx.setLineDash([2, 6]);
    this.polyline(this.arc(tiles.word.arc, -fan, fan), AREA_INK.dropoff, 1.5, 0.8);
    this.ctx.setLineDash([]);
  }

  drawWord() {
    const { word, limits } = this.marks;
    if (!word) return;
    for (const step of word.steps) {
      const face = step.corners.map(([x, y]) => this.at([x, y, limits.tiles.height]));
      if (face.some((corner) => !corner)) continue;
      const ink = wordInk(step.status);
      if (step.status === 'placed') this.fill(face, ink, 0.3);
      else this.ctx.setLineDash([5, 4]);
      this.polyline([...face, face[0]], ink, 1.5, 0.9);
      this.ctx.setLineDash([]);
      const { corner } = letterBadge({ ...step, is_read: true, upright: step.facing });
      const at = this.at([corner[0], corner[1], limits.tiles.height]);
      if (at) this.badge(step.letter, at, ink, false);
    }
  }

  drawTiles() {
    const { found, limits, picked, moved } = this.marks;
    if (!found) return;
    const hovered = this.hovered();
    for (const tile of found.tiles) {
      const face = tile.corners.map(([x, y]) => this.at([x, y, limits.tiles.height]));
      if (face.some((corner) => !corner)) continue;
      const ink = STATUS_INK[tile.status];
      if (moved && moved.has(tile.n)) {
        this.ctx.setLineDash([4, 4]);
        this.polyline([...face, face[0]], ink, 1, 0.35);
        this.ctx.setLineDash([]);
        continue;
      }
      const isPicked = tile.n === picked || (hovered && tile.n === hovered.n);
      if (isPicked) this.fill(face, ink, 0.12);
      this.polyline([...face, face[0]], ink, isPicked ? 1.5 : 1, isPicked ? 0.9 : 0.6);
      const { text, corner, isMarked } = letterBadge(tile);
      if (corner) {
        const at = this.at([corner[0], corner[1], limits.tiles.height]);
        if (at) this.badge(text, at, ink, isMarked);
      } else {
        const top = face.reduce((best, each) => (each[1] < best[1] ? each : best));
        this.badge(text, [top[0], top[1] - 12], ink, isMarked);
      }
    }
  }

  // The tile under the pointer, in the full-size view: its top face holds the point seen there.
  hovered() {
    const { hover, found, matrix, limits } = this.marks;
    if (!hover || !found) return null;
    const desk = toDesk(matrix, hover, limits.tiles.height);
    return desk ? tileAt(found.tiles, desk) : null;
  }

  // A tile's letter on a small disc of its colour; ringed in white when there is no letter to show.
  badge(text, [x, y], ink, isMarked) {
    const { ctx } = this;
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, 2 * Math.PI);
    ctx.fill();
    if (isMarked) {
      ctx.strokeStyle = INK.white;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.font = '700 12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#0b0f14';
    ctx.fillText(text, x, y + 0.5);
  }
}

// What a tile's badge shows, and where: its letter at the face's corner at the letter's bottom right -- 135 degrees
// on from where the letter's top points -- or '?' or '–' with no corner, for a letter not read or a blank tile.
export function letterBadge(tile) {
  if (tile.letter === undefined) return { text: String(tile.n), corner: null, isMarked: false };
  if (!tile.letter) return { text: '–', corner: null, isMarked: true };
  if (!tile.is_read) return { text: '?', corner: null, isMarked: true };
  const toward = ((tile.upright + 135) * Math.PI) / 180;
  const corner = tile.corners.reduce((best, each) => {
    const along = (each[0] - tile.x) * Math.cos(toward) + (each[1] - tile.y) * Math.sin(toward);
    return along > best.along ? { along, each } : best;
  }, { along: -Infinity, each: null }).each;
  return { text: tile.letter, corner, isMarked: false };
}

// What the tables and hovers call a tile: its number and letter, '?' or 'blank'.
export function tileName(tile) {
  const letter = tile.letter === undefined ? '' : !tile.letter ? ' blank' : tile.is_read ? ` ${tile.letter}` : ' ?';
  return `${tile.n}${letter}`;
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
