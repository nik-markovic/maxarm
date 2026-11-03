// The Tiles page's marks over the camera's live picture, and what its views draw tiles with.
//
// What the picture shows of the tiles -- the work areas and the reach; each tile in its status's colour with its
// letter on a badge, a letter not read in red; where the tiles a word carried were; where a word's letters go -- is
// drawn in Python (applib/overlay/tiles.py), as the app's stream shows it, and laid over the picture as a layer
// (common/layer.js). Drawn here, over that layer:
// - the arm: where it was sent, and where its servos say the cup is;
// - the tile last clicked, and in the full-size view the tile under the pointer, picked out.
// The colours are Python's too (applib/overlay/inks.py), given with the limits: useInks(limits.tiles.ink).

import { ViewMarks } from '/common/marks.js';
import { toDesk } from '/common/shares.js';

export const STATUS_INK = {};
export const AREA_INK = {};
const WORD_INK = {};
export let CARRIED_INK = '';
export let UNREAD_INK = '';

export function useInks(ink) {
  Object.assign(STATUS_INK, ink.status);
  Object.assign(AREA_INK, ink.area);
  Object.assign(WORD_INK, ink.word);
  CARRIED_INK = ink.carried;
  UNREAD_INK = ink.unread;
}

// A letter of the word by how its tile is doing: to come, being carried, set down; stopped, refused or failed.
export const wordInk = (status) => WORD_INK[status] || WORD_INK.failed;
export const isUnread = (tile) => Boolean(tile.letter) && !tile.is_read;
// A tile's colour: its status's, or red for a letter not read (the owner's, PILOT-jackal).
export const tileInk = (tile) => (isUnread(tile) ? UNREAD_INK : STATUS_INK[tile.status]);

export class TilesMarks extends ViewMarks {
  // found: LOCATE TILES' result, as the state has it; moved: the numbers of the tiles PLACE has carried off; hover:
  // [u, v], shares of the view; picked: a tile's number, drawn picked out.
  constructor(canvas) {
    super(canvas, { found: null, moved: null, hover: null, picked: null });
  }

  drawMarks() {
    this.drawPicked();
    this.drawArm();
  }

  drawPicked() {
    const { found, limits, picked, moved } = this.marks;
    if (!found) return;
    const hovered = this.hovered();
    for (const tile of found.tiles) {
      const isPicked = tile.n === picked || (hovered && tile.n === hovered.n);
      if (!isPicked || (moved && moved.has(tile.n))) continue;
      const face = tile.corners.map(([x, y]) => this.at([x, y, limits.tiles.height]));
      if (face.some((corner) => !corner)) continue;
      this.fill(face, tileInk(tile), 0.12);
      this.polyline([...face, face[0]], tileInk(tile), 1.5, 0.9);
    }
  }

  // The tile under the pointer, in the full-size view: its top face holds the point seen there.
  hovered() {
    const { hover, found, matrix, limits } = this.marks;
    if (!hover || !found) return null;
    const desk = toDesk(matrix, hover, limits.tiles.height);
    return desk ? tileAt(found.tiles, desk) : null;
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

// A chevron on a tile's face, its tip towards where the letter's top points, desk mm; none for a letter not read
// or a blank tile.
export function chevronUp(tile) {
  if (!tile.letter || !tile.is_read) return [];
  const angle = (tile.upright * Math.PI) / 180;
  const [ux, uy] = [Math.cos(angle), Math.sin(angle)];
  const at = (along, across) => [tile.x + along * ux - across * uy, tile.y + along * uy + across * ux];
  return [at(1, -6), at(8, 0), at(1, 6)];
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
