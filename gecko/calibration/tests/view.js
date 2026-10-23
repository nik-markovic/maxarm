// The live view's marks, drawn over the camera's picture through the calibration (shares.js):
// - guides: the reach on the desk, 95 and 270 mm, orange and red as on the grid; the arm's target, the cup's rim
//   there, faint green, with a line down to the desk under it; where the servos say the cup is, a white dot;
// - where the arm landed, if its servos say it is more than half a millimetre from where it was sent (the owner's):
//   the cup's rim there, yellow. On the real cup, the camera is right and the arm missed; on the green ring, the
//   arm is right and the camera is not;
// - the red cube as FIND RED CUBE found it: its edges in pink, fainter where hidden, an X under its centre, and its
//   desk x, y over it, clear of the cup's end;
// - ARM TO DESK's point: a crosshair where the view was clicked, half see-through, the cup's rim round the point on
//   the desk, and its desk x, y;
// - with the guides, the red cube's points: each a small ring, numbered, where the camera saw the cube's centre,
//   coloured by how far off it is (refine.js); and a dashed circle on the desk, a place for the next point.
// The cup's rim is drawn its true size (tests/api.py). And the live view as large as the window allows, to click on
// (openFullView).

import { offShown, rank } from './refine.js';
import { circle, toDesk, toView } from './shares.js';

const INK = {
  reach: '#f0524a', reachMin: '#f08c3c', cube: '#ff4df8', green: '#41c363', yellow: '#ffd23f', white: '#ffffff',
  shadow: '#05070a', red: '#ff4d5e',
};
const RANK_INK = { ok: INK.green, warn: '#e6b84a', bad: INK.reach, '': '#8d9aa9' };
const MONO = '"JetBrains Mono", ui-monospace, monospace';
const FULL_MARGIN_PX = 14;
const RETRY_MS = 3000;
// The found cube's coordinates are written this far over the cup's end, which comes down to 4 over the cube's top,
// and at least so many pixels: the cup's end stays in sight.
const LABEL_OVER_CUP_MM = 16;
const LABEL_OVER_CUP_PX = 18;
// Further than this from where it was sent, desk mm across, and the arm has landed off (the owner's).
export const LANDED_OFF_MM = 0.5;

// How far the servos say the cup is from where it was sent, across the desk; null while that is not known.
export function landedOff(sent, says, isReading) {
  return sent && says && !isReading ? Math.hypot(says[0] - sent[0], says[1] - sent[1]) : null;
}

export class LiveMarks {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = 0;
    this.height = 0;
    // matrix: desk to shares of the view; deskToTrue: the arm's x/y error; limits: /api/limits; found: the red
    // cube; touch: { u, v, desk }; hover: [u, v]; target, sent, says: the arm, desk mm; refine: the red cube's
    // points and the cubes, as the tests' state has them; next: { x, y }, desk mm.
    this.marks = { matrix: null, deskToTrue: null, limits: null, found: null, touch: null, hover: null,
      target: null, sent: null, says: null, isReading: false, showGuides: true, refine: null, next: null };
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  update(marks) {
    Object.assign(this.marks, marks);
    this.draw();
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.draw();
  }

  draw() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.width, this.height);
    if (!this.width || !this.marks.matrix || !this.marks.limits) return;
    if (this.marks.showGuides) {
      this.drawReach();
      this.drawNext();
      this.drawPoints();
      this.drawArm();
    }
    this.drawLanding();
    this.drawFound();
    this.drawHover();
    this.drawTouch();
  }

  // Desk point to canvas pixels; null behind the camera.
  at(point) {
    const share = toView(this.marks.matrix, point);
    return share && [share[0] * this.width, share[1] * this.height];
  }

  drawReach() {
    const { reach, reach_min: reachMin, fan } = this.marks.limits;
    for (const [radius, ink] of [[reach, INK.reach], [reachMin, INK.reachMin]]) {
      const points = [];
      for (let bearing = -fan; bearing <= fan; bearing += 2) {
        const angle = (bearing * Math.PI) / 180;
        points.push(this.at([radius * Math.sin(angle), -radius * Math.cos(angle), 0]));
      }
      this.polyline(points, ink, 2, 0.75);
    }
  }

  drawArm() {
    const { ctx } = this;
    const { target, says, isReading } = this.marks;
    if (target) {
      const cup = this.at(target);
      const under = this.at([target[0], target[1], 0]);
      if (cup && under && target[2] > 0.5) {
        ctx.setLineDash([4, 4]);
        this.polyline([cup, under], INK.white, 1, 0.45);
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.6;
        this.dot(under, INK.white);
        ctx.globalAlpha = 1;
      }
      this.polyline(this.ring(target, target[2]), INK.green, 1.5, 0.3);
    }
    const dot = says && this.at(says);
    if (dot) {
      ctx.fillStyle = isReading ? 'rgba(255, 255, 255, 0.35)' : INK.white;
      ctx.beginPath();
      ctx.arc(dot[0], dot[1], 3.5, 0, 2 * Math.PI);
      ctx.fill();
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

  drawLanding() {
    const { sent, says, isReading } = this.marks;
    const off = landedOff(sent, says, isReading);
    if (off === null || off <= LANDED_OFF_MM) return;
    this.polyline(this.ring(says, says[2]), INK.yellow, 2, 0.95);
  }

  drawFound() {
    const { found } = this.marks;
    if (!found) return;
    const corners = found.corners.map((corner) => this.at(corner));
    if (corners.some((corner) => !corner)) return;
    for (const [start, end] of [...found.hidden, [0, 2], [1, 3]]) {
      this.polyline([corners[start], corners[end]], INK.cube, 1.5, 0.35);
    }
    for (const [start, end, isOutline] of found.seen) {
      this.polyline([corners[start], corners[end]], INK.cube, isOutline ? 2.2 : 1.6, 0.95);
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
    this.polyline(this.ring(desk), INK.white, 1.5, 0.55);
    this.label(`${desk[0].toFixed(1)}, ${desk[1].toFixed(1)}`, [hover[0] * this.width + 14,
      hover[1] * this.height + 22], INK.white, 'left');
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
  ring([x, y], height = 0) {
    const { matrix, limits, deskToTrue } = this.marks;
    return circle(matrix, [x, y], limits.tests.touch.cup_radius, height, deskToTrue)
      .map((share) => share && [share[0] * this.width, share[1] * this.height]);
  }

  // --- small drawing helpers ---

  // Points joined up, skipping any gap a point behind the camera leaves.
  polyline(points, ink, width, alpha) {
    const { ctx } = this;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = ink;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let isDrawing = false;
    for (const point of points) {
      if (!point) {
        isDrawing = false;
      } else if (isDrawing) {
        ctx.lineTo(point[0], point[1]);
      } else {
        ctx.moveTo(point[0], point[1]);
        isDrawing = true;
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  dot([x, y], ink) {
    const { ctx } = this;
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Text on a dark pill, kept inside the view.
  label(text, [x, y], ink, align) {
    const { ctx } = this;
    ctx.font = `600 12px ${MONO}`;
    const width = ctx.measureText(text).width + 14;
    let left = align === 'center' ? x - width / 2 : x;
    left = Math.min(Math.max(left, 4), this.width - width - 4);
    const top = Math.min(Math.max(y - 11, 4), this.height - 26);
    ctx.fillStyle = 'rgba(10, 13, 17, 0.82)';
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(left, top, width, 22, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, left + 7, top + 11.5);
  }
}

// The live view (`feed`, the stream's path) as large as the window allows, its marks over it, to click on.
// `onPick(u, v)` gets each click as shares of the view. Esc or × closes it, and `onClose()` is told;
// `update(marks)` keeps its marks as the page's.
export function openFullView({ feed, aspect, title, hint, onPick, onReset, onClose }) {
  const layer = document.createElement('div');
  layer.className = 'fullview';
  layer.innerHTML = `
    <div class="fullview-bar">
      <div class="fullview-title"><b>${title}</b><span>${hint}</span></div>
      <div class="fullview-point"><span>desk</span><b class="num">—</b><small class="landed"></small></div>
      <span class="fullview-doing"></span>
      <button class="btn reset" title="Stop everything, let go, lift clear and go home">RESET</button>
      <button class="btn icon close" title="Close (Esc)">×</button>
    </div>
    <div class="fullview-stage"><div class="fullview-frame"><img alt=""><canvas></canvas></div></div>`;
  document.body.append(layer);
  const stage = layer.querySelector('.fullview-stage');
  const frame = layer.querySelector('.fullview-frame');
  const canvas = layer.querySelector('canvas');
  const image = layer.querySelector('img');
  const marks = new LiveMarks(canvas);
  const connect = () => { if (layer.isConnected) image.src = `${feed}?t=${Date.now()}`; };
  image.addEventListener('error', () => setTimeout(connect, RETRY_MS));
  connect();

  const fit = () => {
    const width = Math.min(stage.clientWidth - 2 * FULL_MARGIN_PX, (stage.clientHeight - 2 * FULL_MARGIN_PX) * aspect);
    frame.style.width = `${width}px`;
    frame.style.height = `${width / aspect}px`;
  };
  const stageWatch = new ResizeObserver(fit);
  stageWatch.observe(stage);
  const share = (event) => {
    const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height];
  };
  canvas.addEventListener('click', (event) => onPick(...share(event)));
  canvas.addEventListener('pointermove', (event) => marks.update({ hover: share(event) }));
  canvas.addEventListener('pointerleave', () => marks.update({ hover: null }));
  layer.querySelector('.reset').addEventListener('click', onReset);

  const onKey = (event) => { if (event.key === 'Escape') close(); };
  function close() {
    stageWatch.disconnect();
    image.removeAttribute('src');                // the stream ends: the camera is let go if nothing else watches
    layer.remove();
    document.removeEventListener('keydown', onKey);
    onClose();
  }
  layer.querySelector('.close').addEventListener('click', close);
  document.addEventListener('keydown', onKey);

  return {
    close,
    isOpen: () => layer.isConnected,
    // `landed`: what the page says of the landing, and `isOff` if it is off.
    update(next, doing, landed = null) {
      marks.update(next);
      const desk = marks.marks.touch && marks.marks.touch.desk;
      layer.querySelector('.fullview-point b').textContent = desk
        ? `${desk[0].toFixed(1)}, ${desk[1].toFixed(1)}` : '—';
      const line = layer.querySelector('.fullview-point .landed');
      line.textContent = landed ? landed.text : '';
      line.classList.toggle('off', Boolean(landed && landed.isOff));
      layer.querySelector('.fullview-doing').textContent = doing || '';
    },
  };
}
