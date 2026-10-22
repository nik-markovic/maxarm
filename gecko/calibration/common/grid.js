// The view from above: reach, the arm-base limit, the arm, and whatever a page marks on it (drawMarks).
// Arm and desk x and y are the same, so it serves both spaces. A click picks the grid point nearest to it.
// The view turns in quarters; at 0 the arm reaches up the screen, +x to the right.

const CENTRE = [0, -65];      // mm: the middle of the reachable fan, which then fills the square
const HALF = 290;             // mm from the centre to an edge
const TURNTABLE_MM = 30;      // the base's own disc, as drawn

export const STEPS = [10, 20];
export const STEP_DEFAULT = 20;
export const FACINGS = ['arm up', 'arm right', 'arm down', 'arm left'];
// The grid step and the view's turn are the same on every page: the arm stands the same way whichever is open.
const VIEW_KEY = 'grid-view';

export function rememberedView() {
  return JSON.parse(localStorage.getItem(VIEW_KEY) || '{}');
}

export function rememberView(step, rotation) {
  localStorage.setItem(VIEW_KEY, JSON.stringify({ step, rotation }));
}

export const INK = {
  grid: 'rgba(138, 151, 166, 0.07)', major: 'rgba(138, 151, 166, 0.17)', axis: 'rgba(138, 151, 166, 0.38)',
  label: 'rgba(138, 151, 166, 0.55)', reach: '#f0524a', reachMin: '#f08c3c', fan: 'rgba(65, 195, 99, 0.07)',
  base: '#0c1116', baseEdge: '#3a4553', hatch: 'rgba(80, 95, 112, 0.35)',
  arm: 'rgba(205, 215, 228, 0.16)', armLine: 'rgba(205, 215, 228, 0.5)',
  green: '#41c363', says: '#ffffff', saysStale: 'rgba(255, 255, 255, 0.3)', neutral: '#8d9aa9', warn: '#e6b84a',
  bad: '#f0524a',
};

export class GridView {
  constructor(canvas, limits, onPick) {
    this.canvas = canvas;
    this.limits = limits;
    this.onPick = onPick;
    this.ctx = canvas.getContext('2d');
    this.size = 0;
    this.hover = null;
    this.view = { step: STEP_DEFAULT, rotation: 0, sent: null, says: null, target: null, isReading: false };
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener('pointermove', (event) => { this.hover = this.snapped(event); this.draw(); });
    canvas.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
    canvas.addEventListener('click', (event) => this.onPick(...this.snapped(event)));
  }

  update(view) {
    Object.assign(this.view, view);
    this.draw();
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.size = this.canvas.clientWidth;
    this.canvas.width = Math.round(this.size * ratio);
    this.canvas.height = Math.round(this.size * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.scale = this.size / (2 * HALF);
    this.draw();
  }

  // Arm (x, y) to canvas pixels: the view's quarter turn is clockwise on screen.
  toScreen(x, y) {
    const [c, s] = this.turn();
    const dx = x - CENTRE[0];
    const dy = y - CENTRE[1];
    return [this.size / 2 + (dx * c - dy * s) * this.scale, this.size / 2 + (dx * s + dy * c) * this.scale];
  }

  toArm(px, py) {
    const [c, s] = this.turn();
    const rx = (px - this.size / 2) / this.scale;
    const ry = (py - this.size / 2) / this.scale;
    return [CENTRE[0] + rx * c + ry * s, CENTRE[1] - rx * s + ry * c];
  }

  turn() {
    const angle = (this.view.rotation * Math.PI) / 180;
    return [Math.round(Math.cos(angle)), Math.round(Math.sin(angle))];
  }

  snapped(event) {
    const box = this.canvas.getBoundingClientRect();
    const [x, y] = this.toArm(event.clientX - box.left, event.clientY - box.top);
    const step = this.view.step;
    return [Math.round(x / step) * step, Math.round(y / step) * step];
  }

  draw() {
    if (!this.size) return;
    this.ctx.clearRect(0, 0, this.size, this.size);
    this.drawGrid();
    this.drawReach();
    this.drawBase();
    this.drawMarks();
    this.drawArm();
    this.drawHover();
    this.drawCompass();
  }

  drawGrid() {
    const { ctx } = this;
    const low = [CENTRE[0] - HALF, CENTRE[1] - HALF];
    const high = [CENTRE[0] + HALF, CENTRE[1] + HALF];
    ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
    ctx.lineWidth = 1;
    for (const axis of [0, 1]) {
      for (let value = Math.ceil(low[axis] / this.view.step) * this.view.step; value <= high[axis];
        value += this.view.step) {
        const isMajor = Math.abs(value % 100) < 1e-9;
        const ends = axis === 0 ? [[value, low[1]], [value, high[1]]] : [[low[0], value], [high[0], value]];
        const [a, b] = ends.map(([x, y]) => this.toScreen(x, y));
        ctx.strokeStyle = value === 0 ? INK.axis : isMajor ? INK.major : INK.grid;
        this.line(a, b);
        if (!isMajor) continue;
        const isUpright = Math.abs(a[0] - b[0]) < 0.5;
        ctx.fillStyle = INK.label;
        ctx.textAlign = isUpright ? 'center' : 'left';
        ctx.textBaseline = isUpright ? 'bottom' : 'middle';
        const text = `${'xy'[axis]} ${value}`;
        if (isUpright) ctx.fillText(text, a[0], this.size - 4);
        else ctx.fillText(text, 5, a[1]);
      }
    }
  }

  drawReach() {
    const { ctx } = this;
    const { reach, reach_min: reachMin, fan } = this.limits;
    ctx.beginPath();
    this.trace(this.arc(reach, -fan, fan).concat(this.arc(reachMin, fan, -fan)));
    ctx.closePath();
    ctx.fillStyle = INK.fan;
    ctx.fill();
    ctx.lineWidth = 2;
    for (const [radius, ink] of [[reach, INK.reach], [reachMin, INK.reachMin]]) {
      ctx.strokeStyle = ink;
      ctx.beginPath();
      this.trace(this.arc(radius, -fan, fan));
      ctx.stroke();
    }
    ctx.strokeStyle = INK.reach;
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1.2;
    for (const bearing of [-fan, fan]) {
      this.line(this.atBearing(reachMin, bearing), this.atBearing(reach, bearing));
    }
    ctx.setLineDash([]);
  }

  drawBase() {
    const { ctx } = this;
    const half = this.limits.base_half;
    const corners = [[-half, -half], [half, -half], [half, half], [-half, half]].map(([x, y]) => this.toScreen(x, y));
    ctx.save();
    ctx.beginPath();
    this.trace(corners);
    ctx.closePath();
    ctx.fillStyle = INK.base;
    ctx.fill();
    ctx.clip();
    ctx.strokeStyle = INK.hatch;
    ctx.lineWidth = 1;
    for (let offset = -this.size; offset < this.size; offset += 7) {
      this.line([offset, 0], [offset + this.size, this.size]);
    }
    ctx.restore();
    ctx.beginPath();
    this.trace(corners);
    ctx.closePath();
    ctx.strokeStyle = INK.baseEdge;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const [cx, cy] = this.toScreen(0, 0);
    ctx.beginPath();
    ctx.arc(cx, cy, TURNTABLE_MM * this.scale, 0, 2 * Math.PI);
    ctx.fillStyle = '#18202a';
    ctx.fill();
    ctx.stroke();
  }

  // A page's own marks, between the base and the arm: touch points, cubes. Nothing here.
  drawMarks() {}

  drawArm() {
    const { ctx } = this;
    const { sent, says, target } = this.view;
    const base = this.toScreen(0, 0);
    ctx.lineCap = 'round';
    if (sent) {
      const cup = this.toScreen(sent[0], sent[1]);
      ctx.strokeStyle = INK.arm;
      ctx.lineWidth = 14;
      this.line(base, cup);
      ctx.strokeStyle = INK.armLine;
      ctx.lineWidth = 2;
      this.line(base, cup);
      ctx.beginPath();
      ctx.arc(base[0], base[1], 6, 0, 2 * Math.PI);
      ctx.fillStyle = INK.armLine;
      ctx.fill();
    }
    if (target) {
      const [x, y] = this.toScreen(target[0], target[1]);
      if (sent && Math.hypot(target[0] - sent[0], target[1] - sent[1]) > 0.5) {
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = INK.green;
        ctx.lineWidth = 1;
        this.line(this.toScreen(sent[0], sent[1]), [x, y]);
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = INK.green;
      ctx.lineWidth = 2;
      ctx.shadowColor = INK.green;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 1.5;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        this.line([x + dx * 6, y + dy * 6], [x + dx * 16, y + dy * 16]);
      }
    }
    if (says) {
      const [x, y] = this.toScreen(says[0], says[1]);
      ctx.fillStyle = this.view.isReading ? INK.saysStale : INK.says;
      ctx.shadowColor = INK.says;
      ctx.shadowBlur = this.view.isReading ? 0 : 8;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  drawHover() {
    if (!this.hover) return;
    const { ctx } = this;
    const [x, y] = this.toScreen(...this.hover);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.stroke();
    const text = `${this.hover[0]}, ${this.hover[1]}`;
    ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace';
    const width = ctx.measureText(text).width + 12;
    const left = Math.min(x + 12, this.size - width - 4);
    const top = Math.max(y - 30, 4);
    ctx.fillStyle = 'rgba(18, 23, 30, 0.92)';
    ctx.strokeStyle = 'rgba(65, 195, 99, 0.6)';
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

  drawCompass() {
    const { ctx } = this;
    const origin = [this.size - 44, 44];
    const zero = this.toScreen(0, 0);
    ctx.font = '600 10.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const [name, x, y] of [['+x', 1, 0], ['+y', 0, 1]]) {
      const tip = this.toScreen(x, y);
      const direction = [tip[0] - zero[0], tip[1] - zero[1]].map((value) => value / this.scale);
      const end = [origin[0] + direction[0] * 18, origin[1] + direction[1] * 18];
      ctx.strokeStyle = INK.label;
      ctx.lineWidth = 1.5;
      this.line(origin, end);
      ctx.fillStyle = INK.label;
      ctx.fillText(name, origin[0] + direction[0] * 28, origin[1] + direction[1] * 28);
    }
  }

  // --- small drawing helpers ---

  atBearing(radius, bearing) {
    const angle = (bearing * Math.PI) / 180;
    return this.toScreen(radius * Math.sin(angle), -radius * Math.cos(angle));
  }

  arc(radius, from, to) {
    const count = Math.max(2, Math.ceil(Math.abs(to - from) / 2));
    return Array.from({ length: count + 1 }, (_, index) => this.atBearing(radius, from + ((to - from) * index) / count));
  }

  trace(points) {
    points.forEach(([x, y], index) => (index ? this.ctx.lineTo(x, y) : this.ctx.moveTo(x, y)));
  }

  line(a, b) {
    this.ctx.beginPath();
    this.ctx.moveTo(a[0], a[1]);
    this.ctx.lineTo(b[0], b[1]);
    this.ctx.stroke();
  }
}
