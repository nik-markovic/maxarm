// Marks drawn over the camera's picture through the calibration (shares.js): a canvas laid over the live view or a
// snapshot, sized with it, and what every page's marks draw with -- the reach on the desk, the arm, lines, areas,
// labels. A page's own marks extend ViewMarks and draw in drawMarks(), from `this.marks`, which update() fills:
// `matrix` (desk to shares of the view), `deskToTrue` (the arm's x/y error), `limits` (/api/limits), and their own.

import { circle, toView } from './shares.js';

export const INK = {
  reach: '#f0524a', reachMin: '#f08c3c', green: '#41c363', yellow: '#ffd23f', white: '#ffffff', shadow: '#05070a',
};
export const MONO = '"JetBrains Mono", ui-monospace, monospace';

export class ViewMarks {
  constructor(canvas, marks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = 0;
    this.height = 0;
    // The arm's marks: target, sent and says, desk mm; isReading while what it says is out of date; cupRadius, mm.
    this.marks = { matrix: null, deskToTrue: null, limits: null, target: null, says: null, isReading: false,
      cupRadius: 10.5, ...marks };
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
    this.ctx.clearRect(0, 0, this.width, this.height);
    if (!this.width || !this.marks.matrix || !this.marks.limits) return;
    this.drawMarks();
  }

  // A page's own marks.
  drawMarks() {}

  // Desk point to canvas pixels; null behind the camera.
  at(point) {
    const share = toView(this.marks.matrix, point);
    return share && [share[0] * this.width, share[1] * this.height];
  }

  // A canvas point to shares of the view.
  share([x, y]) {
    return [x / this.width, y / this.height];
  }

  // The arm's reach on the desk: 95 and 270 mm, orange and red, as on the grid.
  drawReach(alpha = 0.75) {
    const { reach, reach_min: reachMin, fan } = this.marks.limits;
    for (const [radius, ink] of [[reach, INK.reach], [reachMin, INK.reachMin]]) {
      this.polyline(this.arc(radius, -fan, fan), ink, 2, alpha);
    }
  }

  // The arm's target, the cup's rim at its height, faint, with a line down to the desk under it; where its servos
  // say the cup is, a white dot.
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

  // The cup's rim round a desk point, `height` over the desk, its true size, in canvas pixels.
  ring([x, y], height = 0, radius = this.marks.cupRadius) {
    const { matrix, deskToTrue } = this.marks;
    return circle(matrix, [x, y], radius, height, deskToTrue)
      .map((share) => share && [share[0] * this.width, share[1] * this.height]);
  }

  // Canvas points along an arc round the base axis on the desk, `from` to `to` degrees from straight out.
  arc(radius, from, to, height = 0) {
    const count = Math.max(2, Math.ceil(Math.abs(to - from) / 2));
    return Array.from({ length: count + 1 }, (_, index) => {
      const angle = ((from + ((to - from) * index) / count) * Math.PI) / 180;
      return this.at([radius * Math.sin(angle), -radius * Math.cos(angle), height]);
    });
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

  // A closed shape filled, if every point of it is in front of the camera.
  fill(points, ink, alpha) {
    if (!points.length || points.some((point) => !point)) return;
    const { ctx } = this;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = ink;
    ctx.beginPath();
    points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fill();
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
