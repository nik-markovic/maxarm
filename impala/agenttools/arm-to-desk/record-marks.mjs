// What the tests page's live view draws for ARM TO DESK's point, phase by phase, recorded from a stand-in canvas:
// every stroke, fill and text with its colour, as JSON on stdout, for draw-marks.py to paint on a frame.
//
//   node --experimental-default-type=module impala/agenttools/arm-to-desk/record-marks.mjs SCENE.json
//
// SCENE.json: { matrix, deskToTrue, size: [w, h], limits, target, touch: { u, v, desk }, says, phases: [...] }.

import { readFileSync } from 'node:fs';
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = pathToFileURL(path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../calibration')).href;
register(`data:text/javascript,${encodeURIComponent(`
export async function resolve(specifier, context, next) {
  return next(specifier.startsWith('/') ? ${JSON.stringify(root)} + specifier : specifier, context);
}`)}`);
globalThis.document = { getElementById: () => null, querySelector: () => null, addEventListener: () => {} };
globalThis.window = globalThis;
globalThis.devicePixelRatio = 1;
globalThis.ResizeObserver = class { observe() {} };

class Recorder {
  constructor() {
    this.records = [];
    this.paths = [];
    Object.assign(this, { globalAlpha: 1, strokeStyle: '#000', fillStyle: '#000', lineWidth: 1, font: '', dash: [] });
  }
  clearRect() { this.records = []; }
  setTransform() {}
  setLineDash(dash) { this.dash = dash; }
  beginPath() { this.paths = []; }
  closePath() {}
  moveTo(x, y) { this.paths.push([[x, y]]); }
  lineTo(x, y) { this.paths[this.paths.length - 1].push([x, y]); }
  arc(x, y, r) { this.paths.push(Array.from({ length: 25 }, (_, i) => [x + r * Math.cos(i * Math.PI / 12), y + r * Math.sin(i * Math.PI / 12)])); }
  roundRect(x, y, w, h) { this.paths.push([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]); }
  measureText(text) { return { width: text.length * 7.2 }; }
  stroke() { this.records.push({ kind: 'stroke', ink: this.strokeStyle, alpha: this.globalAlpha, width: this.lineWidth, dash: this.dash.length > 0, paths: this.paths }); }
  fill() { this.records.push({ kind: 'fill', ink: this.fillStyle, alpha: this.globalAlpha, paths: this.paths }); }
  fillText(text, x, y) { this.records.push({ kind: 'text', ink: this.fillStyle, alpha: this.globalAlpha, text, at: [x, y] }); }
}

const scene = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const { LiveMarks } = await import(`${root}/tests/view.js`);
const ctx = new Recorder();
const canvas = { getContext: () => ctx, clientWidth: scene.size[0], clientHeight: scene.size[1] };
const marks = new LiveMarks(canvas);
marks.resize();
const out = {};
for (const phase of scene.phases) {
  const touch = { ...scene.touch, phase: phase.phase, says: phase.phase === 'read' ? scene.says : null };
  marks.update({ matrix: scene.matrix, deskToTrue: scene.deskToTrue, limits: scene.limits, target: phase.target,
    touch, hover: phase.hover || null, showGuides: true, found: null, refine: null, next: null });
  out[phase.phase] = ctx.records;
}
console.log(JSON.stringify(out));
