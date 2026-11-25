// The desk page: the arm moved by hand, in its own coordinates, to get to know it.
//
// The arm's target: x, y the grid point last clicked plus the nudges since, z the height column. Whatever moves
// the arm from elsewhere (a point, RESET) becomes the target. A move the arm refuses puts every
// control back where it was when the arm last went where they said.
//
// Heights are true z here and in every request. TRUE / RAW only changes how arm z is shown and typed in: RAW is
// the firmware's own. Heights above the desk are the same either way.

import { $, api, copyReadout, fmt, mountHeader, report, setText, showHeader, startPolling } from '/common/page.js';
import { GridView, STEPS, STEP_DEFAULT, rememberView, rememberedView } from '/common/grid.js';

const SCREEN = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const ARROWS = { up: '▲', down: '▼', left: '◀', right: '▶' };
// A press of the pad, mm: under TRUE along true x or y, under RAW along the firmware's.
const NUDGE_MM = { true: 0.5, raw: 0.25 };
const SAME_MM = 0.005;
// The height column: the slider runs from 90 down to 40 in 1 mm steps, for fine control near the desk and up to
// over a cube; SAFE sends the arm up to 90; each ▼ lets the slider go 2 mm lower and each ▲ 20 mm higher, without
// end. The desk is at about 44 under most arms, so the slider is yellow from 44 down to 42, and red below.
const HEIGHT = { top: 90, bottom: 40, safe: 90, lower: 2, higher: 20, caution: 44, danger: 42 };
// Whether the cup sealed when the pump went on (maxarm/seal.py), shown while it is on. Not known once the arm has
// moved: the servos pull on the supply the seal is told by.
const SEAL = {
  sealed: ['SEALED', 'ok', 'the cup is stuck to what is under it'],
  loose: ['LOOSE', 'warn', 'the cup draws air: nothing is sealed'],
  uncertain: ['UNCERTAIN', 'bad', 'between sealed and loose: the cup may be only just touching'],
  unknown: ['UNKNOWN', 'bad', 'the arm\'s supply could not be read'],
};

mountHeader('desk');
const limits = await api('limits');
const remembered = rememberedView();
const ui = {
  step: STEPS.includes(remembered.step) ? remembered.step : STEP_DEFAULT,
  rotation: remembered.rotation || 0,
  isRaw: false,
  top: HEIGHT.top,
  bottom: HEIGHT.bottom,
  base: { x: limits.home[0], y: limits.home[1], z: Math.min(limits.home[2], HEIGHT.safe) },
};
let state = null;
let inFlight = 0;
let accepted = '';
let isDragging = false;

const grid = new GridView($('grid'), limits, pick);
const target = () => [ui.base.x, ui.base.y, ui.base.z];
const controls = () => JSON.stringify({ base: ui.base });
accepted = controls();

// The library's true position, as maxarm/raw.py has it: the raw point of a true one, and the true point of a raw one.
const TRUE_Z = limits.true_z;
const TRUE_XY = limits.true_xy;
const tooTallPerMm2 = (out) => (TRUE_Z.tall_at_stack + TRUE_Z.tall_per_mm_of_reach * out) / TRUE_Z.stack ** 2;
// A table of [at, value] rows read at `at`: straight between its rows, held beyond its ends.
function between(rows, at) {
  if (at <= rows[0][0]) return rows[0][1];
  for (let row = 1; row < rows.length; row += 1) {
    const [near, nearValue] = rows[row - 1];
    const [far, farValue] = rows[row];
    if (at <= far) return nearValue + ((farValue - nearValue) * (at - near)) / (far - near);
  }
  return rows[rows.length - 1][1];
}
// The desk's wave by the firmware's reach; the base's turn by the firmware's bearing.
const deskWave = (reach) => between(TRUE_Z.desk_wave, reach);
const furtherOut = (height) => TRUE_XY.reach_at_desk
  - TRUE_XY.reach_less_per_mm * Math.min(Math.max(height, 0), TRUE_XY.reach_top);
const furtherRound = (bearing) => (Math.PI / 180) * between(TRUE_XY.turn, (bearing * 180) / Math.PI);
// x and y from reach and bearing: 0 straight out (-y), positive toward +x.
const fromPolar = (reach, bearing) => [reach * Math.sin(bearing), -reach * Math.cos(bearing)];
// The camera calibration's correction on top of the formula (maxarm/mesh.py), as the server had it when the page
// loaded: further out (mm) and further round (degrees) by the points' pattern over the whole desk -- the turn's
// widening held from a little past the points' bearings -- and what it leaves at each point straight across the mesh's triangles,
// beyond them its nearest edge's fading out. Null: the formula alone. A CALIBRATE or SAVE since shows after a reload.
const MESH = TRUE_XY.mesh;
const TO_RAW_ROUNDS = 8;
function meshValue(bearing, reach) {
  const { pattern: [out, round, axisX, axisY, widening], per, span: [low, high] } = MESH;
  if (!MESH.left.length) return [0, 0];
  const across = axisX * Math.cos(bearing) + axisY * Math.sin(bearing);
  const held = Math.min(Math.max((bearing * 180) / Math.PI, low), high);
  const [leftOut, leftRound] = meshLeft(bearing, reach);
  return [out + axisX * Math.sin(bearing) - axisY * Math.cos(bearing) + leftOut,
    round + (widening * held) / per + ((across / reach) * 180) / Math.PI + leftRound];
}
// What the pattern leaves.
function meshLeft(bearing, reach) {
  const { plane, left, triangles, edges, scale, fade } = MESH;
  const here = [bearing * scale, reach];
  const blend = (weights, corners) => [0, 1].map((k) => corners.reduce((sum, c, i) => sum + weights[i] * left[c][k], 0));
  for (const [a, b, c] of triangles) {
    const [ax, ay] = plane[a];
    const [ux, uy, vx, vy] = [plane[b][0] - ax, plane[b][1] - ay, plane[c][0] - ax, plane[c][1] - ay];
    const det = ux * vy - uy * vx;
    const s = ((here[0] - ax) * vy - (here[1] - ay) * vx) / det;
    const t = (ux * (here[1] - ay) - uy * (here[0] - ax)) / det;
    if (Math.min(1 - s - t, s, t) >= -1e-9) return blend([1 - s - t, s, t], [a, b, c]);
  }
  let best = { gap: Math.hypot(plane[0][0] - here[0], plane[0][1] - here[1]), value: left[0] };   // one point: its own
  for (const [a, b] of edges) {
    const [dx, dy] = [plane[b][0] - plane[a][0], plane[b][1] - plane[a][1]];
    const share = Math.min(Math.max(((here[0] - plane[a][0]) * dx + (here[1] - plane[a][1]) * dy)
      / Math.max(dx * dx + dy * dy, 1e-12), 0), 1);
    const gap = Math.hypot(plane[a][0] + share * dx - here[0], plane[a][1] + share * dy - here[1]);
    if (gap < best.gap) best = { gap, value: blend([1 - share, share], [a, b]) };
  }
  const keep = Math.max(0, 1 - best.gap / fade);
  return best.value.map((value) => keep * value);
}
// The formula's true x, y corrected by the mesh.
function meshed([x, y]) {
  const reach = Math.hypot(x, y);
  if (!MESH || reach < 1) return [x, y];
  const bearing = Math.atan2(x, -y);
  const [out, round] = meshValue(bearing, reach);
  return fromPolar(reach + out, bearing + (round * Math.PI) / 180);
}
// The formula's true x, y that the mesh corrects to [x, y]: found by repeating.
function unmeshed([x, y]) {
  if (!MESH) return [x, y];
  let estimate = [x, y];
  for (let round = 0; round < TO_RAW_ROUNDS; round += 1) {
    const there = meshed(estimate);
    estimate = [estimate[0] + x - there[0], estimate[1] + y - there[1]];
  }
  return estimate;
}
// The true x and y of the firmware's, at true height z.
function trueXY([x, y], z) {
  const reach = Math.hypot(x, y);
  if (reach === 0) return [x, y];
  const bearing = Math.atan2(x, -y);
  return meshed(fromPolar(reach + furtherOut(Math.max(z - TRUE_Z.desk_z, 0)), bearing + furtherRound(bearing)));
}
function rawPoint([trueX, trueY, z]) {
  const height = Math.max(z - TRUE_Z.desk_z, 0);
  const [x, y] = unmeshed([trueX, trueY]);
  const reach = Math.hypot(x, y);
  const bearing = Math.atan2(x, -y);
  let rawBearing = bearing;
  for (let round = 0; round < TRUE_XY.bearing_rounds; round += 1) rawBearing = bearing - furtherRound(rawBearing);
  const [rawX, rawY] = reach === 0 ? [x, y] : fromPolar(Math.max(reach - furtherOut(height), 0), rawBearing);
  const out = Math.hypot(rawX, rawY) - TRUE_Z.level_reach;
  return [rawX, rawY, z - TRUE_Z.per_mm_of_reach * out - deskWave(Math.hypot(rawX, rawY))
    + tooTallPerMm2(out) * height ** 2];
}
// The true z of a raw z typed in, with the true x and y held: raw z is tried and corrected until it is the one typed.
function trueZ(x, y, z) {
  let height = z - TRUE_Z.desk_z;
  for (let round = 0; round < 6; round += 1) height += z - rawPoint([x, y, TRUE_Z.desk_z + height])[2];
  return TRUE_Z.desk_z + height;
}
// Arm positions as shown, and heights as typed in: true, or under RAW the firmware's own.
const shownPoint = (point) => (ui.isRaw ? rawPoint(point) : point);
const shownZ = (point) => shownPoint(point)[2];
const typedZ = (x, y, z) => (ui.isRaw ? trueZ(x, y, z) : z);
// What the arm says is out of date: a command is on its way, or the servos are still to be read after one.
const isReading = () => inFlight > 0 || !state || state.is_reading;

// --- talking to the app ------------------------------------------------------------------------------

async function command(path, body, what, { isRevertible = false } = {}) {
  const asked = controls();
  inFlight += 1;
  render();
  let reply;
  try {
    reply = await api(path, body);
  } catch {
    reply = { status: 'error', detail: 'the calibration app is not answering' };
  }
  inFlight -= 1;
  if (reply.status === 'superseded') return reply;
  if (isRevertible && reply.status === 'done') accepted = asked;
  else if (isRevertible && inFlight === 0) Object.assign(ui, JSON.parse(accepted));
  report(reply, what);
  show(reply.state || state, true);
  return reply;
}

function show(next, isReply = false) {
  if (next && state && next.stamp < state.stamp) return;        // a poll answered before a newer reply
  state = next;
  showHeader(state);
  if (state && !isDragging && inFlight === 0 && (isReply || (!state.doing && !state.queued))) {
    follow(state.sent);
    accepted = controls();
  }
  render();
}

// Makes where the arm was sent the target; the height no higher than SAFE, so that after RESET the height column
// stands at SAFE, where the next click or nudge goes.
function follow(sent) {
  const [x, y, z] = target();
  const height = Math.min(sent[2], HEIGHT.safe);
  if (Math.abs(x - sent[0]) < SAME_MM && Math.abs(y - sent[1]) < SAME_MM && Math.abs(z - height) < SAME_MM) return;
  ui.base = { x: sent[0], y: sent[1], z: height };
}

const move = () => {
  const [x, y, z] = target();
  return command('move', { x, y, z }, 'Move', { isRevertible: true });
};

// Down onto the target as the library lands: PRECISE.
const land = () => {
  const [x, y, z] = target();
  return command('land', { x, y, z }, 'Land', { isRevertible: true });
};

// A grid click: down onto the grid point, at the height column's height -- no higher than SAFE, so that a click
// after RESET, the column up where reset parks the arm, comes down to SAFE -- as PRECISE lands.
function pick(x, y) {
  ui.base.x = x;
  ui.base.y = y;
  ui.base.z = Math.min(ui.base.z, HEIGHT.safe);
  land();
}

// A press of the pad: the target NUDGE_MM along x or y -- under RAW the firmware's x or y, its true height kept --
// a plain move.
function nudge(axis, delta) {
  if (ui.isRaw) {
    const point = rawPoint(target());
    point[axis === 'x' ? 0 : 1] += delta * NUDGE_MM.raw;
    [ui.base.x, ui.base.y] = trueXY(point, ui.base.z);
  } else {
    ui.base[axis] += delta * NUDGE_MM.true;
  }
  move();
}

function raise(delta) {
  ui.base.z += delta;
  move();
}

// The height as shown, true or raw, at the target's x and y.
function setShownZ(z) {
  const [x, y] = target();
  ui.base.z = typedZ(x, y, z);
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  renderHeight();
  renderControls();
  renderReadout();
  grid.update({
    step: ui.step, rotation: ui.rotation, target: target(), isReading: isReading(),
    sent: state ? state.sent : null, says: state ? state.measured : null,
  });
}

function renderHeight() {
  const shown = shownZ(target());
  const slider = $('z-slider');
  slider.min = ui.bottom;
  slider.max = ui.top;
  if (!isDragging) slider.value = Math.round(shown);
  $('z-out').textContent = fmt(shown);
  $('z-mode').textContent = ui.isRaw ? 'raw' : 'true';
  $('z-safe').classList.toggle('on', shown >= HEIGHT.safe - 0.5);
  $('z-top').textContent = ui.top;
  $('z-bottom').textContent = ui.bottom;
  const at = (z) => `${Math.min(100, Math.max(0, (100 * (ui.top - z)) / (ui.top - ui.bottom)))}%`;
  $('z-scale').style.background = `linear-gradient(to bottom, var(--line-strong) ${at(HEIGHT.caution)}, `
    + `var(--yellow) ${at(HEIGHT.caution)} ${at(HEIGHT.danger)}, var(--red) ${at(HEIGHT.danger)})`;
}

function renderControls() {
  for (const button of document.querySelectorAll('.pad [data-dir]')) {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    button.textContent = `${ARROWS[button.dataset.dir]} ${dx > 0 ? '+X' : dx < 0 ? '−X' : dy > 0 ? '+Y' : '−Y'}`;
  }
  $('pad-hub').textContent = String(ui.isRaw ? NUDGE_MM.raw : NUDGE_MM.true);
  for (const button of $('steps').children) button.classList.toggle('on', Number(button.textContent) === ui.step);
  for (const button of $('heights').children) {
    button.classList.toggle('on', (button.dataset.mode === 'raw') === ui.isRaw);
  }
  if (state && !isDragging && inFlight === 0) {
    $('cup').value = state.cup;
    $('suction').checked = state.suction;
  }
  $('cup-out').textContent = `${fmt(Number($('cup').value), 0, true)}°`;
  renderSeal();
}

function renderSeal() {
  const line = $('seal-line');
  if (!state || !state.suction) {
    line.innerHTML = '';
  } else if (!state.seal) {
    line.innerHTML = '<span class="hint">Seal not known: the arm has moved since the pump went on. Suction off and on '
      + 'again to check.</span>';
  } else {
    const [word, kind, meaning] = SEAL[state.seal.state];
    const { dip, pulled } = state.seal;
    const told = [dip === null ? '' : `pump dip ${fmt(dip, 1)}`,
      pulled === null ? '' : `pulled down ${fmt(pulled, 1)} mm`].filter(Boolean).join(', ');
    line.innerHTML = `<span class="chip ${kind}" title="${told}">${word}</span><span class="hint">${meaning}</span>`;
  }
}

function renderReadout() {
  const shown = shownPoint;
  const aim = shown(target());
  const says = state && state.measured ? state.measured : null;
  ['x', 'y', 'z'].forEach((axis, index) => {
    setText($(`target-${axis}`), fmt(aim[index], 2));
    const cell = $(`says-${axis}`);
    const off = says ? Math.abs(says[index] - state.sent[index]) : 0;
    setText(cell, says ? fmt(shown(says)[index], ui.isRaw ? 2 : 0) : '—');     // RAW: the servos' mean, to 0.01
    const kind = `num says ${off >= 2 ? 'far' : off >= 1 ? 'off' : ''}`;
    if (cell.className !== kind) cell.className = kind;
  });
  $('readout').classList.toggle('stale', isReading());
  for (const axis of ['x', 'y']) setText($(`${axis}-head`), ui.isRaw ? `${axis.toUpperCase()} raw` : axis.toUpperCase());
  setText($('z-head'), ui.isRaw ? 'Z raw' : 'Z');
  const height = state ? state.desk_height : null;
  $('desk-line').innerHTML = height === null || height === undefined ? ''
    : `Cup above the desk: <b>${fmt(height, 1, true)}</b> mm`;
}

// Screen direction to arm direction, undoing the view's quarter turn.
function armStep([sx, sy]) {
  const angle = (ui.rotation * Math.PI) / 180;
  const c = Math.round(Math.cos(angle));
  const s = Math.round(Math.sin(angle));
  return [sx * c + sy * s, -sx * s + sy * c];
}

function keepView() {
  rememberView(ui.step, ui.rotation);
}

// --- wiring ----------------------------------------------------------------------------------------

function onSlide(slider, apply, send) {
  slider.addEventListener('input', () => { isDragging = true; apply(Number(slider.value)); render(); });
  slider.addEventListener('change', () => { isDragging = false; apply(Number(slider.value)); send(); });
}

onSlide($('z-slider'), setShownZ, move);
$('z-up').addEventListener('click', () => raise(1));
$('z-down').addEventListener('click', () => raise(-1));
$('z-safe').innerHTML = `${HEIGHT.safe}<small>SAFE</small>`;
$('z-safe').addEventListener('click', () => {
  setShownZ(HEIGHT.safe);
  move();
});
$('z-lower').addEventListener('click', () => {
  ui.bottom -= HEIGHT.lower;
  render();
});
$('z-higher').addEventListener('click', () => {
  ui.top += HEIGHT.higher;
  render();
});

for (const button of document.querySelectorAll('.pad [data-dir]')) {
  button.addEventListener('click', () => {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    if (dx) nudge('x', dx);
    else nudge('y', dy);
  });
}
$('precise').addEventListener('click', () => land());
$('copy').addEventListener('click', () => copyReadout(ui.isRaw ? 'ARM RAW' : 'ARM TRUE'));


for (const button of $('heights').children) {
  button.addEventListener('click', () => {
    ui.isRaw = button.dataset.mode === 'raw';
    render();
  });
}
$('heights-tip').innerHTML = `<b>RAW</b> is where the arm reports the cup is. It is a few millimeters off, more so
  far out and high up. <b>TRUE</b> is that corrected to real distances on the desk. Only the numbers shown change: the
  arm moves the same either way.`;

$('cup').min = -limits.cup;
$('cup').max = limits.cup;
onSlide($('cup'), () => {}, () => command('cup', { angle: Number($('cup').value) }, 'Cup'));
$('suction').addEventListener('change', () => command('suction', { on: $('suction').checked }, 'Suction'));

$('steps').innerHTML = STEPS.map((step) => `<button>${step}</button>`).join('');
for (const button of $('steps').children) {
  button.addEventListener('click', () => {
    ui.step = Number(button.textContent);
    keepView();
    render();
  });
}
$('rotate').addEventListener('click', () => {
  ui.rotation = (ui.rotation + 90) % 360;
  keepView();
  render();
});

render();
startPolling((next) => show(next));
