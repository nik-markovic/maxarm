// The desk calibration page: bring the cup down onto the desk, record where it touched, fit the plane, try it out.
//
// The arm's target is a base plus offsets: base x, y is the grid point last clicked, base z the height column;
// the X and Y fine sliders and the nudges add to x and y. A grid click puts the offsets back to 0. Whatever moves
// the arm from elsewhere (a point, a desk test, RESET) moves the base, so the offsets stay where the user left
// them. A move the arm refuses puts every control back where it was when the arm last went where they said.
//
// Heights are true z here and in every request. TRUE / RAW only changes how arm z is shown and typed in: RAW is
// the firmware's own. Heights above the desk are the same either way.

import { $, api, fmt, mountHeader, report, showHeader, startPolling, toast } from '/common/page.js';
import { GridView } from './grid.js';

const STEPS = [10, 20];
const STEP_DEFAULT = 20;
const FACINGS = ['arm up', 'arm right', 'arm down', 'arm left'];
const SCREEN = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const ARROWS = { up: '▲', down: '▼', left: '◀', right: '▶' };
const SAME_MM = 0.005;
// The height column, the owner's (PILOT-falcon): the slider runs from 80 down to 40 in 1 mm steps, for fine
// control near the desk; SAFE sends the arm up to 120; each ▼ lets the slider go 2 mm lower, without end. The
// desk is at about 44 under most arms, so the slider is yellow from 44 down to 42, and red below.
const HEIGHT = { top: 80, bottom: 40, safe: 120, lower: 2, caution: 44, danger: 42 };
const OFFSETS = {
  fit: 'the plane fitted through the points', saved: 'the saved desk.json', default: 'the default desk',
};

mountHeader('desk');
const limits = await api('limits');
const remembered = JSON.parse(localStorage.getItem('desk-view') || '{}');
const ui = {
  step: STEPS.includes(remembered.step) ? remembered.step : STEP_DEFAULT,
  rotation: remembered.rotation || 0,
  offsetsFrom: remembered.offsetsFrom in OFFSETS ? remembered.offsetsFrom : 'fit',
  isRaw: false,
  bottom: HEIGHT.bottom,
  base: { x: limits.home[0], y: limits.home[1], z: limits.home[2] },
  fine: { x: 0, y: 0 },
  testHeight: limits.desk.test_height,
};
let state = null;
let inFlight = 0;
let accepted = '';
let isDragging = false;
let pointsShown = '';

const grid = new GridView($('grid'), limits, jump);
const target = () => [ui.base.x + ui.fine.x, ui.base.y + ui.fine.y, ui.base.z];
const controls = () => JSON.stringify({ base: ui.base, fine: ui.fine });
accepted = controls();

// How much higher the cup really is than the firmware says, at (x, y): the library's correction (maxarm/raw.py).
const trueMinusRaw = (x, y) => limits.true_z.per_mm_of_reach * (Math.hypot(x, y) - limits.true_z.level_reach);
// Arm z as shown and typed in: true, or under RAW the firmware's own.
const shownZ = ([x, y, z]) => (ui.isRaw ? z - trueMinusRaw(x, y) : z);
const typedZ = (x, y, z) => (ui.isRaw ? z + trueMinusRaw(x, y) : z);
// What the arm says is out of date: a command is on its way, or the servos are still to be read after one.
const isReading = () => inFlight > 0 || !state || state.is_reading;
// The saved desk only while there is one.
const offsetsFrom = () => (ui.offsetsFrom === 'saved' && !(state && state.desk.saved.exists) ? 'fit' : ui.offsetsFrom);

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

// Moves the base so that base + offsets is where the arm was sent.
function follow(sent) {
  const [x, y, z] = target();
  if (Math.abs(x - sent[0]) < SAME_MM && Math.abs(y - sent[1]) < SAME_MM && Math.abs(z - sent[2]) < SAME_MM) return;
  ui.base = { x: sent[0] - ui.fine.x, y: sent[1] - ui.fine.y, z: sent[2] };
}

const move = () => {
  const [x, y, z] = target();
  return command('move', { x, y, z }, 'Move', { isRevertible: true });
};

function jump(x, y) {
  ui.base.x = x;
  ui.base.y = y;
  ui.fine = { x: 0, y: 0 };
  command('jump', { x, y }, 'Over to the grid point', { isRevertible: true });
}

// One step of X or Y fine; past the end of its slider the base takes it, so a nudge always moves.
function nudge(axis, delta) {
  if (Math.abs(ui.fine[axis] + delta) <= ui.step) ui.fine[axis] += delta;
  else ui.base[axis] += delta;
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
  if (state) renderDesk();
  grid.update({
    step: ui.step, rotation: ui.rotation, target: target(), isReading: isReading(),
    points: state ? state.desk.points.map((point) => ({ ...point, deviation: point.off[offsetsFrom()] })) : [],
    sent: state ? state.sent : null, says: state ? state.measured : null,
  });
}

function renderHeight() {
  const shown = shownZ(target());
  const slider = $('z-slider');
  slider.min = ui.bottom;
  slider.max = HEIGHT.top;
  if (!isDragging) slider.value = Math.round(shown);
  $('z-out').textContent = fmt(shown);
  $('z-mode').textContent = ui.isRaw ? 'raw' : 'true';
  $('z-safe').classList.toggle('on', shown >= HEIGHT.safe - 0.5);
  $('z-top').textContent = HEIGHT.top;
  $('z-bottom').textContent = ui.bottom;
  const at = (z) => `${Math.min(100, Math.max(0, (100 * (HEIGHT.top - z)) / (HEIGHT.top - ui.bottom)))}%`;
  $('z-scale').style.background = `linear-gradient(to bottom, var(--line-strong) ${at(HEIGHT.caution)}, `
    + `var(--yellow) ${at(HEIGHT.caution)} ${at(HEIGHT.danger)}, var(--red) ${at(HEIGHT.danger)})`;
}

function renderControls() {
  for (const row of document.querySelectorAll('.fine')) {
    const axis = row.dataset.axis;
    const slider = row.querySelector('input');
    slider.min = -ui.step;
    slider.max = ui.step;
    if (!isDragging) slider.value = ui.fine[axis];
    row.querySelector('output').textContent = fmt(ui.fine[axis], 0, true);
  }
  for (const button of document.querySelectorAll('.pad [data-dir]')) {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    button.textContent = `${ARROWS[button.dataset.dir]} ${dx > 0 ? '+X' : dx < 0 ? '−X' : dy > 0 ? '+Y' : '−Y'}`;
  }
  $('facing').textContent = FACINGS[ui.rotation / 90];
  for (const button of $('steps').children) button.classList.toggle('on', Number(button.textContent) === ui.step);
  for (const button of $('heights').children) {
    button.classList.toggle('on', (button.dataset.mode === 'raw') === ui.isRaw);
  }
  if (state && !isDragging && inFlight === 0) {
    $('cup').value = state.cup;
    $('suction').checked = state.suction;
  }
  $('cup-out').textContent = `${fmt(Number($('cup').value), 0, true)}°`;
  $('record').disabled = isReading();
  $('save').disabled = isReading();
}

function renderReadout() {
  const shown = (point) => [point[0], point[1], shownZ(point)];
  const aim = shown(target());
  const says = state && state.measured ? state.measured : null;
  ['x', 'y', 'z'].forEach((axis, index) => {
    $(`target-${axis}`).textContent = fmt(aim[index]);
    const cell = $(`says-${axis}`);
    const off = says ? Math.abs(says[index] - state.sent[index]) : 0;
    cell.textContent = says ? fmt(shown(says)[index], 0) : '—';
    cell.className = `num says ${off >= 2 ? 'far' : off >= 1 ? 'off' : ''}`;
  });
  $('readout').classList.toggle('stale', isReading());
  $('z-head').textContent = ui.isRaw ? 'Z raw' : 'Z';
  const desk = state ? state.desk : null;
  const parts = [];
  if (desk && desk.fit_height !== null) parts.push(`the fit of the points <b>${fmt(desk.fit_height, 1, true)}</b>`);
  if (desk && desk.position) {
    const which = desk.saved.exists ? 'the saved desk' : 'the default desk';
    parts.push(`${which} <b>${fmt(desk.position[2], 1, true)}</b>`);
  }
  $('desk-line').innerHTML = parts.length ? `Cup above the desk, mm: ${parts.join(' · ')}` : '';
}

function renderDesk() {
  const desk = state.desk;
  const from = offsetsFrom();
  renderPoints(desk.points, from);
  const { x_span: xNeed, y_span: yNeed } = limits.desk;
  $('span').innerHTML = desk.span ? ['x', 'y'].map((axis) => {
    const [low, high] = desk.span[axis];
    const need = axis === 'x' ? xNeed : yNeed;
    return `${axis.toUpperCase()} ${fmt(low, 0)} … ${fmt(high, 0)}: <b>${fmt(high - low, 0)}</b> mm
      <span class="chip ${high - low >= need ? 'ok' : 'bad'}">≥ ${need}</span>`;
  }).join(' &nbsp; ') : 'Range covered: no points yet';
  $('fit').innerHTML = desk.fit ? `Plane through the points: tilted <b>${fmt(desk.fit.tilt_deg, 2)}°</b>`
    : 'Plane: needs 3 points, not in a row';
  for (const button of $('offsets-from').children) {
    button.classList.toggle('on', button.dataset.from === from);
    button.disabled = button.dataset.from === 'saved' && !desk.saved.exists;
  }
  const spread = desk.off[from];
  $('offsets').innerHTML = spread ? `<b>${fmt(spread.rms, 2)}</b> mm RMS, <b>${fmt(spread.worst, 2)}</b> at worst`
    : from === 'fit' && desk.fit ? 'a 4th point shows how well they agree' : '';
  $('save-state').innerHTML = desk.save_problem ? `<span class="chip bad">not ready</span> ${desk.save_problem}`
    : '<span class="chip ok">ready to save</span>';
  const ignored = desk.saved.summary ? `config/desk.json ${desk.saved.summary}` : 'nothing saved yet';
  $('desk-file').textContent = desk.saved.exists ? `config/desk.json: ${desk.saved.summary}`
    : `the default desk, arm z ${fmt(limits.desk.default_z, 0)} and level: ${ignored}`;
  $('glide-hint').innerHTML = desk.is_gliding
    ? `Next point test <b>glides</b> there, slowly and level at ${fmt(ui.testHeight)} mm.`
    : 'Next point test comes down from above, the way grab() does.';
  $('test-height-out').textContent = `${fmt(ui.testHeight)} mm`;
  if (desk.position && !isDragging && inFlight === 0) {
    ['x', 'y', 'z'].forEach((axis, index) => {
      const field = $(`desk-${axis}`);
      if (document.activeElement !== field) field.value = fmt(desk.position[index]);
    });
  }
}

function renderPoints(points, from) {
  const shown = JSON.stringify([points, from, ui.isRaw]);
  if (shown === pointsShown) return;
  pointsShown = shown;
  const [warn, bad] = limits.desk.deviation;
  const rank = (value) => (value === null ? '' : Math.abs(value) <= warn ? 'ok' : Math.abs(value) <= bad ? 'warn' : 'bad');
  $('points-z').textContent = ui.isRaw ? 'Z raw' : 'Z';
  $('off-head').textContent = `off ${from}`;
  $('off-head').title = `How far above ${OFFSETS[from]} each point is`;
  $('points').tBodies[0].innerHTML = points.length ? points.map((point) => `
    <tr>
      <td><span class="badge ${rank(point.off[from])}">${point.id}</span></td>
      <td class="num">${fmt(point.x)}</td><td class="num">${fmt(point.y)}</td>
      <td class="num">${fmt(shownZ([point.x, point.y, point.z]))}</td>
      <td class="num faint">${point.sent ? fmt(shownZ(point.sent)) : '—'}</td>
      <td class="num dev ${rank(point.off[from])}">${fmt(point.off[from], 2, true)}</td>
      <td class="actions">
        <button class="btn small" data-act="visit" data-id="${point.id}" title="Back to it in arm space, from above">arm</button>
        <button class="btn small" data-act="test" data-id="${point.id}"
          title="Over it in desk space, at the point test height">desk</button>
        <button class="btn small ghost" data-act="delete" data-id="${point.id}" title="Forget it">×</button>
      </td>
    </tr>`).join('') : '<tr class="empty"><td colspan="7">Bring the cup down onto the desk, then RECORD.</td></tr>';
}

// Screen direction to arm direction, undoing the view's quarter turn.
function armStep([sx, sy]) {
  const angle = (ui.rotation * Math.PI) / 180;
  const c = Math.round(Math.cos(angle));
  const s = Math.round(Math.sin(angle));
  return [sx * c + sy * s, -sx * s + sy * c];
}

function keepView() {
  const view = { step: ui.step, rotation: ui.rotation, offsetsFrom: ui.offsetsFrom };
  localStorage.setItem('desk-view', JSON.stringify(view));
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

for (const row of document.querySelectorAll('.fine')) {
  const axis = row.dataset.axis;
  onSlide(row.querySelector('input'), (value) => { ui.fine[axis] = value; }, move);
}
for (const button of document.querySelectorAll('.pad [data-dir]')) {
  button.addEventListener('click', () => {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    if (dx) nudge('x', dx);
    else nudge('y', dy);
  });
}

for (const button of $('heights').children) {
  button.addEventListener('click', () => {
    ui.isRaw = button.dataset.mode === 'raw';
    render();
  });
}
const { per_mm_of_reach: perMm, level_reach: level } = limits.true_z;
$('heights-tip').innerHTML = `<b>True z</b> is where the cup really is. <b>Raw z</b> is the arm firmware's own
  number, which is off by an amount that grows with the distance from the base: 100 mm out the firmware says
  ${fmt(perMm * (level - 100))} mm higher than the cup is, at ${limits.reach} mm ${fmt(perMm * (limits.reach - level))}
  mm lower. The library corrects every height it sends and reads:
  <code>true z = raw z + ${perMm} × (reach − ${level} mm)</code>
  ${level} mm out the two agree. RAW only changes the numbers shown: the arm moves the same, and touch points,
  offsets and SAVE always use true z.`;

$('cup').min = -limits.cup;
$('cup').max = limits.cup;
onSlide($('cup'), () => {}, () => command('cup', { angle: Number($('cup').value) }, 'Cup'));
$('suction').addEventListener('change', () => command('suction', { on: $('suction').checked }, 'Suction'));

$('steps').innerHTML = STEPS.map((step) => `<button>${step}</button>`).join('');
for (const button of $('steps').children) {
  button.addEventListener('click', () => {
    ui.step = Number(button.textContent);
    for (const axis of ['x', 'y']) {
      const kept = Math.max(-ui.step, Math.min(ui.step, ui.fine[axis]));
      ui.base[axis] += ui.fine[axis] - kept;
      ui.fine[axis] = kept;
    }
    keepView();
    render();
  });
}
$('rotate').addEventListener('click', () => {
  ui.rotation = (ui.rotation + 90) % 360;
  keepView();
  render();
});

$('offsets-from').innerHTML = Object.entries(OFFSETS)
  .map(([from, what]) => `<button data-from="${from}" title="How far above ${what} each point is">${from}</button>`)
  .join('');
for (const button of $('offsets-from').children) {
  button.addEventListener('click', () => {
    ui.offsetsFrom = button.dataset.from;
    keepView();
    render();
  });
}

$('record').addEventListener('click', () => command('desk/record', {}, 'RECORD'));
$('save').addEventListener('click', () => command('desk/save', {}, 'SAVE CALIBRATION'));
$('clear').addEventListener('click', () => {
  const count = state ? state.desk.points.length : 0;
  if (count && window.confirm(`Forget all ${count} touch points?`)) command('desk/clear', {}, 'CLEAR');
});
$('points').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-act]');
  if (!button) return;
  const id = Number(button.dataset.id);
  if (button.dataset.act === 'visit') command('desk/visit', { id }, `Back to #${id}`);
  if (button.dataset.act === 'test') command('desk/test-point', { id, height: ui.testHeight }, `Point test #${id}`);
  if (button.dataset.act === 'delete') command('desk/delete', { id }, `#${id}`);
});

const testHeight = $('test-height');
testHeight.max = limits.desk.test_height;
testHeight.value = ui.testHeight;
onSlide(testHeight, (value) => { ui.testHeight = value; },
  () => command('desk/test-height', { height: ui.testHeight }, 'Point test height'));

// The desk fields as a point, or null while one is blank: an empty field must not read as 0.
function deskPoint() {
  const fields = ['x', 'y', 'z'].map((axis) => $(`desk-${axis}`).value);
  if (fields.some((text) => text.trim() === '' || !Number.isFinite(Number(text)))) {
    toast('Fill in X, Y and Z first', 'warn');
    return null;
  }
  const [x, y, z] = fields.map(Number);
  return { x, y, z };
}
for (const axis of ['x', 'y', 'z']) {
  $(`desk-${axis}`).addEventListener('change', () => {
    const point = deskPoint();
    if (point) command('desk/move', point, 'Desk move');
  });
}
$('test-input').addEventListener('click', () => {
  const point = deskPoint();
  if (point) command('desk/test-input', { ...point, hold: Number($('desk-hold').value) }, 'TEST INPUT');
});

render();
startPolling((next) => show(next));
