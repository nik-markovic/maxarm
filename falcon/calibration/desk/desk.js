// The desk calibration page: bring the cup down onto the desk, record where it touched, fit the plane, try it out.
//
// The arm's target is a base plus offsets: base x, y is the grid point last clicked, base z the height slider;
// the fine sliders, the nudges and the tenth slider add to them. Whatever moves the arm from elsewhere (a point,
// a desk test, RESET) moves the base, so the offsets stay where the user left them. A move the arm refuses puts
// every control back where it was when the arm last went where they said.

import { $, api, fmt, mountHeader, report, showHeader, startPolling, toast } from '/common/page.js';
import { GridView } from './grid.js';

const STEPS = [20, 25, 50];
const FACINGS = ['arm up', 'arm right', 'arm down', 'arm left'];
const SCREEN = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const ARROWS = { up: '▲', down: '▼', left: '◀', right: '▶' };
const SAME_MM = 0.005;

mountHeader('desk');
const limits = await api('limits');
const saved = JSON.parse(localStorage.getItem('desk-view') || '{}');
const ui = {
  step: STEPS.includes(saved.step) ? saved.step : STEPS[0],
  rotation: saved.rotation || 0,
  isLow: false,
  base: { x: limits.home[0], y: limits.home[1], z: limits.home[2] },
  fine: { x: 0, y: 0, z: 0, f: 0 },
  testHeight: limits.desk.test_height,
};
let state = null;
let inFlight = 0;
let accepted = '';
let isDragging = false;
let pointsShown = '';

const grid = new GridView($('grid'), limits, jump);
const target = () => [ui.base.x + ui.fine.x, ui.base.y + ui.fine.y, ui.base.z + ui.fine.z + ui.fine.f];
const controls = () => JSON.stringify({ base: ui.base, fine: ui.fine });
accepted = controls();

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
  ui.base = { x: sent[0] - ui.fine.x, y: sent[1] - ui.fine.y, z: sent[2] - ui.fine.z - ui.fine.f };
}

const move = () => {
  const [x, y, z] = target();
  return command('move', { x, y, z }, 'Move', { isRevertible: true });
};

function jump(x, y) {
  ui.base.x = x;
  ui.base.y = y;
  command('jump', { x: x + ui.fine.x, y: y + ui.fine.y }, 'Over to the grid point', { isRevertible: true });
}

// One step of an offset; past the end of its slider the base takes it, so a nudge always moves.
function nudge(axis, delta) {
  const reach = Number(document.querySelector(`.fine[data-axis="${axis}"] input`).max);
  const next = Math.round((ui.fine[axis] + delta) * 10) / 10;
  if (Math.abs(next) <= reach + 1e-9) ui.fine[axis] = next;
  else ui.base[axis === 'f' ? 'z' : axis] += delta;
  move();
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  renderControls();
  renderReadout();
  if (state) renderDesk();
  grid.update({
    step: ui.step, rotation: ui.rotation, fine: [ui.fine.x, ui.fine.y], target: target(),
    points: state ? state.desk.points : [], sent: state ? state.sent : null, says: state ? state.measured : null,
  });
}

function renderControls() {
  const coarse = $('z-coarse');
  coarse.min = ui.isLow ? limits.z_low : limits.z_min;
  coarse.max = limits.z_max;
  if (!isDragging) coarse.value = Math.round(ui.base.z);
  $('z-coarse-out').textContent = fmt(ui.base.z, 0);
  $('zcol').classList.toggle('low', ui.isLow);
  $('low').classList.toggle('on', ui.isLow);
  for (const row of document.querySelectorAll('.fine')) {
    const axis = row.dataset.axis;
    const slider = row.querySelector('input');
    if (axis === 'x' || axis === 'y') {
      slider.min = -ui.step;
      slider.max = ui.step;
    }
    if (!isDragging) slider.value = ui.fine[axis];
    row.querySelector('output').textContent = fmt(ui.fine[axis], axis === 'f' ? 1 : 0, true);
  }
  for (const button of document.querySelectorAll('.pad [data-dir]')) {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    button.textContent = `${ARROWS[button.dataset.dir]} ${dx > 0 ? '+X' : dx < 0 ? '−X' : dy > 0 ? '+Y' : '−Y'}`;
  }
  $('facing').textContent = FACINGS[ui.rotation / 90];
  for (const button of $('steps').children) button.classList.toggle('on', Number(button.textContent) === ui.step);
  if (state && !isDragging && inFlight === 0) {
    $('cup').value = state.cup;
    $('suction').checked = state.suction;
  }
  $('cup-out').textContent = `${fmt(Number($('cup').value), 0, true)}°`;
}

function renderReadout() {
  const aim = target();
  ['x', 'y', 'z'].forEach((axis, index) => {
    $(`target-${axis}`).textContent = fmt(aim[index]);
    const says = $(`says-${axis}`);
    const value = state && state.measured ? state.measured[index] : null;
    const off = value === null ? 0 : Math.abs(value - state.sent[index]);
    says.textContent = fmt(value, 0);
    says.className = `num ${off >= 2 ? 'far' : off >= 1 ? 'off' : ''}`;
  });
  const desk = state ? state.desk : null;
  const parts = [];
  if (desk && desk.fit_height !== null) parts.push(`fit of the points: cup <b>${fmt(desk.fit_height, 1, true)}</b> mm`);
  if (desk && desk.position) parts.push(`desk.json: cup <b>${fmt(desk.position[2], 1, true)}</b> mm`);
  $('desk-line').innerHTML = parts.length ? `Above the desk — ${parts.join(' · ')}` : 'Above the desk: record 3 points';
}

function renderDesk() {
  const desk = state.desk;
  renderPoints(desk.points);
  const { x_span: xNeed, y_span: yNeed } = limits.desk;
  $('span').innerHTML = desk.span ? ['x', 'y'].map((axis) => {
    const [low, high] = desk.span[axis];
    const need = axis === 'x' ? xNeed : yNeed;
    return `${axis.toUpperCase()} ${fmt(low, 0)} … ${fmt(high, 0)}: <b>${fmt(high - low, 0)}</b> mm
      <span class="chip ${high - low >= need ? 'ok' : 'bad'}">≥ ${need}</span>`;
  }).join(' &nbsp; ') : 'Range covered: no points yet';
  const fit = desk.fit;
  const { elbow_points: elbowPoints, elbow_reach: elbowReach } = limits.desk;
  $('fit').innerHTML = !fit ? 'Plane: needs 3 points, not in a row' : `Plane: tilted <b>${fmt(fit.tilt_deg, 2)}°</b>`
    + (fit.rms === null ? ' — a 4th point shows how well they agree'
      : `, points off it by <b>${fmt(fit.rms, 2)}</b> mm RMS, <b>${fmt(fit.worst, 2)}</b> at worst`)
    + '<br>' + (fit.is_elbow_fitted
      ? `Elbow servo <b>${fmt(fit.elbow_deg, 2, true)}°</b> off the firmware's zero`
        + ` (a plane alone: <b>${fmt(fit.rms_plane_only, 2)}</b> mm RMS)`
      : `Elbow offset: needs ${elbowPoints} points, ${elbowReach} mm apart in reach`);
  $('save-state').innerHTML = desk.save_problem ? `<span class="chip bad">not ready</span> ${desk.save_problem}`
    : '<span class="chip ok">ready to SAVE</span>';
  const test = $('test');
  test.classList.toggle('unavailable', !desk.saved.exists);
  $('desk-file').textContent = desk.saved.exists ? `config/desk.json: ${desk.saved.summary}` : 'no config/desk.json yet';
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

function renderPoints(points) {
  const shown = JSON.stringify([points, state.desk.saved.exists]);
  if (shown === pointsShown) return;
  pointsShown = shown;
  const [warn, bad] = limits.desk.deviation;
  const rank = (value) => (value === null ? '' : Math.abs(value) <= warn ? 'ok' : Math.abs(value) <= bad ? 'warn' : 'bad');
  const canTest = state.desk.saved.exists ? '' : 'disabled';
  $('points').tBodies[0].innerHTML = points.length ? points.map((point) => `
    <tr>
      <td><span class="badge ${rank(point.deviation)}">${point.id}</span></td>
      <td class="num">${fmt(point.x)}</td><td class="num">${fmt(point.y)}</td><td class="num">${fmt(point.z)}</td>
      <td class="num faint">${point.sent ? fmt(point.sent[2]) : '—'}</td>
      <td class="num dev ${rank(point.deviation)}">${fmt(point.deviation, 2, true)}</td>
      <td class="actions">
        <button class="btn small" data-act="visit" data-id="${point.id}" title="Back to it in arm space, from above">arm</button>
        <button class="btn small" data-act="test" data-id="${point.id}" ${canTest}
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
  localStorage.setItem('desk-view', JSON.stringify({ step: ui.step, rotation: ui.rotation }));
}

// --- wiring ----------------------------------------------------------------------------------------

function onSlide(slider, apply, send) {
  slider.addEventListener('input', () => { isDragging = true; apply(Number(slider.value)); render(); });
  slider.addEventListener('change', () => { isDragging = false; apply(Number(slider.value)); send(); });
}

onSlide($('z-coarse'), (value) => { ui.base.z = value; }, move);
for (const row of document.querySelectorAll('.fine')) {
  const axis = row.dataset.axis;
  const slider = row.querySelector('input');
  onSlide(slider, (value) => { ui.fine[axis] = value; }, move);
  for (const button of row.querySelectorAll('button')) {
    button.addEventListener('click', () => nudge(axis, Number(button.dataset.sign) * Number(slider.step)));
  }
}
for (const button of document.querySelectorAll('.pad [data-dir]')) {
  button.addEventListener('click', () => {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    if (dx) nudge('x', dx);
    else nudge('y', dy);
  });
}
for (const button of document.querySelectorAll('.zpad [data-dir]')) {
  button.addEventListener('click', () => nudge('z', button.dataset.dir === 'z+' ? 1 : -1));
}
$('low').addEventListener('click', () => {
  ui.isLow = !ui.isLow;
  if (!ui.isLow && ui.base.z < limits.z_min) {
    ui.base.z = limits.z_min;
    move();
  }
  render();
});

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

$('record').addEventListener('click', () => command('desk/record', {}, 'RECORD'));
$('save').addEventListener('click', () => command('desk/save', {}, 'SAVE'));
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
