// The camera calibration page: aim the camera, set each cube down under the cup and record where the cup is, then
// CALIBRATE, which parks the arm and takes the frame the calibration is fitted to.
//
// Everything is desk space: millimetres, z the height over the desk. The arm's target is the grid point last
// clicked and the height column. Whatever moves the arm from elsewhere (a cube's "go", CALIBRATE, RESET) moves the
// target with it, and a move the arm refuses puts it back where the arm last went.

import { $, api, fmt, mountHeader, openModal, report, showHeader, startPolling } from '/common/page.js';
import { FACINGS, STEPS, STEP_DEFAULT, rememberView, rememberedView } from '/common/grid.js';
import { CubeGrid } from './grid.js';
import { showGuide } from './guide.js';

const SAME_MM = 0.005;
// The height column, the owner's (PILOT-gecko): the slider runs from 70 down to -2 over the desk, yellow from the
// desk down a millimetre and red below, where the cup presses into it. The notch over a cube's top catches the
// slider from a millimetre either side. SAFE is well clear of the cubes anywhere in reach.
const HEIGHT = { top: 70, bottom: -2, safe: 80, lower: 2, higher: 20, caution: 0, danger: -1, snap: 1 };
// Where the cubes might go, desk mm (PILOT-gecko). Red goes on the side nearer the camera: either of its two.
const SUGGESTED = [
  { colour: 'blue', x: -160, y: -220 }, { colour: 'green', x: 160, y: -220 },
  { colour: 'red', x: -160, y: 0 }, { colour: 'red', x: 160, y: 0 },
];
const GUIDE_SEEN = 'camera-guide-seen';
const FEED = '/api/camera/feed';
const FEED_RETRY_MS = 3000;
const FEED_NOTES = {
  idle: 'starting the camera…', starting: 'starting the camera…', snapshot: 'taking the snapshot…',
};
const FEED_BADGES = { live: '● LIVE', snapshot: '◉ SNAPSHOT', failed: 'NO CAMERA' };

mountHeader('camera');
const limits = await api('limits');
const { notch: NOTCH, colours: COLOURS } = limits.camera;
const remembered = rememberedView();
const ui = {
  step: STEPS.includes(remembered.step) ? remembered.step : STEP_DEFAULT,
  rotation: remembered.rotation || 0,
  top: HEIGHT.top,
  bottom: HEIGHT.bottom,
  target: { x: limits.home[0], y: limits.home[1], z: HEIGHT.safe },
};
let state = null;
let inFlight = 0;
let accepted = JSON.stringify(ui.target);
let isDragging = false;
let isLost = false;
let checkShown = null;

const grid = new CubeGrid($('grid'), limits, jump);
const target = () => [ui.target.x, ui.target.y, ui.target.z];
const isAtNotch = (z) => Math.abs(z - NOTCH) < 0.05;
// What the arm says is out of date: a command is on its way, or the servos are still to be read after one.
const isReading = () => inFlight > 0 || !state || state.is_reading;
const isBusy = () => inFlight > 0 || !state || Boolean(state.doing) || state.queued > 0;

// --- talking to the app ------------------------------------------------------------------------------

async function command(path, body, what, { isRevertible = false } = {}) {
  const asked = JSON.stringify(ui.target);
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
  else if (isRevertible && inFlight === 0) ui.target = JSON.parse(accepted);
  report(reply, what);
  show(reply.state || state, true);
  return reply;
}

function show(next, isReply = false) {
  if (next && state && next.stamp < state.stamp) return;        // a poll answered before a newer reply
  if (next && isLost) connectFeed();                            // the app is back: its live view is a new one
  isLost = !next;
  state = next;
  showHeader(state);
  if (state && !isDragging && inFlight === 0 && (isReply || (!state.doing && !state.queued))) {
    follow(state.camera.position);
    accepted = JSON.stringify(ui.target);
  }
  render();
}

// The target becomes where the arm was sent.
function follow([x, y, z]) {
  const now = ui.target;
  if (Math.abs(now.x - x) < SAME_MM && Math.abs(now.y - y) < SAME_MM && Math.abs(now.z - z) < SAME_MM) return;
  ui.target = { x, y, z };
}

const move = () => command('camera/move', { ...ui.target }, 'Move', { isRevertible: true });

function jump(x, y) {
  ui.target = { ...ui.target, x, y };
  command('camera/jump', { x, y }, 'Over to the grid point', { isRevertible: true });
}

function setHeight(z) {
  ui.target.z = z;
  move();
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  renderHeight();
  renderReadout();
  renderView();
  if (state) {
    renderCubes();
    renderCamera();
  }
  grid.update({
    step: ui.step, rotation: ui.rotation, target: target(), isReading: isReading(), suggested: SUGGESTED,
    cubes: state ? state.camera.cubes : {}, sent: state ? state.sent : null,
    says: state ? state.camera.measured : null,
  });
}

function renderHeight() {
  const { z } = ui.target;
  const slider = $('z-slider');
  slider.min = ui.bottom;
  slider.max = ui.top;
  if (!isDragging) slider.value = Math.round(z);
  $('z-out').textContent = fmt(z);
  $('z-out').classList.toggle('at-notch', isAtNotch(z));
  $('z-notch').classList.toggle('on', isAtNotch(z));
  $('z-safe').classList.toggle('on', z >= HEIGHT.safe - 0.5);
  $('z-top').textContent = ui.top;
  $('z-bottom').textContent = ui.bottom;
  const share = (value) => Math.min(1, Math.max(0, (ui.top - value) / (ui.top - ui.bottom)));
  const at = (value) => `${100 * share(value)}%`;
  $('z-scale').style.background = `linear-gradient(to bottom, var(--line-strong) ${at(HEIGHT.caution)}, `
    + `var(--yellow) ${at(HEIGHT.caution)} ${at(HEIGHT.danger)}, var(--red) ${at(HEIGHT.danger)})`;
  const mark = $('z-notch-mark');
  mark.hidden = NOTCH > ui.top || NOTCH < ui.bottom;
  mark.style.top = `calc(8px + (100% - 16px) * ${share(NOTCH)})`;
}

function renderReadout() {
  const aim = target();
  const says = state ? state.camera.measured : null;
  const sent = state ? state.camera.position : null;
  ['x', 'y', 'z'].forEach((axis, index) => {
    $(`target-${axis}`).textContent = fmt(aim[index]);
    const cell = $(`says-${axis}`);
    const off = says && sent ? Math.abs(says[index] - sent[index]) : 0;
    cell.textContent = says ? fmt(says[index]) : '—';
    cell.className = `num says ${off >= 2 ? 'far' : off >= 1 ? 'off' : ''}`;
  });
  $('readout').classList.toggle('stale', isReading());
}

function renderView() {
  $('facing').textContent = FACINGS[ui.rotation / 90];
  for (const button of $('steps').children) button.classList.toggle('on', Number(button.textContent) === ui.step);
}

function renderCubes() {
  const { cubes } = state.camera;
  const count = COLOURS.filter((colour) => cubes[colour]).length;
  $('progress').innerHTML = `<span class="chip ${count === COLOURS.length ? 'ok' : ''}">${count} of
    ${COLOURS.length} recorded</span>`;
  for (const row of $('cube-rows').children) {
    const cube = cubes[row.dataset.colour];
    row.classList.toggle('recorded', Boolean(cube));
    row.querySelector('.cube-at').innerHTML = cube
      ? `X <b>${fmt(cube.x)}</b> Y <b>${fmt(cube.y)}</b> Z <b>${fmt(cube.z)}</b>` : 'not recorded yet';
    const record = row.querySelector('[data-act="record"]');
    record.disabled = isReading();
    record.textContent = cube ? '● RECORD AGAIN' : '● RECORD';
    for (const act of ['visit', 'forget']) row.querySelector(`[data-act="${act}"]`).disabled = !cube;
  }
}

function renderCamera() {
  const { cubes, check, calibration, feed } = state.camera;
  $('feed-badge').textContent = FEED_BADGES[feed.status] || '';
  $('feed-badge').className = `feed-badge ${feed.status}`;
  $('feed-note').textContent = feed.status === 'failed' ? feed.detail : FEED_NOTES[feed.status] || '';
  $('feed-device').textContent = feed.status === 'live' ? feed.detail : '';
  const missing = COLOURS.filter((colour) => !cubes[colour]);
  $('calibrate').disabled = missing.length > 0 || isBusy();
  $('calibrate').title = missing.length ? `Record ${missing.join(', ')} first`
    : 'Park the arm, then take the frame the calibration is fitted to';
  $('check').disabled = isBusy();
  $('calibration-file').textContent = `config/${calibration.summary}`;
  $('check-box').classList.toggle('empty', !check);
  if (check && check.stamp !== checkShown) {
    $('check-image').src = `/api/camera/check.jpg?v=${check.stamp}`;
    checkShown = check.stamp;
  }
  $('check-info').innerHTML = check
    ? `<b>Calibration check</b><br>taken ${check.taken}, ${check.size[0]} × ${check.size[1]}<br>`
      + 'click it to see it full size'
    : '<b>No check image yet</b><br>CALIBRATE or CHECK CALIBRATION takes one, with the arm out of the way.';
}

// An isometric cube in the cube's colour: lit from above.
function cubeIcon(colour) {
  return `<svg class="cube-icon ${colour}" viewBox="0 0 32 32" aria-hidden="true">
    <path class="top" d="M16 3 28 9.5 16 16 4 9.5Z"/><path class="left" d="M4 9.5 16 16v13L4 22.5Z"/>
    <path class="right" d="M28 9.5 16 16v13l12-6.5Z"/></svg>`;
}

function buildCubeRows() {
  const where = (colour) => SUGGESTED.filter((spot) => spot.colour === colour)
    .map((spot) => `${spot.x}, ${spot.y}`).join(' or ');
  $('cube-rows').innerHTML = COLOURS.map((colour) => `
    <div class="cube-row" data-colour="${colour}">
      ${cubeIcon(colour)}
      <div class="cube-name"><b>${colour}</b><small>suggested ${where(colour)}</small></div>
      <div class="cube-at num"></div>
      <div class="cube-actions">
        <button class="btn record" data-act="record" title="Keep where the arm says the cup is, over this cube">
          ● RECORD</button>
        <button class="btn small" data-act="visit" title="Back over it, from above">go</button>
        <button class="btn small ghost" data-act="forget" title="Forget it">×</button>
      </div>
    </div>`).join('');
}

function showCheck() {
  const { body } = openModal('viewer');
  const image = document.createElement('img');
  image.src = $('check-image').src;
  image.title = 'Click for actual size, and back';
  image.addEventListener('click', () => image.classList.toggle('actual'));
  body.append(image);
}

// The live view only while the page is in sight: a hidden tab lets the camera go.
function connectFeed() {
  if (document.hidden) $('feed').removeAttribute('src');
  else $('feed').src = `${FEED}?t=${Date.now()}`;
}

// --- wiring ----------------------------------------------------------------------------------------

function onSlide(slider, apply, send) {
  slider.addEventListener('input', () => { isDragging = true; apply(Number(slider.value)); render(); });
  slider.addEventListener('change', () => { isDragging = false; apply(Number(slider.value)); send(); });
}

onSlide($('z-slider'), (value) => { ui.target.z = Math.abs(value - NOTCH) <= HEIGHT.snap ? NOTCH : value; }, move);
$('z-up').addEventListener('click', () => setHeight(ui.target.z + 1));
$('z-down').addEventListener('click', () => setHeight(ui.target.z - 1));
$('z-safe').innerHTML = `${HEIGHT.safe}<small>SAFE</small>`;
$('z-safe').addEventListener('click', () => setHeight(HEIGHT.safe));
$('z-notch').innerHTML = `${cubeIcon('notch')}<span>${fmt(NOTCH, 0)}<small>CUBE</small></span>`;
$('z-notch').addEventListener('click', () => setHeight(NOTCH));
$('z-notch-mark').firstElementChild.textContent = fmt(NOTCH, 0);
$('z-lower').addEventListener('click', () => { ui.bottom -= HEIGHT.lower; render(); });
$('z-higher').addEventListener('click', () => { ui.top += HEIGHT.higher; render(); });

$('steps').innerHTML = STEPS.map((step) => `<button>${step}</button>`).join('');
for (const button of $('steps').children) {
  button.addEventListener('click', () => {
    ui.step = Number(button.textContent);
    rememberView(ui.step, ui.rotation);
    render();
  });
}
$('rotate').addEventListener('click', () => {
  ui.rotation = (ui.rotation + 90) % 360;
  rememberView(ui.step, ui.rotation);
  render();
});

buildCubeRows();
$('cube-rows').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-act]');
  if (!button) return;
  const { colour } = button.closest('.cube-row').dataset;
  if (button.dataset.act === 'record') command('camera/record', { colour }, `RECORD ${colour}`);
  if (button.dataset.act === 'visit') command('camera/visit', { colour }, `Back over ${colour}`);
  if (button.dataset.act === 'forget') command('camera/forget', { colour }, `Forget ${colour}`);
});

$('calibrate').addEventListener('click', () => command('camera/calibrate', {}, 'CALIBRATE'));
$('check').addEventListener('click', () => command('camera/check', {}, 'CHECK CALIBRATION'));
$('check-box').addEventListener('click', () => { if (checkShown !== null) showCheck(); });
$('guide').addEventListener('click', showGuide);
if (!localStorage.getItem(GUIDE_SEEN)) {
  localStorage.setItem(GUIDE_SEEN, '1');
  showGuide();
}

$('feed').addEventListener('error', () => { if (!document.hidden) setTimeout(connectFeed, FEED_RETRY_MS); });
document.addEventListener('visibilitychange', connectFeed);
connectFeed();
render();
startPolling((next) => show(next));
