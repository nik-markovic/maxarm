// The camera calibration page: aim the camera, set each cube down under the cup and record where the cup is, then
// CALIBRATE, which parks the arm, takes the frame and fits the camera to the cubes in it. VALIDATE CALIBRATION draws
// the saved calibration on a new frame.
//
// Everything is desk space: millimetres, z the height over the desk. The arm's target is the grid point last
// clicked and the height column. Whatever moves the arm from elsewhere (a cube's "go", CALIBRATE, RESET) moves the
// target with it, and a move the arm refuses puts it back where the arm last went. The height column is shared
// with the tests page (common/height.js).

import { $, api, fmt, mountHeader, openModal, report, showHeader, startPolling } from '/common/page.js';
import { FACINGS, STEPS, STEP_DEFAULT, rememberView, rememberedView } from '/common/grid.js';
import { HeightColumn, SAFE_MM, cubeIcon } from '/common/height.js';
import { CubeGrid } from './grid.js';
import { showGuide } from './guide.js';

const SAME_MM = 0.005;
// Where the cubes might go, desk mm (PILOT-gecko). Red goes on the side nearer the camera: either of its two.
const SUGGESTED = [
  { colour: 'blue', x: -140, y: -220 }, { colour: 'green', x: 140, y: -220 },
  { colour: 'red', x: -140, y: 0 }, { colour: 'red', x: 140, y: 0 },
];
const GUIDE_SEEN = 'camera-guide-seen';
const FEED = '/api/camera/feed';
const FEED_RETRY_MS = 3000;
const FEED_NOTES = {
  idle: 'starting the camera…', starting: 'starting the camera…',
};
const FEED_BADGES = { live: '● LIVE', failed: 'NO CAMERA' };

mountHeader('camera');
const limits = await api('limits');
const { notch: NOTCH, colours: COLOURS } = limits.camera;
const remembered = rememberedView();
const ui = {
  step: STEPS.includes(remembered.step) ? remembered.step : STEP_DEFAULT,
  rotation: remembered.rotation || 0,
  target: { x: limits.home[0], y: limits.home[1], z: SAFE_MM },
};
let state = null;
let inFlight = 0;
let movesInFlight = 0;         // the page's own moves on their way: the target is theirs until they answer
let accepted = JSON.stringify(ui.target);
let isLost = false;
let checkShown = null;

const grid = new CubeGrid($('grid'), limits, jump);
const column = new HeightColumn(NOTCH, {
  get: () => ui.target.z, set: (z) => { ui.target.z = z; }, render: () => render(), send: () => move(),
});
const target = () => [ui.target.x, ui.target.y, ui.target.z];
// What the arm says is out of date: a command is on its way, or the servos are still to be read after one.
const isReading = () => inFlight > 0 || !state || state.is_reading;
const isBusy = () => inFlight > 0 || !state || Boolean(state.doing) || state.queued > 0;

// --- talking to the app ------------------------------------------------------------------------------

async function command(path, body, what, { isRevertible = false } = {}) {
  const asked = JSON.stringify(ui.target);
  inFlight += 1;
  if (isRevertible) movesInFlight += 1;
  render();
  let reply;
  try {
    reply = await api(path, body);
  } catch {
    reply = { status: 'error', detail: 'the calibration app is not answering' };
  }
  inFlight -= 1;
  if (isRevertible) movesInFlight -= 1;
  if (reply.status === 'superseded') return reply;
  if (isRevertible && reply.status === 'done') accepted = asked;
  else if (isRevertible && inFlight === 0) ui.target = JSON.parse(accepted);
  report(reply, what);
  show(reply.state || state);
  return reply;
}

function show(next) {
  if (next && state && next.stamp < state.stamp) return;        // a poll answered before a newer reply
  if (next && isLost) connectFeed();                            // the app is back: its live view is a new one
  isLost = !next;
  state = next;
  showHeader(state);
  // The target follows the arm, as RESET, FIND RED CUBE or CALIBRATE move it too, step by step.
  if (state && !column.isDragging && movesInFlight === 0) {
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

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  column.render(ui.target.z);
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
    : 'Park the arm, take the frame, and fit the camera to the cubes in it';
  $('validate').disabled = isBusy();
  $('calibration-file').textContent = `config/${calibration.summary}`;
  $('check-box').classList.toggle('empty', !check);
  if (check && check.stamp !== checkShown) {
    $('check-image').src = `/api/camera/check.jpg?v=${check.stamp}`;
    checkShown = check.stamp;
  }
  $('check-info').innerHTML = check
    ? `<b>Calibration check</b><br>taken ${check.taken}, ${check.size[0]} × ${check.size[1]}<br>`
      + 'click it to see it full size'
    : '<b>No check image yet</b><br>CALIBRATE or VALIDATE CALIBRATION takes one, with the arm out of the way.';
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
$('validate').addEventListener('click', () => command('camera/validate', {}, 'VALIDATE CALIBRATION'));
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
