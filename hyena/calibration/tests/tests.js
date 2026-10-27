// The tests page: does the camera calibration bring the arm to what the camera sees?
//
// FIND RED CUBE parks the arm, finds the red cube in the live view, and brings the cup down over it. ARM TO DESK
// sends the arm to the point of the desk clicked in the live view, shown full size. The grid, the height column and
// the arrows move the arm by hand, in desk space, with the camera page's moves. Over the live view: the reach, the
// arm, the cube found and the point clicked, drawn through the calibration.
//
// And the red cube's test in reverse: with the cube centred under the cup, RECORD POINT keeps a point that refines
// the calibration; SAVE CALIBRATION writes it (refine.js).
//
// The arm's target is the grid point last clicked, the arrows' steps and the height column. Whatever moves the arm
// from elsewhere (a test, RESET, another page) moves the target with it, and a move the arm refuses puts it back.

import { $, api, fmt, mountHeader, report, showHeader, startPolling } from '/common/page.js';
import { FACINGS, STEPS, STEP_DEFAULT, rememberView, rememberedView } from '/common/grid.js';
import { HeightColumn, SAFE_MM, cubeIcon } from '/common/height.js';
import { TestsGrid } from './grid.js';
import { fitLine, nextSpot, savedLine, tableRows } from './refine.js';
import { openFullView } from '/common/fullview.js';
import { toDesk } from '/common/shares.js';
import { LANDED_OFF_MM, LiveMarks, landedOff } from './view.js';

const SAME_MM = 0.005;
const SCREEN = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const ARROWS = { up: '▲', down: '▼', left: '◀', right: '▶' };
const FEED = '/api/camera/feed';
const FEED_RETRY_MS = 3000;
const FEED_NOTES = {
  idle: 'starting the camera…', starting: 'starting the camera…', snapshot: 'the camera is taking a snapshot…',
};
const FEED_BADGES = { live: '● LIVE', snapshot: '◉ SNAPSHOT', failed: 'NO CAMERA' };
const GUIDES_KEY = 'tests-guides';

mountHeader('tests');
const limits = await api('limits');
const { notch: NOTCH } = limits.camera;
const { touch: TOUCH, find: FIND, refine: REFINE } = limits.tests;
const remembered = rememberedView();
const ui = {
  step: STEPS.includes(remembered.step) ? remembered.step : STEP_DEFAULT,
  rotation: remembered.rotation || 0,
  target: { x: limits.home[0], y: limits.home[1], z: SAFE_MM },
  showGuides: localStorage.getItem(GUIDES_KEY) !== 'off',
  touch: null,                  // the point of desk last clicked: { u, v } in the view, desk [x, y]
  findNote: '',                 // why the last FIND RED CUBE found nothing
};
let state = null;
let inFlight = 0;
let accepted = JSON.stringify(ui.target);
let isLost = false;
let full = null;                // the live view full size, while it is open
let pointsShown = '';

const grid = new TestsGrid($('grid'), limits, jump);
const column = new HeightColumn(NOTCH, {
  get: () => ui.target.z, set: (z) => { ui.target.z = z; }, render: () => render(), send: () => move(),
});
const marks = new LiveMarks($('marks'));
const target = () => [ui.target.x, ui.target.y, ui.target.z];
const tests = () => (state ? state.tests : null);
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
  if (state && !column.isDragging && inFlight === 0 && (isReply || (!state.doing && !state.queued))) {
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

function nudge([dx, dy]) {
  ui.target = { ...ui.target, x: ui.target.x + dx, y: ui.target.y + dy };
  move();
}

async function findRedCube() {
  ui.findNote = '';
  const reply = await command('tests/find-red', {}, 'FIND RED CUBE');
  if (reply.status !== 'done' && reply.status !== 'superseded') ui.findNote = reply.detail || reply.status;
  render();
}

function clearPoints() {
  const count = tests() && tests().refine ? tests().refine.points.length : 0;
  if (count && window.confirm(`Forget all ${count} red cube points?`)) command('tests/clear', {}, 'CLEAR');
}

// A click in the live view, full size: marked at once, then the arm is sent. A newer click replaces one still
// waiting for the arm.
async function touch(u, v) {
  const matrix = tests() ? tests().desk_to_view : null;
  if (!matrix) return;
  ui.touch = { u, v, desk: toDesk(matrix, [u, v]) };
  render();
  const reply = await command('tests/touch', { u, v }, 'ARM TO DESK');
  if (reply.point && ui.touch && ui.touch.u === u && ui.touch.v === v) ui.touch.desk = reply.point;
  render();
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  const found = tests() ? tests().found : null;
  const matrix = tests() ? tests().desk_to_view : null;
  const refine = tests() ? tests().refine : null;
  const next = refine ? nextSpot([...refine.cubes, ...refine.points], limits, matrix) : null;
  column.render(ui.target.z);
  renderReadout();
  renderView();
  if (state) {
    renderTests();
    renderFeed();
  }
  if (refine) renderRefine(refine, next);
  const view = {
    matrix, deskToTrue: tests() ? tests().desk_to_true : null, limits, found, touch: ui.touch, target: target(),
    sent: state ? state.camera.position : null, says: state ? state.camera.measured : null, isReading: isReading(),
    showGuides: ui.showGuides, refine, next,
  };
  marks.update(view);
  if (full && full.isOpen()) full.update(view, state ? state.doing : '', clickedPoint());
  grid.update({
    step: ui.step, rotation: ui.rotation, target: target(), isReading: isReading(), found,
    touch: ui.touch ? ui.touch.desk : null, sent: state ? state.sent : null,
    says: state ? state.camera.measured : null, refine, next,
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
  for (const button of document.querySelectorAll('.pad [data-dir]')) {
    const [dx, dy] = armStep(SCREEN[button.dataset.dir]);
    button.textContent = `${ARROWS[button.dataset.dir]} ${dx > 0 ? '+X' : dx < 0 ? '−X' : dy > 0 ? '+Y' : '−Y'}`;
  }
}

function renderTests() {
  const { problem, view_problem: viewProblem, frame_size: size, found } = tests();
  $('blocked').hidden = !problem;
  document.querySelector('main.tests').classList.toggle('is-blocked', Boolean(problem));
  $('blocked-why').textContent = problem ? `${problem[0].toUpperCase()}${problem.slice(1)}.` : '';
  if (size) $('feed-box').style.aspectRatio = `${size[0]} / ${size[1]}`;
  const why = problem || viewProblem;
  $('find').disabled = Boolean(why) || isBusy();
  const steps = `come over it at ${fmt(FIND.over, 0)} mm and down to ${fmt(FIND.down, 0)}`;
  $('find').title = why || `Park the arm, find the red cube in the live view, ${steps}`;
  $('record').disabled = Boolean(why) || isBusy() || isReading();
  $('record').title = why || 'Keep where the arm says the cup is, over the red cube centred under it; then park '
    + 'the arm and see where the camera has the cube: a point to refine the calibration with';
  for (const id of ['touch', 'expand']) {
    $(id).disabled = Boolean(why);
    $(id).title = why || 'The live view full size: click a point of the desk to send the arm there';
  }
  $('find-result').innerHTML = found
    ? `desk <b>${fmt(found.x)}, ${fmt(found.y)}</b> · turned ${fmt(found.turn_deg, 0)}° · edges within
      ${fmt(found.edge_px, 1)} px · ${found.taken}`
    : `<span class="note">${ui.findNote || viewProblem || 'not looked for yet'}</span>`;
  const landed = landing();
  $('touch-result').innerHTML = ui.touch && ui.touch.desk
    ? `desk <b>${fmt(ui.touch.desk[0])}, ${fmt(ui.touch.desk[1])}</b>, down to ${fmt(TOUCH.down, 0)} mm`
      + (landed ? ` · <span class="${landed.isOff ? 'off' : 'on'}">${landed.text}</span>` : '')
    : `<span class="note">${ui.touch ? 'that point is not on the desk' : 'no point clicked yet'}</span>`;
}

function renderRefine(refine, next) {
  const count = refine.points.length;
  $('refine-count').innerHTML = `<span class="chip ${count >= REFINE.suggested ? 'ok' : ''}">${count} of
    ${REFINE.suggested} points</span>`;
  $('refine-ask').innerHTML = `At least <b>${REFINE.suggested}</b>, spread over where pieces will go: near the arm and
    far out, left and right, as far as the desk allows. `
    + (next ? `The dashed circle on the grid and in the live view is a place far from every point so far, around
      (${fmt(next.x, 0)}, ${fmt(next.y, 0)}).` : 'Every place in reach and in view has a point near it.');
  const offLimits = REFINE.off;
  const shown = JSON.stringify([refine.points, refine.cubes]);
  if (shown !== pointsShown) {
    pointsShown = shown;
    $('points').tBodies[0].innerHTML = tableRows(refine, offLimits);
  }
  $('refine-fit').innerHTML = fitLine(refine);
  $('refine-saved').textContent = savedLine(refine);
  $('save').disabled = !refine.fit || isBusy();
  $('save').title = refine.fit ? 'Write camera.json with the calibration these points make' : refine.problem;
  $('clear').disabled = count === 0 || isBusy();
}

// The point clicked, as the full view's bar shows it: its desk x, y, and how the arm landed there.
function clickedPoint() {
  const desk = ui.touch ? ui.touch.desk : null;
  const landed = landing();
  return { label: 'desk', value: desk ? `${desk[0].toFixed(1)}, ${desk[1].toFixed(1)}` : '',
    note: landed ? landed.text : '', isOff: Boolean(landed && landed.isOff) };
}

// How the arm landed at the point clicked, by its servos, once it is there and read; null otherwise.
function landing() {
  const sent = state ? state.camera.position : null;
  const desk = ui.touch ? ui.touch.desk : null;
  if (!desk || !sent || Math.hypot(sent[0] - desk[0], sent[1] - desk[1]) > 0.05) return null;
  const off = landedOff(sent, state.camera.measured, isReading());
  if (off === null) return null;
  return off > LANDED_OFF_MM ? { text: `landed ${fmt(off)} mm off, by the arm's servos`, isOff: true }
    : { text: `landed within ${fmt(LANDED_OFF_MM)} mm`, isOff: false };
}

function renderFeed() {
  const { feed } = state.camera;
  $('feed-badge').textContent = FEED_BADGES[feed.status] || '';
  $('feed-badge').className = `feed-badge ${feed.status}`;
  $('feed-note').textContent = full && full.isOpen() ? 'shown full size'
    : feed.status === 'failed' ? feed.detail : FEED_NOTES[feed.status] || '';
  $('feed-device').textContent = feed.status === 'live' ? feed.detail : '';
}

// Screen direction to desk direction, undoing the view's quarter turn.
function armStep([sx, sy]) {
  const angle = (ui.rotation * Math.PI) / 180;
  const c = Math.round(Math.cos(angle));
  const s = Math.round(Math.sin(angle));
  return [sx * c + sy * s, -sx * s + sy * c];
}

// The live view only while the page is in sight, and not while it is shown full size: either lets the camera go.
function connectFeed() {
  if (document.hidden || (full && full.isOpen())) $('feed').removeAttribute('src');
  else $('feed').src = `${FEED}?t=${Date.now()}`;
}

function openFull() {
  const { frame_size: size, problem, view_problem: viewProblem } = tests() || {};
  if (!size || problem || viewProblem || (full && full.isOpen())) return;
  full = openFullView({
    image: FEED,
    Marks: LiveMarks,
    aspect: size[0] / size[1],
    title: 'ARM TO DESK',
    hint: `Click a point of the desk: the arm comes over it at ${fmt(TOUCH.over, 0)} mm, then down to
      ${fmt(TOUCH.down, 0)} mm over the desk -- the desk, not the top of whatever stands there.`,
    onPick: touch,
    onReset: async () => report(await api('reset', {}), 'RESET'),
    onClose: () => { connectFeed(); render(); },
  });
  connectFeed();
  render();
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
for (const button of document.querySelectorAll('.pad [data-dir]')) {
  button.addEventListener('click', () => nudge(armStep(SCREEN[button.dataset.dir])));
}

$('guides').checked = ui.showGuides;
$('guides').addEventListener('change', () => {
  ui.showGuides = $('guides').checked;
  localStorage.setItem(GUIDES_KEY, ui.showGuides ? 'on' : 'off');
  render();
});
$('find-icon').innerHTML = cubeIcon('red');
$('find').addEventListener('click', findRedCube);
$('record').addEventListener('click', () => command('tests/record', {}, 'RECORD POINT'));
$('save').addEventListener('click', () => command('tests/save', {}, 'SAVE CALIBRATION'));
$('clear').addEventListener('click', clearPoints);
$('points').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-id]');
  if (button) command('tests/forget', { id: Number(button.dataset.id) }, `Forget point ${button.dataset.id}`);
});
$('touch').addEventListener('click', openFull);
$('feed-box').addEventListener('click', openFull);
$('feed').addEventListener('error', () => { if (!document.hidden) setTimeout(connectFeed, FEED_RETRY_MS); });
document.addEventListener('visibilitychange', connectFeed);
connectFeed();
render();
startPolling((next) => show(next));
