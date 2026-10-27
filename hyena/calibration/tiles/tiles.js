// The Tiles page: where the 25 mm tiles are, and which of them the arm can pick up.
//
// LOCATE TILES parks the arm, takes the snapshot at the camera's full size and finds the tiles in it; each is placed
// in the work areas -- ready to pick up, too close to another, outside the pickup area, or out of reach -- and shown
// in that colour on the grid, over the camera's picture and in the table. Each tile's letter is read too: shown on a
// badge at the letter's bottom right corner, '?' where it could not be read, '–' on a blank tile; the table has how
// sure the reader is, its next best, the dot, and which way the letter's top points. The picture is the live view or
// the snapshot the tiles were found in; FULL SIZE shows it as large as the window allows. A tile clicked, anywhere, brings
// the cup over its centre and down to 4 mm over its top, to see whether it is centred.
// Each LOCATE TILES puts five letters at random from the tiles ready to pick up in the box beside PLACE; PLACE carries
// the tiles with the letters typed, in order, to the dropoff arc, each the right way up, the cup gripping each at the
// desk height in the grip box. Where each goes is drawn, filled once the tile is there. The tiles carried are no
// longer where the snapshot has them: LOCATE TILES again.

import { $, api, fmt, mountHeader, report, showHeader, startPolling } from '/common/page.js';
import { openFullView } from '/common/fullview.js';
import { FACINGS, STEPS, STEP_DEFAULT, rememberView, rememberedView } from '/common/grid.js';
import { toDesk } from '/common/shares.js';
import { TilesGrid } from './grid.js';
import { STATUS_INK, TilesMarks, tileAt, tileName } from './view.js';

const FEED = '/api/camera/feed';
const SNAPSHOT = '/api/tiles/snapshot.jpg';
const FEED_RETRY_MS = 3000;
const FEED_NOTES = {
  idle: 'starting the camera…', starting: 'starting the camera…', snapshot: 'the camera is taking the snapshot…',
};
const FEED_BADGES = { live: '● LIVE', failed: 'NO CAMERA' };
const AREAS_KEY = 'tiles-areas';
const STATUS_WORDS = {
  ready: 'ready', 'too close': 'too close', elsewhere: 'outside the pickup area', 'out of reach': 'out of reach',
};

mountHeader('tiles');
const limits = await api('limits');
const { tiles: TILES } = limits;
const remembered = rememberedView();
const ui = {
  step: STEPS.includes(remembered.step) ? remembered.step : STEP_DEFAULT,
  rotation: remembered.rotation || 0,
  source: 'live',               // the picture shown: 'live', or 'snapshot', the one the tiles were found in
  showAreas: localStorage.getItem(AREAS_KEY) !== 'off',
  picked: null,                 // the tile last sent to, its number
  offered: null,                // the stamp of the snapshot whose letters were put in the box
};
let state = null;
let inFlight = 0;
let isLost = false;
let full = null;                // the picture full size, while it is open
let rowsShown = '';
let snapshotShown = null;       // the stamp of the snapshot the page has loaded

const grid = new TilesGrid($('grid'), limits, visit);
const marks = new TilesMarks($('marks'));
const tiles = () => (state ? state.tiles : null);
const found = () => (tiles() ? tiles().found : null);
const placing = () => (tiles() ? tiles().word : null);
const isBusy = () => inFlight > 0 || !state || Boolean(state.doing) || state.queued > 0;
// The tiles PLACE has started on, by number: no longer, or perhaps not, where the snapshot has them.
const moved = () => new Set(placing() ? placing().steps.filter((step) => step.status !== 'waiting')
  .map((step) => step.n) : []);

// --- talking to the app ------------------------------------------------------------------------------

async function command(path, body, what) {
  inFlight += 1;
  render();
  let reply;
  try {
    reply = await api(path, body);
  } catch {
    reply = { status: 'error', detail: 'the calibration app is not answering' };
  }
  inFlight -= 1;
  if (reply.status !== 'superseded') report(reply, what);
  show(reply.state || state);
  return reply;
}

function show(next) {
  if (next && state && next.stamp < state.stamp) return;        // a poll answered before a newer reply
  if (next && isLost) connectFeed();                            // the app is back: its live view is a new one
  isLost = !next;
  state = next;
  showHeader(state);
  render();
}

async function locate() {
  const reply = await command('tiles/locate', {}, 'LOCATE TILES');
  if (reply.status === 'done') {
    ui.source = 'snapshot';
    connectFeed();
    render();
  }
}

function visit(n) {
  const tile = found() ? found().tiles.find((each) => each.n === n) : null;
  if (!tile || isBusy()) return;
  ui.picked = n;
  command('tiles/visit', { n }, `Tile ${n}`);
}

function place() {
  const letters = $('word').value.trim().toUpperCase();
  if (!letters || $('place').disabled) return;
  command('tiles/place', { letters, grip: Number($('grip').value) }, `PLACE ${letters}`);
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  const view = {
    matrix: tiles() ? tiles().desk_to_view : null, deskToTrue: tiles() ? tiles().desk_to_true : null, limits,
    found: found(), word: placing(), moved: moved(), showAreas: ui.showAreas, picked: ui.picked,
    target: state ? state.camera.position : null, says: state ? state.camera.measured : null,
    isReading: !state || state.is_reading,
  };
  marks.update(view);
  if (full && full.isOpen()) full.update(view, state ? state.doing : '', pointedTile());
  grid.update({ step: ui.step, rotation: ui.rotation, found: found(), word: placing(), moved: moved(),
    picked: ui.picked, sent: state ? state.sent : null, says: state ? state.camera.measured : null,
    isReading: !state || state.is_reading });
  renderView();
  if (!state) return;
  renderPage();
  renderPicture();
  renderTable();
}

function renderView() {
  $('facing').textContent = FACINGS[ui.rotation / 90];
  for (const button of $('steps').children) button.classList.toggle('on', Number(button.textContent) === ui.step);
  for (const button of $('source').children) button.classList.toggle('on', button.dataset.source === ui.source);
  const [pickup, dropoff] = ['pickup', 'dropoff'].map((name) => TILES.areas.find((area) => area.name === name));
  $('areas-line').textContent = `Tile centres ${fmt(dropoff.inner, 0)} to ${fmt(pickup.outer, 0)} mm from the base `
    + `axis, ${fmt(TILES.margin, 0)} mm inside the reach. Pickup: ${fmt(pickup.inner, 0)} to ${fmt(pickup.outer, 0)} `
    + `mm${pickup.side < 0 ? ', left of the arm\'s centre line (x < 0)' : ''}`
    + `${pickup.side > 0 ? ', right of the arm\'s centre line (x > 0)' : ''}`
    + `; dropoff: ${fmt(dropoff.inner, 0)} to ${fmt(dropoff.outer, 0)} mm. Tiles in the pickup area at least `
    + `${fmt(TILES.gap, 0)} mm apart.`;
}

function renderPage() {
  const { problem } = tiles();
  $('blocked').hidden = !problem;
  document.querySelector('main.tiles').classList.toggle('is-blocked', Boolean(problem));
  $('blocked-why').textContent = problem ? `${problem[0].toUpperCase()}${problem.slice(1)}.` : '';
  $('locate').disabled = Boolean(problem) || isBusy();
  const result = found();
  $('locate-line').innerHTML = result
    ? `${countsLine(result)}${lettersLine(result)} <span class="note">· found in ${fmt(result.seconds, 1)} s, read in
      ${fmt(result.reading, 1)} s, in the snapshot of ${result.taken}, ${result.frame[0]} x ${result.frame[1]}</span>`
    : '<span class="note">Set the tiles down, then LOCATE TILES: the arm moves out of the picture, the camera takes '
      + 'its full-size snapshot -- about 11 s -- and the tiles are found in it.</span>';
  $('counts').innerHTML = result ? Object.entries(STATUS_WORDS).filter(([status]) => result.counts[status])
    .map(([status, words]) => `<span class="chip tile-chip" style="--ink: ${STATUS_INK[status]}">
      ${result.counts[status]} ${words}</span>`).join('') : '';
  $('visit-hint').textContent = result && result.tiles.length
    ? `go, or a click on a tile in the grid or the picture: the cup over its centre at ${fmt(TILES.visit.over, 0)} mm, `
      + `down to ${fmt(TILES.visit.down, 0)}` : '';
  if (result && ui.offered !== result.stamp) {
    $('word').value = result.offer;
    ui.offered = result.stamp;
  }
  $('word').disabled = Boolean(problem) || !result;
  $('grip').disabled = $('word').disabled;
  $('place').disabled = Boolean(problem) || isBusy() || !result || !$('word').value.trim() || moved().size > 0
    || $('grip').value === '' || Number.isNaN(Number($('grip').value));
  $('word-line').innerHTML = wordLine(result, placing());
}

// What PLACE can take, or how the word it is setting down is going.
function wordLine(result, word) {
  if (!result) return '';
  const head = word ? `PLACE <b class="letters">${word.letters}</b>: ` : '';
  const placed = word ? word.steps.filter((step) => step.status === 'placed').length : 0;
  const again = ' <span class="note">· LOCATE TILES again before the next PLACE</span>';
  if (!word) {
    return result.on_offer ? `On the tiles ready to pick up: <b class="letters">${result.on_offer}</b> <span `
      + `class="note">· PLACE carries the letters typed, in order, to the arc ${fmt(TILES.word.arc, 0)} mm out, `
      + `${fmt(TILES.word.gap, 0)} mm apart, each facing ${TILES.word.is_away ? 'away from' : 'towards'} the arm; `
      + `the cup grips at the height in the box and lets go at ${fmt(TILES.word.drop, 0)} mm</span>`
      : '<span class="note">No letters on tiles ready to pick up: nothing for PLACE to carry.</span>';
  }
  if (word.status === 'placing') {
    const carrying = word.steps.find((step) => step.status === 'carrying');
    return `${head}${placed} of ${word.steps.length} set down`
      + `${carrying ? `, carrying ${carrying.letter} from tile ${carrying.n}` : ''}`;
  }
  if (word.status === 'done') return `${head}all ${word.steps.length} set down${again}`;
  return `${head}${word.status} with ${placed} of ${word.steps.length} set down: ${escaped(word.detail)}${again}`;
}

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
function escaped(text) {
  return String(text).replace(/[&<>"]/g, (character) => ESCAPES[character]);
}

function countsLine(result) {
  const parts = Object.entries(STATUS_WORDS).filter(([status]) => result.counts[status])
    .map(([status, words]) => `<b style="color: ${STATUS_INK[status]}">${result.counts[status]}</b> ${words}`);
  return result.tiles.length ? `<b>${result.tiles.length}</b> tiles: ${parts.join(', ')}`
    : 'No tiles found: are they in view, the arm out of the picture?';
}

function lettersLine(result) {
  if (!result.tiles.length) return '';
  const shown = result.tiles.map((tile) => (!tile.letter ? '–' : tile.is_read ? tile.letter : '?')).join('');
  const unread = result.tiles.filter((tile) => tile.letter && !tile.is_read).length;
  const blank = result.tiles.filter((tile) => !tile.letter).length;
  const notes = [unread ? `${unread} not read` : '', blank ? `${blank} blank` : ''].filter(Boolean);
  return `; letters <b class="letters">${shown}</b>${notes.length ? `, ${notes.join(', ')}` : ''}`;
}

// The reading, for the table: the letter or '?', how sure, the next best, the dot.
function letterCells(tile) {
  if (!tile.letter) {
    return '<td class="letter"><span class="letter-key marked">–</span></td>'
      + '<td class="read"><small>blank: no ink</small></td><td>—</td>';
  }
  const guess = tile.is_read ? '' : `<small>best guess ${tile.letter}</small>`;
  const also = tile.uprights.filter((angle) => angle !== tile.upright).map((angle) => `${fmt(angle, 0)}°`);
  const alsoNote = also.length ? `reads the same at ${also.join(', ')}` : '';
  return `<td class="letter">${tile.is_read ? tile.letter : '<span class="letter-key marked">?</span>'}</td>
    <td class="read num">${fmt(100 * tile.confidence, 0)}% ${guess}
      <small>next ${tile.runner_up} ${fmt(100 * tile.runner_up_confidence, 0)}%${tile.has_dot ? ' · dot' : ''}
      </small></td>
    <td class="num" title="${alsoNote}">${fmt(tile.upright, 0)}°${also.length ? '*' : ''}</td>`;
}

function renderPicture() {
  const { frame_size: size, view_problem: viewProblem } = tiles();
  if (size) $('feed-box').style.aspectRatio = `${size[0]} / ${size[1]}`;
  const result = found();
  const isSnapshot = ui.source === 'snapshot' && result;
  if (isSnapshot && snapshotShown !== result.stamp) {
    $('snapshot').src = `${SNAPSHOT}?stamp=${result.stamp}`;
    snapshotShown = result.stamp;
  }
  $('snapshot').hidden = !isSnapshot;
  $('feed').hidden = Boolean(isSnapshot);
  const { feed } = state.camera;
  $('feed-badge').textContent = isSnapshot ? `◉ SNAPSHOT ${result.taken}` : FEED_BADGES[feed.status] || '';
  $('feed-badge').className = `feed-badge ${isSnapshot ? 'snapshot' : feed.status}`;
  $('feed-note').textContent = full && full.isOpen() ? 'shown full size'
    : isSnapshot ? '' : viewProblem || (feed.status === 'failed' ? feed.detail : FEED_NOTES[feed.status] || '');
  $('feed-device').textContent = !isSnapshot && feed.status === 'live' ? feed.detail : '';
  for (const button of $('source').children) {
    button.disabled = button.dataset.source === 'snapshot' && !result;
  }
  $('expand').disabled = Boolean(tiles().problem);
}

function renderTable() {
  const result = found();
  const word = placing();
  const shown = JSON.stringify([result ? result.stamp : null, isBusy(), word ? word.steps : null]);
  if (shown === rowsShown) return;
  rowsShown = shown;
  const carried = moved();
  const rows = result ? result.tiles.map((tile) => `
    <tr class="${tile.status.replace(/ /g, '-')}" data-n="${tile.n}">
      <td><span class="badge" style="background: ${STATUS_INK[tile.status]}">${tile.n}</span></td>
      ${letterCells(tile)}
      <td class="num">${fmt(tile.x)}</td><td class="num">${fmt(tile.y)}</td>
      <td class="num">${fmt(tile.turn, 1)}°</td><td class="num">${fmt(tile.reach, 0)}</td>
      <td>${tile.area || '<span class="faint">—</span>'}</td>
      <td class="num">${tile.nearest ? `${fmt(tile.gap, 1)} mm to ${tile.nearest}` : '—'}</td>
      <td class="status"><b style="color: ${STATUS_INK[tile.status]}">${STATUS_WORDS[tile.status]}</b>
        ${tile.detail && tile.status !== 'elsewhere' ? `<small>${tile.detail}</small>` : ''}${wordNote(word, tile)}</td>
      <td class="actions"><button class="btn small" data-n="${tile.n}"
        ${tile.status === 'out of reach' || isBusy() || carried.has(tile.n) ? 'disabled' : ''}
        title="The cup over this tile's centre, down to ${fmt(TILES.visit.down, 0)} mm">go</button></td>
    </tr>`) : [];
  const empty = result && !result.tiles.length ? '<tr class="empty"><td colspan="12">No tiles found.</td></tr>'
    : !result ? '<tr class="empty"><td colspan="12">LOCATE TILES to find them.</td></tr>' : '';
  $('tiles').tBodies[0].innerHTML = rows.join('') + empty;
}

// A tile PLACE takes: which letter of the word it is, and how that went.
function wordNote(word, tile) {
  const index = word ? word.steps.findIndex((step) => step.n === tile.n) : -1;
  if (index < 0) return '';
  const { status } = word.steps[index];
  const how = { waiting: 'next', carrying: 'being carried', placed: 'set down' }[status] || status;
  return `<small class="word-note">PLACE: letter ${index + 1} of ${word.letters}, ${how}</small>`;
}

// The tile under the pointer in the full-size view, as its bar shows it.
function pointedTile() {
  const result = found();
  const hover = full && full.marks.marks.hover;
  const desk = hover && tiles() && toDesk(tiles().desk_to_view, hover, TILES.height);
  const tile = desk && result ? tileAt(result.tiles, desk) : null;
  if (!tile) return { label: 'tile', value: '', note: desk ? `desk ${desk[0].toFixed(1)}, ${desk[1].toFixed(1)}` : '' };
  return { label: `tile ${tileName(tile)}`, value: `${tile.x.toFixed(1)}, ${tile.y.toFixed(1)}`,
    note: STATUS_WORDS[tile.status], isOff: tile.status !== 'ready' };
}

// The live view only while it is shown and the page is in sight; the full-size view takes it over.
function connectFeed() {
  const isWanted = !document.hidden && ui.source === 'live' && !(full && full.isOpen());
  if (isWanted) $('feed').src = `${FEED}?t=${Date.now()}`;
  else $('feed').removeAttribute('src');
}

function openFull() {
  const { frame_size: size, problem } = tiles() || {};
  if (!size || problem || (full && full.isOpen())) return;
  const result = found();
  const isSnapshot = ui.source === 'snapshot' && result;
  full = openFullView({
    image: isSnapshot ? `${SNAPSHOT}?stamp=${result.stamp}` : FEED,
    Marks: TilesMarks,
    aspect: size[0] / size[1],
    title: isSnapshot ? `THE SNAPSHOT OF ${result.taken}` : 'THE LIVE VIEW',
    hint: 'Click a tile: the cup comes over its centre and down to '
      + `${fmt(TILES.visit.down, 0)} mm, ${fmt(TILES.visit.down - TILES.height, 0)} over its top.`,
    onPick: (u, v) => {
      const desk = toDesk(tiles().desk_to_view, [u, v], TILES.height);
      const tile = desk && found() ? tileAt(found().tiles, desk) : null;
      if (tile) visit(tile.n);
    },
    onHover: () => { if (full) full.update({}, state ? state.doing : '', pointedTile()); },
    onReset: async () => report(await api('reset', {}), 'RESET'),
    onClose: () => { full = null; connectFeed(); render(); },
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
for (const button of $('source').children) {
  button.addEventListener('click', () => {
    ui.source = button.dataset.source;
    connectFeed();
    render();
  });
}
$('guides').checked = ui.showAreas;
$('guides').addEventListener('change', () => {
  ui.showAreas = $('guides').checked;
  localStorage.setItem(AREAS_KEY, ui.showAreas ? 'on' : 'off');
  render();
});
$('locate').addEventListener('click', locate);
$('word').maxLength = TILES.word.most;
[$('grip').min, $('grip').max] = TILES.word.grip_range;
$('grip').value = TILES.word.grip;
$('grip').addEventListener('input', render);
$('word').addEventListener('input', render);
$('word').addEventListener('keydown', (event) => { if (event.key === 'Enter') place(); });
$('place').addEventListener('click', place);
$('expand').addEventListener('click', openFull);
$('tiles').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-n]');
  if (button) visit(Number(button.dataset.n));
});
$('feed').addEventListener('error', () => { if (!document.hidden) setTimeout(connectFeed, FEED_RETRY_MS); });
document.addEventListener('visibilitychange', connectFeed);
connectFeed();
render();
startPolling((next) => show(next));
