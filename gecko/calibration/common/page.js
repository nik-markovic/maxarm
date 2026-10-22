// Shared by every calibration page: the API, the header with RESET, the state poll, the toasts, the dialogs.

const PAGES = [
  { id: 'desk', title: 'Desk', href: '/desk/' },
  { id: 'camera', title: 'Camera', href: '/camera/' },
];
const POLL_MS = 500;
const TOAST_MS = 5000;

const LOGO = `<svg viewBox="0 0 32 32" fill="none" stroke="#41c363" stroke-width="2.4" stroke-linecap="round"
  stroke-linejoin="round"><path d="M4 28h16"/><path d="M8 28v-6h8v6"/><circle cx="12" cy="18" r="2.6"/>
  <path d="M13.6 16l6.4-8"/><circle cx="21.5" cy="6.5" r="2.2"/><path d="M23.4 8l4 6.5"/><path d="M24.5 17.5h5"/></svg>`;

export const $ = (id) => document.getElementById(id);

// GET without a body, POST with one. A refusal or an error is a reply, not an exception.
export async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const response = await fetch(`/api/${path}`, options);
  return response.json();
}

export function fmt(value, digits = 1, sign = false) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const text = Number(value).toFixed(digits);
  return sign && value > 0 ? `+${text}` : text.replace(/^-(0\.0*)$/, '$1');
}

export function toast(text, kind = 'ok') {
  const box = document.createElement('div');
  box.className = `toast ${kind}`;
  box.textContent = text;
  $('toasts').append(box);
  setTimeout(() => box.remove(), TOAST_MS);
}

// A dialog over the page, closed by Esc, its × or a click beside it. Fill `body`; `close()` closes it too.
export function openModal(kind = '') {
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = `<div class="modal ${kind}" role="dialog" aria-modal="true">
    <button class="btn icon close" title="Close (Esc)">×</button><div class="modal-body"></div></div>`;
  const onKey = (event) => { if (event.key === 'Escape') close(); };
  function close() {
    backdrop.remove();
    document.removeEventListener('keydown', onKey);
  }
  backdrop.addEventListener('click', (event) => { if (event.target === backdrop) close(); });
  backdrop.querySelector('.close').addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.append(backdrop);
  return { body: backdrop.querySelector('.modal-body'), close };
}

// Says how a command went, unless it simply did what it was asked and has nothing to add.
export function report(reply, what) {
  const detail = reply.detail ? `: ${reply.detail}` : '';
  if (reply.status === 'done' && reply.detail) toast(`${what}${detail}`);
  else if (reply.status === 'refused') toast(`${what} refused${detail}`, 'warn');
  else if (reply.status === 'stopped') toast(`${what} stopped${detail}`, 'warn');
  else if (reply.status === 'error') toast(`${what} failed${detail}`, 'bad');
}

export function mountHeader(active) {
  const tabs = PAGES.map((page) => `<a href="${page.href}" class="${page.id === active ? 'active' : ''}
    ${page.isSoon ? 'soon' : ''}">${page.title}</a>`).join('');
  document.querySelector('header.top').innerHTML = `
    <div class="brand">${LOGO}<div>MaxArm calibration<small>Avnet demo</small></div></div>
    <nav class="tabs">${tabs}</nav>
    <div class="status">
      <span class="pill idle" id="doing"><span class="dot"></span><span>idle</span></span>
      <span class="pill connecting" id="connection"><span class="dot"></span><span>connecting</span></span>
      <button class="btn small" id="retry" hidden>Retry</button>
      <button class="btn reset" id="reset" title="Stop everything, let go, lift clear and go home">RESET</button>
    </div>`;
  $('reset').onclick = async () => report(await api('reset', {}), 'RESET');
  $('retry').onclick = async () => report(await api('connect', {}), 'Connecting');
}

export function showHeader(state) {
  const connection = state ? state.connection : { status: 'failed', detail: 'the calibration app is not answering' };
  const pill = $('connection');
  pill.className = `pill ${connection.status}`;
  pill.title = connection.detail || '';
  pill.lastElementChild.textContent = {
    ready: 'arm ready', connecting: 'connecting…', failed: connection.detail || 'not connected',
  }[connection.status];
  $('retry').hidden = connection.status !== 'failed' || !state;
  const doing = $('doing');
  const waiting = state && state.queued ? ` · ${state.queued} waiting` : '';
  doing.className = `pill ${state && state.doing ? 'busy' : 'idle'}`;
  doing.lastElementChild.textContent = (state && state.doing ? state.doing : 'idle') + waiting;
}

// Calls onState with each state, or null while the app does not answer. One request at a time.
export function startPolling(onState) {
  const tick = async () => {
    try {
      onState(await api('state'));
    } catch {
      onState(null);
    }
    setTimeout(tick, POLL_MS);
  };
  tick();
}
