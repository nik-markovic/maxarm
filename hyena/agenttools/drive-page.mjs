// A calibration page's script run in node, against the app's own replies: does it render them without throwing?
//
//   node --experimental-default-type=module agenttools/drive-page.mjs <page.js> <replies.json>
//
// replies.json: { "limits": {...}, "states": [...], "clicks": ["record", ...], "watch": ["save", ...] }. The page
// loads with the first state; then each click is that button's click handler, its request answered with the next
// state. The DOM is a stub: elements keep what is set on them and remember their listeners; a canvas's context
// draws nothing. The page's own polling takes any failure for the app not answering, so here it is one that lets
// the page's errors through. Prints, as JSON, the errors thrown, and for each state what the page wrote into the
// elements watched: their text, markup, rows (a table's body), whether disabled, and title.

import { readFileSync } from 'node:fs';
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [pagePath, repliesPath] = process.argv.slice(2);
const root = pathToFileURL(path.resolve(path.dirname(pagePath), '..')).href;
const polling = `data:text/javascript,${encodeURIComponent(`
export * from '${root}/common/page.js';
import { api } from '${root}/common/page.js';
export function startPolling(onState) {
  api('state').then((state) => onState(state)).catch((error) => globalThis.pageErrors.push(String(error.stack)));
}`)}`;
register(`data:text/javascript,${encodeURIComponent(`
export async function resolve(specifier, context, next) {
  if (specifier === '/common/page.js') return next(${JSON.stringify(polling)}, context);
  return next(specifier.startsWith('/') ? ${JSON.stringify(root)} + specifier : specifier, context);
}`)}`);

const replies = JSON.parse(readFileSync(repliesPath, 'utf8'));
const errors = [];
globalThis.pageErrors = errors;
const elements = new Map();
let stateIndex = 0;

function stub(name, extra = {}) {
  const own = { textContent: '', innerHTML: '', value: '', src: '', title: '', disabled: false, hidden: false,
    checked: false, dataset: {}, style: {}, children: [], clientWidth: 640, clientHeight: 480, listeners: {},
    ...extra };
  return new Proxy(own, {
    get(target, key) {
      if (key in target) return target[key];
      if (key === Symbol.iterator) return function* nothing() {};
      if (typeof key === 'symbol') return undefined;
      if (key === 'addEventListener') return (type, listener) => { target.listeners[type] = listener; };
      if (key === 'getContext') return () => context();
      if (key === 'getBoundingClientRect') return () => ({ left: 0, top: 0, width: 640, height: 480 });
      if (key === 'classList') return { toggle() {}, add() {}, remove() {}, contains: () => false };
      if (key === 'tBodies') {
        target.tBodies = [stub(`${name}.tbody`)];
        return target.tBodies;
      }
      return () => stub(`${name}.${key}()`);
    },
    set(target, key, value) {
      target[key] = value;
      return true;
    },
  });
}

function context() {
  return new Proxy({}, {
    get(target, key) {
      if (key in target) return target[key];
      if (key === 'measureText') return (text) => ({ width: 7 * String(text).length });
      return () => {};
    },
    set(target, key, value) {
      target[key] = value;
      return true;
    },
  });
}

globalThis.document = stub('document', {
  hidden: false,
  getElementById: (id) => {
    if (!elements.has(id)) elements.set(id, stub(`#${id}`));
    return elements.get(id);
  },
});
globalThis.window = { devicePixelRatio: 1, confirm: () => true };
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.ResizeObserver = class {
  constructor(callback) { this.callback = callback; }
  observe() { queueMicrotask(() => this.callback()); }          // after the constructor, as a browser does
  disconnect() {}
};
globalThis.setTimeout = () => 0;
globalThis.fetch = async (url) => {
  const state = replies.states[Math.min(stateIndex, replies.states.length - 1)];
  const body = url.endsWith('/api/limits') ? replies.limits : url.endsWith('/api/state') ? state
    : { status: 'done', detail: '', state };
  return { json: async () => body };
};
process.on('unhandledRejection', (error) => errors.push(String(error && error.stack ? error.stack : error)));
process.on('uncaughtException', (error) => errors.push(String(error.stack || error)));

const settle = () => new Promise((resolve) => setImmediate(resolve));
const steps = [];
const look = () => steps.push(Object.fromEntries((replies.watch || []).map((id) => {
  const element = document.getElementById(id);
  return [id, { text: element.textContent, html: element.innerHTML, rows: element.tBodies[0].innerHTML,
    disabled: element.disabled, title: element.title }];
})));
try {
  await import(pathToFileURL(path.resolve(pagePath)).href);
} catch (error) {
  errors.push(String(error.stack || error));
}
await settle();
look();
for (const id of replies.clicks || []) {
  stateIndex += 1;
  const listener = elements.has(id) ? elements.get(id).listeners.click : null;
  if (!listener) {
    errors.push(`no click handler on #${id}`);
    continue;
  }
  try {
    await listener({ target: stub('event.target') });
  } catch (error) {
    errors.push(String(error.stack || error));
  }
  await settle();
  look();
}
console.log(JSON.stringify({ errors, steps }));
