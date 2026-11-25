// A page's help: a markdown file beside the page, shown in a dialog. Plain text to edit as wanted, with a little of
// markdown: "## " starts a heading; a line starting "- " or "* " is a bullet and "1. " a numbered step, an indented
// line under either carries it on; **two stars** make bold, *one star* italic; a blank line ends a paragraph or a
// list. Nothing else is special.

import { openModal } from './page.js';

export async function showHelp(file, title) {
  const { body } = openModal('help');
  body.innerHTML = '<p class="hint">Reading the help…</p>';
  try {
    const response = await fetch(file, { cache: 'no-store' });
    if (!response.ok) throw new Error(`the app answered ${response.status}`);
    body.innerHTML = `<h2>${escape(title)}</h2>${renderHelp(await response.text())}`;
  } catch (error) {
    body.innerHTML = `<p>The help, ${escape(file)}, could not be read: ${escape(error.message)}</p>`;
  }
}

export function renderHelp(text) {
  const html = [];
  let paragraph = [];
  let items = [];
  let list = 'ul';
  const flush = () => {
    if (paragraph.length) html.push(`<p>${inline(paragraph.join(' '))}</p>`);
    if (items.length) html.push(`<${list}>${items.map((item) => `<li>${inline(item)}</li>`).join('')}</${list}>`);
    paragraph = [];
    items = [];
  };
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    const bullet = /^([-*]|\d+\.) /.exec(line);
    if (!line) {
      flush();
    } else if (line.startsWith('## ')) {
      flush();
      html.push(`<h3>${inline(line.slice(3))}</h3>`);
    } else if (bullet) {
      const kind = bullet[1].endsWith('.') ? 'ol' : 'ul';
      if (paragraph.length || (items.length && kind !== list)) flush();
      list = kind;
      items.push(line.slice(bullet[0].length));
    } else if (items.length && /^\s/.test(raw)) {
      items[items.length - 1] += ` ${line}`;
    } else {
      if (items.length) flush();
      paragraph.push(line);
    }
  }
  flush();
  return html.join('\n');
}

function inline(text) {
  return escape(text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\*(.+?)\*/g, '<i>$1</i>');
}

function escape(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
