// The GUIDE: guide.md beside this file, shown in a dialog. It is plain text to edit as wanted, with a little of
// markdown: a line starting "- " or "* " is a bullet, and an indented line under it carries the bullet on;
// **two stars** make bold, *one star* italic; a blank line ends a paragraph or a list. Nothing else is special.

import { openModal } from '/common/page.js';

export async function showGuide() {
  const { body } = openModal('guide');
  body.innerHTML = '<p class="hint">Reading the guide…</p>';
  try {
    const response = await fetch('guide.md', { cache: 'no-store' });
    if (!response.ok) throw new Error(`the app answered ${response.status}`);
    body.innerHTML = `<h2>Camera calibration</h2>${renderGuide(await response.text())}`;
  } catch (error) {
    body.innerHTML = `<p>The guide, calibration/camera/guide.md, could not be read: ${escape(error.message)}</p>`;
  }
}

export function renderGuide(text) {
  const html = [];
  let paragraph = [];
  let items = [];
  const flush = () => {
    if (paragraph.length) html.push(`<p>${inline(paragraph.join(' '))}</p>`);
    if (items.length) html.push(`<ul>${items.map((item) => `<li>${inline(item)}</li>`).join('')}</ul>`);
    paragraph = [];
    items = [];
  };
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      flush();
    } else if (/^[-*] /.test(line)) {
      if (paragraph.length) flush();
      items.push(line.slice(2));
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
