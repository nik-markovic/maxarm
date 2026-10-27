// The height column beside the view, for the pages that work in desk space with cubes on the desk (camera, tests).
// The owner's (PILOT-gecko): the slider runs from 70 down to -2 mm over the desk, yellow from the desk down a
// millimetre and red below, where the cup presses into it. The notch, 4 mm over a cube's top, catches the slider
// from a millimetre either side. SAFE is well clear of the cubes anywhere in reach. ▲ and ▼ take the slider's ends
// further.
//
// The page owns the height: the column shows it, and says when it is changed and when to send it.

import { $, fmt } from './page.js';

const HEIGHT = { top: 70, bottom: -2, safe: 80, lower: 2, higher: 20, caution: 0, danger: -1, snap: 1 };
export const SAFE_MM = HEIGHT.safe;

// An isometric cube, lit from above, in a cube's colour: blue, green, red, or the notch's grey.
export function cubeIcon(colour) {
  return `<svg class="cube-icon ${colour}" viewBox="0 0 32 32" aria-hidden="true">
    <path class="top" d="M16 3 28 9.5 16 16 4 9.5Z"/><path class="left" d="M4 9.5 16 16v13L4 22.5Z"/>
    <path class="right" d="M28 9.5 16 16v13l12-6.5Z"/></svg>`;
}

export class HeightColumn {
  // notch: desk mm, 4 over a cube's top. get() is the height the arm is sent to; set(z) changes it, render() shows
  // it, send() moves the arm there.
  constructor(notch, { get, set, render, send }) {
    this.notch = notch;
    this.top = HEIGHT.top;
    this.bottom = HEIGHT.bottom;
    this.isDragging = false;
    const slider = $('z-slider');
    const snapped = (value) => (Math.abs(value - notch) <= HEIGHT.snap ? notch : value);
    slider.addEventListener('input', () => { this.isDragging = true; set(snapped(Number(slider.value))); render(); });
    slider.addEventListener('change', () => { this.isDragging = false; set(snapped(Number(slider.value))); send(); });
    const go = (z) => { set(z); send(); };
    $('z-up').addEventListener('click', () => go(get() + 1));
    $('z-down').addEventListener('click', () => go(get() - 1));
    $('z-safe').innerHTML = `${HEIGHT.safe}<small>SAFE</small>`;
    $('z-safe').addEventListener('click', () => go(HEIGHT.safe));
    $('z-notch').innerHTML = `${cubeIcon('notch')}<span>${fmt(notch, 0)}<small>CUBE</small></span>`;
    $('z-notch').addEventListener('click', () => go(notch));
    $('z-notch-mark').firstElementChild.textContent = fmt(notch, 0);
    $('z-lower').addEventListener('click', () => { this.bottom -= HEIGHT.lower; render(); });
    $('z-higher').addEventListener('click', () => { this.top += HEIGHT.higher; render(); });
  }

  render(z) {
    const isAtNotch = Math.abs(z - this.notch) < 0.05;
    const slider = $('z-slider');
    slider.min = this.bottom;
    slider.max = this.top;
    if (!this.isDragging) slider.value = Math.round(z);
    $('z-out').textContent = fmt(z);
    $('z-out').classList.toggle('at-notch', isAtNotch);
    $('z-notch').classList.toggle('on', isAtNotch);
    $('z-safe').classList.toggle('on', z >= HEIGHT.safe - 0.5);
    $('z-top').textContent = this.top;
    $('z-bottom').textContent = this.bottom;
    const share = (value) => Math.min(1, Math.max(0, (this.top - value) / (this.top - this.bottom)));
    const at = (value) => `${100 * share(value)}%`;
    $('z-scale').style.background = `linear-gradient(to bottom, var(--line-strong) ${at(HEIGHT.caution)}, `
      + `var(--yellow) ${at(HEIGHT.caution)} ${at(HEIGHT.danger)}, var(--red) ${at(HEIGHT.danger)})`;
    const mark = $('z-notch-mark');
    mark.hidden = this.notch > this.top || this.notch < this.bottom;
    mark.style.top = `calc(8px + (100% - 16px) * ${share(this.notch)})`;
  }
}
