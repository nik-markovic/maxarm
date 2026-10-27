// The camera's picture as large as the window allows, a page's marks over it, to look closely and click on.
//
//   const full = openFullView({ image, aspect, title, hint, Marks, onPick, onHover, onReset, onClose });
//   full.update(marks, doing, { label, value, note, isOff });   // the marks; the bar's point and what is running
//
// `image` is the picture's address: the live view's stream, or a still. `Marks` draws over it (a ViewMarks).
// `onPick(u, v)` gets each click as shares of the picture, `onHover()` is told the pointer moved, if given. Esc or x
// closes it, and `onClose()` is told.

const MARGIN_PX = 14;
const RETRY_MS = 3000;

export function openFullView({ image, aspect, title, hint, Marks, onPick, onHover = () => {}, onReset, onClose }) {
  const layer = document.createElement('div');
  layer.className = 'fullview';
  layer.innerHTML = `
    <div class="fullview-bar">
      <div class="fullview-title"><b>${title}</b><span>${hint}</span></div>
      <div class="fullview-point"><span></span><b class="num">—</b><small class="landed"></small></div>
      <span class="fullview-doing"></span>
      <button class="btn reset" title="Stop everything, let go, lift clear and go home">RESET</button>
      <button class="btn icon close" title="Close (Esc)">×</button>
    </div>
    <div class="fullview-stage"><div class="fullview-frame"><img alt=""><canvas></canvas></div></div>`;
  document.body.append(layer);
  const stage = layer.querySelector('.fullview-stage');
  const frame = layer.querySelector('.fullview-frame');
  const canvas = layer.querySelector('canvas');
  const picture = layer.querySelector('img');
  const marks = new Marks(canvas);
  const connect = () => {
    if (layer.isConnected) picture.src = image.includes('?') ? image : `${image}?t=${Date.now()}`;
  };
  picture.addEventListener('error', () => setTimeout(connect, RETRY_MS));
  connect();

  const fit = () => {
    const width = Math.min(stage.clientWidth - 2 * MARGIN_PX, (stage.clientHeight - 2 * MARGIN_PX) * aspect);
    frame.style.width = `${width}px`;
    frame.style.height = `${width / aspect}px`;
  };
  const stageWatch = new ResizeObserver(fit);
  stageWatch.observe(stage);
  const share = (event) => {
    const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height];
  };
  canvas.addEventListener('click', (event) => onPick(...share(event)));
  canvas.addEventListener('pointermove', (event) => { marks.update({ hover: share(event) }); onHover(); });
  canvas.addEventListener('pointerleave', () => { marks.update({ hover: null }); onHover(); });
  layer.querySelector('.reset').addEventListener('click', onReset);

  const onKey = (event) => { if (event.key === 'Escape') close(); };
  function close() {
    stageWatch.disconnect();
    picture.removeAttribute('src');                // a stream ends: the camera is let go if nothing else watches
    layer.remove();
    document.removeEventListener('keydown', onKey);
    onClose();
  }
  layer.querySelector('.close').addEventListener('click', close);
  document.addEventListener('keydown', onKey);

  return {
    close,
    isOpen: () => layer.isConnected,
    marks,
    // `point`: what the bar shows on the right -- a label, a value, and a note under it, yellow if `isOff`.
    update(next, doing, point = null) {
      marks.update(next);
      layer.querySelector('.fullview-point span').textContent = point ? point.label : '';
      layer.querySelector('.fullview-point b').textContent = point && point.value ? point.value : '—';
      const line = layer.querySelector('.fullview-point .landed');
      line.textContent = point && point.note ? point.note : '';
      line.classList.toggle('off', Boolean(point && point.isOff));
      layer.querySelector('.fullview-doing').textContent = doing || '';
    },
  };
}
