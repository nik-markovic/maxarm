// A layer drawn in Python over the camera's picture (applib/overlay): a transparent PNG laid over the live view,
// asked for at the size the picture is shown, in device pixels, whenever what it shows or that size changes, and
// hidden while there is nothing to show. The stream's overlay is drawn by the same code.
//
//   const layer = new PictureLayer(img, (width, scale) => `/api/tiles/scene.png?w=${width}&s=${scale}`);
//   layer.show(key);          // key: what it shows now, as text -- or null for nothing

// Asked for again when the picture's width changes by more than this share of it.
const RESIZE_SHARE = 0.03;

export class PictureLayer {
  constructor(img, address) {
    this.img = img;
    this.address = address;
    this.key = null;
    this.asked = { key: null, width: 0 };
    this.count = 0;
    img.hidden = true;
    img.addEventListener('load', () => { img.hidden = this.key === null; });
    img.addEventListener('error', () => { img.hidden = true; });
    new ResizeObserver(() => this.refresh()).observe(img.parentElement);
  }

  show(key) {
    this.key = key;
    this.refresh();
  }

  refresh() {
    if (this.key === null) {
      this.img.hidden = true;
      this.asked = { key: null, width: 0 };
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(this.img.parentElement.clientWidth * ratio);
    const { key, width: asked } = this.asked;
    if (!width || (key === this.key && Math.abs(width - asked) <= asked * RESIZE_SHARE)) return;
    this.asked = { key: this.key, width };
    this.count += 1;
    this.img.src = `${this.address(width, ratio)}&k=${this.count}`;
  }
}
