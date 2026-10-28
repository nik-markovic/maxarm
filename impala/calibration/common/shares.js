// Desk space and the live view, through the camera calibration. `matrix` is 3x4: desk (x, y, z) to shares of the
// view's width and height, 0 to 1 across it, as the tests' state carries it -- the same at any size of the picture.
// The app converts with the library; these are for drawing, and for saying at once where a click is.

// Where in the view a desk point is seen, as shares; null if it is behind the camera.
export function toView(matrix, [x, y, z]) {
  const [u, v, w] = matrix.map((row) => row[0] * x + row[1] * y + row[2] * z + row[3]);
  return w > 0 ? [u / w, v / w] : null;
}

// The desk point seen at a place in the view, on the plane `height` over the desk; null if that plane is not seen
// there (above the horizon).
export function toDesk(matrix, [u, v], height = 0) {
  const plane = matrix.map((row) => [row[0], row[1], row[3] + height * row[2]]);
  const solved = solve(plane, [u, v, 1]);
  return solved && solved[2] > 0 ? [solved[0] / solved[2], solved[1] / solved[2]] : null;
}

// A circle round a desk point on the plane `height` over the desk, as shares of the view: a ring drawn in perspective.
// `deskToTrue` (2x2, the calibration's) makes it a circle of true mm, the size of a real thing: desk mm carry the
// arm's x/y error.
export function circle(matrix, [x, y], radius, height = 0, deskToTrue = null, count = 48) {
  const [[a, b], [c, d]] = deskToTrue ? inverse(deskToTrue) : [[1, 0], [0, 1]];
  return Array.from({ length: count + 1 }, (_, index) => {
    const turn = (2 * Math.PI * index) / count;
    const [dx, dy] = [radius * Math.cos(turn), radius * Math.sin(turn)];
    return toView(matrix, [x + a * dx + b * dy, y + c * dx + d * dy, height]);
  });
}

function inverse([[a, b], [c, d]]) {
  const det = a * d - b * c;
  return [[d / det, -b / det], [-c / det, a / det]];
}

// a x = b, for a 3x3 a: Cramer's rule. Null if a is singular.
function solve(a, b) {
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const whole = det(a);
  if (Math.abs(whole) < 1e-15) return null;
  const replaced = (column) => a.map((row, index) => row.map((value, at) => (at === column ? b[index] : value)));
  return [0, 1, 2].map((column) => det(replaced(column)) / whole);
}
