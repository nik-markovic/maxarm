// The Fine Tune table and its line under SAVE, as tests/refine.js writes them, for describe-check.py's JSON files.
//
//   node --experimental-default-type=module impala/agenttools/height/render-check.mjs FILE.json ...

import { readFileSync } from 'node:fs';
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = pathToFileURL(path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../calibration')).href;
register(`data:text/javascript,${encodeURIComponent(`
export async function resolve(specifier, context, next) {
  return next(specifier.startsWith('/') ? ${JSON.stringify(root)} + specifier : specifier, context);
}`)}`);
globalThis.document = { getElementById: () => null, querySelector: () => null, addEventListener: () => {} };
globalThis.window = globalThis;
const { tableRows, fitLine } = await import(`${root}/tests/refine.js`);
for (const file of process.argv.slice(2)) {
  const described = JSON.parse(readFileSync(file, 'utf8'));
  const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
  console.log(`== ${path.basename(file)}`);
  console.log(text(fitLine(described)));
  console.log(text(tableRows(described, [1.0, 2.0])).slice(0, 400));
}
