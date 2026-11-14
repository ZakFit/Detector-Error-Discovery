import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const resolve = createRequire(join(process.argv[2], 'runtime-loader.cjs')).resolve;
const { Workbook } = await import(pathToFileURL(resolve('@oai/artifact-tool')).href);
const outputs = fileURLToPath(new URL('../../../../outputs/', import.meta.url));
const auditDir = join(outputs, 'manual_bbox_review');

async function loadCsv(path) {
  const csv = await readFile(path, 'utf8');
  const workbook = await Workbook.fromCSV(csv, { sheetName: 'Data' });
  const [header, ...rows] = workbook.worksheets.getItem('Data').getUsedRange().values;
  return { header, rows: rows.map(row => Object.fromEntries(header.map((key, i) => [key, row[i]]))) };
}

function indexById(rows) {
  const indexed = new Map(rows.map(row => [row.id, row]));
  assert.equal(indexed.size, rows.length, 'Duplicate IDs');
  return indexed;
}

function unitValue(value) {
  assert(value !== null && value !== undefined && value !== '', 'Missing score');
  const number = Number(value);
  assert(Number.isFinite(number) && number >= 0 && number <= 1, 'Score outside [0, 1]');
  return number;
}

function blendManual(base, row) {
  const failure = unitValue(row.boundary_failure_fraction);
  const nominal = unitValue(row.manual_error_nominal);
  assert(nominal === 0 || nominal === 1);
  const manual = 0.85 * failure + 0.15 * nominal;
  const qualityWeight = { high: 0.22, medium: 0.12, low: 0.05 }[row.quality];
  assert(qualityWeight !== undefined);
  let weight = qualityWeight * (0.5 + Math.abs(manual - 0.5));
  if (String(row.conservative_override).toLowerCase() === 'true') {
    weight = row.error_reason === 'no_same_class_object' ? 0.75 : 0.55;
  }
  return base + weight * (manual - base);
}

const [template, baseline, audit] = await Promise.all([
  loadCsv(join(auditDir, 'source_tables', 'sample_submission.csv')),
  loadCsv(join(outputs, 'submission_person_refiner_280.csv')),
  loadCsv(join(auditDir, 'detection_audit.csv')),
]);
assert.deepEqual(template.header, ['id', 'score']);
const templateIds = [...indexById(template.rows).keys()];
assert.equal(templateIds.length, 362);
const model = indexById(baseline.rows);
const review = indexById(audit.rows);
assert.equal(model.size, templateIds.length);
assert.equal(review.size, templateIds.length);

const baselineScores = [];
const resultRows = templateIds.map(id => {
  assert.match(id, /^[0-9a-f]{16}$/);
  const base = unitValue(model.get(id).score);
  const row = review.get(id);
  assert(Math.abs(base - unitValue(row.baseline_091063)) < 1e-12);
  baselineScores.push(base);
  return [id, Number(blendManual(base, row).toFixed(12))];
});
const submission = Workbook.create().worksheets.add('Submission');
const outputRange = submission.getRange(`A1:B${resultRows.length + 1}`);
outputRange.values = [template.header, ...resultRows];
const path = join(outputs, 'submission_manual_review_full.csv');
await writeFile(path, outputRange.values.map(row => row.join(',')).join('\n') + '\n', 'utf8');

const checked = await loadCsv(path);
assert.deepEqual(checked.header, template.header);
assert.deepEqual(checked.rows.map(row => row.id), templateIds);
const scores = checked.rows.map(row => unitValue(row.score));
assert.deepEqual(scores, resultRows.map(row => row[1]));
const top = values => new Set([...values.keys()].sort((i, j) => values[j] - values[i]).slice(0, 50));
const baselineTop = top(baselineScores);
const updatedTop = top(scores);
console.log(JSON.stringify({
  path,
  rows: scores.length,
  distinctScores: new Set(scores).size,
  changed: scores.filter((score, i) => Math.abs(score - baselineScores[i]) > 1e-10).length,
  top50Overlap: [...updatedTop].filter(i => baselineTop.has(i)).length,
  minimum: Math.min(...scores),
  maximum: Math.max(...scores),
}));
