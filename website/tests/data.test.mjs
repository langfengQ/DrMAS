import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { experiments, formatGain, getResults } from '../src/results.js';

test('every configuration has complete, valid scores and a reported average', () => {
  let configurations = 0;
  for (const [task, experiment] of Object.entries(experiments)) {
    for (const [model, modelData] of Object.entries(experiment.models)) {
      for (const setting of ['shared', 'separate']) {
        for (const method of ['grpo', 'drmas']) {
          const scores = modelData[setting][method];
          assert.equal(scores.length, experiment.benchmarks.length);
          for (const [avg, pass] of scores) {
            assert.ok(Number.isFinite(avg) && Number.isFinite(pass));
            assert.ok(avg >= 0 && pass <= 100 && avg <= pass);
          }
        }
        for (const metric of ['avg', 'pass']) {
          const result = getResults({ task, model, setting, metric });
          assert.equal(result.average.benchmark, 'Average');
          assert.ok(result.description.includes(modelData.label));
          configurations++;
        }
      }
    }
  }
  assert.equal(configurations, 16);
});

test('representative math results match Table 1', () => {
  const result = getResults({ task: 'math', model: '8b', setting: 'shared', metric: 'avg' });
  assert.deepEqual(result.average, { benchmark: 'Average', grpo: 57.8, drmas: 62.3, gain: 4.5 });
  assert.deepEqual(result.rows[0], { benchmark: "AIME '24", grpo: 42.7, drmas: 54.8, gain: 12.1 });
});

test('representative search results match Table 2', () => {
  const avg = getResults({ task: 'search', model: '7b', setting: 'separate', metric: 'avg' });
  const pass = getResults({ task: 'search', model: '7b', setting: 'separate', metric: 'pass' });
  assert.deepEqual(avg.average, { benchmark: 'Average', grpo: 28.0, drmas: 43.8, gain: 15.8 });
  assert.deepEqual(pass.average, { benchmark: 'Average', grpo: 40.5, drmas: 58.3, gain: 17.8 });
});

test('zero and negative gains are retained, not hidden', () => {
  const result = getResults({ task: 'math', model: '4b', setting: 'shared', metric: 'pass' });
  assert.equal(result.rows[0].gain, -0.7);
  assert.equal(result.rows[2].gain, 0);
  assert.equal(formatGain(-0.7), '-0.7');
  assert.equal(formatGain(0), '0.0');
  assert.equal(formatGain(4.5), '+4.5');
});

test('invalid combinations fail explicitly', () => {
  assert.throws(() => getResults({ task: 'search', model: '8b', setting: 'shared', metric: 'avg' }), RangeError);
  assert.throws(() => getResults({ task: 'math', model: '8b', setting: 'shared', metric: 'unknown' }), RangeError);
});

test('downloadable BibTeX matches the displayed citation', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const displayed = html.match(/<code id="bibtex">([\s\S]*?)<\/code>/)[1].trim();
  const downloadable = readFileSync(new URL('../public/citation.bib', import.meta.url), 'utf8').trim();
  assert.equal(displayed, downloadable);
});

test('published PDF is an exact copy of the source paper', () => {
  assert.deepEqual(
    readFileSync(new URL('../public/paper.pdf', import.meta.url)),
    readFileSync(new URL('../Dr__MAS_final.pdf', import.meta.url)),
  );
});
