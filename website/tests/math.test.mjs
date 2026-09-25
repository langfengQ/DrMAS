import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { renderMath } from '../scripts/render-math.mjs';

test('display math produces KaTeX HTML and accessible MathML at build time', () => {
  const html = renderMath(String.raw`<div class="math-block" data-math>$$ A^{i,k}_{\mathrm{agent}} = \frac{R^i-\mu_k}{\sigma_k} $$</div>`);
  assert.ok(html.includes('class="katex-display"'));
  assert.ok(html.includes('class="katex-html" aria-hidden="true"'));
  assert.ok(html.includes('<math xmlns="http://www.w3.org/1998/Math/MathML"'));
  assert.ok(html.includes('<mfrac>'));
  assert.ok(!html.includes('$$'));
});

test('inline math does not become a display block', () => {
  const html = renderMath(String.raw`<span data-math>\(Y_k\)</span>`);
  assert.ok(html.includes('class="katex"'));
  assert.ok(!html.includes('katex-display'));
});

test('ordinary HTML and monetary dollar amounts are left untouched', () => {
  const html = '<p>Cost: $97.5 to $56.7.</p><code>$$ raw example $$</code>';
  assert.equal(renderMath(html), html);
});

test('invalid LaTeX or missing delimiters fail the build instead of publishing broken math', () => {
  assert.throws(() => renderMath(String.raw`<div data-math>$$ \frac{1}{ $$</div>`));
  assert.throws(() => renderMath('<div data-math>x + y</div>'), /delimiters/);
});

test('all equations in the manuscript page compile successfully', () => {
  const source = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const expected = (source.match(/\bdata-math\b/g) || []).length;
  const rendered = renderMath(source);
  assert.ok(expected >= 9);
  assert.equal((rendered.match(/class="katex-mathml"/g) || []).length, expected);
  assert.ok(!rendered.includes('katex-error'));
});
