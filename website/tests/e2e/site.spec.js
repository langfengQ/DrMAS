import { test, expect } from '@playwright/test';
import { experiments, getResults, formatGain } from '../../src/results.js';

test('production page loads under /DrMAS/ without broken local assets', async ({ page, request }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('response', (response) => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  await page.goto('./');
  await expect(page).toHaveTitle(/Dr\. MAS/);
  await expect(page.locator('h1')).toContainText('Stable Reinforcement Learning');
  await expect(page.locator('#results-body tr')).toHaveCount(7);
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.fonts.check('500 20px "Space Grotesk"'))).toBe(true);
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim())).toBe('#8c569b');
  await expect(page.locator('.grpo-bar').first()).toHaveCSS('background-color', 'rgb(247, 176, 111)');
  await expect(page.locator('.drmas-cell').first()).toHaveCSS('background-color', 'rgb(238, 240, 242)');
  for (const asset of ['paper.pdf', 'citation.bib', 'figures/framework.png', 'figures/training-dynamics.png', 'favicon.svg']) {
    const response = await request.get(asset);
    expect(response.status()).toBe(200);
    if (asset.endsWith('.pdf')) expect(response.headers()['content-type']).toBe('application/pdf');
  }
  expect(errors).toEqual([]);
});

test('all 16 result configurations render the exact source data', async ({ page }) => {
  await page.goto('./');
  for (const [task, experiment] of Object.entries(experiments)) {
    await page.locator(`[data-task="${task}"]`).click();
    for (const model of Object.keys(experiment.models)) {
      await page.locator('#model-select').selectOption(model);
      for (const setting of ['shared', 'separate']) {
        await page.locator(`[data-setting="${setting}"]`).click();
        for (const metric of ['avg', 'pass']) {
          await page.locator(`[data-metric="${metric}"]`).click();
          const result = getResults({ task, model, setting, metric });
          await expect(page.locator('#results-caption')).toHaveText(result.description);
          await expect(page.locator('#average-gain')).toHaveText(`${formatGain(result.average.gain)} pp`);
          await expect(page.locator(`[data-metric="${metric}"]`)).toHaveAttribute('aria-pressed', 'true');
          const rendered = await page.locator('#results-body tr').evaluateAll((rows) => rows.map((row) => [...row.querySelectorAll('th, td:not(.bar-column)')].map((cell) => cell.textContent)));
          expect(rendered).toEqual(result.rows.map((row) => [row.benchmark, row.grpo.toFixed(1), row.drmas.toFixed(1), formatGain(row.gain)]));
        }
      }
    }
  }
});

test('image dialog opens, closes with Escape, and returns focus', async ({ page }) => {
  await page.goto('./');
  const trigger = page.getByRole('button', { name: 'Enlarge framework diagram' });
  await trigger.click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.locator('#dialog-image')).toHaveAttribute('src', './figures/framework.png');
  await expect(page.locator('#figure-caption')).toContainText('Figure 2');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await trigger.click();
  await page.getByRole('button', { name: 'Close enlarged figure' }).click();
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await expect(page.locator('body')).not.toHaveClass('dialog-open');
});

test('citation copies to the clipboard', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('./');
  await page.getByRole('button', { name: 'Copy citation' }).click();
  await expect(page.locator('#copy-status')).toHaveText('Citation copied to clipboard.');
  const clipboard = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboard).toEqual(await page.locator('#bibtex').textContent());
});

test('clipboard denial falls back without a page error', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: () => Promise.reject(new Error('Denied')) } });
    document.execCommand = () => false;
  });
  await page.goto('./');
  await page.getByRole('button', { name: 'Copy citation' }).click();
  await expect(page.locator('#copy-status')).toContainText('Text selected');
  expect(await page.evaluate(() => window.getSelection().toString())).toContain('@misc{feng2026drmas');
});

test('layouts fit narrow phones, tablets, and desktop screens', async ({ page }) => {
  await page.goto('./');
  for (const width of [320, 375, 390, 540, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth), `overflow at ${width}px`).toBeLessThanOrEqual(width);
  }
});

test('mobile navigation closes after selection and Escape', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('./');
  const menu = page.locator('.menu-toggle');
  await menu.click();
  await expect(menu).toHaveAttribute('aria-expanded', 'true');
  await page.getByRole('navigation').getByRole('link', { name: 'Results', exact: true }).click();
  await expect(page).toHaveURL(/#results$/);
  await expect(menu).toHaveAttribute('aria-expanded', 'false');
  await menu.click();
  await page.keyboard.press('Escape');
  await expect(menu).toHaveAttribute('aria-expanded', 'false');
  await expect(menu).toBeFocused();
});

test('paper and citation remain available without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4173/DrMAS/');
  await expect(page.getByRole('link', { name: 'Read the paper' })).toBeVisible();
  await expect(page.locator('#bibtex')).toContainText('@misc{feng2026drmas');
  await expect(page.getByRole('link', { name: 'Tables 1 and 2 of the paper' })).toBeVisible();
  await expect(page.getByRole('navigation')).toBeVisible();
  await context.close();
});
