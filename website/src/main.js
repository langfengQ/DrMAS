import { experiments, formatGain, getResults } from './results.js';

document.documentElement.classList.add('js');

const state = { task: 'math', model: '8b', setting: 'shared', metric: 'avg' };
const modelSelect = document.querySelector('#model-select');
const explorer = document.querySelector('#results-explorer');

function renderResults(announce = true) {
  const result = getResults(state);
  document.querySelector('#results-body').innerHTML = result.rows.map((row) => `
    <tr class="${row.benchmark === 'Average' ? 'average-row' : ''}">
      <th scope="row">${row.benchmark}</th>
      <td class="bar-column" aria-hidden="true"><div class="bar-pair"><span class="bar grpo-bar" style="width:${row.grpo}%"></span><span class="bar drmas-bar" style="width:${row.drmas}%"></span></div></td>
      <td>${row.grpo.toFixed(1)}</td>
      <td class="drmas-cell">${row.drmas.toFixed(1)}</td>
      <td class="gain-cell ${row.gain < 0 ? 'negative' : row.gain === 0 ? 'neutral' : ''}">${formatGain(row.gain)}</td>
    </tr>
  `).join('');
  document.querySelector('#baseline-average').innerHTML = `${result.average.grpo.toFixed(1)}<span>%</span>`;
  document.querySelector('#drmas-average').innerHTML = `${result.average.drmas.toFixed(1)}<span>%</span>`;
  document.querySelector('#average-gain').textContent = `${formatGain(result.average.gain)} pp`;
  document.querySelector('#results-caption').textContent = result.description;
  document.querySelector('#table-number').textContent = result.table;
  document.querySelector('.table-source').href = `./paper.pdf#page=${result.page}`;
  for (const key of ['task', 'setting', 'metric']) {
    explorer.querySelectorAll(`[data-${key}]`).forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset[key] === state[key]));
    });
  }
  if (announce) {
    document.querySelector('#result-announcement').textContent = `${result.description}. Average: GRPO ${result.average.grpo.toFixed(1)} percent; Dr. MAS ${result.average.drmas.toFixed(1)} percent. Gain ${formatGain(result.average.gain)} percentage points.`;
  }
}

explorer.addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  if (button.dataset.task && button.dataset.task !== state.task) {
    state.task = button.dataset.task;
    state.model = experiments[state.task].defaultModel;
    modelSelect.replaceChildren(...Object.entries(experiments[state.task].models).map(([value, model]) => new Option(model.label, value)));
    modelSelect.value = state.model;
  } else if (button.dataset.setting) {
    state.setting = button.dataset.setting;
  } else if (button.dataset.metric) {
    state.metric = button.dataset.metric;
  } else {
    return;
  }
  renderResults();
});
modelSelect.addEventListener('change', () => {
  state.model = modelSelect.value;
  renderResults();
});
renderResults(false);
explorer.hidden = false;

const menuButton = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#navigation');
function closeMenu() {
  navigation.classList.remove('open');
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-label', 'Open navigation');
}
menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') !== 'true';
  navigation.classList.toggle('open', open);
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
});
navigation.addEventListener('click', (event) => {
  if (event.target.closest('a')) closeMenu();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') {
    closeMenu();
    menuButton.focus();
  }
});
document.addEventListener('click', (event) => {
  if (!event.target.closest('.site-header')) closeMenu();
});
window.matchMedia('(min-width: 541px)').addEventListener('change', closeMenu);

const dialog = document.querySelector('#figure-dialog');
const dialogImage = document.querySelector('#dialog-image');
document.querySelectorAll('[data-figure]').forEach((button) => {
  button.addEventListener('click', () => {
    dialogImage.src = button.dataset.figure;
    dialogImage.alt = button.querySelector('img').alt;
    document.querySelector('#figure-caption').textContent = button.dataset.caption;
    document.querySelector('#figure-original').href = button.dataset.figure;
    dialog.showModal();
    document.body.classList.add('dialog-open');
  });
});
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target !== dialog) return;
  const { left, right, top, bottom } = dialog.getBoundingClientRect();
  if (event.clientX < left || event.clientX > right || event.clientY < top || event.clientY > bottom) dialog.close();
});
dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));

const copyButton = document.querySelector('#copy-citation');
const copyStatus = document.querySelector('#copy-status');
copyButton.hidden = false;
let copyTimeout;

function legacyCopy(text) {
  const input = document.createElement('textarea');
  input.value = text;
  input.style.cssText = 'position:fixed;top:0;left:-9999px;';
  document.body.append(input);
  input.select();
  let copied = false;
  try {
    copied = document.execCommand('copy');
  } finally {
    input.remove();
    copyButton.focus({ preventScroll: true });
  }
  return copied;
}

copyButton.addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent;
  clearTimeout(copyTimeout);
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    try { copied = legacyCopy(text); } catch { /* Offer manual selection below. */ }
  }
  if (copied) {
    copyButton.querySelector('span').textContent = 'Copied!';
    copyStatus.textContent = 'Citation copied to clipboard.';
  } else {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Copy unavailable. Text selected; press Ctrl+C or Cmd+C.';
  }
  copyTimeout = setTimeout(() => {
    copyButton.querySelector('span').textContent = 'Copy citation';
    copyStatus.textContent = '';
  }, 5000);
});

if ('IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.07 });
  document.querySelectorAll('.reveal').forEach((element) => {
    element.classList.add('ready');
    revealObserver.observe(element);
  });

  const sectionObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      navigation.querySelectorAll('a').forEach((link) => {
        if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
  }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
  document.querySelectorAll('#overview, #method, #results, #citation').forEach((section) => sectionObserver.observe(section));
}
