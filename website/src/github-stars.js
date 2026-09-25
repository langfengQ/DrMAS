const cacheKey = 'drmas:github-stars:v1';
const cacheLifetime = 6 * 60 * 60 * 1000;
const sources = [
  { url: 'https://api.github.com/repos/langfengQ/DrMAS', field: 'stargazers_count', label: 'GitHub' },
  { url: 'https://img.shields.io/github/stars/langfengQ/DrMAS.json', field: 'value', label: 'Shields.io cache' },
];

export function formatStarCount(value) {
  if (Number.isSafeInteger(value) && value >= 0) return value.toLocaleString('en-US');
  if (typeof value !== 'string') return null;
  const label = value.trim();
  // Shields may abbreviate large counts. Preserve that label rather than inventing precision.
  if (/^(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d*(?:\.\d{1,2})?[kM])$/.test(label)) return label;
  return null;
}

export async function fetchStarCount(fetcher = fetch) {
  for (const source of sources) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetcher(source.url, {
        signal: controller.signal,
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) continue;
      const data = await response.json();
      const label = formatStarCount(data[source.field]);
      if (label !== null) return { label, source: source.label, savedAt: Date.now() };
    } catch {
      // Counts are supplementary; a blocked endpoint must not break the project page.
    } finally {
      clearTimeout(timer);
    }
  }
  return null;
}

export async function initGitHubStars() {
  const counters = document.querySelectorAll('[data-github-stars]');
  if (!counters.length) return;

  function render(record) {
    for (const counter of counters) {
      counter.textContent = record.label;
      counter.dataset.starsSource = record.source;
      counter.closest('.github-stars').title = `GitHub stars: ${record.label}. Last retrieved ${new Date(record.savedAt).toLocaleDateString('en-GB')} via ${record.source}.`;
    }
  }

  try {
    const cached = JSON.parse(localStorage.getItem(cacheKey));
    if (cached && formatStarCount(cached.label) !== null && Number.isFinite(cached.savedAt) &&
        typeof cached.source === 'string' && cached.savedAt <= Date.now()) {
      render(cached);
      if (Date.now() - cached.savedAt < cacheLifetime) return;
    }
  } catch { /* Storage can be disabled; the embedded count stays readable. */ }

  const latest = await fetchStarCount();
  if (!latest) return;
  render(latest);
  try { localStorage.setItem(cacheKey, JSON.stringify(latest)); } catch { /* Optional cache. */ }
}
