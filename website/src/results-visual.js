function requireFinite(value, name) {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be a finite number`);
}

export function getGainScale(gains) {
  if (!Array.isArray(gains)) throw new TypeError('gains must be an array');
  let maximum = 0;
  for (const gain of gains) {
    requireFinite(gain, 'gain');
    maximum = Math.max(maximum, Math.abs(gain));
  }

  // Round the shared, symmetric extent outwards, never crop an observed gain.
  const extent = Math.max(1, maximum);
  const step = 10 ** Math.floor(Math.log10(extent)) / 2;
  const limit = Math.ceil(extent / step) * step;
  if (!Number.isFinite(limit)) throw new RangeError('gains exceed a finite chart scale');

  return { limit, ticks: [-limit, -limit / 2, 0, limit / 2, limit] };
}

export function getGainGeometry(gain, limit) {
  requireFinite(gain, 'gain');
  requireFinite(limit, 'limit');
  if (limit <= 0) throw new RangeError('limit must be greater than zero');
  if (Math.abs(gain) > limit) throw new RangeError('gain must fit within the chart scale');

  const end = 50 + gain / limit * 50;
  return {
    start: Math.min(50, end),
    width: Math.abs(gain) / limit * 50,
    end,
    direction: gain > 0 ? 'positive' : gain < 0 ? 'negative' : 'neutral',
  };
}

export function formatGainTick(value) {
  requireFinite(value, 'tick');
  const rounded = Number(value.toFixed(2));
  return `${rounded > 0 ? '+' : ''}${rounded}`;
}
