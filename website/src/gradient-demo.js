import { gradientMismatch } from './gradient-model.js';

const presets = {
  matched: { mean: 0, agentMean: 0, std: 0.5, agentStd: 0.5 },
  mean: { mean: 0, agentMean: 1, std: 0.5, agentStd: 0.5 },
  variance: { mean: 0, agentMean: 0, std: 0.5, agentStd: 1 },
  combined: { mean: 0, agentMean: 1, std: 0.5, agentStd: 1 },
};
const momentKeys = ['mean', 'agentMean', 'std', 'agentStd'];
const initialized = new WeakSet();
const svgNamespace = 'http://www.w3.org/2000/svg';
const curveStyles = {
  global: { class: 'gm-curve-global', stroke: '#d28a48', 'stroke-width': 3 },
  agent: { class: 'gm-curve-agent', stroke: '#8c569b', 'stroke-width': 2.5, 'stroke-dasharray': '7 5' },
};

function format(value) {
  return (Math.abs(value) < 1e-10 ? 0 : value).toFixed(2);
}

function formatPercent(value) {
  const rounded = Number(value.toFixed(2));
  return `${rounded > 0 ? '+' : ''}${format(rounded)}%`;
}

function svgElement(name, attributes = {}, text) {
  const node = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawSecondMomentResponse(svg, { moments, minStd, maxStd, remedy }) {
  const left = 70;
  const right = 620;
  const top = 24;
  const bottom = 255;
  const valueAt = (std) => gradientMismatch({ ...moments, std }).inflation;
  // The sweep is monotone in sigma. Its endpoint fixes the scale independently
  // of the current sigma, so moving that control only moves the selected point.
  const yMax = Math.max(valueAt(minStd), 1) * 1.15;
  const scaleX = (x) => left + (x - minStd) / (maxStd - minStd) * (right - left);
  const scaleY = (y) => bottom - y / yMax * (bottom - top);
  const marks = svgElement('g', { 'aria-hidden': 'true' });

  for (let index = 0; index < 5; index++) {
    const fraction = index / 4;
    const xValue = minStd + (maxStd - minStd) * fraction;
    const x = scaleX(xValue);
    const yValue = yMax * fraction;
    const y = scaleY(yValue);
    marks.append(
      svgElement('line', { x1: x, y1: top, x2: x, y2: bottom, class: 'gm-grid', stroke: '#e7e2df' }),
      svgElement('line', { x1: left, y1: y, x2: right, y2: y, class: 'gm-grid', stroke: '#e7e2df' }),
      svgElement('text', {
        x,
        y: bottom + 21,
        class: 'gm-axis-label',
        'text-anchor': index === 0 ? 'start' : index === 4 ? 'end' : 'middle',
        'font-size': 12,
        fill: '#706864',
      }, format(xValue)),
      svgElement('text', {
        x: left - 8,
        y: y + 4,
        class: 'gm-axis-label',
        'text-anchor': 'end',
        'font-size': 11,
        fill: '#706864',
      }, format(yValue)),
    );
  }

  marks.append(
    svgElement('path', {
      d: `M${left},${top} V${bottom} H${right}`,
      class: 'gm-axis',
      fill: 'none',
      stroke: '#a69a94',
    }),
    svgElement('text', {
      x: (left + right) / 2,
      y: 299,
      class: 'gm-axis-label gm-axis-title',
      'text-anchor': 'middle',
      'font-size': 13,
      fill: '#706864',
    }, 'Global standard deviation (\u03c3)'),
    svgElement('text', {
      x: 15,
      y: (top + bottom) / 2,
      transform: `rotate(-90 15 ${(top + bottom) / 2})`,
      class: 'gm-axis-label gm-axis-title',
      'text-anchor': 'middle',
      'font-size': 12,
      fill: '#706864',
    }, 'Second-moment multiplier'),
  );

  const samples = Array.from({ length: 321 }, (_, index) => minStd + (maxStd - minStd) * index / 320);
  const globalPath = samples.map((std, index) =>
    `${index === 0 ? 'M' : 'L'}${scaleX(std).toFixed(3)},${scaleY(valueAt(std)).toFixed(3)}`).join(' ');
  const selectedValue = remedy ? 1 : valueAt(moments.std);
  const selectedX = scaleX(moments.std);
  const selectedY = scaleY(selectedValue);
  const selectedColor = remedy ? curveStyles.agent.stroke : curveStyles.global.stroke;
  const labelOnLeft = selectedX > (left + right) / 2;

  marks.append(
    svgElement('path', {
      ...curveStyles.global,
      d: globalPath,
      fill: 'none',
      'stroke-linejoin': 'round',
      'stroke-linecap': 'round',
    }),
    svgElement('path', {
      ...curveStyles.agent,
      d: `M${left},${scaleY(1).toFixed(3)} L${right},${scaleY(1).toFixed(3)}`,
      fill: 'none',
    }),
    svgElement('line', {
      x1: selectedX,
      x2: selectedX,
      y1: top,
      y2: bottom,
      class: 'gm-current-guide',
      stroke: '#96918f',
      'stroke-dasharray': '3 4',
    }),
    svgElement('circle', {
      cx: selectedX,
      cy: selectedY,
      r: 5,
      class: 'gm-current-point',
      'data-std': moments.std,
      'data-value': selectedValue,
      'data-mode': remedy ? 'agent' : 'global',
      fill: selectedColor,
      stroke: '#ffffff',
      'stroke-width': 2,
    }),
    svgElement('text', {
      x: selectedX + (labelOnLeft ? -9 : 9),
      y: Math.max(top + 12, selectedY - 10),
      class: 'gm-current-label',
      'text-anchor': labelOnLeft ? 'end' : 'start',
      'font-size': 13,
      fill: selectedColor,
    }, `${format(selectedValue)}\u00d7`),
  );

  const titleId = `${svg.id}-title`;
  const descriptionId = `${svg.id}-description`;
  const mode = remedy ? 'agent-wise' : 'global';
  svg.setAttribute('viewBox', '0 0 640 310');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-labelledby', titleId);
  svg.setAttribute('aria-describedby', descriptionId);
  svg.dataset.xMin = String(minStd);
  svg.dataset.xMax = String(maxStd);
  svg.dataset.yMax = String(yMax);
  svg.replaceChildren(
    svgElement('title', { id: titleId }, 'Gradient second-moment multiplier: illustrative response to global standard deviation'),
    svgElement('desc', { id: descriptionId }, `Global standard deviation varies from ${format(minStd)} to ${format(maxStd)}, holding global mean ${format(moments.mean)}, agent mean ${format(moments.agentMean)}, and agent standard deviation ${format(moments.agentStd)} fixed. Orange shows global normalization; the dashed purple agent-wise reference is 1.00. The selected ${mode} point at standard deviation ${format(moments.std)} has multiplier ${format(selectedValue)}. The score-function second moment is fixed at 1 and both covariance corrections are zero. This is not a measured gradient trace.`),
    marks,
  );
}

export function initGradientDemo() {
  if (typeof document === 'undefined') return;
  const root = document.querySelector('#mismatch-demo');
  if (!root || initialized.has(root)) return;

  const controls = Object.fromEntries(momentKeys.map((key) => [key, root.querySelector(`[data-moment="${key}"]`)]));
  const secondMomentChart = root.querySelector('#second-moment-response');
  const remedyButton = root.querySelector('#apply-remedy');
  if (Object.values(controls).some((control) => !control) || !secondMomentChart || !remedyButton) return;
  const remedyLabel = remedyButton.firstChild?.nodeType === Node.TEXT_NODE ? remedyButton.firstChild : remedyButton;
  const minStd = Number(controls.std.min);
  const maxStd = Number(controls.std.max);

  let moments = { ...presets.combined };
  let remedy = false;

  function setText(selector, text) {
    const node = root.querySelector(selector);
    if (node && node.textContent !== text) node.textContent = text;
  }

  function readControls() {
    moments = Object.fromEntries(momentKeys.map((key) => [key, Number(controls[key].value)]));
  }

  function applyMoments(next) {
    moments = { ...next };
    for (const key of momentKeys) controls[key].value = moments[key];
  }

  function render(announce = false) {
    const result = gradientMismatch(moments);
    const activeFactor = remedy ? result.remedyInflation : result.inflation;
    const mode = remedy ? 'Agent-wise normalization' : 'Global normalization';
    const factorState = Math.abs(result.inflation - 1) < 1e-9
      ? 'Matched scale'
      : result.inflation > 1 ? 'Amplification' : 'Attenuation';

    for (const key of momentKeys) {
      const control = controls[key];
      const progress = 100 * (moments[key] - Number(control.min)) / (Number(control.max) - Number(control.min));
      setText(`[data-value="${key}"]`, format(moments[key]));
      control.setAttribute('aria-valuetext', format(moments[key]));
      control.style.setProperty('--range-fill', `${progress}%`);
    }
    const numericReadouts = {
      '#gm-factor': result.inflation,
      '#gm-variance-term': result.varianceRatio,
      '#gm-mean-term': result.meanMismatch,
      '#gm-active-factor': activeFactor,
    };
    for (const [selector, value] of Object.entries(numericReadouts)) setText(selector, format(value));
    setText('#gm-active-second-change', formatPercent(remedy ? 0 : result.secondMomentChangePercent));
    setText('#gm-factor-state', factorState);
    setText('#gm-demo-state', mode);

    root.dataset.remedy = String(remedy);
    remedyButton.setAttribute('aria-pressed', String(remedy));
    remedyLabel.textContent = remedy ? 'Restore global baseline' : 'Apply the remedy';
    root.querySelectorAll('.gm-formula-global').forEach((node) => { node.hidden = remedy; });
    root.querySelectorAll('.gm-formula-agent').forEach((node) => { node.hidden = !remedy; });
    root.querySelectorAll('[data-preset]').forEach((button) => {
      const preset = presets[button.dataset.preset];
      const selected = preset && momentKeys.every((key) => Math.abs(preset[key] - moments[key]) < 1e-9);
      button.setAttribute('aria-pressed', String(Boolean(selected)));
    });

    drawSecondMomentResponse(secondMomentChart, { moments, minStd, maxStd, remedy });

    if (announce) {
      setText('#gradient-status', `${mode}. Global reward mean ${format(moments.mean)}, standard deviation ${format(moments.std)}. Agent reward mean ${format(moments.agentMean)}, standard deviation ${format(moments.agentStd)}. Illustrative gradient second-moment multiplier ${format(activeFactor)}, ${formatPercent(remedy ? 0 : result.secondMomentChangePercent)} versus agent-wise. The score second moment is fixed at 1 and both covariance corrections are zero.`);
    }
  }

  for (const control of Object.values(controls)) {
    control.addEventListener('input', () => {
      readControls();
      render();
    });
    control.addEventListener('change', () => {
      readControls();
      render(true);
    });
  }
  root.querySelectorAll('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const name = button.dataset.preset;
      if (!Object.hasOwn(presets, name)) return;
      applyMoments(presets[name]);
      render(true);
    });
  });
  remedyButton.addEventListener('click', () => {
    remedy = !remedy;
    render(true);
  });

  applyMoments(presets.combined);
  render();
  initialized.add(root);
  root.hidden = false;
}
