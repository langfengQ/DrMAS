// Tables 1 and 2 of Dr__MAS_final.pdf. Each pair is [avg@16, pass@16].
// The last row is the paper's reported average, not a recomputed rounded mean.
export const experiments = {
  math: {
    label: 'Math reasoning',
    table: 1,
    page: 7,
    defaultModel: '8b',
    benchmarks: ["AIME '24", "AIME '25", "AMC '23", 'MATH500', 'Minerva', 'Olympiad', 'Average'],
    models: {
      '8b': {
        label: 'Qwen3-8B',
        shared: {
          grpo: [[42.7, 66.7], [31.4, 53.3], [87.3, 95.0], [89.6, 96.2], [37.5, 50.0], [58.2, 71.4], [57.8, 72.1]],
          drmas: [[54.8, 80.0], [39.4, 70.0], [88.9, 97.5], [91.3, 96.0], [39.9, 49.6], [59.3, 72.4], [62.3, 77.6]],
        },
        separate: {
          grpo: [[42.9, 70.0], [31.8, 53.3], [86.1, 95.0], [90.5, 96.6], [39.2, 50.7], [58.2, 67.6], [58.1, 72.2]],
          drmas: [[44.6, 73.3], [41.5, 56.7], [87.5, 95.0], [90.7, 96.2], [40.9, 54.0], [59.0, 70.2], [60.7, 74.2]],
        },
      },
      '4b': {
        label: 'Qwen3-4B',
        shared: {
          grpo: [[39.3, 64.0], [31.4, 53.3], [85.6, 95.0], [89.5, 96.2], [37.5, 50.0], [57.6, 65.6], [56.8, 70.7]],
          drmas: [[39.3, 63.3], [38.1, 63.3], [87.3, 95.0], [90.5, 96.0], [40.9, 53.3], [58.2, 68.6], [59.0, 73.2]],
        },
        separate: {
          grpo: [[42.7, 73.3], [35.6, 63.3], [83.5, 95.0], [89.6, 95.0], [37.5, 50.7], [56.3, 68.9], [57.5, 74.4]],
          drmas: [[46.9, 80.0], [38.1, 66.7], [89.5, 97.5], [92.4, 97.0], [39.0, 51.5], [60.9, 73.6], [61.1, 77.7]],
        },
      },
    },
  },
  search: {
    label: 'Multi-turn search',
    table: 2,
    page: 8,
    defaultModel: '7b',
    benchmarks: ['NQ', 'TriviaQA', 'PopQA', 'HotpotQA', '2Wiki', 'MuSiQue', 'Bamboogle', 'Average'],
    models: {
      '7b': {
        label: 'Qwen2.5-7B',
        shared: {
          grpo: [[45.2, 60.0], [63.9, 70.9], [43.9, 55.0], [40.3, 55.0], [41.6, 67.8], [15.2, 31.7], [40.1, 58.4], [41.5, 57.0]],
          drmas: [[47.4, 60.7], [63.1, 71.2], [45.9, 57.3], [42.5, 56.0], [42.0, 67.1], [16.7, 32.1], [40.1, 59.2], [42.5, 57.7]],
        },
        separate: {
          grpo: [[27.1, 39.0], [53.1, 64.4], [20.7, 27.9], [24.4, 36.2], [30.3, 51.2], [8.3, 18.1], [31.9, 46.4], [28.0, 40.5]],
          drmas: [[47.7, 59.5], [63.4, 72.7], [46.7, 57.8], [44.0, 57.5], [45.4, 68.1], [19.4, 34.9], [39.8, 57.6], [43.8, 58.3]],
        },
      },
      '3b': {
        label: 'Qwen2.5-3B',
        shared: {
          grpo: [[41.0, 59.0], [57.9, 68.4], [43.2, 58.0], [32.5, 48.0], [33.7, 64.0], [9.1, 26.5], [26.4, 46.4], [34.8, 52.9]],
          drmas: [[43.8, 58.5], [61.7, 70.1], [45.0, 57.6], [33.3, 51.2], [34.1, 64.0], [10.2, 25.8], [28.6, 49.6], [36.7, 53.8]],
        },
        separate: {
          grpo: [[43.8, 54.5], [60.6, 70.8], [45.6, 54.5], [32.5, 45.2], [29.2, 48.9], [8.6, 19.2], [21.0, 33.6], [34.5, 46.7]],
          drmas: [[44.6, 58.1], [61.1, 71.7], [46.5, 57.4], [35.3, 51.1], [34.9, 60.2], [10.4, 26.1], [25.4, 46.4], [36.9, 53.0]],
        },
      },
    },
  },
};

export function getResults({ task, model, setting, metric }) {
  const experiment = experiments[task];
  const modelData = experiment?.models[model];
  const data = modelData?.[setting];
  if (!data || !['avg', 'pass'].includes(metric)) {
    throw new RangeError('Unknown experiment configuration');
  }
  const column = metric === 'avg' ? 0 : 1;
  const rows = experiment.benchmarks.map((benchmark, index) => ({
    benchmark,
    grpo: data.grpo[index][column],
    drmas: data.drmas[index][column],
    gain: Number((data.drmas[index][column] - data.grpo[index][column]).toFixed(1)),
  }));
  return {
    rows,
    average: rows.at(-1),
    description: `${experiment.label} with ${modelData.label}, ${setting === 'shared' ? 'shared' : 'separate'} weights, ${metric}@16`,
    table: experiment.table,
    page: experiment.page,
  };
}

export function formatGain(gain) {
  return `${gain > 0 ? '+' : ''}${gain.toFixed(1)}`;
}
