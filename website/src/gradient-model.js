function requireFinite(value, name) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
}

function requirePositiveStd(value, name) {
  requireFinite(value, name);
  if (value <= 0) {
    throw new RangeError(`${name} must be greater than zero`);
  }
}

// Lemma 4.2's reward-statistics factor, not a measured gradient norm.
// Second-moment comparisons fix the positive score second moment and set
// both global and agent-wise score-reward covariance corrections to zero.
export function gradientMismatch({ mean, agentMean, std, agentStd }) {
  requireFinite(mean, 'mean');
  requireFinite(agentMean, 'agentMean');
  requirePositiveStd(std, 'std');
  requirePositiveStd(agentStd, 'agentStd');

  const meanMismatch = ((agentMean - mean) / std) ** 2;
  const varianceRatio = (agentStd / std) ** 2;
  const inflation = varianceRatio + meanMismatch;

  return {
    varianceRatio,
    meanMismatch,
    inflation,
    secondMomentChangePercent: (inflation - 1) * 100,
    remedyInflation: 1,
  };
}
