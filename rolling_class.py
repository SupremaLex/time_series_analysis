import numpy as np
from scipy.stats import skew, kurtosis


class Rolling:
    def __init__(self, series, window):
        self.series = series
        self.window = window
        self.rolling = [series[i:i+window] for i in range(series.size - 1)]

    def apply(self, func):
        return np.array([func(e) for e in self.rolling])

    def mean(self):
        return self.apply(np.mean)

    def median(self):
        return self.apply(np.median)

    def std(self):
        return self.apply(np.std)

    def var(self):
        return self.apply(np.var)

    def kurt(self):
        return self.apply(kurtosis)

    def cov(self):
        return self.apply(np.cov)

    def skew(self):
        return self.apply(skew)

    def sum(self):
        return self.apply(np.sum)

    def corr(self):
        return self.apply(np.corrcoef)

    def exponential_rolling(self, alpha):
        smoothed = [self.series[1]]
        for i in range(len(self.series)):
            smoothed.append(smoothed[-1] + alpha * (self.series[i] - smoothed[-1]))
        return smoothed