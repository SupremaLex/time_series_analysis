import numpy as np
from scipy import stats
from critical_values import estimate_gamma_alpha_min


def series_based_on_median(series):
    med = np.median(series)
    y = series[series != med]
    series_plus_minus = y > med
    return series_plus_minus


def series_ascending_and_descending(series):
    series_plus_minus = []
    for i in range(1, series.size):
        if series[i] == series[i-1]:
            continue
        series_plus_minus.append(series[i] > series[i-1])
    return series_plus_minus


def test_series_based_on_median(series):
    series_plus_minus = series_based_on_median(series)
    n = series.size
    k1 = round(0.5 * (n + 2 - 1.96 * np.sqrt(n - 1)))
    k2 = round(1.43 * np.log(n + 1))
    return test_series(series_plus_minus, k1, k2)


def test_series_ascending_and_descending(series):
    series_plus_minus = series_ascending_and_descending(series)
    n = series.size
    k1 = round((2 * n - 1) / 3 - 1.96 * np.sqrt((16 * n - 29) / 90))
    if n <= 26:
        k2 = 5
    elif n <= 153:
        k2 = 6.26
    elif n <= 1170:
        k2 = 7.153
    return test_series(series_plus_minus, k1, k2)


def test_series(series_plus_minus, k1, k2):
    p_m = " ".join([["-", "+"][e] for e in series_plus_minus])
    print(p_m)
    previous = series_plus_minus[0]
    n = len(series_plus_minus)
    series_list = []
    for i in range(n):
        if series_plus_minus[i] != previous:
            series_list.append(i)
            previous = series_plus_minus[i]
    series_list.append(n)
    m = len(series_list)
    t = np.max([series_list[i] - series_list[i-1] for i in range(1, m)])
    if m <= k1 or t >= k2:
        return True
    return False


def test_abbe(series, alpha):
    n = series.size
    quad_dif = np.sum(np.array([series[i] - series[i-1] for i in range(1, n)])**2) / (2 * (n - 1))
    quad_dif_avg = np.sum((series - series.mean())**2) / (n - 1)
    gamma = quad_dif / quad_dif_avg
    return gamma < estimate_gamma_alpha_min(alpha, n)


def test_average_levels(series, alpha, beta):
    n = series.size
    part = round(beta * n)
    left, right = series[:part], series[part:]
    n1, n2 = left.size, right.size
    var1, var2 = left.var(), right.var()
    tmp = np.abs(left.mean() - right.mean()) / np.sqrt((n1 - 1) * var1 + (n2 - 1) * var2)
    t = tmp * np.sqrt(n1 * n2 * (n1 + n2 - 2) / (n1 + n2))
    trend_level = t < stats.t.ppf(alpha, n - 2)
    if var1 > var2:
        trend_variance = (var1 / var2) > stats.f.ppf(1 - alpha, n1 - 1, n2 - 1)
    else:
        trend_variance = (var2 / var1) > stats.f.ppf(1 - alpha, n2 - 1, n1 - 1)
    return trend_level, trend_variance


def test_foster_stuart(series, alpha):
    def func(series, ind, cmp):
        return 1 if np.sum(cmp(series[ind], series[:ind])) == ind else 0
    n = series.size
    kt = np.array([func(series, i, np.greater) for i in range(1, n)])
    lt = np.array([func(series, i, np.less) for i in range(1, n)])
    s = np.sum(kt + lt)
    d = np.sum(kt - lt)
    #print(kt, "\n", lt)
    #print(s, alpha)
    mu = 2 * np.sum([1 / i for i in range(2, n)])
    t_s = np.abs(s - mu) / np.sqrt(2 * np.log(n) - 3.4253)
    t_alpha = d / np.sqrt(2 * np.log(n) - 0.8456)
    #print(t_s, t_alpha, stats.t.ppf(1 - alpha, n - 1))
    return t_s > stats.t.ppf(1 - alpha, n - 1), t_alpha > stats.t.ppf(1 - alpha, n - 1)


if __name__ == "__main__":
    arr = np.array([5, 8, 6, 7, 7, 10, 13, 9, 8, 6, 1,2, 4, 5, 10, 17, 9, 11, 8 , 20, 16])
    alpha = [0.1, 0.05, 0.01]
    msg = "Trend exists: %r"
    print("Series test based on median:")
    print(msg % test_series_based_on_median(arr))
    print("Test of ascending and descending series:")
    print(msg % test_series_ascending_and_descending(arr))
    msg1 = "Trend exists in average: %r\nTrend exists in variance: %r"
    for a in alpha:
        print("Test Abbe:")
        print(msg % test_abbe(arr, a))
        print("Test of average levels:")
        print(msg1 % test_average_levels(arr, a, 0.5))
        print("Foster - Stuart test:")
        print(msg1 % test_foster_stuart(arr, a))
