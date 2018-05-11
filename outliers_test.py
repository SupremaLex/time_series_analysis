import numpy as np
import critical_values


def test_grubbs(series, alpha):
    n = series.size
    sorted_series = np.sort(series)
    avg = series.mean()
    ind_max = np.argmax(series)
    ind_min = np.argmin(series)
    sigma = np.sqrt((np.sum((series - avg)**2) / (n - 1)))
    t_n = (sorted_series[-1] - avg) / sigma
    t_1 = (avg - sorted_series[0]) / sigma
    g = critical_values.get_grubbs_criteria_critical_value(alpha, n)
    result = series
    if t_n > g:
        print("Remove highest outlier")
        result = np.delete(series, ind_max)
    if t_1 > g:
        print("Remove lowest outlier")
        result = np.delete(series, ind_min)
    return result


def __test_titjen_moore(series, k, m):
        sorted_series = np.sort(series)
        avg = series.mean()
        series_without_outliers = sorted_series[k:-m]
        avg_without = np.average(series_without_outliers)
        k = np.sum((series_without_outliers - avg_without) ** 2) / np.sum(((sorted_series - avg) ** 2))
        return k


def test_tietjen_moore(series, k, m, alpha):
    if m == 0:
        m = 1
    K = __test_titjen_moore(series, k, m)
    CV = critical_values.monte_carlo(10000, series.size,
                                     __test_titjen_moore, alpha, (k, m))
    result = series
    print(K, CV)
    if K < CV:
        print("Remove %i lowest and %i highest outliers" % (k, m))
        ind_less = np.argpartition(series - np.average(series), k)[:k]
        ind_more = np.argpartition(series - np.average(series), -m)[-m:]
        result = np.delete(result, np.concatenate([ind_less, ind_more]))
    return result


def __test_outliers_simultaneously(series, k):
    avg = series.mean()
    z = np.array([np.abs(y - avg) for y in series])
    return __test_titjen_moore(z, 0, k)


def test_outliers_simultaneously(series, k, alpha):
    K = __test_outliers_simultaneously(series, k)
    CV = critical_values.monte_carlo(10000, series.size,
                                     __test_outliers_simultaneously, alpha, (k, ))
    result = series
    print(K, CV)
    if K < CV:
        print("Remove %i highest outliers" % k)
        ind_more = np.argpartition(series - np.average(series), -k)[-k:]
        result = np.delete(result, ind_more)
    return result


def robust_test_poancare(series, k):
    n = int(series.size * critical_values.estimate_alpha(k / series.size))
    if not n:
        return series
    arr = np.array(list(zip(range(len(series)), series)))
    sorted_series = arr[arr[:, 1].argsort()][n:-n]
    print("Remove %i lowest and highest outliers" % n)
    return sorted_series[sorted_series[:, 0].argsort()][:, 1]


def robust_test_winzor(series, alpha):
    n = int(series.size * alpha)
    if not n:
        return series
    arr = np.array(list(zip(range(series.size), series)))
    sorted_series = arr[arr[:, 1].argsort()]
    left = sorted_series[:n]
    right = sorted_series[-n:]
    l, h = sorted_series[n][1], sorted_series[-n-1][1]
    left[:, 1] = [l] * n
    right[:, 1] = [h] * n
    winsorized = np.concatenate([left, sorted_series[n:-n], right])
    msg = "Change %i %s outliers with %0.4f"
    print(msg % (n, "lowest", l))
    print(msg % (n, "highest", h))
    return winsorized[winsorized[:, 0].argsort()][:, 1]


def robust_test_huber(series, outliers_number):
    n = series.size
    k = critical_values.estimate_k(outliers_number / n)
    series = np.sort(series)
    avg_p = series.mean()
    group_2 = np.array([])
    while len(group_2) != n:
        group_1 = series[series < avg_p - k] + k
        group_2 = series[np.abs(series - avg_p) < k]
        group_3 = series[series > avg_p + k] - k
        series = np.concatenate([group_1, group_2, group_3])
        new_group_2 = np.concatenate([group_1[np.abs(group_1 - avg_p) < k],
                                      group_2,
                                      group_3[np.abs(group_3 - avg_p) < k]])
        n1 = np.sum(group_1 < avg_p - k)
        n2 = np.sum(group_3 > avg_p + k)
        tmp = (n2 - n1) * k + (n2 + n1) * avg_p
        avg_p = (np.sum(new_group_2) + tmp) / n
        #print("Group 1:", group_1)
        #print("Group 2:", group_2)
        #print("Group 3:", group_3)
        #print("%i levels was modificated on this iteration" % group_1.size + group_3.size)
        #print("New avg", avg_p)
    return series


if __name__ == "__main__":
    alpha = [0.1, 0.05, 0.01]
    test_array = np.array([0, 0.5, 0.6, 2, 3, 4, 5, 6, 7, 8, 5, 6, 4, 3, 9, 10, 20, 22, 25])
    print("Not robust tests:\n")
    for a in alpha:
        print("Grubbs test")
        print(test_grubbs(test_array, a))
        print("Tietjen-Moore E-test")
        print(test_tietjen_moore(test_array, 3, 3, a))
        print("Simultaneously estimating of outliers test:")
        print(test_outliers_simultaneously(test_array, 3, a))

    print("Robust tests:\n")
    print("Poancare method:")
    for i in range(1, 5):
        print(robust_test_poancare(test_array, i))
    print("Winsor method:")
    for a in alpha:
        print(robust_test_winzor(test_array, 0.1))
    print("Huber method:")
    for i in range(1, 5):
        print(robust_test_huber(test_array, i))