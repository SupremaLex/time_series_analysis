import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def calculate_all(series, period, trend_func):
    season_component_estimation = calculate_season_component_estimation(series, period)
    season_component = calculate_season_component(season_component_estimation, period)
    level_with_error = calculate_level_with_error(series, season_component, period)
    level, level_plus_season, trend = calculate_level(level_with_error, season_component,
                                                      period, trend_func)
    error = series - level_plus_season
    print(error)
    table = [[series[i],season_component[i % period],
              level[i], error[i], error[i]**2] for i in range(len(series))]
    print('{:<4} {:^8} {:^8} {:^8} {:^8} {:^8}'.format('t', 'y', 's', 't', 'e', 'e*e'))
    for i, (y, s, t, e, ee) in enumerate(table, 1):
        print('{:<4d} {:^8.2f} {:^8.2f} {:^8.2f} {:^8.2f} {:^8.4f}'.format(i, y, s, t, e, ee))
    return level, trend, np.sum(np.array(error) ** 2)


def calculate_season_component_estimation(series, window):
    rolling_mean = pd.Series(series).rolling(window=window, center=True).mean()
    #centered_rolling_mean = pd.Series(rolling_mean).rolling(window=2).mean()[1:]
    #centered_rolling_mean = centered_rolling_mean.append(pd.Series(np.nan), ignore_index=True)
    season_component_estimation = series - rolling_mean
    return season_component_estimation


def calculate_season_component(estimation, n):
    n_periodic_season_component = [np.mean(estimation[i::n]) for i in range(n)]
    corrective_coefficient = sum(n_periodic_season_component) / n
    adjusted_season_component = n_periodic_season_component - corrective_coefficient
    return adjusted_season_component


def calculate_level_with_error(series, season_component, n):
    level_with_error = [series[i] - season_component[i % n] for i in range(len(series))]
    return level_with_error


def calculate_level(level_with_error, season_component, n, trend_func):
    k = len(level_with_error)
    time = range(1, k + 1)
    popt, pcov = scipy.optimize.curve_fit(trend_func, time, level_with_error)
    level = np.array([trend_func(t, *popt) for t in time])
    level_plus_season_component = np.array([level[i] + season_component[i % n] for i in range(k)])
    trend = lambda t: trend_func(t, *popt)
    return level, level_plus_season_component, trend
