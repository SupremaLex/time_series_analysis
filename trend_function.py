import numpy as np
import matplotlib.pyplot as plt
from rolling_class import Rolling


polynom_1 = lambda t, a, b: a*t + b
polynom_2 = lambda t, a, b, c: a*np.power(t, 2) + b*t + c
polynom_3 = lambda t, a, b, c, d: a*np.power(t, 3) + b*np.power(t, 2) + c*t + d
exponent = lambda t, a, b: a*np.power(b, t)
mod_exponent = lambda t, a, b, c: a*np.power(b, t) + c
gomperc_curve = lambda t, k, a, b: k * a**np.power(b, t)
logistic_curve = lambda t, k, a:  k / (1 + a * np.exp(np.negative(t)))
log_parabola = lambda t, a, b, c: a * np.power(b, t) * c**np.power(t, 2)

functions = {"polynom_1": polynom_1, "polynom_2": polynom_2, "polynom_3": polynom_3, "exponent": exponent,
             "mod_exponent": mod_exponent, "gomperc_curve": gomperc_curve, "logistic_curve": logistic_curve,
             "log_parabola": log_parabola}


def estimate_trend_function(series, window, rolling_func, args=()):
    smoothed_series = np.array(rolling_func(Rolling(series, window), *args))
    first_mean_gain = [(smoothed_series[i+1] - smoothed_series[i-1]) / 2
                       for i in range(1, len(smoothed_series) - 1)]
    second_mean_gain = [(first_mean_gain[i+1] - first_mean_gain[i-1]) / 2
                        for i in range(1, len(first_mean_gain) - 1)]
    first_mean_gain = np.array(first_mean_gain)
    second_mean_gain = np.array(second_mean_gain)
    u1t_yt = first_mean_gain / smoothed_series[1:-1]
    log_u1t = np.log(np.abs(first_mean_gain))
    log_u1t_yt = np.log(u1t_yt)
    plt.figure("First mean gain")
    plt.plot(first_mean_gain)
    plt.plot()
    plt.figure("Second mean gain")
    plt.plot(second_mean_gain)
    plt.figure("u1 / y1")
    plt.plot(u1t_yt)
    plt.figure("log(u1)")
    plt.plot(log_u1t)
    plt.figure("log (u1 / y1)")
    plt.plot(log_u1t_yt)
    plt.show()


if __name__ == "__main__":
    milk_data_infagro = [2472, 2479, 2396, 2521, 2486, 2528, 2917, 3069, 3333, 3405, 3283, 3146,
                         3083, 2979, 2667, 2479, 2375, 2375, 2396, 2403, 2542, 2646, 2667, 2875,
                         2979, 3042, 3417, 3250, 3250, 3354, 3333, 3458, 3521, 3625, 3833, 4063,
                         4097, 4021, 3875, 3448, 3130, 3125, 3333, 3464, 3500, 3615, 3750, 3870,
                         3896, 4021, 4479, 4417, 4177, 4031, 4021, 4063, 4193, 4417, 4854, 5167,
                         5281, 5333, 5229, 5094, 4833, 4604, 4604, 4885, 5479, 6167, 6875, 7458,
                         7479, 7417]
    arr = np.array(milk_data_infagro)
    estimate_trend_function(arr, 4, Rolling.median)
