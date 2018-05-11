import numpy as np
import scipy.optimize as opt
from scipy import stats


def get_grubbs_criteria_critical_value(alpha, n, two_sided=False):
    nn = n
    if two_sided:
        nn *= 2
    t_alpha = stats.t.ppf(alpha / nn, n - 2)**2
    return np.sqrt(t_alpha / (n - 2 + t_alpha)) * (n - 1) / np.sqrt(n)


def monte_carlo(n, s, test, alpha, args=()):
    e_norm = np.zeros(n)
    for i in np.arange(n):
        norm = np.random.normal(size=s)
        e_norm[i] = test(norm, *args)
    return np.percentile(e_norm, alpha * 100)


def estimate_alpha(epsilon):
    k = estimate_k(epsilon)
    return (1 - epsilon) * stats.norm.cdf(-k) + epsilon / 2


def estimate_k(epsilon):
    if epsilon == 1:
        return 0

    def equation(k, e):
        return 2 * stats.norm.pdf(k) / k - 2 * stats.norm.cdf(-k) - e / (1 - e)

    return opt.root(equation, 1, tol=0.001, args=(epsilon,)).x[0]


def estimate_gamma_alpha_min(alpha, n):
    alpha_quantile = stats.norm.ppf(alpha)
    return 1 + alpha_quantile / np.sqrt((n + 0.5 * (1 + alpha_quantile**2)))


def range_moments(n, m):
    from scipy.integrate import quad, nquad, fixed_quad
    F = stats.norm.cdf
    f = lambda x: 1 - (1 - F(x))**n - F(x)**n
    mean = quad(f, -np.inf, np.inf)[0]
    if m == 1:
        return mean

    f = lambda x: quad(lambda t: (1 - F(t)**n - (1 - F(x))**n + (F(t) - F(x))**n)
                        * (t - x - mean)**(m-2), -1000, x)[0]

    return m * (m - 1) * quad(f, -np.inf, np.inf)[0] - (m - 1) * (-mean)**m


if __name__ == "__main__":
    alpha = [0.1, 0.05, 0.01]
    eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10,
           0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.65, 0.80, 1]
    grubbs = get_grubbs_criteria_critical_value
    gamma_min = estimate_gamma_alpha_min
    print("Grubbs criterion critical value:")
    for i in range(3, 100):
        print(i, grubbs(alpha[0], i), grubbs(alpha[1], i), grubbs(alpha[2], i))
    print("Value of k for Huber test:")
    print([estimate_k(e) for e in eps])
    print("Value of alpha for Pouncare test:")
    print([estimate_alpha(e) for e in eps])
    print("Gamma value for Abbe test:")
    for i in range(3, 100):
        print(i, gamma_min(alpha[0], i), gamma_min(alpha[1], i), gamma_min(alpha[2], i))
