import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rr_pso
import training_rr_pso


def verhulst(t, k=1, p0=1, r=1):
    k = np.max([k, 1e-3])
    p0 = np.max([p0, 1e-3])
    r = np.max([r, 0])

    return k * p0 / (p0 + (k - p0) * np.exp(- r * t))


def gompertz(t, n0=1, c=1, a=1):
    return n0 * np.exp(-c*(np.exp(a * t) - 1))


def covid_fitness(x, data, estimator, xlim, tol=None):
    t = np.arange(len(data))
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if tol is None:
            fitness[i] = - np.sqrt(np.sum((data - estimator(t, *x[i]))**2) / np.sum(data**2))
        else:
            fitness[i] = - np.max([np.sqrt(np.sum((data - estimator(t, *x[i]))**2) / np.sum(data**2)), tol])
        #fitness[i] += 1e6 * np.all(x[i] >= xlim[:, 0])
        #fitness[i] += 1e6 * np.all(x[i] <= xlim[:, 1])
    return fitness


def fit_covid_wave(data, max_iter=100, test_number=10, tol=.15):
    data = data - np.min(data)

    estimator = verhulst
    t = np.arange(len(data))

    xlim = np.array([[0, np.max(data) * 1.25],  # k
                     [1, 1000],  # p0
                     [0, 10]  # r
                     ])
    best_fit = -1e17

    for test in range(test_number):
        X, V = training_rr_pso.training_rr_pso(criteria=lambda x: covid_fitness(x, data, estimator, xlim, tol),
                                               nparams=3, xlim=xlim, max_iter=max_iter,
                                               parameter_gemerator=rr_pso.constant_parameter_generator)

        fval = np.array([covid_fitness(xi, data, estimator, xlim) for xi in X])

        if np.max(fval) > best_fit:
            Xbest = X
            x = X[-1]

            fit_arr = covid_fitness(x, data, estimator, xlim)
            x0 = x[np.where(fit_arr == np.nanmax(fit_arr))][0]
    return fval, x, x0, Xbest


if __name__ == '__main__':
    data = np.array(pd.read_csv('covid_aragon_ola_2')['PCR'])
    data = data - np.min(data)

    estimator = verhulst
    t = np.arange(len(data))

    xlim = np.array([[0, np.max(data)*1.25], [1, 10], [0, 100]])

    X, V = training_rr_pso.training_rr_pso(criteria=lambda x: covid_fitness(x, data, estimator, xlim),
                                           nparams=3,
                                           xlim=xlim)
    x = X[-1]

    fit_arr = covid_fitness(x, data, estimator, xlim)
    print(fit_arr)
    x0 = x[np.where(fit_arr == np.nanmax(fit_arr))][0]

    plt.plot(data)
    plt.plot(verhulst(t, *x0))
    plt.show()

    plt.plot(np.diff(data))
    plt.plot(np.diff(verhulst(t, *x0)))
    plt.show()

    plt.hist(x[:, 0])
    plt.title('k')
    plt.show()

    plt.hist(x[:, 1])
    plt.title('p0')
    plt.show()

    plt.hist(x[:, 2])
    plt.title('r')
    plt.show()
