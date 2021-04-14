import numpy as np
import matplotlib.pyplot as plt
import rr_pso


def griewank(x):
    # [-600, 600]^D
    sqrti = np.sqrt(np.arange(1, x.shape[1]+1))
    g = 1 + 1 / 4000 * np.sum(x**2, axis=1) - np.prod(np.cos(x / sqrti), axis=1)
    return g


def rosenbrock(x):
    # [-30, 30]^D
    r = np.sum(100 * (np.roll(x, shift=-1, axis=1) - x**2)[:, :-1] ** 2
               + (1 - x)[:, :-1] ** 2, axis=1)
    return r


def training_rr_pso(criteria, nparticles=40, nparams=30):
    # Griewank: dx = 1200
    # Rosenbrock: dx = 60
    dx = 1200
    x0 = dx * (0.5 - np.random.random((nparticles, nparams)))
    v0 = 0.6 * (0.5 - np.random.random((nparticles, nparams)))

    X, V = rr_pso.rr_pso(x0,
                         v0,
                         parameter_generator=rr_pso.constant_parameter_generator,
                         criteria=criteria,
                         memory=True,
                         max_iter=200)
    return X, V


def mean_error(x, x_opt=None):
    if x_opt is None:
        x_opt = np.zeros(x.shape[1])
    errors = np.std(x-x_opt, axis=1)
    me = np.min(errors)
    return me


if __name__ == '__main__':
    me_arr = []
    error_iter_arr = []
    for run in range(100):
        X, V = training_rr_pso(criteria=lambda y: -griewank(y))
        x = X[-1]
        me_arr.append(mean_error(x))

        error_iter_arr.append(np.array([mean_error(xi) for xi in X]))

    print(np.median(np.array(me_arr)))

    error_iter = np.mean(np.array(error_iter_arr), axis=0)

    plt.plot(np.log10(error_iter))
    plt.show()
