import numpy as np
import rr_pso


def griewank(x):
    sqrti = np.sqrt(np.arange(1, x.shape[1]+1))
    g = 1 + 1 / 4000 * np.sum(x**2, axis=1) - np.prod(np.cos(x / sqrti), axis=1)
    return g


def rosenbrock(x):
    r = np.sum(100 * (np.roll(x, shift=-1, axis=1) - x**2)[:, :-1] ** 2
               + (1 - x)[:, :-1] ** 2, axis=1)
    return r


def training_rr_pso(criteria, nparticles=10, nparams=3):
    x0 = 10 * (0.5 - np.random.random((nparticles, nparams)))
    v0 = 0.6 * (0.5 - np.random.random((nparticles, nparams)))

    x1 = rr_pso.rr_pso(x0,
                       v0,
                       parameter_generator=rr_pso.parameter_generator_dt_constant,
                       criteria=criteria,
                       max_iter=1000)
    return x1


if __name__ == '__main__':
    x = training_rr_pso(criteria=lambda y: -rosenbrock(y))
    print(x)
