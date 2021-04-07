import numpy as np


def constant_parameter_generator(nparticles):
    dt = np.full((nparticles, 1), 1)
    a1 = np.full((nparticles, 1), 0.6)
    a2 = np.full((nparticles, 1), 0.4)
    omega = np.full((nparticles, 1), 0.3)
    return dt, a1, a2, omega


def update_xv(x, v, dt, a1, a2, gbest, lbest, omega):
    """
    Updates x and v for RR-PSO.

    :param x: vector de partículas (numero_de_partículas x numero_de_parametros)
    :param v: vector de velocidades (numero_de_partículas x numero_de_parametros)
    :param dt: vector con los saltos de tiempo, (numero_de_partículas)
    :param a1: vector con las aceleraciones al global best, (numero_de_partículas)
    :param a2: vector con las aceleraciones al local best, (numero_de_partículas)
    :param gbest: global best particle, (numero_de_parametros)
    :param lbest: local best, (numero_de_partículas x numero_de_parametros)
    :param omega: vector con las inercias, (numero_de_partículas)
    :return:
    """

    nparticles, nparameters = x.shape
    r1 = np.random.random((nparticles, 1))
    r2 = np.random.random((nparticles, 1))
    phi1 = a1 * r1
    phi2 = a2 * r2

    acglobal = dt * phi1 * (gbest - x)
    aclocal = dt * phi2 * (lbest - x)
    phi = phi1 + phi2

    v_new = (v + acglobal + aclocal) / (1 + (1 - omega) * dt + phi * dt ** 2)
    x_new = x + v_new * dt
    return x_new, v_new


def rr_pso(x0, v0, parameter_generator, criteria, memory=False, max_iter=int(1e6)):
    """

    :param x0: vector de valores inciales, (numero_de_partículas x numero_de_parametros)
    :param v0: vector de velocidades iniciales, (numero_de_partículas x numero_de_parametros)
    :param parameter_generator: función que genera valores para cada partícula de los siguientes parámetros:
            * dt: vector con los saltos de tiempo, (numero_de_partículas)
            * a1: vector con las aceleraciones al global best, (numero_de_partículas)
            * a2: vector con las aceleraciones al local best, (numero_de_partículas)
            * omega: vector con las inercias, (numero_de_partículas)
    :param criteria: función objetivo, function
        criteria(x) -> fx : vector de valores de la función objetivo para cada partícula (numero_de_partículas)
    :param memory: si quieres guardar memoria, boolean
    :param max_iter: número máximo de interaciones, int
    :return:
    """

    i = 0
    x = x0
    v = v0
    lbest = x0

    while i < max_iter:
        clbest = criteria(lbest)
        options = (criteria(x) >= clbest).reshape((x.shape[0], 1))
        lbest = options * x + np.logical_not(options) * lbest
        gbest = lbest[np.where(np.max(clbest) == clbest)][0]

        dt, a1, a2, omega = parameter_generator(x.shape[0])

        x, v = update_xv(x, v, dt, a1, a2, gbest, lbest, omega)

        i += 1

    return x


def zero_criteria(x):
    return -np.max(abs(x), axis=1)


if __name__ == '__main__':
    npart = 10
    nparams = 3
    x0 = np.random.random((npart, nparams))
    v0 = 0.1 * np.random.random((npart, nparams))

    x = rr_pso(x0,
               v0,
               parameter_generator=constant_parameter_generator,
               criteria=zero_criteria,
               max_iter=10)
    print("x0: \n", x0)
    print("\nx: \n", x)