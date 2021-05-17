import numpy as np


def constant_parameter_generator(nparticles):
    dt = np.full((nparticles, 1), 1)
    omega = np.full((nparticles, 1), 3.9)
    phibar = np.full((nparticles, 1), 6.97)

    alpha = 1.5 * np.random.random((nparticles, 1))
    a1 = 2 * phibar * alpha
    a2 = 2 * phibar * (1 - alpha)
    return dt, a1, a2, omega


def parameter_generator_dt_constant(nparticles):
    dt = np.full((nparticles, 1), 1)

    #omega = 2 + np.random.random((nparticles, 1))
    omega = 3.9 + np.random.normal(0.0, 1.0, (nparticles, 1))

    phibar = 3 * (omega - 3/2)

    alpha = 1.5 * np.random.random((nparticles, 1))
    a1 = 2 * phibar * alpha
    a2 = 2 * phibar * (1 - alpha)
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


def rr_pso(x0, v0, parameter_generator, criteria, memory=False, max_iter=500):
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
    if memory:
        X = np.zeros((max_iter+1, x.shape[0], x.shape[1]))
        X[0] = x0
        V = np.zeros((max_iter+1,  v.shape[0], v.shape[1]))
        V[0] = v0

    while i < max_iter:
        clbest = criteria(lbest)
        options = (criteria(x) >= clbest).reshape((x.shape[0], 1))
        lbest = options * x + np.logical_not(options) * lbest

        gbest = lbest[np.where(np.nanmax(clbest) == clbest)][0]

        dt, a1, a2, omega = parameter_generator(x.shape[0])

        x, v = update_xv(x, v, dt, a1, a2, gbest, lbest, omega)

        if memory:
            X[i+1] = x
            V[i+1] = v

        i += 1

    if not memory:
        return x
    else:
        return X, V


def zero_criteria(x):
    return -np.mean(abs(x), axis=1)


if __name__ == '__main__':
    npart = 10
    nparams = 3
    x0 = 10 * (0.5 - np.random.random((npart, nparams)))
    v0 = 0.6 * (0.5 - np.random.random((npart, nparams)))

    x1 = rr_pso(x0,
                v0,
                parameter_generator=parameter_generator_dt_constant,
                criteria=zero_criteria,
                max_iter=1e5)

    print("av fit x0: ", np.mean(zero_criteria(x0)))
    print("x0: \n", x0)
    print("\nav fit x1: ", np.mean(zero_criteria(x1)))
    print("x: \n", x1)
