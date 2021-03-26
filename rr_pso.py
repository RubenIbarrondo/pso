import numpy as np


def update_xv(x, v, dt, a1, a2, gbest, lbest, omega):
    """
    Updates x and v for RR-PSO.

    :param x: vector de partículas (numero_de_partículas x numero_de_parametros)
    :param v: vector de velocidades (numero_de_partículas x numero_de_parametros)
    :param dt: salto de tiempo, float
    :param a1: aceleración al grobal best, float
    :param a2: aceleración al local best, float
    :param gbest: global best particle, (numero_de_parametros)
    :param lbest: local best, (numero_de_partículas x numero_de_parametros)
    :param omega: inercia, float
    :return:
    """
    nparticles, nparameters = x.shape
    phi1 = a1 * np.array([np.random.random(nparticles)] * nparameters).transpose()
    phi2 = a2 * np.array([np.random.random(nparticles)] * nparameters).transpose()

    acglobal = phi1 * dt * (gbest - x)
    aclocal = phi2 * dt * (lbest - x)
    phi = phi1 + phi2

    v_new = (v + acglobal + aclocal) / (1 + (1 - omega) * dt + phi * dt ** 2)
    x_new = x + v_new * dt
    return x_new, v_new


def rr_pso(x0, v0, dt, a1, a2, omega, criteria, memory=False, max_iter=int(1e6)):
    """

    :param x0: vector de valores inciales, (numero_de_partículas x numero_de_parametros)
    :param v0: vector de velocidades iniciales, (numero_de_partículas x numero_de_parametros)
    :param dt: salto de tiempo, float
    :param a1: aceleración al grobal best, float
    :param a2: aceleración al local best, float :param dt:
    :param omega: inercia, float
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
        options = criteria(x) > criteria(lbest)
        lbest = options * x + (1 - options) * lbest
        gbest = lbest[np.where(np.max(criteria(lbest)) == criteria(lbest))][0]

        x, v = update_xv(x, v, dt, a1, a2, gbest, lbest, omega)

        i += 1
