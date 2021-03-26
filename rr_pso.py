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


