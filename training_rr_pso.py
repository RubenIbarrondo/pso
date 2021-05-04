import numpy as np
import matplotlib.pyplot as plt
import rr_pso

from mayavi import mlab
import moviepy.editor as mpy


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


def training_rr_pso(criteria, nparticles=40, nparams=30, xlim=np.array([-1, 1])):
    # Griewank: dx = 1200
    # Rosenbrock: dx = 60
    dx = 60
    if xlim.shape[0] == 1:
        x0 = np.min(xlim) + (np.max(xlim)-np.min(xlim)) * np.random.random((nparticles, nparams))
    else:
        x0 = np.min(xlim, axis=1) + (np.max(xlim, axis=1) - np.min(xlim, axis=1)) * np.random.random((nparticles, nparams))
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
    # TEST CON 30 PARÁMETROS

    # me_arr = []
    # error_iter_arr = []
    # for run in range(100):
    #     X, V = training_rr_pso(criteria=lambda y: -griewank(y))
    #     x = X[-1]
    #     me_arr.append(mean_error(x))
    #
    #     error_iter_arr.append(np.array([mean_error(xi) for xi in X]))
    #
    # print(np.median(np.array(me_arr)))
    #
    # error_iter = np.mean(np.array(error_iter_arr), axis=0)
    #
    # plt.plot(np.log10(error_iter))
    # plt.show()

    # TEST CON 2 PARÁMETROS

    me_arr = []
    error_iter_arr = []

    X, V = training_rr_pso(criteria=lambda y: -griewank(y), nparams=2, xlim=np.array([-600, 600]))
    N = X.shape[0]

    xp = X[:, :, 0]
    yp = X[:, :, 1]
    zp = np.array([griewank(X[g, :, :]) for g in range(N)])

    d = 500
    Xmesh, Ymesh = np.meshgrid(np.linspace(-d, d, 500), np.linspace(-d, d, 500))
    Xflat = Xmesh.flatten()
    Yflat = Ymesh.flatten()
    Zmesh = griewank(np.column_stack([Xflat, Yflat])).reshape(Xmesh.shape)

    mlab.mesh(Xmesh, Ymesh, Zmesh)
    sp = mlab.points3d(xp[N-1], yp[N-1], zp[N-1], np.ones(zp[0].shape), scale_factor=10)
    #mlab.show()

    @mlab.animate(delay=700, support_movie=True)
    def anim():
       for i in range(len(xp)):
           #s.mlab_source.scalars = np.sin((x*x+y*y+i*np.pi)/10)

           sp.mlab_source.y = yp[i]
           sp.mlab_source.x = xp[i]
           sp.mlab_source.z = zp[i]
           yield

    anim()
    mlab.show()
