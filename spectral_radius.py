import numpy as np
import matplotlib.pyplot as plt

def spectral_radius(oemga, phi):
    lambdam = abs((oemga - 3 + np.sqrt(oemga**2 - 2*oemga - 4*phi + 1, dtype=complex))/(2*(-oemga + phi + 2)))
    lambdap = abs((oemga - 3 - np.sqrt(oemga**2 - 2*oemga - 4*phi + 1, dtype=complex))/(2*(-oemga + phi + 2)))
    return np.max([lambdam, lambdap], axis=0)

def spectral_radius_2(omega,phi):
    n = np.shape(omega)
    sp = np.zeros((n[0],n[1]))
    for i in range(n[0]):
        for j in range(n[1]):
            a3 = -1
            a2 = ((3-omega[i,j])**2-(2-omega[i,j]+phi[i,j]))/(2-omega[i,j]+phi[i,j])**2
            a1 = -(3-omega[i,j])**2/(2-omega[i,j]+phi[i,j])**3
            a0 = 1/(2-omega[i,j]+phi[i,j])**3
            sp[i,j] = np.max(np.abs(np.roots(np.array([a3,a2,a1,a0]))), axis=0)
    return sp


ngridx, ngridy = 550, 550
xmin, xmax = [-10, 10]
ymin, ymax = [-10, 10]

x = np.linspace(xmin, xmax, ngridx)
y = np.linspace(ymin, ymax, ngridy)
X,Y = np.meshgrid(x, y)
Z = spectral_radius(X, Y)
Z2 = spectral_radius_2(X,Y)

im = plt.imshow(Z,cmap=plt.cm.RdBu,
                vmin = 0,
                vmax = 1,
                extent=[xmin,xmax,ymin,ymax],
                origin='lower')

# adding the Contour lines with labels
#cset = plt.contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=plt.cm.Set2)
#plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)

plt.colorbar(im)

plt.xlabel('$\omega$')
plt.ylabel('$\overline{\phi}$')
plt.title('Spectral radius')
plt.savefig("spectralradius1", dpi=300)
plt.show()

im2 = plt.imshow(Z2,cmap=plt.cm.RdBu,
                vmin = 0,
                vmax = 1,
                extent=[xmin,xmax,ymin,ymax],
                origin='lower')

# adding the Contour lines with labels
#cset = plt.contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=plt.cm.Set2)
#plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)

plt.colorbar(im2)

plt.xlabel('$\omega$')
plt.ylabel('$\overline{\phi}$')
plt.title('Spectral radius')
plt.savefig("spectralradius2", dpi=300)
plt.show()


    