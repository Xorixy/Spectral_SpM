import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from math import factorial
def LorentzianSpectral(w0, L, omegas, norm = 1.0):
    return (L/np.pi)/((omegas - w0)**2 + L**2)*norm

def vec_sign(v):
    vsum = np.sum(v)
    if vsum < 0:
        return -1
    else:
        return 1

def choose(n, k):
    return factorial(n)/(factorial(n-k)*factorial(k))
def MTR(x, n, L):
    num = 0
    for k in range(n+1):
        num += choose(2*n+1, 2*k)*x**(2*n + 1 - 2*k)*L**(2*k)*(-1)**k
    return np.sqrt(L/np.pi)*num/(x**2 + L**2)**(n+1)

def MTI(x, n, L):
    num = 0
    for k in range(n+1):
        num += choose(2*n+1, 2*k+1)*x**(2*n - 2*k)*L**(2*k+1)*(-1)**k
    return np.sqrt(L/np.pi)*num/(x**2 + L**2)**(n+1)

def MTR2(x, n, L):
    return np.real(np.sqrt(L/(2*np.pi))*1j*((x - 1j*L)**(2*n+1) - (x + 1j*L)**(2*n+1))/(x**2 + L**2)**(n+1))

def MTI2(x, n, L):
    return np.real(np.sqrt(L/(2*np.pi))*((x - 1j*L)**(2*n+1) + (x + 1j*L)**(2*n+1))/(x**2 + L**2)**(n+1))

def MTRT(x, n, L, a):
    return MTR2(np.arctanh(x/a), n, L)*np.sqrt(a/(a**2 - x**2))

def MTIT(x, n, L, a):
    return MTR2(np.arctanh(x/a), n, L)*np.sqrt(a/(a**2 - x**2))
def plot_svs(filename, nmax, ax_t, ax_o, c = 1.0, nmin = 0):
    f = h5.File(filename, "r")

    beta = f["beta"][()]
    SVs = f["SVs"][()]
    V = f["V"][()]
    U = f["U"][()]

    taus = np.linspace(0, beta, U.shape[0])
    omegas = f["omegas"][()]

    for i in range(nmin, nmax+1):
        ax_t.plot(taus, c*U[:, i], label = f"S({i}) = {SVs[i]}")
        ax_o.plot(omegas, c*V[:, i], label = f"S({i}) = {SVs[i]}")


def test():
    filename = "../results/svd.h5"
    f = h5.File(filename, "r")

    beta = f["beta"][()]
    SVs = f["SVs"][()]
    V = f["V"][()]
    U = f["U"][()]

    taus = np.linspace(0, beta, U.shape[0])
    omegas = f["omegas"][()]

    plt.plot(omegas, V[:,0])
    L = 2
    c = 1
    nu = 0.45
    s = np.max(V[:, 0])*L**(2*nu)
    b = 0.00
    a = omegas[-1]
    plt.plot(omegas, s/(L**2 + omegas**2)**nu)
    #plt.plot(omegas, s*MTI2(a*omegas/(a**2 - omegas**2), 0, L)*np.sqrt(a**2 + omegas**2)/(a**2 - omegas**2))
    #plt.plot(omegas[1:-1], s*MTRT(omegas[1:-1], n, L, omegas[-1]))

#fig_t, ax_t = plt.subplots()
#fig_o, ax_o = plt.subplots()
n = 0
c = 1
nmin = n
nmax = n

test()
#filename = "../results/svd.h5"
#plot_svs(filename, nmax, ax_t, ax_o, nmin = nmin, c=1.3)
#filename = "../results/svd_long.h5"
#plot_svs(filename, nmax, ax_t, ax_o, nmin=nmin, c=0.5)
#filename = "../results/svd_long_long.h5"
#plot_svs(filename, nmax, ax_t, ax_o, nmin=nmin, c=0.7)

#ax_t.set_title("Columns of U (time)")
#ax_o.set_title("Columns of V (freq)")

#ax_t.legend()
#ax_o.legend()


plt.show()

