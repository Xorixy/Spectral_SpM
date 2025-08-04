import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

def QuadraticSpectral(w0, w1, omegas, norm = 1.0):
    spectral = (w1**2 - (w0 - omegas)**2)*0.75/w1**3
    spectral += np.abs(spectral)
    spectral /= 2
    return spectral*norm

def GaussianSpectral(w0, sigma, omegas, norm = 1.0):
    return np.exp(-(omegas-w0)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)*norm

def LorentzianSpectral(w0, L, omegas, norm = 1.0):
    return (L/np.pi)/((omegas - w0)**2 + L**2)*norm
def SpectralGreen(spectral, omegas, taus, beta, noise = 0.0):
    dE = omegas[1]-omegas[0]
    green = np.zeros(len(taus))
    for i in range(len(taus)):
        green[i] = sum(-dE*spectral*np.exp(-taus[i]*omegas)/(1 + np.exp(-beta*omegas)))
    green += np.random.normal(0.0, noise, len(green))
    return green

def SaveGreen(filename, green, beta):
    f = h5.File(filename, "w")
    f["green"] = green
    f["beta"] = beta

def SaveSpectral(filename, spectral, omegas, beta):
    f = h5.File(filename, "w")
    f["spectral"] = spectral
    f["omegas"] = omegas
    f["beta"] = beta


filename = "../data/gaussian.h5"
omega_min = -10
omega_max = 10
l = 0.5
w0 = 1.3
w1 = 1.1
n_omega = 4000
beta = 2.3
n_tau = 2000

omegas = np.linspace(omega_min, omega_max, n_omega)
taus = np.linspace(0, beta, n_tau)

spectral = GaussianSpectral(-4, 0.8, omegas, 0.9) + GaussianSpectral(4, 0.8, omegas, 0.1)
SaveSpectral("../data/spectral.h5", spectral, omegas, beta)
green = SpectralGreen(spectral, omegas, taus, beta, 0)

plt.plot(omegas, spectral)
plt.figure()
plt.plot(taus, green)
#plt.plot(taus, np.exp(-taus*w0)/(1 + np.exp(-beta*w0)))
plt.show()

SaveGreen(filename, green, beta)