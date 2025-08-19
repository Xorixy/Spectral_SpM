import scipy as scp
import numpy as np
import matplotlib.pyplot as plt


def freq_kernel(n_real, omega_max, n_imag_max, beta):
    kernel = np.zeros([2*(n_imag_max + 1), n_real], dtype=complex)
    omegas = np.linspace(-omega_max, omega_max, n_real)
    iomegas = np.pi*(2*np.linspace(-n_imag_max - 1, n_imag_max, 2*(n_imag_max + 1)) + 1)/beta
    for i in range(2*(n_imag_max + 1)):
        for j in range(n_real):
            kernel[i, j] = 1/(1j*iomegas[i] - omegas[j])
    return kernel


def geo(n, k):
    return (n**k)/(n+1)**(k+1)

omega_max = 1000
n_real = 101
n_imag_max = 420
beta = 1
omegas = np.linspace(-omega_max, omega_max, n_real)


kernel = freq_kernel(n_real, omega_max, n_imag_max, beta)

U, S, Vh = scp.linalg.svd(kernel)

ns = np.linspace(-n_imag_max - 1, n_imag_max, 2*(n_imag_max + 1))
n_imag = 2*(n_imag_max + 1)
s = 0
k = 1
c = -1
b = -3.8
#plt.plot(ns[n_imag_max + 1:], np.imag(U[n_imag_max + 1:, s]))
#plt.plot(ns[n_imag_max + 1:], c*np.real(U[n_imag_max + 1:, s+1]))
#plt.plot(ns[n_imag_max + 1:], c*geo(ns[n_imag_max + 1:], k))
#plt.plot(ns[n_imag_max + 1:], np.imag(U[n_imag_max + 1:, s]))
#plt.plot(ns[n_imag_max + 1:], c*geo(ns[n_imag_max + 1:], 0) + c*b*geo(ns[n_imag_max + 1:], 1))
#plt.plot(ns[n_imag_max + 1:], np.real(U[n_imag_max + 1:, s+1]))
#plt.plot(ns[n_imag_max + 1:], np.imag(U[n_imag_max + 1:, s]))
#plt.plot(ns[n_imag_max + 1:], (0.65/(2*ns[n_imag_max + 1:] + 1))**(1))
#plt.plot(ns[n_imag_max + 1:], (30/(2*ns[n_imag_max + 1:] + 1))**(2))
#plt.yscale('log')
plt.plot(omegas, np.real(Vh[0,:]))
print(np.sum(Vh[2,:]*Vh[0,:]))
print(S[s])

plt.show()

