import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


def PlotSpectralRun(filename):
    f = h5.File(filename, 'r')
    print(f.keys())
    omegas = f["omegas"][()]
    spectral = f["spectral"][()]
    taus = f["taus"][()]
    green = f["green"][()]
    green_rc = f["green_rc"][()]
    beta = f["beta"][()]
    #green_rc = SpectralGreen(spectral, omegas, taus, beta)

    plt.plot(omegas, spectral, label="RC")
    plt.legend()
    plt.figure()
    plt.plot(omegas, (spectral - np.abs(spectral))/2)
    plt.figure()
    plt.plot(taus, green, label="Green sim")
    plt.plot(taus, green_rc, label="Green RC")
    plt.legend()
    plt.figure()
    plt.plot(taus, green_rc/green - 1)

    if "lambdas" in f and "errors" in f:
        fig_l, ax_l = plt.subplots()
        lambdas = f["lambdas"][()]
        errors = f["errors"][()]
        ax_l.plot(lambdas, errors)
        lin = (errors[-1] - errors[0])/(np.log(lambdas[-1]) - np.log(lambdas[0]))*(np.log(lambdas) - np.log(lambdas[0])) + errors[0]
        ax_l.plot(lambdas, lin)
        div = lin/errors
        ax_l.plot(lambdas, div*np.max(lin)/np.max(div))
        ax_l.set_xscale('log')


    plt.show()

def SpectralGreen(spectral, omegas, taus, beta):
    dE = omegas[1]-omegas[0]
    green = np.zeros(len(taus))
    for i in range(len(taus)):
        print(i)
        green[i] = sum(-dE*spectral*np.exp(-taus[i]*omegas)/(1 + np.exp(-beta*omegas)))

    return green

def GetSpectral(filename):
    f = h5.File(filename, 'r')
    omegas = f["omegas"][()]
    spectral = f["spectral"][()]

    return omegas, spectral

def GetGreen(filename):
    f = h5.File(filename, 'r')
    taus = f["taus"][()]
    green = f["green"][()]
    green_rc = f["green_rc"][()]

    return taus, green, green_rc

def Compare2():
    fig_s, ax_s = plt.subplots()
    omegas, spectral = GetSpectral("../results/f18h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="Direct")
    omegas, spectral = GetSpectral("../results/f18h0_spec_rec.h5")
    ax_s.plot(omegas, spectral, label="Recursive")
    ax_s.legend()

    fig_g, ax_g = plt.subplots()
    taus, green, green_rc = GetGreen("../results/f18h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="Direct")
    taus, green, green_rc = GetGreen("../results/f18h0_spec_rec.h5")
    ax_g.plot(green_rc/green - 1, label="Recursive")
    ax_g.legend()
def Compare():
    fig_s, ax_s = plt.subplots()
    omegas, spectral = GetSpectral("../results/f12h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="β = 1.2")
    omegas, spectral = GetSpectral("../results/f14h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="β = 1.423")
    omegas, spectral = GetSpectral("../results/f16h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="β = 1.635")
    omegas, spectral = GetSpectral("../results/f17h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="β = 1.75")
    omegas, spectral = GetSpectral("../results/f18h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="β = 1.85")
    ax_s.legend()

    fig_g, ax_g = plt.subplots()
    taus, green, green_rc = GetGreen("../results/f12h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.2")
    taus, green, green_rc = GetGreen("../results/f14h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.423")
    taus, green, green_rc = GetGreen("../results/f16h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.635")
    taus, green, green_rc = GetGreen("../results/f17h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.75")
    taus, green, green_rc = GetGreen("../results/f18h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.85")
    ax_g.legend()

def CompareQ():
    fig_s, ax_s = plt.subplots()
    omegas, spectral = GetSpectral("../results/fq00h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="q = 0.0")
    omegas, spectral = GetSpectral("../results/fq10h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="q = 1.0")
    omegas, spectral = GetSpectral("../results/f16h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="q = 2.5")
    omegas, spectral = GetSpectral("../results/fq50h0_spec_1000.h5")
    ax_s.plot(omegas, spectral, label="q = 5.0")
    ax_s.legend()

    fig_g, ax_g = plt.subplots()
    taus, green, green_rc = GetGreen("../results/fq00h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="q = 0.0")
    taus, green, green_rc = GetGreen("../results/fq10h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="q = 1.0")
    taus, green, green_rc = GetGreen("../results/f16h0_spec_1000.h5")
    ax_g.plot(green_rc/green - 1, label="q = 2.5")
    taus, green, green_rc = GetGreen("../results/fq50h0_spec_2000.h5")
    ax_g.plot(green_rc/green - 1, label="q = 5.0")
    ax_g.legend()

#PlotSpectralRun("../results/fq50h0_spec_2000.h5")
CompareQ()
Compare()
plt.show()



