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
    omegas, spectral = GetSpectral("../results/xtest_2.h5")
    ax_s.plot(omegas, spectral, label="Double")
    omegas, spectral = GetSpectral("../results/xtest_3.h5")
    ax_s.plot(omegas, spectral, label="MPReal")
    ax_s.legend()

    fig_g, ax_g = plt.subplots()
    taus, green, green_rc = GetGreen("../results/xtest_2.h5")
    ax_g.plot(taus, green_rc/green - 1, label="Double")
    taus, green, green_rc = GetGreen("../results/xtest_3.h5")
    ax_g.plot(taus, green_rc/green - 1, label="MPReal")
    ax_g.legend()
def Compare():
    fig_s, ax_s = plt.subplots()
    omegas, spectral = GetSpectral("../results/f12h0_spec_norec.h5")
    ax_s.plot(omegas, spectral, label="β = 1.2")
    omegas, spectral = GetSpectral("../results/f14h0_spec_norec.h5")
    ax_s.plot(omegas, spectral, label="β = 1.423")
    omegas, spectral = GetSpectral("../results/f16h0_spec_norec.h5")
    ax_s.plot(omegas, spectral, label="β = 1.635")
    omegas, spectral = GetSpectral("../results/f17h0_spec_norec.h5")
    ax_s.plot(omegas, spectral, label="β = 1.75")
    omegas, spectral = GetSpectral("../results/f18h0_spec_norec.h5")
    ax_s.plot(omegas, spectral, label="β = 1.85")
    ax_s.legend()

    fig_g, ax_g = plt.subplots()
    taus, green, green_rc = GetGreen("../results/f12h0_spec_norec.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.2")
    taus, green, green_rc = GetGreen("../results/f14h0_spec_norec.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.423")
    taus, green, green_rc = GetGreen("../results/f16h0_spec_norec.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.635")
    taus, green, green_rc = GetGreen("../results/f17h0_spec_norec.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.75")
    taus, green, green_rc = GetGreen("../results/f18h0_spec_norec.h5")
    ax_g.plot(green_rc/green - 1, label="β = 1.85")
    ax_g.legend()


PlotSpectralRun("../results/f12h3_spec_500.h5")
#Compare2()
plt.show()



