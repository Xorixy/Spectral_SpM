import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


def ReconstructGreen(spectral, omegas, domegas, taus):
    green = np.zeros(len(taus))
    beta = taus[-1]
    for i in range(len(taus)):
        green[i] = -np.sum(domegas*spectral*np.exp(-taus[i]*omegas)/(1 + np.exp(-beta*omegas)))
    return green

def PlotSpectralRun(filename):
    f = h5.File(filename, 'r')
    print(f.keys())
    omegas = f["omegas"][()]
    domegas = f["domegas"][()]
    spectral = f["spectral"][()]
    taus = f["taus"][()]
    green = f["green"][()]
    green_rc = f["green_rc"][()]
    beta = f["beta"][()]
    #green_rc = SpectralGreen(spectral, omegas, taus, beta)
    print(len(taus))

    plt.plot(omegas, spectral, label="RC")
    plt.plot(omegas, np.flip(spectral), label="Rev")
    plt.legend()
    plt.figure()
    plt.plot(omegas, (spectral - np.abs(spectral))/2)
    plt.figure()
    plt.plot(taus, green, label="Green sim")
    plt.plot(taus, green_rc, label="Green RC")
    green_p = ReconstructGreen(spectral, omegas, domegas, taus)
    plt.plot(taus, green_p, '--', label="Green Python")
    plt.legend()
    plt.figure()
    plt.plot(taus, green_p/green - 1)

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

    print(np.sum(domegas*spectral))
    plt.show()


def analyze(filename):
    f = h5.File(filename, 'r')
    print(f.keys())
    omegas = f["omegas"][()]
    domegas = f["domegas"][()]
    spectral = f["spectral"][()]
    taus = f["taus"][()]
    green = f["green"][()]
    green_rc = f["green_rc"][()]
    beta = f["beta"][()]
    #green_rc = SpectralGreen(spectral, omegas, taus, beta)

    wqs = 10.8
    wqs = 15
    wqe = 40
    wps = 1
    wpe = 11
    wpe = 15

    iq_neg = np.where(np.logical_and(-wqe < omegas, omegas < -wqs))
    iq_pos = np.where(np.logical_and(wqs < omegas, omegas < wqe))
    ip_neg = np.where(np.logical_and(-wpe < omegas, omegas < -wps))
    ip_pos = np.where(np.logical_and(wps < omegas, omegas < wpe))

    plt.plot(omegas, spectral)
    plt.plot(omegas[iq_neg], spectral[iq_neg], label="Q neg")
    plt.plot(omegas[iq_pos], spectral[iq_pos], label="Q pos")
    plt.plot(omegas[ip_neg], spectral[ip_neg], label="P neg")
    plt.plot(omegas[ip_pos], spectral[ip_pos], label="P pos")

    wq_neg = np.sum(omegas[iq_neg]*spectral[iq_neg])/np.sum(spectral[iq_neg])
    wq_pos = np.sum(omegas[iq_pos]*spectral[iq_pos])/np.sum(spectral[iq_pos])
    wp_neg = np.sum(omegas[ip_neg]*spectral[ip_neg])/np.sum(spectral[ip_neg])
    wp_pos = np.sum(omegas[ip_pos]*spectral[ip_pos])/np.sum(spectral[ip_pos])

    wp_neg = omegas[ip_neg][np.argmax(spectral[ip_neg])]
    wp_pos = omegas[ip_pos][np.argmax(spectral[ip_pos])]

    domega = omegas[1] - omegas[0]
    aq_neg = np.sum(spectral[iq_neg])*domega
    ap_neg = np.sum(spectral[ip_neg])*domega
    aq_pos = np.sum(spectral[iq_pos])*domega
    ap_pos = np.sum(spectral[ip_pos])*domega

    print(aq_neg, wq_neg)
    print(aq_pos, wq_pos)
    print(ap_neg, wp_neg)
    print(ap_pos, wp_pos)
    print(aq_neg + aq_pos + ap_neg + ap_pos)

    Es = np.mean([wp_pos, -wp_neg])
    Q = Es-np.mean([wq_pos, -wq_neg])

    print(Es)
    print(Q)

    aq = (aq_neg + aq_pos) / 2
    ap = (ap_neg + ap_pos) / 2

    print(aq/ap)

    plt.plot(wq_neg, 0.1, 'or')
    plt.plot(wq_pos, 0.1, 'or')
    plt.plot(wp_neg, 0.1, 'or')
    plt.plot(wp_pos, 0.1, 'or')


    plt.legend()
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

#analyze("../results/f16h0_spec_1000.h5")
#PlotSpectralRun("../results/f16h0_spec_1000.h5")
Compare()
plt.show()



