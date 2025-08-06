import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

def PlotConvergence(filename):
    f = h5.File(filename, 'r')
    errors = f["errors"][()]
    plt.plot(errors)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

filename = "../results/f12h0_conv_1000_norec.h5"
PlotConvergence(filename)