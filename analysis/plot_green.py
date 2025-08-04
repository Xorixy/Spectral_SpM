import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

def plot_green(filename):
    f = h5.File(filename)
    beta = f["beta"][()]
    green = f["green"][()]
    taus = np.linspace(0, beta, len(green))
    plt.plot(taus, green)


plot_green("../data/f16h0.h5")
plot_green("../data/b16h0.h5")

plt.show()