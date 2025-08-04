import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

def halven(array, n, norm = True):
    n = int(n)
    npow = int(2**n)
    halved = np.zeros(len(array)//npow)
    for i in range(len(halved)):
        halved[i] = np.sum(array[i*npow:(i+1)*npow])
    if norm:
        halved/=npow
    return halved

def condense_array(array):
    return (array[:-1] + array[1:])/2


def TidyData(filename, type="boson", number="single"):
    f = h5.File(filename, 'r')
    keys = np.array(list(f.keys()))
    keys = keys[:]
    taus = f["1/taus"][:]
    beta = f["1/beta"][()]
    size = f["1/N"][()]
    n_sims = len(keys)
    true_densities = np.zeros(len(keys))
    true_double_densities = np.zeros(len(keys))
    greens = np.zeros([len(keys), len(taus)])
    for i in range(len(keys)):
        key = keys[i]
        z = f[key][type + "ic_partition"][()]
        greens[i, :] = f[key][number + "_green_" + type][:]/z
        true_densities[i] = np.mean(f[key][type + "ic_occupation"][:])/(size*z)
        true_double_densities[i] = np.mean(f[key][type + "ic_double_occupation"][:])/(size*z)
        #green += f[key]["double_green_boson"][:]/z
        #true_density += np.mean(f[key]["bosonic_double_occupation"][:])/z


    true_density = np.mean(true_densities)
    true_double_density = np.mean(true_double_densities)
    true_single_density = true_density - true_double_density
    green = np.mean(greens, axis=0)
    td_err = np.std(true_densities)/np.sqrt(len(keys))
    green_err = np.std(greens, axis=0)/np.sqrt(len(keys))
    c = 1.5*green[0] - 0.5*green[1] + 1.5*green[-1] - 0.5*green[-2]
    green /= c
    true_double_density /= 1 - 2*true_single_density
    n_halve = num
    d = 1.5*green[-1] - 0.5*green[-2]
    #print(len(taus))
    taus = np.append(-beta, np.append(condense_array(halven(taus, n_halve, True)), 0))
    #print(len(taus))
    green = np.append(1-d, np.append(condense_array(halven(green, n_halve, True)), d))
    short_greens = np.zeros([len(keys), len(green)])
    for i in range(len(keys)):
        print(i)
        ci = 1.5*greens[i][0] - 0.5*greens[i][1] + 1.5*greens[i][-1] - 0.5*greens[i][-2]
        greens[i] /= ci
        di = 1.5*greens[i][-1] - 0.5*greens[i][-2]
        short_greens[i] = np.append(1-di, np.append(condense_array(halven(greens[i], n_halve, True)), di))
    short_green = np.mean(short_greens, axis=0)
    short_err = np.std(short_greens, axis=0)/np.sqrt(len(keys))
    #plt.plot(taus, green, color='black')
    #plt.errorbar(taus, short_green, yerr=short_err, color='red')
    #plt.figure()
    #plt.plot(taus, short_green)
    #plt.errorbar([-beta, 0], [1 - true_density, true_density], yerr=[td_err, td_err], fmt='.')
    return beta, -short_green, taus + beta

def SaveGreen(filename, green, beta):
    print(f"Saving {filename}")
    f = h5.File(filename, "w")
    f["green"] = green
    f["beta"] = beta

num = 0

type = "fermion"
prefix = "f"

beta, green, taus = TidyData("../data/q25_green_12_small.h5", type=type)
SaveGreen(f"../data/{prefix}12h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_14_small.h5", type=type)
SaveGreen(f"../data/{prefix}14h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_16_small.h5", type=type)
SaveGreen(f"../data/{prefix}16h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_17_small.h5", type=type)
SaveGreen(f"../data/{prefix}17h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_18_small.h5", type=type)
SaveGreen(f"../data/{prefix}18h{num}.h5", green, beta)
#plt.figure()

type = "boson"
prefix = "b"

beta, green, taus = TidyData("../data/q25_green_12_small.h5", type=type)
SaveGreen(f"../data/{prefix}12h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_14_small.h5", type=type)
SaveGreen(f"../data/{prefix}14h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_16_small.h5", type=type)
SaveGreen(f"../data/{prefix}16h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_17_small.h5", type=type)
SaveGreen(f"../data/{prefix}17h{num}.h5", green, beta)
beta, green, taus = TidyData("../data/q25_green_18_small.h5", type=type)
SaveGreen(f"../data/{prefix}18h{num}.h5", green, beta)
plt.show()

