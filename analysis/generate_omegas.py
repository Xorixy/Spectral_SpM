import numpy as np
import h5py as h5
import matplotlib.pyplot as plt




def OmegasFromSeparationPoints(points):
    omegas = np.zeros(len(points) - 1)
    domegas = np.zeros(len(points) - 1)
    for i in range(len(points) - 1):
        omegas[i] = (points[i+1] + points[i])/2
        domegas[i] = points[i+1] - points[i]
    return omegas, domegas

def SaveOmegas(filename, omegas, domegas):
    f = h5.File(filename, "w")
    f["omegas"] = omegas
    f["domegas"] = domegas


i1s = -23
i1e = -13
i2s = -13
i2e = -3
i3s = -3
i3e = 3
i4s = 3
i4e = 13
i5s = 13
i5e = 23

d1 = 100/44
d2 = 500/44
d3 = 50/44
d4 = 500/44
d5 = 100/44

n1 = int((i1e - i1s)*d1)
n2 = int((i2e - i2s)*d2)
n3 = int((i3e - i3s)*d3)
n4 = int((i4e - i4s)*d4)
n5 = int((i5e - i5s)*d5)

p1 = np.linspace(i1s, i1e, n1)
p2 = np.linspace(i2s, i2e, n2)
p3 = np.linspace(i3s, i3e, n3)
p4 = np.linspace(i4s, i4e, n4)
p5 = np.linspace(i5s, i5e, n5)

points = np.unique(np.append(np.append(np.append(p1, p2), np.append(p3, p4)), p5))

omegas, domegas = OmegasFromSeparationPoints(points)

plt.plot(points, len(points)*[0], 'o')
plt.plot(omegas, domegas, 'o')

print(len(omegas))

SaveOmegas("../data/omegas.h5", omegas, domegas)

plt.show()


