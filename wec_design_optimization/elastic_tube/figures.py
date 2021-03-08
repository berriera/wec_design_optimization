import matplotlib.pyplot as plt
import numpy  as np
from tube_class import ElasticTube

def mode_roots():
    wave_frequencies = np.linspace(0.1, 5.0, 100)
    tube = ElasticTube(np.array([0.9, 60, -1.35, 120e3]))
    boundary = np.zeros_like(wave_frequencies)
    k = 0
    for f in wave_frequencies:
        boundary[k] = tube._mode_type_1__boundary_conditions(f)
        k += 1
    plt.plot(wave_frequencies, boundary)
    plt.xlabel('$\omega$')
    plt.hlines(y=0, xmin=wave_frequencies[0], xmax=wave_frequencies[-1])
    plt.show()

mode_roots()