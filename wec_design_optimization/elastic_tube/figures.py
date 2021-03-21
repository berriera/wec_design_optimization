import matplotlib.pyplot as plt
import numpy  as np
from tube_class import ElasticTube

def plot_dispersion_formula():
    wave_frequencies = np.linspace(0.1, 70.0, 100000)
    tube = ElasticTube(np.array([0.9, 60, -1.35, 120e3]))
    boundary1 = np.zeros_like(wave_frequencies)
    boundary2 = np.zeros_like(wave_frequencies)
    k = 0
    for f in wave_frequencies:
        boundary1[k] = tube._mode_type_1__boundary_conditions(f)
        boundary2[k] = tube._mode_type_2__boundary_conditions(f)
        k += 1
    plt.plot(wave_frequencies, boundary1)
    plt.plot(wave_frequencies, boundary2)
    plt.xlabel('$\omega$')
    plt.hlines(y=0, xmin=wave_frequencies[0], xmax=wave_frequencies[-1])
    plt.show()

#plot_dispersion_formula()

def plot_modes():
    tube = ElasticTube(np.array([0.274, 10, -1.35, 120e3]))
    tube.evaluate_modal_frequency_information()
    tube.normalize_mode_shapes()

    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)
    #print(tube.mode_lower_wavenumber_list)
    #print(tube.mode_upper_wavenumber_list)
    tube.plot_mode_shapes()

plot_modes()

def normalize_modes():
    #tube = ElasticTube(np.array([0.90, 60, -1.35, 120e3]))
    tube = ElasticTube(np.array([0.274, 10, -1.35, 120e3]))

    tube.evaluate_modal_frequency_information()
    tube.normalize_mode_shapes()
    print(tube.normalization_factor_matrix)

#normalize_modes()
