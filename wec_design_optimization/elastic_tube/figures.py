import matplotlib.pyplot as plt
import numpy  as np
from tube_class import ElasticTube
from math import pi

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
    tube = ElasticTube(np.array([0.90, 60, -1.35, 120e3]))
    tube.evaluate_modal_frequency_information()
    tube.normalize_mode_shapes()

    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)
    #print(tube.mode_lower_wavenumber_list)
    #print(tube.mode_upper_wavenumber_list)
    #ube.plot_mode_shapes()

#plot_modes()

def normalize_modes():
    #tube = ElasticTube(np.array([0.90, 60, -1.35, 120e3]))
    tube = ElasticTube(np.array([0.274, 10, -1.35, 120e3]))

    tube.evaluate_modal_frequency_information()
    tube.normalize_mode_shapes()
    print(tube.normalization_factor_matrix)

#normalize_modes()

def damping_figure():
    elastic_tube_instance = ElasticTube(np.array([0.90, 60, -1.35]))
    elastic_tube_instance.load_environmental_data()
    elastic_tube_instance.evaluate_modal_frequency_information()
    elastic_tube_instance.normalize_mode_shapes()
    elastic_tube_instance.generate_tube()
    elastic_tube_instance.solve_tube_hydrodynamics()

    damping_values = np.linspace(0, 2e7, 100)
    power_values = np.zeros_like(damping_values)

    k = 0
    for b in damping_values:
        power_values[k] = elastic_tube_instance._optimal_damping(b)
        k += 1

    power_values = np.abs(power_values)

    plt.plot(damping_values, power_values)
    plt.show()

    return

#damping_figure()


def mode_convergence_figure():
    mode_count = np.array([0,1,2,3,4,5,6,7,8,9,10,15,20,25])
    dissipated_power = np.array([0.0, 8709.57, 9275.78, 10038.33, 10177.16, 10413.67, 10472.76, 10588.84, 10620.17, 10691.56, 10710.14, 10849.69, 10911.23, 10973.66])

    dissipated_power_percent = 100 * dissipated_power / dissipated_power[-1]

    plt.plot(mode_count, dissipated_power_percent, marker='o')
    plt.hlines(y=(95), xmin=-2, xmax=27, linestyle='dotted')
    plt.vlines(x=(5), ymin=0, ymax=105, linestyle='dotted')
    plt.hlines(y=(100), xmin=-2, xmax=27, linestyle='solid', color='black')
    plt.xlabel('Number of Modes', fontsize=14)
    plt.ylabel('Percent of Capable Power', fontsize=14)
    plt.xlim((0, 26))
    plt.ylim((0, 105))
    plt.show()

#mode_convergence_figure()
