import matplotlib.pyplot as plt
import capytaine as cpt
import numpy  as np
import pickle
import scipy.io
import xarray as xr
from math import pi, asin, nan
from tube_class import ElasticTube, evaluate_tube_design

# Globally update plotting parameters
parameters = {'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.labelsize': 14, 'legend.fontsize': 14}
plt.rcParams.update(parameters)

# Mesh convergence test with 3 modes, design vars = [2.5, 20.0, -2.50]
cells = [0, 540, 720, 1440]
power = [0, -60749.728,-61025.426, -74551.217]

# Mesh convergence test with 3 modes, design vars = [1.0, 20.0, -2.50]
cells = [0,        70,       240,       380,       520,       660,       700,       800,       900,      1260,      1600]
power = [0, -7204.216, -4001.619, -3750.571, -3680.533, -3650.361, -3650.750, -3688.604, -3689.265, -3655.903, -3605.987]


def run_tube():
    design_variables = np.array([1.0, 20.0, -2.5])
    tube = ElasticTube(design_variables, mode_count=1)
    print('\tNumber of cells= {}'.format(tube.tube.mesh.nb_faces))
    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)
    return


def design_variable_history():
    file_location = r'C:\Users\13365\Box\flexible_tube_optimization_results\flexible_and_rigid_dofs\tube_history__unconstrained_geometry_opt__full_hydrodynamic_model.pkl'
    with open(file_location, 'rb') as file:

        location_history = pickle.load(file)
        function_history = pickle.load(file)

    optimal_power = np.min(function_history)
    optimal_design_index = np.where(function_history == optimal_power)[0][0]
    opt_design = location_history[optimal_design_index]
    optimal_design_length = opt_design[1]

    variable_history = np.zeros_like(function_history)
    k = 0
    for design in location_history:
        print(design)
        variable_history[k] = design[0]
        k += 1
    moves = range(1, len(variable_history) + 1)
    plt.plot(variable_history, moves)
    plt.vlines(x=optimal_design_length, ymin=len(function_history), ymax=0, linestyles='dotted')
    plt.scatter(x=[optimal_design_length], y=[len(function_history)], marker='*', s=200)
    plt.xlim((20, 200))
    plt.ylim((len(function_history) + 1, 0))
    plt.xlabel('Evaluated Tube Design Length $L$')
    plt.ylabel('Iteration Number')
    plt.show()
    return


def load_data(radius, length, submergence, cell_count):

    # Load saved dataset and return the data
    load_file_name = 'flexible_tube_results__rs_L_zs__{}_{}_{}__with_{}_cells.nc'.format(radius, length, submergence, cell_count)
    results_data = cpt.io.xarray.merge_complex_values(xr.open_dataset(load_file_name))

    return results_data

def save_hydrodynamic_result_figures(tube, result_data, sorted_dof_integers):

    sorted_dof_names = []
    for dof in sorted_dof_integers:
        sorted_dof_names.append('Bulge Mode' + str(dof))

    plt.figure()
    k=0
    for dof in sorted_dof_names:
        plt.plot(
            tube.wave_frequencies,
            result_data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
            label='Bulge Mode ' + str(k)
        )
        k += 1
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('Added Mass')
    plt.legend()
    plt.show()

    plt.figure()
    k=0
    for dof in sorted_dof_names:
        plt.plot(
                tube.wave_frequencies,
                result_data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
                label='Bulge Mode ' + str(k)
            )
        k += 1
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('Radiation damping')
    plt.legend()
    plt.show()
        
    plt.figure()
    plt.plot(
        tube.wave_periods, 
        0.001 * tube.dissipated_power_spectrum)
    plt.xlabel('Wave Period $T$ [s]')
    plt.ylabel('Dissipated Power Spectrum $P(T)$ [kW]')
    plt.show()

    plt.figure()
    k=0
    for dof in sorted_dof_names:
        plt.plot(
            tube.wave_periods,
            np.abs(tube.modal_response_amplitude_data.sel(radiating_dof=dof)).data,
            label='Bulge Mode ' + str(k)
        )
    k += 1
    plt.xlabel('Wave Period $T$ (s)')
    plt.ylabel('Response Amplitude Operator $|\hat{a}|$')
    plt.legend()
    plt.show()
    return


def plot_mode_shapes(tube, sorted_dof_integers):
        plt.figure()
        j = 0
        for mode_number in sorted_dof_integers:
            tube_x = np.linspace(tube.integration_bounds[0], tube.integration_bounds[1], 250)
            tube_radial_deformation = np.zeros_like(tube_x)

            k = 0
            for x in tube_x:
                tube_radial_deformation[k] = tube.mode_shape_derivatives(x, y=nan, z=nan, mode_number=mode_number, plot_flag=True)
                k += 1
            plt.plot(tube_x, tube_radial_deformation, label='Mode {}'.format(j))
            j += 1
        
        plt.xlabel('Tube Length $x$ (m)')
        plt.ylabel('Radial Deformation $\delta r$ (m)')
        plt.legend(loc='upper right')
        plt.show()

        return

def plot_dissipated_power_statistics(tube):
        damping_values = np.linspace(0, 5.0 * tube.optimal_damping_value, 300)
        power_mean_values = np.zeros_like(damping_values)
        power_standard_deviation_values = np.zeros_like(damping_values)

        k = 0
        for b in damping_values:
            tube._pto_damping_dissipated_power(b)
            power_mean_values[k] = tube.power_mean
            power_standard_deviation_values[k] = tube.power_standard_deviation
            k += 1

        plt.figure()
        plt.plot(0.001 * damping_values, 0.001 * power_mean_values, label='Mean Power')
        plt.plot(0.001 * damping_values, 0.001 * power_standard_deviation_values, label='Standard Deviation in Power')
        plt.xlabel('Power Take Off Damping Value (kPa $ \cdot $ s /m$^2$)')
        plt.ylabel('Dissipated PTO Power (kW)')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(0.001 * damping_values, 0.001 * power_mean_values)
        plt.xlabel('Power Take Off Damping Value (kPa $ \cdot $ s /m$^2$)', fontsize=14)
        plt.ylabel('Dissipated PTO Power (kW)', fontsize=14)

        tube._pto_damping_dissipated_power(optim)
        plt.vlines(x=1e-3 * damping_value, ymin=0, ymax=0.001 * tube.power_mean, linestyles='dashed')
        plt.scatter([1e-3*damping_value], [0.001*tube.power_mean], marker='*', s=150)

        plt.show()


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
    return


def damping_figure(tube):
    damping_values = np.linspace(0, 2e7, 100)
    power_values = np.zeros_like(damping_values)

    k = 0
    for b in damping_values:
        power_values[k] = tube._optimal_damping(b)
        k += 1

    power_values = np.abs(power_values)

    plt.plot(damping_values, power_values)
    plt.show()

    return


def mode_convergence_figure():
    mode_count = np.array([0,1,2,3,4,5,6,7,8,9,10,15,20,25])
    dissipated_power = np.array([0.0, 8709.57, 9275.78, 10038.33, 10177.16, 10413.67, 10472.76, 10588.84, 10620.17, 10691.56, 10710.14, 10849.69, 10911.23, 10973.66])
    #mode_count = np.array([      0,         1,         2,         3,          4,         5,         6,        7,        8,         9,        10,         15,       20,         25])
    #dissipated_power = np.array([0, -3203.403, -3212.627, -3650.750,  -3654.096, -3800.032, -3811.432,-3794.230, -3713.957, -3731.736, -3723.279, -3776.180, -3820.286, -3837.304])

    # Mode convergence test, design vars = [1.0, 20.0, -2.5]
    mode_count = [0,         1,         2,         3,          4,         5,         6,        7,        8,         9,        10,         15,       20,         25]
    power_mean = [0, -3203.403, -3212.627, -3650.750,  -3654.096, -3800.032, -3811.432,-3794.230, -3713.957, -3731.736, -3723.279, -3776.180, -3820.286, -3837.304]

    dissipated_power_percent = 100 * dissipated_power / dissipated_power[-1]

    plt.plot(mode_count, dissipated_power_percent, marker='o')
    plt.hlines(y=(95), xmin=-2, xmax=27, linestyle='dotted')
    plt.vlines(x=(5), ymin=0, ymax=100, linestyle='dotted')
    plt.hlines(y=(100), xmin=-2, xmax=27, linestyle='solid', color='black')
    plt.xlabel('Number of Modes $N$')
    plt.ylabel('Proportion of Capable Power [%]')
    plt.xlim((0, 26))
    plt.ylim((0, 105))
    plt.show()
    return


def submergence_power_multiplier():
    z_s = np.linspace(-2, 1, 3000)
    c_s = np.zeros_like(z_s)

    parameters = {'xtick.labelsize': 14, 'ytick.labelsize': 14}
    plt.rcParams.update(parameters)

    k = 0
    for z in z_s:
        if z <= -1:
            c_s[k] = 100.0
        else:
            c_s[k] = 100 * (pi - 2*asin(z)) / (2 * pi)
        k += 1

    plt.plot(z_s, c_s)
    plt.vlines(x=(-1, 1), ymin=0, ymax=100, linestyles='dotted')
    plt.xlabel('$ z_s/r_s $ []', fontsize=14)
    plt.ylabel('Percent of Theoretically \n Available Power $ C_s $ [%]', fontsize=14)
    plt.show()
    return


def plot_wave_probability_distribution():
    wave_data = scipy.io.loadmat(r'wec_design_optimization/elastic_tube/period_probability_distribution.mat')
    wave_periods = np.array(wave_data['Ta'][0])
    wave_probabilities = 0.01 * np.array(wave_data['Pa'][0])
    plt.figure()
    plt.plot(wave_periods, wave_probabilities)
    plt.xlabel('Wave Period $T$ [s]')
    plt.ylabel('Wave Probability []')
    plt.ylim((0, 0.05))
    plt.show()
    return
