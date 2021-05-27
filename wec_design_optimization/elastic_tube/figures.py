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


def intrinsic_impedance(tube, dof_ints, rs, le, zs, ce):
    resorted_dofs = []
    for dof_int in dof_ints:
        dof_name = 'Bulge Mode ' + str(dof_int)
        resorted_dofs.append(dof_name)

    dataset = load_data(rs, le, zs, ce)

    tube.result_data = dataset
    tube.optimize_damping()
    dissipation = tube.dissipation

    omega = dataset.coords['omega']
    A = (-omega**2*(dataset['mass'] + dataset['added_mass'])
            + 1j*omega*dataset['radiation_damping']
            + dataset['hydrostatic_stiffness'])
    A = A + 1j*omega*dissipation

    intrinsic_impedance = xr.DataArray(A, coords=[omega, dataset.coords['radiating_dof'], dataset.coords['influenced_dof']], dims=['omega', 'radiating_dof', 'influenced_dof'])

    plt.figure()
    k=0
    for dof in resorted_dofs:
        plt.plot(omega,
                np.abs(intrinsic_impedance.sel(radiating_dof=dof, influenced_dof=dof)),
                label='Bulge Mode ' + str(k)
                )
        k += 1
    plt.xlabel('Wave Frequency $\omega$ [rad/s]')
    plt.ylabel('Intrinsic Impedance Magnitude')
    plt.legend()
    plt.show()

    plt.figure()
    k=0
    for dof in resorted_dofs:
        plt.plot(omega,
                np.angle(intrinsic_impedance.sel(radiating_dof=dof, influenced_dof=dof), deg=True),
                label='Bulge Mode ' + str(k)
                )
        k += 1
    plt.xlabel('Wave Frequency $\omega$ [rad/s]')
    plt.ylabel('Intrinsic Impedance Angle')
    plt.legend()
    plt.show()
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
    load_file_name = 'flexible_tube_results__rs_le_zs__{}_{}_{}__with_{}_cells.nc'.format(radius, length, submergence, cell_count)
    results_data = cpt.io.xarray.merge_complex_values(xr.open_dataset(load_file_name))

    return results_data

def hydrodynamic_results(tube, result_data, sorted_dof_integers):

    sorted_dof_names = []
    for dof in sorted_dof_integers:
        sorted_dof_names.append('Bulge Mode ' + str(dof))

    plt.figure()
    k=0
    for dof in sorted_dof_names:
        plt.plot(
            tube.wave_frequencies,
            result_data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
            label='Bulge Mode ' + str(k)
        )
        k += 1
    plt.xlabel('Wave Frequency $\omega$ [rad/s]')
    plt.ylabel('Added Mass [kg]')
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
    plt.xlabel('Wave Frequency $\omega$ [rad/s]')
    plt.ylabel('Radiation Damping [N$\cdot$s/m]')
    plt.legend()
    plt.show()
        
    plt.figure()
    plt.plot(
        tube.wave_frequencies,
        0.001 * tube.dissipated_power_spectrum)
    power_mean = 0.001 * np.sum(tube.dissipated_power_spectrum * tube.frequency_probability_distribution)
    plt.hlines(y=power_mean, xmin=0.0, xmax=2.30, color='black', linestyles='dotted', label='Probability Averaged Mean Power')

    plt.xlabel('Wave Frequency $\omega$ [rad/s]')
    plt.ylabel('Dissipated Power Spectrum $P_{PTO}(\omega)$ [kW]')
    plt.show()

    plt.figure()
    k=0
    for dof in sorted_dof_names:
        plt.plot(
            tube.wave_frequencies,
            np.abs(tube.modal_response_amplitude_data.sel(radiating_dof=dof)).data,
            label='Bulge Mode ' + str(k)
        )
        k += 1
    plt.xlabel('Wave Frequency $\omega$ (rad/s)')
    plt.ylabel('Response Amplitude Operator\nMagnitude $|\hat{a}|$')
    plt.legend()
    plt.show()
    return


def plot_mode_shapes(tube, sorted_dof_integers):
        plt.figure()
        j = 0
        for mode_number in sorted_dof_integers:
            tube_x = np.linspace(tube.integration_bounds[0], tube.integration_bounds[1], 20000)
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

def plot_tube_shapes(tube, sorted_dof_integers):
    plt.figure()
    plt.show()
    pass

def plot_dissipated_power_statistics(tube, penalties=[0]):
        damping_values = np.linspace(0, 5.0 * tube.optimal_damping_value, 300)
        power_mean_values = np.zeros_like(damping_values)
        power_standard_deviation_values = np.zeros_like(damping_values)

        k = 0
        for b in damping_values:
            tube._pto_damping_dissipated_power(b)
            power_mean_values[k] = tube.power_mean
            power_standard_deviation_values[k] = tube.power_standard_deviation
            k += 1

        tube._pto_damping_dissipated_power(tube.optimal_damping_value)

        damping_values = 0.001 * damping_values
        power_mean_values = 0.001 * power_mean_values
        power_standard_deviation_values = 0.001 * power_standard_deviation_values

        plt.figure()
        plt.plot(damping_values, power_mean_values, label='Mean Power')
        plt.plot(damping_values, power_standard_deviation_values, '.-', label='Standard Deviation in Power')
        for penalty in penalties:
            if penalty != 0:
                combined_objective = power_mean_values - penalty * (power_standard_deviation_values ** 2)
                plt.plot(damping_values, combined_objective, '.', label='r={}'.format(penalty))

        
        plt.vlines(x=1e-3 * tube.optimal_damping_value, ymin=0, ymax=0.001 * tube.power_mean, linestyles='dashed')
        plt.scatter([1e-3*tube.optimal_damping_value], [0.001*tube.power_mean], marker='*', s=150)
        plt.xlabel('Power Take Off Damping Value [kPa $ \cdot $ s /m$^2$]')
        plt.ylabel('Dissipated PTO Power [kW]')
        plt.legend()
        plt.show()

        #plt.figure()
        #plt.plot(0.001 * damping_values, 0.001 * power_mean_values)
        #plt.xlabel('Power Take Off Damping Value [kPa $ \cdot $ s /m$^2$]', fontsize=14)
        #plt.ylabel('Dissipated PTO Power [kW]', fontsize=14)

        #tube._pto_damping_dissipated_power(tube.optimal_damping_value)
        #plt.vlines(x=1e-3 * tube.optimal_damping_value, ymin=0, ymax=0.001 * tube.power_mean, linestyles='dashed')
        #plt.scatter([1e-3*tube.optimal_damping_value], [0.001*tube.power_mean], marker='*', s=150)

        #plt.show()


def plot_dispersion_formula(tube):
    wave_frequencies = np.linspace(0.215, 2.085, 500)
    boundary1 = np.zeros_like(wave_frequencies)
    boundary2 = np.zeros_like(wave_frequencies)
    k = 0
    for f in wave_frequencies:
        boundary1[k] = tube._mode_type_1__boundary_conditions(f)
        boundary2[k] = tube._mode_type_2__boundary_conditions(f)
        k += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(wave_frequencies, boundary1)
    ax2.plot(wave_frequencies, boundary2)
    ax2.set_xlabel('Wave Frequency $\omega$ [rad/s]')
    ax1.set_ylabel('$f_1(\omega$)')
    ax2.set_ylabel('$f_2(\omega$)')

    ax1.hlines(y=0, xmin=0.215, xmax=2.085, color='black', linestyles ='dotted')
    ax2.hlines(y=0, xmin=0.215, xmax=2.085, color='black', linestyles ='dotted')
    ax2.set_xlim(xmin=0.215, xmax=2.085)
    ax1.set_ylim(ymin=-5000, ymax=5000)
    ax2.set_ylim(ymin=-5000, ymax=5000)
    ax1.scatter(np.array(tube.mode_type_1_frequency_list), np.zeros_like(tube.mode_type_1_frequency_list), s=100)
    ax2.scatter(np.array(tube.mode_type_2_frequency_list), np.zeros_like(tube.mode_type_2_frequency_list), s=100)
    plt.show()

    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)

    return


def mode_convergence_figure(modes, power_list):
    design_count = len(power_list)
    for k in range(design_count):
        dissipated_power = power_list[k]
        dissipated_power_percent = 100 * dissipated_power / dissipated_power[-1]
        plt.plot(modes, dissipated_power_percent, marker='o', label='Design {}'.format(k+1))

    plt.hlines(y=(98, 102), xmin=-2, xmax=14, linestyle='dotted')
    plt.hlines(y=(100), xmin=-2, xmax=27, linestyle='solid', color='black')
    #plt.vlines(x=(10), ymin=0, ymax=100, linestyle='dotted')
    plt.xlabel('Number of Modes $N$')
    plt.ylabel('Proportion of Capable Power [%]')
    plt.xlim((0, 12))
    plt.ylim((0, 110))
    plt.legend()
    plt.show()

    for k in range(design_count):
        dissipated_power = power_list[k]
        dissipated_power_percent = 100 * (dissipated_power - dissipated_power[-1]) / dissipated_power[-1]
        plt.plot(modes, dissipated_power_percent, marker='o', label='Design {}'.format(k+1))

    plt.hlines(y=(2, -2), xmin=-2, xmax=14, linestyle='dotted')
    #plt.vlines(x=(10), ymin=-10, ymax=10, linestyle='dotted')
    plt.hlines(y=(0), xmin=-2, xmax=14, linestyle='solid', color='black')
    plt.xlabel('Number of Modes $N$')
    plt.ylabel('Relative Error in Total Capable Power [%]')
    plt.xlim((0, 12))
    plt.ylim((-4, 4))
    #plt.legend()
    plt.show()

    return

def mesh_convergence_figure(cell_list, power_list):
    design_count = len(cell_list)
    assert design_count == len(power_list)

    for k in range(design_count):
        cells = cell_list[k]
        power = power_list[k]
        dissipated_power_percent = 100 * np.absolute(power - power[-1]) / power[-1]
        plt.plot(cells, dissipated_power_percent, marker='o', label='Design {}'.format(k+1))

    #plt.hlines(y=(98, 102), xmin=-2, xmax=14, linestyle='dotted')
    #plt.hlines(y=(0), xmin=-2, xmax=27, linestyle='solid', color='black')
    #plt.vlines(x=(10), ymin=0, ymax=100, linestyle='dotted')
    plt.xlabel('Mesh Cell Count')
    plt.ylabel('Relative Error in \n Average Capable Power [%]')
    plt.xlim((0, 8000))
    plt.ylim((0, 30))
    plt.legend()
    plt.show()

def frequency_convergence_figure(frequency_list, power_list):
    design_count = len(frequency_list)
    assert design_count == len(power_list)

    for k in range(design_count):
        frequencies = frequency_list[k]
        power = power_list[k]
        dissipated_power_percent = 100 * (power - power[-1]) / power[-1]
        plt.plot(frequencies, dissipated_power_percent, marker='o', label='Design {}'.format(k+1))

    plt.hlines(y=(0.25, -0.25), xmin=-2, xmax=102, linestyle='dotted')
    #plt.hlines(y=(0), xmin=-2, xmax=27, linestyle='solid', color='black')
    #plt.vlines(x=(50), ymin=0, ymax=100, linestyle='dotted')
    plt.xlabel('Incident Frequency Count')
    plt.ylabel('Relative Error in \n Average Capable Power [%]')
    plt.xlim((20, 100))
    plt.ylim((-4.0, 4.0))
    plt.legend()
    plt.show()


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

def evaluate_tube(design_vars, modes=10):
    tube = ElasticTube(tube_design_variables=design_vars, mode_count=modes)
    print('Modal frequencies: ')
    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)
    print('Cell count = {} cells'.format(tube.tube.mesh.nb_faces))
    f = -1 * evaluate_tube_design(design_variables=design_vars, mode_count=modes)
    print('Absolute value of objective function with {} modes is {:.2f} W'.format(modes, f))
        
    return

def plot_all_tube_results(design_vars, cells, modes=10, mode_ints=[0]):
    tube = ElasticTube(tube_design_variables=design_vars, mode_count=modes)
    plot_mode_shapes(tube, mode_ints)
    rs = design_vars[0]
    le = design_vars[1]
    zs = design_vars[2]

    dataset = load_data(rs, le, zs, cells)
    tube.result_data = dataset
    tube.optimize_damping()

    #hydrodynamic_results(tube, dataset, mode_ints)
    plot_dissipated_power_statistics(tube, penalties=[0.0001, 0.0005, 0.001, 0.005])

def plot_sampled_designs():
    design1 = np.array([0.650, 90.0, -11.25])
    design2 = np.array([1.10, 192.0, -4.00])
    design3 = np.array([2.75, 22.0, -1.50])
    design4 = np.array([1.10, 35.0, -1.25])
    #design5 = np.array([1.25, 35.0, -1.25])
    designs = [design1, design2, design3, design4]
    cells_list = [3384, 7152, 1186, 1488, 1536]
    modes_list = [[4, 0, 5], [4, 0, 5], [3, 4, 5], [4, 0, 5], [4, 5, 6]]
    k = 0
    for k in range(4):
        #evaluate_tube(design_vars=design)
        design_vars = designs[k]
        cells = cells_list[k]
        mode_ints = modes_list[k]
        plot_all_tube_results(design_vars, cells, mode_ints=mode_ints)

#plot_sampled_designs()
#evaluate_tube(np.array( [1.15, 200., -1.25]))
#plot_all_tube_results(np.array([1.15, 200., -1.25]), 7450, mode_ints=[4, 0, 5])

tube = ElasticTube(tube_design_variables=np.array([1.10, 192.0, -4.00]), mode_count=12)
#tube = ElasticTube(tube_design_variables=np.array([1.15, 200., -1.25]), mode_count=12)

#plot_dispersion_formula(tube)
#intrinsic_impedance(tube, [4, 0, 5], 1.15, 200.0, -1.25, 7450)

### After fixing the wave period probability distribution function
## Mode shape convergence data, refined mesh, 80 equally spaced wave frequencies: 10 modes final answer
# [0.650, 90.0, -11.25]
modes =   np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
power1 =   np.array([0, 311.97, 1650.89, 3363.75, 4335.06, 4919.97, 5024.29, 5057.75, 5064.12, 5046.68, 5018.87, 5011.46, 5012.87])

# [1.10, 192.0, -4.00]
power2 = np.array([0, 88.24, 2250.79, 7624.85, 69942.22, 133480.10, 205137.71, 220561.67, 247367.72, 254794.85, 260017.45, 262235.45, 263590.23])

# [2.75, 22.0, -1.50]
power3 = np.array([0, 0.99, 198.18, 765.07, 859.46, 938.12, 942.50, 965.62, 966.51, 975.87, 975.87, 975.87, 975.87])

# [1.10, 35.0, -1.25]
power4 = np.array([0, 2834.25, 11467.10, 17221.75, 22578.98, 21678.91, 21974.80, 21660.98, 21687.69, 21575.48, 21580.05, 21533.22, 21534.42])

#mode_convergence_figure(modes, [power1, power2, power3, power4])

## Mesh cell convergence data, 10 modes and 80 wave frequencies
# [0.650, 90.0, -11.25]
cells1 = np.array([470, 1038, 1880, 2360, 3390])
power1 = np.array([4520.04 , 4839.04 , 5013.62, 5018.87, 5072.83])

# [1.10, 192.0, -4.00]
cells2 = np.array([1000, 2232, 4000, 5000, 7162])
power2 = np.array([238199.56, 252413.99, 260152.51, 260017.45, 262253.18])

# [2.75, 22.0, -1.50]
cells3 = np.array([192, 404, 732, 874, 1206, 1542, 2106])
power3 = np.array([1015.82, 955.05, 980.10, 975.87, 1047.38, 1009.54, 1003.59])

# [1.10, 35.0, -1.25]
cells4 = np.array([210, 482, 860, 1060, 1498, 1980, 2970, 4130])
power4 = np.array([51040.38, 34142.98, 25995.07, 21580.05, 19960.21, 18791.41, 18526.93, 18416.32])

#mesh_convergence_figure([cells1, cells2, cells3, cells4], [power1, power2, power3, power4])

## Frequency divisions, 10 modes, fully refined mesh: 50 frequencies final answer
# [0.650, 90.0, -11.25]
divisions1 = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
power1 = np.array([14371.15, 4864.22, 4984.34, 5036.17, 4979.15, 5015.43, 5018.37, 5018.25, 5018.87, 5019.25])

# [1.10, 192.0, -4.00]
divisions2 = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
power2 = np.array([209931.46, 165139.33, 293578.73, 250560.80, 255950.94, 260428.95, 259606.60, 260112.74, 260017.45, 259887.74])

# [2.75, 22.0, -1.50]
divisions3 = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
power3 = np.array([923.64, 974.84, 971.98, 978.75, 973.45, 975.61, 975.87, 975.04, 975.87, 975.67])

# [1.10, 35.0, -1.25]
divisions4 = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
power4 = np.array([23974.17, 23645.14, 21660.26, 21624.84, 21501.35, 21572.35, 21588.47, 21565.20, 21580.05, 10051.49, 21573.71])

divisions = [divisions1, divisions2, divisions3, divisions4]
powers = [power1, power2, power3, power4]
#frequency_convergence_figure(divisions, powers)
