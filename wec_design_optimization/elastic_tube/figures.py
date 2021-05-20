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
#cells = [0, 540, 720, 1440]
#power = [0, -60749.728,-61025.426, -74551.217]

# Mesh convergence test with 3 modes, design vars = [1.0, 20.0, -2.50]
#cells = [0,        70,       240,       380,       520,       660,       700,       800,       900,      1260,      1600]
#power = [0, -7204.216, -4001.619, -3750.571, -3680.533, -3650.361, -3650.750, -3688.604, -3689.265, -3655.903, -3605.987]


def tube_descriptors():
    design_variables = np.array([2.5, 90.0, -3.0])
    #tube = ElasticTube(design_variables, mode_count=5)
    f = evaluate_tube_design(design_variables)
    print(f)
    #obj = evaluate_tube_design(design_variables, mode_count=10)
    #print('\tNumber of cells= {}'.format(tube.tube.mesh.nb_faces))
    #print(tube.mode_type_1_frequency_list)
    #print(tube.mode_type_2_frequency_list)
    #print(obj)
    return

#tube_descriptors()

def intrinsic_impedance():
    
    tube = ElasticTube(tube_design_variables=np.array([2.5, 117.5, -2.75]), mode_count=5)
    resorted_dofs = ['Bulge Mode 2', 'Bulge Mode 0', 'Bulge Mode 3']
    dataset = load_data(2.5, 117.5, -2.75, 3400)

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
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('Radiation damping')
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

        tube._pto_damping_dissipated_power(tube.optimal_damping_value)
        plt.vlines(x=1e-3 * tube.optimal_damping_value, ymin=0, ymax=0.001 * tube.power_mean, linestyles='dashed')
        plt.scatter([1e-3*tube.optimal_damping_value], [0.001*tube.power_mean], marker='*', s=150)

        plt.show()


def plot_dispersion_formula():
    wave_frequencies = np.linspace(1e-4, 10*pi, 63000)
    tube = ElasticTube(np.array([0.9, 60, -1.35]), mode_count=5)
    animation = tube.tube.animate(motion={'Bulge Mode 0': 0.18}, loop_duration=6.0)
    animation.run()
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
    plt.xlim(xmin=0, xmax=10)
    plt.ylim(ymin=-2, ymax=2)
    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)
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

#tube = ElasticTube(tube_design_variables=np.array([2.5, 200.0, -3.0]), mode_count=5)

## Mode shape convergence data, 82 wave periods
# [0.650, 90.0, -11.25]
modes =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
power =   [7666.73, 13731.63, 13823.66, 13841.86, 13843.08, 13844.86, 13842.81, 13843.14, 13828.53, 13794.49]
damping = [5484016.73, 104617.46, 104409.37, 105389.42, 105456.17, 105572.98, 105477.24, 105504.52, 104728.45, 102915.79]

# [1.10, 192.0, -4.00]
modes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
power = [14822.17, 166569.90, 281448.16, 283388.01, 279700.78, 280397.38, 276994.78, 277091.68, 275966.01, 275988.33]
damping = [13962611.31, 445590.67, 327046.14, 326392.08, 320709.79, 320071.01, 309918.49, 310187.52, 306120.26, 306198.29]

# [2.75, 22.0, -1.50]
modes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
power = [22.29, 46.84, 208.32, 208.46, 236.87, 237.08, 245.78, 246.06, 249.61, 249.61]
damping = [847376.38, 7001507.75, 1592576.91, 1592392.21, 1582473.59, 1582845.42, 1581828.95, 1582335.14, 1582896.52, 1582896.52]


## Mesh cell convergence data, 5 modes and 82 wave periods
# [0.650, 90.0, -11.25]
cells = [470, 1460, 1880, 2360, 3390]
power = [12322.01, 13742.08, 13796.73, 13843.08, 14016.48]
damping = [104855.62, 105155.37, 105314.76, 105456.17, 105557.87]

# [1.10, 192.0, -4.00]
cells = [1000, 2232, 4000, 5000, 7162]
power = [253510.43, 270451.01, 279776.33, 279700.78, 282443.16]
damping = [319927.97, 320537.26, 320781.84, 320709.79, 320714.58]

# [2.75, 22.0, -1.50]
cells = [192, 404, 732, 874, 1206]
power = [224.85, 231.30, 237.58, 236.87, 238.85]
damping = [1582746.28, 1582810.40, 1582561.74, 1582473.59, 1581810.83]

f = -1 * evaluate_tube_design(design_variables=np.array([2.75, 22.0, -1.50]), mode_count=5)
print('Objective function for 5 modes is {:.2f} W'.format(f))

#print(tube.mode_type_1_frequency_list)
#print(tube.mode_type_2_frequency_list)

#plot_mode_shapes(tube, [1, 2, 3, 0, 4])
#dataset = load_data(2.5, 50.0, -3.0, 1720)
#tube.result_data = dataset
#tube.optimize_damping()
#dissipation = tube.dissipation

#hydrodynamic_results(tube, dataset, [1, 2, 0])
#plot_dissipated_power_statistics(tube)

### After fixing the wave period probability distribution function
## Mode shape convergence data, refined mesh, 80 equally spaced wave frequencies
# [0.650, 90.0, -11.25]
modes =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
power =   [311.97, 1650.89, 3363.75, 4335.06, 4919.97, 5024.29, 5057.75, 5064.12, 5046.68, 5018.87, 5011.46, 5012.87]

# [1.10, 192.0, -4.00]
modes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
power = [88.24, 2250.79, 7624.85, 69942.22, 133480.10, 205137.71, 220561.67, 247367.72, 254794.85, 260017.45, 262235.45, 263590.23]

# [2.75, 22.0, -1.50]
modes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
power = [0.99, 198.18, 765.07, 859.46, 938.12, 942.50, 965.62, 966.51, 975.87, 975.87, 975.87, 975.87]


## Mesh cell convergence data, 10 modes and 80 wave periods
# [0.650, 90.0, -11.25]
cells = [470, 1038, 1880, 2360, 3390]
power = [4520.04 , 4839.04 , 5013.62, 5018.87, 5072.83]

# [1.10, 192.0, -4.00]
cells = [1000, 2232, 4000, 5000, 7162]
power = [238199.56, 252413.99, 260152.51, 260017.45, 262253.18]

# [2.75, 22.0, -1.50]
cells = [192, 404, 732, 874, 1206, 1542, 2106]
power = [1015.82, 955.05, 980.10, 975.87, 1047.38, 1009.54, 1003.59]


## Frequency divisions, 10 modes, fully refined mesh
# [0.650, 90.0, -11.25]
divisions= [10, 20, 30]
power = [4864.22, 4984.34, 5036.17, 4979.15, 5015.43, 5018.37, 5018.25, 5018.87, 5019.25]
#### Catch that you've already run everything for 80 frequencies
# [1.10, 192.0, -4.00]
divisions= [10, 20]
power = [165139.33, 293578.73]

# [2.75, 22.0, -1.50]
divisions= [10, 20, 30, 40, 50, 60, 70, 80, 90]
power = [974.84, 971.98, 978.75, 973.45, 975.61, 975.87, 975.04, 975.87, 975.67]
