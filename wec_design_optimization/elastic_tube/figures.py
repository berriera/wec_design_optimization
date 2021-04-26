import matplotlib.pyplot as plt
import numpy  as np
from tube_class import ElasticTube, evaluate_tube_design
from math import pi
import capytaine as cpt


parameters = {'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.labelsize': 14, 'legend.fontsize': 14}
plt.rcParams.update(parameters)

design_variables = np.array([1.0, 15.0, -1.25])
tube = ElasticTube(design_variables, mode_count=3)
tube.solve_tube_hydrodynamics()
damping_value, optimal_power = tube.optimize_damping()
print('Optimal damping = {:.3f}'.format(damping_value))
print('\n')
print('Optimal power = {:.3f}'.format(optimal_power))


#f = evaluate_tube_design(design_variables=np.array([1.1, 60.0, -1.25]), mode_count=1)
#print(f)

def design_variable_history():
    import pickle
    import numpy as np
    from math import pi
    import matplotlib.pyplot as plt


    parameters = {'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.labelsize': 14, 'legend.fontsize': 14}
    plt.rcParams.update(parameters)


    geometry = True
    file_location = r'C:/Users/13365/Box/flexible_tube_optimization_results/tube_history__constrained_geometry_optimization.pkl'
    #file_location = r'C:/Users/13365/Box/flexible_tube_optimization_results/tube_history__unconstrained_material_optimization.pkl'
    with open(file_location, 'rb') as file:

        location_history = pickle.load(file)
        function_history = pickle.load(file)

        #print(location_history)
        #print(function_history)

    print('Design count = ')
    print(len(function_history))
    print(len(location_history))

    length_history = np.zeros_like(function_history)
    k = 0
    for design in location_history:
        print(design)
        length_history[k] = design[1]
        k += 1
    moves = range(1, len(length_history) + 1)
    plt.plot(length_history, moves)
    plt.vlines(x=145.0, ymin=len(function_history), ymax=0, linestyles='dotted')
    plt.scatter(x=[145.0], y=[len(function_history)], marker='*', s=200)
    plt.xlim((20, 200))
    plt.ylim((len(function_history) + 1, 0))
    plt.xlabel('Evaluated Tube Design Length $L$')
    plt.ylabel('Iteration Number')
    plt.show()

    opt_power = np.min(function_history)
    opt_index = np.where(function_history == opt_power)[0][0]
    opt_design = location_history[opt_index]
    r_opt = opt_design[0]
    L_opt = opt_design[1]
    area_opt = 2 * pi * r_opt * L_opt
    print('\nOptimal design: ')
    print(r_opt, L_opt, area_opt)

    print('')
    print('Optimal power = {} '.format(np.min(function_history)))

    print('\nOptimal design = {}'.format(opt_design))

    if geometry:
        initial_design = location_history[0]
        initial_power = function_history[0]
        r_o = initial_design[0]
        L_o = initial_design[1]
        initial_area = 2*pi*r_o*L_o

        print('Initial design: ')
        print(r_o, L_o, initial_area)
        print(initial_area)

        power_ratio = opt_power / initial_power
        area_ratio = area_opt / initial_area

        power_per_area = power_ratio / area_ratio

        print('Power ratio = {}'.format(power_ratio))
        print('\nArea ratio = {}'.format(area_ratio))

        print('Power per area ratio = {}'.format(power_per_area))


def load_data(self):
    import xarray

    # Load saved dataset and return the data
    load_file_name = 'flexible_tube_results__rs_L_zs__{}_{}_{}.nc'.format(self.static_radius, self.length, self.submergence)
    results_data = cpt.io.xarray.merge_complex_values(xarray.open_dataset(load_file_name))

    return results_data

def save_hydrodynamic_result_figures(tube):
        import matplotlib.pyplot as plt

        tube.resorted_dofs = ['Bulge Mode 1', 'Bulge Mode 0', 'Bulge Mode 2']  # 'Bulge Mode 4', 'Bulge Mode 2'
        tube.resorted_dofs = ['Bulge Mode 1', 'Bulge Mode 0', 'Bulge Mode 2']
        print(tube.mode_type_1_frequency_list)
        print(tube.mode_type_2_frequency_list)
        plt.figure()
        k=0
        for dof in tube.resorted_dofs:
            if dof.startswith('Bulge Mode'):
                plt.plot(
                    tube.wave_frequencies,
                    tube.result_data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
                    label='Bulge Mode ' + str(k)
                )
                k += 1
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('Added Mass')
        plt.legend()
        plt.show()
        #plt.savefig('added_mass.png', bbox_inches='tight')


        plt.figure()
        k=0
        for dof in tube.resorted_dofs:
            if dof.startswith('Bulge Mode'):
                plt.plot(
                    tube.wave_frequencies,
                    tube.result_data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
                    label='Bulge Mode ' + str(k)
                )
                k += 1
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('Radiation damping')
        plt.legend()
        plt.show()
        #plt.savefig('radiation_damping.png', bbox_inches='tight')

        plt.figure()
        plt.plot(
            tube.wave_periods, 
            0.001 * tube.dissipated_power_spectrum)
        plt.xlabel('Wave Period $T$ (s)')
        plt.ylabel('$P(T)$ (kW)')
        plt.show()
        #plt.savefig('dissipated_power.png', bbox_inches='tight')

        plt.figure()
        k=0
        for dof in tube.resorted_dofs:
            #if dof.startswith('Bulge Mode'):
            plt.plot(
                    tube.wave_periods,
                    np.abs(tube.modal_response_amplitude_data.sel(radiating_dof=dof)).data,
                    label='Bulge Mode ' + str(k)
            )
            k += 1
        plt.xlabel('Wave Period $T$ (s)')
        plt.ylabel('Response Amplitude Operator $\hat{a}$')
        plt.legend()
        plt.show()
        #plt.savefig('response_amplitude_operator.png', bbox_inches='tight')

#save_hydrodynamic_result_figures(tube)

def plot_mode_shapes(tube):
        import matplotlib.pyplot as plt
        from math import nan

        plt.figure()
        j = 0
        for mode_number in [3, 0, 1]:
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
        #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        #plt.savefig('tube_mode_shapes.png', bbox_inches='tight')
        plt.show()

        return

def plot_dissipated_power_statistics(tube):
        import matplotlib.pyplot as plt

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

        tube._pto_damping_dissipated_power(damping_value)
        plt.vlines(x=1e-3 * damping_value, ymin=0, ymax=0.001 * tube.power_mean, linestyles='dashed')
        plt.scatter([1e-3*damping_value], [0.001*tube.power_mean], marker='*', s=150)

        plt.show()

plot_dissipated_power_statistics(tube)

def plot_tube_design():
        tube = ElasticTube(np.array([2.5, 145, -2.75]))
        tube.load_environmental_data()
        tube.evaluate_modal_frequency_information()
        tube.normalize_mode_shapes()
        tube.plot_mode_shapes()
        print(tube.mode_type_1_frequency_list)
        print(tube.mode_type_2_frequency_list)
        tube.generate_tube()
        tube.solve_tube_hydrodynamics()
        damping_value, objective_function_value = tube.optimize_damping()
        print('\tOptimal damping value = {:.2f}'.format(damping_value))
        print('\tObjective function value = {:.2f}'.format(objective_function_value))

        tube.plot_dissipated_power_statistics()
        tube.save_hydrodynamic_result_figures()       


#plot_tube_design()

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
    from math import pi

    tube = ElasticTube(np.array([1.1, 145.0, -1.25]))
    tube.evaluate_modal_frequency_information()
    tube.normalize_mode_shapes()

    print(tube.mode_type_1_frequency_list)
    print(tube.mode_type_2_frequency_list)
    #print(tube.mode_lower_wavenumber_list)
    #print(tube.mode_upper_wavenumber_list)
    tube.plot_mode_shapes()

#plot_modes()


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
    dissipated_power = np.array([0.0, 8942.42])  # 2 rigid bodies
    dissipated_power = np.array([0.0, 14257.85]) # 6 rigid dofs
    dissipated_power_percent = 100 * dissipated_power / dissipated_power[-1]

    plt.plot(mode_count, dissipated_power_percent, marker='o')
    plt.hlines(y=(95), xmin=-2, xmax=27, linestyle='dotted')
    plt.vlines(x=(5), ymin=0, ymax=105, linestyle='dotted')
    plt.hlines(y=(100), xmin=-2, xmax=27, linestyle='solid', color='black')
    plt.xlabel('Number of Modes $N$')
    plt.ylabel('Proportion of Capable Power (%)')
    plt.xlim((0, 26))
    plt.ylim((0, 105))
    plt.show()

#mode_convergence_figure()

def submergence_power_multiplier():
    from math import asin, pi
    z_s = np.linspace(-2, 1, 3000)
    c_s = np.zeros_like(z_s)

    parameters = {'xtick.labelsize': 14, 'ytick.labelsize': 14}
    plt.rcParams.update(parameters)

    k = 0
    for z in z_s:
        if z <= -1:
            c_s[k] = 100.0
        elif z >= 1:
            c_s[k] = 0.0
        else:
            c_s[k] = 100 * (pi - 2*asin(z)) / (2 * pi)
        k += 1

    plt.plot(z_s, c_s)
    plt.vlines(x=(-1, 1), ymin=0, ymax=100, linestyles='dotted')

    plt.xlabel('$ z_s/r_s $', fontsize=14)
    plt.ylabel('Percent of Theoretically \n Available Power $ C_s $ (%)', fontsize=14)

    plt.show()
    return

#submergence_power_multiplier()

def plot_wave_probability_distribution():
    import numpy as np
    import scipy.interpolate
    import scipy.io
    import matplotlib.pyplot as plt

    from math import pi
    import os

    print('YOU ARE HERE')
    print(os.path.abspath(os.curdir))
    print('\n\n\n\n\n\n')
    parameters = {'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.labelsize': 14, 'legend.fontsize': 14}
    plt.rcParams.update(parameters)

    wave_data = scipy.io.loadmat(r'wec_design_optimization/elastic_tube/period_probability_distribution.mat')
    wave_periods = np.array(wave_data['Ta'][0])
    print(wave_periods)
    wave_probabilities = 0.01 * np.array(wave_data['Pa'][0])
    #print(wave_probabilities)

    wave_frequencies = (2 * pi) / wave_periods
    print(wave_frequencies)

    period_probability_function = scipy.interpolate.interp1d(wave_periods, wave_probabilities, bounds_error=False, fill_value=0.0)

    plt.figure()
    plt.plot(wave_periods, wave_probabilities)
    plt.xlabel('T (s)', fontsize=14)
    plt.ylabel('Wave Probability', fontsize=14)
    #plt.xlim((0, 18))
    plt.ylim((0, 0.05))
    plt.show()
