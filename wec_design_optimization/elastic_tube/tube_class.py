import numpy as np
import capytaine as cpt
import logging

def evaluate_tube_design(design_variables):
        """Evaluates a complete tube design from start to finish

        """
        elastic_tube_instance = ElasticTube(design_variables)
        elastic_tube_instance.load_environmental_data()
        elastic_tube_instance.evaluate_modal_frequency_information()
        elastic_tube_instance.normalize_mode_shapes()
        elastic_tube_instance.generate_tube()
        elastic_tube_instance.solve_tube_hydrodynamics()
        damping_value, objective_function_value = elastic_tube_instance.optimize_damping()
        print('\tOptimal damping value = {:.2f}'.format(damping_value))

        return objective_function_value


class ElasticTube(object):


    def __init__(self, tube_design_variables):
        from math import inf, pi

        #logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")
        self.save_results = True

        # Unpack independent design variables
        # Variable notation: r_s, L, z_s
        self.static_radius = tube_design_variables[0]
        self.length = tube_design_variables[1]
        self.submergence = tube_design_variables[2]

        # Environment and incident wave constants
        self.rho = 1000
        self.water_depth = -inf
        self.wave_direction = 0.0
        self.mode_count = 5
        self.wave_periods = np.linspace(3.2409, 17.8155, 82)
        self.wave_frequencies = (2 * pi) / self.wave_periods
        self.wave_height = 1.0
        self.rigid_dofs = 0

        # Tube material constants
        self.viscous_damping_parameter = 8 * pi * 1e-6
        self.thickness = 0.1
        self.tube_density = 532.6
        self.wall_stiffness = 9e5
        self.material_damping_coefficient = 17.8e3 # {Pa * s}, also called B_{vis}
        self.fiber_pretension = 3.8e4  # {N} From Energies 2020 paper doi:10.3390/en13205499
        self.mooring_stiffness = 510.0e3    # Froude scaled by a factor of 10 from the original value of 510.0 N/m in
                                            # Journal of Fluids and Structures 2017 paper doi.org/10.1016/j.jfluidstructs.2017.06.003

        # Initialize PTO damping value B_{PTO}
        self.power_take_off_damping = 0.0

        # Dependent geometry variables
        # Variable notation: A, V, M
        self.cross_sectional_area = pi * (self.static_radius ** 2)
        self.displaced_volume = pi * (self.static_radius ** 2) * self.length
        self.tube_mass = (2 * pi * self.static_radius * self.thickness * self.length) * self.tube_density
        # TODO: add 2 * towhead masses if needed

        # Dependent miscellaneous variables
        # Variable notation: [-L/2, L/2], D, B_{mat}, eta
        self.integration_bounds = [-self.length / 2, self.length / 2]
        self.distensibility = (self.static_radius) * (self.thickness * self.cross_sectional_area * self.wall_stiffness)
        self.wall_damping = (1 / (2 * self.cross_sectional_area)) * self.material_damping_coefficient
        self.dissipation_coefficient = ((self.thickness * self.cross_sectional_area) / (self.rho * self.static_radius)) \
            * (self.power_take_off_damping + self.wall_damping)

        # Dependent inertia variables
        self.displaced_mass = self.rho * self.displaced_volume
        self.rotational_inertia_x_axis = (1 / 2) * self.displaced_mass * (self.static_radius ** 2)
        self.rotational_inertia_y_axis = (1 / 12) * self.displaced_mass * (3 * (self.static_radius ** 2) + (self.length ** 2))
        self.rotational_inertia_z_axis = self.rotational_inertia_y_axis
    
    def load_environmental_data(self):
        import scipy.io
        import scipy.interpolate
        from math import pi

        # Load and extract environmental data
        wave_data = scipy.io.loadmat(r'wec_design_optimization/elastic_tube/period_probability_distribution.mat')
        wave_periods = np.array(wave_data['Ta'][0])
        wave_probabilities = 0.01 * np.array(wave_data['Pa'][0])

        # Set up interpolation of probability function
        wave_period_probability_function = scipy.interpolate.interp1d(wave_periods, wave_probabilities, bounds_error=False, fill_value=0.0)
        frequency_probabilities = np.zeros_like(self.wave_frequencies)
        k = 0
        for omega in self.wave_frequencies:
            frequency_probabilities[k] = wave_period_probability_function((2 * pi) / omega)
            k += 1

        self.frequency_probabilities = frequency_probabilities
        #print('\nTotal wave resource seen = {:.3f}%\n'.format(100*np.sum(frequency_probabilities)))

    def objective_function(self):

        power_mean = np.sum(self.power_take_off_power_mean_power * self.frequency_probabilities)
        power_variance = np.sum(self.frequency_probabilities * (self.power_take_off_power_mean_power - power_mean) ** 2)
        power_standard_deviation = power_variance ** (1/2)

        self.power_mean = power_mean
        self.power_standard_deviation = power_standard_deviation

        # TODO: rename self.power_take_off_power_mean_power
        return - 1.0 * power_mean.data

    def generate_tube(self):
        """Generates an elastic tube mesh with all attached rigid body (if relevant) and modal degrees of freedom

        Args:
            None

        Returns:
            tube (an instance of a Capytaine FloatingBody)

        """
        print('\tGenerating tube.')
        tube = cpt.HorizontalCylinder(
            radius=self.static_radius, length=self.length, center=(0, 0, self.submergence),
            nx=int(self.length), ntheta=20, nr=int(self.static_radius / 0.2), clever=False)
        tube.keep_immersed_part()

        # Add relevant rigid DOFs
        #tube.add_all_rigid_body_dofs()
        #tube.add_translation_dof(name='Surge')

        # Add all elastic mode DOFs
        for k in range(self.mode_count):
            key_name = 'Bulge Mode ' + str(k)
            tube.dofs[key_name] = np.array([self.mode_shape_derivatives(x, y, z, mode_number=k) for x, y, z in tube.mesh.faces_centers])    

        tube.mass = tube.add_dofs_labels_to_matrix(self.mass_matrix())
        tube.dissipation = tube.add_dofs_labels_to_matrix(self.damping_matrix())
        tube.hydrostatic_stiffness = tube.add_dofs_labels_to_matrix(self.stiffness_matrix())

        self.tube = tube

    def solve_tube_hydrodynamics(self):
        """

        """
        print('\tSolving tube hydrodynamics.')

        solver = cpt.BEMSolver()
        problems = [cpt.RadiationProblem(omega=omega, body=self.tube, radiating_dof=dof) for dof in self.tube.dofs for omega in self.wave_frequencies]
        problems += [cpt.DiffractionProblem(omega=omega, body=self.tube, wave_direction=self.wave_direction) for omega in self.wave_frequencies]
        results = solver.solve_all(problems, keep_details=False)
        result_data = cpt.assemble_dataset(results)
        
        if self.save_results:
            from capytaine.io.xarray import separate_complex_values
            separate_complex_values(result_data).to_netcdf('flexible_tube_results.nc',
                                    encoding={'radiating_dof': {'dtype': 'U'},
                                    'influenced_dof': {'dtype': 'U'}}
                                    )

        self.result_data = result_data

    def load_data():
        import xarray

        # Load saved dataset and return the data
        results_data = cpt.io.xarray.merge_complex_values(xarray.open_dataset('flexible_tube_results.nc'))

        return results_data

    def evaluate_tube_modal_response_amplitudes(self):
        
        modal_response_amplitude_data = cpt.post_pro.rao(self.result_data, wave_direction=self.wave_direction, dissipation=self.tube.dissipation)
        self.modal_response_amplitude_data = modal_response_amplitude_data

        return

    def optimize_damping(self):
        from scipy.optimize import minimize_scalar

        print('\tOptimizing damping value.')
        damping_optimization_result = minimize_scalar(self._optimal_damping, method='golden')
        optimal_damping = damping_optimization_result.x
        optimal_objective_function = damping_optimization_result.fun

        self.optimal_damping_value = optimal_damping

        return optimal_damping, optimal_objective_function

    def _optimal_damping(self, power_take_off_damping):
        self.power_take_off_damping = power_take_off_damping

        # Update dissipation value
        self.dissipation_coefficient = ((self.thickness * self.cross_sectional_area) / (self.rho * self.static_radius)) \
            * (self.power_take_off_damping + self.wall_damping)
        
        # Update dissipation matrix in xarray format
        damping_matrix = self.rho * self.cross_sectional_area * self.dissipation_coefficient * self.wall_damping_matrix \
            + self.rho * self.viscous_damping_parameter * self.inner_flow_damping_matrix
        self.tube.dissipation.data = damping_matrix

        # Update modal RAOs
        self.evaluate_tube_modal_response_amplitudes()

        # Calculate power spectrum
        self.evaluate_dissipated_power()

        # Calculate objective function value
        objective_function_value = self.objective_function()

        return objective_function_value


    def evaluate_dissipated_power(self):
        """Calculates the mean power dissipated by the material as a function of wave frequency

        Args:
            None
        
        Returns:
            material_mean_power_dissipation (1d np array)

        """
        modal_response_amplitudes = self.wave_height * self.modal_response_amplitude_data
        total_damping_response = 0
        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                index_1 = 'Bulge Mode ' + str(k1)
                index_2 = 'Bulge Mode ' + str(k2)
                total_damping_response += (modal_response_amplitudes.sel(radiating_dof=index_1) * np.conjugate(modal_response_amplitudes.sel(radiating_dof=index_2))).real \
                                            * self.wall_damping_matrix[self.rigid_dofs+k1][self.rigid_dofs+k2]

        material_mean_power_dissipation = (1 / 2) * self.rho * self.cross_sectional_area * self.dissipation_coefficient * self.wave_frequencies * total_damping_response
        power_take_off_power_mean_power = (self.power_take_off_damping / (self.power_take_off_damping + self.wall_damping)) \
                                            * material_mean_power_dissipation
        self.material_mean_total_power_dissipation = material_mean_power_dissipation
        self.power_take_off_power_mean_power = power_take_off_power_mean_power


    def mode_shapes(self, x, mode_number):
        # Define chi(x)
        # k1 is lowercase k in A Barbarit et al.
        # k2 is uppercase K in A Barbarit et al.
        from math import sin, cos, tan, sinh, cosh, tanh, sqrt

        # Find, modal frequency and modal wavenumbers
        modal_frequency = self.mode_frequency_list[mode_number]
        k1 = self.mode_lower_wavenumber_list[mode_number]
        k2 = self.mode_upper_wavenumber_list[mode_number]
        try:
            normalization_factor = self.normalization_factor_matrix[mode_number]
        except:
            normalization_factor = 1
        
        # Mode type 1 as a function of x
        if modal_frequency in self.mode_type_1_frequency_list:
            c1 = tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
            c2 = tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
            chi = c1 * sin(k1 * x) - c2 * sinh(k2 * x)
        # Mode type 2 as a function of x
        elif modal_frequency in self.mode_type_2_frequency_list:
            c3 = k2 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
            c4 = k1 * tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
            chi = c3 * cos(k1 * x) + c4 * cosh(k2 * x)
        chi = chi / sqrt(normalization_factor)

        return chi

    def mode_shape_derivatives(self, x, y, z, mode_number, integration_flag=False, plot_flag=False):
        # Defines del{chi}/del{x}(x)
        # k1 is lowercase k in A Barbarit et al.
        # k2 is uppercase K in A Barbarit et al.
        from math import sin, cos, tan, sinh, cosh, tanh, sqrt

        modal_frequency = self.mode_frequency_list[mode_number]
        k1 = self.mode_lower_wavenumber_list[mode_number]
        k2 = self.mode_upper_wavenumber_list[mode_number]
        try:
            normalization_factor = self.normalization_factor_matrix[mode_number]
        except:
            normalization_factor = 1
        
        # Mode type 1 as a function of x
        if modal_frequency in self.mode_type_1_frequency_list:
            c1 = k1 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
            c2 = k2 * tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
            chi_dx = c1 * cos(k1 * x) - c2 * cosh(k2 * x)
        # Mode type 2 as a function of x
        elif modal_frequency in self.mode_type_2_frequency_list:
            c3 = k1 * k2 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
            c4 = k1 * k2 * tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
            chi_dx = -c3 * sin(k1 * x) + c4 * sinh(k2 * x)

        chi_dx = chi_dx / sqrt(normalization_factor)

        radial_deformation = (-self.static_radius / 2) * chi_dx

        if integration_flag:
            return chi_dx
        if plot_flag:
            return radial_deformation

        u = 0.0
        v = (y / self.static_radius) * radial_deformation
        w = ((z - self.submergence) / self.static_radius) * radial_deformation

        return (u, v, w)

    def normalize_mode_shapes(self):
        """Defines normalization factors for each modal frequency

        Args:
            None

        Returns:
            None

        """
        from scipy.integrate import quad

        normalization_factor_matrix = np.zeros(shape=self.mode_count)
        for k in range(self.mode_count):
            modal_product_integration = quad(func=self._mode_shape_product, a=self.integration_bounds[0], b=self.integration_bounds[1], args=(k, k))[0]
            normalization_factor_matrix[k] = (1 / self.length) * modal_product_integration + (self.tube_mass / self.displaced_mass) \
                    * self.mode_shapes(self.integration_bounds[1], k) * self.mode_shapes(self.integration_bounds[1], k)
        
        self.normalization_factor_matrix = normalization_factor_matrix

    def mass_matrix(self):
        """Defines an (6+n) x (6+n) mass matrix for the tube, where n is the modal degrees of freedom

        Args:
            None

        Returns:
            mass_matrix (2d np array)

        """
        rigid_body_mass_matrix = np.array([self.displaced_mass, self.displaced_mass, self.displaced_mass,
                                            self.rotational_inertia_x_axis,
                                            self.rotational_inertia_y_axis,
                                            self.rotational_inertia_z_axis
                                            ])
        rigid_body_mass_matrix = rigid_body_mass_matrix[0:self.rigid_dofs]
        modal_mass_matrix = self.displaced_mass * np.ones(shape=self.mode_count)
        mass_matrix = np.concatenate((rigid_body_mass_matrix, modal_mass_matrix), axis=0)
        mass_matrix = np.diag(mass_matrix)

        return mass_matrix

    def damping_matrix(self):
        """Defines an (6+n) x (6+n) damping matrix for the tube's modal degrees of freedom

        Args:
            None
        
        Returns:
            damping matrix (2d np array)

        """
        from scipy.integrate import quad

        wall_damping_matrix = np.zeros(shape=(self.rigid_dofs + self.mode_count, self.rigid_dofs + self.mode_count))
        inner_flow_damping_matrix = np.zeros(shape=(self.rigid_dofs + self.mode_count, self.rigid_dofs + self.mode_count))

        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                wall_damping_matrix[self.rigid_dofs+k1][self.rigid_dofs+k2] = quad(func=self._mode_shape_derivative_product, a=self.integration_bounds[0], b=self.integration_bounds[1], args=(k1, k2))[0]

        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                inner_flow_damping_matrix[self.rigid_dofs+k1][self.rigid_dofs+k2] = quad(func=self._mode_shape_product, a=self.integration_bounds[0], b=self.integration_bounds[1], args=(k1, k2))[0]

        damping_matrix = self.rho * self.cross_sectional_area * self.dissipation_coefficient * wall_damping_matrix \
            + self.rho * self.viscous_damping_parameter * inner_flow_damping_matrix

        self.total_damping_matrix = damping_matrix
        self.wall_damping_matrix = wall_damping_matrix
        self.inner_flow_damping_matrix = inner_flow_damping_matrix

        return damping_matrix

    def stiffness_matrix(self):
        """Defines an (6+n) x (6+n) stiffness matrix for the tube's modal degrees of freedom

        Args:
            None
        
        Returns:
            stiffness matrix (2d np array)

        """
        # Generate diagonal stiffness matrix
        # The rigid body matrices do not really matter here for assessing power since we are only evaluating the power
        # generation due to modal response amplitudes. Adding the surge mode only appears to have a minimal impact on 
        # assessing a design's produced power.
        rigid_body_stiffness_matrix = np.zeros(shape=self.rigid_dofs)
        #rigid_body_stiffness_matrix[0] = 2 * self.mooring_stiffness
        #rigid_body_stiffness_matrix[1:5] = 0
        modal_stiffness_matrix = self.displaced_mass * (self.mode_frequency_list ** 2)

        stiffness_matrix = np.concatenate((rigid_body_stiffness_matrix, modal_stiffness_matrix))
        stiffness_matrix = np.diag(stiffness_matrix)

        return stiffness_matrix

    def evaluate_modal_frequency_information(self):
        """Gathers information for both types of mode shapes by getting all modal frequencies in the bounds of self.omega_range.
        Finds only the lowest user specified integer number of modes.

        Args:
            None

        Returns:
            None

        """
        # Find all exact modal frequencies for each type of mode
        mode_type_1_frequency_list = self._calculate_dispersion_roots(self._mode_type_1__boundary_conditions)
        mode_type_2_frequency_list = self._calculate_dispersion_roots(self._mode_type_2__boundary_conditions)

        # Limit the number of modes to the user specified integer number by sorting all of them
        unsorted_mode_list = np.concatenate((mode_type_1_frequency_list, mode_type_2_frequency_list))
        sorted_mode_list = np.sort(unsorted_mode_list)

        # Error catching if less modes exist than modeled
        if len(sorted_mode_list) < self.mode_count:
            self.mode_count = len(sorted_mode_list)

        maximum_modal_frequency = sorted_mode_list[self.mode_count-1]
        mode_type_1_frequency_list = mode_type_1_frequency_list[np.where(mode_type_1_frequency_list <= maximum_modal_frequency)]
        mode_type_2_frequency_list = mode_type_2_frequency_list[np.where(mode_type_2_frequency_list <= maximum_modal_frequency)]

        # Find wavenumbers for each modal frequency
        mode_type_1_lower_wavenumber_list = np.zeros_like(mode_type_1_frequency_list)
        mode_type_1_upper_wavenumber_list = np.zeros_like(mode_type_1_frequency_list)
        mode_type_2_lower_wavenumber_list = np.zeros_like(mode_type_2_frequency_list)
        mode_type_2_upper_wavenumber_list = np.zeros_like(mode_type_2_frequency_list)

        for k in range(len(mode_type_1_frequency_list)):
            mode_type_1_lower_wavenumber_list[k], mode_type_1_upper_wavenumber_list[k] = self._mode_type_1__boundary_conditions(mode_type_1_frequency_list[k], wavenumber_flag=True)
        for k in range(len(mode_type_2_frequency_list)):
            mode_type_2_lower_wavenumber_list[k], mode_type_2_upper_wavenumber_list[k] = self._mode_type_2__boundary_conditions(mode_type_2_frequency_list[k], wavenumber_flag=True)

        # Stores frequency and wavenumber lists for instance
        self.mode_type_1_frequency_list = mode_type_1_frequency_list
        self.mode_type_2_frequency_list = mode_type_2_frequency_list
        self.mode_frequency_list = np.concatenate((mode_type_1_frequency_list, mode_type_2_frequency_list))
        self.mode_lower_wavenumber_list = np.concatenate((mode_type_1_lower_wavenumber_list, mode_type_2_lower_wavenumber_list))
        self.mode_upper_wavenumber_list = np.concatenate((mode_type_1_upper_wavenumber_list, mode_type_2_upper_wavenumber_list))

        return

    def _calculate_dispersion_roots(self, function_name, eps=1e-3, reltol=1e-3):
        """Calculates the roots of the nonlinear dispersion relationship governing the elastic tube
        
        Args:
            function_name (callable): the dispersion function used to find modal frequencies w and wavenumbers k and K
            eps (float): all found frequencies need be greater than eps to avoid finding zero  # TODO: consider changing to self.wavefrequencies[0]
            reltol (float): all numerically found frequencies from each starting points need to (100 * reltol) percent different 
                            from all found frequencies so far

        Returns:
            modal_frequency_array (np array): row of tube modal frequencies found from the dispersion relationship;
                                                size is the number of modes mode_count
        """
        from math import pi
        import scipy.optimize

        # Calculate values of the dispersion relationship whose roots correspond to modal frequencies of the tube.
        discretized_wave_frequencies = np.linspace(eps, 2 * pi, 1000)
        dispersion_function = np.zeros_like(discretized_wave_frequencies)
        k = 0
        for frequency in discretized_wave_frequencies:
            dispersion_function[k] = function_name(frequency)
            k += 1

        # Approximate roots by looking for changes in function sign. This is an appropriate method for this problem because all of the
        # dispersion roots come from where the tan() function factor changes sign instead of something like a quadratic function with one root.
        approximate_modal_frequency_list = np.asarray([])
        exact_modal_frequency_list = np.asarray([])
        for k in range(len(discretized_wave_frequencies) - 1):
            if dispersion_function[k] * dispersion_function[k+1] <= 0.0:
                approximate_modal_frequency_list = np.append(approximate_modal_frequency_list, discretized_wave_frequencies[k])

        # Use approximate roots as starting points for finding each actual root. Only accept a new root if it is both new and non-zero
        for approximate_modal_frequency in approximate_modal_frequency_list:
            exact_modal_frequency = scipy.optimize.fsolve(func=function_name, x0=approximate_modal_frequency)[0]
            if exact_modal_frequency > eps and not np.any(abs(exact_modal_frequency_list - exact_modal_frequency) / exact_modal_frequency < reltol):
                exact_modal_frequency_list = np.append(exact_modal_frequency_list, exact_modal_frequency)

        return exact_modal_frequency_list

    def _mode_shape_product(self, x, index_1, index_2):
        return self.mode_shapes(x, mode_number=index_1) \
            * self.mode_shapes(x, mode_number=index_2)

    def _mode_shape_derivative_product(self, x, index_1, index_2):
        from math import nan
        return self.mode_shape_derivatives(x, y=nan, z=nan, mode_number=index_1, integration_flag=True) \
            * self.mode_shape_derivatives(x, y=nan, z=nan, mode_number=index_2, integration_flag=True)
    

    def save_hydrodynamic_result_figures(self):
        import matplotlib.pyplot as plt

        plt.figure()
        for dof in self.tube.dofs:
            if dof.startswith('Bulge Mode'):
                plt.plot(
                    self.wave_frequencies,
                    self.result_data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
                    label=dof
                )
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('Added Mass')
        plt.legend()
        plt.savefig('added_mass.png', bbox_inches='tight')


        plt.figure()
        for dof in self.tube.dofs:
            if dof.startswith('Bulge Mode'):
                plt.plot(
                    self.wave_frequencies,
                    self.result_data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
                    label=dof
                )
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('Radiation damping')
        plt.legend()
        plt.savefig('radiation_damping.png', bbox_inches='tight')

        plt.figure()
        plt.plot(
            self.wave_frequencies, 
            self.power_take_off_power_mean_power,
            label='PTO Power Dissipation')
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('$P(\omega)$')
        plt.legend()
        plt.savefig('dissipated_power.png', bbox_inches='tight')

        plt.figure()
        for dof in self.tube.dofs:
            if dof.startswith('Bulge Mode'):
                plt.plot(
                    self.wave_frequencies,
                    np.abs(self.modal_response_amplitude_data.sel(radiating_dof=dof)).data,
                    label=dof
            )
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('RAO')
        plt.legend()
        plt.savefig('response_amplitude_operator.png', bbox_inches='tight')

    # From Babarit et al. 2017. Modal frequencies are the zeroes of both boundary condition functions.
    # Note that due to the tan() components of each function, roots showing up on their graphs 
    # may only be discontinuities instead of actual roots.
    def _mode_type_1__boundary_conditions(self, w, wavenumber_flag=False):
        from math import sqrt, tanh, tan, pi

        di = self.distensibility
        ts = self.fiber_pretension

        lowercase_wavenumber_1 = sqrt(((2 * pi) / (di * ts)) * (sqrt(1 + (ts * self.rho * (di ** 2) * (w ** 2) / pi)) - 1))
        uppercase_wavenumber_1 = sqrt(((2 * pi) / (di * ts)) * (sqrt(1 + (ts * self.rho * (di ** 2) * (w ** 2) / pi)) + 1))

        if wavenumber_flag:
            return lowercase_wavenumber_1, uppercase_wavenumber_1

        return (lowercase_wavenumber_1 * self.length / 2) * tanh(uppercase_wavenumber_1 * self.length / 2) \
                - (uppercase_wavenumber_1 * self.length / 2) * tan(lowercase_wavenumber_1 * self.length / 2)

    def _mode_type_2__boundary_conditions(self, w, wavenumber_flag=False):
        from math import sqrt, tanh, tan, pi
        
        di = self.distensibility
        ts = self.fiber_pretension      

        lowercase_wavenumber_2 = sqrt(((2 * pi) / (di * ts)) * (sqrt(1 + (ts * self.rho * (di ** 2) * (w ** 2) / pi)) - 1))
        uppercase_wavenumber_2 = sqrt(((2 * pi) / (di * ts)) * (sqrt(1 + (ts * self.rho * (di ** 2) * (w ** 2) / pi)) + 1))

        if wavenumber_flag:
            return lowercase_wavenumber_2, uppercase_wavenumber_2

        return (uppercase_wavenumber_2 * self.length / 2) * tanh(uppercase_wavenumber_2 * self.length / 2) \
                + (lowercase_wavenumber_2 * self.length / 2) * tan(lowercase_wavenumber_2 * self.length / 2) \
                - (((w ** 2) * self.rho * self.cross_sectional_area * self.length) / (-self.tube_mass * (w ** 2) + 2 * self.mooring_stiffness)) \
                * (uppercase_wavenumber_2 / lowercase_wavenumber_2 + lowercase_wavenumber_2 / uppercase_wavenumber_2) \
                * tanh(uppercase_wavenumber_2 * self.length / 2) * tan(lowercase_wavenumber_2 * self.length / 2)

    def plot_mode_shapes(self):
        import matplotlib.pyplot as plt
        from math import nan

        plt.figure()
        for mode_number in range(self.mode_count):
            tube_x = np.linspace(self.integration_bounds[0], self.integration_bounds[1], 250)
            tube_radial_deformation = np.zeros_like(tube_x)

            k = 0
            for x in tube_x:
                tube_radial_deformation[k] = self.mode_shape_derivatives(x, y=nan, z=nan, mode_number=mode_number, plot_flag=True)
                k += 1
            plt.plot(tube_x, tube_radial_deformation, label='Mode {} ($\omega = {:.3f}$ rad/s)'.format(mode_number, self.mode_frequency_list[mode_number]))
        
        plt.xlabel('$x (m)$')
        plt.ylabel('$\delta r (m)$')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig('tube_mode_shapes.png', bbox_inches='tight')

        return

    def plot_dissipated_power_statistics(self):
        import matplotlib.pyplot as plt

        damping_values = np.linspace(0, 3e6, 200)
        power_mean_values = np.zeros_like(damping_values)
        power_standard_deviation_values = np.zeros_like(damping_values)

        k = 0
        for b in damping_values:
            self._optimal_damping(b)
            power_mean_values[k] = self.power_mean
            power_standard_deviation_values[k] = self.power_standard_deviation
            k += 1

        plt.figure()
        plt.plot(0.001 * damping_values, 0.001 * power_mean_values, label='Mean Power')
        plt.plot(0.001 * damping_values, 0.001 * power_standard_deviation_values, label='Standard Deviation in Power')
        plt.xlabel('Power Take Off Damping Value (kPa $ \cdot $ s /m$^2$)')
        plt.ylabel('Dissipated PTO Power [kW]')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(0.001 * damping_values, 0.001 * power_mean_values)
        plt.xlabel('Power Take Off Damping Value (kPa $ \cdot $ s /m$^2$)', fontsize=14)
        plt.ylabel('Dissipated PTO Power (kW)', fontsize=14)

        self._optimal_damping(self.optimal_damping_value)
        plt.vlines(x=1e-3 * self.optimal_damping_value, ymin=0, ymax=0.001 * self.power_mean, linestyles='dashed')
        plt.scatter([1e-3*self.optimal_damping_value], [0.001*self.power_mean], marker='*', s=150)
        plt.xlim((0, 3000))
        plt.ylim((0, 12))

        plt.show()