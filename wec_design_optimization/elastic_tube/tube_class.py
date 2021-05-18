from math import pi
import numpy as np
import capytaine as cpt
import logging

def evaluate_tube_design(design_variables, mode_count=5):
        """Evaluates a complete tube design from start to finish

        """
        # Return 0 power if the entire tube is above the free surface
        if design_variables[2] >= design_variables[0]:
            return 0.0

        elastic_tube_instance = ElasticTube(tube_design_variables=design_variables, mode_count=mode_count)

        # Solve for motion and optimize damping for optimal power
        elastic_tube_instance.solve_tube_hydrodynamics()
        damping_value, optimal_objective_function = elastic_tube_instance.optimize_damping()
        print('\tOptimal damping value = {:.2f}'.format(damping_value))

        # TODO: update to include variance from objective function instead of from .optimize_damping()

        return optimal_objective_function


class ElasticTube(object):


    def __init__(self, tube_design_variables, mode_count):
        from math import inf, pi
        import scipy.io

        #logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")
        self.save_results = True

        # Unpack independent design variables
        # Variable notation: r_s, L, z_s
        self.static_radius = tube_design_variables[0]
        self.length = tube_design_variables[1]
        self.submergence = tube_design_variables[2]

        # Unpack independent design variables for material optimization
        #self.wall_stiffness = tube_design_variables[0]
        #self.fiber_pretension = tube_design_variables[1]

        # Fixed geometry for material optimization
        #self.static_radius = 0.9
        #self.length = 60.0
        #self.submergence = -1.25

        # Environment and incident wave constants
        self.rho = 1025
        self.water_depth = -45
        self.wave_direction = 0.0
        self.mode_count = mode_count
        self.wave_periods = np.linspace(3.2409, 17.8155, 82)
        self.wave_frequencies = (2 * pi) / self.wave_periods
        self.wave_height = 1.0

        # Tube material constants
        self.viscous_damping_parameter = 8 * pi * 1e-6
        self.thickness = 0.04 * self.static_radius
        tube_density = 532.6
        self.wall_stiffness = 9e5
        self.material_damping_coefficient = 17.8e3 # {Pa * s}, also called B_{vis}
        self.fiber_pretension = 3.8e4  # {N} From Energies 2020 paper doi:10.3390/en13205499
        self.mooring_stiffness = 51.0e3  # Froude scaled by a factor of 10 from the original value of 510.0 N/m in
                                         # Journal of Fluids and Structures 2017 paper doi.org/10.1016/j.jfluidstructs.2017.06.003

        # Initialize PTO damping value B_{PTO}
        self.power_take_off_damping = 0.0

        # Dependent geometry and inertia variables
        # Variable notation: A, V, m, M
        self.cross_sectional_area = pi * (self.static_radius ** 2)
        self.total_volume = pi * (self.static_radius ** 2) * self.length
        self.total_mass = self.rho * self.total_volume  # Assuming each device is neutrally buoyant
        # self.inner_radius = self.static_radius - self.thickness
        # self.tube_area = pi * (self.static_radius ** 2 - self.inner_radius ** 2)
        # self.tube_mass = (self.tube_area * self.length) * tube_density
        # self.towhead_mass = 1e4
        # self.system_mass = self.tube_mass + 2*self.towhead_mass
        # Simplified tube and towhead mass
        self.system_mass = 0.15 * self.total_mass

        # Dependent miscellaneous variables
        # Variable notation: [-L/2, L/2], D, B_{mat}, eta
        self.integration_bounds = [-self.length / 2, self.length / 2]
        self.distensibility = 1 / (pi * self.static_radius * self.thickness * self.wall_stiffness)
        self.wall_damping = (1 / (2 * self.cross_sectional_area)) * self.material_damping_coefficient
        self.dissipation_coefficient = ((self.thickness * self.cross_sectional_area) / (self.rho * self.static_radius)) \
            * (self.power_take_off_damping + self.wall_damping)

        # Load environmental data
        wave_data = scipy.io.loadmat(r'wec_design_optimization/elastic_tube/period_probability_distribution.mat')
        self.frequency_probability_distribution = 0.01 * np.array(wave_data['Pa'][0])

        # Solve for goemetric modes and add them as custom degrees of freedom
        self.evaluate_modal_frequency_information()
        self.normalize_mode_shapes()
        self.generate_tube()


    def objective_function(self, power_spectrum):
        from math import asin

        # Adjust for submerged cicumference
        if self.submergence <= -self.static_radius:
            circumference_ratio = 1.0
        elif self.submergence >= self.static_radius:
            circumference_ratio = 0.0
        else:
            circumference_ratio = (pi - 2*asin(self.submergence / self.static_radius)) / (2*pi)

        geometry_weighted_power_spectrum = circumference_ratio * power_spectrum.data

        power_mean = np.sum(geometry_weighted_power_spectrum * self.frequency_probability_distribution)
        power_variance = np.sum(self.frequency_probability_distribution * (geometry_weighted_power_spectrum - power_mean) ** 2)
        power_standard_deviation = power_variance ** (1/2)

        self.power_mean = power_mean
        self.power_standard_deviation = power_standard_deviation

        return -1.0 * power_mean

    def generate_tube(self):
        """Generates an elastic tube mesh with all attached rigid body and modal degrees of freedom

        Args:
            None

        Returns:
            tube (an instance of a Capytaine FloatingBody)

        """
        print('\tGenerating tube.')
        tube = cpt.HorizontalCylinder(
            radius=self.static_radius, length=self.length, center=(0, 0, self.submergence),
            nx=int(1.25*self.length), ntheta=20, nr=int(5*self.static_radius), clever=False)
        tube.keep_immersed_part()

        # Add all elastic mode DOFs
        for k in range(self.mode_count):
            key_name = 'Bulge Mode ' + str(k)
            tube.dofs[key_name] = np.array([self.mode_shape_derivatives(x, y, z, mode_number=k) for x, y, z in tube.mesh.faces_centers])    

        modal_mass_matrix = self.total_mass * np.eye(N=self.mode_count)
        modal_stiffness_matrix = np.diag(self.total_mass * (self.mode_frequency_list ** 2))

        tube.mass = tube.add_dofs_labels_to_matrix(modal_mass_matrix)
        tube.hydrostatic_stiffness = tube.add_dofs_labels_to_matrix(modal_stiffness_matrix)

        self.tube = tube

        self.dissipation = tube.add_dofs_labels_to_matrix(self.damping_matrix())

    def damping_matrix(self):
        """Defines an n x n damping matrix for the tube's modal degrees of freedom

        Args:
            None
        
        Returns:
            damping matrix (2d np array)

        """
        from scipy.integrate import quad

        wall_damping_matrix = np.zeros(shape=(self.mode_count, self.mode_count))
        inner_flow_damping_matrix = np.zeros(shape=(self.mode_count, self.mode_count))

        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                wall_damping_matrix[k1][k2] = quad(func=self._mode_shape_derivative_product, a=self.integration_bounds[0], b=self.integration_bounds[1], args=(k1, k2))[0]

        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                inner_flow_damping_matrix[k1][k2] = quad(func=self._mode_shape_product, a=self.integration_bounds[0], b=self.integration_bounds[1], args=(k1, k2))[0]

        damping_matrix = self.rho * self.cross_sectional_area * self.dissipation_coefficient * wall_damping_matrix \
            + self.rho * self.viscous_damping_parameter * inner_flow_damping_matrix

        self.total_damping_matrix = damping_matrix
        self.wall_damping_matrix = wall_damping_matrix
        self.inner_flow_damping_matrix = inner_flow_damping_matrix

        return damping_matrix


    def solve_tube_hydrodynamics(self):
        """

        """
        print('\tSolving tube hydrodynamics.')

        solver = cpt.BEMSolver()
        problems = [cpt.RadiationProblem(omega=omega, 
                        body=self.tube, 
                        radiating_dof=dof, 
                        rho=self.rho, 
                        sea_bottom=self.water_depth)
                        for dof in self.tube.dofs for omega in self.wave_frequencies]
        problems += [cpt.DiffractionProblem(omega=omega, 
                        body=self.tube, 
                        wave_direction=self.wave_direction, 
                        rho=self.rho, 
                        sea_bottom=self.water_depth) 
                        for omega in self.wave_frequencies]
        results = solver.solve_all(problems, keep_details=False)
        result_data = cpt.assemble_dataset(results)
        
        if self.save_results:
            from capytaine.io.xarray import separate_complex_values
            save_file_name = 'flexible_tube_results__rs_le_zs__{}_{}_{}__with_{}_cells.nc'.format(self.static_radius, self.length, self.submergence, self.tube.mesh.nb_faces)
            separate_complex_values(result_data).to_netcdf(save_file_name,
                                    encoding={'radiating_dof': {'dtype': 'U'},
                                    'influenced_dof': {'dtype': 'U'}}
                                    )

        self.result_data = result_data


    def optimize_damping(self):
        from scipy.optimize import minimize_scalar

        print('\tOptimizing damping value.')
        damping_optimization_result = minimize_scalar(self._pto_damping_dissipated_power, method='golden')
        optimal_damping = damping_optimization_result.x
        optimal_power = damping_optimization_result.fun

        self.optimal_damping_value = optimal_damping

        return optimal_damping, optimal_power

    def _pto_damping_dissipated_power(self, power_take_off_damping):
        self.power_take_off_damping = power_take_off_damping

        # Update dissipation value
        self.dissipation_coefficient = ((self.thickness * self.cross_sectional_area) / (self.rho * self.static_radius)) \
            * (self.power_take_off_damping + self.wall_damping)
        
        # Update dissipation matrix in xarray format
        damping_matrix = self.rho * self.cross_sectional_area * self.dissipation_coefficient * self.wall_damping_matrix \
            + self.rho * self.viscous_damping_parameter * self.inner_flow_damping_matrix
        self.dissipation.data = damping_matrix

        # Update modal RAOs
        modal_response_amplitude_data = cpt.post_pro.rao(self.result_data,
                                                            wave_direction=self.wave_direction, 
                                                            dissipation=self.dissipation)

        # Calculate power spectrum
        dissipated_power_spectrum = self.evaluate_dissipated_power(modal_response_amplitude_data)

        # Calculate objective function value
        objective_function_value = self.objective_function(dissipated_power_spectrum)

        # Update variables for figures
        self.dissipated_power_spectrum = dissipated_power_spectrum
        self.modal_response_amplitude_data = modal_response_amplitude_data

        return objective_function_value


    def evaluate_dissipated_power(self, mode_response_dataset):
        """Calculates the mean power dissipated by the material as a function of wave frequency

        Args:
            None
        
        Returns:
            material_mean_power_dissipation (1d np array)

        """
        modal_response_amplitudes = self.wave_height * mode_response_dataset
        total_damping_response = 0
        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                index_1 = 'Bulge Mode ' + str(k1)
                index_2 = 'Bulge Mode ' + str(k2)
                total_damping_response += (modal_response_amplitudes.sel(radiating_dof=index_1) * np.conjugate(modal_response_amplitudes.sel(radiating_dof=index_2))).real \
                                            * self.wall_damping_matrix[k1][k2]

        material_mean_power_dissipation = (1 / 2) * self.rho * self.cross_sectional_area * self.dissipation_coefficient * (self.wave_frequencies ** 2) * total_damping_response
        power_take_off_power_spectrum = (self.power_take_off_damping / (self.power_take_off_damping + self.wall_damping)) \
                                            * material_mean_power_dissipation
        return power_take_off_power_spectrum


    def mode_shapes(self, x, mode_number):
        # Define chi(x)
        # k1 is lowercase k in A Barbarit et al.
        # k2 is uppercase K in A Barbarit et al.
        from math import sin, cos, tan, sinh, cosh, tanh, sqrt, exp

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
            try:
                c1 = tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c2 = tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
                chi = c1 * sin(k1 * x) - c2 * sinh(k2 * x)
            except OverflowError:
                c1 = tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c2 = tan(k1 * self.length / 2)
                if x >= 0:
                    chi = c1 * sin(k1 * x) - c2 * exp(k2 * (x - self.length / 2))
                else:
                    chi = c1 * sin(k1 * x) - c2 * -exp(k2 * (-x - self.length / 2))
        # Mode type 2 as a function of x
        elif modal_frequency in self.mode_type_2_frequency_list:
            try:
                c3 = k2 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c4 = k1 * tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
                chi = c3 * cos(k1 * x) + c4 * cosh(k2 * x)
            except OverflowError:
                c3 = k2 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c4 = k1 * tan(k1 * self.length / 2)
                if x >= 0:
                    chi = c3 * cos(k1 * x) + c4 * exp(k2 * (x - self.length / 2))
                else:
                    chi = c3 * cos(k1 * x) + c4 * exp(k2 * (-x - self.length / 2))
        chi = chi / sqrt(normalization_factor)

        return chi

    def mode_shape_derivatives(self, x, y, z, mode_number, integration_flag=False, plot_flag=False):
        # Defines del{chi}/del{x}(x)
        # k1 is lowercase k in A Barbarit et al.
        # k2 is uppercase K in A Barbarit et al.
        from math import sin, cos, tan, sinh, cosh, tanh, sqrt, exp

        modal_frequency = self.mode_frequency_list[mode_number]
        k1 = self.mode_lower_wavenumber_list[mode_number]
        k2 = self.mode_upper_wavenumber_list[mode_number]
        try:
            normalization_factor = self.normalization_factor_matrix[mode_number]
        except:
            normalization_factor = 1
        
        # Mode type 1 as a function of x
        if modal_frequency in self.mode_type_1_frequency_list:
            try:
                c1 = k1 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c2 = k2 * tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
                chi_dx = c1 * cos(k1 * x) - c2 * cosh(k2 * x)
            except OverflowError:
                c1 = k1 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c2 = k2 * tan(k1 * self.length / 2)
                if x >= 0:
                    chi_dx = c1 * cos(k1 * x) - c2 * exp(k2 * (x - self.length / 2))
                else:
                    chi_dx = c1 * cos(k1 * x) - c2 * exp(k2 * (-x - self.length / 2))
        # Mode type 2 as a function of x
        elif modal_frequency in self.mode_type_2_frequency_list:
            try:
                c3 = k1 * k2 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c4 = k1 * k2 * tan(k1 * self.length / 2) / cosh(k2 * self.length / 2)
                chi_dx = -c3 * sin(k1 * x) + c4 * sinh(k2 * x)
            except OverflowError:
                c3 = k1 * k2 * tanh(k2 * self.length / 2) / cos(k1 * self.length / 2)
                c4 = k1 * k2 * tan(k1 * self.length / 2)
                if x >= 0:
                    chi_dx = -c3 * sin(k1 * x) + c4 * exp(k2 * (x - self.length / 2))
                else:
                    chi_dx = -c3 * sin(k1 * x) + c4 * -exp(k2 * (-x - self.length / 2))

        chi_dx = chi_dx / sqrt(normalization_factor)

        radial_deformation = (-self.static_radius / 2) * chi_dx  # variable is delta_radius

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
            normalization_factor_matrix[k] = (1 / self.length) * modal_product_integration \
                + (self.system_mass / self.total_mass) * (self.mode_shapes(self.integration_bounds[1], k) ** 2)
        # Each factor is equal to N_i**2
        self.normalization_factor_matrix = normalization_factor_matrix
        

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

    def _calculate_dispersion_roots(self, function_name, eps=1e-4, reltol=1e-3):
        """Calculates the roots of the nonlinear dispersion relationship governing the elastic tube
        
        Args:
            function_name (callable): the dispersion function used to find modal frequencies w and wavenumbers k and K
            eps (float): all found frequencies need be greater than eps to avoid finding zero
            reltol (float): all numerically found frequencies from each starting points need to (100 * reltol) percent different 
                            from all found frequencies so far

        Returns:
            modal_frequency_array (np array): row of tube modal frequencies found from the dispersion relationship;
                                                size is the number of modes mode_count
        """
        from math import pi
        import scipy.optimize

        # Calculate values of the dispersion relationship whose roots correspond to modal frequencies of the tube.
        discretized_wave_frequencies = np.linspace(eps, 10 * pi, 63000)
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
                - (((w ** 2) * self.rho * self.cross_sectional_area * self.length) / (-self.system_mass * (w ** 2) + 2 * self.mooring_stiffness)) \
                * (uppercase_wavenumber_2 / lowercase_wavenumber_2 + lowercase_wavenumber_2 / uppercase_wavenumber_2) \
                * tanh(uppercase_wavenumber_2 * self.length / 2) * tan(lowercase_wavenumber_2 * self.length / 2)
