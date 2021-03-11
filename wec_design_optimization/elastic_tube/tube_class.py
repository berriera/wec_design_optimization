import numpy as np
import capytaine as cpt
import logging

def evaluate_tube_design(design_variables):
        """Evaluates a complete tube design from start to finish

        """
        elastic_tube_instance = ElasticTube(design_variables)
        elastic_tube_instance.evaluate_modal_frequency_information()
        elastic_tube_instance.normalize_mode_shapes()
        elastic_tube_instance.generate_tube()
        elastic_tube_instance.evaluate_tube_modal_response_amplitudes()
        elastic_tube_instance.evaluate_dissipated_power()
        elastic_tube_instance.save_hydrodynamic_result_figures()

        objective_function_value = elastic_tube_instance.objective_function()

        return objective_function_value


class ElasticTube(object):


    def __init__(self, tube_design_variables):
        from math import inf, pi

        logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

        # Constants
        self.rho = 1025
        self.water_depth = -inf
        self.wave_direction = 0.0
        self.mode_count = 4
        self.viscous_damping_parameter = pi * 8e-6
        self.wave_frequencies = np.linspace(0.1, 5.0, 50)
        self.thickness = 0.01 # units: {m}
        self.fiber_pretension = 3.8e4  # From Energies 2020 paper doi:10.3390/en13205499
        self.mooring_stiffness = 510.0e3    # Froude scaled by a factor of 10 from the original value of 510.0 N/m in
                                            # Journal of Fluids and Structures 2017 paper doi.org/10.1016/j.jfluidstructs.2017.06.003

        # Unpack independent design variables
        self.static_radius = tube_design_variables[0]
        self.length = tube_design_variables[1]
        self.submergence = tube_design_variables[2]
        self.power_take_off_damping = tube_design_variables[3]

        # Dependent geometry variables
        self.cross_sectional_area = pi * (self.static_radius ** 2)
        self.displaced_volume = pi * (self.static_radius ** 2) * self.length
        self.system_mass = self.rho * self.displaced_volume  # TODO: add 2 * towhead masses

        # Dependent miscellaneous variables
        self.integration_bounds = [-self.length / 2, self.length / 2]
        self.distensibility = 2.248e-5  # TODO: make this dependent on elasticity and geometry design variables
        self.dissipation_coefficient = ((self.thickness * self.cross_sectional_area) / (self.rho * self.static_radius)) * self.power_take_off_damping

        # Dependent inertia variables
        self.displaced_mass = self.rho * self.displaced_volume
        self.rotational_inertia_x_axis = (1 / 2) * self.displaced_mass * (self.static_radius ** 2)
        self.rotational_inertia_y_axis = (1 / 12) * self.displaced_mass * (3 * (self.static_radius ** 2) + (self.length ** 2))
        self.rotational_inertia_z_axis = self.rotational_inertia_y_axis
    
    def objective_function(self):
        return np.sum(self.material_mean_power_dissipation) / self.displaced_volume

    def generate_tube(self):
        """Generates an elastic tube mesh with all attached rigid body (if relevant) and modal degrees of freedom

        Args:
            None

        Returns:
            tube (an instance of a Capytaine FloatingBody)

        """
        tube = cpt.HorizontalCylinder(
            radius=self.static_radius, length=self.length, center=(0, 0, self.submergence),
            nx=240, ntheta=20, nr=6, clever=False) # TODO: adjust nx, nr, clever args here
        tube.keep_immersed_part()

        # Add all rigid DOFs
        #tube.add_all_rigid_body_dofs()

        # Add all elastic mode DOFs
        for k in range(self.mode_count):
            key_name = 'Bulge Mode ' + str(k)
            tube.dofs[key_name] = np.array([self.mode_shape_derivatives(x, y, z, mode_number=k) for x, y, z in tube.mesh.faces_centers])    

        tube.mass = tube.add_dofs_labels_to_matrix(self.mass_matrix())
        tube.dissipation = tube.add_dofs_labels_to_matrix(self.damping_matrix())
        tube.hydrostatic_stiffness = tube.add_dofs_labels_to_matrix(self.stiffness_matrix())

        self.tube = tube

    def evaluate_tube_modal_response_amplitudes(self):
        """

        """
        solver = cpt.BEMSolver()
        problems = [cpt.RadiationProblem(omega=omega, body=self.tube, radiating_dof=dof) for dof in self.tube.dofs for omega in self.wave_frequencies]
        problems += [cpt.DiffractionProblem(omega=omega, body=self.tube, wave_direction=self.wave_direction) for omega in self.wave_frequencies]
        results = [solver.solve(problem) for problem in problems]
        result_data = cpt.assemble_dataset(results)
        modal_response_amplitude_data = cpt.post_pro.rao(result_data, wave_direction=self.wave_direction)

        self.result_data = result_data
        self.modal_response_amplitude_data = modal_response_amplitude_data

    def evaluate_dissipated_power(self):
        """Calculates the mean power dissipated by the material as a function of wave frequency

        Args:
            None
        
        Returns:
            material_mean_power_dissipation (1d np array)

        """
        modal_response_amplitudes = self.modal_response_amplitude_data
        total_damping_response = 0
        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                total_damping_response += (modal_response_amplitudes[:, k1] * np.conjugate(modal_response_amplitudes[:, k2])).real * self.wall_damping_matrix[k1][k2]
        material_mean_power_dissipation = (1 / 2) * self.rho * self.cross_sectional_area * self.dissipation_coefficient * self.wave_frequencies * total_damping_response

        self.material_mean_power_dissipation = material_mean_power_dissipation

    def mode_shapes(self, x, mode_number):
        # Define chi(x)
        # k1 is lowercase k in A Barbarit et al.
        # k2 is uppercase K in A Barbarit et al.
        from math import sin, cos, tan, sinh, cosh, tanh

        # Find, modal frequency and modal wavenumbers
        modal_frequency = self.mode_frequency_list[mode_number]
        k1 = self.mode_lower_wavenumber_list[mode_number]
        k2 = self.mode_upper_wavenumber_list[mode_number]
        
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

        return chi

    def mode_shape_derivatives(self, x, y, z, mode_number, integration_flag=False, plot_flag=False):
        # Defines del{chi}/del{x}(x)
        # k1 is lowercase k in A Barbarit et al.
        # k2 is uppercase K in A Barbarit et al.
        from math import sin, cos, tan, sinh, cosh, tanh

        modal_frequency = self.mode_frequency_list[mode_number]
        k1 = self.mode_lower_wavenumber_list[mode_number]
        k2 = self.mode_upper_wavenumber_list[mode_number]
        
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

        normalization_factor_matrix = np.zeros(shape=(self.mode_count, self.mode_count))
        for k1 in range(self.mode_count):
            for k2 in range(self.mode_count):
                modal_product_integration = quad(func=self._mode_shape_product, a=self.integration_bounds[0], b=self.integration_bounds[1], args=(k1, k2))[0]
                normalization_factor_matrix[k1][k2] = (1 / self.length) * modal_product_integration + (self.system_mass / self.displaced_mass) \
                    * self.mode_shapes(self.integration_bounds[1], k1) * self.mode_shapes(self.integration_bounds[1], k2)
        
        self.normalization_factor_matrix = normalization_factor_matrix

    def mass_matrix(self):
        """Defines an n x n mass matrix for the tube, where n is the modal degrees of freedom

        Args:
            None

        Returns:
            mass_matrix (2d np array)

        """
        mass_matrix = self.displaced_mass * np.ones(shape = self.mode_count)
        mass_matrix = np.diag(mass_matrix)
     
        #elif self.rigid_body_degrees_of_freedom ==  6:
        #        
        #    for k in [0, 1, 2]:
        #        mass_matrix[k][k] = self.displaced_mass
        #    mass_matrix[3][3] = self.rotational_inertia_x_axis
        #    mass_matrix[4][4] = self.rotational_inertia_y_axis
        #    mass_matrix[5][5] = self.rotational_inertia_z_axis        
        #
        #    for k in range(6, 6 + self.mode_count):
        #        mass_matrix[k][k] = self.displaced_mass

        return mass_matrix

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

        self.wall_damping_matrix = wall_damping_matrix
        return damping_matrix

    def stiffness_matrix(self):
        """Defines an n x n stiffness matrix for the tube's modal degrees of freedom

        Args:
            None
        
        Returns:
            stiffness matrix (2d np array)

        """
        stiffness_matrix = self.displaced_mass * np.diag(self.mode_frequency_list ** 2)

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
        import scipy.optimize

        # Calculate values of the dispersion relationship whose roots correspond to modal frequencies of the tube.
        discretized_wave_frequencies = np.linspace(self.wave_frequencies[0], self.wave_frequencies[-1], 50)
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

    def log_history(self):
        """Records all design variables and objective function values in saved text files
        
        """
        import time
        
        # TODO: save vars in a dictionary
        # TODO: save dictionary to .txt file

        pass

    def save_hydrodynamic_result_figures(self):
        import matplotlib.pyplot as plt

        plt.figure()
        for dof in self.tube.dofs:
            plt.plot(
                self.wave_frequencies,
                self.result_data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
                label=dof,
                marker='o'
            )
        plt.xlabel('$\omega$')
        plt.ylabel('Added Mass')
        plt.legend()
        plt.savefig('added_mass.png', bbox_inches='tight')


        plt.figure()
        for dof in self.tube.dofs:
            plt.plot(
                self.wave_frequencies,
                self.result_data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
                label=dof,
                marker='o'
            )
        plt.xlabel('$\omega$')
        plt.ylabel('Radiation damping')
        plt.legend()
        plt.savefig('radiation_damping.png', bbox_inches='tight')

        plt.figure()
        plt.plot(
            self.wave_frequencies,
            self.material_mean_power_dissipation,
            marker='o'
        )
        plt.xlabel('$\omega$')
        plt.ylabel('$P(\omega)$')
        plt.savefig('dissipated_power.png', bbox_inches='tight')

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
        plt.savefig('mode_shapes.png', bbox_inches='tight')

        return