import capytaine as cpt
import numpy as np

class FlexibleColumn(object):

    def __init__(self, radius, height, elastic_modulus, density, logging=False):
        from math import pi

        # Constant wave conditions
        self.water_depth = -height
        self.wave_periods = np.linspace(3, 20, 50)
        self.wave_direction = 0.0

        # Constant beam modal properties
        self.eigenvalue_list = np.array([1.8751, 4.6941, 7.8458, 10.9955])
        self.mode_count = 4

        # Unpack independent design variables
        self.radius = radius
        self.height = height
        self.elastic_modulus = elastic_modulus
        self.density = density

        # Initialize logger
        if logging:
            import logging
            logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

        # Dependent incident wave frequency
        self.wave_frequencies = 2 * pi / self.wave_periods

        # Dependent geometry variables
        self.cross_sectional_area = pi * (self.radius ** 2)
        self.volume = self.cross_sectional_area * self.height
        
        # Dependent inertia variables
        self.mass = self.density * self.volume
        self.area_y_moment_of_inertia = (pi / 4) * (self.radius ** 4)

        # Dependent modal values
        self.kappa_list = self.eigenvalue_list / self.height
        self.modal_frequency_list = np.sqrt(((self.elastic_modulus * self.area_y_moment_of_inertia) / (self.density * self.cross_sectional_area)) \
                                        * (self.kappa_list ** 4))
        self.modal_period_list = (2 * pi) / self.modal_frequency_list


    def generate_column(self):
        column = cpt.VerticalCylinder(length=self.height + 2, radius=self.radius, 
                                        center=(0, 0, -self.height / 2), 
                                        nx=50, ntheta=30, nr=8,
                                        clever=False)

        for k in range(self.mode_count):
                key_name = 'Flexure Mode ' + str(k)
                column.dofs[key_name] = np.array([self.cantilever_beam_flexure_mode_shape(x, y, z, mode_number=k)
                                        for x, y, z in column.mesh.faces_centers])    

        column.keep_immersed_part(sea_bottom=self.water_depth)

        column.mass = column.add_dofs_labels_to_matrix(self.mass_matrix())
        column.dissipation = column.add_dofs_labels_to_matrix(self.damping_matrix())
        column.hydrostatic_stiffness = column.add_dofs_labels_to_matrix(self.stiffness_matrix())

        self.column_mesh = column

    def cantilever_beam_flexure_mode_shape(self, x, y, z, mode_number, plot_flag=False):
        from math import cosh, sinh, cos, sin

        eigenvalue = self.eigenvalue_list[mode_number]

        # Normalize z coordinate value to q from 0 (sea bottom) to 1 (free surface)
        q = 1 + z / self.height

        c1 = (sinh(eigenvalue) - sin(eigenvalue)) / (cosh(eigenvalue) + cos(eigenvalue))
        end_displacement = cosh(eigenvalue) - cos(eigenvalue) - c1 * (sinh(eigenvalue) - sin(eigenvalue))

        u = (cosh(eigenvalue*q) - cos(eigenvalue*q) - c1*(sinh(eigenvalue*q) - sin(eigenvalue*q))) / end_displacement
        v = 0.0
        w = 0.0

        if plot_flag:
            return u

        return (u, v, w)


    def mass_matrix(self):
        modal_mass = 0.25 * self.mass
        mass_matrix = modal_mass * np.ones(shape = self.mode_count)
        mass_matrix = np.diag(mass_matrix)
        
        return mass_matrix


    def damping_matrix(self):
        damping_matrix = np.zeros(shape=(self.mode_count, self.mode_count))

        return damping_matrix


    def stiffness_matrix(self):
        modal_mass = 0.25 * self.mass
        stiffness_matrix = modal_mass * np.diag(self.modal_frequency_list ** 2)

        return stiffness_matrix

    def evaluate_column_modal_response_amplitudes(self):
        """

        """
        bem_solver = cpt.BEMSolver()
        problems = [cpt.RadiationProblem(sea_bottom=self.water_depth, body=self.column_mesh, 
                                        radiating_dof=dof, omega=omega)
                                        for dof in self.column_mesh.dofs for omega in self.wave_frequencies]
        problems += [cpt.DiffractionProblem(sea_bottom=self.water_depth, body=self.column_mesh,
                                        wave_direction=self.wave_direction, omega=omega)
                                        for omega in self.wave_frequencies]
        results = [bem_solver.solve(problem) for problem in problems]
        result_data = cpt.assemble_dataset(results)
        modal_response_amplitude_data = cpt.post_pro.rao(result_data, wave_direction=self.wave_direction)

        self.result_data = result_data
        self.modal_response_amplitude_data = modal_response_amplitude_data

    def plot_mode_shapes(self):
        import matplotlib.pyplot as plt
        from math import nan

        plt.figure()
        for mode_number in range(self.mode_count):
            column_q = np.linspace(0, 1, 250)
            column_z = -self.height + column_q * self.height
            column_deformation = np.zeros_like(column_z)

            k = 0
            for z in column_z:
                column_deformation[k] = self.cantilever_beam_flexure_mode_shape(x=nan, y=nan, z=z, mode_number=mode_number, plot_flag=True)
                k += 1
            plt.plot(column_deformation, column_q, label='Mode {} ($T = {:.3f}$ s)'
                    .format(mode_number, self.modal_period_list[mode_number]))
        
        plt.xlabel('$ q $')
        plt.ylabel('$ f(q) $')
        plt.xlim((-1, 1))
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig('column_mode_shapes.png', bbox_inches='tight')

        return

    def save_hydrodynamic_result_figures(self):
        import matplotlib.pyplot as plt

        plt.figure()
        for dof in self.column_mesh.dofs:
            plt.plot(
                self.wave_periods,
                self.result_data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
                label=dof,
                marker='o'
            )
        plt.xlabel('T (s)')
        plt.ylabel('Added Mass')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('added_mass.png', bbox_inches='tight')

        plt.figure()
        for dof in self.column_mesh.dofs:
            plt.plot(
                self.wave_periods,
                self.result_data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
                label=dof,
                marker='o'
            )
        plt.xlabel('T (s)')
        plt.ylabel('Radiation damping')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('radiation_damping.png', bbox_inches='tight')

        plt.figure()
        for dof in self.column_mesh.dofs:
            plt.plot(
                self.wave_periods, 
                np.abs(self.result_data['RAO'].sel(radiating_dof=dof).data),
                label=dof,
                marker='o'
            )
        plt.xlabel('T (s)')
        plt.ylabel('RAO')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('response_amplitude_operators.png', bbox_inches='tight')

    def animate_column(self, wave_period):
        motion_dict = {}
        for dof in self.column_mesh.dofs:
            motion_dict[dof] = self.modal_response_amplitude_data  
            # TODO: go off nearest wave frequency or interpolate between two neighbors
        animation = self.column_mesh.animate(motion=motion_dict, loop_duration=wave_period)
        animation.run()
