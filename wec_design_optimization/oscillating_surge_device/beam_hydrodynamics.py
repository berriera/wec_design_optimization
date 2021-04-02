import capytaine as cpt
import numpy as np

def solve_beam(design_geometry, logging, save_results):
    # Unpack design geometry
    submerged_height = design_geometry[0]
    width = design_geometry[1]
    thickness= design_geometry[2]

    # Create and run device simulation
    beam_instance = FlexibleBeamHydrodynamicsModel(submerged_height=submerged_height, width=width, thickness=thickness, 
                                                    logging=logging)
    beam_instance.generate_beam()
    #beam_instance.animate_beam()
    beam_data = beam_instance.solve_beam_hydrodynamics(save_results=save_results)
    return beam_data

class FlexibleBeamHydrodynamicsModel(object):

    def __init__(self, submerged_height, width, thickness, logging=False):
        from math import pi

        # Constant wave conditions
        self.water_depth = -submerged_height
        self.wave_frequencies = np.linspace((2 * pi) / 20, (2 * pi / 5), 50)
        self.wave_direction = 0.0

        # Constant beam modal properties
        self.eigenvalue_list = np.array([1.8751, 4.6941, 7.8548, 10.9955])
        self.mode_count = 1

        # Unpack independent design variables
        self.height = submerged_height
        self.width = width
        self.thickness = thickness

        # Dependent geometry variables
        self.z_center = self.water_depth + self.height / 2

        # Initialize logger
        if logging:
            import logging
            logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

    def generate_beam(self):
        # Create mesh. The height is extended by 1 m in each direction for trimming at the surface
        beam = cpt.RectangularParallelepiped(
            size=(self.thickness, self.width, self.height + 2),
            resolution=(int(4*self.thickness), int(4*self.width), int(4*(self.height + 2))),
            center = (0.0, 0.0, self.z_center)
            )
        
        # Add custom defined pitch degree of freedom about constrained axis
        pitch_axis = cpt.Axis()
        pitch_axis.point = np.array([0.0, 0.0, self.water_depth])
        pitch_axis.vector = np.array([0.0, 1.0, 0.0])
        #beam.add_rotation_dof(name='Pitch', axis=pitch_axis)

        # Add flexible degrees of freedom
        for k in range(self.mode_count):
                key_name = 'Flexure Mode ' + str(k)
                beam.dofs[key_name] = np.array([self._cantilever_beam_flexure_mode_shape(x, y, z, mode_number=k)
                                        for x, y, z in beam.mesh.faces_centers])

        # Trim mesh at free surface and sea floor
        beam.keep_immersed_part(sea_bottom=self.water_depth)

        # Store mesh object
        self.beam_mesh = beam

        return beam

    def _cantilever_beam_flexure_mode_shape(self, x, y, z, mode_number, plot_flag=False):
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


    def solve_beam_hydrodynamics(self, save_results):
        """

        """
        bem_solver = cpt.BEMSolver()
        problems = [cpt.RadiationProblem(sea_bottom=self.water_depth, body=self.beam_mesh, 
                                        radiating_dof=dof, omega=omega)
                                        for dof in self.beam_mesh.dofs for omega in self.wave_frequencies]
        problems += [cpt.DiffractionProblem(sea_bottom=self.water_depth, body=self.beam_mesh,
                                        wave_direction=self.wave_direction, omega=omega)
                                        for omega in self.wave_frequencies]
        results = [bem_solver.solve(problem) for problem in problems]
        result_data = cpt.assemble_dataset(results)

        if save_results:
            from capytaine.io.xarray import separate_complex_values
            separate_complex_values(result_data).to_netcdf('beam_hydrodynamics_results.nc',
                                    encoding={'radiating_dof': {'dtype': 'U'},
                                    'influenced_dof': {'dtype': 'U'}}
                                    )

        return result_data

    def plot_mode_shapes(self):
        import matplotlib.pyplot as plt
        from math import nan

        plt.figure()
        for mode_number in range(self.mode_count):
            beam_q = np.linspace(0, 1, 250)
            beam_z = -self.height + beam_q * self.height
            beam_deformation = np.zeros_like(beam_z)

            k = 0
            for z in beam_z:
                beam_deformation[k] = self._cantilever_beam_flexure_mode_shape(x=nan, y=nan, z=z, mode_number=mode_number, plot_flag=True)
                k += 1
            plt.plot(beam_deformation, beam_q, label='Mode {} ($T = {:.3f}$ s)'
                    .format(mode_number, self.modal_period_list[mode_number]))
        
        plt.xlabel('$ q $')
        plt.ylabel('$ f(q) $')
        plt.xlim((-1, 1))
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.show()
        # plt.savefig('beam_mode_shapes.png', bbox_inches='tight')

        return

    def animate_beam(self, wave_period=6.0):
        #motion_dict = {}
        #for dof in self.beam_mesh.dofs:
        #    motion_dict[dof] = self.modal_response_amplitude_data  
        #    # TODO: go off nearest wave frequency or interpolate between two neighbors
        #animation = self.beam_mesh.animate(motion=motion_dict, loop_duration=wave_period)
        animation = self.beam_mesh.animate(motion={'Pitch': 0.0, 'Flexure Mode 0': 1.80}, loop_duration=6.0)
        animation.run()
        
        return

    def save_hydrodynamic_result_figures(self):
        import matplotlib.pyplot as plt

        plt.figure()
        for dof in self.beam_mesh.dofs:
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
        for dof in self.beam_mesh.dofs:
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


