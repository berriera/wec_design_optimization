from beam_hydrodynamics import solve_beam

import capytaine as cpt
import numpy as np
import xarray

solve_hydro = False
solve_stress = True
beam_geometry = np.array([10.0, 8.0, 0.5])
elastic_modulus = 69e9
density = 2700

# Unpack design geometry list
submerged_height = beam_geometry[0]
width = beam_geometry[1]
thickness= beam_geometry[2]

def run_structural_model():
    structural_model = FlexibleBeamStructuralModel(submerged_height, width, thickness, elastic_modulus, density)
    hydro_results = structural_model.load_data()
    structural_model.add_matrices(hydro_results)
    structural_model.evaluate_response_amplitudes(hydro_results)
    structural_model.save_response_amplitude_figure()

if solve_hydro:
    # Create hydrodynamics results
    model_results = solve_beam(beam_geometry, logging=True, save_results=True)
if solve_stress:
    run_structural_model()

class FlexibleBeamStructuralModel(object):

    def __init__(self, height, width, thickness, elastic_modulus, density):
        from math import pi

        # Structural simulation constant
        self.mode_count = 1
        self.hydrodynamics_file_name = 'beam_hydrodynamic_results.nc'

        # Unpack material parameters
        self.elastic_modulus = elastic_modulus
        self.density = density

        # Dependent geometry values
        self.cross_sectional_area = width * thickness
        self.area_y_moment_of_inertia = (1/12) * width * (thickness ** 3)

        # Dependent inertia values
        self.volume = height * width * thickness
        self.mass = density * self.volume
        self.pitch_rotational_inertia = (1/3) * self.mass * (height ** 2)

        # Indenpendent modal values
        self.eigenvalue_list = np.array([1.8751, 4.6941, 7.8458, 10.9955])

        # Dependent modal values
        self.wavenumber_list = self.eigenvalue_list / self.height
        self.modal_frequency_list = np.sqrt(((self.elastic_modulus * self.area_y_moment_of_inertia) / (self.density * self.cross_sectional_area)) \
                                        * (self.wavenumber_list ** 4))
        self.modal_period_list = (2 * pi) / self.modal_frequency_list


    def load_data(self):
        from capytaine.io.xarray import merge_complex_values

        results_data = merge_complex_values(xarray.open_dataset(self.hydrodynamics_file_name))

        return results_data

    def add_matrices(results):
        results.mass = results.add_dofs_labels_to_matrix(FlexibleBeamStructuralModel._mass_matrix())
        results.dissipation = results.add_dofs_labels_to_matrix(FlexibleBeamStructuralModel._damping_matrix())
        results.hydrostatic_stiffness = results.add_dofs_labels_to_matrix(FlexibleBeamStructuralModel._stiffness_matrix())

        return

    def evaluate_response_amplitudes(results):
        response_amplitudes = cpt.post_pro.rao(results, wave_direction=0.0)

        return response_amplitudes

    def _mass_matrix(self):
        modal_mass = 0.25 * self.mass
        mass_matrix = modal_mass * np.ones(shape=self.mode_count)
        mass_matrix = np.diag(mass_matrix)
            
        return mass_matrix

    def _damping_matrix(self):
        damping_matrix = np.zeros(shape=(self.mode_count, self.mode_count))

        return damping_matrix

    def _stiffness_matrix(self):
        modal_mass = 0.25 * self.mass
        stiffness_matrix = modal_mass * np.diag(self.modal_frequency_list ** 2)

        return stiffness_matrix

    def save_response_amplitude_figure(results):
        import matplotlib.pyplot as plt
        
        plt.figure()
        for dof in results.dofs:
            plt.plot(
                results.coords['omega'], 
                np.abs(results.sel(radiating_dof=dof).data)
            )
        plt.xlabel('T (s)', fontsize=14)
        plt.ylabel('Maximum Deflection [m]', fontsize=14)
        #plt.legend()
        plt.tight_layout()
        plt.show()
        #plt.savefig('response_amplitude_operators.png', bbox_inches='tight')
