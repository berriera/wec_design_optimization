import os

from capytaine import *
import capytaine.post_pro
import numpy as np
import logging
import matplotlib.pyplot as plt

os.system('cls')


def plate_flexure_mode_shape(x, y, z):
    from math import pi, cos, sin, cosh, sinh

    device_height = 10.0
    device_width = 6.0
    device_thickness = 0.50
    base_height = 0.3
    depth = -10.0

    z_center = depth + base_height + device_height / 2
    
    u = cos(pi * y / device_width) * cos(pi*(z - z_center) / device_height)
    v = 0.0
    w = 0.0

    # x_disp = (z + height) ** 2 / (height ** 2)  # Initialize parabolic mode shape for estimates

    return (u, v, w)


# Set logger configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

# Material parameters
density = 5000
elastic_modulus = 1e6
nu = 0.3


# Create OSWEC mesh
device_height = 10.0
device_width = 6.0
device_thickness = 0.50
base_height = 0.3
depth = -10.0
wave_direction = 0.0

volume = device_width * device_height * device_thickness
mass = density * volume

omega_range = np.linspace(0.1, 5.0, 50)

full_oswec = RectangularParallelepiped(size=(device_thickness, device_width, device_height + 2),
            resolution=(4, 40, 32),
            center = (0.0, 0.0, depth + base_height + device_height / 2))

dissipation_matrix = np.zeros(shape=(2, 2))
mass_matrix = mass * np.array([[1.0, 0.0], [0.0, 0.25]])
stiffness_matrix = 1e5 * np.eye(N=2)

# Add custom defined pitch axis about constrained axis
pitch_axis = Axis()
pitch_axis.point = np.array([0.0, 0.0, depth + base_height])
pitch_axis.vector = np.array([0.0, 1.0, 0.0])

full_oswec.add_rotation_dof(name='Pitch', axis = pitch_axis)
full_oswec.dofs['Plate Flexure'] = np.array([plate_flexure_mode_shape(x, y, z) for x, y, z, in full_oswec.mesh.faces_centers])

full_oswec.mass = full_oswec.add_dofs_labels_to_matrix(mass_matrix)
full_oswec.dissipation = full_oswec.add_dofs_labels_to_matrix(dissipation_matrix)
full_oswec.hydrostatic_stiffness = full_oswec.add_dofs_labels_to_matrix(stiffness_matrix)

oswec = full_oswec.copy()
oswec.keep_immersed_part(sea_bottom=depth)
full_oswec.show()
oswec.show()

# Animate rigid body pitch DOF along with modal flexure DOF
animation = full_oswec.animate(motion={'Pitch': 0.40, 'Plate Flexure': 1.25}, loop_duration=6.0)
animation.run()

# Problem definition
oswec_problems = [RadiationProblem(body=oswec,sea_bottom=depth, 
                    radiating_dof=dof, omega=omega) 
                    for dof in oswec.dofs 
                    for omega in omega_range]
oswec_problems += [DiffractionProblem(body=oswec, sea_bottom=depth,
                    omega=omega, wave_direction=wave_direction) 
                    for omega in omega_range]

# Solve for results and assemble data
solver = BEMSolver()
results = [solver.solve(problem) for problem in sorted(oswec_problems)]
data = assemble_dataset(results)
rao_data = capytaine.post_pro.rao(dataset=data, wave_direction=0.0, dissipation=None, stiffness=None)

# Plot results
for dof in full_oswec.dofs:
    plt.figure()
    plt.plot(
        omega_range,
        data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
        label='Added Mass',
        marker='o'
    )
    plt.plot(
        omega_range,
        data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
        label='Radiation Damping',
        marker='o'
    )
    plt.xlabel('$\omega$')
    plt.legend()
    plt.title(dof)
    #plt.savefig(dof + 'results.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

for dof in full_oswec.dofs:
    plt.figure()
    plt.plot(
        omega_range,
        np.abs(rao_data.sel(radiating_dof=dof))
    )
    plt.xlabel('$\omega$')
    plt.ylabel('RAO')
    plt.title(dof)
    #plt.savefig(dof + 'rao_results.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()