import os
import logging
import numpy as np
from math import pi

import capytaine as cpt
from capytaine.ui.vtk.animation import Animation

os.system('cls')
logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

mode_count = 4
tube_radius = 0.20
tube_length = 10.0
tube_submergence = -0.3
omega_range = np.linspace(0.3, 8.0, 10)

def bulging_mode(x, y, z, mode_number, radius, length, z_s):
    from math import sin, pi

    dr = sin(mode_number * 2*pi*x / length)

    u = 0.0
    v = (y / radius) * dr
    w = ((z - z_s) / radius) * dr

    return (u, v, w)

# Generate tube mesh from input values
tube = cpt.HorizontalCylinder(
    radius=tube_radius, length=tube_length, center=(0, 0, tube_submergence),
    nx=10, ntheta=15, nr=3, clever=False,
)
tube.keep_immersed_part()
#tube.show()

# Add all rigid DOFs
tube.add_all_rigid_body_dofs()

# Add all flexible DOFs
for k in range(mode_count):
    key_name = 'bulge_' + str(k+1)
    tube.dofs[key_name] = np.array([bulging_mode(x, y, z, mode_number=k+1, 
                                    radius=tube_radius, length=tube_length, z_s=tube_submergence)
                                     for x, y, z, in tube.mesh.faces_centers])    

#print(tube.dofs.keys())

tube.mass = tube.add_dofs_labels_to_matrix(
        [[2e7, 0,   0,   0,   0,   0,  0,  0,  0,  0],
         [0,   2e7, 0,   0,   0,   0,  0,  0,  0,  0],
         [0,   0,   2e7, 0,   0,   0,  0,  0,  0,  0],
         [0,   0,   0,   1e10, 0,   0,  0,  0,  0,  0],
         [0,   0,   0,   0,   2e9, 0,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   1e10,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  4e6,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  4e6,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  4e6,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  0,  4e6]]
         )
tube.hydrostatic_stiffness = tube.add_dofs_labels_to_matrix(
        [[2e7, 0,   0,   0,   0,   0,  0,  0,  0,  0],
         [0,   2e7, 0,   0,   0,   0,  0,  0,  0,  0],
         [0,   0,   2e7, 0,   0,   0,  0,  0,  0,  0],
         [0,   0,   0,   1e10, 0,   0,  0,  0,  0,  0],
         [0,   0,   0,   0,   2e9, 0,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   1e10,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  4e6,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  4e7,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  4e8,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  0,  4e9]]
    )

tube.stiffness = tube.add_dofs_labels_to_matrix(
        [[2e7, 0,   0,   0,   0,   0,  0,  0,  0,  0],
         [0,   2e7, 0,   0,   0,   0,  0,  0,  0,  0],
         [0,   0,   2e7, 0,   0,   0,  0,  0,  0,  0],
         [0,   0,   0,   1e10, 0,   0,  0,  0,  0,  0],
         [0,   0,   0,   0,   2e9, 0,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   1e10,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  4e6,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  4e7,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  4e8,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  0,  4e9]]
    )

tube.dissipation = tube.add_dofs_labels_to_matrix(
        [[2e7, 0,   0,   0,   0,   0,  0,  0,  0,  0],
         [0,   2e7, 0,   0,   0,   0,  0,  0,  0,  0],
         [0,   0,   2e7, 0,   0,   0,  0,  0,  0,  0],
         [0,   0,   0,   1e10, 0,   0,  0,  0,  0,  0],
         [0,   0,   0,   0,   2e9, 0,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   1e10,  0,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  4e6,  0,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  4e6,  0,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  4e6,  0],
         [0,   0,   0,   0,  0,   0,  0,  0,  0,  4e6]]
    )


animation = tube.animate(motion={'Surge': 1.0, 'bulge_1': 0.08, 'bulge_2': 0.03 + 0.07j, 'bulge_3': -0.05 - 0.03j, 'bulge_4': -0.01-0.02j}, loop_duration=4.0)
#animation.run()

wave_direction = 0.0

bem_solver = cpt.BEMSolver()
problems = [cpt.RadiationProblem(omega=omega, body=tube, radiating_dof=dof) for dof in tube.dofs for omega in omega_range]
problems += [cpt.DiffractionProblem(omega=omega, body=tube, wave_direction=wave_direction) for omega in omega_range]
results = [bem_solver.solve(problem) for problem in problems]
*radiation_results, diffraction_result = results
data = cpt.assemble_dataset(results)

data['RAO'] = cpt.post_pro.rao(data, wave_direction=wave_direction)

import matplotlib.pyplot as plt

plt.figure()
for dof in tube.dofs:
    plt.plot(
        omega_range,
        data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
        label=dof,
        marker='o',
    )
plt.xlabel('omega')
plt.ylabel('Added Mass')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for dof in tube.dofs:
    plt.plot(
        omega_range,
        data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
        label=dof,
        marker='o',
    )
plt.xlabel('omega')
plt.ylabel('Radiation damping')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for dof in tube.dofs:
    print(dof)
    plt.plot(
        omega_range, 
        sum(abs(data['RAO'].sel(radiating_dof=dof).data) for dof in tube.dofs),
        #data['RAO'].sel(radiating_dof=dof, influenced_dof=dof),
        label=dof,
        marker='o',
    )
#rao_faces_motion = sum(data['RAO'].sel(radiating_dof=dof).data for dof in tube.dofs)
plt.xlabel('omega')
plt.ylabel('RAO')
plt.legend()
plt.tight_layout()
plt.show()
