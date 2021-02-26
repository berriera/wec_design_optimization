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

# Create OSWEC mesh
device_height = 10.0
device_width = 6.0
device_thickness = 0.50
base_height = 0.3
depth = -10.0

full_oswec = RectangularParallelepiped(size=(device_thickness, device_width, device_height),
            resolution=(4, 40, 32),
            center = (0.0, 0.0, depth + base_height + device_height / 2))

# Add custom defined pitch axis about constrained axis
pitch_axis = Axis()
pitch_axis.point = np.array([0.0, 0.0, depth + base_height])
pitch_axis.vector = np.array([0.0, 1.0, 0.0])

full_oswec.add_rotation_dof(name='Pitch', axis = pitch_axis)
full_oswec.dofs['plate_flexure'] = np.array([plate_flexure_mode_shape(x, y, z) for x, y, z, in full_oswec.mesh.faces_centers])

oswec = full_oswec.copy()
oswec.keep_immersed_part() # TODO: copy here instead
#full_oswec.show()
#oswec.show()

# Animate rigid body pitch DOF along with modal flexure DOF
animation = full_oswec.animate(motion={'Pitch': 0.40, 'plate_flexure': 1.00}, loop_duration=6.0)
animation.run()

# Problem definition
omega_range = np.linspace(0.1, 7.0, 40)
oswec_problems = [RadiationProblem(body=oswec, radiating_dof=dof, omega=omega, sea_bottom=depth) for dof in oswec.dofs for omega in omega_range]

# Solve for results and assemble data
solver = BEMSolver()
results = [solver.solve(pb) for pb in sorted(oswec_problems)]
data = assemble_dataset(results)
# rao_data = capytaine.post_pro.rao(dataset=data, wave_direction=0.0, dissipation=None, stiffness=None)

# Plot added mass
plt.figure()
plt.plot(omega_range, data['added_mass'].sel(radiating_dof="Pitch"), label="Pitch", marker='o')
plt.show()

plt.figure()
plt.plot(omega_range, data['radiation_damping'].sel(radiating_dof="Pitch"), label="Pitch", marker='o')
plt.show()
