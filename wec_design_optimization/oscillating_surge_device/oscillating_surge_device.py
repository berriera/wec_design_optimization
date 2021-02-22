import os

from capytaine import *
import capytaine.post_pro
import numpy as np
import logging
import matplotlib.pyplot as plt

os.system('cls')


def bending_mode(x, y, z):
    from math import pi, cos, sin, cosh, sinh
    height = 8.0
    z = z + height
    
    k = (1 / 8.0) * (5 * pi / 2.0)

    c2 = 1.0
    c1 = ((cos(k*height) + cosh(k*height)) / (sin(k*height + sinh(k*height))))*c2

    c3 = c1
    c4 = -c2

    x_disp = c1*sin(k*z) + c2*cos(k*z) + c3*sinh(k*z) + c4*cosh(k*z)
    x_end_disp = c1*sin(k*height) + c2*cos(k*height) + c3*sinh(k*height) + c4*cosh(k*height)

    x_disp = x_disp / x_end_disp  # Normalize end displacement to unity

    # x_disp = (z + height) ** 2 / (height ** 2)  # Initialize parabolic mode shape for estimates

    return (x_disp, 0, 0)


# Set logger configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

# Create OSWEC mesh
height = 8.0
full_oswec = RectangularParallelepiped(size=(0.5, 5.0, height), resolution=(4, 40, 32), center = (0.0, 0.0, 0.0))
oswec = full_oswec.keep_immersed_part()
#oswec.show()

# Add all relevant DOFs
#  oswec.add_rotation_dof(name="Pitch")
#  oswec.dofs['vertical_pitch'] = np.array([(1 + (z / height), 0, 0) for x, y, z, in oswec.mesh.faces_centers])
oswec.dofs['pitch_bend'] = np.array([bending_mode(x, y, z) for x, y, z, in oswec.mesh.faces_centers])

# Problem definition
omega_range = np.linspace(0.1, 7.0, 40)
#oswec_problems = [RadiationProblem(body=oswec, radiating_dof=dof, omega=omega, sea_bottom=-10.0) for dof in oswec.dofs for omega in omega_range]
oswec_problems = [RadiationProblem(body=oswec, radiating_dof='pitch_bend', omega=omega, sea_bottom=-10.0) for omega in omega_range]

# Solve for results and assemble data
solver = BEMSolver()
results = [solver.solve(pb) for pb in sorted(oswec_problems)]
data = assemble_dataset(results)
# rao_data = capytaine.post_pro.rao(dataset=data, wave_direction=0.0, dissipation=None, stiffness=None)

# Plot added mass
plt.figure()
plt.plot(omega_range, data['added_mass'].sel(radiating_dof="pitch_bend"), label="pitch_bend", marker='o')
plt.show()

# Animate modal DOF
animation = full_oswec.animate(motion={'pitch_bend': 1.0}, loop_duration=2.5)
animation.run()
