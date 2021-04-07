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
omega_range = np.linspace(0.3, 8.0, 5)

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
    nx=50, ntheta=15, nr=3, clever=False,
)
tube.keep_immersed_part()
tube.show()

# Add all rigid DOFs
tube.add_all_rigid_body_dofs()

# Add all flexible DOFs
for k in range(mode_count):
    key_name = 'bulge_' + str(k+1)
    tube.dofs[key_name] = np.array([bulging_mode(x, y, z, mode_number=k+1, 
                                    radius=tube_radius, length=tube_length, z_s=tube_submergence)
                                     for x, y, z, in tube.mesh.faces_centers])    

animation = tube.animate(motion={'Surge': 1.0, 'bulge_1': 0.30, 'bulge_2': 0.12 + 0.07j, 'bulge_3': -0.15 - 0.03j, 'bulge_4': -0.03-0.02j}, loop_duration=4.0)
animation.run()
