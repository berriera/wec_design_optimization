import os
from capytaine import *

import numpy as np
from math import sin, pi

os.system('cls')

barge = RectangularParallelepiped(size=(80, 10, 10), resolution=(16, 10, 10))
# barge.show()

length = 20
barge.add_all_rigid_body_dofs()
barge.dofs['bending_1'] = np.array([(0, 0, sin(0.25*pi * x/length) )for x, y, z in barge.mesh.faces_centers])
barge_animation = barge.animate(motion={'Surge': 0.5, 'Heave': 1.0, 'Pitch': 0.01, 'bending_1': 3.0}, loop_duration=3.0)
barge_animation.run()
