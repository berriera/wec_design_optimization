import numpy as np
from tube_class import evaluate_tube_design

# Design variables in order: 
#   static_radius
#   length 
#   submergence
#   power_take_off_damping
#   wall_stiffness
#   fiber_pretension

design_variables = np.array([0.9, 60, -1.35])
tube_objective = evaluate_tube_design(design_variables)
print(tube_objective)

# TODO: potentially specify optimization type as a string: 'geometry', 'material', 'all' to pick the variables

moves_list = []
converged = False
