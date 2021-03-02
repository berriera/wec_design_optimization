import numpy as np
from tube_class import evaluate_tube_design

# Design variables in order: 
#   static_radius
#   length 
#   submergence
#   power_take_off_damping

design_variables = np.array([0.9, 60, -1.35, 120e3])
tube_objective = evaluate_tube_design(design_variables)
print(tube_objective)
