import numpy as np
np.random.seed(42)
np.set_printoptions(precision=3)

def uniform_projection_plan(m, n):
    permutations = [np.random.permutation(m) for k in range(n)]
    plan = np.array([[permutations[i][j] for i in range(n)] for j in range(m)])
    return plan

vars = 3
design_count = 3
lower_bounds = np.array([0.05,  20, -12.0])
upper_bounds = np.array([ 3.0, 200,   3.0])

var_spacing = np.array([np.linspace(lower_bounds[k], upper_bounds[k], design_count + 1) for k in range(vars)])
var_spacing = np.transpose(var_spacing)

sampling_plan = uniform_projection_plan(design_count, vars)

random_factors = np.random.random(size=(design_count, vars))
for k in range(design_count):
    sample = sampling_plan[k]
    random_design_factors = random_factors[k]
    design = []
    for j in range(vars):
        choice = sample[j]
        variable_bottom = var_spacing[choice][j]
        variable_top = var_spacing[choice+1][j]
        random_variable_factor = random_design_factors[j]
        design_variable = variable_bottom + random_variable_factor * (variable_top - variable_bottom)
        design.append(design_variable)
    design = np.array(design)

    print('Design {} = {}'.format(k+1, design))
