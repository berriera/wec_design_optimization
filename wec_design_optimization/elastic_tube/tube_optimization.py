import numpy as np
from tube_class import evaluate_tube_design

# Design variables in order: 
#   static_radius
#   length 
#   submergence
#   power_take_off_damping
#   wall_stiffness
#   fiber_pretension


def obj_func(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

# TODO: potentially specify optimization type as a string: 'geometry', 'material', 'all' to pick the variables

location_history = []
function_history = []
converged = False

# Initialize starting location and objective function value
current_location = np.array([0.9, 60, -0.9])
current_objective_value = evaluate_tube_design(current_location)

print('Objective function value = {:.2f}'.format(current_objective_value))
z
#current_objective_value = obj_func(current_location)
iteration_count = 0

variable_count = 3
moves_list = np.eye(N=variable_count)
moves_list[0][0] = 0.1
moves_list[1][1] = 2.5
moves_list[2][2] = 0.25


move_number = 0
move_direction = 1
move_multiplier = 16

def check_bounds(potential_design):
    upper_bounds = np.array([6.0, -200.0, -0.1])
    lower_bounds = np.array([0.1, 20.0, -20.0])

    return np.all(potential_design <= upper_bounds) and np.all(potential_design >= lower_bounds)


def check_constraints(potential_design):
    from math import fabs
    static_radius = potential_design[0]
    submergence = fabs(potential_design[2])

    return submergence >= static_radius


while not converged:
    # Pick new move
    movement = move_direction * move_multiplier * moves_list[move_number]
    print('\nMoving in direction: ', movement)

    # Update design variables
    new_location = current_location + movement

    # Check bounds first
    in_bounds = check_bounds(new_location)

    # Try new move if in bounds
    if in_bounds:
        iteration_count = iteration_count + 1

        print('Iteration number: ', iteration_count)
        print('\n\tWith design variables: ', new_location)

        location_history.append(new_location)
        new_objective_value = evaluate_tube_design(new_location)
        #new_objective_value = obj_func(new_location)

        function_history.append(new_objective_value)
        print('\n\tObjective function value: ', new_objective_value)

    # Update location if better
    if new_objective_value < current_objective_value:
        print('Moving to new location')
        current_location = np.copy(new_location)
        current_objective_value = new_objective_value
    
    # Switch direction if movement if worse
    elif move_direction == 1:
        move_direction = -1

    # Pick next variable to move if both move directions have been worse
    elif move_number < (variable_count - 1):
        move_number = move_number + 1
        move_direction = 1

    elif move_multiplier > 1:
        move_multiplier = (1/2) * move_multiplier
        move_number = 0

    # Converged if multipler is now equal to 1 for all variables
    else:
        print('\n\n\nOptimization converged!')
        print('Final converged design variables: ', current_location)
        print('\nFinal converged objective function value: ', current_objective_value)

        print('Iteration count: ', iteration_count)
        converged = True

    

