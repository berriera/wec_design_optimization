import numpy as np
import pickle
import datetime
from tube_class import evaluate_tube_design

# Test objective function with global minimum at (2.15, 32.5, -8.75)
def obj_func(x):
    return (x[0] - 2.15) ** 2 + (x[1] - 32.5) ** 2 + (x[2] + 8.75) ** 2

# Design variables in order: 
#   static_radius r_s
#   length L
#   submergence z_s
#   wall_stiffness K_{mat}
#   fiber_pretension T_s

# Initialize starting location and objective function value
current_location = np.array([0.9, 60, -1.25])
#current_location = np.array([9e5, 3.8e4])
current_objective_value = evaluate_tube_design(current_location)
#current_objective_value = obj_func(current_location)

# Set up move matrix for geometry optimization
variable_count = 3
moves_list = np.eye(N=variable_count)
moves_list[0][0] = 0.05
moves_list[1][1] = 2.5
moves_list[2][2] = 0.25

# Set up move matrix for material optimization
# Variable notation: K_{mat}, T_s
#variable_count = 2
#moves_list = np.eye(N=variable_count)
#moves_list[0][0] = 50.0e3
#moves_list[1][1] = 2.0e3


# Initialize history variables
location_history = []
function_history = []
converged = False
iteration_count = 1

# Print log file header
f = open('optimization_history_file.txt', 'w')
print('Geometry optimization run, 5 modes:', file=f)
print(datetime.datetime.today(), file=f)
variables = '\nVariables:\n' + '\tStatic radius r_s' + '\tLength L' + '\tSubmergence z_s'
bounds = 'Bounds:\n' + '\tr_s   in [0.05, 2.5]\n' + '\tL in [20.0, 200.0]\n' + '\tz_s in [-25.0, 2.5]'
print(variables, file=f)
print(bounds, file=f)
print('\nMoves list: ', file=f)
print(moves_list, file=f)


# Set up first history values
location_history.append(current_location)
function_history.append(current_objective_value)
print('\nFirst design is {}'.format(current_location), file=f)
print('Objective function value = {:.2f}'.format(current_objective_value), file=f)

# Initialize potential move counters
move_number = 0
move_direction = 1
move_multiplier = 16

def check_bounds(potential_design):
    # Bounds for geometry optimization
    upper_bounds = np.array([2.5, 200.0, 2.5])
    lower_bounds = np.array([0.05, 20.0, -25.0])

    # Bounds for material optimization
    #upper_bounds = np.array([100.0, 3000.0])
    #lower_bounds = np.array([10.0, 200.0])

    return np.all(potential_design <= upper_bounds) and np.all(potential_design >= lower_bounds)


def check_history(history_list, potential_design):
    for design in history_list:
        if np.array_equal(potential_design, design):
            return True
    return False

while not converged:
    # Pick new move
    movement = move_direction * move_multiplier * moves_list[move_number]
    print('\nMoving in direction: {}'.format(movement), file=f)

    # Update design variables
    new_location = current_location + movement
    print('\tTo test design variables: {}'.format(new_location), file=f)

    # Check bounds first and constraint if relevant
    bounds_check = check_bounds(new_location)
    if check_history(location_history, new_location):
        print('\tDesign {} already visited'.format(new_location), file=f)
        past_location_index = np.where(new_location == location_history)[0][0]
        new_objective_value = function_history[past_location_index]
        

    # Try new move if in bounds and has not been visited before
    elif bounds_check:
        iteration_count = iteration_count + 1

        print('\tNew design is in bounds. Evaluating iteration number: {}'.format(iteration_count), file=f)
        print('Evaluating design {} with location {} !'.format(iteration_count, new_location))

        new_objective_value = evaluate_tube_design(new_location)
        #new_objective_value = obj_func(new_location)
        
        # Update histories
        location_history.append(new_location)
        function_history.append(new_objective_value)
        print('\tObjective function value: {}'.format(new_objective_value), file=f)

    else:
        print('\tNew design not in bounds. Trying next move.', file=f)
        new_objective_value = 1e23

    # Update design if tested one is better
    if new_objective_value < current_objective_value:
        print('\tMoving to new location', file=f)
        current_location = np.copy(new_location)
        current_objective_value = new_objective_value

    # Tries a different move if the new objective function value is worse than the current one.
    # This will happen if the new location is out of bounds or if the returned power value is worse.

    # Reverses the search direction
    elif move_direction == 1:
        move_direction = -1

    # Pick next variable to move if both search directions along the current variable have been worse
    elif move_number < (variable_count - 1):
        move_number = move_number + 1
        move_direction = 1

    # Make lower discretization level when out of moves at the current level
    elif move_multiplier > 1:
        move_multiplier = (1/2) * move_multiplier
        move_direction = 1
        move_number = 0

    # Converged if multipler is now equal to 1 for all variables
    else:
        print('\n\n\nOptimization converged!', file=f)
        print('Final converged design variables: {}'.format(current_location), file=f)
        print('\nFinal converged objective function value: {}'.format(current_objective_value), file=f)
        print('Iteration count: {}'.format(iteration_count), file=f)

        # Save results
        with open('tube_history.pkl', 'wb') as history_file:
            pickle.dump(location_history, history_file)
            pickle.dump(function_history, history_file)

        converged = True

f.close()