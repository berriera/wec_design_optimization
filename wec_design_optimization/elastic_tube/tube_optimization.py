def hooke_jeeves__greedy(obj_function, starting_point, upper_bounds, lower_bounds, moves_list,
                            alpha_0=16, history_file = 'optimization_history.txt', header=''):
    import numpy as np
    import pickle
    import time

    start_time = time.time()
    var_count = len(starting_point)
    search_direction_count = 2 * var_count
    assert len(moves_list) == var_count

    moves_array = np.zeros(shape=(search_direction_count, var_count))
    for n in range(var_count):
        moves_array[2*n][n] = moves_list[n]
        moves_array[2*n+1][n] = -moves_list[n]

    # Initialize history variables
    location_history = []
    function_history = []
    converged = False
    iteration_count = 1

    best_objective_value = obj_function(starting_point)
    best_location = np.copy(starting_point)

    # Print log file header
    f = open(history_file, 'w')
    print(header, file=f)

    # Set up first history values
    location_history.append(best_location)
    function_history.append(best_objective_value)
    print('\nFirst design is {}'.format(best_location), file=f)
    print('Objective function value = {:.2f}'.format(best_objective_value), file=f)

    # Initialize potential move counters
    search_direction_int = 0
    alpha = alpha_0

    def check_history(history_list, potential_design):
        for design in history_list:
            if np.allclose(potential_design, design):
                return True
        return False

    while not converged:
        # Pick new move
        search_direction = moves_array[search_direction_int]
        move = alpha * search_direction
        print('\nMoving in direction: {}'.format(move), file=f)

        # Update design variables
        new_location = best_location + move
        print('\tTo test design variables: {}'.format(new_location), file=f)

        # Try new move if in bounds and has not been visited before
        bounds_check = np.all(new_location <= upper_bounds) and np.all(new_location >= lower_bounds)

        if check_history(location_history, new_location):
            print('\tDesign {} already visited'.format(new_location), file=f)
            past_location_index = np.where(new_location == location_history)[0][0]
            new_objective_value = function_history[past_location_index]

        elif bounds_check:
            iteration_count = iteration_count + 1

            print('\tNew design is in bounds. Evaluating iteration number: {}'.format(iteration_count), file=f)
            
            # Output to terminal
            print('Evaluating design {} with location {} !'.format(iteration_count, new_location))

            new_objective_value = obj_function(new_location)
            
            # Update histories
            location_history.append(new_location)
            function_history.append(new_objective_value)
            print('\tObjective function value: {}'.format(new_objective_value), file=f)

        else:
            print('\tNew design not in bounds. Trying next move.', file=f)
            new_objective_value = 1e23

        # Update design if tested one is better
        if new_objective_value < best_objective_value:
            print('\tMoving to new location', file=f)
            best_location = np.copy(new_location)
            best_objective_value = new_objective_value

            # Dynamic ordering: update moves list to repeat the current move if it performs better
            tmp = np.zeros_like(moves_array)
            tmp[0] = search_direction
            row = 1
            for k in range(search_direction_count):
                if np.any(moves_array[k] != search_direction):
                    tmp[row] = moves_array[k]
                    row += 1
            moves_array = np.copy(tmp)

            # Reset the search direction
            search_direction_int = 0

        # Pick next move if the current one performed worse.
        # This will happen if the new location is out of bounds or if the returned power value is worse.
        elif search_direction_int < (search_direction_count - 1):
            search_direction_int = search_direction_int + 1

        # Make lower discretization level when out of moves at the current alpha level
        elif alpha > 1:
            alpha = (1/2) * alpha
            search_direction_int = 0

        # Converged if multipler is now equal to 1 for all variables
        else:
            finish_time = time.time()
            run_time_minutes = (finish_time - start_time) / 60
            print('\n\n\nOptimization converged!', file=f)
            print('Final converged design variables: {}'.format(best_location), file=f)
            print('\nFinal converged objective function value: {}'.format(best_objective_value), file=f)
            print('Iteration count: {}'.format(iteration_count), file=f)

            print('Total optimization run time: {:.2f} minutes'.format(run_time_minutes), file=f)

            # Save results
            with open('optimization_history__x0={}.pkl'.format(starting_point), 'wb') as history_file:
                pickle.dump(location_history, history_file)
                pickle.dump(function_history, history_file)

            converged = True

    f.close()
