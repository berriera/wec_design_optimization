import numpy as np
import datetime

from tube_class import evaluate_tube_design
from tube_optimization import hooke_jeeves__greedy

# Test objective function with global minimum at (2.15, 42.0, -8.75)
def test_quadratic_function(x):
    return (x[0] - 2.15) ** 2 + (x[1] - 42.0) ** 2 + (x[2] + 8.75) ** 2

# Design variables in order: 
#   static_radius r_s
#   length L
#   submergence z_s

starting_locations = [  np.array([ 2.50,  30.0,  2.25]), np.array([ 0.45,  50.0, -8.75]), np.array([ 1.60, 170.0, -11.25]), 
                        np.array([ 0.30, 114.0, -2.25]), np.array([ 2.30,  74.0,  1.00]), np.array([ 0.70,  94.0,  -3.00]),
                        np.array([ 3.00, 160.0, -7.00]), np.array([ 1.25, 194.0, -0.75]), np.array([ 0.95, 136.0, -10.50]), 
                        np.array([ 2.10,  60.0, -5.00])]
moves_list = [0.05, 2.0, 0.25]
upper_bounds = np.array([3.00, 200.0,   3.0]) + 1e-4
lower_bounds = np.array([0.05,  20.0, -12.0]) - 1e-4

header = 'Geometry optimization run, 10 modes, 50 wave frequencies:'
method = '\nAlgorithm: Greedy hooke and jeeves with dynamic move ordering'
variables = '\nVariables:\n' + '\tStatic radius rs' + '\tLength le' + '\tSubmergence zs'
bounds = '\nBounds:\n' + '\tr_s   in [0.05, 3.0]\n' + '\tL in [20.0, 200.0]\n' + '\tz_s in [-12.0, 3.0]\n'
moves = '\nMoves list: {}'.format(moves_list)
start_time = '\nStart_time: {}.'.format(datetime.datetime.today())

header = header + variables + bounds + moves + start_time

for starting_design in starting_locations:
    history_file = 'geometry_optimization__x0=rs_le_zs=[{}]'.format(starting_design)
    hooke_jeeves__greedy(obj_function=test_quadratic_function,
                            starting_point=starting_design, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
                            moves_list=moves_list, history_file=history_file, header=header)
