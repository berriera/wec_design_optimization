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

start_location = np.array([0.9, 60, -1.25])
moves_list = [0.05, 2.0, 0.25]
upper_bounds = np.array([3.00, 200.0,   3.0])
lower_bounds = np.array([0.05,  20.0, -12.0])
history_file = 'geometry_optimization__x0=rs_le_zs=[{}]'.format(start_location)

header = 'Geometry optimization run, 10 modes, 50 wave frequencies:'
variables = '\nVariables:\n' + '\tStatic radius rs' + '\tLength le' + '\tSubmergence zs'
bounds = '\nBounds:\n' + '\tr_s   in [0.05, 3.0]\n' + '\tL in [20.0, 200.0]\n' + '\tz_s in [-12.0, 3.0]\n'
moves = '\nMoves list: {}'.format(moves_list)
start_time = '\nStart_time: {}.'.format(datetime.datetime.today())

header = header + variables + bounds + moves + start_time

hooke_jeeves__greedy(obj_function=test_quadratic_function,
                        starting_point=start_location, upper_bounds=upper_bounds, lower_bounds=lower_bounds,
                        moves_list=moves_list, history_file=history_file, header=header)
