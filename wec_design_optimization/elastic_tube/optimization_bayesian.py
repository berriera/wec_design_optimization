import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

from tube_class import evaluate_tube_design__bayesian

def test_quadratic_function(radius, length, submergence):
    x = radius
    y = length
    z = submergence
    return -((x - 2.15) ** 2 + (y - 42.0) ** 2 + (z + 8.75) ** 2)

sq_exp = squaredExponential(l=3, sigman=0.0)
gp = GaussianProcess(sq_exp)
acq = Acquisition(mode='ExpectedImprovement')
design_parameters = {'radius': ('cont', [0.05, 3.0]), 'length': ('cont', [20.0, 200.0]), 'submergence': ('cont', [-12.0, 3.0])}

np.random.seed(42)
gpgo = GPGO(gp, acq, evaluate_tube_design__bayesian, design_parameters)
gpgo.run(max_iter=40, init_evals=20)
optimal_design, optimal_power = gpgo.getResult()

print('Best design after {} iterations is {} with objective function value {}'.format(60, optimal_design, optimal_power))
