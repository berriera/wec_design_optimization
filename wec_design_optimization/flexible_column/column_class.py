import capytaine as cpt
import numpy as np
import logging
from math import pi, sqrt
from scipy.integrate import quad
import matplotlib.pyplot as plt

import os
os.system('cls')
logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

# Independent geometry design variables
radius = 10.0
height = 200.0

# Independent material design variables
elastic_modulus = 4.8e9 / 100 # Pa
density = 500.0
nu = 0.3

# Period definitions
t_start = 5.0
t_end = 10.0

# Modal definitions
#k1l = 1.8751
k1l = 4.6941
mode_count = 1

# Incident wave frequencies
wave_periods = np.linspace(t_start, t_end, 30)
wave_frequencies = 2 * pi / wave_periods

# Dependent geometry values
cross_sectional_area = pi * (radius ** 2)
volume = cross_sectional_area * height
mass = density * volume
inertia = (pi / 4) * (radius ** 4)

# Dependent modal values
k = k1l / height
modal_frequency = sqrt(((elastic_modulus * inertia) / (density * cross_sectional_area)) * (k ** 4))
modal_period = (2 * pi) / modal_frequency

# Modal properties
modal_mass = 0.25 * mass
modal_stiffness = modal_mass * (modal_frequency ** 2)
water_depth = -height

def bending_mode(x, y, z):
    from math import cosh, sinh, cos, sin

    q = 1 + z / height

    #k1l = 1.8751
    k1l = 4.6941

    c1 = (sinh(k1l) - sin(k1l)) / (cosh(k1l) + cos(k1l))
    end_displacement = cosh(k1l) - cos(k1l) - c1 * (sinh(k1l) - sin(k1l))

    u = (cosh(k1l * q) - cos(k1l * q) - c1 * (sinh(k1l * q) - sin(k1l * q))) / end_displacement
    v = 0.0
    w = 0.0

    return (u, v, w)

def integration_mode(q):
    from math import cosh, sinh, cos, sin

    #k1l = 1.8751
    k1l = 4.6941

    c1 = (sinh(k1l) - sin(k1l)) / (cosh(k1l) + cos(k1l))
    end_displacement = cosh(k1l) - cos(k1l) - c1 * (sinh(k1l) - sin(k1l))
    u = (cosh(k1l * q) - cos(k1l * q) - c1 * (sinh(k1l * q) - sin(k1l * q))) / end_displacement

    return u

def integration_mode_product(q):
    return integration_mode(q) ** 2

x_plot = np.linspace(0, 1.0, 100)
y_plot = np.zeros_like(x_plot)
m = 0
for x in x_plot:
    y_plot[m] = integration_mode(x)
    m += 1
plt.plot(x_plot, y_plot)
plt.show()


modal_mass_coefficient = quad(func=integration_mode_product, a=0.0, b=1.0)[0]
print(modal_mass_coefficient)
print(modal_mass_coefficient * mass)


column = cpt.VerticalCylinder(length=height + 2, radius=radius, 
                                center=(0, 0, -height / 2), 
                                nx=50, ntheta=30, nr=8,
                                clever=False)
column.keep_immersed_part(sea_bottom=water_depth)

column.dofs['Bending 1'] = np.array([bending_mode(x, y, z)
                                     for x, y, z, in column.mesh.faces_centers])

column.mass = column.add_dofs_labels_to_matrix([[modal_mass]])
column.hydrostatic_stiffness = column.add_dofs_labels_to_matrix([[modal_stiffness]])
column.dissipation = column.add_dofs_labels_to_matrix([[0.0]])

animation = column.animate(motion={'Bending 1': 30.0}, loop_duration=8.0)
animation.run()

wave_direction = 0.0
bem_solver = cpt.BEMSolver()
problems = [cpt.RadiationProblem(sea_bottom=water_depth, body=column, 
            radiating_dof=dof, omega=omega) 
            for dof in column.dofs for omega in wave_frequencies]
problems += [cpt.DiffractionProblem(sea_bottom=water_depth, body=column, 
            wave_direction=wave_direction, omega=omega) 
            for omega in wave_frequencies]

results = [bem_solver.solve(problem) for problem in problems]
data = cpt.assemble_dataset(results)

data['RAO'] = cpt.post_pro.rao(data, wave_direction=wave_direction)


plt.figure()
for dof in column.dofs:
    plt.plot(
        wave_periods,
        data['added_mass'].sel(radiating_dof=dof, influenced_dof=dof),
        label=dof,
        marker='o'
    )
plt.xlabel('T (s)')
plt.ylabel('Added Mass')
plt.ylim((0, 1.6e7))
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('added_mass.png', bbox_inches='tight')

plt.figure()
for dof in column.dofs:
    plt.plot(
        wave_periods,
        data['radiation_damping'].sel(radiating_dof=dof, influenced_dof=dof),
        label=dof,
        marker='o'
    )
#plt.xlabel('$\omega \$ (rad/s)')
plt.xlabel('T (s)')
plt.ylabel('Radiation damping')
plt.ylim((0, 5e6))
plt.legend()
#plt.tight_layout()
#plt.show()
plt.savefig('radiation_damping.png', bbox_inches='tight')

plt.figure()
for dof in column.dofs:
    plt.plot(
        wave_periods, 
        np.abs(data['RAO'].sel(radiating_dof=dof).data),
        label=dof,
        marker='o'
    )
#plt.xlabel('$\omega \ $ (rad/s)')
plt.xlabel('T (s)')
plt.ylabel('RAO')
#plt.ylim((0, 2.0))
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('response_amplitude_operator.png', bbox_inches='tight')
