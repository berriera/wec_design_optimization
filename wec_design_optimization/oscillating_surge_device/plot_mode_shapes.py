import os
import numpy as np
import matplotlib.pyplot as plt

os.system('cls')
height = 8.0
z_range = np.linspace(-height, 0, 50)
x = np.zeros_like(z_range)

def bending_mode(z):
    from math import pi, cos, sin, cosh, sinh
    height = 8.0
    z = z + height
    
    k = (1 / 8.0) * (5 * pi / 2.0)

    c2 = 1.0
    c1 = ((cos(k*height) + cosh(k*height)) / (sin(k*height + sinh(k*height))))*c2

    c3 = c1
    c4 = -c2

    x_disp = c1*sin(k*z) + c2*cos(k*z) + c3*sinh(k*z) + c4*cosh(k*z)
    x_end_disp = c1*sin(k*height) + c2*cos(k*height) + c3*sinh(k*height) + c4*cosh(k*height)

    x_disp = x_disp / x_end_disp  # Normalize end displacement to unity

    # x_disp = (z + height) ** 2 / (height ** 2)  # Initialize parabolic mode shape for estimates

    return x_disp

index = 0
for z in z_range:

    x[index] = bending_mode(z)
    index += 1

plt.plot(z_range, x)
plt.show()