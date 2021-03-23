from math import pi

length = 10
radius = 0.274
distensibility = 1.29e-4  # 1.12e-4 when tuning experimental to numerical model
thickness = 0.01
tube_mass = 91.7
b_visc = 17.8e3
rho = 1000

area = pi * (radius ** 2)
rho_tube = tube_mass / (2 * pi * radius * thickness * length)

b_mat = (1 / (2 * area)) * b_visc
eta = (thickness * area * b_mat) / (rho * radius)
k_mat = (radius) / (thickness * area *  distensibility)

print('Tube density = {:.3f}'.format(rho_tube))
print('Tube material damping = {:.3f}'.format(b_mat))
print('Tube damping coefficient = {:.3f}'.format(eta))
print('Tube stiffness = {:.3f}'.format(k_mat))

# Yeoh model parameters
c1 = 134e3
c2= -22.2e3
c3 = 7.30e3

ls = 1.309
ls_sq = ls ** 2
i1s = ls_sq + (1 / ls_sq) + 1
k_mat = (4 / area) * ((1 / ls_sq) * (c1 + 2 * c2 * (i1s - 3) + 3 * c3 * (i1s- 3) ** 2) + (ls_sq - (1 / ls_sq)) * (c2 + 3 * c3 * (i1s - 3)))

print('Tube stiffness = {:.3f}'.format(k_mat))

dist = (radius) / (thickness * area * k_mat)
print('Tube distensibility = {:.7f}'.format(dist))
