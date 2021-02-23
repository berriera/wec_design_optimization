import numpy as np
import capytaine as cpt

class elasticTube(object):


    def __init__(self, static_radius, length, submergence):
        from math import inf, pi

        # Constants
        self.rho = 1025
        self.water_depth = -inf
        self.wave_direction = 0.0
        self.mode_count = 5

        # Independent design variables
        self.static_radius = static_radius
        self.length = length
        self.submergence = submergence

        # Dependent geometry for integration bounds
        self.bounds = [-length / 2, length / 2]

        # Dependent inertia variables
        self.displaced_volume = pi * (self.static_radius ** 2) * self.length
        self.displaced_mass = self.rho * self.displaced_volume

        self.rotational_inertia_x_axis = (1 / 2) * self.displaced_mass * (self.static_radius ** 2)
        self.rotational_inertia_y_axis = (1 / 12) * self.displaced_mass * (3 * (self.static_radius ** 2) + (self.length ** 2))
        self.rotational_inertia_z_axis = self.rotational_inertia_y_axis

    def generate_tube(self):
        tube = cpt.HorizontalCylinder(
            radius=self.static_radius, length=self.length, center=(0, 0, self.submergence),
            nx=60, ntheta=20, nr=3, clever=False) # TODO: adjust nx, nr, clever args here
        tube.keep_immersed_part()
        #tube.show()

        # Add all rigid DOFs
        tube.add_all_rigid_body_dofs()

        # Add all elastic mode DOFs
        for k in range(self.mode_count):
            key_name = 'bulge_mode_' + str(k+1)
            tube.dofs[key_name] = np.array([elasticTube.mode_shape_derivatives(x, y, z, mode_number=k+1) for x, y, z, in tube.mesh.faces_centers])    

        tube.mass = tube.add_dofs_labels_to_matrix(elasticTube.mass_matrix())
        tube.dissipation = tube.add_dofs_labels_to_matrix(elasticTube.damping_matrix())
        tube.hydrostatic_stiffness = tube.add_dofs_labels_to_matrix(elasticTube.stiffness_matrix())

        return tube

    def mode_shapes(self, x, y, z):
        # Define chi(x)
        pass

    def mode_shape_derivatives(self, x, y, z, mode_number):
        # Defines del{chi}/del{x}(x)
        from math import sin, pi

        dr = sin(mode_number * 2*pi*x / self.length)

        u = 0.0
        v = (y / self.radius) * dr
        w = ((z - self.submergence) / self.radius) * dr

        return (u, v, w)

    def mass_matrix(self):
        # Define rigid body mass matrix
        mass_matrix = np.zeros(shape=(6 + self.mode_count, 6 + self.mode_count))
        for k in [0, 1, 2]:
            mass_matrix[k][k] = self.displaced_mass
        mass_matrix[3][3] = self.rotational_inertia_x_axis
        mass_matrix[4][4] = self.rotational_inertia_y_axis
        mass_matrix[5][5] = self.rotational_inertia_z_axis        

        return mass_matrix

    def damping_matrix(self):
        # Integrates mode_shapes and mode_shape_derivates along bounds
        dissipation_matrix = np.zeros(shape=(6 + self.mode_count, 6 + self.mode_count))
        wall_damping_matrix = np.zeros(shape=(6 + self.mode_count, 6 + self.mode_count))

        pass

    def stiffness_matrix(self):
        stiffness_matrix = np.zeros(shape=(6 + self.mode_count, 6 + self.mode_count))

        pass

    def power(self):
        pass

    def objective(self):
        from math import pi

        displaced_volume = pi * (self.static_radius ** 2) * self.length
        return elasticTube.power() / displaced_volume

    def _mode_shape_product(self):
        pass

    def _mode_shape_derivative_product(self):
        pass
