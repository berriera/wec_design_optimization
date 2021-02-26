import numpy as np
import capytaine as cpt

class ElasticTube(object):


    def __init__(self, static_radius, length, submergence, dissipation_coefficient):
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
        self.dissipation_coefficient = dissipation_coefficient

        # Dependent geometry for integration bounds
        self.bounds = [-length / 2, length / 2]

        # Dependent inertia variables
        self.cross_sectional_area = pi * (self.static_radius ** 2)
        self.displaced_volume = pi * (self.static_radius ** 2) * self.length
        self.displaced_mass = self.rho * self.displaced_volume

        self.rotational_inertia_x_axis = (1 / 2) * self.displaced_mass * (self.static_radius ** 2)
        self.rotational_inertia_y_axis = (1 / 12) * self.displaced_mass * (3 * (self.static_radius ** 2) + (self.length ** 2))
        self.rotational_inertia_z_axis = self.rotational_inertia_y_axis

    def generate_tube(self):
        """Generates an elastic tube mesh with all attached rigid body (if relevant) and modal degrees of freedom

        Args:
            None

        Returns:
            tube (an instance of a Capytaine FloatingBody)

        """
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
            tube.dofs[key_name] = np.array([ElasticTube.mode_shape_derivatives(x, y, z, mode_number=k+1) for x, y, z, in tube.mesh.faces_centers])    

        tube.mass = tube.add_dofs_labels_to_matrix(ElasticTube.mass_matrix())
        tube.dissipation = tube.add_dofs_labels_to_matrix(ElasticTube.damping_matrix())
        tube.hydrostatic_stiffness = tube.add_dofs_labels_to_matrix(ElasticTube.stiffness_matrix())

        return tube

    def mode_shapes(self, x, y, z, mode_number, integration_flag=False):
        # Define chi(x)
        pass

    def mode_shape_derivatives(self, x, y, z, mode_number, integration_flag=False):
        # Defines del{chi}/del{x}(x)
        from math import sin, pi

        dr = sin(mode_number * 2*pi*x / self.length)

        u = 0.0
        v = (y / self.radius) * dr
        w = ((z - self.submergence) / self.radius) * dr

        return (u, v, w)

    def mass_matrix(self):
        """Defines an n x n mass matrix for the tube, where n is the modal degrees of freedom

        Args:
            None

        Returns:
            mass_matrix (2d np array)

        """
        mass_matrix = self.displaced_mass * np.ones(shape = self.mode_count)
        mass_matrix = np.diag(mass_matrix)
     
        #elif self.rigid_body_degrees_of_freedom ==  6:
        #        
        #    for k in [0, 1, 2]:
        #        mass_matrix[k][k] = self.displaced_mass
        #    mass_matrix[3][3] = self.rotational_inertia_x_axis
        #    mass_matrix[4][4] = self.rotational_inertia_y_axis
        #    mass_matrix[5][5] = self.rotational_inertia_z_axis        
        #
        #    for k in range(6, 6 + self.mode_count):
        #        mass_matrix[k][k] = self.displaced_mass

        return mass_matrix

    def damping_matrix(self):
        """Defines an n x n damping matrix for the tube's modal degrees of freedom

        Args:
            None
        
        Returns:
            damping matrix (2d np array)

        """
        wall_damping_matrix = np.zeros(shape=(self.mode_count, self.mode_count))
        inner_flow_damping_matrix = np.zeros(shape=(self.mode_count, self.mode_count))

        damping_matrix = self.rho * self.cross_sectional_area * self.dissipation_coefficient * wall_damping_matrix \
            + self.rho * self.viscous_damping_parameter * inner_flow_damping_matrix
        return damping_matrix

    def stiffness_matrix(self):
        """Defines an n x n stiffness matrix for the tube's modal degrees of freedom

        Args:
            None
        
        Returns:
            stiffness matrix (2d np array)

        """
        modal_frequencies = ElasticTube._find_modal_frequencies()
        stiffness_matrix = self.displaced_mass * np.diag(modal_frequencies ** 2)

        return stiffness_matrix

    def power(self):
        """Calculates the mean power dissipated by the material as a function of wave frequency

        """

        pass

    def objective(self):
        from math import pi

        displaced_volume = pi * (self.static_radius ** 2) * self.length
        return ElasticTube.power() / displaced_volume

    def _find_modal_frequencies(self):
        """Calculates the roots of the nonlinear dispersion relationship governing the elastic tube
        
        Args:
            None

        Returns:
            modal_frequency_array (np array): row of tube modal frequencies found from the dispersion relationship;
                                                size is the number of modes mode_count

        """
        return np.zeros(shape=self.mode_count)

    def _mode_shape_product(self):
        pass

    def _mode_shape_derivative_product(self):
        pass
