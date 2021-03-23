from column_class import FlexibleColumn

def plot_column_modes():
    column = FlexibleColumn(radius=10.0, height=200.0, elastic_modulus=4.800e9, density=500.0)
    column.plot_mode_shapes()

plot_column_modes()