import numpy as np


def create_grid(mesh, domain_length_x, domain_length_y):
    """Create cell-center coordinates and grid spacing."""

    cell_count_x, cell_count_y = mesh

    dx = domain_length_x / cell_count_x
    dy = domain_length_y / cell_count_y

    x = (np.arange(cell_count_x) + 0.5) * dx
    y = (np.arange(cell_count_y) + 0.5) * dy

    return x, y, dx, dy