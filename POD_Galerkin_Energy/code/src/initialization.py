import numpy as np


def inlet_temperature(y):
    """Prescribed inlet temperature profile."""
    return y


def initialize_temperature(mesh, domain_length_y):
    """Initialize T(x,y,0) = y, including ghost-cell locations."""

    _, cell_count_y = mesh
    dy = domain_length_y / cell_count_y

    y = (np.arange(cell_count_y + 2) - 0.5) * dy

    temperature = np.tile(y, (mesh[0] + 2, 1))

    return temperature


def initialize_velocity_fields(
    mesh,
    domain_length_y,
    mean_velocity,
):
    """Initialize prescribed fully developed velocity fields."""

    cell_count_x, cell_count_y = mesh
    dy = domain_length_y / cell_count_y

    y = (np.arange(cell_count_y + 2) - 0.5) * dy

    u_profile = 6.0 * mean_velocity * y * (1.0 - y)

    u_velocity = np.tile(
        u_profile,
        (cell_count_x + 2, 1),
    )

    v_velocity = np.zeros_like(u_velocity)

    return u_velocity, v_velocity