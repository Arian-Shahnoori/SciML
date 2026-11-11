import numpy as np


def apply_temperature_boundary_conditions(
    temperature,
    mesh,
    inlet_temperature,
    domain_length_y,
    bottom_temperature=0.0,
    top_temperature=1.0,
):
    """Apply temperature boundary conditions using ghost cells."""

    cell_count_x, cell_count_y = mesh
    dy = domain_length_y / cell_count_y

    # Bottom and top walls: fixed temperature.
    temperature[:, 0] = (
        2.0 * bottom_temperature - temperature[:, 1]
    )
    temperature[:, -1] = (
        2.0 * top_temperature - temperature[:, -2]
    )

    # Inlet: prescribed temperature profile.
    for y_index in range(1, cell_count_y + 1):
        y = (y_index - 0.5) * dy
        temperature[0, y_index] = (
            2.0 * inlet_temperature(y) - temperature[1, y_index]
        )

    # Outlet: zero streamwise temperature gradient.
    temperature[cell_count_x + 1, 1:-1] = (
        temperature[cell_count_x, 1:-1]
    )

    return temperature