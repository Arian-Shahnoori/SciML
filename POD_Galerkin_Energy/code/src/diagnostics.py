import numpy as np



def l2_error(reference, numerical):
    """Calculate the root-mean-square difference between two arrays."""
    reference = np.asarray(reference, dtype=float)
    numerical = np.asarray(numerical, dtype=float)

    if reference.shape != numerical.shape:
        raise ValueError("Reference and numerical arrays must have the same shape.")

    return float(np.sqrt(np.mean((numerical - reference) ** 2)))


def steady_state_residual_rms(
    mesh,
    temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    convection_coefficient: float = 1.0,
):
    """Calculate the RMS of the discrete steady energy-equation residual."""
    cell_count_x, cell_count_y = mesh
    rhs_matrix = np.zeros_like(temperature, dtype=float)

    for y_index in range(1, cell_count_y + 1):
        rhs_matrix[:, y_index] = build_rhs(
            cell_counts=mesh,
            y_index=y_index,
            temperature=temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

    residual = rhs_matrix[1:-1, 1:-1]

    return float(np.sqrt(np.mean(residual**2)))


def build_rhs(
    cell_counts, y_index, temperature,
    peclet_number, brinkman_number,
    u_velocity, v_velocity, domain_length_x, domain_length_y,
    convection_coefficient: float = 1.0
):
    cell_count_x, cell_count_y = cell_counts

    dx = domain_length_x / cell_count_x
    dy = domain_length_y / cell_count_y

    rhs = np.zeros(cell_count_x + 2, dtype=float)

    for x_index in range(1, cell_count_x + 1):

        # Advection and diffusion in the x-direction
        x_flux_difference = (
            u_velocity[x_index + 1, y_index]
            * temperature[x_index + 1, y_index]
            - u_velocity[x_index - 1, y_index]
            * temperature[x_index - 1, y_index]
        )

        x_diffusion_difference = (
            temperature[x_index + 1, y_index]
            - 2 * temperature[x_index, y_index]
            + temperature[x_index - 1, y_index]
        )

        # Advection and diffusion in the y-direction
        y_flux_difference = (
            v_velocity[x_index, y_index + 1]
            * temperature[x_index, y_index + 1]
            - v_velocity[x_index, y_index - 1]
            * temperature[x_index, y_index - 1]
        )

        y_diffusion_difference = (
            temperature[x_index, y_index + 1]
            - 2 * temperature[x_index, y_index]
            + temperature[x_index, y_index - 1]
        )

        transport_rate = (
            -convection_coefficient * x_flux_difference / (2 * dx)
            + x_diffusion_difference / (peclet_number * dx**2)
            - convection_coefficient * y_flux_difference / (2 * dy)
            + y_diffusion_difference / (peclet_number * dy**2)
        )

        # Velocity gradients for viscous dissipation
        du_dx = (
            u_velocity[x_index + 1, y_index]
            - u_velocity[x_index - 1, y_index]
        ) / (2 * dx)

        dv_dy = (
            v_velocity[x_index, y_index + 1]
            - v_velocity[x_index, y_index - 1]
        ) / (2 * dy)

        dv_dx = (
            v_velocity[x_index + 1, y_index]
            - v_velocity[x_index - 1, y_index]
        ) / (2 * dx)

        du_dy = (
            u_velocity[x_index, y_index + 1]
            - u_velocity[x_index, y_index - 1]
        ) / (2 * dy)

        viscous_heating = (brinkman_number / peclet_number) * (
            2 * du_dx**2
            + 2 * dv_dy**2
            + (dv_dx + du_dy)**2
        )

        rhs[x_index] = transport_rate + viscous_heating

    return rhs
