from time import perf_counter
import time
import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import splu

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    PECLET_NUMBER,
    BRINKMAN_NUMBER,
    MEAN_VELOCITY,
    RESIDUAL_TOLERANCE,
    MAX_ITERATIONS,
)

from src.boundary_conditions import apply_temperature_boundary_conditions
from src.diagnostics import (
    steady_state_residual_rms,
)


# =========================================================
# 1. Assemble the source-free spatial matrix
# =========================================================

def build_spatial_matrix(
    nx,
    ny,
    length_x=DOMAIN_LENGTH_X,
    length_y=DOMAIN_LENGTH_Y,
    peclet_number=PECLET_NUMBER,
    mean_velocity=MEAN_VELOCITY,
    u_velocity=None,
    v_velocity=None,
    convection_coefficient=1.0,
):
    """
    Assemble A for the source-free spatial operator:

        dT/dt = A @ T + b

    Uses second-order central differences and the same ghost-cell
    boundary relations as apply_temperature_boundary_conditions().

    Matrix ordering matches:

        temperature[1:-1, 1:-1].ravel(order="C")

    Thus, the wall-normal index varies fastest.
    """

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive.")

    if length_x <= 0 or length_y <= 0:
        raise ValueError("Domain lengths must be positive.")

    if peclet_number <= 0:
        raise ValueError("Peclet number must be positive.")

    dx = length_x / nx
    dy = length_y / ny

    diffusion_x = 1.0 / (peclet_number * dx**2)
    diffusion_y = 1.0 / (peclet_number * dy**2)

    # Supply the prescribed parabolic velocity field if needed.
    if u_velocity is None:
        y_all = (np.arange(ny + 2) - 0.5) * dy
        u_profile = (
            6.0 * mean_velocity * y_all * (1.0 - y_all)
        )
        u_velocity = np.broadcast_to(
            u_profile[None, :],
            (nx + 2, ny + 2),
        ).copy()

    if v_velocity is None:
        v_velocity = np.zeros((nx + 2, ny + 2), dtype=float)

    if u_velocity.shape != (nx + 2, ny + 2):
        raise ValueError("Unexpected shape for u_velocity.")

    if v_velocity.shape != (nx + 2, ny + 2):
        raise ValueError("Unexpected shape for v_velocity.")

    number_of_cells = nx * ny
    A = lil_matrix(
        (number_of_cells, number_of_cells),
        dtype=float,
    )

    def index(i, j):
        return i * ny + j

    diagonal = -2.0 * (diffusion_x + diffusion_y)

    for i in range(nx):
        for j in range(ny):

            # Indices in the original array, which includes ghosts.
            x_idx = i + 1
            y_idx = j + 1

            # Coefficients consistent with the conservative central
            # convection terms used by build_rhs().
            a_east = (
                diffusion_x
                - convection_coefficient
                * u_velocity[x_idx + 1, y_idx]
                / (2.0 * dx)
            )

            a_west = (
                diffusion_x
                + convection_coefficient
                * u_velocity[x_idx - 1, y_idx]
                / (2.0 * dx)
            )

            a_north = (
                diffusion_y
                - convection_coefficient
                * v_velocity[x_idx, y_idx + 1]
                / (2.0 * dy)
            )

            a_south = (
                diffusion_y
                + convection_coefficient
                * v_velocity[x_idx, y_idx - 1]
                / (2.0 * dy)
            )

            k = index(i, j)
            A[k, k] += diagonal

            # East: zero-gradient outlet.
            # T_ghost = T_adjacent.
            if i + 1 < nx:
                A[k, index(i + 1, j)] += a_east
            else:
                A[k, k] += a_east

            # West: prescribed inlet.
            # T_ghost = 2*T_in - T_adjacent.
            if i > 0:
                A[k, index(i - 1, j)] += a_west
            else:
                A[k, k] -= a_west

            # North: fixed top-wall temperature.
            # T_ghost = 2*T_top - T_adjacent.
            if j + 1 < ny:
                A[k, index(i, j + 1)] += a_north
            else:
                A[k, k] -= a_north

            # South: fixed bottom-wall temperature.
            # T_ghost = 2*T_bottom - T_adjacent.
            if j > 0:
                A[k, index(i, j - 1)] += a_south
            else:
                A[k, k] -= a_south

    return A.tocsr()


#rhs of APF method from before
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


# =========================================================
# 2. Assemble the affine forcing vector
# =========================================================

def build_forcing_field(
    mesh,
    inlet_temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    convection_coefficient=1.0,
):
    """
    Build b in:

        dT/dt = A @ T + b

    Evaluate the existing RHS with all physical-cell temperatures
    set to zero, after applying the boundary conditions.

    Because the velocity field and viscous-heating source are
    independent of temperature, and the boundary conditions are
    affine, this isolates the constant contributions.

    build_rhs() returns one x-direction line at a time. The result
    is explicitly mapped into the same ordering as the spatial matrix.
    """

    nx, ny = mesh

    zero_temperature = np.zeros(
        (nx + 2, ny + 2),
        dtype=float,
    )

    apply_temperature_boundary_conditions(
        temperature=zero_temperature,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=domain_length_y,
    )

    forcing = np.zeros(nx * ny, dtype=float)
    forcing_field = forcing.reshape((nx, ny), order="C")

    for y_index in range(1, ny + 1):

        rhs_line = build_rhs(
            cell_counts=mesh,
            y_index=y_index,
            temperature=zero_temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

        # x positions 1,...,nx are the physical cells.
        # Assign them into the column corresponding to this y index.
        forcing_field[:, y_index - 1] = rhs_line[1:nx + 1]

    return forcing_field


# =========================================================
# 3. Direct backward-Euler solver
# =========================================================

def solve_with_direct(
    initial_temperature,
    time_step,
    mesh,
    inlet_temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    mean_velocity=3.0,
    residual_tolerance=1e-12,
    max_iterations=100000,
    snapshot_interval=10,
    convection_coefficient=1.0,
):
    """
    March the energy equation with fully coupled implicit Euler:

        (I - dt*A) T_new = T_old + dt*b

    The sparse matrix is factorized once and reused.

    Returns the same seven-element structure as solve_with_afi().
    """

    if time_step <= 0:
        raise ValueError("time_step must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    if snapshot_interval < 1:
        raise ValueError("snapshot_interval must be positive.")

    start_time = perf_counter()

    nx, ny = mesh
    number_of_cells = nx * ny

    # Apply boundary conditions to the initial temperature.
    temperature = np.asarray(
        initial_temperature,
        dtype=float,
    ).copy()

    if temperature.shape != (nx + 2, ny + 2):
        raise ValueError(
            f"Expected temperature shape {(nx + 2, ny + 2)}, "
            f"received {temperature.shape}."
        )

    apply_temperature_boundary_conditions(
        temperature=temperature,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=domain_length_y,
    )

    # Assemble the spatial operator using the actual velocity fields.
    A = build_spatial_matrix(
        nx=nx,
        ny=ny,
        length_x=domain_length_x,
        length_y=domain_length_y,
        peclet_number=peclet_number,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        convection_coefficient=convection_coefficient,
    )

    # Constant forcing, including the viscous-heating source and
    # constant contributions from the prescribed boundary values.
    b = build_forcing_field(
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=peclet_number,
        brinkman_number=brinkman_number,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        convection_coefficient=convection_coefficient,
    )

    b = b.ravel(order="C")

    # Fully implicit Euler:
    #
    # (I - dt*A) T_new = T_old + dt*b
    #
    # Factorize once because A and dt do not change during the run.
    implicit_matrix = (
        eye(number_of_cells, format="csc")
        - time_step * A.tocsc()
    )

    factorized_matrix = splu(implicit_matrix)

    # Initial physical-cell vector, with the matrix's ordering.
    temperature_vector = (
        temperature[1:-1, 1:-1]
        .ravel(order="C")
        .copy()
    )

    # Histories returned through the standard solver interface.
    iteration_history = []
    residual_history = []
    time_history = []

    temperature_history = [temperature.copy()]
    snapshot_iterations = [0]

    for iteration in range(1, max_iterations + 1):

        # Implicit Euler linear solve.
        temperature_vector = factorized_matrix.solve(
            temperature_vector + time_step * b
        )

        if not np.all(np.isfinite(temperature_vector)):
            raise FloatingPointError(
                f"Non-finite temperature at iteration {iteration}."
            )

        # Map the vector back to the physical interior.
        temperature[1:-1, 1:-1] = temperature_vector.reshape(
            (nx, ny),
            order="C",
        )

        # Reconstruct ghost-cell values for the next iteration
        # and for the residual calculation.
        temperature = apply_temperature_boundary_conditions(
            temperature=temperature,
            mesh=mesh,
            inlet_temperature=inlet_temperature,
            domain_length_y=domain_length_y,
        )

        # Evaluate the steady-state residual using the same
        # spatial RHS and convection coefficient.
        residual = steady_state_residual_rms(
            mesh=mesh,
            temperature=temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

        iteration_history.append(iteration)
        residual_history.append(float(residual))

        # Cumulative elapsed time at this iteration.
        time_history.append(time.perf_counter() - start_time)

        converged = (
            residual_tolerance > 0.0
            and residual <= residual_tolerance
        )

        # Store snapshots at the specified interval and always
        # store the final state.
        save_snapshot = (
            iteration % snapshot_interval == 0
            or iteration == max_iterations
            or converged
        )

        if save_snapshot:
            temperature_history.append(temperature.copy())
            snapshot_iterations.append(iteration)

        if converged:
            break

    computation_time = perf_counter() - start_time

    return (
        temperature,
        iteration_history,
        residual_history,
        time_history,
        float(computation_time),
        temperature_history,
        snapshot_iterations,
    )