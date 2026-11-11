from pathlib import Path
import numpy as np

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    CELL_COUNT_X,
    CELL_COUNT_Y,
    PECLET_NUMBER,
    BRINKMAN_NUMBER,
    MEAN_VELOCITY,
    TIME_STEP,
    RESIDUAL_TOLERANCE,
    MAX_ITERATIONS,
    convection_coefficient,
)

from src.initialization import (
    inlet_temperature,
    initialize_temperature,
    initialize_velocity_fields,
)

from src.diagnostics import steady_state_residual_rms
from src.solvers.direct import solve_with_direct


def run_direct_solver():

    # Mesh
    mesh = (CELL_COUNT_X, CELL_COUNT_Y)

    # Initial temperature
    T_initial = initialize_temperature(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
    )

    # Velocity fields
    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
    )

    # Run the existing direct solver
    (
        temperature,
        iteration_history,
        residual_history,
        time_history,
        computation_time,
        temperature_history,
        snapshot_iterations,
    ) = solve_with_direct(
        initial_temperature=T_initial,
        time_step=TIME_STEP,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        convection_coefficient=convection_coefficient,
    )

    # Calculate final steady-state residual
    final_residual = steady_state_residual_rms(
        mesh=mesh,
        temperature=temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        convection_coefficient=convection_coefficient,
    )

    # Save final temperature field
    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)

    np.save(output_directory / "T_steady.npy", temperature)

    # Print results
    print("\nDirect solver results")
    print("---------------------")
    print(f"Mesh:              {mesh}")
    print(f"Iterations:        {len(iteration_history)}")
    print(f"Final residual:    {final_residual:.6e}")
    print(f"Tolerance:         {RESIDUAL_TOLERANCE:.6e}")
    print(f"Computation time:  {computation_time:.3f} s")
    print(
        f"Temperature range: "
        f"{temperature.min():.6f} to {temperature.max():.6f}"
    )

    if final_residual <= RESIDUAL_TOLERANCE:
        print("Convergence:       CONVERGED")
    else:
        print("Convergence:       NOT CONVERGED")

    print(f"Saved:             {output_directory / 'T_steady.npy'}")

    return temperature