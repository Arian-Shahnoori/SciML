from pathlib import Path
import csv
import numpy as np

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    CELL_COUNT_X,
    CELL_COUNT_Y,
    MEAN_VELOCITY,
    TIME_STEP,
    RESIDUAL_TOLERANCE,
    MAX_ITERATIONS,
)

from src.initialization import (
    inlet_temperature,
    initialize_temperature,
    initialize_velocity_fields,
)

from src.diagnostics import steady_state_residual_rms

# Replace this module path if your solver is stored in another file.
from src.solvers.direct import solve_with_direct


# Fixed convection coefficient
CONVECTION_COEFFICIENT = 1.0


def run_parameter_sweep():

    # -------------------------------------------------
    # 1. Define paths
    # -------------------------------------------------

    project_directory = Path(__file__).resolve().parent

    parameter_file = (
        project_directory / "results" / "parameter_grid.csv"
    )

    output_directory = (
        Path(__file__).resolve().parent / "results"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    if not parameter_file.exists():
        raise FileNotFoundError(
            f"Parameter grid not found: {parameter_file}"
        )

    # -------------------------------------------------
    # 2. Read parameter combinations
    # -------------------------------------------------

    with open(parameter_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        parameter_cases = list(reader)

    if not parameter_cases:
        raise ValueError("The parameter grid is empty.")

    print(f"Number of parameter cases: {len(parameter_cases)}")

    # -------------------------------------------------
    # 3. Initialize mesh and fields
    # -------------------------------------------------

    mesh = (CELL_COUNT_X, CELL_COUNT_Y)

    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
    )

    # -------------------------------------------------
    # 4. Run each parameter combination
    # -------------------------------------------------

    summary = []

    for case_number, parameters in enumerate(parameter_cases, start=1):

        case_id = int(parameters["case_id"])
        mu = float(parameters["mu"])
        brinkman_number = float(parameters["Br"])

        if mu <= 0:
            raise ValueError(f"Invalid mu in case {case_id}: {mu}")

        if brinkman_number < 0:
            raise ValueError(
                f"Invalid Br in case {case_id}: {brinkman_number}"
            )

        # Since mu = c * Pe and c = 1:
        peclet_number = mu / CONVECTION_COEFFICIENT

        print("\n" + "=" * 50)
        print(f"Case {case_number}/{len(parameter_cases)}")
        print(f"Case ID: {case_id}")
        print(f"mu:      {mu:.6f}")
        print(f"Br:      {brinkman_number:.6f}")
        print(f"Pe:      {peclet_number:.6f}")
        print("=" * 50)

        # Run the existing solver
        (
            temperature,
            iteration_history,
            residual_history,
            time_history,
            computation_time,
            temperature_history,
            snapshot_iterations,
        ) = solve_with_direct(
            initial_temperature=initial_temperature.copy(),
            time_step=TIME_STEP,
            mesh=mesh,
            inlet_temperature=inlet_temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
            mean_velocity=MEAN_VELOCITY,
            residual_tolerance=RESIDUAL_TOLERANCE,
            max_iterations=MAX_ITERATIONS,
            convection_coefficient=CONVECTION_COEFFICIENT,
        )

        # -------------------------------------------------
        # 5. Calculate final steady-state residual
        # -------------------------------------------------

        final_residual = steady_state_residual_rms(
            mesh=mesh,
            temperature=temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
            convection_coefficient=CONVECTION_COEFFICIENT,
        )

        converged = final_residual <= RESIDUAL_TOLERANCE

        # -------------------------------------------------
        # 6. Save temperature field
        # -------------------------------------------------

        temperature_filename = f"T_case_{case_id:03d}.npy"

        np.save(
            output_directory / temperature_filename,
            temperature,
        )

        # -------------------------------------------------
        # 7. Store case information
        # -------------------------------------------------

        summary.append({
            "case_id": case_id,
            "mu": mu,
            "Br": brinkman_number,
            "Pe": peclet_number,
            "c": CONVECTION_COEFFICIENT,
            "iterations": len(iteration_history),
            "final_residual": final_residual,
            "converged": converged,
            "computation_time": computation_time,
            "temperature_file": temperature_filename,
        })

        print(f"Iterations:     {len(iteration_history)}")
        print(f"Final residual: {final_residual:.6e}")
        print(f"Converged:      {converged}")
        print(f"Saved:          {temperature_filename}")

    # -------------------------------------------------
    # 8. Save summary of all simulations
    # -------------------------------------------------

    summary_file = output_directory / "simulation_summary.csv"

    with open(summary_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary)

    successful_cases = sum(item["converged"] for item in summary)

    print("\n" + "=" * 50)
    print("PARAMETER SWEEP COMPLETED")
    print("=" * 50)
    print(f"Total cases:       {len(summary)}")
    print(f"Converged cases:   {successful_cases}")
    print(f"Non-converged:     {len(summary) - successful_cases}")
    print(f"Results directory: {output_directory}")

    return summary