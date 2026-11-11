from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.initialization import (
    initialize_temperature,
    initialize_velocity_fields,
)
from src.solvers.direct import solve_with_direct

from studies.f04_galerkin_rom.f01_operators import build_operator_components
from studies.f04_galerkin_rom.f02_reduced_operators import build_reduced_operators
from studies.f04_galerkin_rom.f03_solve_reduced import (
    construct_reduced_system,
    solve_reduced_system,
)


# ============================================================
# 1. Validation configuration
# ============================================================

MESH = (50, 20)
N_MODES = 3

POD_MODES_PATH = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "pod_modes.npy"
)

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f04_galerkin_rom"
    / "results"
    / f"case_pod_{N_MODES}"
)

CASE_RESULTS_DIRECTORY = OUTPUT_DIRECTORY / "cases"
FIGURES_DIRECTORY = OUTPUT_DIRECTORY / "figures"

CONVECTION_COEFFICIENT = 1.0


def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


# These are example validation points.
# Make sure none of their (c * Pe, Br) combinations were used
# to construct the POD snapshot matrix.
VALIDATION_CASES = [
    {
        "case_id": "val_01",
        "peclet_number": 17.5,
        "brinkman_number": 0.035,
        "convection_coefficient": 1.0,
    },
    {
        "case_id": "val_02",
        "peclet_number": 17.5,
        "brinkman_number": 0.135,
        "convection_coefficient": 1.0,
    },
    {
        "case_id": "val_03",
        "peclet_number": 17.5,
        "brinkman_number": 0.10,
        "convection_coefficient": 0.7,
    },
    {
        "case_id": "val_04",
        "peclet_number": 17.5,
        "brinkman_number": 0.16,
        "convection_coefficient": 1.3,
    },
    {
        "case_id": "val_05",
        "peclet_number": 10.0,
        "brinkman_number": 0.04,
        "convection_coefficient": 1.0,
    },
]


# ============================================================
# 2. Utility functions
# ============================================================

def format_value(value):
    """Format a numerical value for use in a filename."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def safe_filename(value):
    """Remove characters unsuitable for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def extract_physical_temperature(temperature, mesh):
    """
    Extract physical cells from a temperature array.

    Supports either:
        (nx + 2, ny + 2): field including ghost cells
        (nx, ny): physical field only
    """
    nx, ny = mesh
    temperature = np.asarray(temperature, dtype=float)

    if temperature.shape == (nx + 2, ny + 2):
        return temperature[1:-1, 1:-1].copy()

    if temperature.shape == (nx, ny):
        return temperature.copy()

    raise ValueError(
        f"Unexpected temperature shape {temperature.shape}; "
        f"expected {(nx + 2, ny + 2)} or {(nx, ny)}."
    )


def calculate_error_metrics(fom_temperature, rom_temperature):
    """Calculate error measures over the physical domain."""

    fom = np.asarray(fom_temperature, dtype=float)
    rom = np.asarray(rom_temperature, dtype=float)

    if fom.shape != rom.shape:
        raise ValueError(
            f"FOM and ROM shapes differ: {fom.shape} and {rom.shape}."
        )

    if not np.all(np.isfinite(fom)) or not np.all(np.isfinite(rom)):
        raise FloatingPointError(
            "Non-finite values found in the FOM or ROM temperature."
        )

    error = fom - rom

    relative_l2_error = (
        np.linalg.norm(error.ravel())
        / max(np.linalg.norm(fom.ravel()), 1e-14)
    )

    rmse = np.sqrt(np.mean(error**2))
    maximum_absolute_error = np.max(np.abs(error))

    return {
        "error_field": error,
        "relative_l2_error": float(relative_l2_error),
        "rmse": float(rmse),
        "maximum_absolute_error": float(maximum_absolute_error),
    }


# ============================================================
# 3. Run the full-order model
# ============================================================

def run_fom_case(case, mesh, u_velocity, v_velocity):
    """Run the existing direct solver for one parameter case."""

    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
    )

    result = solve_with_direct(
        initial_temperature=initial_temperature,
        time_step=config.TIME_STEP,
        mesh=mesh,
        inlet_temperature=linear_inlet_temperature,
        peclet_number=case["peclet_number"],
        brinkman_number=case["brinkman_number"],
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=config.DOMAIN_LENGTH_X,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
        residual_tolerance=config.RESIDUAL_TOLERANCE,
        max_iterations=config.MAX_ITERATIONS,
        snapshot_interval=config.MAX_ITERATIONS + 1,
        convection_coefficient=case["convection_coefficient"],
    )

    if len(result) != 7:
        raise ValueError(
            f"Expected a seven-element solver result; received {len(result)}."
        )

    temperature = extract_physical_temperature(result[0], mesh)

    iteration_history = result[1]
    residual_history = np.asarray(result[2], dtype=float)
    computation_time = float(result[4])

    final_residual = (
        float(residual_history[-1])
        if residual_history.size
        else float("nan")
    )

    converged = (
        np.all(np.isfinite(temperature))
        and np.isfinite(final_residual)
        and final_residual <= config.RESIDUAL_TOLERANCE
    )

    return {
        "temperature": temperature,
        "converged": bool(converged),
        "final_residual": final_residual,
        "iterations": len(iteration_history),
        "computation_time": computation_time,
    }


# ============================================================
# 4. Run the reduced-order model
# ============================================================

def run_rom_case(
    case,
    mesh,
    u_velocity,
    v_velocity,
    pod_modes,
    n_modes,
):
    """Construct and solve the ROM for one parameter case."""

    start_time = time.perf_counter()

    # Build full-order operator components for this case.
    full_operators = build_operator_components(
        mesh=mesh,
        domain_length_x=config.DOMAIN_LENGTH_X,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        peclet_number=case["peclet_number"],
        mean_velocity=config.MEAN_VELOCITY,
        convection_coefficient=case["convection_coefficient"],
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        inlet_temperature=linear_inlet_temperature,
    )

    # Project the full-order operators onto the POD basis.
    reduced_operators = build_reduced_operators(
        operators=full_operators,
        pod_modes=pod_modes,
        n_modes=n_modes,
    )

    # Construct the reduced steady-state equation.
    reduced_system = construct_reduced_system(
        reduced_operators=reduced_operators,
        peclet_number=case["peclet_number"],
        brinkman_number=case["brinkman_number"],
        convection_coefficient=case["convection_coefficient"],
    )

    # Solve for the modal coefficients.
    solution = solve_reduced_system(reduced_system)
    a = solution["a"]

    # Reconstruct the physical temperature field.
    V_r = reduced_operators["V_r"]

    theta_vector = V_r @ a

    T_ref = np.asarray(
        full_operators["T_ref"],
        dtype=float,
    ).ravel(order="C")

    temperature_vector = T_ref + theta_vector

    rom_temperature = temperature_vector.reshape(
        mesh,
        order="C",
    )

    rom_computation_time = time.perf_counter() - start_time

    return {
        "temperature": rom_temperature,
        "theta": theta_vector.reshape(mesh, order="C"),
        "coefficients": a,
        "reduced_system": reduced_system,
        "reduced_operators": reduced_operators,
        "linear_residual": solution["residual_norm"],
        "condition_number": solution["condition_number"],
        "computation_time": float(rom_computation_time),
    }


# ============================================================
# 5. Save an individual validation case
# ============================================================

def save_case_results(case, results, n_modes):
    """Save FOM, ROM, and error fields for one successful case."""

    case_id = safe_filename(case["case_id"])

    filename = (
        f"{case_id}"
        f"_Pe{format_value(case['peclet_number'])}"
        f"_Br{format_value(case['brinkman_number'])}"
        f"_c{format_value(case['convection_coefficient'])}"
        f"_n{n_modes}.npz"
    )

    output_path = CASE_RESULTS_DIRECTORY / filename

    np.savez_compressed(
        output_path,
        fom_temperature=results["fom_temperature"],
        rom_temperature=results["rom_temperature"],
        error_field=results["error_field"],
        theta=results["theta"],
        coefficients=results["coefficients"],
        peclet_number=case["peclet_number"],
        brinkman_number=case["brinkman_number"],
        convection_coefficient=case["convection_coefficient"],
        n_modes=n_modes,
        relative_l2_error=results["relative_l2_error"],
        rmse=results["rmse"],
        maximum_absolute_error=results["maximum_absolute_error"],
        fom_residual=results["fom_residual"],
        fom_iterations=results["fom_iterations"],
        fom_computation_time=results["fom_computation_time"],
        rom_linear_residual=results["rom_linear_residual"],
        rom_condition_number=results["rom_condition_number"],
        rom_computation_time=results["rom_computation_time"],
    )

    return output_path


# ============================================================
# 6. Plot a selected validation case
# ============================================================

def plot_validation_case(
    case_data,
    mesh,
    domain_length_x,
    domain_length_y,
    plot_label,
):
    """Plot FOM, ROM, and absolute error for one validation case."""

    case = case_data["case"]
    fom = case_data["fom_temperature"]
    rom = case_data["rom_temperature"]
    error = case_data["error_field"]

    nx, ny = mesh

    x = (np.arange(nx) + 0.5) * domain_length_x / nx
    y = (np.arange(ny) + 0.5) * domain_length_y / ny

    # Use identical contour levels for the FOM and ROM fields.
    temperature_min = min(np.min(fom), np.min(rom))
    temperature_max = max(np.max(fom), np.max(rom))

    if np.isclose(temperature_min, temperature_max):
        temperature_max = temperature_min + 1e-12

    temperature_levels = np.linspace(
        temperature_min,
        temperature_max,
        41,
    )

    maximum_error = max(float(np.max(np.abs(error))), 1e-14)

    error_levels = np.linspace(
        0.0,
        maximum_error,
        41,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        constrained_layout=True,
    )

    plot_1 = axes[0].contourf(
        x, y, fom.T,
        levels=temperature_levels,
        cmap="viridis",
    )

    plot_2 = axes[1].contourf(
        x, y, rom.T,
        levels=temperature_levels,
        cmap="viridis",
    )

    plot_3 = axes[2].contourf(
        x, y, np.abs(error).T,
        levels=error_levels,
        cmap="magma",
    )

    axes[0].set_title("FOM temperature")
    axes[1].set_title("ROM temperature")
    axes[2].set_title("Absolute error")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    colorbar_temperature = fig.colorbar(
        plot_1,
        ax=axes[:2],
        shrink=0.9,
    )
    colorbar_temperature.set_label("Temperature")

    colorbar_error = fig.colorbar(
        plot_3,
        ax=axes[2],
        shrink=0.9,
    )
    colorbar_error.set_label("Absolute temperature error")

    fig.suptitle(
        f"{plot_label}: {case['case_id']} | "
        f"Pe={case['peclet_number']:g}, "
        f"Br={case['brinkman_number']:g}, "
        f"c={case['convection_coefficient']:g} | "
        f"Relative L2 error="
        f"{case_data['relative_l2_error']:.4e}"
    )

    case_id = safe_filename(case["case_id"])

    plot_path = FIGURES_DIRECTORY / f"{plot_label}_{case_id}.png"

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")


# ============================================================
# 7. Main validation workflow
# ============================================================

def run_rom_for_cases(
    cases=VALIDATION_CASES,
    n_modes=N_MODES,
):
    """Run FOM-ROM validation, save results, and plot min/max errors."""

    if not cases:
        raise ValueError("No validation cases were provided.")

    if not POD_MODES_PATH.exists():
        raise FileNotFoundError(
            f"POD basis not found: {POD_MODES_PATH}. "
            "Set POD_MODES_PATH to the location of pod_modes.npy."
        )

    CASE_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    mesh = MESH
    nx, ny = mesh

    # Load the complete POD basis.
    pod_modes = np.load(POD_MODES_PATH)

    if n_modes < 1 or n_modes > pod_modes.shape[1]:
        raise ValueError(
            f"n_modes must be between 1 and {pod_modes.shape[1]}."
        )

    # Velocity is prescribed and fixed for all validation cases.
    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    records = []
    successful_cases = []

    print("\n" + "=" * 76)
    print("POD-GALERKIN ROM VALIDATION")
    print("=" * 76)

    print(f"Mesh: {nx} x {ny}")
    print(f"Number of retained modes: {n_modes}")
    print(f"Number of validation cases: {len(cases)}")
    print(f"FOM time step: {config.TIME_STEP}")
    print(f"FOM residual tolerance: {config.RESIDUAL_TOLERANCE:.3e}")
    print(f"Maximum FOM iterations: {config.MAX_ITERATIONS}")
    print(f"POD basis: {POD_MODES_PATH}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")

    for case_number, case in enumerate(cases, start=1):

        case_id = case["case_id"]

        print("\n" + "-" * 76)
        print(f"Case {case_number}/{len(cases)}: {case_id}")
        print("-" * 76)

        print(
            f"Pe = {case['peclet_number']:g}, "
            f"Br = {case['brinkman_number']:g}, "
            f"c = {case['convection_coefficient']:g}, "
            f"c*Pe = "
            f"{case['convection_coefficient'] * case['peclet_number']:g}"
        )

        record = {
            "case_id": case_id,
            "peclet_number": case["peclet_number"],
            "brinkman_number": case["brinkman_number"],
            "convection_coefficient": case["convection_coefficient"],
            "effective_parameter_mu": (
                case["peclet_number"]
                * case["convection_coefficient"]
            ),
            "n_modes": n_modes,
            "status": "failed",
            "fom_residual": np.nan,
            "fom_iterations": np.nan,
            "fom_computation_time_s": np.nan,
            "rom_computation_time_s": np.nan,
            "relative_l2_error": np.nan,
            "rmse": np.nan,
            "maximum_absolute_error": np.nan,
            "rom_linear_residual": np.nan,
            "rom_condition_number": np.nan,
        }

        try:
            # ------------------------------
            # Full-order solution
            # ------------------------------
            print("Running FOM...")

            fom_result = run_fom_case(
                case=case,
                mesh=mesh,
                u_velocity=u_velocity,
                v_velocity=v_velocity,
            )

            record["fom_residual"] = fom_result["final_residual"]
            record["fom_iterations"] = fom_result["iterations"]
            record["fom_computation_time_s"] = (
                fom_result["computation_time"]
            )

            print(
                f"FOM residual: "
                f"{fom_result['final_residual']:.6e}"
            )
            print(
                f"FOM iterations: {fom_result['iterations']}"
            )
            print(
                f"FOM computation time: "
                f"{fom_result['computation_time']:.4f} s"
            )

            if not fom_result["converged"]:
                record["status"] = "FOM_not_converged"
                records.append(record)

                print(
                    "SKIPPED: FOM did not satisfy the "
                    "configured residual tolerance."
                )
                continue

            # ------------------------------
            # Reduced-order solution
            # ------------------------------
            print("Running ROM...")

            rom_result = run_rom_case(
                case=case,
                mesh=mesh,
                u_velocity=u_velocity,
                v_velocity=v_velocity,
                pod_modes=pod_modes,
                n_modes=n_modes,
            )

            # ------------------------------
            # Compare temperature fields
            # ------------------------------
            metrics = calculate_error_metrics(
                fom_temperature=fom_result["temperature"],
                rom_temperature=rom_result["temperature"],
            )

            case_results = {
                "case": case,
                "fom_temperature": fom_result["temperature"],
                "rom_temperature": rom_result["temperature"],
                "error_field": metrics["error_field"],
                "theta": rom_result["theta"],
                "coefficients": rom_result["coefficients"],
                "relative_l2_error": metrics["relative_l2_error"],
                "rmse": metrics["rmse"],
                "maximum_absolute_error": (
                    metrics["maximum_absolute_error"]
                ),
                "fom_residual": fom_result["final_residual"],
                "fom_iterations": fom_result["iterations"],
                "fom_computation_time": fom_result["computation_time"],
                "rom_linear_residual": rom_result["linear_residual"],
                "rom_condition_number": rom_result["condition_number"],
                "rom_computation_time": rom_result["computation_time"],
            }

            record.update({
                "status": "success",
                "rom_computation_time_s": (
                    rom_result["computation_time"]
                ),
                "relative_l2_error": (
                    metrics["relative_l2_error"]
                ),
                "rmse": metrics["rmse"],
                "maximum_absolute_error": (
                    metrics["maximum_absolute_error"]
                ),
                "rom_linear_residual": rom_result["linear_residual"],
                "rom_condition_number": rom_result["condition_number"],
            })

            # Save the case fields and diagnostics.
            output_path = save_case_results(
                case=case,
                results=case_results,
                n_modes=n_modes,
            )

            record["result_file"] = str(output_path)

            successful_cases.append(case_results)
            records.append(record)

            print("\nComparison:")
            print(
                f"Relative L2 error: "
                f"{metrics['relative_l2_error']:.6e}"
            )
            print(f"RMSE: {metrics['rmse']:.6e}")
            print(
                f"Maximum absolute error: "
                f"{metrics['maximum_absolute_error']:.6e}"
            )
            print(
                f"ROM linear residual: "
                f"{rom_result['linear_residual']:.6e}"
            )
            print(f"Saved: {output_path}")

        except Exception as exc:
            record["status"] = (
                f"error: {type(exc).__name__}: {exc}"
            )
            records.append(record)

            print(
                f"FAILED: {type(exc).__name__}: {exc}"
            )

    # ========================================================
    # 8. Save summary table
    # ========================================================

    summary = pd.DataFrame(records)

    summary_path = OUTPUT_DIRECTORY / "validation_summary.csv"

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 76)
    print("VALIDATION SUMMARY")
    print("=" * 76)

    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")

    if not successful_cases:
        print("\nNo successful validation cases were available.")
        return {
            "summary": summary,
            "successful_cases": successful_cases,
        }

    # ========================================================
    # 9. Aggregate error statistics
    # ========================================================

    relative_errors = np.array([
        item["relative_l2_error"]
        for item in successful_cases
    ])

    rmse_values = np.array([
        item["rmse"]
        for item in successful_cases
    ])

    maximum_errors = np.array([
        item["maximum_absolute_error"]
        for item in successful_cases
    ])

    print("\n--- Aggregate validation errors ---")
    print(f"Successful cases: {len(successful_cases)}")
    print(f"Mean relative L2 error: {np.mean(relative_errors):.6e}")
    print(f"Median relative L2 error: {np.median(relative_errors):.6e}")
    print(f"Maximum relative L2 error: {np.max(relative_errors):.6e}")
    print(f"Mean RMSE: {np.mean(rmse_values):.6e}")
    print(f"Mean maximum absolute error: {np.mean(maximum_errors):.6e}")

    # ========================================================
    # 10. Find minimum- and maximum-error cases
    # ========================================================

    minimum_index = int(np.argmin(relative_errors))
    maximum_index = int(np.argmax(relative_errors))

    minimum_case = successful_cases[minimum_index]
    maximum_case = successful_cases[maximum_index]

    print("\n--- Minimum-error case ---")
    print(f"Case: {minimum_case['case']['case_id']}")
    print(
        f"Relative L2 error: "
        f"{minimum_case['relative_l2_error']:.6e}"
    )

    print("\n--- Maximum-error case ---")
    print(f"Case: {maximum_case['case']['case_id']}")
    print(
        f"Relative L2 error: "
        f"{maximum_case['relative_l2_error']:.6e}"
    )

    # ========================================================
    # 11. Plot minimum- and maximum-error cases
    # ========================================================

    plot_validation_case(
        case_data=minimum_case,
        mesh=mesh,
        domain_length_x=config.DOMAIN_LENGTH_X,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        plot_label="minimum_error",
    )

    # Avoid generating the exact same figure twice if there is
    # only one successful case.
    if maximum_index != minimum_index:
        plot_validation_case(
            case_data=maximum_case,
            mesh=mesh,
            domain_length_x=config.DOMAIN_LENGTH_X,
            domain_length_y=config.DOMAIN_LENGTH_Y,
            plot_label="maximum_error",
        )

    print("\nValidation completed.")

    return {
        "summary": summary,
        "successful_cases": successful_cases,
        "minimum_error_case": minimum_case,
        "maximum_error_case": maximum_case,
    }


# ============================================================
# 12. Run
# ============================================================

if __name__ == "__main__":
    validation_results = run_rom_for_cases(
        cases=VALIDATION_CASES,
        n_modes=N_MODES,
    )