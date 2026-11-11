from pathlib import Path
import time

import numpy as np
import pandas as pd

from src import config
from src.initialization import initialize_velocity_fields

# Reuse the existing FOM solver function.
from studies.f04_galerkin_rom.f04_rom_for_case import run_fom_case


# ============================================================
# 1. Parameter-space configuration
# ============================================================

MESH = (50, 20)

MU_MIN = 1.0
MU_MAX = 100.0

BR_MIN = 0.01
BR_MAX = 2.0

# 20 x 20 = 400 parameter combinations.
N_MU = 10
N_BR = 10

# mu = c * Pe. Fixing c = 1 gives Pe = mu.
CONVECTION_COEFFICIENT = 1.0


# ============================================================
# 2. Output paths
# ============================================================

TRAINING_SWEEP_DIRECTORY = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "large_training_sweep"
)

CASE_RESULTS_DIRECTORY = TRAINING_SWEEP_DIRECTORY / "cases"

SNAPSHOT_MATRIX_PATH = (
    TRAINING_SWEEP_DIRECTORY / "snapshot_matrix.npy"
)

SNAPSHOT_CASE_IDS_PATH = (
    TRAINING_SWEEP_DIRECTORY / "snapshot_case_ids.csv"
)

PARAMETER_GRID_PATH = (
    TRAINING_SWEEP_DIRECTORY / "parameter_grid.csv"
)

SUMMARY_PATH = (
    TRAINING_SWEEP_DIRECTORY / "simulation_summary.csv"
)


# Reuse successfully completed cases when rerunning the script.
RESUME_FROM_EXISTING_CASES = True


# ============================================================
# 3. Utility functions
# ============================================================

def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


def format_value(value):
    """Format a number for use in a filename."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_reference_temperature(mesh):
    """
    Construct the reference temperature field:

        T_ref = y

    Evaluated at physical cell centres.
    """

    nx, ny = mesh

    dy = config.DOMAIN_LENGTH_Y / ny

    y_centres = (np.arange(ny) + 0.5) * dy

    return np.broadcast_to(
        y_centres[None, :],
        (nx, ny),
    ).copy()


# ============================================================
# 4. Generate parameter combinations
# ============================================================

def build_parameter_grid(n_mu=N_MU, n_br=N_BR):
    """
    Generate a uniform grid in (mu, Br).

    mu = c * Pe

    Since c is fixed, Pe = mu / c.
    """

    if n_mu < 1 or n_br < 1:
        raise ValueError(
            "N_MU and N_BR must both be at least 1."
        )

    if MU_MIN <= 0 or MU_MAX < MU_MIN:
        raise ValueError("Invalid mu range.")

    if BR_MIN < 0 or BR_MAX < BR_MIN:
        raise ValueError("Invalid Brinkman number range.")

    if CONVECTION_COEFFICIENT <= 0:
        raise ValueError(
            "CONVECTION_COEFFICIENT must be positive."
        )

    mu_values = np.linspace(
        MU_MIN,
        MU_MAX,
        n_mu,
    )

    br_values = np.linspace(
        BR_MIN,
        BR_MAX,
        n_br,
    )

    cases = []

    case_number = 0

    for mu in mu_values:
        for br in br_values:

            case_number += 1

            peclet_number = (
                mu / CONVECTION_COEFFICIENT
            )

            cases.append({
                "case_id": f"train_{case_number:04d}",
                "effective_parameter_mu": float(mu),
                "peclet_number": float(peclet_number),
                "brinkman_number": float(br),
                "convection_coefficient": float(
                    CONVECTION_COEFFICIENT
                ),
            })

    return cases


# ============================================================
# 5. Case file handling
# ============================================================

def make_case_path(case):
    """Build the output path for an individual case."""

    filename = (
        f"{case['case_id']}"
        f"_mu{format_value(case['effective_parameter_mu'])}"
        f"_Br{format_value(case['brinkman_number'])}.npz"
    )

    return CASE_RESULTS_DIRECTORY / filename


def load_existing_case(case, case_path, mesh):
    """
    Load a previously completed case.

    Check its parameter values and mesh before reusing it.
    """

    with np.load(case_path, allow_pickle=False) as saved:

        if not bool(saved["converged"]):
            return None

        if not np.array_equal(
            saved["mesh"],
            np.asarray(mesh),
        ):
            return None

        expected_parameters = {
            "effective_parameter_mu":
                case["effective_parameter_mu"],
            "peclet_number":
                case["peclet_number"],
            "brinkman_number":
                case["brinkman_number"],
            "convection_coefficient":
                case["convection_coefficient"],
        }

        for key, expected_value in expected_parameters.items():

            if not np.isclose(
                float(saved[key]),
                expected_value,
            ):
                return None

        temperature = np.asarray(
            saved["temperature"],
            dtype=float,
        )

        theta = np.asarray(
            saved["theta"],
            dtype=float,
        )

        if temperature.shape != mesh or theta.shape != mesh:
            return None

        if not (
            np.all(np.isfinite(temperature))
            and np.all(np.isfinite(theta))
        ):
            return None

        return {
            "temperature": temperature,
            "theta": theta,
            "final_residual": float(
                saved["final_residual"]
            ),
            "iterations": int(
                saved["iterations"]
            ),
            "computation_time_s": float(
                saved["computation_time_s"]
            ),
        }


def save_case(case, result, mesh, case_path):
    """Save one converged FOM case."""

    np.savez_compressed(
        case_path,

        case_id=case["case_id"],

        effective_parameter_mu=case[
            "effective_parameter_mu"
        ],

        peclet_number=case["peclet_number"],
        brinkman_number=case["brinkman_number"],
        convection_coefficient=case[
            "convection_coefficient"
        ],

        mesh=np.asarray(mesh),

        temperature=result["temperature"],
        theta=result["theta"],

        converged=True,
        final_residual=result["final_residual"],
        iterations=result["iterations"],
        computation_time_s=result["computation_time_s"],
    )


# ============================================================
# 6. Main training-case generation
# ============================================================

def run_training_case_generation(
    n_mu=N_MU,
    n_br=N_BR,
):
    """Run the FOM parameter sweep and assemble POD snapshots."""

    start_total = time.perf_counter()

    TRAINING_SWEEP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CASE_RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh = MESH
    nx, ny = mesh

    number_of_cells = nx * ny

    # Generate the parameter combinations.
    cases = build_parameter_grid(
        n_mu=n_mu,
        n_br=n_br,
    )

    # Save the complete parameter grid before running simulations.
    pd.DataFrame(cases).to_csv(
        PARAMETER_GRID_PATH,
        index=False,
    )

    # Reference temperature for the lifting.
    reference_temperature = make_reference_temperature(mesh)

    # Velocity is fixed across all cases.
    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    records = []

    snapshot_columns = []
    snapshot_case_records = []

    print("\n" + "=" * 76)
    print("LARGE FOM TRAINING-CASE GENERATION")
    print("=" * 76)

    print(f"Mesh: {nx} x {ny}")
    print(f"mu range: [{MU_MIN:g}, {MU_MAX:g}]")
    print(f"Br range: [{BR_MIN:g}, {BR_MAX:g}]")

    print(f"Number of mu samples: {n_mu}")
    print(f"Number of Br samples: {n_br}")
    print(f"Total cases: {len(cases)}")

    print(
        f"Convection coefficient: "
        f"{CONVECTION_COEFFICIENT:g}"
    )

    print(
        f"Residual tolerance: "
        f"{config.RESIDUAL_TOLERANCE:.3e}"
    )

    print(
        f"Maximum iterations: "
        f"{config.MAX_ITERATIONS}"
    )

    print(f"Output directory: {TRAINING_SWEEP_DIRECTORY}")

    # --------------------------------------------------------
    # Loop over the parameter combinations
    # --------------------------------------------------------

    for case_index, case in enumerate(cases, start=1):

        case_id = case["case_id"]

        case_path = make_case_path(case)

        print("\n" + "-" * 76)
        print(
            f"Case {case_index}/{len(cases)}: "
            f"{case_id}"
        )

        print(
            f"mu = {case['effective_parameter_mu']:.6g}, "
            f"Br = {case['brinkman_number']:.6g}, "
            f"Pe = {case['peclet_number']:.6g}, "
            f"c = {case['convection_coefficient']:.6g}"
        )

        record = {
            **case,
            "status": "failed",
            "final_residual": np.nan,
            "iterations": np.nan,
            "computation_time_s": np.nan,
            "result_file": "",
        }

        try:

            case_result = None

            # --------------------------------------------
            # Reuse an existing converged case if possible
            # --------------------------------------------

            if (
                RESUME_FROM_EXISTING_CASES
                and case_path.exists()
            ):

                try:

                    case_result = load_existing_case(
                        case=case,
                        case_path=case_path,
                        mesh=mesh,
                    )

                    if case_result is not None:
                        print("Loaded existing converged case.")

                except (
                    OSError,
                    ValueError,
                    KeyError,
                ) as exc:

                    print(
                        "Could not reuse existing result. "
                        f"Recomputing case. Reason: {exc}"
                    )

            # --------------------------------------------
            # Run FOM if no reusable result exists
            # --------------------------------------------

            if case_result is None:

                fom_case = {
                    "peclet_number": case["peclet_number"],
                    "brinkman_number": case["brinkman_number"],
                    "convection_coefficient": case[
                        "convection_coefficient"
                    ],
                }

                fom_result = run_fom_case(
                    case=fom_case,
                    mesh=mesh,
                    u_velocity=u_velocity,
                    v_velocity=v_velocity,
                )

                temperature = np.asarray(
                    fom_result["temperature"],
                    dtype=float,
                )

                if temperature.shape != mesh:
                    raise ValueError(
                        f"Unexpected temperature shape: "
                        f"{temperature.shape}; expected {mesh}."
                    )

                if not np.all(np.isfinite(temperature)):
                    raise FloatingPointError(
                        "Non-finite values in FOM temperature."
                    )

                theta = (
                    temperature - reference_temperature
                )

                case_result = {
                    "temperature": temperature,
                    "theta": theta,
                    "final_residual": float(
                        fom_result["final_residual"]
                    ),
                    "iterations": int(
                        fom_result["iterations"]
                    ),
                    "computation_time_s": float(
                        fom_result["computation_time"]
                    ),
                }

                if not fom_result["converged"]:

                    record["status"] = "FOM_not_converged"

                    print(
                        "FOM did not converge. "
                        f"Final residual: "
                        f"{case_result['final_residual']:.6e}"
                    )

                else:

                    save_case(
                        case=case,
                        result=case_result,
                        mesh=mesh,
                        case_path=case_path,
                    )

            # --------------------------------------------
            # Add only converged cases to the snapshots
            # --------------------------------------------

            if (
                case_result is not None
                and record["status"] != "FOM_not_converged"
            ):

                record.update({
                    "status": "success",
                    "final_residual": case_result[
                        "final_residual"
                    ],
                    "iterations": case_result["iterations"],
                    "computation_time_s": case_result[
                        "computation_time_s"
                    ],
                    "result_file": str(case_path),
                })

                # Each snapshot is one column.
                # Flattening order matches the POD workflow.
                snapshot_columns.append(
                    case_result["theta"].ravel(order="C")
                )

                snapshot_case_records.append({
                    "snapshot_column": len(snapshot_columns) - 1,
                    "case_id": case_id,
                    "effective_parameter_mu": case[
                        "effective_parameter_mu"
                    ],
                    "peclet_number": case["peclet_number"],
                    "brinkman_number": case[
                        "brinkman_number"
                    ],
                    "convection_coefficient": case[
                        "convection_coefficient"
                    ],
                    "result_file": str(case_path),
                })

                print(
                    f"Converged. Residual: "
                    f"{case_result['final_residual']:.6e}"
                )

                print(
                    f"Iterations: {case_result['iterations']}"
                )

            elif record["status"] != "FOM_not_converged":

                record["status"] = "failed"

        except Exception as exc:

            record["status"] = (
                f"error: {type(exc).__name__}: {exc}"
            )

            print(
                f"FAILED: {type(exc).__name__}: {exc}"
            )

        # Save progress after every case.
        records.append(record)

        pd.DataFrame(records).to_csv(
            SUMMARY_PATH,
            index=False,
        )

    # ========================================================
    # 7. Assemble the snapshot matrix
    # ========================================================

    if snapshot_columns:

        snapshot_matrix = np.column_stack(
            snapshot_columns
        )

    else:

        snapshot_matrix = np.empty(
            (number_of_cells, 0),
            dtype=float,
        )

    # Save the snapshot matrix.
    np.save(
        SNAPSHOT_MATRIX_PATH,
        snapshot_matrix,
    )

    # Save the metadata corresponding to each snapshot column.
    snapshot_case_ids = pd.DataFrame(
        snapshot_case_records
    )

    snapshot_case_ids.to_csv(
        SNAPSHOT_CASE_IDS_PATH,
        index=False,
    )

    # ========================================================
    # 8. Final summary
    # ========================================================

    summary = pd.DataFrame(records)

    successful_count = int(
        (summary["status"] == "success").sum()
    )

    failed_count = len(summary) - successful_count

    total_time = time.perf_counter() - start_total

    print("\n" + "=" * 76)
    print("TRAINING-CASE GENERATION SUMMARY")
    print("=" * 76)

    print(f"Total cases attempted: {len(cases)}")
    print(f"Successful cases: {successful_count}")
    print(f"Failed/non-converged cases: {failed_count}")

    print(
        f"Snapshot matrix shape: "
        f"{snapshot_matrix.shape}"
    )

    print(
        f"Total elapsed time: {total_time:.2f} s"
    )

    print("\nGenerated files:")
    print(f"Parameter grid: {PARAMETER_GRID_PATH}")
    print(f"Simulation summary: {SUMMARY_PATH}")
    print(f"Snapshot matrix: {SNAPSHOT_MATRIX_PATH}")
    print(f"Snapshot case IDs: {SNAPSHOT_CASE_IDS_PATH}")

    print("\nGeneration completed.")

    return {
        "summary": summary,
        "snapshot_matrix": snapshot_matrix,
        "snapshot_case_ids": snapshot_case_ids,
    }


# ============================================================
# 9. Run
# ============================================================

if __name__ == "__main__":

    training_data = run_training_case_generation(
        n_mu=N_MU,
        n_br=N_BR,
    )