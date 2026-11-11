from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import qmc

from src import config
from src.initialization import initialize_velocity_fields

from studies.f04_galerkin_rom.f04_rom_for_case import run_fom_case


# ============================================================
# 1. Configuration
# ============================================================

MESH = (50, 20)

# Parameter-space limits.
MU_MIN = 1.0
MU_MAX = 100.0

BR_MIN = 0.01
BR_MAX = 2.0

CONVECTION_COEFFICIENT = 1.0

# Number of independent test cases.
N_TEST_CASES = 100

# Random seed for reproducibility.
RANDOM_SEED = 20261106

# Reuse previously completed test cases when possible.
RESUME_FROM_EXISTING_CASES = True

# Parameter duplicate detection precision.
PARAMETER_ROUNDING_DECIMALS = 10


# ============================================================
# 2. Input paths
# ============================================================

ORIGINAL_TRAINING_DIRECTORY = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "large_training_sweep"
)

ORIGINAL_PARAMETER_GRID_PATH = (
    ORIGINAL_TRAINING_DIRECTORY / "parameter_grid.csv"
)


AUGMENTED_TRAINING_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "augmented_training_set"
)

AUGMENTED_PARAMETER_GRID_PATH = (
    AUGMENTED_TRAINING_DIRECTORY / "augmented_parameter_grid.csv"
)


# ============================================================
# 3. Output paths
# ============================================================

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "independent_test_set"
)

CASE_RESULTS_DIRECTORY = OUTPUT_DIRECTORY / "cases"

PARAMETER_GRID_PATH = (
    OUTPUT_DIRECTORY / "test_parameter_grid.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "simulation_summary.csv"
)


# ============================================================
# 4. General helper functions
# ============================================================

def format_value(value):
    """Format a number for use in a filename."""

    return (
        f"{value:g}"
        .replace("-", "m")
        .replace(".", "p")
    )


def parameter_key(mu, brinkman_number):
    """
    Create a rounded parameter key for duplicate detection.
    """

    return (
        round(
            float(mu),
            PARAMETER_ROUNDING_DECIMALS,
        ),
        round(
            float(brinkman_number),
            PARAMETER_ROUNDING_DECIMALS,
        ),
    )


def make_reference_temperature(mesh):
    """
    Construct T_ref = y at the physical cell centres.
    """

    nx, ny = mesh

    dy = config.DOMAIN_LENGTH_Y / ny

    y_centres = (
        np.arange(ny) + 0.5
    ) * dy

    return np.broadcast_to(
        y_centres[None, :],
        (nx, ny),
    ).copy()


def make_case_path(case):
    """Construct the output path for one test case."""

    filename = (
        f"{case['case_id']}"
        f"_mu{format_value(case['effective_parameter_mu'])}"
        f"_Br{format_value(case['brinkman_number'])}.npz"
    )

    return CASE_RESULTS_DIRECTORY / filename


def save_summary(records):
    """Save the simulation summary after every case."""

    pd.DataFrame(records).to_csv(
        SUMMARY_PATH,
        index=False,
    )


# ============================================================
# 5. Load existing training parameter combinations
# ============================================================

def load_training_parameter_keys():
    """
    Load parameter combinations from the original and augmented
    training sets.

    These combinations must not appear in the independent test set.
    """

    training_paths = {
        "original": ORIGINAL_PARAMETER_GRID_PATH,
        "augmented": AUGMENTED_PARAMETER_GRID_PATH,
    }

    training_parameter_keys = set()

    print("\n" + "=" * 76)
    print("LOADING EXISTING TRAINING PARAMETERS")
    print("=" * 76)

    for dataset_name, parameter_path in training_paths.items(): # load and check the dataset

        if not parameter_path.exists():
            raise FileNotFoundError(
                f"{dataset_name.capitalize()} parameter grid "
                f"not found:\n{parameter_path}"
            )

        parameter_grid = pd.read_csv(
            parameter_path
        )

        required_columns = {
            "effective_parameter_mu",
            "brinkman_number",
        }

        missing_columns = required_columns.difference(
            parameter_grid.columns
        )

        if missing_columns:
            raise ValueError(
                f"{dataset_name.capitalize()} parameter grid "
                f"is missing columns: {sorted(missing_columns)}"
            )

        parameter_values = parameter_grid[
            [
                "effective_parameter_mu",
                "brinkman_number",
            ]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if parameter_values.isna().any().any():
            raise ValueError(
                f"{dataset_name.capitalize()} parameter grid "
                "contains missing or non-numeric parameter values."
            )

        for _, row in parameter_values.iterrows():

            key = parameter_key(
                row["effective_parameter_mu"],
                row["brinkman_number"],
            )

            training_parameter_keys.add(key)

        print(
            f"{dataset_name.capitalize()} training combinations: "
            f"{len(parameter_grid)}"
        )

    print(
        f"\nUnique existing training combinations: "
        f"{len(training_parameter_keys)}"
    )

    return training_parameter_keys


# ============================================================
# 6. Generate independent test parameters using LHS
# ============================================================

def generate_test_parameter_grid(
    training_parameter_keys,
    n_test_cases=N_TEST_CASES,
):
    """
    Generate independent parameter combinations using
    Latin hypercube sampling.

    Any point that duplicates an existing training parameter
    combination is rejected.
    """

    if n_test_cases < 1:
        raise ValueError(
            "n_test_cases must be at least 1."
        )

    if CONVECTION_COEFFICIENT <= 0.0:
        raise ValueError(
            "CONVECTION_COEFFICIENT must be positive."
        )

    sampler = qmc.LatinHypercube(
        d=2,
        seed=RANDOM_SEED,
    )

    lower_bounds = np.array(
        [MU_MIN, BR_MIN],
        dtype=float,
    )

    upper_bounds = np.array(
        [MU_MAX, BR_MAX],
        dtype=float,
    )

    accepted_parameters = []
    accepted_keys = set()

    # Generate the initial Latin hypercube sample.
    unit_samples = sampler.random(
        n=n_test_cases,
    )

    physical_samples = qmc.scale(
        unit_samples,
        lower_bounds,
        upper_bounds,
    )

    def try_accept_sample(sample):

        mu = float(sample[0])
        brinkman_number = float(sample[1])

        key = parameter_key(
            mu,
            brinkman_number,
        )

        if key in training_parameter_keys:
            return False

        if key in accepted_keys:
            return False

        accepted_parameters.append({
            "effective_parameter_mu": mu,
            "brinkman_number": brinkman_number,
        })

        accepted_keys.add(key)

        return True

    for sample in physical_samples:
        try_accept_sample(sample) # save unique values

    # In the very unlikely event of duplicate points, generate
    # additional candidates using the same LHS sampler.
    additional_attempts = 0
    maximum_attempts = 1000 * n_test_cases

    while len(accepted_parameters) < n_test_cases:

        if additional_attempts >= maximum_attempts:
            raise RuntimeError(
                "Could not generate the requested number of "
                "unique independent test cases."
            )

        additional_sample = qmc.scale(
            sampler.random(n=1),
            lower_bounds,
            upper_bounds,
        )[0]

        try_accept_sample(
            additional_sample
        )

        additional_attempts += 1

    # Assign case IDs and calculate Pe from mu = c * Pe.
    test_cases = []

    for case_number, parameters in enumerate(
        accepted_parameters,
        start=1,
    ):

        mu = parameters["effective_parameter_mu"]
        brinkman_number = parameters["brinkman_number"]

        test_cases.append({
            "case_id": f"test_{case_number:04d}",
            "effective_parameter_mu": mu,
            "peclet_number": float(
                mu / CONVECTION_COEFFICIENT
            ),
            "brinkman_number": brinkman_number,
            "convection_coefficient": float(
                CONVECTION_COEFFICIENT
            ),
        })

    test_parameter_grid = pd.DataFrame(
        test_cases
    )

    # Final checks.
    test_keys = {
        parameter_key(
            row["effective_parameter_mu"],
            row["brinkman_number"],
        )
        for row in test_cases
    }

    overlap = (
        test_keys.intersection(
            training_parameter_keys
        )
    )

    if overlap:
        raise RuntimeError(
            "Independent test parameters overlap with "
            f"training parameters: {overlap}"
        )

    if len(test_keys) != len(test_cases):
        raise RuntimeError(
            "Duplicate parameter combinations were found "
            "inside the test set."
        )

    return test_parameter_grid


# ============================================================
# 7. Load an existing successful FOM case
# ============================================================

def load_existing_case(
    case,
    case_path,
    reference_temperature,
):
    """
    Load a previously saved converged test case.

    The saved parameters and temperature fields are checked
    before the result is reused.
    """

    with np.load(
        case_path,
        allow_pickle=False,
    ) as saved:

        if not bool(saved["converged"].item()):
            return None

        saved_case_id = str(
            saved["case_id"].item()
        )

        if saved_case_id != case["case_id"]:
            raise ValueError(
                f"Case ID mismatch: {saved_case_id} "
                f"versus {case['case_id']}."
            )

        # Verify the saved parameter values.
        saved_parameters = {
            "effective_parameter_mu": float(
                saved["effective_parameter_mu"]
            ),
            "peclet_number": float(
                saved["peclet_number"]
            ),
            "brinkman_number": float(
                saved["brinkman_number"]
            ),
            "convection_coefficient": float(
                saved["convection_coefficient"]
            ),
        }

        for parameter_name, expected_value in (
            saved_parameters.items()
        ):

            if not np.isclose(
                expected_value,
                case[parameter_name],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Saved {parameter_name} does not match "
                    "the requested test case."
                )

        temperature = np.asarray(
            saved["temperature"],
            dtype=float,
        )

        theta = np.asarray(
            saved["theta"],
            dtype=float,
        )

        if temperature.shape != MESH:
            raise ValueError(
                f"Unexpected temperature shape: "
                f"{temperature.shape}; expected {MESH}."
            )

        if theta.shape != MESH:
            raise ValueError(
                f"Unexpected theta shape: "
                f"{theta.shape}; expected {MESH}."
            )

        if (
            not np.all(np.isfinite(temperature))
            or not np.all(np.isfinite(theta))
        ):
            raise FloatingPointError(
                "Saved FOM fields contain non-finite values."
            )

        expected_theta = (
            temperature - reference_temperature
        )

        if not np.allclose(
            theta,
            expected_theta,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "Saved theta does not match T - T_ref."
            )

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


# ============================================================
# 8. Run the independent FOM test set
# ============================================================

def run_independent_test_set_generation():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CASE_RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx, ny = MESH
    number_of_cells = nx * ny

    # --------------------------------------------------------
    # Load existing training parameters.
    # --------------------------------------------------------

    training_parameter_keys = (
        load_training_parameter_keys()
    )

    # --------------------------------------------------------
    # Generate the independent test parameters.
    # --------------------------------------------------------

    test_parameter_grid = generate_test_parameter_grid(
        training_parameter_keys=training_parameter_keys,
        n_test_cases=N_TEST_CASES,
    )

    test_parameter_grid.to_csv(
        PARAMETER_GRID_PATH,
        index=False,
    )

    print("\n" + "=" * 76)
    print("INDEPENDENT TEST PARAMETER GRID")
    print("=" * 76)

    print(f"Number of test cases: {len(test_parameter_grid)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(
        f"mu range: [{MU_MIN:g}, {MU_MAX:g}]"
    )
    print(
        f"Br range: [{BR_MIN:g}, {BR_MAX:g}]"
    )

    print("\nParameter-grid preview:")
    print(
        test_parameter_grid.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nSaved test parameter grid: "
        f"{PARAMETER_GRID_PATH}"
    )

    # --------------------------------------------------------
    # Initialize the FOM setup.
    # --------------------------------------------------------

    mesh = MESH

    reference_temperature = make_reference_temperature(
        mesh
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    # --------------------------------------------------------
    # Run or load each FOM case.
    # --------------------------------------------------------

    records = []

    print("\n" + "=" * 76)
    print("INDEPENDENT FOM TEST-SET GENERATION")
    print("=" * 76)

    print(f"Mesh: {nx} x {ny}")
    print(f"Number of test cases: {len(test_parameter_grid)}")
    print(f"FOM time step: {config.TIME_STEP}")
    print(
        f"FOM residual tolerance: "
        f"{config.RESIDUAL_TOLERANCE:.3e}"
    )
    print(
        f"Maximum FOM iterations: "
        f"{config.MAX_ITERATIONS}"
    )

    for case_index, (_, row) in enumerate(
        test_parameter_grid.iterrows(),
        start=1,
    ):

        case = row.to_dict()

        case_id = case["case_id"]
        case_path = make_case_path(case)

        print("\n" + "-" * 76)
        print(
            f"Test case {case_index}/"
            f"{len(test_parameter_grid)}: {case_id}"
        )

        print(
            f"mu = {case['effective_parameter_mu']:.8f}, "
            f"Pe = {case['peclet_number']:.8f}, "
            f"Br = {case['brinkman_number']:.8f}, "
            f"c = {case['convection_coefficient']:.6f}"
        )

        record = {
            **case,
            "status": "pending",
            "final_residual": np.nan,
            "iterations": np.nan,
            "computation_time_s": np.nan,
            "result_file": "",
        }

        try:

            case_result = None

            # ------------------------------------------------
            # Reuse an existing successful FOM result.
            # ------------------------------------------------

            if (
                RESUME_FROM_EXISTING_CASES
                and case_path.exists()
            ):

                try:

                    case_result = load_existing_case(
                        case=case,
                        case_path=case_path,
                        reference_temperature=reference_temperature,
                    )

                    if case_result is not None:
                        print(
                            "Loaded existing successful FOM case."
                        )

                except (
                    OSError,
                    ValueError,
                    KeyError,
                    FloatingPointError,
                ) as exc:

                    print(
                        "Could not reuse the existing result. "
                        "The case will be recomputed."
                    )

                    print(
                        f"{type(exc).__name__}: {exc}"
                    )

            # ------------------------------------------------
            # Run the FOM if no valid saved result exists.
            # ------------------------------------------------

            if case_result is None:

                fom_case = {
                    "peclet_number": case["peclet_number"],
                    "brinkman_number": case["brinkman_number"],
                    "convection_coefficient": (
                        case["convection_coefficient"]
                    ),
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

                if temperature.shape != MESH:
                    raise ValueError(
                        f"Unexpected temperature shape: "
                        f"{temperature.shape}; expected {MESH}."
                    )

                if not np.all(np.isfinite(temperature)):
                    raise FloatingPointError(
                        "FOM temperature contains non-finite values."
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

                if fom_result["converged"]:

                    np.savez_compressed(
                        case_path,
                        case_id=case_id,
                        effective_parameter_mu=(
                            case["effective_parameter_mu"]
                        ),
                        peclet_number=case["peclet_number"],
                        brinkman_number=case["brinkman_number"],
                        convection_coefficient=(
                            case["convection_coefficient"]
                        ),
                        temperature=temperature,
                        theta=theta,
                        converged=True,
                        final_residual=(
                            case_result["final_residual"]
                        ),
                        iterations=case_result["iterations"],
                        computation_time_s=(
                            case_result["computation_time_s"]
                        ),
                    )

                else:

                    print(
                        "FOM did not converge. "
                        f"Final residual = "
                        f"{case_result['final_residual']:.6e}"
                    )

            # ------------------------------------------------
            # Update the case record.
            # ------------------------------------------------

            if fom_result_converged(case_result, case_path):

                record.update({
                    "status": "success",
                    "final_residual": (
                        case_result["final_residual"]
                    ),
                    "iterations": (
                        case_result["iterations"]
                    ),
                    "computation_time_s": (
                        case_result["computation_time_s"]
                    ),
                    "result_file": str(case_path),
                })

                print(
                    f"Converged: residual = "
                    f"{case_result['final_residual']:.6e}"
                )

                print(
                    f"Iterations: {case_result['iterations']}"
                )

                print(f"Saved: {case_path}")

            else:

                record["status"] = "FOM_not_converged"

        except Exception as exc:

            record["status"] = (
                f"error: {type(exc).__name__}: {exc}"
            )

            print(
                f"FAILED: {type(exc).__name__}: {exc}"
            )

        records.append(record)

        # Save progress after every case.
        save_summary(records)

    # ========================================================
    # 9. Final summary
    # ========================================================

    summary = pd.DataFrame(records)

    successful_count = int(
        (summary["status"] == "success").sum()
    )

    nonconverged_count = int(
        (summary["status"] == "FOM_not_converged").sum()
    )

    error_count = (
        len(summary)
        - successful_count
        - nonconverged_count
    )

    print("\n" + "=" * 76)
    print("INDEPENDENT TEST-SET SUMMARY")
    print("=" * 76)

    print(f"Total test cases: {len(test_parameter_grid)}")
    print(f"Successful FOM cases: {successful_count}")
    print(f"Nonconverged FOM cases: {nonconverged_count}")
    print(f"Cases with errors: {error_count}")

    print(f"\nParameter grid: {PARAMETER_GRID_PATH}")
    print(f"Simulation summary: {SUMMARY_PATH}")
    print(f"FOM case directory: {CASE_RESULTS_DIRECTORY}")

    return {
        "parameter_grid": test_parameter_grid,
        "summary": summary,
    }


# ============================================================
# 10. Helper: determine whether the FOM case converged
# ============================================================

def fom_result_converged(case_result, case_path):
    """
    Determine whether a case is converged.

    Fresh results have their convergence state in the saved
    file when successful. Reused results are loaded only when
    the saved file has converged=True.
    """

    if case_result is None:
        return False

    if not case_path.exists():
        return False

    try:
        with np.load(
            case_path,
            allow_pickle=False,
        ) as saved:

            return bool(
                saved["converged"].item()
            )

    except (
        OSError,
        ValueError,
        KeyError,
    ):
        return False


# ============================================================
# 11. Run
# ============================================================

if __name__ == "__main__":

    independent_test_data = (
        run_independent_test_set_generation()
    )