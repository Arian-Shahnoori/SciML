from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

from src import config
from src.initialization import initialize_velocity_fields

from studies.f04_galerkin_rom.f01_operators import (
    build_operator_components,
)

from studies.f04_galerkin_rom.f02_reduced_operators import (
    build_reduced_operators,
)

from studies.f04_galerkin_rom.f03_solve_reduced import (
    construct_reduced_system,
    solve_reduced_system,
)


# ============================================================
# 1. Configuration
# ============================================================

MESH = (50, 20)

MODE_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10]

BASIS_NAMES = ["original", "augmented"]


# ------------------------------------------------------------
# POD basis paths
# ------------------------------------------------------------

ORIGINAL_POD_MODES_PATH = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "large_training_sweep"
    / "pod"
    / "pod_modes.npy"
)

AUGMENTED_POD_MODES_PATH = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "augmented_pod_basis"
    / "pod_modes.npy"
)


# ------------------------------------------------------------
# Independent test dataset
# ------------------------------------------------------------

TEST_SET_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "independent_test_set"
)

TEST_PARAMETER_GRID_PATH = (
    TEST_SET_DIRECTORY / "test_parameter_grid.csv"
)

TEST_SUMMARY_PATH = (
    TEST_SET_DIRECTORY / "simulation_summary.csv"
)

TEST_CASES_DIRECTORY = (
    TEST_SET_DIRECTORY / "cases"
)


# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "basis_comparison"
)

FIGURES_DIRECTORY = OUTPUT_DIRECTORY / "figures"

CASE_RESULTS_PATH = (
    OUTPUT_DIRECTORY / "comparison_results.csv"
)

COMPARISON_SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "comparison_summary.csv"
)

PAIRED_COMPARISON_PATH = (
    OUTPUT_DIRECTORY / "paired_case_comparison.csv"
)

IMPROVEMENT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "paired_improvement_summary.csv"
)

UNAVAILABLE_CASES_PATH = (
    OUTPUT_DIRECTORY / "unavailable_test_cases.csv"
)

FIGURES_DIRECTORY = OUTPUT_DIRECTORY / "figures"

HEATMAP_DIRECTORY = FIGURES_DIRECTORY / "error_maps"

# ============================================================
# 2. Inlet temperature
# ============================================================

def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


# ============================================================
# 3. Load and validate independent test cases
# ============================================================

def resolve_result_path(result_file):
    """Resolve a result-file path from the test summary."""

    if result_file is None or pd.isna(result_file):
        return None

    if not str(result_file).strip():
        return None

    result_path = Path(str(result_file))

    if result_path.exists():
        return result_path

    fallback_path = (
        TEST_CASES_DIRECTORY / result_path.name
    )

    if fallback_path.exists():
        return fallback_path

    return None


def parameter_key(mu, brinkman_number):
    """Create a rounded key for parameter comparison."""

    return (
        round(float(mu), 10),
        round(float(brinkman_number), 10),
    )


def check_test_set_independence(test_grid):
    """
    Verify that test parameter combinations do not overlap
    with the original or augmented training datasets.
    """

    training_parameter_paths = [
        Path("studies")
        / "f03_pod_analysis"
        / "results"
        / "large_training_sweep"
        / "parameter_grid.csv",

        Path("studies")
        / "f05_refinement"
        / "results"
        / "augmented_training_set"
        / "augmented_parameter_grid.csv",
    ]

    required_columns = {
        "effective_parameter_mu",
        "brinkman_number",
    }

    test_keys = {
        parameter_key(
            row["effective_parameter_mu"],
            row["brinkman_number"],
        )
        for _, row in test_grid.iterrows()
    }

    if len(test_keys) != len(test_grid):
        raise ValueError(
            "The independent test grid contains duplicate "
            "parameter combinations."
        )

    for parameter_path in training_parameter_paths:

        if not parameter_path.exists():
            raise FileNotFoundError(
                f"Training parameter grid not found: "
                f"{parameter_path}"
            )

        training_grid = pd.read_csv(
            parameter_path
        )

        missing_columns = required_columns.difference(
            training_grid.columns
        )

        if missing_columns:
            raise ValueError(
                f"{parameter_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        training_keys = {
            parameter_key(
                row["effective_parameter_mu"],
                row["brinkman_number"],
            )
            for _, row in training_grid.iterrows()
        }

        overlap = test_keys.intersection(
            training_keys
        )

        if overlap:
            raise ValueError(
                "The independent test set overlaps with "
                "a training dataset.\n"
                f"File: {parameter_path}\n"
                f"Overlapping combinations: {sorted(overlap)[:10]}"
            )

    print(
        "Test-set independence check passed."
    )


def load_independent_test_cases():
    """
    Load converged FOM temperature fields from the
    independent test dataset.
    """

    if not TEST_PARAMETER_GRID_PATH.exists():
        raise FileNotFoundError(
            f"Test parameter grid not found: "
            f"{TEST_PARAMETER_GRID_PATH}"
        )

    if not TEST_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Test simulation summary not found: "
            f"{TEST_SUMMARY_PATH}"
        )

    test_grid = pd.read_csv(
        TEST_PARAMETER_GRID_PATH
    )

    test_summary = pd.read_csv(
        TEST_SUMMARY_PATH
    )

    required_grid_columns = {
        "case_id",
        "effective_parameter_mu",
        "peclet_number",
        "brinkman_number",
        "convection_coefficient",
    }

    required_summary_columns = {
        "case_id",
        "status",
        "result_file",
    }

    missing_grid_columns = required_grid_columns.difference(
        test_grid.columns
    )

    missing_summary_columns = required_summary_columns.difference(
        test_summary.columns
    )

    if missing_grid_columns:
        raise ValueError(
            f"Test grid is missing columns: "
            f"{sorted(missing_grid_columns)}"
        )

    if missing_summary_columns:
        raise ValueError(
            f"Test summary is missing columns: "
            f"{sorted(missing_summary_columns)}"
        )

    if test_grid["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs found in test parameter grid."
        )

    if test_summary["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs found in test summary."
        )

    # Check that the test set is independent.
    check_test_set_independence(test_grid)

    summary_by_case = (
        test_summary
        .set_index("case_id")
        .to_dict("index")
    )

    nx, ny = MESH

    loaded_cases = []
    unavailable_cases = []

    for _, row in test_grid.iterrows():

        case = {
            "case_id": str(row["case_id"]),
            "effective_parameter_mu": float(
                row["effective_parameter_mu"]
            ),
            "peclet_number": float(
                row["peclet_number"]
            ),
            "brinkman_number": float(
                row["brinkman_number"]
            ),
            "convection_coefficient": float(
                row["convection_coefficient"]
            ),
        }

        # Check parameter consistency: mu = c * Pe.
        calculated_mu = (
            case["convection_coefficient"]
            * case["peclet_number"]
        )

        if not np.isclose(
            calculated_mu,
            case["effective_parameter_mu"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Inconsistent effective parameter for "
                f"{case['case_id']}: "
                f"mu={case['effective_parameter_mu']}, "
                f"c*Pe={calculated_mu}."
            )

        summary_record = summary_by_case.get(
            case["case_id"]
        )

        if summary_record is None:

            unavailable_cases.append({
                **case,
                "status": "missing_summary_record",
            })

            continue

        if summary_record["status"] != "success":

            unavailable_cases.append({
                **case,
                "status": summary_record["status"],
            })

            continue

        result_path = resolve_result_path(
            summary_record["result_file"]
        )

        if result_path is None:

            unavailable_cases.append({
                **case,
                "status": "FOM_result_file_missing",
            })

            continue

        try:

            with np.load(
                result_path,
                allow_pickle=False,
            ) as saved:

                if "converged" not in saved:
                    raise KeyError(
                        "'converged' field is missing."
                    )

                if not bool(saved["converged"].item()):
                    raise ValueError(
                        "Saved FOM case is not converged."
                    )

                fom_temperature = np.asarray(
                    saved["temperature"],
                    dtype=float,
                )

                if fom_temperature.shape != (nx, ny):
                    raise ValueError(
                        f"Unexpected FOM temperature shape: "
                        f"{fom_temperature.shape}; "
                        f"expected {(nx, ny)}."
                    )

                if not np.all(
                    np.isfinite(fom_temperature)
                ):
                    raise FloatingPointError(
                        "FOM temperature contains non-finite values."
                    )

                # Check saved parameters against the CSV.
                parameter_names = [
                    "effective_parameter_mu",
                    "peclet_number",
                    "brinkman_number",
                    "convection_coefficient",
                ]

                for parameter_name in parameter_names:

                    saved_value = float(
                        saved[parameter_name].item()
                    )

                    expected_value = case[parameter_name]

                    if not np.isclose(
                        saved_value,
                        expected_value,
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"Saved {parameter_name} does not "
                            "match the test parameter grid."
                        )

            loaded_cases.append({
                "case": case,
                "fom_temperature": fom_temperature,
                "fom_result_file": str(result_path),
                "fom_final_residual": float(
                    summary_record.get(
                        "final_residual",
                        np.nan,
                    )
                ),
                "fom_iterations": float(
                    summary_record.get(
                        "iterations",
                        np.nan,
                    )
                ),
            })

        except (
            OSError,
            ValueError,
            KeyError,
            FloatingPointError,
        ) as exc:

            unavailable_cases.append({
                **case,
                "status": (
                    f"FOM_file_error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    print("\n" + "=" * 76)
    print("INDEPENDENT TEST CASES")
    print("=" * 76)

    print(f"Parameter combinations: {len(test_grid)}")
    print(f"Available converged FOM cases: {len(loaded_cases)}")
    print(f"Unavailable cases: {len(unavailable_cases)}")

    if not loaded_cases:
        raise RuntimeError(
            "No valid converged FOM test cases were available."
        )

    return loaded_cases, unavailable_cases


# ============================================================
# 4. Load and validate POD bases
# ============================================================

def load_pod_bases(mode_counts):
    """Load and validate the original and augmented POD bases."""

    basis_paths = {
        "original": ORIGINAL_POD_MODES_PATH,
        "augmented": AUGMENTED_POD_MODES_PATH,
    }

    number_of_cells = MESH[0] * MESH[1]
    maximum_mode_count = max(mode_counts)

    pod_bases = {}

    print("\n" + "=" * 76)
    print("LOADING POD BASES")
    print("=" * 76)

    for basis_name, basis_path in basis_paths.items():

        if not basis_path.exists():
            raise FileNotFoundError(
                f"{basis_name.capitalize()} POD basis not found: "
                f"{basis_path}"
            )

        pod_modes = np.load(
            basis_path,
            allow_pickle=False,
        )

        if pod_modes.ndim != 2:
            raise ValueError(
                f"{basis_name} POD basis must be 2D; "
                f"received shape {pod_modes.shape}."
            )

        if pod_modes.shape[0] != number_of_cells:
            raise ValueError(
                f"{basis_name} POD basis has "
                f"{pod_modes.shape[0]} rows; "
                f"expected {number_of_cells}."
            )

        if pod_modes.shape[1] < maximum_mode_count:
            raise ValueError(
                f"{basis_name} POD basis has only "
                f"{pod_modes.shape[1]} modes, but "
                f"{maximum_mode_count} are requested."
            )

        if not np.all(np.isfinite(pod_modes)):
            raise FloatingPointError(
                f"{basis_name} POD basis contains non-finite values."
            )

        # Check orthonormality of the modes that will be used.
        selected_modes = pod_modes[:, :maximum_mode_count]

        gram_matrix = (
            selected_modes.T @ selected_modes
        )

        orthogonality_error = float(
            np.linalg.norm(
                gram_matrix - np.eye(maximum_mode_count),
                ord="fro",
            )
        )

        pod_bases[basis_name] = pod_modes

        print(
            f"{basis_name.capitalize()} basis shape: "
            f"{pod_modes.shape}"
        )

        print(
            f"Available modes: {pod_modes.shape[1]}"
        )

        print(
            f"Orthogonality error (first "
            f"{maximum_mode_count} modes): "
            f"{orthogonality_error:.6e}"
        )

    return pod_bases


# ============================================================
# 5. Error metrics
# ============================================================

def calculate_error_metrics(
    fom_temperature,
    rom_temperature,
):
    """Calculate relative L2 error, RMSE and maximum absolute error."""

    fom = np.asarray(
        fom_temperature,
        dtype=float,
    )

    rom = np.asarray(
        rom_temperature,
        dtype=float,
    )

    if fom.shape != rom.shape:
        raise ValueError(
            f"FOM and ROM shapes differ: "
            f"{fom.shape} and {rom.shape}."
        )

    if (
        not np.all(np.isfinite(fom))
        or not np.all(np.isfinite(rom))
    ):
        raise FloatingPointError(
            "Non-finite values found in FOM or ROM temperature."
        )

    error = fom - rom

    relative_l2_error = (
        np.linalg.norm(error.ravel())
        / max(np.linalg.norm(fom.ravel()), 1e-14)
    )

    rmse = np.sqrt(
        np.mean(error**2)
    )

    maximum_absolute_error = np.max(
        np.abs(error)
    )

    return {
        "relative_l2_error": float(relative_l2_error),
        "rmse": float(rmse),
        "maximum_absolute_error": float(
            maximum_absolute_error
        ),
    }


# ============================================================
# 6. Build full-order operators
# ============================================================

def build_case_full_operators(
    case,
    mesh,
    u_velocity,
    v_velocity,
):
    """
    Construct full-order operators for one test case.

    They are independent of the POD basis and mode count,
    so they are built only once per test case.
    """

    return build_operator_components(
        mesh=mesh,
        domain_length_x=config.DOMAIN_LENGTH_X,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        peclet_number=case["peclet_number"],
        mean_velocity=config.MEAN_VELOCITY,
        convection_coefficient=case[
            "convection_coefficient"
        ],
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        inlet_temperature=linear_inlet_temperature,
    )


# ============================================================
# 7. Evaluate one ROM
# ============================================================

def evaluate_rom(
    case,
    full_operators,
    pod_modes,
    n_modes,
    fom_temperature,
    mesh,
):
    """Project, solve and reconstruct one ROM."""

    start_time = time.perf_counter()

    reduced_operators = build_reduced_operators(
        operators=full_operators,
        pod_modes=pod_modes,
        n_modes=n_modes,
    )

    reduced_system = construct_reduced_system(
        reduced_operators=reduced_operators,
        peclet_number=case["peclet_number"],
        brinkman_number=case["brinkman_number"],
        convection_coefficient=case[
            "convection_coefficient"
        ],
    )

    solution = solve_reduced_system(
        reduced_system
    )

    coefficients = np.asarray(
        solution["a"],
        dtype=float,
    )

    reduced_basis = reduced_operators["V_r"]

    theta_vector = (
        reduced_basis @ coefficients
    )

    reference_temperature = np.asarray(
        full_operators["T_ref"],
        dtype=float,
    ).ravel(order="C")

    rom_temperature = (
        reference_temperature + theta_vector
    ).reshape(
        mesh,
        order="C",
    )

    rom_computation_time = (
        time.perf_counter() - start_time
    )

    metrics = calculate_error_metrics(
        fom_temperature=fom_temperature,
        rom_temperature=rom_temperature,
    )

    return {
        **metrics,
        "rom_linear_residual": float(
            solution["residual_norm"]
        ),
        "rom_condition_number": float(
            solution["condition_number"]
        ),
        "rom_computation_time_s": float(
            rom_computation_time
        ),
    }


# ============================================================
# 8. Evaluate all bases and mode counts
# ============================================================

def run_rom_evaluations(
    loaded_cases,
    pod_bases,
    mode_counts,
):
    """
    Evaluate both POD bases at every requested mode count
    for exactly the same independent test cases.
    """

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=MESH,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    basis_names = list(pod_bases.keys())

    records = []

    total_evaluations = (
        len(loaded_cases)
        * len(basis_names)
        * len(mode_counts)
    )

    completed_evaluations = 0

    print("\n" + "=" * 76)
    print("RUNNING ROM COMPARISONS")
    print("=" * 76)

    print(f"Test cases: {len(loaded_cases)}")
    print(f"Bases: {basis_names}")
    print(f"Mode counts: {mode_counts}")
    print(f"Total ROM evaluations: {total_evaluations}")

    for case_index, loaded_case in enumerate(
        loaded_cases,
        start=1,
    ):

        case = loaded_case["case"]

        fom_temperature = loaded_case[
            "fom_temperature"
        ]

        print("\n" + "-" * 76)

        print(
            f"Test case {case_index}/{len(loaded_cases)}: "
            f"{case['case_id']}"
        )

        print(
            f"mu = {case['effective_parameter_mu']:.6f}, "
            f"Br = {case['brinkman_number']:.6f}"
        )

        # ----------------------------------------------------
        # Construct full-order operators once per test case.
        # ----------------------------------------------------

        operator_start_time = time.perf_counter()

        try:

            full_operators = build_case_full_operators(
                case=case,
                mesh=MESH,
                u_velocity=u_velocity,
                v_velocity=v_velocity,
            )

            operator_build_time = (
                time.perf_counter() - operator_start_time
            )

        except Exception as exc:

            error_message = (
                f"operator_error: {type(exc).__name__}: {exc}"
            )

            print(
                f"Operator construction failed: {error_message}"
            )

            for basis_name in basis_names:

                for n_modes in mode_counts:

                    records.append({
                        **case,
                        "basis": basis_name,
                        "n_modes": n_modes,
                        "status": error_message,
                        "relative_l2_error": np.nan,
                        "rmse": np.nan,
                        "maximum_absolute_error": np.nan,
                        "rom_linear_residual": np.nan,
                        "rom_condition_number": np.nan,
                        "operator_build_time_s": np.nan,
                        "rom_computation_time_s": np.nan,
                        "fom_result_file": loaded_case[
                            "fom_result_file"
                        ],
                    })

                    completed_evaluations += 1

            # Save progress even when operator construction fails.
            pd.DataFrame(records).to_csv(
                CASE_RESULTS_PATH,
                index=False,
            )

            continue

        # ----------------------------------------------------
        # Evaluate both bases at every requested mode count.
        # ----------------------------------------------------

        for basis_name in basis_names:

            pod_modes = pod_bases[basis_name]

            for n_modes in mode_counts:

                record = {
                    **case,
                    "basis": basis_name,
                    "n_modes": n_modes,
                    "status": "failed",
                    "relative_l2_error": np.nan,
                    "rmse": np.nan,
                    "maximum_absolute_error": np.nan,
                    "rom_linear_residual": np.nan,
                    "rom_condition_number": np.nan,
                    "operator_build_time_s": float(
                        operator_build_time
                    ),
                    "rom_computation_time_s": np.nan,
                    "fom_result_file": loaded_case[
                        "fom_result_file"
                    ],
                }

                try:

                    rom_result = evaluate_rom(
                        case=case,
                        full_operators=full_operators,
                        pod_modes=pod_modes,
                        n_modes=n_modes,
                        fom_temperature=fom_temperature,
                        mesh=MESH,
                    )

                    record.update({
                        "status": "success",
                        **rom_result,
                    })

                    print(
                        f"  {basis_name:10s} | "
                        f"modes = {n_modes:2d} | "
                        f"relative L2 error = "
                        f"{rom_result['relative_l2_error']:.6e}"
                    )

                except Exception as exc:

                    record["status"] = (
                        f"error: {type(exc).__name__}: {exc}"
                    )

                    print(
                        f"  {basis_name:10s} | "
                        f"modes = {n_modes:2d} | "
                        f"FAILED: {type(exc).__name__}: {exc}"
                    )

                records.append(record)
                completed_evaluations += 1

        # Save progress after each complete test case.
        pd.DataFrame(records).to_csv(
            CASE_RESULTS_PATH,
            index=False,
        )

        print(
            f"Progress: {completed_evaluations}/"
            f"{total_evaluations} ROM evaluations"
        )

    return pd.DataFrame(records)


# ============================================================
# 9. Aggregate error statistics
# ============================================================

def build_comparison_summary(
    results,
    mode_counts,
):
    """Calculate aggregate errors for each basis and mode count."""

    successful = results.loc[
        results["status"] == "success"
    ].copy()

    summary_records = []

    for basis_name in BASIS_NAMES:

        for n_modes in mode_counts:

            subset = successful.loc[
                (successful["basis"] == basis_name)
                & (successful["n_modes"] == n_modes)
            ]

            if subset.empty:

                summary_records.append({
                    "basis": basis_name,
                    "n_modes": n_modes,
                    "successful_cases": 0,
                    "mean_relative_l2_error": np.nan,
                    "median_relative_l2_error": np.nan,
                    "p95_relative_l2_error": np.nan,
                    "maximum_relative_l2_error": np.nan,
                    "mean_rmse": np.nan,
                    "median_rmse": np.nan,
                    "mean_maximum_absolute_error": np.nan,
                    "maximum_absolute_error_overall": np.nan,
                    "mean_rom_computation_time_s": np.nan,
                })

                continue

            summary_records.append({
                "basis": basis_name,
                "n_modes": n_modes,
                "successful_cases": len(subset),

                "mean_relative_l2_error": (
                    subset["relative_l2_error"].mean()
                ),

                "median_relative_l2_error": (
                    subset["relative_l2_error"].median()
                ),

                "p95_relative_l2_error": (
                    subset["relative_l2_error"].quantile(0.95)
                ),

                "maximum_relative_l2_error": (
                    subset["relative_l2_error"].max()
                ),

                "mean_rmse": subset["rmse"].mean(),
                "median_rmse": subset["rmse"].median(),

                "mean_maximum_absolute_error": (
                    subset["maximum_absolute_error"].mean()
                ),

                "maximum_absolute_error_overall": (
                    subset["maximum_absolute_error"].max()
                ),

                "mean_rom_computation_time_s": (
                    subset["rom_computation_time_s"].mean()
                ),
            })

    summary = pd.DataFrame(
        summary_records
    )

    summary.to_csv(
        COMPARISON_SUMMARY_PATH,
        index=False,
    )

    return summary


# ============================================================
# 10. Paired case-by-case comparison
# ============================================================

def calculate_percentage_reduction(
    original_errors,
    augmented_errors,
):
    """
    Calculate percentage error reduction.

    Positive values indicate improvement.
    Negative values indicate deterioration.
    """

    original = np.asarray(
        original_errors,
        dtype=float,
    )

    augmented = np.asarray(
        augmented_errors,
        dtype=float,
    )

    reduction = np.full(
        original.shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(original)
        & np.isfinite(augmented)
        & (np.abs(original) > 1e-14)
    )

    reduction[valid] = (
        100.0
        * (original[valid] - augmented[valid])
        / original[valid]
    )

    return reduction


def build_paired_comparison(results):
    """
    Match original and augmented results by case ID and mode count.

    Parameter values are checked after matching.
    """

    parameter_columns = [
        "effective_parameter_mu",
        "peclet_number",
        "brinkman_number",
        "convection_coefficient",
    ]

    metric_columns = [
        "relative_l2_error",
        "rmse",
        "maximum_absolute_error",
    ]

    # The matching keys are case ID and mode count.
    join_columns = [
        "case_id",
        "n_modes",
    ]

    original = results.loc[
        results["basis"] == "original",
        join_columns + parameter_columns + [
            "status",
            *metric_columns,
        ],
    ].copy()

    augmented = results.loc[
        results["basis"] == "augmented",
        join_columns + parameter_columns + [
            "status",
            *metric_columns,
        ],
    ].copy()

    # Rename columns to distinguish the two bases.
    original = original.rename(
        columns={
            "status": "status_original",
            **{
                metric: f"{metric}_original"
                for metric in metric_columns
            },
        }
    )

    augmented = augmented.rename(
        columns={
            "status": "status_augmented",
            **{
                column: f"{column}_augmented"
                for column in parameter_columns
            },
            **{
                metric: f"{metric}_augmented"
                for metric in metric_columns
            },
        }
    )

    paired = original.merge(
        augmented,
        on=join_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    # Verify parameter consistency between the two records.
    for column in parameter_columns:

        original_values = paired[column].to_numpy(
            dtype=float
        )

        augmented_values = paired[
            f"{column}_augmented"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(original_values)
            & np.isfinite(augmented_values)
        )

        if not np.allclose(
            original_values[valid],
            augmented_values[valid],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Parameter mismatch between bases: {column}."
            )

    # Remove the duplicated augmented parameter columns.
    paired = paired.drop(
        columns=[
            f"{column}_augmented"
            for column in parameter_columns
        ]
    )

    paired["paired_success"] = (
        (paired["_merge"] == "both")
        & (paired["status_original"] == "success")
        & (paired["status_augmented"] == "success")
    )

    paired = paired.drop(
        columns="_merge"
    )

    # Calculate paired reductions.
    for metric in metric_columns:

        reduction_column = (
            f"{metric}_reduction_pct"
        )

        paired[reduction_column] = (
            calculate_percentage_reduction(
                paired[f"{metric}_original"],
                paired[f"{metric}_augmented"],
            )
        )

        paired.loc[
            ~paired["paired_success"],
            reduction_column,
        ] = np.nan

    paired.to_csv(
        PAIRED_COMPARISON_PATH,
        index=False,
    )

    return paired


# ============================================================
# 11. Summarize paired improvements
# ============================================================

def build_improvement_summary(
    paired,
    mode_counts,
):
    """
    Summarize the reconstruction error differences between
    the original and augmented POD bases.

    Aggregate percentage reductions are calculated from
    group-level statistics.

    Casewise comparisons use absolute differences in
    relative L2 error to avoid misleading percentages
    when the original error is close to zero.
    """

    summary_records = []

    for n_modes in mode_counts:

        subset = paired.loc[
            (paired["n_modes"] == n_modes)
            & paired["paired_success"]
        ].copy()

        # Keep only cases with valid errors for both bases.
        subset = subset.dropna(
            subset=[
                "relative_l2_error_original",
                "relative_l2_error_augmented",
            ]
        )

        if subset.empty:

            summary_records.append({
                "n_modes": n_modes,
                "paired_test_cases": 0,

                "mean_relative_l2_error_original": np.nan,
                "mean_relative_l2_error_augmented": np.nan,
                "reduction_in_mean_relative_l2_error_pct": np.nan,

                "maximum_relative_l2_error_original": np.nan,
                "maximum_relative_l2_error_augmented": np.nan,
                "reduction_in_maximum_relative_l2_error_pct": np.nan,

                "mean_casewise_error_change": np.nan,
                "median_casewise_error_change": np.nan,
                "p05_casewise_error_change": np.nan,
                "p95_casewise_error_change": np.nan,

                "percentage_of_cases_improved": np.nan,
            })

            continue

        original_errors = subset[
            "relative_l2_error_original"
        ]

        augmented_errors = subset[
            "relative_l2_error_augmented"
        ]

        # --------------------------------------------------
        # Aggregate error statistics
        # --------------------------------------------------

        original_mean = original_errors.mean()
        augmented_mean = augmented_errors.mean()

        original_max = original_errors.max()
        augmented_max = augmented_errors.max()

        mean_reduction = (
            100.0 * (original_mean - augmented_mean)
            / original_mean
            if original_mean > 1e-14
            else np.nan
        )

        maximum_reduction = (
            100.0 * (original_max - augmented_max)
            / original_max
            if original_max > 1e-14
            else np.nan
        )

        # --------------------------------------------------
        # Casewise absolute error differences
        # --------------------------------------------------

        # Negative: augmented basis performs better.
        # Positive: augmented basis performs worse.
        error_changes = (
            augmented_errors - original_errors
        )

        mean_change = error_changes.mean()
        median_change = error_changes.median()

        p05_change = error_changes.quantile(0.05)
        p95_change = error_changes.quantile(0.95)

        # Percentage of cases where augmented error is lower.
        percentage_improved = (
            100.0 * (error_changes < 0.0).mean()
        )

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        summary_records.append({
            "n_modes": n_modes,
            "paired_test_cases": len(subset),

            "mean_relative_l2_error_original": original_mean,
            "mean_relative_l2_error_augmented": augmented_mean,

            "reduction_in_mean_relative_l2_error_pct": (
                mean_reduction
            ),

            "maximum_relative_l2_error_original": original_max,
            "maximum_relative_l2_error_augmented": augmented_max,

            "reduction_in_maximum_relative_l2_error_pct": (
                maximum_reduction
            ),

            "mean_casewise_error_change": mean_change,
            "median_casewise_error_change": median_change,
            "p05_casewise_error_change": p05_change,
            "p95_casewise_error_change": p95_change,

            "percentage_of_cases_improved": percentage_improved,
        })

    improvement_summary = pd.DataFrame(
        summary_records
    )

    improvement_summary.to_csv(
        IMPROVEMENT_SUMMARY_PATH,
        index=False,
    )

    return improvement_summary


# ============================================================
# 12. Plot error statistics
# ============================================================

def plot_error_comparison(
    summary,
    mode_counts,
):
    """
    Compare original and augmented basis errors.

    Four panels show mean, median, 95th percentile and
    maximum relative L2 errors.
    """

    plot_settings = [
        (
            "mean_relative_l2_error",
            "Mean relative L2 error",
        ),
        (
            "median_relative_l2_error",
            "Median relative L2 error",
        ),
        (
            "p95_relative_l2_error",
            "95th-percentile relative L2 error",
        ),
        (
            "maximum_relative_l2_error",
            "Maximum relative L2 error",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    axes = axes.ravel()

    markers = {
        "original": "o",
        "augmented": "s",
    }

    for ax, (metric, title) in zip(
        axes,
        plot_settings,
    ):

        for basis_name in BASIS_NAMES:

            subset = summary.loc[
                summary["basis"] == basis_name
            ].sort_values("n_modes")

            valid = (
                np.isfinite(subset[metric])
                & (subset[metric] > 0.0)
            )

            subset = subset.loc[valid]

            if not subset.empty:

                ax.plot(
                    subset["n_modes"],
                    subset[metric],
                    marker=markers[basis_name],
                    linewidth=1.6,
                    label=basis_name.capitalize(),
                )

        ax.set_yscale("log")

        ax.set_xlabel("Number of POD modes")
        ax.set_ylabel("Relative L2 error")
        ax.set_title(title)

        ax.set_xticks(mode_counts)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Original vs Augmented POD Basis: Test-Set Errors",
        fontsize=14,
    )

    output_path = (
        FIGURES_DIRECTORY
        / "error_comparison_by_mode_count.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# 13. Plot percentage error reduction
# ============================================================

def plot_error_reduction(
    improvement_summary,
    mode_counts,
):
    """
    Plot casewise absolute changes in relative L2 error
    due to basis augmentation.

    Negative values indicate improvement.
    Positive values indicate deterioration.
    Zero indicates equal errors.
    """

    plot_settings = [
        (
            "mean_casewise_error_change",
            "Mean casewise error change",
        ),
        (
            "median_casewise_error_change",
            "Median casewise error change",
        ),
        (
            "p05_casewise_error_change",
            "5th-percentile error change",
        ),
        (
            "p95_casewise_error_change",
            "95th-percentile error change",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (metric, title) in zip(
        axes,
        plot_settings,
    ):

        valid = improvement_summary.loc[
            np.isfinite(
                improvement_summary[metric]
            )
        ]

        if valid.empty:

            ax.set_title(f"{title} (no successful pairs)")
            continue

        ax.plot(
            valid["n_modes"],
            valid[metric],
            marker="o",
            linewidth=1.6,
        )

        # Zero represents equal errors for both bases.
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xlabel("Number of POD modes")
        ax.set_ylabel(
            "Change in relative L2 error\n"
            "(augmented - original)"
        )
        ax.set_title(title)

        ax.set_xticks(mode_counts)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Casewise Error Changes from POD-Basis Augmentation",
        fontsize=14,
    )

    output_path = (
        FIGURES_DIRECTORY
        / "error_change_by_mode_count.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# 14. Triangulated contour error maps
# ============================================================

def get_common_error_scale(results, mode_counts):
    """
    Calculate one logarithmic colour scale for all bases
    and requested mode counts.
    """

    successful = results.loc[
        (results["status"] == "success")
        & results["basis"].isin(BASIS_NAMES)
        & results["n_modes"].isin(mode_counts)
    ]

    errors = successful[
        "relative_l2_error"
    ].to_numpy(dtype=float)

    positive_errors = errors[
        np.isfinite(errors) & (errors > 0.0)
    ]

    if positive_errors.size == 0:
        raise ValueError(
            "No finite positive relative L2 errors available "
            "for the error maps."
        )

    vmin = float(np.min(positive_errors))
    vmax = float(np.max(positive_errors))

    # LogNorm requires strictly increasing limits.
    if vmax <= vmin:
        vmin = vmin / 10.0
        vmax = vmax * 10.0

    norm = LogNorm(
        vmin=vmin,
        vmax=vmax,
    )

    levels = np.geomspace(
        vmin,
        vmax,
        31,
    )

    return norm, levels


def plot_error_maps(results, mode_counts):
    """
    Plot triangulated relative L2 error maps.

    One figure is generated per mode count.

    Each figure contains:
        Left:  Original POD basis.
        Right: Augmented POD basis.

    Both panels use the same colour scale. Actual LHS
    sample locations are marked on the interpolated maps.
    """

    successful = results.loc[
        results["status"] == "success"
    ].copy()

    if successful.empty:
        print("No successful ROM results; error maps skipped.")
        return

    HEATMAP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Common colour scale across all bases and mode counts.
    norm, levels = get_common_error_scale(
        results=results,
        mode_counts=mode_counts,
    )

    print("\nGenerating triangulated error maps...")

    for n_modes in mode_counts:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, 5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        mappable = None

        for ax, basis_name in zip(
            axes,
            BASIS_NAMES,
        ):

            subset = successful.loc[
                (successful["basis"] == basis_name)
                & (successful["n_modes"] == n_modes)
            ].copy()

            if subset.empty:
                ax.set_title(
                    f"{basis_name.capitalize()} basis "
                    "(no successful results)"
                )
                continue

            # Extract parameter coordinates and errors.
            plot_data = subset[
                [
                    "effective_parameter_mu",
                    "brinkman_number",
                    "relative_l2_error",
                ]
            ].copy()

            plot_data = plot_data.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            plot_data = plot_data.loc[
                plot_data["relative_l2_error"] >= 0.0
            ]

            # Each parameter location should have one result
            # per basis and mode count.
            duplicate_count = plot_data.duplicated(
                subset=[
                    "effective_parameter_mu",
                    "brinkman_number",
                ]
            ).sum()

            if duplicate_count:
                print(
                    f"Warning: {duplicate_count} duplicate "
                    f"parameter locations for {basis_name}, "
                    f"{n_modes} modes. Averaging duplicate errors."
                )

                plot_data = (
                    plot_data.groupby(
                        [
                            "effective_parameter_mu",
                            "brinkman_number",
                        ],
                        as_index=False,
                    )
                    ["relative_l2_error"]
                    .mean()
                )

            mu = plot_data[
                "effective_parameter_mu"
            ].to_numpy(dtype=float)

            br = plot_data[
                "brinkman_number"
            ].to_numpy(dtype=float)

            error = plot_data[
                "relative_l2_error"
            ].to_numpy(dtype=float)

            if len(error) < 3:
                ax.set_title(
                    f"{basis_name.capitalize()} basis "
                    "(insufficient samples)"
                )
                continue

            # Require at least three unique, non-collinear
            # parameter locations for triangulation.
            coordinates = np.column_stack((mu, br))

            if np.linalg.matrix_rank(
                coordinates - coordinates.mean(axis=0)
            ) < 2:

                ax.set_title(
                    f"{basis_name.capitalize()} basis "
                    "(degenerate parameter distribution)"
                )
                continue

            # Logarithmic colour scales cannot display zero.
            # Clip zero errors to the lower colour limit.
            plot_error = np.maximum(
                error,
                norm.vmin,
            )

            try:

                triangulation = mtri.Triangulation(
                    mu,
                    br,
                )

                contour = ax.tricontourf(
                    triangulation,
                    plot_error,
                    levels=levels,
                    norm=norm,
                    cmap="viridis",
                    extend="both",
                )

                mappable = contour

                # Mark actual LHS sample locations.
                ax.scatter(
                    mu,
                    br,
                    s=12,
                    color="black",
                    alpha=0.5,
                    linewidths=0,
                    zorder=3,
                )

            except (ValueError, RuntimeError) as exc:

                print(
                    f"Triangulation failed for {basis_name}, "
                    f"{n_modes} modes: {exc}. "
                    "Using sample-point plot instead."
                )

                mappable = ax.scatter(
                    mu,
                    br,
                    c=plot_error,
                    norm=norm,
                    cmap="viridis",
                    s=40,
                    edgecolors="black",
                    linewidths=0.3,
                )

            ax.set_title(
                f"{basis_name.capitalize()} POD basis"
            )

            ax.set_xlabel(
                r"Effective parameter $\mu=cPe$"
            )

            ax.set_ylabel(
                r"Brinkman number $Br$"
            )

            ax.grid(
                True,
                alpha=0.2,
            )

            print(
                f"{basis_name.capitalize()}, "
                f"{n_modes} modes: "
                f"{len(error)} samples, "
                f"mean error = {np.mean(error):.6e}, "
                f"maximum error = {np.max(error):.6e}"
            )

        # One shared colourbar for both basis comparisons.
        if mappable is not None:

            colorbar = fig.colorbar(
                mappable,
                ax=axes.ravel().tolist(),
                fraction=0.035,
                pad=0.04,
            )

            colorbar.set_label(
                "Relative L2 error"
            )

            colorbar.ax.yaxis.set_major_formatter(
                plt.FormatStrFormatter("%.1e")
            )

        fig.suptitle(
            f"Original vs Augmented POD Basis "
            f"({n_modes} Modes)",
            fontsize=14,
        )

        output_path = (
            HEATMAP_DIRECTORY
            / f"error_map_{n_modes}_modes.pdf"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")

    print("\nError-map generation completed.")



# ============================================================
# 14. Main workflow
# ============================================================

def run_basis_comparison(
    mode_counts=MODE_COUNTS,
):
    """
    Compare original and augmented POD bases on the same
    independent test set for multiple mode counts.
    """

    mode_counts = [
        int(value)
        for value in mode_counts
    ]

    if not mode_counts:
        raise ValueError(
            "mode_counts cannot be empty."
        )

    if any(value < 1 for value in mode_counts):
        raise ValueError(
            "Every mode count must be at least 1."
        )

    if len(set(mode_counts)) != len(mode_counts):
        raise ValueError(
            "Duplicate mode counts were provided."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load independent FOM test cases.
    # --------------------------------------------------------

    loaded_cases, unavailable_cases = (
        load_independent_test_cases()
    )

    # --------------------------------------------------------
    # Load both POD bases.
    # --------------------------------------------------------

    pod_bases = load_pod_bases(
        mode_counts=mode_counts
    )

    # --------------------------------------------------------
    # Evaluate both bases at every mode count.
    # --------------------------------------------------------

    results = run_rom_evaluations(
        loaded_cases=loaded_cases,
        pod_bases=pod_bases,
        mode_counts=mode_counts,
    )

    if results.empty:
        raise RuntimeError(
            "No ROM comparison results were generated."
        )

    results.to_csv(
        CASE_RESULTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate error statistics.
    # --------------------------------------------------------

    comparison_summary = build_comparison_summary(
        results=results,
        mode_counts=mode_counts,
    )

    # --------------------------------------------------------
    # Pair original and augmented results.
    # --------------------------------------------------------

    paired_comparison = build_paired_comparison(
        results
    )

    improvement_summary = build_improvement_summary(
        paired=paired_comparison,
        mode_counts=mode_counts,
    )

    # --------------------------------------------------------
    # Save unavailable test cases.
    # --------------------------------------------------------

    pd.DataFrame(
        unavailable_cases
    ).to_csv(
        UNAVAILABLE_CASES_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Generate plots.
    # --------------------------------------------------------

    plot_error_comparison(
        summary=comparison_summary,
        mode_counts=mode_counts,
    )

    plot_error_reduction(
        improvement_summary=improvement_summary,
        mode_counts=mode_counts,
    )

    plot_error_maps(
        results=results,
        mode_counts=mode_counts,
    )

    # --------------------------------------------------------
    # Print final results.
    # --------------------------------------------------------

    print("\n" + "=" * 82)
    print("ORIGINAL VS AUGMENTED POD BASIS: FINAL COMPARISON")
    print("=" * 82)

    print("\nAggregate error statistics:")
    print(
        comparison_summary.to_string(
            index=False
        )
    )

    print("\nPaired improvement statistics:")
    print(
        improvement_summary.to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print(f"Case-level results: {CASE_RESULTS_PATH}")
    print(f"Comparison summary: {COMPARISON_SUMMARY_PATH}")
    print(f"Paired comparison: {PAIRED_COMPARISON_PATH}")
    print(f"Improvement summary: {IMPROVEMENT_SUMMARY_PATH}")
    print(f"Unavailable test cases: {UNAVAILABLE_CASES_PATH}")
    print(f"Figures: {FIGURES_DIRECTORY}")

    return {
        "case_results": results,
        "comparison_summary": comparison_summary,
        "paired_comparison": paired_comparison,
        "improvement_summary": improvement_summary,
        "unavailable_test_cases": pd.DataFrame(
            unavailable_cases
        ),
    }


# ============================================================
# 15. Run
# ============================================================

if __name__ == "__main__":

    comparison_results = run_basis_comparison(
        mode_counts=MODE_COUNTS,
    )