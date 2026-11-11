from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src import config
from src.initialization import initialize_velocity_fields

from studies.f04_galerkin_rom.f01_operators import build_operator_components
from studies.f04_galerkin_rom.f02_reduced_operators import build_reduced_operators
from studies.f04_galerkin_rom.f03_solve_reduced import (
    construct_reduced_system,
    solve_reduced_system,
)


# ============================================================
# 1. Study configuration
# ============================================================

MESH = (50, 20)

# Test several truncation levels using the same POD basis.
# Change this list as needed. Every value must be <= pod_modes.shape[1].
MODE_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10]

POD_MODES_PATH = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "large_training_sweep"
    / "pod"
    / "pod_modes.npy"
)

# These files are created by generate_large_training_cases.py.
TEST_SET_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "independent_test_set_lhs"
)

PARAMETER_GRID_PATH = TEST_SET_DIRECTORY / "test_parameter_grid.csv"
TRAINING_SUMMARY_PATH = TEST_SET_DIRECTORY / "simulation_summary.csv"
TRAINING_CASES_DIRECTORY = TEST_SET_DIRECTORY / "cases"

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "mode_count_study"
)

HEATMAP_DIRECTORY = OUTPUT_DIRECTORY / "error_maps"

# Keep the largest-error cases for later adaptive sampling.
NUMBER_OF_WORST_CASES_TO_SAVE = 20


def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


# ============================================================
# 2. Load the existing cases FOM sweep
# ============================================================

def resolve_result_path(result_file):
    """Resolve a result path stored in simulation_summary.csv."""
    if not result_file or pd.isna(result_file):
        return None

    result_path = Path(str(result_file))

    if result_path.exists():
        return result_path

    # Fallback if the summary contains a path from another working directory.
    fallback_path = TRAINING_CASES_DIRECTORY / result_path.name
    if fallback_path.exists():
        return fallback_path

    return None


def load_training_cases():
    """Load converged FOM fields for every point in the parameter grid."""
    if not PARAMETER_GRID_PATH.exists():
        raise FileNotFoundError(
            f"Parameter grid not found: {PARAMETER_GRID_PATH}. "
            "Run generate_large_training_cases.py first."
        )

    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Training summary not found: {TRAINING_SUMMARY_PATH}. "
            "Run generate_large_training_cases.py first."
        )

    parameter_grid = pd.read_csv(PARAMETER_GRID_PATH)
    training_summary = pd.read_csv(TRAINING_SUMMARY_PATH)

    required_grid_columns = {
        "case_id",
        "effective_parameter_mu",
        "peclet_number",
        "brinkman_number",
        "convection_coefficient",
    }
    missing_grid_columns = required_grid_columns.difference(
        parameter_grid.columns
    )
    if missing_grid_columns:
        raise ValueError(
            "The parameter grid is missing columns: "
            f"{sorted(missing_grid_columns)}"
        )

    required_summary_columns = {"case_id", "status", "result_file"}
    missing_summary_columns = required_summary_columns.difference(
        training_summary.columns
    )
    if missing_summary_columns:
        raise ValueError(
            "The training summary is missing columns: "
            f"{sorted(missing_summary_columns)}"
        )

    summary_by_case = training_summary.set_index("case_id").to_dict("index")
    loaded_cases = []
    unavailable_cases = []

    nx, ny = MESH

    for _, grid_row in parameter_grid.iterrows():
        case_id = str(grid_row["case_id"])
        case = {
            "case_id": case_id,
            "effective_parameter_mu": float(
                grid_row["effective_parameter_mu"]
            ),
            "peclet_number": float(grid_row["peclet_number"]),
            "brinkman_number": float(grid_row["brinkman_number"]),
            "convection_coefficient": float(
                grid_row["convection_coefficient"]
            ),
        }

        summary_record = summary_by_case.get(case_id)
        if summary_record is None or summary_record.get("status") != "success":
            unavailable_cases.append({
                **case,
                "status": "FOM_not_available_or_not_converged",
            })
            continue

        result_path = resolve_result_path(
            summary_record.get("result_file", "")
        )
        if result_path is None:
            unavailable_cases.append({
                **case,
                "status": "FOM_result_file_missing",
            })
            continue

        try:
            with np.load(result_path, allow_pickle=False) as saved:
                if "temperature" not in saved:
                    raise KeyError("'temperature' was not found in the case file.")

                fom_temperature = np.asarray(
                    saved["temperature"],
                    dtype=float,
                )

                if fom_temperature.shape != (nx, ny):
                    raise ValueError(
                        f"Unexpected FOM temperature shape "
                        f"{fom_temperature.shape}; expected {(nx, ny)}."
                    )

                if not np.all(np.isfinite(fom_temperature)):
                    raise FloatingPointError(
                        "The FOM temperature contains non-finite values."
                    )

            loaded_cases.append({
                "case": case,
                "fom_temperature": fom_temperature,
                "fom_result_file": str(result_path),
            })

        except (OSError, ValueError, KeyError, FloatingPointError) as exc:
            unavailable_cases.append({
                **case,
                "status": f"FOM_file_error: {type(exc).__name__}: {exc}",
            })

    return parameter_grid, loaded_cases, unavailable_cases


# ============================================================
# 3. Error metrics and ROM evaluation
# ============================================================

def calculate_error_metrics(fom_temperature, rom_temperature):
    """Calculate relative L2 error, RMSE, and maximum absolute error."""
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

    return {
        "relative_l2_error": float(relative_l2_error),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def build_case_full_operators(case, mesh, u_velocity, v_velocity):
    """Build the full-order operators once for a parameter case."""
    return build_operator_components(
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


def run_rom_for_mode_count(
    case,
    full_operators,
    pod_modes,
    n_modes,
    fom_temperature,
    mesh,
):
    """Project and solve the ROM for one case and one mode count."""
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
        convection_coefficient=case["convection_coefficient"],
    )

    solution = solve_reduced_system(reduced_system)
    coefficients = np.asarray(solution["a"], dtype=float)

    reduced_basis = reduced_operators["V_r"]
    theta_vector = reduced_basis @ coefficients

    reference_temperature = np.asarray(
        full_operators["T_ref"],
        dtype=float,
    ).ravel(order="C")

    rom_temperature = (
        reference_temperature + theta_vector
    ).reshape(mesh, order="C")

    computation_time = time.perf_counter() - start_time

    metrics = calculate_error_metrics(
        fom_temperature=fom_temperature,
        rom_temperature=rom_temperature,
    )

    return {
        **metrics,
        "rom_linear_residual": float(solution["residual_norm"]),
        "rom_condition_number": float(solution["condition_number"]),
        "rom_computation_time_s": float(computation_time),
    }


# ============================================================
# 4. ROM error visualisation for LHS test samples
# ============================================================

def get_common_error_scale(results, mode_counts):
    """
    Construct a common logarithmic colour scale across all
    requested mode counts.

    Returns:
        norm   : shared LogNorm instance
        levels : logarithmically spaced contour levels
    """

    successful = results.loc[
        (results["status"] == "success")
        & results["n_modes"].isin(mode_counts)
    ]

    errors = successful["relative_l2_error"].to_numpy(dtype=float)

    positive_errors = errors[
        np.isfinite(errors) & (errors > 0.0)
    ]

    if positive_errors.size == 0:
        raise ValueError(
            "No finite positive relative L2 errors are available."
        )

    vmin = float(np.min(positive_errors))
    vmax = float(np.max(positive_errors))

    # LogNorm requires strictly increasing limits.
    if np.isclose(vmin, vmax):
        vmin = max(vmin * 0.99, np.finfo(float).tiny)
        vmax = vmax * 1.01

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


# ============================================================
# 4.1 Triangulated contour error maps
# ============================================================

def plot_error_maps(results, mode_counts):
    """
    Plot triangulated error maps for the LHS test samples.

    The triangular contours interpolate between the actual
    sample points. The sample locations are also marked.

    All mode counts use the same logarithmic colour scale.
    """

    successful = results.loc[
        results["status"] == "success"
    ].copy()

    if successful.empty:
        print("No successful ROM results; contour plots skipped.")
        return

    HEATMAP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    norm, levels = get_common_error_scale(
        results=results,
        mode_counts=mode_counts,
    )

    number_of_modes = len(mode_counts)

    if number_of_modes == 0:
        return

    ncols = min(2, number_of_modes)
    nrows = int(np.ceil(number_of_modes / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for index, n_modes in enumerate(mode_counts):

        row = index // ncols
        col = index % ncols
        ax = axes[row, col]

        subset = successful.loc[
            successful["n_modes"] == n_modes
        ].copy()

        if subset.empty:
            ax.set_title(f"{n_modes} POD modes (no results)")
            continue

        mu = subset[
            "effective_parameter_mu"
        ].to_numpy(dtype=float)

        br = subset[
            "brinkman_number"
        ].to_numpy(dtype=float)

        error = subset[
            "relative_l2_error"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(mu)
            & np.isfinite(br)
            & np.isfinite(error)
            & (error >= 0.0)
        )

        mu = mu[valid]
        br = br[valid]
        error = error[valid]

        if len(error) < 3:
            ax.set_title(f"{n_modes} POD modes (insufficient samples)")
            continue

        # A logarithmic scale cannot represent zero.
        # Values below the common minimum are displayed at the
        # lower limit of the colour scale.
        plot_error = np.maximum(error, norm.vmin)

        # Check that at least three distinct parameter points exist.
        coordinates = np.column_stack((mu, br))

        if np.unique(coordinates, axis=0).shape[0] < 3:
            ax.set_title(f"{n_modes} POD modes (insufficient unique points)")
            continue

        # Interpolated triangular contour map.
        try:
            contour = ax.tricontourf(
                mu,
                br,
                plot_error,
                levels=levels,
                norm=norm,
                cmap="viridis",
                extend="both",
            )

        except (ValueError, RuntimeError) as exc:
            ax.set_title(f"{n_modes} POD modes (triangulation failed)")
            print(
                f"Triangulation failed for {n_modes} modes: {exc}"
            )
            continue

        # Mark actual LHS sample locations.
        ax.scatter(
            mu,
            br,
            s=12,
            color="black",
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )

        ax.set_title(f"{n_modes} POD modes")
        ax.set_xlabel(r"Effective parameter $\mu=cPe$")
        ax.set_ylabel(r"Brinkman number $Br$")
        ax.grid(True, alpha=0.2)

        colorbar = fig.colorbar(
            contour,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        colorbar.set_label("Relative L2 error")

        colorbar.ax.yaxis.set_major_formatter(
            plt.FormatStrFormatter("%.1e")
        )

        print(
            f"{n_modes} modes: "
            f"samples = {len(error)}, "
            f"mean error = {np.mean(error):.6e}, "
            f"maximum error = {np.max(error):.6e}"
        )

    # Hide unused subplots.
    for index in range(number_of_modes, nrows * ncols):
        row = index // ncols
        col = index % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(
        "ROM Relative L2 Error: Triangulated LHS Test Results",
        fontsize=15,
    )

    output_path = (
        HEATMAP_DIRECTORY
        / "relative_l2_error_triangulated_lhs.pdf"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved triangulated error maps: {output_path}")


# ============================================================
# 4.2 Actual sample-point error plots
# ============================================================

def plot_error_sample_points(results, mode_counts):
    """
    Plot the actual LHS test locations coloured by ROM error.

    No interpolation is performed. All mode counts use the
    same logarithmic colour scale.
    """

    successful = results.loc[
        results["status"] == "success"
    ].copy()

    if successful.empty:
        print("No successful ROM results; sample plots skipped.")
        return

    HEATMAP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    norm, _ = get_common_error_scale(
        results=results,
        mode_counts=mode_counts,
    )

    number_of_modes = len(mode_counts)

    if number_of_modes == 0:
        return

    ncols = min(2, number_of_modes)
    nrows = int(np.ceil(number_of_modes / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for index, n_modes in enumerate(mode_counts):

        row = index // ncols
        col = index % ncols
        ax = axes[row, col]

        subset = successful.loc[
            successful["n_modes"] == n_modes
        ].copy()

        if subset.empty:
            ax.set_title(f"{n_modes} POD modes (no results)")
            continue

        mu = subset[
            "effective_parameter_mu"
        ].to_numpy(dtype=float)

        br = subset[
            "brinkman_number"
        ].to_numpy(dtype=float)

        error = subset[
            "relative_l2_error"
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(mu)
            & np.isfinite(br)
            & np.isfinite(error)
            & (error >= 0.0)
        )

        mu = mu[valid]
        br = br[valid]
        error = error[valid]

        if error.size == 0:
            ax.set_title(f"{n_modes} POD modes (no finite results)")
            continue

        # Display zero errors at the lower colour-scale limit.
        plot_error = np.maximum(error, norm.vmin)

        scatter = ax.scatter(
            mu,
            br,
            c=plot_error,
            cmap="viridis",
            norm=norm,
            s=48,
            edgecolors="black",
            linewidths=0.35,
        )

        ax.set_title(f"{n_modes} POD modes")
        ax.set_xlabel(r"Effective parameter $\mu=cPe$")
        ax.set_ylabel(r"Brinkman number $Br$")
        ax.grid(True, alpha=0.25)

        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        colorbar.set_label("Relative L2 error")

        colorbar.ax.yaxis.set_major_formatter(
            plt.FormatStrFormatter("%.1e")
        )

    # Hide unused subplots.
    for index in range(number_of_modes, nrows * ncols):
        row = index // ncols
        col = index % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(
        "ROM Relative L2 Error: Actual LHS Test Locations",
        fontsize=15,
    )

    output_path = (
        HEATMAP_DIRECTORY
        / "relative_l2_error_actual_lhs_samples.pdf"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved actual sample-point plots: {output_path}")


# ============================================================
# 4.3 Aggregate error statistics
# ============================================================

def plot_error_statistics(mode_summary):
    """
    Plot aggregate ROM error statistics against the POD mode count.
    Uses a logarithmic y-axis to show the range of error magnitudes.
    """

    if mode_summary.empty:
        print("No mode-count statistics available.")
        return

    mode_summary = mode_summary.sort_values(
        "n_modes"
    ).copy()

    fig, ax = plt.subplots(
        figsize=(8, 5),
        constrained_layout=True,
    )

    statistics = [
        ("mean_relative_l2_error", "Mean", "o"),
        ("median_relative_l2_error", "Median", "s"),
        ("p95_relative_l2_error", "95th percentile", "^"),
        ("maximum_relative_l2_error", "Maximum", "d"),
    ]

    for column, label, marker in statistics:

        if column not in mode_summary.columns:
            continue

        values = mode_summary[column].to_numpy(dtype=float, copy=True)

        # Logarithmic axes cannot display zero or negative values.
        values[~np.isfinite(values) | (values <= 0.0)] = np.nan

        ax.plot(
            mode_summary["n_modes"],
            values,
            marker=marker,
            linewidth=1.5,
            label=label,
        )

    ax.set_yscale("log")

    ax.set_xlabel("Number of POD modes")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("ROM Error Statistics on Independent LHS Test Set")

    ax.set_xticks(
        mode_summary["n_modes"].to_numpy(dtype=int)
    )

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    output_path = (
        OUTPUT_DIRECTORY
        / "error_statistics_vs_modes.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved mode-count statistics plot: {output_path}")

# ============================================================
# 5. Main mode-count study
# ============================================================

def run_mode_count_study(mode_counts=MODE_COUNTS):
    """Evaluate multiple ROM dimensions over the complete parameter grid."""
    mode_counts = [int(value) for value in mode_counts]

    if not mode_counts:
        raise ValueError("mode_counts cannot be empty.")
    if any(value < 1 for value in mode_counts):
        raise ValueError("Every mode count must be at least 1.")
    if len(set(mode_counts)) != len(mode_counts):
        raise ValueError("mode_counts contains duplicate values.")

    if not POD_MODES_PATH.exists():
        raise FileNotFoundError(
            f"POD modes not found: {POD_MODES_PATH}. "
            "Generate pod_modes.npy from the large training snapshot matrix first."
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    pod_modes = np.load(POD_MODES_PATH)
    if pod_modes.ndim != 2:
        raise ValueError(
            f"POD basis must be 2D; received shape {pod_modes.shape}."
        )

    maximum_mode_count = max(mode_counts)
    if maximum_mode_count > pod_modes.shape[1]:
        raise ValueError(
            f"Requested {maximum_mode_count} modes, but the POD basis has "
            f"only {pod_modes.shape[1]} columns."
        )

    nx, ny = MESH
    if pod_modes.shape[0] != nx * ny:
        raise ValueError(
            f"POD basis has {pod_modes.shape[0]} rows, but the mesh requires "
            f"{nx * ny} cells. Check MESH and POD_MODES_PATH."
        )

    parameter_grid, loaded_cases, unavailable_cases = load_training_cases()

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=MESH,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    print("\n" + "=" * 78)
    print("POD MODE-COUNT STUDY")
    print("=" * 78)
    print(f"Parameter-grid cases: {len(parameter_grid)}")
    print(f"Cases with available converged FOM fields: {len(loaded_cases)}")
    print(f"Unavailable FOM cases: {len(unavailable_cases)}")
    print(f"Mode counts: {mode_counts}")
    print(f"POD basis: {POD_MODES_PATH}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print(
        "NOTE: If this POD basis was built from these same 400 snapshots, "
        "these errors are in-sample diagnostics, not independent validation."
    )

    result_records = []
    total_case_mode_runs = len(loaded_cases) * len(mode_counts)
    completed_runs = 0

    for case_index, loaded_case in enumerate(loaded_cases, start=1):
        case = loaded_case["case"]
        fom_temperature = loaded_case["fom_temperature"]

        print(
            f"\nCase {case_index}/{len(loaded_cases)}: {case['case_id']} "
            f"(mu={case['effective_parameter_mu']:.6g}, "
            f"Br={case['brinkman_number']:.6g})"
        )

        try:
            operator_start = time.perf_counter()
            full_operators = build_case_full_operators(
                case=case,
                mesh=MESH,
                u_velocity=u_velocity,
                v_velocity=v_velocity,
            )
            operator_build_time = time.perf_counter() - operator_start

        except Exception as exc:
            for n_modes in mode_counts:
                result_records.append({
                    **case,
                    "n_modes": n_modes,
                    "status": (
                        f"operator_error: {type(exc).__name__}: {exc}"
                    ),
                    "relative_l2_error": np.nan,
                    "rmse": np.nan,
                    "maximum_absolute_error": np.nan,
                    "rom_linear_residual": np.nan,
                    "rom_condition_number": np.nan,
                    "operator_build_time_s": np.nan,
                    "rom_computation_time_s": np.nan,
                    "fom_result_file": loaded_case["fom_result_file"],
                })
                completed_runs += 1
            print(f"Operator construction failed: {type(exc).__name__}: {exc}")
            continue

        # The full operators are independent of n_modes, so they are built
        # once per parameter case and reused for every requested ROM size.
        for n_modes in mode_counts:
            try:
                rom_result = run_rom_for_mode_count(
                    case=case,
                    full_operators=full_operators,
                    pod_modes=pod_modes,
                    n_modes=n_modes,
                    fom_temperature=fom_temperature,
                    mesh=MESH,
                )

                result_record = {
                    **case,
                    "n_modes": n_modes,
                    "status": "success",
                    **rom_result,
                    "operator_build_time_s": float(operator_build_time),
                    "fom_result_file": loaded_case["fom_result_file"],
                }

                print(
                    f"  n={n_modes:2d}: relative L2 error = "
                    f"{rom_result['relative_l2_error']:.6e}"
                )

            except Exception as exc:
                result_record = {
                    **case,
                    "n_modes": n_modes,
                    "status": f"error: {type(exc).__name__}: {exc}",
                    "relative_l2_error": np.nan,
                    "rmse": np.nan,
                    "maximum_absolute_error": np.nan,
                    "rom_linear_residual": np.nan,
                    "rom_condition_number": np.nan,
                    "operator_build_time_s": float(operator_build_time),
                    "rom_computation_time_s": np.nan,
                    "fom_result_file": loaded_case["fom_result_file"],
                }
                print(
                    f"  n={n_modes:2d}: FAILED - "
                    f"{type(exc).__name__}: {exc}"
                )

            result_records.append(result_record)
            completed_runs += 1

            if completed_runs % 25 == 0 or completed_runs == total_case_mode_runs:
                print(
                    f"Progress: {completed_runs}/{total_case_mode_runs} "
                    "case-mode evaluations"
                )

    # Add unavailable FOM cases to the result table for traceability.
    for unavailable in unavailable_cases:
        for n_modes in mode_counts:
            result_records.append({
                **unavailable,
                "n_modes": n_modes,
                "relative_l2_error": np.nan,
                "rmse": np.nan,
                "maximum_absolute_error": np.nan,
                "rom_linear_residual": np.nan,
                "rom_condition_number": np.nan,
                "operator_build_time_s": np.nan,
                "rom_computation_time_s": np.nan,
                "fom_result_file": "",
            })

    results = pd.DataFrame(result_records)
    results_path = OUTPUT_DIRECTORY / "case_error_results.csv"
    results.to_csv(results_path, index=False)

    successful = results.loc[results["status"] == "success"].copy()
    summary_records = []
    worst_case_records = []

    for n_modes in mode_counts:
        subset = successful.loc[successful["n_modes"] == n_modes].copy()
        if subset.empty:
            summary_records.append({
                "n_modes": n_modes,
                "successful_cases": 0,
                "mean_relative_l2_error": np.nan,
                "median_relative_l2_error": np.nan,
                "p95_relative_l2_error": np.nan,
                "maximum_relative_l2_error": np.nan,
                "mean_rmse": np.nan,
                "max_error_case_id": "",
                "max_error_mu": np.nan,
                "max_error_br": np.nan,
            })
            continue

        max_index = subset["relative_l2_error"].idxmax()
        max_case = subset.loc[max_index]

        summary_records.append({
            "n_modes": n_modes,
            "successful_cases": len(subset),
            "mean_relative_l2_error": subset[
                "relative_l2_error"
            ].mean(),
            "median_relative_l2_error": subset[
                "relative_l2_error"
            ].median(),
            "p95_relative_l2_error": subset[
                "relative_l2_error"
            ].quantile(0.95),
            "maximum_relative_l2_error": subset[
                "relative_l2_error"
            ].max(),
            "mean_rmse": subset["rmse"].mean(),
            "max_error_case_id": max_case["case_id"],
            "max_error_mu": max_case["effective_parameter_mu"],
            "max_error_br": max_case["brinkman_number"],
        })

        worst = subset.nlargest(
            NUMBER_OF_WORST_CASES_TO_SAVE,
            "relative_l2_error",
        ).copy()
        worst_case_records.append(worst)

    mode_summary = pd.DataFrame(summary_records)
    summary_path = OUTPUT_DIRECTORY / "mode_count_summary.csv"
    mode_summary.to_csv(summary_path, index=False)

    if worst_case_records:
        worst_cases = pd.concat(worst_case_records, ignore_index=True)
    else:
        worst_cases = pd.DataFrame()
    worst_cases_path = OUTPUT_DIRECTORY / "worst_error_cases.csv"
    worst_cases.to_csv(worst_cases_path, index=False)

    unavailable_path = OUTPUT_DIRECTORY / "unavailable_fom_cases.csv"
    pd.DataFrame(unavailable_cases).to_csv(unavailable_path, index=False)

    plot_error_maps(results, mode_counts)
    plot_error_statistics(mode_summary)

    print("\n" + "=" * 78)
    print("MODE-COUNT STUDY SUMMARY")
    print("=" * 78)
    if not mode_summary.empty:
        print(mode_summary.to_string(index=False))

    print(f"\nCase-level results: {results_path}")
    print(f"Mode-count summary: {summary_path}")
    print(f"Worst-error cases: {worst_cases_path}")
    print(f"Unavailable FOM cases: {unavailable_path}")
    print(f"Error maps: {HEATMAP_DIRECTORY}")

    return {
        "case_results": results,
        "mode_count_summary": mode_summary,
        "worst_error_cases": worst_cases,
        "unavailable_fom_cases": pd.DataFrame(unavailable_cases),
    }


# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":
    study_results = run_mode_count_study(
        mode_counts=MODE_COUNTS,
    )
