from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    CELL_COUNT_X,
    CELL_COUNT_Y,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "image.cmap": "plasma"
})

def visualize_reconstruction(retained_modes=3):

    # -------------------------------------------------
    # 1. Define paths
    # -------------------------------------------------

    project_directory = Path(__file__).resolve().parents[1]

    simulation_directory = (
        project_directory / "results"
    )

    reconstruction_directory = (
        Path(__file__).resolve().parent
        / "results"
        / f"reconstruction_r{retained_modes:02d}"
    )

    summary_file = simulation_directory / "simulation_summary.csv"

    # -------------------------------------------------
    # 2. Read converged cases
    # -------------------------------------------------

    with open(summary_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        summary = list(reader)

    converged_cases = [
        case for case in summary
        if case["converged"].strip().lower() == "true"
    ]

    if not converged_cases:
        raise ValueError("No converged cases found.")

    # Select first, middle and last cases
    indices = sorted(set([
        0,
        len(converged_cases) // 2,
        len(converged_cases) - 1,
    ]))

    selected_cases = [
        converged_cases[i] for i in indices
    ]

    # -------------------------------------------------
    # 3. Load actual and reconstructed fields
    # -------------------------------------------------

    nx = CELL_COUNT_X
    ny = CELL_COUNT_Y

    comparisons = []

    for case in selected_cases:

        case_id = int(case["case_id"])

        actual_file = (
            simulation_directory / f"T_case_{case_id:03d}.npy"
        )

        reconstructed_file = (
            reconstruction_directory
            / f"T_reconstructed_case_{case_id:03d}.npy"
        )

        if not actual_file.exists():
            raise FileNotFoundError(actual_file)

        if not reconstructed_file.exists():
            raise FileNotFoundError(reconstructed_file)

        T_actual = np.load(actual_file, allow_pickle=False)
        T_reconstructed = np.load(
            reconstructed_file,
            allow_pickle=False,
        )

        expected_shape = (nx + 2, ny + 2)

        if T_actual.shape != expected_shape:
            raise ValueError(
                f"Unexpected actual temperature shape for case {case_id}."
            )

        if T_reconstructed.shape != expected_shape:
            raise ValueError(
                f"Unexpected reconstructed temperature shape "
                f"for case {case_id}."
            )

        # Remove ghost cells
        T_actual = T_actual[1:-1, 1:-1]
        T_reconstructed = T_reconstructed[1:-1, 1:-1]

        # Absolute difference
        error = np.abs(T_actual - T_reconstructed)

        comparisons.append({
            "case_id": case_id,
            "mu": float(case["mu"]),
            "Br": float(case["Br"]),
            "actual": T_actual,
            "reconstructed": T_reconstructed,
            "error": error,
        })

    # -------------------------------------------------
    # 4. Define plotting coordinates
    # -------------------------------------------------

    x = (np.arange(nx) + 0.5) * DOMAIN_LENGTH_X / nx
    y = (np.arange(ny) + 0.5) * DOMAIN_LENGTH_Y / ny

    # -------------------------------------------------
    # 5. Define common colour scales
    # -------------------------------------------------

    temperature_min = min(
        min(case["actual"].min(), case["reconstructed"].min())
        for case in comparisons
    )

    temperature_max = max(
        max(case["actual"].max(), case["reconstructed"].max())
        for case in comparisons
    )

    error_max = max(case["error"].max() for case in comparisons)

    if temperature_min == temperature_max:
        temperature_max += 1e-12

    if error_max == 0:
        error_max = 1e-12

    temperature_norm = Normalize(
        vmin=temperature_min,
        vmax=temperature_max,
    )

    error_norm = Normalize(
        vmin=0,
        vmax=error_max,
    )

    # -------------------------------------------------
    # 6. Plot actual, reconstructed and error fields
    # -------------------------------------------------

    number_of_cases = len(comparisons)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(20, 10),
        constrained_layout=True,
    )

    for row, case in enumerate(comparisons):

        case_id = case["case_id"]
        mu = case["mu"]
        br = case["Br"]

        actual = case["actual"]
        reconstructed = case["reconstructed"]
        error = case["error"]

        temperature_levels = np.linspace(
        temperature_norm.vmin,
        temperature_norm.vmax,
        101,
        )

        error_levels = np.linspace(
            error_norm.vmin,
            error_norm.vmax,
            101,
        )

        # Actual temperature
        actual_contour = axes[row, 0].contourf(
            x, y, actual.T,
            levels=temperature_levels,
            cmap="viridis",
            norm=temperature_norm,
        )

        # Reconstructed temperature
        reconstructed_contour = axes[row, 1].contourf(
            x, y, reconstructed.T,
            levels=temperature_levels,
            cmap="viridis",
            norm=temperature_norm,
        )

        # Absolute error
        error_contour = axes[row, 2].contourf(
            x, y, error.T,
            levels=error_levels,
            cmap="magma",
            norm=error_norm,
        )

        axes[row, 0].set_title(
            f"Actual | Case {case_id}\n"
            f"$\\mu$={mu:.3f}, $Br$={br:.3f}"
        )

        axes[row, 1].set_title(
            f"POD reconstruction | Case {case_id}"
        )

        axes[row, 2].set_title(
            f"Absolute error | Case {case_id}"
        )

        for ax in axes[row, :]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("auto")

    # -------------------------------------------------
    # 7. Add colour bars
    # -------------------------------------------------

    temperature_mappable = ScalarMappable(
        norm=temperature_norm,
        cmap="plasma",
    )

    error_mappable = ScalarMappable(
        norm=error_norm,
        cmap="inferno",
    )

    fig.colorbar(
        temperature_mappable,
        location="right",
        fraction=0.025,
        pad=0.08,
        shrink=0.90,
        ax=axes[:, :2].ravel().tolist(),
        label="Temperature",
    )

    fig.colorbar(
        error_mappable,
        ax=axes[:, 2].ravel().tolist(),
        location="right",
        fraction=0.05,
        pad=0.12,
        shrink=0.90,
        label="Absolute temperature error",
    )

    fig.suptitle(
        f"POD Reconstruction Comparison ({retained_modes} modes)",
        fontsize=15,
    )

    

    # -------------------------------------------------
    # 8. Save figure
    # -------------------------------------------------

    figure_file = (
        reconstruction_directory
        / "reconstruction_comparison.png"
    )

    fig.savefig(figure_file, dpi=300, bbox_inches="tight")

    print(f"Figure saved to: {figure_file}")

    plt.show()


if __name__ == "__main__":
    visualize_reconstruction(retained_modes=3)