from pathlib import Path
import csv
import numpy as np

from src.config import (
    CELL_COUNT_X,
    CELL_COUNT_Y,
    DOMAIN_LENGTH_Y,
)

from src.initialization import inlet_temperature
from src.boundary_conditions import apply_temperature_boundary_conditions


def reconstruct_snapshots(retained_modes=3):

    # -------------------------------------------------
    # 1. Define paths
    # -------------------------------------------------

    project_directory = Path(__file__).resolve().parents[1]

    results_directory = Path(__file__).resolve().parent / "results"

    simulation_directory = (
        project_directory / "results"
    )

    snapshot_file = results_directory / "snapshot_matrix.npy"
    modes_file = results_directory / "pod_modes.npy"
    singular_values_file = results_directory / "singular_values.npy"
    vt_file = results_directory / "Vt.npy"

    summary_file = simulation_directory / "simulation_summary.csv"

    # -------------------------------------------------
    # 2. Load POD data
    # -------------------------------------------------

    X = np.load(snapshot_file, allow_pickle=False)
    U = np.load(modes_file, allow_pickle=False)
    singular_values = np.load(singular_values_file, allow_pickle=False)
    Vt = np.load(vt_file, allow_pickle=False)

    nx = CELL_COUNT_X
    ny = CELL_COUNT_Y
    number_of_snapshots = X.shape[1]

    # Check dimensions
    if X.shape[0] != nx * ny:
        raise ValueError("Snapshot matrix has an unexpected spatial dimension.")

    if U.shape[0] != X.shape[0]:
        raise ValueError("Incompatible dimensions between X and U.")

    if Vt.shape[1] != number_of_snapshots:
        raise ValueError("Incompatible dimensions between X and Vt.")

    if U.shape[1] != len(singular_values) or Vt.shape[0] != len(singular_values):
        raise ValueError("Incompatible SVD dimensions.")

    if not 1 <= retained_modes <= len(singular_values):
        raise ValueError("Invalid number of retained modes.")

    # -------------------------------------------------
    # 3. Read parameter information
    # -------------------------------------------------

    with open(summary_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        summary = list(reader)

    converged_cases = [
        case for case in summary
        if case["converged"].strip().lower() == "true"
    ]

    if len(converged_cases) != number_of_snapshots:
        raise ValueError(
            "The number of converged cases does not match "
            "the number of snapshot columns."
        )

    # -------------------------------------------------
    # 4. Truncate the SVD
    # -------------------------------------------------

    r = retained_modes

    U_r = U[:, :r]
    sigma_r = singular_values[:r]
    Vt_r = Vt[:r, :]

    # -------------------------------------------------
    # 5. Calculate POD coefficients
    # -------------------------------------------------

    # A_r = Sigma_r @ Vt_r

    A_r = sigma_r[:, None] * Vt_r

    # -------------------------------------------------
    # 6. Reconstruct the snapshot matrix
    # -------------------------------------------------

    # X_r = U_r @ A_r

    X_reconstructed = U_r @ A_r

    # -------------------------------------------------
    # 7. Calculate reconstruction errors
    # -------------------------------------------------

    differences = X - X_reconstructed

    absolute_errors = np.linalg.norm(differences, axis=0)
    original_norms = np.linalg.norm(X, axis=0)

    reconstruction_errors = np.divide(
        absolute_errors,
        original_norms,
        out=np.where(absolute_errors == 0.0, 0.0, np.inf),
        where=original_norms > 0.0,
    )

    cumulative_energy = (
        np.sum(sigma_r**2) / np.sum(singular_values**2)
    )

    # -------------------------------------------------
    # 8. Define reference temperature field
    # -------------------------------------------------

    dy = DOMAIN_LENGTH_Y / ny

    y = (np.arange(ny) + 0.5) * dy

    T_reference = np.tile(y, (nx, 1))

    # -------------------------------------------------
    # 9. Create output directory
    # -------------------------------------------------

    output_directory = (
        results_directory / f"reconstruction_r{r:02d}"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    # Save reconstructed snapshot matrix and coefficients
    np.save(
        output_directory / "reconstructed_snapshot_matrix.npy",
        X_reconstructed,
    )

    np.save(
        output_directory / "pod_coefficients.npy",
        A_r,
    )

    # -------------------------------------------------
    # 10. Reconstruct individual temperature fields
    # -------------------------------------------------

    error_records = []

    for column, case in enumerate(converged_cases):

        case_id = int(case["case_id"])
        mu = float(case["mu"])
        br = float(case["Br"])

        # Extract reconstructed temperature deviation
        theta_reconstructed = X_reconstructed[:, column].reshape(
            (nx, ny),
            order="C",
        )

        # Reverse the lifting transformation
        T_physical = T_reference + theta_reconstructed

        # Reconstruct the full array, including ghost cells
        T_full = np.zeros((nx + 2, ny + 2), dtype=float)

        T_full[1:-1, 1:-1] = T_physical

        T_full = apply_temperature_boundary_conditions(
            temperature=T_full,
            mesh=(nx, ny),
            inlet_temperature=inlet_temperature,
            domain_length_y=DOMAIN_LENGTH_Y,
        )

        # Save reconstructed temperature field
        temperature_file = (
            output_directory / f"T_reconstructed_case_{case_id:03d}.npy"
        )

        np.save(temperature_file, T_full)

        # Record reconstruction error
        error_records.append({
            "snapshot_column": column + 1,
            "case_id": case_id,
            "relative_l2_error": reconstruction_errors[column],
        })

        print(
            f"Case {case_id:03d} | "
            f"mu={mu:.4f} | "
            f"Br={br:.4f} | "
            f"Error={reconstruction_errors[column]:.6e}"
        )

    # -------------------------------------------------
    # 11. Save reconstruction errors
    # -------------------------------------------------

    error_file = output_directory / "reconstruction_errors.csv"

    with open(error_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=error_records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(error_records)

    # -------------------------------------------------
    # 12. Report
    # -------------------------------------------------

    print("\nPOD reconstruction")
    print("------------------")
    print(f"Retained modes:       {r}")
    print(f"Original matrix:      {X.shape}")
    print(f"Reconstructed matrix:{X_reconstructed.shape}")
    print(f"Coefficient matrix:   {A_r.shape}")
    print(f"Cumulative energy:    {cumulative_energy:.8%}")
    print(f"Mean relative error:  {np.mean(reconstruction_errors):.6e}")
    print(f"Maximum relative error: {np.max(reconstruction_errors):.6e}")

    print(f"\nResults saved in: {output_directory}")

    return X_reconstructed, A_r, reconstruction_errors


if __name__ == "__main__":
    X_reconstructed, A_r, errors = reconstruct_snapshots(
        retained_modes=3
    )