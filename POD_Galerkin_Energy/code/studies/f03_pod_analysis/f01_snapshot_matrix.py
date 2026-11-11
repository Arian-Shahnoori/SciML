from pathlib import Path
import csv
import numpy as np

from src.config import (
    CELL_COUNT_X,
    CELL_COUNT_Y,
    DOMAIN_LENGTH_Y,
)


def build_snapshot_matrix():

    # -------------------------------------------------
    # 1. Define input and output paths
    # -------------------------------------------------

    project_directory = Path(__file__).resolve().parents[1]

    simulation_directory = (
        project_directory / "results"
    )

    summary_file = simulation_directory / "simulation_summary.csv"

    output_directory = (
        Path(__file__).resolve().parent / "results"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Simulation summary not found: {summary_file}"
        )

    # -------------------------------------------------
    # 2. Read simulation summary
    # -------------------------------------------------

    with open(summary_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        summary = list(reader)

    # Keep only converged simulations
    converged_cases = [
        case for case in summary
        if case["converged"].strip().lower() == "true"
    ]

    if not converged_cases:
        raise ValueError("No converged simulations found.")

    print(f"Total simulations:     {len(summary)}")
    print(f"Converged simulations: {len(converged_cases)}")

    # -------------------------------------------------
    # 3. Define reference temperature field
    # -------------------------------------------------

    nx = CELL_COUNT_X
    ny = CELL_COUNT_Y

    dy = DOMAIN_LENGTH_Y / ny

    # Wall-normal coordinates at physical cell centres
    y = (np.arange(ny) + 0.5) * dy

    # T_ref(x,y) = y
    T_reference = np.tile(y, (nx, 1))

    # -------------------------------------------------
    # 4. Load temperature fields and construct snapshots
    # -------------------------------------------------

    snapshots = []

    for case in converged_cases:

        case_id = int(case["case_id"])
        temperature_file = case["temperature_file"]

        file_path = simulation_directory / temperature_file

        if not file_path.exists():
            raise FileNotFoundError(
                f"Temperature file not found: {file_path}"
            )

        # Load full temperature field, including ghost cells
        T = np.load(file_path, allow_pickle=False)

        expected_shape = (nx + 2, ny + 2)

        if T.shape != expected_shape:
            raise ValueError(
                f"Unexpected temperature shape in case {case_id}. "
                f"Expected {expected_shape}, received {T.shape}."
            )

        if not np.all(np.isfinite(T)):
            raise ValueError(
                f"Non-finite temperature values in case {case_id}."
            )

        # Remove ghost cells
        T_physical = T[1:-1, 1:-1]

        # Apply lifting transformation
        theta = T_physical - T_reference

        # Flatten each snapshot using C ordering
        snapshot = theta.ravel(order="C")

        snapshots.append(snapshot)

        print(f"Loaded case {case_id}")

    # -------------------------------------------------
    # 5. Assemble snapshot matrix
    # -------------------------------------------------

    # Each snapshot becomes one column
    X = np.column_stack(snapshots)

    # -------------------------------------------------
    # 6. Save snapshot matrix and metadata
    # -------------------------------------------------

    np.save(output_directory / "snapshot_matrix.npy", X)

    metadata_file = output_directory / "snapshot_metadata.csv"

    with open(metadata_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=converged_cases[0].keys(),
        )

        writer.writeheader()
        writer.writerows(converged_cases)

    # -------------------------------------------------
    # 7. Report
    # -------------------------------------------------

    print("\nSnapshot matrix")
    print("----------------")
    print(f"Matrix shape: {X.shape}")
    print(f"Spatial degrees of freedom: {nx * ny}")
    print(f"Number of snapshots: {len(snapshots)}")

    print(f"\nSaved matrix: {output_directory / 'snapshot_matrix.npy'}")
    print(f"Saved metadata: {metadata_file}")

    return X


if __name__ == "__main__":
    X = build_snapshot_matrix()