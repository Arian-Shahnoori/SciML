from pathlib import Path
import numpy as np


def perform_pod():

    # -------------------------------------------------
    # 1. Define paths
    # -------------------------------------------------

    output_directory = Path(__file__).resolve().parent / "results"

    snapshot_file = output_directory / "snapshot_matrix.npy"

    if not snapshot_file.exists():
        raise FileNotFoundError(
            f"Snapshot matrix not found: {snapshot_file}"
        )

    # -------------------------------------------------
    # 2. Load snapshot matrix
    # -------------------------------------------------

    X = np.load(snapshot_file, allow_pickle=False)

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Snapshot matrix must be a non-empty 2D array.")

    if not np.all(np.isfinite(X)):
        raise ValueError("Snapshot matrix contains non-finite values.")

    print(f"Snapshot matrix shape: {X.shape}")

    # -------------------------------------------------
    # 3. Perform economy SVD
    # -------------------------------------------------

    U, singular_values, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    # -------------------------------------------------
    # 4. Calculate cumulative energy
    # -------------------------------------------------

    modal_energy = singular_values**2

    total_energy = np.sum(modal_energy)

    if total_energy == 0:
        raise ValueError("Snapshot matrix has zero energy.")

    cumulative_energy = np.cumsum(modal_energy) / total_energy

    # -------------------------------------------------
    # 5. Save POD results
    # -------------------------------------------------

    np.save(output_directory / "pod_modes.npy", U)
    np.save(output_directory / "singular_values.npy", singular_values)
    np.save(output_directory / "Vt.npy", Vt)
    np.save(output_directory / "cumulative_energy.npy", cumulative_energy)

    # -------------------------------------------------
    # 6. Report
    # -------------------------------------------------

    print("\nPOD results")
    print("-----------")
    print(f"Number of modes: {len(singular_values)}")
    print(f"Mode matrix shape: {U.shape}")
    print(f"Vt matrix shape: {Vt.shape}")

    print("\nCumulative energy:")
    for i, energy in enumerate(cumulative_energy, start=1):
        print(f"Mode {i}: {energy:.6%}")

    print(f"\nResults saved in: {output_directory}")

    return U, singular_values, Vt, cumulative_energy


if __name__ == "__main__":
    U, singular_values, Vt, cumulative_energy = perform_pod()