from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Input and output paths
# ============================================================

TRAINING_SWEEP_DIRECTORY = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "large_training_sweep"
)

SNAPSHOT_MATRIX_PATH = TRAINING_SWEEP_DIRECTORY / "snapshot_matrix.npy"
SNAPSHOT_CASE_IDS_PATH = TRAINING_SWEEP_DIRECTORY / "snapshot_case_ids.csv"

# Save the POD basis beside the snapshot matrix. This does not overwrite
# the original studies/f03_pod_analysis/results/pod_modes.npy.
POD_MODES_PATH = TRAINING_SWEEP_DIRECTORY / "pod" / "pod_modes.npy"
SINGULAR_VALUES_PATH = TRAINING_SWEEP_DIRECTORY / "pod" / "singular_values.npy"
RIGHT_SINGULAR_VECTORS_PATH = TRAINING_SWEEP_DIRECTORY / "pod" / "Vt.npy"
CUMULATIVE_ENERGY_PATH = TRAINING_SWEEP_DIRECTORY / "pod" / "cumulative_energy.npy"
POD_SUMMARY_PATH = TRAINING_SWEEP_DIRECTORY / "pod" / "pod_summary.csv"


# ============================================================
# 2. Compute POD using the snapshot-matrix SVD
# ============================================================

def build_pod_basis():
    """Compute and save the economy-SVD POD basis from training snapshots."""

    if not SNAPSHOT_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Snapshot matrix not found: {SNAPSHOT_MATRIX_PATH}. "
            "Run generate_large_training_cases.py first."
        )

    if not SNAPSHOT_CASE_IDS_PATH.exists():
        raise FileNotFoundError(
            f"Snapshot case IDs not found: {SNAPSHOT_CASE_IDS_PATH}."
        )

    snapshot_matrix = np.load(SNAPSHOT_MATRIX_PATH)

    if snapshot_matrix.ndim != 2:
        raise ValueError(
            f"Snapshot matrix must be 2D; got shape {snapshot_matrix.shape}."
        )

    if snapshot_matrix.shape[0] == 0 or snapshot_matrix.shape[1] == 0:
        raise ValueError(
            f"Snapshot matrix is empty; got shape {snapshot_matrix.shape}."
        )

    if not np.all(np.isfinite(snapshot_matrix)):
        raise FloatingPointError(
            "Snapshot matrix contains NaN or infinite values."
        )

    snapshot_case_ids = pd.read_csv(SNAPSHOT_CASE_IDS_PATH)
    if "case_id" not in snapshot_case_ids.columns:
        raise ValueError(
            f"{SNAPSHOT_CASE_IDS_PATH} must contain a 'case_id' column."
        )

    if len(snapshot_case_ids) != snapshot_matrix.shape[1]:
        raise ValueError(
            "The number of snapshot case IDs does not match the number "
            "of snapshot-matrix columns: "
            f"{len(snapshot_case_ids)} IDs versus "
            f"{snapshot_matrix.shape[1]} columns."
        )

    print("\n" + "=" * 72)
    print("POD BASIS GENERATION")
    print("=" * 72)
    print(f"Snapshot matrix: {SNAPSHOT_MATRIX_PATH}")
    print(f"Snapshot matrix shape: {snapshot_matrix.shape}")
    print(f"Number of snapshots: {snapshot_matrix.shape[1]}")
    print("Computing economy SVD...")

    # X = U Sigma V^T. The columns of U are the spatial POD modes.
    U, singular_values, Vt = np.linalg.svd(
        snapshot_matrix,
        full_matrices=False,
    )

    singular_value_energy = singular_values**2
    total_energy = float(np.sum(singular_value_energy))

    if total_energy > 0.0:
        energy_fraction = singular_value_energy / total_energy
        cumulative_energy = np.cumsum(energy_fraction)
    else:
        energy_fraction = np.zeros_like(singular_values)
        cumulative_energy = np.zeros_like(singular_values)

    # Save the spatial modes and SVD diagnostics.
    np.save(POD_MODES_PATH, U)
    np.save(SINGULAR_VALUES_PATH, singular_values)
    np.save(RIGHT_SINGULAR_VECTORS_PATH, Vt)
    np.save(CUMULATIVE_ENERGY_PATH, cumulative_energy)

    pod_summary = pd.DataFrame({
        "mode": np.arange(1, len(singular_values) + 1),
        "singular_value": singular_values,
        "energy_fraction": energy_fraction,
        "cumulative_energy": cumulative_energy,
    })
    pod_summary.to_csv(POD_SUMMARY_PATH, index=False)

    orthogonality_error = float(
        np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord="fro")
    )

    print("\nPOD computation completed.")
    print(f"POD modes shape: {U.shape}")
    print(f"Available modes: {U.shape[1]}")
    print(f"Orthogonality error (Frobenius norm): {orthogonality_error:.6e}")

    for mode_number in (1, 2, 3, 4, 5, 10, 20):
        if mode_number <= len(cumulative_energy):
            print(
                f"Cumulative energy, first {mode_number:2d} modes: "
                f"{100.0 * cumulative_energy[mode_number - 1]:.8f}%"
            )

    print("\nSaved files:")
    print(f"POD modes: {POD_MODES_PATH}")
    print(f"Singular values: {SINGULAR_VALUES_PATH}")
    print(f"Vt: {RIGHT_SINGULAR_VECTORS_PATH}")
    print(f"Cumulative energy: {CUMULATIVE_ENERGY_PATH}")
    print(f"POD summary: {POD_SUMMARY_PATH}")

    return {
        "pod_modes": U,
        "singular_values": singular_values,
        "Vt": Vt,
        "cumulative_energy": cumulative_energy,
        "pod_summary": pod_summary,
        "orthogonality_error": orthogonality_error,
    }


if __name__ == "__main__":
    pod_results = build_pod_basis()
