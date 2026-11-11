from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

MESH = (50, 20)

# ------------------------------------------------------------
# Original training dataset
# ------------------------------------------------------------

ORIGINAL_TRAINING_DIRECTORY = (
    Path("studies")
    / "f03_pod_analysis"
    / "results"
    / "large_training_sweep"
)

ORIGINAL_SNAPSHOT_MATRIX_PATH = (
    ORIGINAL_TRAINING_DIRECTORY / "snapshot_matrix.npy"
)

ORIGINAL_SNAPSHOT_CASE_IDS_PATH = (
    ORIGINAL_TRAINING_DIRECTORY / "snapshot_case_ids.csv"
)

ORIGINAL_PARAMETER_GRID_PATH = (
    ORIGINAL_TRAINING_DIRECTORY / "parameter_grid.csv"
)

# ------------------------------------------------------------
# Augmented training dataset
# ------------------------------------------------------------

AUGMENTED_TRAINING_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "augmented_training_set"
)

AUGMENTED_SNAPSHOT_MATRIX_PATH = (
    AUGMENTED_TRAINING_DIRECTORY / "snapshot_matrix.npy"
)

AUGMENTED_SNAPSHOT_CASE_IDS_PATH = (
    AUGMENTED_TRAINING_DIRECTORY / "snapshot_case_ids.csv"
)

AUGMENTED_PARAMETER_GRID_PATH = (
    AUGMENTED_TRAINING_DIRECTORY / "augmented_parameter_grid.csv"
)

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "augmented_pod_basis"
)

COMBINED_SNAPSHOT_MATRIX_PATH = (
    OUTPUT_DIRECTORY / "combined_snapshot_matrix.npy"
)

COMBINED_SNAPSHOT_CASE_IDS_PATH = (
    OUTPUT_DIRECTORY / "combined_snapshot_case_ids.csv"
)

POD_MODES_PATH = OUTPUT_DIRECTORY / "pod_modes.npy"

SINGULAR_VALUES_PATH = (
    OUTPUT_DIRECTORY / "singular_values.npy"
)

RIGHT_SINGULAR_VECTORS_PATH = (
    OUTPUT_DIRECTORY / "Vt.npy"
)

CUMULATIVE_ENERGY_PATH = (
    OUTPUT_DIRECTORY / "cumulative_energy.npy"
)

POD_SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "pod_summary.csv"
)


# ============================================================
# 2. Load and validate one snapshot dataset
# ============================================================

def load_snapshot_dataset(
    snapshot_matrix_path,
    snapshot_case_ids_path,
    parameter_grid_path,
    dataset_name,
):
    """
    Load a snapshot matrix and its corresponding case metadata.

    The number of metadata rows must equal the number of
    snapshot columns.
    """

    if not snapshot_matrix_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} snapshot matrix not found:\n"
            f"{snapshot_matrix_path}"
        )

    if not snapshot_case_ids_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} snapshot metadata not found:\n"
            f"{snapshot_case_ids_path}"
        )

    snapshot_matrix = np.load(
        snapshot_matrix_path,
        allow_pickle=False,
    )

    if snapshot_matrix.ndim != 2:
        raise ValueError(
            f"{dataset_name} snapshot matrix must be 2D; "
            f"received shape {snapshot_matrix.shape}."
        )

    expected_rows = MESH[0] * MESH[1]

    if snapshot_matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{dataset_name} snapshot matrix has "
            f"{snapshot_matrix.shape[0]} rows; "
            f"expected {expected_rows} for mesh {MESH}."
        )

    if snapshot_matrix.shape[1] == 0:
        raise ValueError(
            f"{dataset_name} snapshot matrix contains no snapshots."
        )

    if not np.all(np.isfinite(snapshot_matrix)):
        raise FloatingPointError(
            f"{dataset_name} snapshot matrix contains "
            "NaN or infinite values."
        )

    snapshot_case_ids = pd.read_csv(
        snapshot_case_ids_path
    )

    if "case_id" not in snapshot_case_ids.columns:
        raise ValueError(
            f"{dataset_name} metadata must contain a 'case_id' column."
        )

    if len(snapshot_case_ids) != snapshot_matrix.shape[1]:
        raise ValueError(
            f"{dataset_name}: snapshot metadata rows do not match "
            "snapshot matrix columns.\n"
            f"Metadata rows: {len(snapshot_case_ids)}\n"
            f"Snapshot columns: {snapshot_matrix.shape[1]}"
        )

    if snapshot_case_ids["case_id"].isna().any():
        raise ValueError(
            f"{dataset_name} metadata contains missing case IDs."
        )

    if snapshot_case_ids["case_id"].duplicated().any():
        duplicate_ids = snapshot_case_ids.loc[
            snapshot_case_ids["case_id"].duplicated(),
            "case_id",
        ].tolist()

        raise ValueError(
            f"{dataset_name} metadata contains duplicate case IDs: "
            f"{duplicate_ids[:10]}"
        )

    snapshot_case_ids = snapshot_case_ids.copy()

    # Preserve the original column index if it exists.
    if "snapshot_column" in snapshot_case_ids.columns:
        snapshot_case_ids = snapshot_case_ids.rename(
            columns={
                "snapshot_column": "source_snapshot_column"
            }
        )

    # Add parameter information from the parameter grid
    # when it is not already present in the metadata.
    if parameter_grid_path.exists():

        parameter_grid = pd.read_csv(
            parameter_grid_path
        )

        if "case_id" not in parameter_grid.columns:
            raise ValueError(
                f"{dataset_name} parameter grid must contain "
                "a 'case_id' column."
            )

        if parameter_grid["case_id"].duplicated().any():
            raise ValueError(
                f"{dataset_name} parameter grid contains "
                "duplicate case IDs."
            )

        parameter_columns = [
            "effective_parameter_mu",
            "peclet_number",
            "brinkman_number",
            "convection_coefficient",
            "source_hotspot_ids",
        ]

        columns_to_add = [
            column
            for column in parameter_columns
            if column in parameter_grid.columns
            and column not in snapshot_case_ids.columns
        ]

        if columns_to_add:

            snapshot_case_ids = snapshot_case_ids.merge(
                parameter_grid[
                    ["case_id"] + columns_to_add
                ],
                on="case_id",
                how="left",
                validate="one_to_one",
            )

            if snapshot_case_ids[columns_to_add].isna().any().any():
                raise ValueError(
                    f"Some {dataset_name} snapshot cases could not "
                    "be matched to their parameter-grid records."
                )

    # Ensure the metadata order corresponds to the snapshot columns.
    snapshot_case_ids["snapshot_column"] = np.arange(
        snapshot_matrix.shape[1]
    )

    snapshot_case_ids.insert(
        0,
        "source_dataset",
        dataset_name,
    )

    print(f"\n{dataset_name.upper()} DATASET")
    print("-" * 60)
    print(f"Snapshot matrix shape: {snapshot_matrix.shape}")
    print(f"Number of snapshots: {snapshot_matrix.shape[1]}")

    return snapshot_matrix, snapshot_case_ids


# ============================================================
# 3. Check for duplicate parameter combinations
# ============================================================

def check_duplicate_parameters(combined_metadata):
    """
    Check whether original and augmented snapshots contain
    duplicate parameter combinations.

    Parameter values are rounded to 10 decimal places only
    for duplicate detection.
    """

    required_columns = {
        "effective_parameter_mu",
        "brinkman_number",
    }

    if not required_columns.issubset(
        combined_metadata.columns
    ):
        print(
            "\nParameter-duplicate check skipped because "
            "parameter values are unavailable in the metadata."
        )
        return

    metadata = combined_metadata.copy()

    metadata["mu_key"] = metadata[
        "effective_parameter_mu"
    ].round(10)

    metadata["br_key"] = metadata[
        "brinkman_number"
    ].round(10)

    duplicated = metadata.duplicated(
        subset=["mu_key", "br_key"],
        keep=False,
    )

    if duplicated.any():

        duplicate_records = metadata.loc[
            duplicated,
            [
                "source_dataset",
                "case_id",
                "effective_parameter_mu",
                "brinkman_number",
            ],
        ]

        raise ValueError(
            "Duplicate parameter combinations were found "
            "between the original and augmented datasets.\n\n"
            f"{duplicate_records.to_string(index=False)}"
        )

    print(
        "\nParameter uniqueness check passed: "
        "no duplicate parameter combinations."
    )


# ============================================================
# 4. Combine the snapshot matrices
# ============================================================

def combine_snapshot_datasets():

    print("\n" + "=" * 76)
    print("COMBINING ORIGINAL AND AUGMENTED SNAPSHOTS")
    print("=" * 76)

    original_snapshots, original_metadata = (
        load_snapshot_dataset(
            snapshot_matrix_path=ORIGINAL_SNAPSHOT_MATRIX_PATH,
            snapshot_case_ids_path=ORIGINAL_SNAPSHOT_CASE_IDS_PATH,
            parameter_grid_path=ORIGINAL_PARAMETER_GRID_PATH,
            dataset_name="original",
        )
    )

    augmented_snapshots, augmented_metadata = (
        load_snapshot_dataset(
            snapshot_matrix_path=AUGMENTED_SNAPSHOT_MATRIX_PATH,
            snapshot_case_ids_path=AUGMENTED_SNAPSHOT_CASE_IDS_PATH,
            parameter_grid_path=AUGMENTED_PARAMETER_GRID_PATH,
            dataset_name="augmented",
        )
    )

    # Both datasets must use the same physical ordering.
    if original_snapshots.shape[0] != augmented_snapshots.shape[0]:
        raise ValueError(
            "Original and augmented snapshots have different "
            "numbers of spatial degrees of freedom."
        )

    # Check that case IDs are unique across both datasets.
    original_ids = set(
        original_metadata["case_id"].astype(str)
    )

    augmented_ids = set(
        augmented_metadata["case_id"].astype(str)
    )

    overlapping_ids = original_ids.intersection(
        augmented_ids
    )

    if overlapping_ids:
        raise ValueError(
            "Case IDs overlap between the two datasets: "
            f"{sorted(overlapping_ids)[:10]}"
        )

    # Concatenate columns: one column per snapshot.
    combined_snapshots = np.concatenate(
        [
            original_snapshots,
            augmented_snapshots,
        ],
        axis=1,
    )

    # Combine metadata in exactly the same column order.
    combined_metadata = pd.concat(
        [
            original_metadata,
            augmented_metadata,
        ],
        ignore_index=True,
        sort=False,
    )

    combined_metadata["snapshot_column"] = np.arange(
        combined_snapshots.shape[1]
    )

    if len(combined_metadata) != combined_snapshots.shape[1]:
        raise RuntimeError(
            "Combined metadata rows do not match "
            "combined snapshot columns."
        )

    check_duplicate_parameters(
        combined_metadata
    )

    if not np.all(np.isfinite(combined_snapshots)):
        raise FloatingPointError(
            "Combined snapshot matrix contains non-finite values."
        )

    print("\nCOMBINED DATASET")
    print("-" * 60)
    print(f"Original snapshots: {original_snapshots.shape[1]}")
    print(f"Augmented snapshots: {augmented_snapshots.shape[1]}")
    print(f"Total snapshots: {combined_snapshots.shape[1]}")
    print(f"Combined matrix shape: {combined_snapshots.shape}")

    return combined_snapshots, combined_metadata


# ============================================================
# 5. Compute POD basis using economy SVD
# ============================================================

def compute_pod_basis(snapshot_matrix):
    """
    Compute the POD basis from the combined snapshot matrix.

    X = U Sigma Vt

    The columns of U are the spatial POD modes.
    """

    print("\n" + "=" * 76)
    print("COMPUTING AUGMENTED POD BASIS")
    print("=" * 76)

    print(
        f"Snapshot matrix shape: {snapshot_matrix.shape}"
    )

    print("Computing economy SVD...")

    U, singular_values, Vt = np.linalg.svd(
        snapshot_matrix,
        full_matrices=False,
    )

    # POD energy is proportional to the squared singular values.
    singular_value_energy = singular_values**2

    total_energy = float(
        np.sum(singular_value_energy)
    )

    if total_energy > 0.0:

        energy_fraction = (
            singular_value_energy / total_energy
        )

        cumulative_energy = np.cumsum(
            energy_fraction
        )

    else:

        energy_fraction = np.zeros_like(
            singular_values
        )

        cumulative_energy = np.zeros_like(
            singular_values
        )

    pod_summary = pd.DataFrame({
        "mode": np.arange(
            1,
            len(singular_values) + 1,
        ),
        "singular_value": singular_values,
        "energy_fraction": energy_fraction,
        "cumulative_energy": cumulative_energy,
    })

    orthogonality_error = float(
        np.linalg.norm(
            U.T @ U - np.eye(U.shape[1]),
            ord="fro",
        )
    )

    return {
        "pod_modes": U,
        "singular_values": singular_values,
        "Vt": Vt,
        "energy_fraction": energy_fraction,
        "cumulative_energy": cumulative_energy,
        "pod_summary": pod_summary,
        "orthogonality_error": orthogonality_error,
    }


# ============================================================
# 6. Save combined snapshots and POD results
# ============================================================

def save_pod_results(
    snapshot_matrix,
    snapshot_metadata,
    pod_results,
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        COMBINED_SNAPSHOT_MATRIX_PATH,
        snapshot_matrix,
    )

    snapshot_metadata.to_csv(
        COMBINED_SNAPSHOT_CASE_IDS_PATH,
        index=False,
    )

    np.save(
        POD_MODES_PATH,
        pod_results["pod_modes"],
    )

    np.save(
        SINGULAR_VALUES_PATH,
        pod_results["singular_values"],
    )

    np.save(
        RIGHT_SINGULAR_VECTORS_PATH,
        pod_results["Vt"],
    )

    np.save(
        CUMULATIVE_ENERGY_PATH,
        pod_results["cumulative_energy"],
    )

    pod_results["pod_summary"].to_csv(
        POD_SUMMARY_PATH,
        index=False,
    )

    print("\nSaved files:")
    print(f"Combined snapshots: {COMBINED_SNAPSHOT_MATRIX_PATH}")
    print(f"Combined metadata: {COMBINED_SNAPSHOT_CASE_IDS_PATH}")
    print(f"POD modes: {POD_MODES_PATH}")
    print(f"Singular values: {SINGULAR_VALUES_PATH}")
    print(f"Vt: {RIGHT_SINGULAR_VECTORS_PATH}")
    print(f"Cumulative energy: {CUMULATIVE_ENERGY_PATH}")
    print(f"POD summary: {POD_SUMMARY_PATH}")


# ============================================================
# 7. Main workflow
# ============================================================

def build_augmented_pod_basis():

    snapshot_matrix, snapshot_metadata = (
        combine_snapshot_datasets()
    )

    pod_results = compute_pod_basis(
        snapshot_matrix
    )

    save_pod_results(
        snapshot_matrix=snapshot_matrix,
        snapshot_metadata=snapshot_metadata,
        pod_results=pod_results,
    )

    # --------------------------------------------------------
    # Print POD diagnostics
    # --------------------------------------------------------

    pod_modes = pod_results["pod_modes"]
    cumulative_energy = pod_results["cumulative_energy"]

    print("\n" + "=" * 76)
    print("AUGMENTED POD SUMMARY")
    print("=" * 76)

    print(f"Combined snapshot matrix: {snapshot_matrix.shape}")
    print(f"POD modes shape: {pod_modes.shape}")
    print(f"Available modes: {pod_modes.shape[1]}")

    print(
        "Orthogonality error (Frobenius norm): "
        f"{pod_results['orthogonality_error']:.6e}"
    )

    print("\nCumulative POD energy:")

    for mode_number in (1, 2, 3, 4, 5, 6, 8, 10, 20, 50, 100):

        if mode_number <= len(cumulative_energy):

            print(
                f"First {mode_number:3d} modes: "
                f"{100.0 * cumulative_energy[mode_number - 1]:.8f}%"
            )

    print("\nPOD basis generation completed.")

    return {
        "snapshot_matrix": snapshot_matrix,
        "snapshot_metadata": snapshot_metadata,
        **pod_results,
    }


# ============================================================
# 8. Run
# ============================================================

if __name__ == "__main__":

    augmented_pod_results = build_augmented_pod_basis()