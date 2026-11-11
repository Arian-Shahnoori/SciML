from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.initialization import initialize_velocity_fields

from studies.f04_galerkin_rom.f04_rom_for_case import run_fom_case


# ============================================================
# 1. Configuration
# ============================================================

MESH = (50, 20)

# Original parameter-space bounds.
MU_MIN = 1.0
MU_MAX = 100.0

BR_MIN = 0.01
BR_MAX = 2.0

# Original training-grid resolution.
ORIGINAL_N_MU = 10
ORIGINAL_N_BR = 10

# Keep mu = c * Pe.
CONVECTION_COEFFICIENT = 1.0

# ------------------------------------------------------------
# Hotspot selection
# ------------------------------------------------------------

# Focus on the higher-mode ROM errors.
TARGET_MODE_COUNTS = [5, 6, 8, 10]

# Number of worst cases considered for each selected mode.
TOP_CASES_PER_MODE = 6

# Maximum number of spatially separated hotspot centres.
MAX_HOTSPOT_CENTERS = 6

# Minimum normalized distance between hotspot centres.
# 0.08 approximately excludes immediately neighbouring
# points on the original parameter grid.
MIN_HOTSPOT_SEPARATION = 0.12

# ------------------------------------------------------------
# Local parameter refinement
# ------------------------------------------------------------

# Half-grid-step offsets in both parameter directions.
# Excluding (0, 0) gives up to 8 new points per hotspot.
AUGMENTATION_OFFSETS = [-0.5, 0.0, 0.5]

# Resume previously completed FOM cases when possible.
RESUME_FROM_EXISTING_CASES = True


# ============================================================
# 2. Input and output paths
# ============================================================

MODE_COUNT_STUDY_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "mode_count_study"
)

WORST_ERROR_CASES_PATH = (
    MODE_COUNT_STUDY_DIRECTORY / "worst_error_cases.csv"
)

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "augmented_training_set"
)

CASE_RESULTS_DIRECTORY = OUTPUT_DIRECTORY / "cases"

HOTSPOT_CANDIDATES_PATH = (
    OUTPUT_DIRECTORY / "hotspot_candidates.csv"
)

SELECTED_HOTSPOTS_PATH = (
    OUTPUT_DIRECTORY / "selected_hotspots.csv"
)

PARAMETER_GRID_PATH = (
    OUTPUT_DIRECTORY / "augmented_parameter_grid.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "simulation_summary.csv"
)

SNAPSHOT_MATRIX_PATH = (
    OUTPUT_DIRECTORY / "snapshot_matrix.npy"
)

SNAPSHOT_CASE_IDS_PATH = (
    OUTPUT_DIRECTORY / "snapshot_case_ids.csv"
)


# ============================================================
# 3. General helper functions
# ============================================================

def format_value(value):
    """Format a number for use in a filename."""
    return (
        f"{value:g}"
        .replace("-", "m")
        .replace(".", "p")
    )


def make_reference_temperature(mesh):
    """
    Construct the reference temperature:

        T_ref = y

    at physical cell centres.
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
    """Construct the output path for one augmented FOM case."""

    filename = (
        f"{case['case_id']}"
        f"_mu{format_value(case['effective_parameter_mu'])}"
        f"_Br{format_value(case['brinkman_number'])}.npz"
    )

    return CASE_RESULTS_DIRECTORY / filename


def save_summary(records):
    """Save the summary after every case to preserve progress."""

    pd.DataFrame(records).to_csv(
        SUMMARY_PATH,
        index=False,
    )


def parameter_key(mu, brinkman_number):
    """Create a rounded key for reliable parameter deduplication."""

    return (
        round(float(mu), 10),
        round(float(brinkman_number), 10),
    )


# ============================================================
# 4. Load and rank the worst-error cases
# ============================================================

def load_worst_error_cases():
    """Read and validate the worst-error cases CSV."""

    if not WORST_ERROR_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Worst-error file not found:\n"
            f"{WORST_ERROR_CASES_PATH}\n\n"
            "Run f01_mode_count_study.py first."
        )

    worst_cases = pd.read_csv(
        WORST_ERROR_CASES_PATH
    )

    required_columns = {
        "effective_parameter_mu",
        "brinkman_number",
        "n_modes",
        "status",
        "relative_l2_error",
    }

    missing_columns = required_columns.difference(
        worst_cases.columns
    )

    if missing_columns:
        raise ValueError(
            "The worst-error file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    worst_cases = worst_cases.copy()

    numeric_columns = [
        "effective_parameter_mu",
        "brinkman_number",
        "n_modes",
        "relative_l2_error",
    ]

    for column in numeric_columns:
        worst_cases[column] = pd.to_numeric(
            worst_cases[column],
            errors="coerce",
        )

    worst_cases = worst_cases.dropna(
        subset=numeric_columns
    )

    if "success" in worst_cases["status"].unique():
        worst_cases = worst_cases.loc[
            worst_cases["status"] == "success"
        ].copy()

    worst_cases["n_modes"] = (
        worst_cases["n_modes"].astype(int)
    )

    if worst_cases.empty:
        raise ValueError(
            "No successful worst-error cases were found."
        )

    available_modes = sorted(
        worst_cases["n_modes"].unique().tolist()
    )

    selected_modes = [
        mode
        for mode in TARGET_MODE_COUNTS
        if mode in available_modes
    ] # target and available modes

    missing_modes = sorted(
        set(TARGET_MODE_COUNTS) - set(selected_modes)
    )

    if missing_modes:
        print(
            "Warning: the following requested mode counts "
            f"are unavailable: {missing_modes}"
        )

    if not selected_modes:
        raise ValueError(
            "None of TARGET_MODE_COUNTS are present in "
            "the worst-error file."
        )

    print("\nAvailable mode counts:", available_modes)
    print("Mode counts used for hotspot selection:", selected_modes)

    return worst_cases, selected_modes


# ============================================================
# 5. Build hotspot candidates
# ============================================================

def build_hotspot_candidates(
    worst_cases, # 20 worst errors for each n_mode
    selected_modes, # target and available modes
):
    """
    Select the top-error cases for each target mode.

    Errors are ranked separately within each mode count.
    Their ranks are used instead of comparing raw errors
    across different mode counts.

    Parameter combinations appearing in several mode counts
    are consolidated into one hotspot candidate.
    """

    candidate_records = []

    for n_modes in selected_modes:

        subset = worst_cases.loc[
            worst_cases["n_modes"] == n_modes
        ].copy()    # Filters worst_cases to keep only rows where n_modes equals the current n_modes value.

        subset = subset.nlargest(
            TOP_CASES_PER_MODE,
            "relative_l2_error",
        ).reset_index(drop=True) # It selects the TOP_CASES_PER_MODE rows with the largest relative_l2_error, in descending order

        if subset.empty:
            continue

        number_of_cases = len(subset)

        for rank_index, row in subset.iterrows():

            # Rank score is 1 for the largest error and
            # decreases towards 0 for the last selected case.
            if number_of_cases > 1:
                rank_score = (
                    1.0
                    - rank_index / (number_of_cases - 1)
                )
            else:
                rank_score = 1.0

            candidate_records.append({
                "effective_parameter_mu": float(
                    row["effective_parameter_mu"]
                ),
                "brinkman_number": float(
                    row["brinkman_number"]
                ),
                "n_modes": int(n_modes),
                "rank_within_mode": int(rank_index + 1),
                "rank_score": float(rank_score),
                "relative_l2_error": float(
                    row["relative_l2_error"]
                ),
                "case_id": str(
                    row["case_id"]
                ) if "case_id" in row else "",
            })

    candidates = pd.DataFrame(candidate_records) # contains the top-error cases (12 cases each n_mode)
                                                 # across all selected POD mode counts with their rank within mode and other info.

    if candidates.empty:
        raise ValueError(
            "No hotspot candidates could be constructed."
        )

    # Rounded coordinates allow identical parameter locations
    # from different mode counts to be combined.
    candidates["mu_key"] = candidates[
        "effective_parameter_mu"
    ].round(10)

    candidates["br_key"] = candidates[
        "brinkman_number"
    ].round(10)

    grouped_records = []

    for (mu_key, br_key), group in candidates.groupby(
        ["mu_key", "br_key"],
        sort=False,
    ): # group is a smaller DataFrame containing all rows from candidates that share the same mu_key and br_key.

        modes_at_location = sorted(
            group["n_modes"].unique().tolist()
        ) # modes_at_location means the POD mode counts for which 
          # this particular parameter combination appeared among the selected high-error cases.

        mode_ranks = [
            f"{int(row.n_modes)}:{int(row.rank_within_mode)}"
            for row in group.itertuples()
        ] # mode_ranks records the error rank of this specific parameter combination within each POD mode count.

        mode_support = len(modes_at_location)

        mean_rank_score = float(
            group["rank_score"].mean()
        )

        support_fraction = (
            mode_support / len(selected_modes)
        )

        # Reward locations that rank highly across several
        # mode counts, while retaining strong individual cases.
        base_score = (
            0.70 * mean_rank_score
            + 0.30 * support_fraction
        )

        grouped_records.append({
            "effective_parameter_mu": float(mu_key),
            "brinkman_number": float(br_key),
            "mode_support": mode_support,
            "source_modes": ",".join(
                map(str, modes_at_location)
            ),
            "source_case_ids": ",".join(
                sorted(set(group["case_id"].tolist()))
            ),
            "mode_ranks": ";".join(mode_ranks),
            "mean_rank_score": mean_rank_score,
            "base_score": float(base_score),
            "maximum_relative_l2_error": float(
                group["relative_l2_error"].max()
            ),
        })

    hotspot_candidates = pd.DataFrame(
        grouped_records # candidates with rank scores (each set of parameters with their score)
    )

    # The highest base score identifies the strongest
    # candidate before considering spatial diversity.
    hotspot_candidates = hotspot_candidates.sort_values(
        by=[
            "base_score",
            "mode_support",
            "mean_rank_score",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return candidates, hotspot_candidates # TOP_CASES_PER_MODE * mode_count cases with each having base score


# ============================================================
# 6. Select spatially separated hotspots
# ============================================================

def normalized_parameter_distance(
    mu_1,
    br_1,
    mu_2,
    br_2,
):
    """
    Euclidean distance in normalized parameter space.

    Dividing by sqrt(2) gives a distance between 0 and 1.
    """

    delta_mu = (
        (mu_1 - mu_2)
        / (MU_MAX - MU_MIN)
    )

    delta_br = (
        (br_1 - br_2)
        / (BR_MAX - BR_MIN)
    )

    return float(
        np.sqrt(delta_mu**2 + delta_br**2)
        / np.sqrt(2.0)
    )


def select_hotspots(hotspot_candidates):
    """
    Greedily select high-scoring, spatially separated locations.

    The selection balances:
        1. Error ranking across mode counts.
        2. Spatial distance from already selected hotspots.
    """

    candidates = hotspot_candidates.copy()

    if candidates.empty:
        raise ValueError(
            "No hotspot candidates are available."
        )

    selected_records = [] # This code selects hotspots one at a time, based on error severity and distance from previously selected hotspots.

    print("\n" + "=" * 76)
    print("HOTSPOT SELECTION")
    print("=" * 76)

    for selection_index in range(MAX_HOTSPOT_CENTERS):

        if selection_index == 0:

            candidates["minimum_distance"] = np.inf

            candidates["selection_score"] = (
                candidates["base_score"]
            )

        else:

            distances_to_selected = []

            for _, candidate in candidates.iterrows():

                distances = [
                    normalized_parameter_distance(
                        candidate["effective_parameter_mu"],
                        candidate["brinkman_number"],
                        selected["effective_parameter_mu"],
                        selected["brinkman_number"],
                    )
                    for selected in selected_records
                ]

                distances_to_selected.append(
                    min(distances)
                )

            candidates["minimum_distance"] = (
                distances_to_selected # distance to selected is min distance of each candidate to the already selected candidates
            )

            # Exclude locations too close to a selected hotspot.
            candidates = candidates.loc[
                candidates["minimum_distance"]
                >= MIN_HOTSPOT_SEPARATION
            ].copy() # Keeps only the candidate hotspots whose minimum_distance is greater than or equal to MIN_HOTSPOT_SEPARATION.

            if candidates.empty:
                print(
                    "No additional candidates satisfy the "
                    "minimum hotspot separation."
                )
                break

            # Combine error ranking with spatial diversity.
            candidates["selection_score"] = (
                0.55 * candidates["base_score"]
                + 0.45 * candidates["minimum_distance"]
            )

        best_index = candidates[
            "selection_score"
        ].idxmax()

        best_candidate = candidates.loc[
            best_index
        ].to_dict()

        best_candidate["hotspot_id"] = (
            f"hotspot_{len(selected_records) + 1:02d}"
        )

        selected_records.append(best_candidate)

        print(
            f"{best_candidate['hotspot_id']}: "
            f"mu = {best_candidate['effective_parameter_mu']:.6f}, "
            f"Br = {best_candidate['brinkman_number']:.6f}, "
            f"modes = {best_candidate['source_modes']}, "
            f"max listed error = "
            f"{best_candidate['maximum_relative_l2_error']:.6e}"
        )

        # Remove the selected candidate.
        candidates = candidates.drop(
            index=best_index
        ).reset_index(drop=True)

    selected_hotspots = pd.DataFrame(
        selected_records
    )

    if selected_hotspots.empty:
        raise RuntimeError(
            "Hotspot selection produced no selected locations."
        )

    print(
        f"\nSelected hotspot centres: "
        f"{len(selected_hotspots)}"
    )

    return selected_hotspots


# ============================================================
# 7. Generate augmented parameter combinations
# ============================================================

def build_augmented_parameter_grid(
    selected_hotspots,
):
    """
    Generate new parameter combinations around every hotspot.

    Each hotspot generates up to eight points using half-grid
    increments in mu and Brinkman number.

    The original hotspot itself is excluded because its FOM
    solution is already in the original training set.
    """

    delta_mu = (
        (MU_MAX - MU_MIN)
        / (ORIGINAL_N_MU - 1)
    )

    delta_br = (
        (BR_MAX - BR_MIN)
        / (ORIGINAL_N_BR - 1)
    )

    print("\nOriginal grid spacing:")
    print(f"Delta mu = {delta_mu:.8f}")
    print(f"Delta Br = {delta_br:.8f}")

    # Build a set of all original parameter combinations.
    original_mu_values = np.linspace(
        MU_MIN,
        MU_MAX,
        ORIGINAL_N_MU,
    )

    original_br_values = np.linspace(
        BR_MIN,
        BR_MAX,
        ORIGINAL_N_BR,
    )

    original_parameter_keys = {
        parameter_key(mu, br)
        for mu in original_mu_values
        for br in original_br_values
    }

    # A dictionary prevents duplicate augmented points.
    generated_parameters = {}

    for _, hotspot in selected_hotspots.iterrows():

        hotspot_id = hotspot["hotspot_id"]

        center_mu = float(
            hotspot["effective_parameter_mu"]
        )

        center_br = float(
            hotspot["brinkman_number"]
        )

        for mu_offset in AUGMENTATION_OFFSETS:
            for br_offset in AUGMENTATION_OFFSETS:

                # Do not include the hotspot centre itself.
                if (
                    mu_offset == 0.0
                    and br_offset == 0.0
                ):
                    continue

                augmented_mu = (
                    center_mu + mu_offset * delta_mu
                )

                augmented_br = (
                    center_br + br_offset * delta_br
                )

                # Discard parameters outside the original bounds.
                if not (
                    MU_MIN <= augmented_mu <= MU_MAX
                ):
                    continue

                if not (
                    BR_MIN <= augmented_br <= BR_MAX
                ):
                    continue

                key = parameter_key(
                    augmented_mu,
                    augmented_br,
                )

                # Do not regenerate points already in the
                # original 400-case training grid.
                if key in original_parameter_keys:
                    continue

                if key not in generated_parameters:

                    generated_parameters[key] = {
                        "effective_parameter_mu": float(
                            augmented_mu
                        ),
                        "brinkman_number": float(
                            augmented_br
                        ),
                        "source_hotspot_ids": set(),
                    }

                generated_parameters[key][
                    "source_hotspot_ids"
                ].add(hotspot_id)

    if not generated_parameters:
        raise RuntimeError(
            "No new augmented parameter combinations were generated."
        )

    # Assign stable case IDs.
    augmented_cases = []

    sorted_parameters = sorted(
        generated_parameters.values(),
        key=lambda item: (
            item["effective_parameter_mu"],
            item["brinkman_number"],
        ),
    )

    for case_number, parameter in enumerate(
        sorted_parameters,
        start=1,
    ):

        mu = parameter["effective_parameter_mu"]
        brinkman_number = parameter["brinkman_number"]

        augmented_cases.append({
            "case_id": f"aug_{case_number:04d}",
            "effective_parameter_mu": float(mu),
            "peclet_number": float(
                mu / CONVECTION_COEFFICIENT
            ),
            "brinkman_number": float(
                brinkman_number
            ),
            "convection_coefficient": float(
                CONVECTION_COEFFICIENT
            ),
            "source_hotspot_ids": ",".join(
                sorted(parameter["source_hotspot_ids"])
            ),
        })

    return augmented_cases


# ============================================================
# 8. Load an existing augmented FOM case
# ============================================================

def load_existing_case(
    case,
    case_path,
    reference_temperature,
):
    """Load a previously completed augmented FOM case."""

    with np.load(
        case_path,
        allow_pickle=False,
    ) as saved:

        if not bool(saved["converged"]):
            return None

        saved_case_id = str(
            saved["case_id"].item()
        )

        if saved_case_id != case["case_id"]:
            raise ValueError(
                f"Case ID mismatch in {case_path}: "
                f"{saved_case_id} != {case['case_id']}"
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
                "The saved FOM fields contain non-finite values."
            )

        # Check that the saved perturbation corresponds to
        # the current reference-temperature definition.
        expected_theta = temperature - reference_temperature

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
            "converged": True,
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
# 9. Run the augmented FOM training cases
# ============================================================

def run_augmented_training_set_generation():

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
    # Load worst-error cases and select hotspots.
    # --------------------------------------------------------

    worst_cases, selected_modes = load_worst_error_cases()

    candidates, hotspot_candidates = (
        build_hotspot_candidates(
            worst_cases=worst_cases,
            selected_modes=selected_modes,
        )
    )

    candidates.to_csv(
        HOTSPOT_CANDIDATES_PATH,
        index=False,
    )

    selected_hotspots = select_hotspots(
        hotspot_candidates
    )

    selected_hotspots.to_csv(
        SELECTED_HOTSPOTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Generate new parameter combinations.
    # --------------------------------------------------------

    augmented_cases = build_augmented_parameter_grid(
        selected_hotspots
    )

    parameter_grid = pd.DataFrame(
        augmented_cases
    )

    parameter_grid.to_csv(
        PARAMETER_GRID_PATH,
        index=False,
    )

    print("\n" + "=" * 76)
    print("AUGMENTED PARAMETER GRID")
    print("=" * 76)

    print(f"Selected hotspots: {len(selected_hotspots)}")
    print(f"Generated FOM cases: {len(augmented_cases)}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")

    print("\nParameter-grid preview:")
    print(parameter_grid.head(10).to_string(index=False))

    # --------------------------------------------------------
    # Initialize FOM solver fields.
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
    # Run all augmented FOM cases.
    # --------------------------------------------------------

    records = []

    snapshot_columns = []
    snapshot_case_records = []

    print("\n" + "=" * 76)
    print("AUGMENTED FOM TRAINING-CASE GENERATION")
    print("=" * 76)

    print(f"Mesh: {nx} x {ny}")
    print(f"Number of augmented cases: {len(augmented_cases)}")
    print(f"FOM time step: {config.TIME_STEP}")
    print(
        f"FOM residual tolerance: "
        f"{config.RESIDUAL_TOLERANCE:.3e}"
    )
    print(
        f"Maximum FOM iterations: "
        f"{config.MAX_ITERATIONS}"
    )

    for case_index, case in enumerate(
        augmented_cases,
        start=1,
    ):

        case_id = case["case_id"]
        case_path = make_case_path(case)

        print("\n" + "-" * 76)
        print(
            f"Case {case_index}/{len(augmented_cases)}: "
            f"{case_id}"
        )

        print(
            f"mu = {case['effective_parameter_mu']:.8f}, "
            f"Br = {case['brinkman_number']:.8f}, "
            f"Pe = {case['peclet_number']:.8f}, "
            f"c = {case['convection_coefficient']:.6f}"
        )

        print(
            f"Source hotspots: "
            f"{case['source_hotspot_ids']}"
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
            # Try to reuse a previously converged FOM case.
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
            # Run the FOM if no reusable result exists.
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
                    "converged": bool(
                        fom_result["converged"]
                    ),
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

                # Save only converged FOM cases.
                if case_result["converged"]:

                    np.savez_compressed(
                        case_path,
                        case_id=case_id,
                        effective_parameter_mu=case[
                            "effective_parameter_mu"
                        ],
                        peclet_number=case[
                            "peclet_number"
                        ],
                        brinkman_number=case[
                            "brinkman_number"
                        ],
                        convection_coefficient=case[
                            "convection_coefficient"
                        ],
                        source_hotspot_ids=case[
                            "source_hotspot_ids"
                        ],
                        temperature=case_result[
                            "temperature"
                        ],
                        theta=case_result["theta"],
                        converged=True,
                        final_residual=case_result[
                            "final_residual"
                        ],
                        iterations=case_result[
                            "iterations"
                        ],
                        computation_time_s=case_result[
                            "computation_time_s"
                        ],
                    )

                else:
                    print(
                        "FOM did not converge. "
                        f"Final residual = "
                        f"{case_result['final_residual']:.6e}"
                    )

            # ------------------------------------------------
            # Record successful and failed cases.
            # ------------------------------------------------

            if case_result["converged"]:

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

                # Append theta, not the full temperature field,
                # to preserve the original POD snapshot convention.
                snapshot_columns.append(
                    case_result["theta"].ravel(
                        order="C"
                    )
                )

                snapshot_column = (
                    len(snapshot_columns) - 1
                )

                snapshot_case_records.append({
                    "snapshot_column": snapshot_column,
                    "case_id": case_id,
                    "effective_parameter_mu": (
                        case["effective_parameter_mu"]
                    ),
                    "peclet_number": case["peclet_number"],
                    "brinkman_number": case["brinkman_number"],
                    "convection_coefficient": (
                        case["convection_coefficient"]
                    ),
                    "source_hotspot_ids": (
                        case["source_hotspot_ids"]
                    ),
                    "result_file": str(case_path),
                })

                print(
                    f"Converged: residual = "
                    f"{case_result['final_residual']:.6e}"
                )

                print(
                    f"Iterations: "
                    f"{case_result['iterations']}"
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

        # Update summary after every case.
        save_summary(records)

    # ========================================================
    # 10. Assemble augmented snapshot matrix
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

    np.save(
        SNAPSHOT_MATRIX_PATH,
        snapshot_matrix,
    )

    snapshot_case_ids = pd.DataFrame(
        snapshot_case_records
    )

    snapshot_case_ids.to_csv(
        SNAPSHOT_CASE_IDS_PATH,
        index=False,
    )

    # ========================================================
    # 11. Final summary
    # ========================================================

    summary = pd.DataFrame(records)

    successful_count = int(
        (summary["status"] == "success").sum()
    )

    nonconverged_count = int(
        (summary["status"] == "FOM_not_converged").sum()
    )

    error_count = len(summary) - successful_count - nonconverged_count

    print("\n" + "=" * 76)
    print("AUGMENTED TRAINING-SET SUMMARY")
    print("=" * 76)

    print(f"Selected hotspots: {len(selected_hotspots)}")
    print(f"Generated parameter combinations: {len(augmented_cases)}")
    print(f"Successful FOM cases: {successful_count}")
    print(f"Nonconverged FOM cases: {nonconverged_count}")
    print(f"Cases with errors: {error_count}")

    print(
        f"Snapshot matrix shape: "
        f"{snapshot_matrix.shape}"
    )

    print("\nOutput files:")
    print(f"Hotspot candidates: {HOTSPOT_CANDIDATES_PATH}")
    print(f"Selected hotspots: {SELECTED_HOTSPOTS_PATH}")
    print(f"Parameter grid: {PARAMETER_GRID_PATH}")
    print(f"Simulation summary: {SUMMARY_PATH}")
    print(f"Snapshot matrix: {SNAPSHOT_MATRIX_PATH}")
    print(f"Snapshot metadata: {SNAPSHOT_CASE_IDS_PATH}")

    return {
        "hotspot_candidates": hotspot_candidates,
        "selected_hotspots": selected_hotspots,
        "parameter_grid": parameter_grid,
        "summary": summary,
        "snapshot_matrix": snapshot_matrix,
        "snapshot_case_ids": snapshot_case_ids,
    }


# ============================================================
# 12. Run
# ============================================================

if __name__ == "__main__":

    augmented_training_data = (
        run_augmented_training_set_generation()
    )