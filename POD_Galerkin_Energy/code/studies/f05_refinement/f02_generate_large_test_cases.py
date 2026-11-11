from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src import config
from src.initialization import initialize_velocity_fields
from studies.f04_galerkin_rom.f04_rom_for_case import run_fom_case


# ============================================================
# 1. Parameter-space configuration
# ============================================================

MESH = (50, 20)

MU_MIN = 1.0
MU_MAX = 100.0
BR_MIN = 0.01
BR_MAX = 2.0

N_TEST_CASES = 100
LHS_SEED = 20261107

# mu = c * Pe. With c fixed at 1, Pe = mu.
CONVECTION_COEFFICIENT = 1.0

# Generate a fresh set, separate from the existing test set.
TEST_SET_DIRECTORY = (
    Path("studies")
    / "f05_refinement"
    / "results"
    / "independent_test_set_lhs"
)
CASE_RESULTS_DIRECTORY = TEST_SET_DIRECTORY / "cases"
PARAMETER_GRID_PATH = TEST_SET_DIRECTORY / "test_parameter_grid.csv"
SUMMARY_PATH = TEST_SET_DIRECTORY / "simulation_summary.csv"

RESUME_FROM_EXISTING_CASES = True

# Exclude parameter pairs already used to train either POD basis, as well
# as points in the previously generated independent test set.
EXCLUSION_GRID_PATHS = (
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
    Path("studies")
    / "f05_refinement"
    / "results"
    / "independent_test_set"
    / "test_parameter_grid.csv",
)


# ============================================================
# 2. Shared problem setup
# ============================================================

def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


def make_reference_temperature(mesh):
    """Construct T_ref = y at the physical cell centres."""
    nx, ny = mesh
    dy = config.DOMAIN_LENGTH_Y / ny
    y_centres = (np.arange(ny) + 0.5) * dy
    return np.broadcast_to(y_centres[None, :], (nx, ny)).copy()


def format_value(value):
    """Format a number for use in a filename."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parameter_key(mu, br):
    """Create a stable key for comparing parameter pairs."""
    return round(float(mu), 10), round(float(br), 10)


# ============================================================
# 3. Exclude previously used parameter combinations
# ============================================================

def load_excluded_parameter_keys():
    """Load parameter pairs from existing training/test grids when present."""
    excluded = set()
    loaded_files = []

    for grid_path in EXCLUSION_GRID_PATHS:
        if not grid_path.exists():
            continue

        grid = pd.read_csv(grid_path)
        required = {"effective_parameter_mu", "brinkman_number"}
        missing = required.difference(grid.columns)
        if missing:
            raise ValueError(
                f"Cannot read exclusion grid {grid_path}; missing columns: "
                f"{sorted(missing)}"
            )

        for mu, br in zip(
            grid["effective_parameter_mu"],
            grid["brinkman_number"],
        ):
            if pd.notna(mu) and pd.notna(br):
                excluded.add(parameter_key(mu, br))

        loaded_files.append(grid_path)

    print("Existing parameter grids used for exclusion:")
    if loaded_files:
        for path in loaded_files:
            print(f"  - {path}")
    else:
        print("  None found; no previous parameter pairs were excluded.")

    return excluded


# ============================================================
# 4. Generate a Latin-hypercube parameter set
# ============================================================

def build_latin_hypercube_cases(
    n_cases=N_TEST_CASES,
    seed=LHS_SEED,
):
    """Generate an LHS design in (mu, Br), avoiding prior parameter pairs."""
    if n_cases < 1:
        raise ValueError("n_cases must be at least 1.")
    if not (0.0 < MU_MIN < MU_MAX):
        raise ValueError("Invalid mu bounds.")
    if not (0.0 <= BR_MIN < BR_MAX):
        raise ValueError("Invalid Brinkman-number bounds.")
    if CONVECTION_COEFFICIENT <= 0.0:
        raise ValueError("CONVECTION_COEFFICIENT must be positive.")

    excluded_keys = load_excluded_parameter_keys()

    # Exact/rounded collisions are extremely unlikely for continuous LHS
    # samples, but check them explicitly rather than silently accepting one.
    for attempt in range(100):
        actual_seed = int(seed) + attempt
        sampler = qmc.LatinHypercube(d=2, seed=actual_seed)
        unit_samples = sampler.random(n=n_cases)
        scaled_samples = qmc.scale(
            unit_samples,
            l_bounds=[MU_MIN, BR_MIN],
            u_bounds=[MU_MAX, BR_MAX],
        )

        keys = [parameter_key(mu, br) for mu, br in scaled_samples]
        has_internal_duplicates = len(set(keys)) != len(keys)
        overlaps_previous_data = any(key in excluded_keys for key in keys)

        if not has_internal_duplicates and not overlaps_previous_data:
            break
    else:
        raise RuntimeError(
            "Could not generate an LHS design without duplicate or previously "
            "used parameter pairs after 100 deterministic seed attempts."
        )

    cases = []
    for index, (mu, br) in enumerate(scaled_samples, start=1):
        mu = float(mu)
        br = float(br)
        peclet_number = mu / CONVECTION_COEFFICIENT

        cases.append({
            "case_id": f"lhs_test_{index:04d}",
            "sampling_method": "LatinHypercube",
            "lhs_seed": actual_seed,
            "effective_parameter_mu": mu,
            "peclet_number": float(peclet_number),
            "brinkman_number": br,
            "convection_coefficient": float(CONVECTION_COEFFICIENT),
        })

    return cases


# ============================================================
# 5. Case-file handling
# ============================================================

def make_case_path(case):
    filename = (
        f"{case['case_id']}"
        f"_mu{format_value(case['effective_parameter_mu'])}"
        f"_Br{format_value(case['brinkman_number'])}.npz"
    )
    return CASE_RESULTS_DIRECTORY / filename


def load_existing_case(case, case_path, mesh):
    """Load a matching, converged case from a previous run."""
    with np.load(case_path, allow_pickle=False) as saved:
        if not bool(saved["converged"]):
            return None

        if not np.array_equal(saved["mesh"], np.asarray(mesh)):
            return None

        expected_parameters = {
            "effective_parameter_mu": case["effective_parameter_mu"],
            "peclet_number": case["peclet_number"],
            "brinkman_number": case["brinkman_number"],
            "convection_coefficient": case["convection_coefficient"],
        }
        for key, expected_value in expected_parameters.items():
            if not np.isclose(float(saved[key]), expected_value):
                return None

        temperature = np.asarray(saved["temperature"], dtype=float)
        theta = np.asarray(saved["theta"], dtype=float)
        if temperature.shape != mesh or theta.shape != mesh:
            return None
        if not (
            np.all(np.isfinite(temperature))
            and np.all(np.isfinite(theta))
        ):
            return None

        return {
            "temperature": temperature,
            "theta": theta,
            "final_residual": float(saved["final_residual"]),
            "iterations": int(saved["iterations"]),
            "computation_time_s": float(saved["computation_time_s"]),
        }


def save_case(case, result, mesh, case_path):
    """Save one converged FOM solution and its metadata."""
    np.savez_compressed(
        case_path,
        case_id=case["case_id"],
        sampling_method=case["sampling_method"],
        lhs_seed=case["lhs_seed"],
        effective_parameter_mu=case["effective_parameter_mu"],
        peclet_number=case["peclet_number"],
        brinkman_number=case["brinkman_number"],
        convection_coefficient=case["convection_coefficient"],
        mesh=np.asarray(mesh),
        temperature=result["temperature"],
        theta=result["theta"],
        converged=True,
        final_residual=result["final_residual"],
        iterations=result["iterations"],
        computation_time_s=result["computation_time_s"],
    )


# ============================================================
# 6. Run FOM for every LHS point
# ============================================================

def run_lhs_test_case_generation(n_cases=N_TEST_CASES, seed=LHS_SEED):
    """Generate independent LHS parameter points and save their FOM results."""
    start_total = time.perf_counter()

    TEST_SET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CASE_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    mesh = MESH
    nx, ny = mesh
    reference_temperature = make_reference_temperature(mesh)

    cases = build_latin_hypercube_cases(n_cases=n_cases, seed=seed)
    parameter_grid = pd.DataFrame(cases)
    parameter_grid.to_csv(PARAMETER_GRID_PATH, index=False)

    # Velocity is fixed for all parameter combinations.
    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    records = []

    print("\n" + "=" * 76)
    print("INDEPENDENT LATIN-HYPERCUBE FOM TEST-CASE GENERATION")
    print("=" * 76)
    print(f"Mesh: {nx} x {ny}")
    print(f"mu range: [{MU_MIN:g}, {MU_MAX:g}]")
    print(f"Br range: [{BR_MIN:g}, {BR_MAX:g}]")
    print(f"Number of LHS cases: {len(cases)}")
    print(f"LHS seed: {cases[0]['lhs_seed']}")
    print(f"Convection coefficient: {CONVECTION_COEFFICIENT:g}")
    print(f"Output directory: {TEST_SET_DIRECTORY}")

    for case_index, case in enumerate(cases, start=1):
        case_path = make_case_path(case)
        print("\n" + "-" * 76)
        print(f"Case {case_index}/{len(cases)}: {case['case_id']}")
        print(
            f"mu = {case['effective_parameter_mu']:.8g}, "
            f"Br = {case['brinkman_number']:.8g}, "
            f"Pe = {case['peclet_number']:.8g}, "
            f"c = {case['convection_coefficient']:.8g}"
        )

        record = {
            **case,
            "status": "failed",
            "final_residual": np.nan,
            "iterations": np.nan,
            "computation_time_s": np.nan,
            "result_file": "",
        }

        try:
            case_result = None

            if RESUME_FROM_EXISTING_CASES and case_path.exists():
                try:
                    case_result = load_existing_case(
                        case=case,
                        case_path=case_path,
                        mesh=mesh,
                    )
                    if case_result is not None:
                        print("Loaded existing converged FOM result.")
                except (OSError, ValueError, KeyError) as exc:
                    print(
                        "Could not reuse existing result; recomputing. "
                        f"Reason: {exc}"
                    )

            if case_result is None:
                fom_case = {
                    "peclet_number": case["peclet_number"],
                    "brinkman_number": case["brinkman_number"],
                    "convection_coefficient": case[
                        "convection_coefficient"
                    ],
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
                if temperature.shape != mesh:
                    raise ValueError(
                        f"Unexpected temperature shape {temperature.shape}; "
                        f"expected {mesh}."
                    )
                if not np.all(np.isfinite(temperature)):
                    raise FloatingPointError(
                        "The FOM temperature contains non-finite values."
                    )

                if not bool(fom_result["converged"]):
                    record.update({
                        "status": "FOM_not_converged",
                        "final_residual": float(
                            fom_result["final_residual"]
                        ),
                        "iterations": int(fom_result["iterations"]),
                        "computation_time_s": float(
                            fom_result["computation_time"]
                        ),
                    })
                    print(
                        "FOM did not converge. Final residual: "
                        f"{record['final_residual']:.6e}"
                    )
                else:
                    theta = temperature - reference_temperature
                    case_result = {
                        "temperature": temperature,
                        "theta": theta,
                        "final_residual": float(
                            fom_result["final_residual"]
                        ),
                        "iterations": int(fom_result["iterations"]),
                        "computation_time_s": float(
                            fom_result["computation_time"]
                        ),
                    }
                    save_case(
                        case=case,
                        result=case_result,
                        mesh=mesh,
                        case_path=case_path,
                    )

            if case_result is not None:
                record.update({
                    "status": "success",
                    "final_residual": case_result["final_residual"],
                    "iterations": case_result["iterations"],
                    "computation_time_s": case_result[
                        "computation_time_s"
                    ],
                    "result_file": str(case_path),
                })
                print(
                    f"Converged. Residual: "
                    f"{case_result['final_residual']:.6e}"
                )
                print(f"Iterations: {case_result['iterations']}")

        except Exception as exc:
            record["status"] = f"error: {type(exc).__name__}: {exc}"
            print(f"FAILED: {type(exc).__name__}: {exc}")

        records.append(record)
        pd.DataFrame(records).to_csv(SUMMARY_PATH, index=False)

    summary = pd.DataFrame(records)
    successful_count = int((summary["status"] == "success").sum())
    failed_count = len(summary) - successful_count
    total_time = time.perf_counter() - start_total

    print("\n" + "=" * 76)
    print("LATIN-HYPERCUBE FOM TEST-CASE GENERATION SUMMARY")
    print("=" * 76)
    print(f"Total cases attempted: {len(cases)}")
    print(f"Successful cases: {successful_count}")
    print(f"Failed/non-converged cases: {failed_count}")
    print(f"Total elapsed time: {total_time:.2f} s")
    print(f"Parameter grid: {PARAMETER_GRID_PATH}")
    print(f"Simulation summary: {SUMMARY_PATH}")
    print(f"Individual FOM cases: {CASE_RESULTS_DIRECTORY}")
    print("No snapshot matrix was created; these cases are reserved for testing.")

    return {
        "parameter_grid": parameter_grid,
        "summary": summary,
    }


if __name__ == "__main__":
    test_data = run_lhs_test_case_generation(
        n_cases=N_TEST_CASES,
        seed=LHS_SEED,
    )
