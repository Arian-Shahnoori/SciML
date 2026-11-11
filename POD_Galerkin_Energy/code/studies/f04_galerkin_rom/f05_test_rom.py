from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.initialization import initialize_velocity_fields

# Reuse the FOM case runner from the validation script.
from studies.f04_galerkin_rom.f04_rom_for_case import run_fom_case
from studies.f04_galerkin_rom.f04_rom_for_case import run_rom_for_cases

# ============================================================
# 1. Parameter-space and run configuration
# ============================================================

MESH = (50, 20)

MU_MIN = 1.0
MU_MAX = 100.0

BR_MIN = 0.01
BR_MAX = 2.0

# A 20 x 20 grid gives 400 parameter combinations.
N_MU = 10
N_BR = 10

# Since mu = c * Pe, fixing c = 1 makes Pe = mu.
CONVECTION_COEFFICIENT = 1.0

OUTPUT_DIRECTORY = (
    Path("studies")
    / "f04_galerkin_rom"
    / "results"
    / "large_training_sweep"
)

CASE_RESULTS_DIRECTORY = OUTPUT_DIRECTORY / "cases"
SNAPSHOT_MATRIX_PATH = OUTPUT_DIRECTORY / "snapshot_matrix.npy"
SNAPSHOT_CASE_IDS_PATH = OUTPUT_DIRECTORY / "snapshot_case_ids.csv"
PARAMETER_GRID_PATH = OUTPUT_DIRECTORY / "parameter_grid.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "simulation_summary.csv"

# Existing successful cases are loaded instead of being run again.
RESUME_FROM_EXISTING_CASES = True


def format_value(value):
    """Format a number so it can safely be included in a filename."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


def build_parameter_grid(n_mu=N_MU, n_br=N_BR):
    """Build a uniform grid in effective parameter mu and Brinkman number."""
    if n_mu < 1 or n_br < 1:
        raise ValueError("n_mu and n_br must both be at least 1.")

    if MU_MIN <= 0.0 or MU_MAX < MU_MIN:
        raise ValueError("Check MU_MIN and MU_MAX; both must be positive.")

    if BR_MIN < 0.0 or BR_MAX < BR_MIN:
        raise ValueError("Check BR_MIN and BR_MAX.")

    if CONVECTION_COEFFICIENT <= 0.0:
        raise ValueError("CONVECTION_COEFFICIENT must be positive.")

    mu_values = np.linspace(MU_MIN, MU_MAX, n_mu)
    br_values = np.linspace(BR_MIN, BR_MAX, n_br)

    cases = []
    case_number = 0

    for mu in mu_values:
        for brinkman_number in br_values:
            case_number += 1

            # Preserve mu = c * Pe for every generated case.
            peclet_number = float(mu / CONVECTION_COEFFICIENT)

            cases.append({
                "case_id": f"train_{case_number:04d}",
                "effective_parameter_mu": float(mu),
                "peclet_number": peclet_number,
                "brinkman_number": float(brinkman_number),
                "convection_coefficient": float(
                    CONVECTION_COEFFICIENT
                ),
            })

    return cases


def run_rom_test():
    TEST_CASES = build_parameter_grid()

    validation_results = run_rom_for_cases(
        cases=TEST_CASES,
    )