from pathlib import Path
import numpy as np
import csv

from src.config import PECLET_NUMBER, BRINKMAN_NUMBER, convection_coefficient


# Baseline parameters
MU_BASELINE = convection_coefficient * PECLET_NUMBER
BR_BASELINE = BRINKMAN_NUMBER

# Parameter ranges: set these for your study
MU_MIN = 1
MU_MAX = 100

BR_MIN = 0.01
BR_MAX = 2

# Number of parameter values along each axis
N_MU = 5
N_BR = 5


def generate_parameter_grid():

    if None in (MU_MIN, MU_MAX, BR_MIN, BR_MAX):
        raise ValueError("Set the parameter bounds before generating the grid.")

    if MU_MIN <= 0 or MU_MAX <= MU_MIN:
        raise ValueError("Invalid mu range.")

    if BR_MIN < 0 or BR_MAX <= BR_MIN:
        raise ValueError("Invalid Brinkman range.")

    if N_MU < 2 or N_BR < 2:
        raise ValueError("Each grid dimension must contain at least two values.")

    # Generate parameter values
    mu_values = np.linspace(MU_MIN, MU_MAX, N_MU)
    br_values = np.linspace(BR_MIN, BR_MAX, N_BR)

    # Generate the tensor-product grid
    mu_grid, br_grid = np.meshgrid(
        mu_values,
        br_values,
        indexing="ij",
    )

    parameter_grid = np.column_stack((
        mu_grid.ravel(),
        br_grid.ravel(),
    ))

    return mu_values, br_values, parameter_grid


def save_parameter_grid(parameter_grid):

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "parameter_grid.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["case_id", "mu", "Br"])

        for case_id, (mu, br) in enumerate(parameter_grid, start=1):
            writer.writerow([case_id, mu, br])

    print(f"Parameter grid saved to: {output_file}")


def run_parameter_grid():

    mu_values, br_values, parameter_grid = generate_parameter_grid()

    print("\nParameter grid")
    print("--------------")
    print(f"Baseline mu: {MU_BASELINE:.6f}")
    print(f"Baseline Br: {BR_BASELINE:.6f}")
    print(f"Number of mu values: {len(mu_values)}")
    print(f"Number of Br values: {len(br_values)}")
    print(f"Total cases: {len(parameter_grid)}")

    print("\nmu values:", mu_values)
    print("Br values:", br_values)

    print(parameter_grid)

    save_parameter_grid(parameter_grid)

    return parameter_grid


