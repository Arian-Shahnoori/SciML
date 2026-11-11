# ============================================================
# Imports
# ============================================================

# 1. Direct solver and parameter studies
from studies.f00_solve_with_direct import run_direct_solver
from studies.f01_parameter_grid import run_parameter_grid
from studies.f02_parameter_sweep import run_parameter_sweep

# 2. POD analysis
from studies.f03_pod_analysis.f01_snapshot_matrix import build_snapshot_matrix
from studies.f03_pod_analysis.f02_pod import perform_pod
from studies.f03_pod_analysis.f03_visualize_pod_modes import visualize_pod_modes
from studies.f03_pod_analysis.f04_pod_reconstruction import reconstruct_snapshots
from studies.f03_pod_analysis.f05_visualize_reconstruction import visualize_reconstruction

# 3. Galerkin ROM
from studies.f04_galerkin_rom.f01_operators import run_operator_construction_verification
from studies.f04_galerkin_rom.f02_reduced_operators import run_reduced_operator_example
from studies.f04_galerkin_rom.f03_solve_reduced import run_reduced_model
from studies.f04_galerkin_rom.f04_rom_for_case import run_rom_for_cases
from studies.f04_galerkin_rom.f05_test_rom import run_rom_test

# 4. POD refinement and validation
from studies.f05_refinement.f00_generate_large_training_cases import run_training_case_generation
from studies.f05_refinement.f01_build_pod_basis import build_pod_basis
from studies.f05_refinement.f02_generate_large_test_cases import run_lhs_test_case_generation
from studies.f05_refinement.f03_mode_count_study import run_mode_count_study
from studies.f05_refinement.f04_generate_augmented_training_set import run_augmented_training_set_generation
from studies.f05_refinement.f05_build_augmented_pod_basis import build_augmented_pod_basis
from studies.f05_refinement.f06_generate_independent_test_set import run_independent_test_set_generation
from studies.f05_refinement.f07_compare_original_and_augmented_bases import run_basis_comparison


# ============================================================
# Global configuration
# ============================================================

MODE_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10]


# ============================================================
# Main execution
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Direct solver and parameter studies
    # --------------------------------------------------------

    # run_direct_solver()
    # run_parameter_grid()
    # run_parameter_sweep()


    # --------------------------------------------------------
    # 2. POD analysis
    # --------------------------------------------------------

    # build_snapshot_matrix()

    # U, singular_values, Vt, cumulative_energy = perform_pod()

    #visualize_pod_modes(
    #    retained_modes=3,
    #)

    # X_reconstructed, A_r, reconstruction_errors = reconstruct_snapshots(
    #     retained_modes=6,
    # )

    #visualize_reconstruction(
    #    retained_modes=3,
    #)


    # --------------------------------------------------------
    # 3. Galerkin ROM
    # --------------------------------------------------------

    #run_operator_construction_verification()

    # run_reduced_operator_example()

    # run_reduced_model(
    #     n_modes=3,
    #     peclet_number=5,
    #     brinkman_number=0.2,
    #     convection_coefficient=1.0,
    #     save_results=False,
    # )

    # run_rom_for_cases()
    #run_rom_test()


    # --------------------------------------------------------
    # 4. POD refinement
    # --------------------------------------------------------

    # 4.1 Original training dataset and POD basis

    #run_training_case_generation()
    #build_pod_basis()


    # 4.2 Original basis: independent test and mode-count study

    #run_lhs_test_case_generation()

    #run_mode_count_study(
    #    mode_counts=MODE_COUNTS,
    #)


    # 4.3 Augmented training dataset and POD basis

    #run_augmented_training_set_generation()
    #build_augmented_pod_basis()


    # 4.4 Original vs augmented basis comparison

    #run_independent_test_set_generation()

    run_basis_comparison(
       mode_counts=MODE_COUNTS,
    )


if __name__ == "__main__":
    main()