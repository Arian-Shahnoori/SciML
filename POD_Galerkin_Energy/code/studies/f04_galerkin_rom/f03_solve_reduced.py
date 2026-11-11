import numpy as np
from pathlib import Path
from studies.f04_galerkin_rom.f01_operators import build_operator_components
from studies.f04_galerkin_rom.f02_reduced_operators import build_reduced_operators


directory = Path(__file__).resolve().parents[1] 
directory = (
        directory / "f03_pod_analysis" / "results"
    )


def construct_reduced_system(
    reduced_operators,
    peclet_number,
    brinkman_number,
    convection_coefficient=1.0,
):
    """
    Construct the reduced steady-state system:

        A_r @ a = b_r

    where:

        A_r = c * Pe * C_r + L_r
        b_r = -Br * Phi_r - Pe * g_r
    """

    if peclet_number <= 0:
        raise ValueError("peclet_number must be positive.")

    C_r = np.asarray(reduced_operators["C_r"], dtype=float)
    L_r = np.asarray(reduced_operators["L_r"], dtype=float)

    Phi_r = np.asarray(
        reduced_operators["Phi_r"], dtype=float
    ).ravel()

    g_r = np.asarray(
        reduced_operators["g_r"], dtype=float
    ).ravel()

    n_modes = reduced_operators["n_modes"]

    if C_r.shape != (n_modes, n_modes):
        raise ValueError("Invalid dimensions for C_r.")

    if L_r.shape != (n_modes, n_modes):
        raise ValueError("Invalid dimensions for L_r.")

    if Phi_r.size != n_modes or g_r.size != n_modes:
        raise ValueError("Reduced source dimensions do not match n_modes.")

    # Reduced system matrix.
    A_r = (
        convection_coefficient * peclet_number * C_r
        + L_r
    )

    # Reduced right-hand side.
    b_r = (
        -brinkman_number * Phi_r
        - peclet_number * g_r
    )

    return {
        "A_r": A_r,
        "b_r": b_r,
    }


def solve_reduced_system(reduced_system):
    """
    Solve the reduced linear system:

        A_r @ a = b_r

    Returns the reduced coefficients and solver diagnostics.
    """

    A_r = reduced_system["A_r"]
    b_r = reduced_system["b_r"]

    try:
        a = np.linalg.solve(A_r, b_r)

    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "The reduced system could not be solved. "
            "The reduced matrix may be singular."
        ) from error

    # Check the linear-system residual.
    residual = A_r @ a - b_r

    residual_norm = float(np.linalg.norm(residual))

    relative_residual = residual_norm / max(
        float(np.linalg.norm(b_r)),
        1e-14,
    )

    return {
        "a": a,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "condition_number": float(np.linalg.cond(A_r)),
    }


def reconstruct_temperature(
    a,
    reduced_operators,
    reference_temperature,
    mesh,
):
    """
    Reconstruct the perturbation and physical temperature:

        theta_r = V_r @ a
        T_r = T_ref + theta_r

    Returns two-dimensional fields with shape (nx, ny).
    """

    V_r = reduced_operators["V_r"]

    reference_temperature = np.asarray(
        reference_temperature,
        dtype=float,
    ).ravel(order="C")

    # Reconstruct the temperature perturbation.
    theta_vector = V_r @ a

    if theta_vector.size != reference_temperature.size:
        raise ValueError(
            "The reconstructed perturbation and reference "
            "temperature have incompatible dimensions."
        )

    # Recover the physical temperature.
    temperature_vector = reference_temperature + theta_vector

    nx, ny = mesh

    if temperature_vector.size != nx * ny:
        raise ValueError("Mesh dimensions do not match the temperature field.")

    theta_field = theta_vector.reshape(mesh, order="C")
    temperature_field = temperature_vector.reshape(mesh, order="C")

    return {
        "theta": theta_field,
        "temperature": temperature_field,
    }


def run_reduced_model(
    n_modes=3,
    brinkman_number=0.2,
    convection_coefficient=1.0,
    peclet_number = 5,
    save_results=True,
    pod_modes_path= directory / "pod_modes.npy",
):
    """
    Construct and solve the reduced-order model, reconstruct
    the temperature field, and optionally save the results.
    """

    from src.config import (
        DOMAIN_LENGTH_X,
        DOMAIN_LENGTH_Y,
        MEAN_VELOCITY,
    )

    # --------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------

    mesh = (50, 20)

    print("\n========================================")
    print("       REDUCED-ORDER MODEL")
    print("========================================")

    print(f"Mesh: {mesh}")
    print(f"Number of modes: {n_modes}")
    print(f"Pe: {peclet_number}")
    print(f"Br: {brinkman_number}")
    print(f"c:  {convection_coefficient}")

    # --------------------------------------------------
    # 2. Load POD basis
    # --------------------------------------------------

    pod_modes = np.load(pod_modes_path)

    # --------------------------------------------------
    # 3. Construct full-order operators
    # --------------------------------------------------

    full_operators = build_operator_components(
        mesh=mesh,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        peclet_number=peclet_number,
        mean_velocity=MEAN_VELOCITY,
        convection_coefficient=convection_coefficient,
    )

    # --------------------------------------------------
    # 4. Construct reduced operators
    # --------------------------------------------------

    reduced_operators = build_reduced_operators(
        operators=full_operators,
        pod_modes=pod_modes,
        n_modes=n_modes,
    )

    # --------------------------------------------------
    # 5. Construct reduced system
    # --------------------------------------------------

    reduced_system = construct_reduced_system(
        reduced_operators=reduced_operators,
        peclet_number=peclet_number,
        brinkman_number=brinkman_number,
        convection_coefficient=convection_coefficient,
    )

    # --------------------------------------------------
    # 6. Solve for reduced coefficients
    # --------------------------------------------------

    solution = solve_reduced_system(reduced_system)

    a = solution["a"]

    print("\n--- Reduced system ---")
    print(f"A_r shape: {reduced_system['A_r'].shape}")
    print(f"b_r shape: {reduced_system['b_r'].shape}")

    print("\n--- Reduced solution ---")
    print(f"Coefficients shape: {a.shape}")
    print(f"Coefficients:\n{a}")

    print(f"\nLinear-system residual: {solution['residual_norm']:.6e}")
    print(f"Relative residual: {solution['relative_residual']:.6e}")
    print(f"Condition number: {solution['condition_number']:.6e}")

    # --------------------------------------------------
    # 7. Reconstruct temperature
    # --------------------------------------------------

    reconstruction = reconstruct_temperature(
        a=a,
        reduced_operators=reduced_operators,
        reference_temperature=full_operators["T_ref"],
        mesh=mesh,
    )

    theta = reconstruction["theta"]
    temperature = reconstruction["temperature"]

    print("\n--- Reconstructed fields ---")
    print(f"Perturbation shape: {theta.shape}")
    print(f"Temperature shape: {temperature.shape}")
    print(f"Temperature minimum: {np.min(temperature):.6e}")
    print(f"Temperature maximum: {np.max(temperature):.6e}")

    # --------------------------------------------------
    # 8. Save results
    # --------------------------------------------------

    if save_results:

        def format_value(value):
            return f"{value:g}".replace("-", "m").replace(".", "p")

        nx, ny = mesh

        filename = (
            f"ROM_n{n_modes}"
            f"_Pe{format_value(peclet_number)}"
            f"_Br{format_value(brinkman_number)}"
            f"_c{format_value(convection_coefficient)}"
            f"_U{format_value(MEAN_VELOCITY)}"
            f"_nx{nx}_ny{ny}.npz"
        )

        output_directory = Path("reduced_results")
        output_directory.mkdir(parents=True, exist_ok=True)

        output_path = output_directory / filename

        np.savez_compressed(
            output_path,
            a=a,
            A_r=reduced_system["A_r"],
            b_r=reduced_system["b_r"],
            theta=theta,
            temperature=temperature,
            n_modes=n_modes,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            convection_coefficient=convection_coefficient,
            mean_velocity=MEAN_VELOCITY,
            mesh=mesh,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
        )

        print(f"\nResults saved to: {output_path}")

    return {
        "reduced_operators": reduced_operators,
        "reduced_system": reduced_system,
        "solution": solution,
        "reconstruction": reconstruction,
    }


if __name__ == "__main__":
    run_reduced_model(n_modes=3)