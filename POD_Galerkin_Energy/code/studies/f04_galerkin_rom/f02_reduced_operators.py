import numpy as np
from studies.f04_galerkin_rom.f01_operators import build_operator_components
from pathlib import Path

directory = Path(__file__).resolve().parents[1] 
directory = (
        directory / "f03_pod_analysis" / "results"
    )


def build_reduced_operators(
    operators,
    pod_modes,
    n_modes,
):
    """
    Construct reduced-order operators using the first n_modes POD modes.

    Full-order operators:
        C_h, L_h, Phi_h, g_lift

    Reduced-order operators:
        C_r   = V_r.T @ C_h @ V_r
        L_r   = V_r.T @ L_h @ V_r
        Phi_r = V_r.T @ Phi_h
        g_r   = V_r.T @ g_lift
    """

    # Validate the number of modes.
    if not isinstance(n_modes, (int, np.integer)):
        raise TypeError("n_modes must be an integer.")

    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")

    # Load POD basis as a NumPy array.
    V = np.asarray(pod_modes, dtype=float)

    if V.ndim != 2:
        raise ValueError("pod_modes must be a two-dimensional array.")

    if n_modes > V.shape[1]:
        raise ValueError(
            f"Requested {n_modes} modes, but only "
            f"{V.shape[1]} modes are available."
        )

    # Extract the retained POD basis.
    V_r = V[:, :n_modes]

    # Extract full-order operators.
    C_h = operators["C_h"]
    L_h = operators["L_h"]

    Phi_h = np.asarray(operators["Phi_h"]).ravel(order="C")
    g_lift = np.asarray(operators["g_lift"]).ravel(order="C")

    # Check dimensions.
    n_dof = V_r.shape[0]

    if C_h.shape != (n_dof, n_dof):
        raise ValueError("C_h dimensions do not match the POD basis.")

    if L_h.shape != (n_dof, n_dof):
        raise ValueError("L_h dimensions do not match the POD basis.")

    if Phi_h.size != n_dof:
        raise ValueError("Phi_h dimensions do not match the POD basis.")

    if g_lift.size != n_dof:
        raise ValueError("g_lift dimensions do not match the POD basis.")

    # Check POD orthonormality.
    orthogonality_error = np.linalg.norm(
        V_r.T @ V_r - np.eye(n_modes),
        ord="fro",
    )

    if not np.isclose(orthogonality_error, 0.0, atol=1e-8):
        raise ValueError(
            "The retained POD modes are not orthonormal. "
            f"Orthogonality error: {orthogonality_error:.6e}"
        )

    # Project the full-order operators.
    C_r = V_r.T @ (C_h @ V_r)

    L_r = V_r.T @ (L_h @ V_r)

    Phi_r = V_r.T @ Phi_h

    g_r = V_r.T @ g_lift

    return {
        "V_r": V_r,
        "C_r": np.asarray(C_r),
        "L_r": np.asarray(L_r),
        "Phi_r": np.asarray(Phi_r),
        "g_r": np.asarray(g_r),
        "n_modes": n_modes,
        "orthogonality_error": orthogonality_error,
    }



def run_reduced_operator_example(n_modes=3):
    """Build and inspect reduced operators."""

    from src.config import (
        DOMAIN_LENGTH_X,
        DOMAIN_LENGTH_Y,
        MEAN_VELOCITY,
        PECLET_NUMBER,
    )

    # Project configuration.
    mesh = (50, 20)
    convection_coefficient = 1.0

    # Load full POD basis.
    pod_modes = np.load(directory / "pod_modes.npy")

    # Construct full-order operators.
    operators = build_operator_components(
        mesh=mesh,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        peclet_number=PECLET_NUMBER,
        mean_velocity=MEAN_VELOCITY,
        convection_coefficient=convection_coefficient,
    )

    # Construct reduced-order operators.
    reduced_operators = build_reduced_operators(
        operators=operators,
        pod_modes=pod_modes,
        n_modes=n_modes,
    )

    print("\n--- Reduced operator dimensions ---")
    print(f"Number of modes: {reduced_operators['n_modes']}")
    print(f"V_r:   {reduced_operators['V_r'].shape}")
    print(f"C_r:   {reduced_operators['C_r'].shape}")
    print(f"L_r:   {reduced_operators['L_r'].shape}")
    print(f"Phi_r: {reduced_operators['Phi_r'].shape}")
    print(f"g_r:   {reduced_operators['g_r'].shape}")

    print("\n--- POD basis verification ---")
    print(
        "Orthogonality error:",
        f"{reduced_operators['orthogonality_error']:.6e}",
    )

    return reduced_operators


if __name__ == "__main__":
    run_reduced_operator_example(n_modes=3)