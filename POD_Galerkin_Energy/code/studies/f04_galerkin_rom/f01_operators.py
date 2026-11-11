import numpy as np

from scipy.sparse.linalg import norm as sparse_norm

from src.boundary_conditions import apply_temperature_boundary_conditions
from src.initialization import initialize_velocity_fields
from src.solvers.direct import (
    build_spatial_matrix,
    build_forcing_field,
)


# ============================================================
# 1. Inlet temperature
# ============================================================

def linear_inlet_temperature(y):
    """Linear inlet temperature profile: T_in(y) = y."""
    return y


# ============================================================
# 2. Prepare velocity fields
# ============================================================

def prepare_velocity_fields(
    mesh,
    domain_length_y,
    mean_velocity,
    u_velocity=None,
    v_velocity=None,
):
    """
    Generate or validate the prescribed velocity fields.

    Both velocity components must be supplied together.
    If neither is supplied, they are initialized using the
    existing velocity initialization function.
    """

    nx, ny = mesh
    expected_shape = (nx + 2, ny + 2)

    if u_velocity is None and v_velocity is None:

        u_velocity, v_velocity = initialize_velocity_fields(
            mesh=mesh,
            domain_length_y=domain_length_y,
            mean_velocity=mean_velocity,
        )

    elif u_velocity is None or v_velocity is None:

        raise ValueError(
            "Provide both u_velocity and v_velocity, or neither."
        )

    u_velocity = np.asarray(u_velocity, dtype=float)
    v_velocity = np.asarray(v_velocity, dtype=float)

    if u_velocity.shape != expected_shape:
        raise ValueError(
            f"Unexpected u_velocity shape: {u_velocity.shape}. "
            f"Expected {expected_shape}."
        )

    if v_velocity.shape != expected_shape:
        raise ValueError(
            f"Unexpected v_velocity shape: {v_velocity.shape}. "
            f"Expected {expected_shape}."
        )

    return u_velocity, v_velocity


# ============================================================
# 3. Internal spatial matrix assembly
# ============================================================

def _assemble_spatial_matrix(
    mesh,
    domain_length_x,
    domain_length_y,
    peclet_number,
    mean_velocity,
    u_velocity,
    v_velocity,
    convection_coefficient,
):
    """
    Call the existing full-order spatial matrix assembler.

    The resulting matrix represents:

        dT/dt = A @ T + b
    """

    nx, ny = mesh

    return build_spatial_matrix(
        nx=nx,
        ny=ny,
        length_x=domain_length_x,
        length_y=domain_length_y,
        peclet_number=peclet_number,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        convection_coefficient=convection_coefficient,
    )


# ============================================================
# 4. Construct convection and diffusion operators
# ============================================================

def build_linear_operators(
    mesh,
    domain_length_x,
    domain_length_y,
    mean_velocity,
    u_velocity=None,
    v_velocity=None,
):
    """
    Construct the parameter-independent discrete operators:

        C_h : convection operator
        L_h : diffusion operator

    The existing full-order matrix has the decomposition:

        A = c*C_h + (1/Pe)*L_h

    Both operators use exactly the same spatial discretization
    and boundary closures as build_spatial_matrix().
    """

    u_velocity, v_velocity = prepare_velocity_fields(
        mesh=mesh,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # Unit Peclet number and unit convection coefficient:
    # A_unit = C_h + L_h
    A_unit = _assemble_spatial_matrix(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        peclet_number=1.0,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        convection_coefficient=1.0,
    )

    # Unit Peclet number and zero convection coefficient:
    # This isolates L_h.
    L_h = _assemble_spatial_matrix(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        peclet_number=1.0,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        convection_coefficient=0.0,
    )

    # Extract convection operator.
    C_h = (A_unit - L_h).tocsr()

    return C_h, L_h, u_velocity, v_velocity


def build_convection_operator(
    mesh,
    domain_length_x,
    domain_length_y,
    mean_velocity,
    u_velocity=None,
    v_velocity=None,
):
    """Build and return the discrete convection operator C_h."""

    C_h, _, _, _ = build_linear_operators(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    return C_h


def build_diffusion_operator(
    mesh,
    domain_length_x,
    domain_length_y,
    mean_velocity,
    u_velocity=None,
    v_velocity=None,
):
    """Build and return the discrete diffusion operator L_h."""

    _, L_h, _, _ = build_linear_operators(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    return L_h


# ============================================================
# 5. Internal forcing-field assembly
# ============================================================

def _assemble_forcing_field(
    mesh,
    inlet_temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    convection_coefficient,
):
    """
    Call the existing full-order forcing-field assembler.

    The output contains the affine boundary contributions
    and viscous-dissipation source.
    """

    return build_forcing_field(
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=peclet_number,
        brinkman_number=brinkman_number,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        convection_coefficient=convection_coefficient,
    )


# ============================================================
# 6. Construct viscous-dissipation source
# ============================================================

def build_viscous_dissipation_source(
    mesh,
    domain_length_x,
    domain_length_y,
    u_velocity,
    v_velocity,
):
    """
    Calculate the parameter-independent viscous-dissipation
    source vector Phi_h.

    Does not include Br/Pe.
    """

    nx, ny = mesh

    dx = domain_length_x / nx
    dy = domain_length_y / ny

    Phi = np.zeros((nx, ny), dtype=float)

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):

            du_dx = (
                u_velocity[i + 1, j] - u_velocity[i - 1, j]
            ) / (2 * dx)

            dv_dy = (
                v_velocity[i, j + 1] - v_velocity[i, j - 1]
            ) / (2 * dy)

            dv_dx = (
                v_velocity[i + 1, j] - v_velocity[i - 1, j]
            ) / (2 * dx)

            du_dy = (
                u_velocity[i, j + 1] - u_velocity[i, j - 1]
            ) / (2 * dy)

            Phi[i - 1, j - 1] = (
                2 * du_dx**2
                + 2 * dv_dy**2
                + (dv_dx + du_dy)**2
            )

    return Phi.ravel(order="C")


# ============================================================
# 7. Construct reference temperature vector
# ============================================================

def build_reference_temperature(
    mesh,
    domain_length_y,
):
    """
    Construct the reference field T_ref = y.

    Uses physical cell-centre values and the same C-order
    vectorization as the full-order spatial matrix.
    """

    nx, ny = mesh

    dy = domain_length_y / ny

    y_centres = (
        np.arange(ny) + 0.5
    ) * dy

    reference_field = np.broadcast_to(
        y_centres[None, :],
        (nx, ny),
    ).copy()

    return reference_field.ravel(order="C")


# ============================================================
# 8. Construct discrete lifting residual
# ============================================================

def build_lifting_residual(
    mesh,
    domain_length_x,
    domain_length_y,
    peclet_number,
    mean_velocity,
    convection_coefficient=1.0,
    u_velocity=None,
    v_velocity=None,
    inlet_temperature=linear_inlet_temperature,
):
    """
    Calculate the discrete lifting residual directly from the
    convection and diffusion stencils.

        g_lift = A @ T_ref + b_boundary

    The viscous-dissipation source is excluded.
    """

    if peclet_number <= 0:
        raise ValueError("peclet_number must be positive.")

    nx, ny = mesh

    dx = domain_length_x / nx
    dy = domain_length_y / ny

    # Prepare velocity fields.
    u_velocity, v_velocity = prepare_velocity_fields(
        mesh=mesh,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # Reference temperature at physical cell centres.
    temperature_ref = np.zeros((nx + 2, ny + 2), dtype=float)

    y_centres = (np.arange(ny) + 0.5) * dy

    temperature_ref[1:nx + 1, 1:ny + 1] = y_centres[np.newaxis, :]

    # Apply the same ghost-cell boundary conditions as the FOM.
    temperature_ref = apply_temperature_boundary_conditions(
        temperature=temperature_ref,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=domain_length_y,
    )

    # Evaluate the transport operator directly.
    g_lift = np.zeros((nx, ny), dtype=float)

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):

            # Convection in the x-direction.
            x_flux_difference = (
                u_velocity[i + 1, j] * temperature_ref[i + 1, j]
                - u_velocity[i - 1, j] * temperature_ref[i - 1, j]
            )

            # Diffusion in the x-direction.
            x_diffusion_difference = (
                temperature_ref[i + 1, j]
                - 2.0 * temperature_ref[i, j]
                + temperature_ref[i - 1, j]
            )

            # Convection in the y-direction.
            y_flux_difference = (
                v_velocity[i, j + 1] * temperature_ref[i, j + 1]
                - v_velocity[i, j - 1] * temperature_ref[i, j - 1]
            )

            # Diffusion in the y-direction.
            y_diffusion_difference = (
                temperature_ref[i, j + 1]
                - 2.0 * temperature_ref[i, j]
                + temperature_ref[i, j - 1]
            )

            g_lift[i - 1, j - 1] = (
                -convection_coefficient * x_flux_difference / (2.0 * dx)
                + x_diffusion_difference / (peclet_number * dx**2)
                -convection_coefficient * y_flux_difference / (2.0 * dy)
                + y_diffusion_difference / (peclet_number * dy**2)
            )

    return g_lift.ravel(order="C")


# ============================================================
# 9. Verify operator decomposition
# ============================================================

def verify_operator_decomposition(
    mesh,
    domain_length_x,
    domain_length_y,
    peclet_number,
    mean_velocity,
    convection_coefficient=1.0,
    u_velocity=None,
    v_velocity=None,
    tolerance=1e-10,
    inlet_temperature=linear_inlet_temperature,
    check_zero_lifting=False,
):
    """
    Verify the spatial operator decomposition:

        A = c * C_h + (1 / Pe) * L_h

    Also verify that the directly calculated lifting residual
    agrees with its original definition:

        g_lift = A @ T_ref + b_boundary

    The boundary forcing is evaluated with Br=0.

    If check_zero_lifting=True, additionally verify that g_lift
    is approximately zero. This is expected for the standard
    parallel-flow velocity field and linear reference temperature,
    but not necessarily for arbitrary velocity fields.
    """

    if peclet_number <= 0:
        raise ValueError("peclet_number must be positive.")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    # Prepare velocity fields.
    u_velocity, v_velocity = prepare_velocity_fields(
        mesh=mesh,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # --------------------------------------------------
    # 1. Verify spatial operator decomposition
    # --------------------------------------------------

    C_h, L_h, _, _ = build_linear_operators(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # Original full-order spatial matrix.
    A_full = _assemble_spatial_matrix(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        peclet_number=peclet_number,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        convection_coefficient=convection_coefficient,
    )

    # Matrix reconstructed from separated operators.
    A_reconstructed = (
        convection_coefficient * C_h
        + (1.0 / peclet_number) * L_h
    )

    difference = A_full - A_reconstructed

    operator_absolute_error = float(sparse_norm(difference))

    operator_relative_error = operator_absolute_error / max(
        float(sparse_norm(A_full)),
        1e-14,
    )

    # --------------------------------------------------
    # 2. Verify lifting residual
    # --------------------------------------------------

    # Calculate lifting residual using the direct stencil method.
    g_lift_direct = build_lifting_residual(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        peclet_number=peclet_number,
        mean_velocity=mean_velocity,
        convection_coefficient=convection_coefficient,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        inlet_temperature=inlet_temperature,
    )

    # Reference temperature at physical cell centres.
    T_ref = np.asarray(
        build_reference_temperature(
            mesh=mesh,
            domain_length_y=domain_length_y,
        )
    ).ravel(order="C")

    # Original boundary forcing, with viscous dissipation excluded.
    boundary_forcing = _assemble_forcing_field(
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=peclet_number,
        brinkman_number=0.0,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        convection_coefficient=convection_coefficient,
    ).ravel(order="C")

    # Lifting residual from the original matrix-based definition.
    A_T_ref = np.asarray(A_full @ T_ref).ravel()

    g_lift_reference = A_T_ref + boundary_forcing

    # Compare direct and matrix-based lifting residuals.
    lifting_difference = g_lift_direct - g_lift_reference

    lifting_absolute_error = float(np.linalg.norm(lifting_difference))

    lifting_scale = max(
        float(np.linalg.norm(A_T_ref)),
        float(np.linalg.norm(boundary_forcing)),
        1e-14,
    )

    lifting_relative_error = lifting_absolute_error / lifting_scale

    # Magnitude of the direct lifting residual.
    lifting_residual_l2 = float(np.linalg.norm(g_lift_direct))

    lifting_residual_max = float(np.max(np.abs(g_lift_direct)))

    lifting_is_near_zero = lifting_residual_max <= tolerance

    # --------------------------------------------------
    # 3. Report verification results
    # --------------------------------------------------

    print("\n--- Spatial operator decomposition ---")
    print(f"Absolute error: {operator_absolute_error:.6e}")
    print(f"Relative error: {operator_relative_error:.6e}")

    if operator_relative_error <= tolerance:
        print("PASS: A = c * C_h + (1/Pe) * L_h")
    else:
        print("FAIL: Spatial operator decomposition")

    print("\n--- Lifting residual verification ---")
    print(f"Direct/assembled absolute error: {lifting_absolute_error:.6e}")
    print(f"Direct/assembled relative error: {lifting_relative_error:.6e}")
    print(f"Lifting residual L2 norm: {lifting_residual_l2:.6e}")
    print(f"Lifting residual maximum: {lifting_residual_max:.6e}")

    if lifting_relative_error <= tolerance:
        print("PASS: Direct g_lift agrees with A @ T_ref + b_boundary")
    else:
        print("FAIL: Lifting residual implementations disagree")

    if lifting_is_near_zero:
        print("PASS: g_lift is approximately zero")
    else:
        print("INFO: g_lift is nonzero")

    # --------------------------------------------------
    # 4. Raise errors if verification fails
    # --------------------------------------------------

    failures = []

    if operator_relative_error > tolerance:
        failures.append(
            "The separated operators do not reproduce "
            "the original full-order spatial matrix."
        )

    if lifting_relative_error > tolerance:
        failures.append(
            "The direct lifting residual does not agree "
            "with A @ T_ref + b_boundary."
        )

    if check_zero_lifting and not lifting_is_near_zero:
        failures.append(
            "The lifting residual is not approximately zero."
        )

    if failures:
        raise AssertionError("\n".join(failures))

    return {
        "operator_absolute_error": operator_absolute_error,
        "operator_relative_error": operator_relative_error,
        "lifting_absolute_error": lifting_absolute_error,
        "lifting_relative_error": lifting_relative_error,
        "lifting_residual_l2": lifting_residual_l2,
        "lifting_residual_max": lifting_residual_max,
        "lifting_is_near_zero": lifting_is_near_zero,
        "g_lift": g_lift_direct,
    }


# ============================================================
# 10. Build complete operator set
# ============================================================

def build_operator_components(
    mesh,
    domain_length_x,
    domain_length_y,
    peclet_number,
    mean_velocity,
    convection_coefficient=1.0,
    u_velocity=None,
    v_velocity=None,
    inlet_temperature=linear_inlet_temperature,
):
    """
    Construct all spatial operators and source terms needed
    for the discrete Galerkin-POD model.

    Returns:
        C_h       : discrete convection operator
        L_h       : discrete diffusion operator
        Phi_h     : viscous-dissipation source vector
        g_lift    : discrete lifting residual
        T_ref     : reference temperature vector
        u_velocity: prescribed x-velocity field
        v_velocity: prescribed y-velocity field

    C_h, L_h and Phi_h are parameter-independent provided
    the grid and prescribed velocity field remain fixed.

    g_lift depends on Pe and c and must be evaluated for
    the relevant parameter combination.
    """

    if peclet_number <= 0:
        raise ValueError("peclet_number must be positive.")

    # Prepare velocity fields once.
    u_velocity, v_velocity = prepare_velocity_fields(
        mesh=mesh,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # Convection and diffusion operators.
    C_h, L_h, _, _ = build_linear_operators(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        mean_velocity=mean_velocity,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # Viscous-dissipation source.
    Phi_h = build_viscous_dissipation_source(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
    )

    # Discrete lifting residual.
    g_lift = build_lifting_residual(
        mesh=mesh,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        peclet_number=peclet_number,
        mean_velocity=mean_velocity,
        convection_coefficient=convection_coefficient,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        inlet_temperature=inlet_temperature,
    )

    # Reference temperature vector.
    T_ref = build_reference_temperature(
        mesh=mesh,
        domain_length_y=domain_length_y,
    )

    return {
        "C_h": C_h,
        "L_h": L_h,
        "Phi_h": Phi_h,
        "g_lift": g_lift,
        "T_ref": T_ref,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
    }


# ============================================================
# 11. Example execution
# ============================================================

def run_operator_example():
    """Build operator components using the current project configuration."""

    from src.config import (
        DOMAIN_LENGTH_X,
        DOMAIN_LENGTH_Y,
        MEAN_VELOCITY,
        PECLET_NUMBER,
    )

    mesh = (50, 20)
    convection_coefficient = 1.0

    operators = build_operator_components(
        mesh=mesh,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        peclet_number=PECLET_NUMBER,
        mean_velocity=MEAN_VELOCITY,
        convection_coefficient=convection_coefficient,
    )

    print("\n--- Operator dimensions ---")
    print(f"C_h:    {operators['C_h'].shape}")
    print(f"L_h:    {operators['L_h'].shape}")
    print(f"Phi_h:  {operators['Phi_h'].shape}")
    print(f"g_lift: {operators['g_lift'].shape}")
    print(f"T_ref:  {operators['T_ref'].shape}")

    print("\n--- Lifting residual ---")
    print(f"L2 norm: {np.linalg.norm(operators['g_lift']):.6e}")
    print(
        f"Maximum absolute value: "
        f"{np.max(np.abs(operators['g_lift'])):.6e}"
    )

    return operators


def run_operator_construction_verification():
    """Verify operator decomposition and lifting residual, then run the example."""

    from src.config import (
        DOMAIN_LENGTH_X,
        DOMAIN_LENGTH_Y,
        MEAN_VELOCITY,
        PECLET_NUMBER,
    )

    mesh = (50, 20)
    convection_coefficient = 1.0

    print("\n========================================")
    print("       OPERATOR VERIFICATION")
    print("========================================")

    verification_results = verify_operator_decomposition(
        mesh=mesh,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        peclet_number=PECLET_NUMBER,
        mean_velocity=MEAN_VELOCITY,
        convection_coefficient=convection_coefficient,
        check_zero_lifting=True,
    )

    print("\n========================================")
    print("       OPERATOR CONSTRUCTION")
    print("========================================")

    operators = run_operator_example()

    return {
        "verification": verification_results,
        "operators": operators,
    }


if __name__ == "__main__":
    run_operator_construction_verification()