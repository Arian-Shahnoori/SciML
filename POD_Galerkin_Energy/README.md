# Parameterized 2D Steady-State Energy Equation: Galerkin–POD Reduced-Order Modeling

A Python implementation of a full-order numerical model and a discrete Galerkin–POD reduced-order model (ROM) for a parameterized two-dimensional steady-state energy equation with viscous dissipation. The project also investigates error-guided snapshot enrichment and compares the original and augmented POD bases on a common test set.

## Overview

The model considers thermal convection, diffusion, and viscous dissipation in a rectangular channel. The velocity field is prescribed analytically and remains fixed; only the temperature field is solved. A cell-centred spatial discretization with second-order central differences and ghost-cell boundary closures is used to construct the full-order model (FOM).

A POD basis is obtained from full-order temperature-deviation snapshots. The discretized spatial operators are then projected onto the retained modes to form a Galerkin ROM. The ROM calculates modal coefficients for a parameter combination by solving a reduced linear system; the resulting temperature field is reconstructed from the POD basis and a reference-temperature profile.

The original basis is evaluated over a Latin hypercube sampling (LHS) dataset. High-error parameter regions are used to guide local snapshot enrichment. The original and augmented bases are then compared using the same final set of full-order reference solutions.

## Problem formulation

### Domain and prescribed flow

The computational domain is

\[
\Omega = [0,5]\times[0,1],
\]

with the prescribed parabolic velocity field

\[
u(y)=18y(1-y), \qquad v(y)=0.
\]

The maximum streamwise velocity is 4.5 at the channel centreline, and the mean streamwise velocity is 3. The momentum equations are not solved, and the velocity field does not change with the thermal parameters.

### Governing equation

The steady dimensionless energy equation is

\[
c\left(u\frac{\partial T}{\partial x}+v\frac{\partial T}{\partial y}\right)
=\frac{1}{Pe}\nabla^2T+\frac{Br}{Pe}\Phi,
\]

where `Pe` is the Peclet number, `Br` is the Brinkman number, and `c` is the convection coefficient. For the prescribed velocity field, the continuous viscous-dissipation function is

\[
\Phi(y)=324(1-2y)^2.
\]

The effective parameter is defined as

\[
\mu=cPe.
\]

The study fixes `c = 1`, so `Pe = μ`, and investigates

\[
1\leq\mu\leq100, \qquad 0.01\leq Br\leq2.
\]

### Thermal boundary conditions

\[
T(0,y)=y, \qquad T(x,0)=0, \qquad T(x,1)=1,
\]

\[
\left.\frac{\partial T}{\partial x}\right|_{x=5}=0.
\]

The non-homogeneous inlet and wall conditions are handled by decomposing the temperature into a reference field and a deviation field:

\[
T(x,y)=T_{ref}(y)+\theta(x,y), \qquad T_{ref}(y)=y.
\]

The deviation field therefore has homogeneous boundary conditions. POD snapshots are formed from `θ`, and the reference profile is added back when reconstructing the temperature.

## Full-order numerical model

The domain is discretized using a uniform, cell-centred Cartesian grid.

| Quantity | Value |
|---|---:|
| Domain length, `Lx` | 5 |
| Domain height, `Ly` | 1 |
| Grid | `50 × 20` |
| Physical unknowns | 1,000 |
| Grid spacing, `Δx` | 0.1 |
| Grid spacing, `Δy` | 0.05 |
| Pseudo-time step, `Δt` | 1 |
| FOM RMS residual tolerance | `1e-10` |
| Maximum iterations | `100,000` |

Second-order central differences are used for convection and diffusion. One layer of ghost cells is used to impose the inlet and wall Dirichlet conditions and the zero-gradient outlet condition. The physical field is flattened in C order, with the wall-normal index varying fastest.

After spatial discretization, the FOM is written in affine semi-discrete form:

\[
\frac{d\mathbf T}{dt}=\mathbf A\mathbf T+\mathbf b.
\]

The steady solution is obtained by backward-Euler pseudo-time marching:

\[
(\mathbf I-\Delta t\mathbf A)\mathbf T^{n+1}
=\mathbf T^n+\Delta t\mathbf b.
\]

A sparse LU factorization is reused during the iterations for each fixed parameter case. Pseudo-time is a numerical procedure for reaching a steady solution; the ROM developed in this project is a steady-state model, not a transient reduced-order model.

## POD basis and discrete Galerkin ROM

### Original POD basis

The main training dataset contains 100 full-order solutions from a uniform `10 × 10` grid in `(μ, Br)`. All 100 simulations are reported as successfully converged.

For each case, the reference temperature is subtracted, ghost cells are removed, and the deviation field is flattened to form one column of the snapshot matrix:

\[
\mathbf X=\begin{bmatrix}\boldsymbol\theta_1&\boldsymbol\theta_2&\cdots&\boldsymbol\theta_{100}\end{bmatrix}
\in\mathbb R^{1000\times100}.
\]

The snapshot matrix is decomposed using singular value decomposition:

\[
\mathbf X=\mathbf U\boldsymbol\Sigma\mathbf W^T.
\]

The first `r` columns of `U` form the orthonormal POD basis `V_r`. For the original 100-snapshot basis, the first three modes capture **99.9833%** of the snapshot energy, and the first four capture **99.9995%**. These energy values describe reconstruction of the training snapshot matrix; they do not, by themselves, establish prediction accuracy at new parameter values.

### Discrete Galerkin projection

The full-order spatial operator is decomposed as

\[
\mathbf A=c\mathbf C_h+\frac{1}{Pe}\mathbf L_h,
\]

where `C_h` and `L_h` are the discrete convection and diffusion operators. The dissipation source is represented by `Φ_h`. The reference-temperature transformation also introduces a discrete lifting residual, `g_lift`, which accounts for the reference field and affine boundary contributions.

For `\theta \approx V_r a`, the projected quantities are

\[
\mathbf C_r=\mathbf V_r^T\mathbf C_h\mathbf V_r,\qquad
\mathbf L_r=\mathbf V_r^T\mathbf L_h\mathbf V_r,
\]

\[
\boldsymbol\Phi_r=\mathbf V_r^T\boldsymbol\Phi_h,\qquad
\mathbf g_r=\mathbf V_r^T\mathbf g_{lift}.
\]

The steady reduced system is

\[
(\mu\mathbf C_r+\mathbf L_r)\mathbf a
=-Br\,\boldsymbol\Phi_r-Pe\,\mathbf g_r.
\]

The lifting contribution is retained in the general formulation and should only be omitted when its discrete value has been verified as negligible. The reduced temperature field is reconstructed as

\[
\mathbf T_{ROM}=\mathbf T_{ref}+\mathbf V_r\mathbf a.
\]

This is a discrete Galerkin projection of the spatial operators assembled by the FOM. Its matrix sign convention follows the FOM right-hand-side operators and should not be confused with a separately derived continuous weak-form stiffness matrix.

## Error assessment and snapshot enrichment

### Initial LHS screening

The original basis was evaluated using 100 LHS parameter samples, separate from the original POD training grid. The selected mode counts were

\[
r\in\{1,2,3,4,5,6,8,10\}.
\]

The mean relative discrete L2 error decreased as more modes were retained:

| POD modes | Mean relative L2 error | Maximum relative L2 error |
|---:|---:|---:|
| 1 | 0.395587 | 0.636144 |
| 2 | 0.171552 | 0.410623 |
| 3 | 0.027889 | 0.079345 |
| 4 | 0.007663 | 0.044846 |
| 5 | 0.002532 | 0.031450 |
| 6 | 0.001377 | 0.024884 |
| 8 | 0.000726 | 0.015513 |
| 10 | 0.000468 | 0.010435 |

The triangulated error maps in the report visualize interpolation over the irregular LHS samples; they do not represent additional ROM evaluations between sampled points.

### Hotspot selection and local enrichment

High-error candidates were identified from the screening results at 5, 6, 8, and 10 modes. For each mode count, the 20 largest-error cases were considered and the six highest-ranked cases were retained as candidates. Candidates were ranked separately because the error scales differ across mode counts. Repeated parameter combinations were grouped, and their mean rank score and support fraction were combined as `S_base = 0.70 S_rank + 0.30 S_support`. A normalized Euclidean distance in `(μ, Br)` was then used to encourage spatial diversity through `S_selection = 0.55 S_base + 0.45 d_min`. The algorithm allowed up to six centres; three were selected for this dataset.

Three hotspot centres were selected:

| Hotspot | `μ` | `Br` |
|---:|---:|---:|
| 1 | 4.066653 | 1.769485 |
| 2 | 3.035937 | 0.320114 |
| 3 | 5.720724 | 1.320725 |

Local `3 × 3` candidate grids were generated around the centres using half-grid-spacing offsets. Candidates outside the parameter domain, duplicates, and already represented points were removed. This produced **15 additional full-order snapshots**, and all 15 simulations converged.

The augmented snapshot matrix therefore has dimensions `1000 × 115`. Its first three modes capture 99.9699% of its snapshot energy, compared with 99.9833% for the original basis. The change in cumulative POD energy is not, by itself, a measure of predictive improvement; the two bases are compared below using common full-order test cases.

## Independent comparison: original vs. augmented basis

A final LHS dataset containing 100 full-order reference solutions was used to compare the two bases. Both bases were evaluated on the same cases and at the same eight mode counts, giving 1,600 ROM evaluations in total.

The relative L2 error is calculated from the complete reconstructed temperature fields as `||T_FOM - T_ROM||₂ / ||T_FOM||₂`. RMSE and maximum absolute error are also recorded. The table reports the mean and maximum relative L2 errors for each basis, the percentage reduction in those aggregate errors, and the percentage of individual test cases for which the augmented basis reduced the error. Positive reductions indicate lower errors with the augmented basis; negative reductions indicate higher errors.

| Modes | Mean error, original | Mean error, augmented | Mean reduction | Max. error, original | Max. error, augmented | Max. reduction | Cases improved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.401768 | 0.384505 | 4.30% | 0.634832 | 0.608058 | 4.22% | 100% |
| 2 | 0.176187 | 0.193602 | -9.88% | 0.411629 | 0.468897 | -13.91% | 33% |
| 3 | 0.028506 | 0.036678 | -28.67% | 0.081035 | 0.095283 | -17.58% | 32% |
| 4 | 0.008127 | 0.013629 | -67.69% | 0.044410 | 0.037714 | 15.08% | 28% |
| 5 | 0.002725 | 0.003841 | -40.94% | 0.030992 | 0.009805 | 68.36% | 22% |
| 6 | 0.001571 | 0.000930 | 40.77% | 0.026048 | 0.005040 | 80.65% | 22% |
| 8 | 0.000876 | 0.000141 | 83.91% | 0.018205 | 0.001747 | 90.40% | 11% |
| 10 | 0.000577 | 0.0000174 | 96.98% | 0.013008 | 0.000401 | 96.92% | 20% |

### Interpretation

- Increasing the number of retained modes substantially reduces the original-basis ROM error on the LHS screening dataset.
- Snapshot enrichment does **not** improve the final test-set results uniformly. At 2–5 modes, the augmented basis has a higher mean relative L2 error. At 6, 8, and 10 modes, the mean error decreases by 40.77%, 83.91%, and 96.98%, respectively.
- The reduction in aggregate error at higher mode counts is concentrated in a relatively small subset of cases. Only 22%, 11%, and 20% of cases improve at 6, 8, and 10 modes, respectively. A lower mean and maximum error therefore does not mean that most test cases improve.
- At 10 modes, the mean RMSE decreases from `3.786e-3` to `7.24e-5`, and the largest absolute error across the test set decreases from `0.434450` to `0.014533`.

## Computational cost

The FOM contains 1,000 physical unknowns. A three-mode ROM solves for three modal coefficients, a **99.7% reduction in the dimension of the linear system**. The full temperature field is still reconstructed on all 1,000 grid cells.

The report records a mean FOM time of `0.137740 s` over 100 cases and a mean ROM time of `0.054623 s` over 99 timing records. One ROM timing record is missing, so these values do not support a definitive paired speedup claim. In addition, the current scripts reconstruct full-order operator components for individual cases rather than fully reusing precomputed reduced operators. A complete offline–online implementation and a breakdown of operator construction, projection, reduced solve, and reconstruction costs remain future work.

## Software organization

The report describes two main parts of the codebase:

- `src/`: shared grid, initialization, boundary-condition, diagnostic, and full-order solver modules.
- `Studies/`: scripts for Galerkin ROM evaluation, parameter studies, enrichment, and final comparison.

Key modules and scripts documented in the report include:

```text
src/
├── grid.py
├── initialization.py
├── boundary_conditions.py
├── diagnostics.py
└── solvers/
    └── direct.py

Studies/
├── F04_galerkin_rom/
│   ├── f01_operators.py
│   ├── f02_reduced_operators.py
│   ├── f03_solve_reduced.py
│   ├── f04_rom_for_case.py
│   └── f05_test_rom.py
└── F05_refinement/
    ├── f00_generate_large_training_cases.py
    ├── f01_build_pod_basis.py
    ├── f02_generate_large_test_cases.py
    ├── f03_mode_count_study.py
    ├── f04_generate_augmented_training_set.py
    ├── f05_build_augmented_pod_basis.py
    ├── f06_generate_independent_test_set.py
    └── f07_compare_original_and_augmented_bases.py
```

The full-order solver is implemented in `src/solvers/direct.py`. The Galerkin scripts construct and project the discrete operators, assemble and solve the reduced system, and reconstruct the temperature field. The refinement scripts generate the training and LHS datasets, perform the mode-count study, select hotspots, build the augmented basis, and run the paired comparison.

> **Repository check:** The technical report uses the directory name `src/` and describes the study scripts under `Studies/`. Confirm exact directory capitalization and file names against the current repository before using this tree as an executable path guide.

## Workflow

The computational sequence documented in the report is:

1. Generate the 100-case full-order training grid and solve the FOM.
2. Build the original snapshot matrix and POD basis.
3. Construct the discrete Galerkin operators and evaluate the ROM.
4. Generate the first 100-case LHS screening dataset and run the mode-count study.
5. Rank high-error parameter combinations, select hotspots, and generate additional FOM snapshots.
6. Combine the original and additional snapshots and build the augmented POD basis.
7. Generate the final 100-case LHS dataset and corresponding FOM reference solutions.
8. Evaluate both POD bases on the same test cases and compute aggregate and paired error statistics.

The report identifies Python as the implementation language. Exact dependency versions and a pinned environment specification are not provided in the report; consult the repository's environment or requirements files for installation instructions.

## Limitations and next steps

- The velocity field is prescribed and fixed. Flow–temperature coupling and changes in the flow field are not modeled.
- The parameter study fixes `c = 1` and varies only `μ` and `Br` over the stated ranges.
- All simulations use a fixed `50 × 20` grid, and the ROM predicts steady-state solutions only.
- Enrichment is guided by the first LHS screening dataset and does not improve accuracy uniformly across mode counts or parameter combinations.
- The final test-set generation described in the report excludes original and augmented POD-training parameter combinations. The report also flags that overlap with the initial LHS screening set should be checked explicitly. Until that check is completed, independence from all development data should not be claimed.
- The current implementation does not fully exploit an offline–online decomposition. Precomputing and reusing reduced operators is a planned improvement.
- A larger independent test set and alternative hotspot-selection and enrichment strategies would help assess the robustness of the observed improvements.

## Technical report

The accompanying technical report, *Parameterized Two-Dimensional Steady-State Energy Equation: Full-Order Numerical Modeling, Discrete Galerkin–POD Reduction, and Error-Guided Snapshot Enrichment*, contains the full mathematical derivation, numerical implementation details, error statistics, plots, and supplementary parameter-space error maps.
