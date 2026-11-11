from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.config import (
    CELL_COUNT_X,
    CELL_COUNT_Y,
    DOMAIN_LENGTH_Y,
)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "image.cmap": "plasma"
})


def visualize_pod_modes(retained_modes=3):

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    analysis_dir = Path(__file__).resolve().parent
    results_dir = analysis_dir / "results"

    modes_file = results_dir / "pod_modes.npy"
    output_file = results_dir / "pod_modes.png"

    # --------------------------------------------------
    # Load POD modes
    # --------------------------------------------------

    U = np.load(modes_file)

    nx = CELL_COUNT_X
    ny = CELL_COUNT_Y

    if U.shape[0] != nx * ny:
        raise ValueError(
            f"Mode size {U.shape[0]} does not match "
            f"the grid size {nx} x {ny}."
        )

    if retained_modes < 1:
        raise ValueError("retained_modes must be at least 1.")

    r = min(retained_modes, U.shape[1])

    # --------------------------------------------------
    # Grid coordinates (cell centres)
    # --------------------------------------------------

    x = (np.arange(nx) + 0.5) * (5.0 / nx)
    y = (np.arange(ny) + 0.5) * (DOMAIN_LENGTH_Y / ny)

    # --------------------------------------------------
    # Reshape selected modes into spatial fields
    # --------------------------------------------------

    modes = [
        U[:, i].reshape((nx, ny), order="C")
        for i in range(r)
    ]

    # --------------------------------------------------
    # Common symmetric colour scale
    # --------------------------------------------------

    max_abs = max(
        np.max(np.abs(mode))
        for mode in modes
    )

    if max_abs == 0:
        raise ValueError("The selected POD modes are all zero.")

    levels = np.linspace(-max_abs, max_abs, 101)

    norm = TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0.0,
        vmax=max_abs,
    )

    # --------------------------------------------------
    # Adaptive subplot layout
    # --------------------------------------------------

    layouts = {
        3: (3, 1),
        4: (2, 2),
        6: (3, 2),
        9: (3, 3),
    }

    if r in layouts:
        nrows, ncols = layouts[r]
    else:
        ncols = int(np.ceil(np.sqrt(r)))
        nrows = int(np.ceil(r / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10,10),
        squeeze=False,
        constrained_layout=True,
    )

    axes_flat = axes.ravel()

    # --------------------------------------------------
    # Plot POD modes
    # --------------------------------------------------

    for i, mode in enumerate(modes):

        ax = axes_flat[i]

        contour = ax.contourf(
            x, y, mode.T,
            levels=levels,
            cmap="viridis",
            norm=norm,
            extend="both",
        )

        ax.set_title(f"POD Mode {i + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("auto")

    # --------------------------------------------------
    # Hide unused subplot positions
    # --------------------------------------------------

    for i in range(r, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # --------------------------------------------------
    # Shared colourbar
    # --------------------------------------------------

    fig.colorbar(
        contour,
        ax=axes_flat[:r].tolist(),
        location="right",
        fraction=0.035,
        pad=0.08,
        label="Mode amplitude",
    )

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"POD modes figure saved to: {output_file}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    visualize_pod_modes(retained_modes=3)