import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.io_paths import ensure_dir


def plot_prebeck_weight_matrix(
    W: np.ndarray,
    ns: int,
    ne: int = 100,
    nt: int = 125,
    title: str = "",
    out_dir: str | Path = "data/goal2/fig2_7_like",
    fname_stem: str = "W",
):
    if W.shape != (nt, nt):
        raise ValueError("W must be (125,125).")

    out_dir = ensure_dir(Path(out_dir))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(W, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("target node j")
    ax.set_ylabel("source node i")

    for x in [ns - 0.5, ne - 0.5]:
        ax.axvline(x, color="white", linewidth=1.5)
        ax.axhline(x, color="white", linewidth=1.5)

    ax.axhline(ne - 0.5, color="white", linewidth=2.5)
    ax.axvline(ne - 0.5, color="white", linewidth=2.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("weight w(i->j)")

    plt.tight_layout()

    png_path = out_dir / f"{fname_stem}.png"
    pdf_path = out_dir / f"{fname_stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path
