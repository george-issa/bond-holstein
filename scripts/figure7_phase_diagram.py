"""Reproduce Fig. 7 of the paper: phase diagram of the half-filled bond-Holstein
model in the (1/lambda, T) plane.

The script reads the published reference values from
`data/published_phase_diagram.csv` and overlays them on the canonical
phase-region shading.  Each row is one (method, 1/lambda, T) tuple.

Methods:
    DQMC_FSS              -- T_cdw from finite-size scaling of S(pi,pi)
    LBC, PCA, tSNE         -- T_cdw from the corresponding ML method
    LBC_crossover         -- bipolaron-liquid / metal crossover from LBC
    Broecker_crossover    -- crossover from the Broecker, Assaad, Trebst method
    site_Holstein         -- T_cdw of the site-Holstein model (previous work)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting_style import apply_paper_style, fig_size  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path,
                   default=REPO_ROOT / "data" / "published_phase_diagram.csv",
                   help="CSV with (method, inv_lambda, T) rows.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "figures" / "figure7_phase_diagram.pdf")
    return p.parse_args()


def shade_regions(ax):
    """Background shading: CDW (below FSS line), metal (right of crossover),
    bipolaron liquid (left of crossover, above FSS line)."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # FSS T_cdw curve, defining the upper edge of the CDW region
    fss_x = np.array([0.4, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 3.3, 4.0, 5.2])
    fss_y = np.array([2.05, 2.05, 2.20, 2.04, 1.78, 1.42, 1.15, 1.00, 0.79, 0.55])

    # Bipolaron-liquid / metal crossover line (averaged from LBC + Broecker points)
    cross_x = np.array([0.4, 1.4, 1.5, 1.7, 2.5, 3.0, 5.2])
    cross_y = np.array([5.2, 5.0, 4.0,  2.5, 2.22, ylim[1] * 0 + 2.20, 2.20])

    x = np.linspace(*xlim, 600)
    cdw_top = np.interp(x, fss_x, fss_y)
    cross_top = np.interp(x, cross_x, cross_y)

    # CDW: below FSS line
    ax.fill_between(x, ylim[0], cdw_top, color="#cfe7c7", alpha=0.75, linewidth=0)
    # Metal: above FSS line and to the right of the crossover line
    ax.fill_between(x, cdw_top, ylim[1], color="#ffd9b3", alpha=0.55, linewidth=0)
    # Bipolaron liquid: above the crossover line (overrides metal in that wedge)
    bp_floor = np.maximum(cross_top, cdw_top)
    ax.fill_between(x, bp_floor, ylim[1], color="#cfe7f5", alpha=0.85, linewidth=0)
    # Soft fade across the crossover so the boundary looks gradual
    for w, alpha in [(0.06, 0.18), (0.12, 0.10)]:
        ax.fill_between(x, np.maximum(cdw_top, cross_top - w),
                        np.maximum(cdw_top, cross_top + w),
                        color="#cfe7f5", alpha=alpha, linewidth=0)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=fig_size(85, 0.75))
    ax.set_xlim(0.4, 5.2)
    ax.set_ylim(0.0, 5.2)

    shade_regions(ax)

    style = {
        "DQMC_FSS":           dict(marker="o", color="#228B22", ms=3.0, label="DQMC, FSS"),
        "LBC":                dict(marker="s", mfc="white", mec="#B22222", color="#B22222",
                                   ms=4.0, ls="", label="LBC"),
        "PCA":                dict(marker="^", mfc="white", mec="#FFD700", color="#FFD700",
                                   ms=4.0, ls="", label="PCA"),
        "tSNE":               dict(marker="v", mfc="white", mec="#4682B4", color="#4682B4",
                                   ms=4.0, ls="", label="t-SNE"),
        "LBC_crossover":      dict(marker="s", color="#B22222", ms=4.0, ls="",
                                   label="crossover, LBC"),
        "Broecker_crossover": dict(marker="x", color="black", ms=4.0, ls="",
                                   label="crossover, Broecker et al."),
        "site_Holstein":      dict(marker="", color="#D2691E", lw=1.4,
                                   label=r"site Holstein $T_{\rm cdw}$ (prev. work)"),
    }

    # DQMC_FSS as a connected line, others as scatter
    for method, sty in style.items():
        sub = df[df["method"] == method].sort_values("inv_lambda")
        if sub.empty:
            continue
        if method == "DQMC_FSS":
            ax.plot(sub["inv_lambda"], sub["T"], "-", color=sty["color"], lw=1.6,
                    label=sty["label"])
            ax.plot(sub["inv_lambda"], sub["T"], "o", color=sty["color"], ms=3.0)
        elif method == "site_Holstein":
            ax.plot(sub["inv_lambda"], sub["T"], "-", color=sty["color"], lw=1.4,
                    label=sty["label"])
        else:
            ax.plot(sub["inv_lambda"], sub["T"], **sty)

    # Phase-region annotations
    ax.text(0.7, 1.0, "CDW", ha="center", va="center", fontsize=8)
    ax.text(3.5, 3.5, "metal", ha="center", va="center", fontsize=8)
    ax.text(0.85, 4.2, "bipolaron\nliquid", ha="center", va="center", fontsize=7.5)

    ax.set_xlabel(r"$1/\lambda$")
    ax.set_ylabel(r"Temperature [$t$]")
    ax.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.9,
              edgecolor="0.7")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
