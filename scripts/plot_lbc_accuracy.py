"""Plot the LBC test accuracy curve(s) as a function of the guessed critical
parameter (e.g. inverse temperature beta or inverse coupling 1/lambda).

The script reads one or more accs CSV files produced by `lbc_train.py`
(each containing one accuracy per guessed critical value) and plots the
W-shape that the LBC method uses to identify the true critical value.

Examples
--------
Single accs file with an explicit sweep grid:

    python plot_lbc_accuracy.py \
        data/accs_w1.00_a1.0000_L12_seed1000-4.csv \
        --grid 0.15 1.10 \
        --xlabel beta --out figures/figure6d_lbc_accuracy.pdf

Average over many accs files (different seeds), explicit grid:

    python plot_lbc_accuracy.py \
        data/accs_w1.00_a1.0000_L12_seed*-4.csv \
        --grid 0.15 1.10 --xlabel beta

Use a custom grid CSV (one column of guess values, one row per accs entry):

    python plot_lbc_accuracy.py data/accs_*.csv --grid_file my_grid.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting_style import apply_paper_style, fig_size  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("accs", nargs="+", type=Path,
                   help="One or more accs CSV files. Curves are averaged.")
    grid = p.add_mutually_exclusive_group(required=True)
    grid.add_argument("--grid", nargs=2, type=float, metavar=("MIN", "MAX"),
                      help="Linear grid endpoints; length is inferred from the accs file.")
    grid.add_argument("--grid_file", type=Path,
                      help="CSV file with guess values, one per accs entry.")
    p.add_argument("--xlabel", default="beta",
                   help="x-axis label, e.g. 'beta', '1/lambda' (default: beta).")
    p.add_argument("--out", type=Path, default=None, help="Output PDF path.")
    return p.parse_args()


def load_curves(paths):
    arrs = [np.loadtxt(p) for p in paths]
    n = arrs[0].size
    if not all(a.size == n for a in arrs):
        raise ValueError(
            "All accs files must have the same length; got "
            + ", ".join(f"{p.name}: {a.size}" for p, a in zip(paths, arrs))
        )
    return np.stack(arrs, axis=0)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    curves = load_curves(args.accs)
    n = curves.shape[1]

    if args.grid_file is not None:
        grid = np.loadtxt(args.grid_file)
        if grid.size != n:
            raise ValueError(f"grid_file has {grid.size} entries, accs files have {n}")
    else:
        grid = np.linspace(args.grid[0], args.grid[1], n)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    label_map = {
        "beta": r"$\beta$ guess",
        "1/lambda": r"$1/\lambda_c$",
        "lambda": r"$\lambda_c$",
    }
    xlabel = label_map.get(args.xlabel, args.xlabel)

    fig, ax = plt.subplots(figsize=fig_size(85, 0.6))
    if curves.shape[0] > 1:
        ax.fill_between(grid, mean - std, mean + std, alpha=0.25, color="C0", linewidth=0)
    ax.plot(grid, mean, "-o", color="C0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")

    # The LBC W-shape has the true critical value at the high-accuracy peak
    # between the two interior minima.  Find the two interior minima first,
    # then take the highest-accuracy point strictly between them.
    interior = mean.copy()
    interior[0] = interior[-1] = np.inf
    imin1 = int(np.argmin(interior))
    masked = interior.copy()
    lo = max(imin1 - 2, 1)
    hi = min(imin1 + 2, len(masked) - 1) + 1
    masked[lo:hi] = np.inf
    imin2 = int(np.argmin(masked))
    lo, hi = sorted((imin1, imin2))
    if hi - lo >= 2:
        peak_idx_local = int(np.argmax(mean[lo + 1:hi]))
        ipeak = lo + 1 + peak_idx_local
    else:
        ipeak = int(np.argmax(mean[1:-1])) + 1
    ax.axvline(grid[ipeak], color="0.4", linestyle="--", linewidth=0.6,
               label=fr"peak: {args.xlabel}={grid[ipeak]:.3f}, acc={mean[ipeak]:.3f}")
    ax.legend()

    out = args.out
    if out is None:
        out = Path("figures") / "lbc_accuracy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  averaged {curves.shape[0]} curve(s); high-accuracy peak between "
          f"W minima at {args.xlabel}={grid[ipeak]:.4f} (acc={mean[ipeak]:.4f})")


if __name__ == "__main__":
    main()
