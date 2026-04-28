"""Reproduce panels (a)-(c) of Fig. 6 (PCA and t-SNE on electron-density snapshots).

The script loads DQMC-generated electron-density snapshots collected over a
range of inverse temperatures beta at a fixed dimensionless coupling
(default 1/lambda = 2, i.e. alpha = 1.0), runs principal component analysis
and t-SNE on the snapshots, and writes:

    figures/figure6a_pca_scatter.pdf       -- P1 vs P2 colored by beta
    figures/figure6b_pca_average.pdf       -- |<P1>| and |<P2>| vs beta
    figures/figure6c_tsne_embedding.pdf    -- 2D t-SNE embedding colored by beta

Input data layout
-----------------
The expected CSV files are produced by the DQMC pipeline and live under
data/electron_densities_w<w>_a<alpha>_L<L>_csv-<sID>/.  Each combined CSV
has shape (N_samples, L*L + 1): the first L*L columns are the L x L
electron density n_i for one snapshot, and the last column is the
inverse temperature beta the snapshot was generated at.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting_style import apply_paper_style, fig_size  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--w", type=float, default=1.0, help="phonon frequency omega_0")
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help="electron-phonon coupling strength alpha (1.0 = 1/lambda_bond = 2)",
    )
    p.add_argument("--L", type=int, default=12, help="lattice linear size")
    p.add_argument("--sID", type=int, default=7, help="simulation ID (data subdir suffix)")
    p.add_argument("--perplexity", type=float, default=40.0, help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=0, help="t-SNE random seed")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def load_snapshots(data_dir: Path, w: float, alpha: float, L: int, sID: int):
    """Return (X, betas) for the combined CSV file at the given parameters.

    X has shape (N_samples, L*L); betas is the per-snapshot inverse temperature.
    """
    sub = data_dir / f"electron_densities_w{w:.2f}_a{alpha:.4f}_L{L}_csv-{sID}"
    csv = sub / f"electron_densities_w{w:.2f}_a{alpha:.4f}_L{L}-{sID}.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Expected combined CSV at {csv}.\n"
            f"Available subdirectories under {data_dir}:\n  "
            + "\n  ".join(sorted(p.name for p in data_dir.glob('electron_densities_*')))
        )

    arr = np.loadtxt(csv, delimiter=",")
    N = L * L
    X = arr[:, :N].astype(np.float64)
    betas = arr[:, -1].astype(np.float64)
    return X, betas


def run_pca(X: np.ndarray):
    """Center the data and return (P1, P2, eigvals) using a covariance eigendecomposition."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; flip to descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    P1 = Xc @ eigvecs[:, 0]
    P2 = Xc @ eigvecs[:, 1]
    return P1, P2, eigvals


def plot_pca_scatter(P1, P2, betas, out_path: Path):
    fig, ax = plt.subplots(figsize=fig_size(85, 0.85))
    sc = ax.scatter(P1, P2, c=betas, cmap="bwr_r", s=2, alpha=0.85, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\beta$", rotation=0)
    ax.set_xlabel(r"$P_1$")
    ax.set_ylabel(r"$P_2$")
    fig.savefig(out_path)
    plt.close(fig)


def plot_pca_average(P1, P2, betas, L, out_path: Path):
    """Mean |P_n| vs beta. P1 is rescaled by L so the curves share an axis.

    beta_c is taken to be the beta where |P_1|/L and 10|P_2| intersect, with
    sub-grid resolution from linear interpolation between adjacent points.
    """
    unique_betas = np.unique(betas)
    P1_avg = np.array([np.abs(P1[betas == b]).mean() for b in unique_betas])
    P2_avg = np.array([np.abs(P2[betas == b]).mean() for b in unique_betas])

    fig, ax = plt.subplots(figsize=fig_size(85, 0.7))
    ax.plot(unique_betas, P1_avg / L, "-^", label=fr"$|P_1|/{L}$")
    ax.plot(unique_betas, 10 * P2_avg, "-v", label=r"$10\,|P_2|$")
    ax.set_xlabel(r"$\beta$")

    diff = (10 * P2_avg) - (P1_avg / L)
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes):
        i = sign_changes[-1]
        x0, x1 = unique_betas[i], unique_betas[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        bc = x0 - d0 * (x1 - x0) / (d1 - d0) if d1 != d0 else x0
        ax.axvline(bc, color="0.4", linestyle="--", linewidth=0.6,
                   label=fr"$\beta_c \approx {bc:.2f}$")

    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tsne(X, betas, perplexity, seed, out_path: Path):
    """t-SNE of the snapshots, colored by beta."""
    model = TSNE(perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    emb = model.fit_transform(X)

    fig, ax = plt.subplots(figsize=fig_size(85, 0.85))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=betas, cmap="bwr_r",
                    s=4, alpha=0.85, linewidths=0.35, edgecolors="black")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\beta$", rotation=0)
    ax.set_xlabel(r"$1^{\mathrm{st}}$ embed. dim.")
    ax.set_ylabel(r"$2^{\mathrm{nd}}$ embed. dim.")
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading snapshots: w={args.w}, alpha={args.alpha}, L={args.L}, sID={args.sID}")
    X, betas = load_snapshots(args.data_dir, args.w, args.alpha, args.L, args.sID)
    n_betas = len(np.unique(betas))
    print(f"  X.shape = {X.shape}, distinct betas = {n_betas}, "
          f"beta in [{betas.min():.3f}, {betas.max():.3f}]")

    print("Running PCA...")
    P1, P2, eigvals = run_pca(X)
    print(f"  top-5 eigenvalues: {eigvals[:5]}")

    plot_pca_scatter(P1, P2, betas, args.out_dir / "figure6a_pca_scatter.pdf")
    plot_pca_average(P1, P2, betas, args.L, args.out_dir / "figure6b_pca_average.pdf")
    print(f"  wrote PCA panels to {args.out_dir}")

    print(f"Running t-SNE (perplexity={args.perplexity}, seed={args.seed})...")
    plot_tsne(X, betas, args.perplexity, args.seed,
              args.out_dir / "figure6c_tsne_embedding.pdf")
    print(f"  wrote t-SNE panel to {args.out_dir}")


if __name__ == "__main__":
    main()
