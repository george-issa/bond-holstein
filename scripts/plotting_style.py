"""Shared matplotlib style for the bond-Holstein paper figures."""

import matplotlib as mpl
from cycler import cycler

paper_rc = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.family": "STIXGeneral",
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 5.0,
    "mathtext.fontset": "stix",

    "lines.linewidth": 0.8,
    "lines.markersize": 1.5,
    "errorbar.capsize": 2.0,

    "axes.linewidth": 0.8,
    "axes.labelpad": 2.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    "axes.grid": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,

    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.5,

    "figure.autolayout": False,
    "figure.constrained_layout.use": True,
}

pro_cycle = [
    "#6A5ACD",  # slate blue
    "#4682B4",  # steel blue
    "#D2691E",  # burnt orange
    "#228B22",  # forest green
    "#B22222",  # firebrick
    "#8B008B",  # dark magenta
    "#708090",  # slate gray
    "#FFD700",  # goldenrod
]


def apply_paper_style():
    mpl.rcParams.update(paper_rc)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=pro_cycle)


def fig_size(width_mm=85, ratio=0.62):
    inch = 1 / 25.4
    return (width_mm * inch, width_mm * inch * ratio)
