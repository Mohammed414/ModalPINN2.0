"""Minimal shared matplotlib style for R9 deliverable figures."""
import matplotlib as mpl


def apply():
    mpl.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 7,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 120,
        'savefig.dpi': 300,
    })


apply()
