"""Voyager deliverable plot theme.

Import at the top of any notebook or script:

    from plot_style import apply_theme
    apply_theme()

Navy + antique brass, no gridlines, despined top/right axes,
tight figsize, 200dpi on save.
"""
import matplotlib.pyplot as plt
import seaborn as sns

VOYAGER_NAVY = "#142454"
VOYAGER_BRASS = "#C89B3C"
FILL = "#EBECF5"


def apply_theme():
    sns.set_theme(style="white", context="paper", font="sans-serif")
    sns.set_palette([VOYAGER_NAVY, VOYAGER_BRASS])
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
