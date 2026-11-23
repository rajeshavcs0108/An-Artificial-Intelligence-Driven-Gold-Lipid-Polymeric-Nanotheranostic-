"""Shared figure style: Okabe-Ito palette, 300 DPI PNG, RNG_SEED=42."""
import matplotlib as mpl
RNG_SEED = 42
DPI = 300
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })
