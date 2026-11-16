"""Shared chart style so every figure looks consistent."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = ["COMPLIANT", "NON_COMPLIANT", "NOT_ADDRESSED"]
PRETTY = {"COMPLIANT": "Compliant", "NON_COMPLIANT": "Non-compliant", "NOT_ADDRESSED": "Not addressed"}
COLORS = {"COMPLIANT": "#0ca30c", "NON_COMPLIANT": "#d03b3b", "NOT_ADDRESSED": "#898781"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})
