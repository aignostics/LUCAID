"""
Shared figure style for the LUCAID analysis figures
===================================================
Single source of truth for the look-and-feel of every figure in this repository:
the muted blue/orange "severity" palette, LUCAID branding, clean spines, and
PNG+PDF+SVG export. Import this instead of re-defining colours and rcParams in
each script so the whole figure set stays visually consistent.

Design language
---------------
* Hero colour is a calm blue (`MODEL`): LUCAID is always the blue series.
* Pathologists are a neutral grey (`PATHOLOGIST`).
* Category agreement uses a 3-step severity ramp:
      concordant (blue) -> off-by-1 (light orange) -> off-by->=2 (dark brown).
* Model points are circles, pathologist points are diamonds.
* Top/right spines are dropped; type is Arial-like, embedded as real text in
  PDF/SVG so a typesetter can restyle it.

Usage
-----
    import style as ps
    ps.apply_house_style()
    ...
    ax.scatter(x, y, color=ps.MODEL, marker=ps.MODEL_MARKER)
    ps.style_axes(ax)
    ps.save_figure(fig, output_dir / "my_figure")   # -> .png/.pdf/.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# Palette
# =============================================================================
# Severity ramp (category distance to the consensus).
CONCORDANT = "#2c6fb0"   # exact category match  (also the LUCAID/model hero blue)
OFF_BY_ONE = "#f3c08a"   # off by one category    (light orange)
OFF_BY_MANY = "#8a4b1a"  # off by >=2 categories  (dark brown)

# Discrete series colours.
MODEL = "#2c6fb0"        # LUCAID — same blue as "concordant" on purpose
MODEL_LIGHT = "#7ea8d4"  # lighter tint of the model blue (secondary model series)
PATHOLOGIST = "#9aa4ac"  # pathologist grey
ACCENT = "#e08214"       # orange accent (discordance / molecular ground truth)
NOT_SCORED = "#d9dde1"   # "not scored" light grey

# Neutral line work.
IDENTITY_LINE = "#b9c0c6"   # y = x reference line
THRESHOLD_LINE = "#c8ccd0"  # clinical cut-off guide lines

# Faint fills for highlighting a model row/column background.
MODEL_BG = "#eaf1f8"        # very faint tint of the model blue
NEUTRAL_BAR = "#c7ccd1"     # light grey for secondary summary bars

# Ordered ramps for convenience.
SEVERITY_COLORS: List[str] = [CONCORDANT, OFF_BY_ONE, OFF_BY_MANY]
SEVERITY_LABELS: List[str] = ["concordant", "off by 1", "off by >=2"]

# Extra model series (e.g. count- vs area-based cellularity) stay in the model's
# blue family so they read as "the model", leaving ACCENT free for highlights.
MODEL_SERIES: List[str] = [MODEL, "#7ea8d4", "#12395f"]

# =============================================================================
# Naming + marker conventions
# =============================================================================
MODEL_LABEL = "LUCAID"       # display name for the model everywhere
MODEL_MARKER = "o"           # circles for the model
PATHOLOGIST_MARKER = "D"     # diamonds for pathologists

# Substrings that identify a model column/annotator (case-insensitive).
_MODEL_TOKENS = ("model", "lucaid")


def is_model_series(name: str) -> bool:
    """True if an annotator/column name refers to the model rather than a human."""
    low = str(name).lower()
    return any(tok in low for tok in _MODEL_TOKENS)


def display_name(name: str) -> str:
    """Map an internal annotator name to its published label (model -> LUCAID)."""
    return MODEL_LABEL if is_model_series(name) else str(name)


def series_color(name: str) -> str:
    """House colour for an annotator: blue for the model, grey for pathologists."""
    return MODEL if is_model_series(name) else PATHOLOGIST


def series_marker(name: str) -> str:
    """House marker for an annotator: circle for the model, diamond otherwise."""
    return MODEL_MARKER if is_model_series(name) else PATHOLOGIST_MARKER


def severity_color(distance: int) -> str:
    """Colour for a category distance: 0 -> concordant, 1 -> off-by-1, else off-by-many."""
    if distance <= 0:
        return CONCORDANT
    if distance == 1:
        return OFF_BY_ONE
    return OFF_BY_MANY


# =============================================================================
# rcParams / global style
# =============================================================================
def apply_house_style() -> None:
    """Apply the shared rcParams. Call once, early, before creating figures."""
    plt.rcParams.update({
        # Arial-like stack; falls back cleanly to whatever the host has.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
        # Embed real text (not paths) so the figures stay editable downstream.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Clean, publication-ready axes.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.edgecolor": "#4a5663",
        "axes.linewidth": 0.9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_axes(ax: plt.Axes, *, grid: bool = False, top_right: bool = False) -> plt.Axes:
    """Drop the top/right spines (the house default) and optionally add a light grid."""
    ax.spines["top"].set_visible(top_right)
    ax.spines["right"].set_visible(top_right)
    if grid:
        ax.grid(True, color="#e6e9ec", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    return ax


# =============================================================================
# Export
# =============================================================================
def save_figure(fig: plt.Figure, path: Union[str, Path],
                formats: Iterable[str] = ("png", "pdf", "svg"),
                dpi: int = 300, close: bool = True,
                pad_inches: float = 0.08) -> List[Path]:
    """Save a figure to several vector/raster formats sharing one path stem.

    ``path`` may carry an extension (it is stripped); one file per entry in
    ``formats`` is written next to it. Returns the paths written.
    """
    stem = Path(path)
    if stem.suffix.lower().lstrip(".") in {"png", "pdf", "svg", "eps", "tif", "tiff", "jpg"}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        written.append(out)
    if close:
        plt.close(fig)
    return written


# =============================================================================
# Small shared annotations
# =============================================================================
def significance_stars(p: float) -> str:
    """APA-style significance stars for a p-value (``ns`` when not significant)."""
    if p is None or p != p:  # None or NaN
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def severity_legend_handles() -> List[Patch]:
    """Legend patches for the concordant / off-by-1 / off-by->=2 severity ramp."""
    return [Patch(color=c, label=lbl)
            for c, lbl in zip(SEVERITY_COLORS, SEVERITY_LABELS)]


def rater_legend_handles() -> List[Line2D]:
    """Legend handles distinguishing the LUCAID circle from the pathologist diamond."""
    return [
        Line2D([0], [0], marker=MODEL_MARKER, color="none",
               markerfacecolor=MODEL, markersize=9, label=MODEL_LABEL),
        Line2D([0], [0], marker=PATHOLOGIST_MARKER, color="none",
               markerfacecolor=PATHOLOGIST, markersize=8, label="Pathologist"),
    ]
