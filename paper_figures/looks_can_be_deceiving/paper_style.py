"""Shared visual style for the CHANGE paper figures.

Style only -- no data, no statistics, no class ordering. The set of damage classes a figure plots
comes from the stats JSON, which derives it from the model's channels YAML, so this module only maps
a class name to how it should look.

Every figure is authored at COL_W (a single paper column) so it is placed 1:1 and never downscaled;
scaling a wide figure into one column is what makes its text illegible.
"""
import matplotlib
matplotlib.use("Agg")
# pylint: disable=wrong-import-position
# The Agg backend must be selected before pyplot is imported, so this import cannot move up.
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# AUC gets its own olive family: dark = pretrained backbone, light = trained from scratch.
PRETRAINED_COLOR, SCRATCH_COLOR = "#5c6f2b", "#b3c17a"
# The "Class Oracle + Change Model Average" fuses the black oracle curve with an olive model curve,
# so it must own a hue belonging to neither. Deep teal stays legible against both the olive family
# and the green/red win-loss shading on the ROC figure.
COMBO_COLOR = "#0f6e7d"

# Single paper column, in inches.
COL_W = 3.5

DAMAGE_LABEL = {"no damage": "No Damage", "minor damage": "Minor Damage",
                "major damage": "Major Damage", "destroyed": "Destroyed",
                "un-classified": "Un-classified"}
# Line-broken variant for matrix ticks, where "Minor Damage"/"Major Damage" on one line collided
# with their neighbours; the other three fit on one line without breaking.
DAMAGE_LABEL_WRAPPED = {"no damage": "No Damage", "minor damage": "Minor\nDamage",
                        "major damage": "Major\nDamage", "destroyed": "Destroyed",
                        "un-classified": "Un-classified"}
# Tightest variant, for the column-width heatmap: five classes share ~2.2in there, so even the
# two-line names run into their neighbours. Severity is still unambiguous from these.
DAMAGE_LABEL_TIGHT = {"no damage": "None", "minor damage": "Minor", "major damage": "Major",
                      "destroyed": "Destr.", "un-classified": "Unclass."}
DAMAGE_COLOR = {"no damage": "#3fa34d", "minor damage": "#f2c230", "major damage": "#ec7014",
                "destroyed": "#c1272d", "un-classified": "#7b52ab"}

# The train-vs-test data-characterisation figure is deliberately neutral so it does not read as a
# model result.
TRAIN_FILL, TEST_FILL = "#b8b5a8", "#4a4a45"

# ROC shading: model ahead of the oracle / behind it.
WIN_COLOR, LOSS_COLOR = "#2e9e4f", "#c1272d"

# Truth-conditioned score histogram.
UNCHANGED_COLOR, CHANGED_COLOR = "#2a78d6", "#eb6834"


def apply_rc():
    """Install the shared rcParams. Called by the render entry point."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    })
