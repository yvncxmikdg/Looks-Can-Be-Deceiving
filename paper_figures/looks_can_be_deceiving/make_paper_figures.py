"""Stage 3: figure_stats.json -> figures/*.{pdf,png}.

Pure rendering. Every number drawn here was computed by compute_figure_stats.py and read out of
figure_stats.json -- this module derives nothing, so re-rendering can change how a figure looks but
never what it claims. If a value needs to change, it changes in the stats stage.

Figure numbers are the PAPER's:

  fig2  disagreement rate per satellite damage class, training split vs test split (no model)
  fig3  every architecture alone and averaged with the Class Oracle, with 95% intervals
  fig4  ROC: best native arm vs the Class Oracle, plus the best averaged score and chance
  fig5  AUC vs the top-1 accuracy of a simulated satellite damage-class detector
  fig6  per-class AUC for the whole fleet
  fig7  (supplemental) truth-conditioned predicted-score distributions for one arm
  fig8  (supplemental) satellite x sUAS pairwise AUC against each row's agreement cell

Usage:
    python make_paper_figures.py          # --data_dir / --out_dir default beside this script
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from paper_style import (
    INK, INK_SECONDARY, GRID, PRETRAINED_COLOR, SCRATCH_COLOR, COMBO_COLOR, COL_W,
    DAMAGE_LABEL, DAMAGE_LABEL_WRAPPED, DAMAGE_LABEL_TIGHT, DAMAGE_COLOR,
    TRAIN_FILL, TEST_FILL, WIN_COLOR, LOSS_COLOR, UNCHANGED_COLOR, CHANGED_COLOR, apply_rc,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png", flush=True)


def _nan(v):
    """Stats JSON stores an undefined metric as null; plotting wants NaN."""
    return np.nan if v is None else v


def _matrix_axes(ax, classes, title):
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([DAMAGE_LABEL_WRAPPED[c] for c in classes], fontsize=8)
    ax.set_yticklabels([DAMAGE_LABEL_WRAPPED[c] for c in classes], fontsize=8)
    ax.set_xlabel("sUAS Damage Class")
    ax.set_ylabel("Satellite Damage Class")
    # The title carries a parenthetical N/cell-count line; give it its own line rather than wrapping
    # mid-sentence, and enough pad that two lines don't collide with the top row of the matrix.
    ax.set_title(title, fontsize=10.5, loc="center", pad=20)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


# ---- fig 2 -------------------------------------------------------------------------------------
def draw_fig2(st, out_dir):
    """Disagreement rate per satellite damage class, training split vs test split."""
    f = st["fig2"]
    classes = f["classes"]
    # Double column, deliberately short: the tallest bar is ~0.64, so a 0-1 axis would spend a third of
    # the panel height on empty space above the data.
    fig, ax = plt.subplots(figsize=(COL_W * 2, 2.8))
    w = 0.38
    for i, c in enumerate(classes):
        for off, (rows, fill, lab) in enumerate([
                (f["train"], TRAIN_FILL, "training events (Ian + Harvey)"),
                (f["test"], TEST_FILL, "test event (Michael)")]):
            rec = rows[i]
            x = i + (off - 0.5) * w
            ax.bar(x, rec["p"], width=w * 0.92, color=fill, edgecolor="none", zorder=2,
                   label=lab if i == 0 else None)
            ax.text(x, rec["p"] + 0.015, f"{rec['p']:.2f}", ha="center", va="bottom",
                    fontsize=7.6, color=INK)
            # "destroyed" bars are short enough (~0.05-0.10) that a vertical N= label runs past the
            # bar top and collides with the value label above it; lay it flat instead. Every other
            # class's bars are tall enough for the vertical label to fit inside cleanly.
            ax.text(x, 0.012, f"N={rec['n']}", ha="center", va="bottom", fontsize=6.6,
                    color="#ffffff" if off else INK,
                    rotation=0 if c == "destroyed" else 90, zorder=5)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([DAMAGE_LABEL[c] for c in classes], fontsize=8.4)
    for tick, c in zip(ax.get_xticklabels(), classes):
        tick.set_color(DAMAGE_COLOR[c])
    ax.set_ylabel("P(sUAS and satellite disagree)", fontsize=8.5)
    ax.set_xlabel("Satellite-Derived Damage Class", fontsize=8.5)
    # 0.7 rather than 1.0. Two jobs: it drops the empty band above the tallest bar, and it stretches
    # every bar ~14% taller in absolute terms, which is what keeps the in-bar "N=" label from
    # overflowing the shortest bar.
    ax.set_ylim(0, 0.7)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right", handlelength=1.1, borderpad=0.2)
    ax.set_title("Disagreement Rate Per Satellite Damage Class, by Dataset Split\n"
                 f"($N_{{train}}$ = {f['n_train']:,}, $N_{{test}}$ = {f['n_test']:,})",
                 fontsize=9.5, loc="center", pad=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_disagreement_rate_by_split")


# ---- fig 3 -------------------------------------------------------------------------------------
def draw_fig3(st, out_dir):
    """Every architecture alone and averaged with the Class Oracle, on one row each.

    Dark = the model's own AUC, light = the same model averaged with the Class Oracle. Both bars are
    anchored at the 0.5 null and the dark one is drawn ON TOP of the light one rather than stacked
    end-to-end. True stacking is undefined here: averaging pulls a sub-chance arm back TOWARD 0.5 and
    pushes others across it, so a segment measured from the native value can point either direction.
    Overlaying reads correctly in both cases -- light visible beyond dark means the average gained,
    dark visible beyond light means it lost.

    Both values carry a 95% building-clustered bootstrap interval: black for the model alone, grey for
    the average. The averaged intervals are the ones that reach the Class Oracle rule, so that rule is
    drawn BLACK rather than grey -- a grey interval landing on a grey dashed rule read as one object.
    """
    series = st["fig3"]["series"]
    ceiling = st["class_oracle"]["auc_global"]

    fig, ax = plt.subplots(figsize=(COL_W, 4.9))
    data_lo = min(min(r["native_lo"], r["combined_lo"]) for r in series)
    data_hi = max(max(r["native_hi"], r["combined_hi"]) for r in series)

    def whisker(x_lo, x_hi, row, color):
        ax.plot([x_lo, x_hi], [row, row], color=color, lw=1.1, zorder=5, solid_capstyle="butt")
        ax.plot([x_lo, x_lo], [row - .15, row + .15], color=color, lw=1.1, zorder=5)
        ax.plot([x_hi, x_hi], [row - .15, row + .15], color=color, lw=1.1, zorder=5)

    for i, r in enumerate(series):
        native, combined = r["native"], r["combined"]
        # Draw the LONGER bar behind so both stay visible. Normally the average is longer and the
        # native bar sits on top of it; where averaging pulls a value back toward the null the native
        # bar is longer and would hide the average entirely, so the order flips and a sliver of light
        # shows at the base. The 0.01 margin keeps near-equal rows drawing dark on top, since "the
        # model alone reaches this" is the honest reading of such a row.
        if abs(combined - 0.5) < abs(native - 0.5) - 0.01:
            first, first_c, second, second_c = native, PRETRAINED_COLOR, combined, SCRATCH_COLOR
        else:
            first, first_c, second, second_c = combined, SCRATCH_COLOR, native, PRETRAINED_COLOR
        ax.barh(i, first - 0.5, left=0.5, height=0.66, color=first_c, edgecolor="none", zorder=2)
        ax.barh(i, second - 0.5, left=0.5, height=0.66, color=second_c, edgecolor="none", zorder=3)
        whisker(r["combined_lo"], r["combined_hi"], i, INK_SECONDARY)
        whisker(r["native_lo"], r["native_hi"], i, INK)
        # The label goes on whichever side of the row is free and clears every drawn element on that
        # row -- both bars AND both intervals -- rather than hugging the 0.5 null: some rows run their
        # native bar left of the null and their average right of it, so a label pinned beside the null
        # lands on the native bar. The white backing stops the Class Oracle rule striking through the
        # one label long enough to reach it.
        row_lo = min(r["native_lo"], r["combined_lo"], native, combined, 0.5)
        row_hi = max(r["native_hi"], r["combined_hi"], native, combined, 0.5)
        text = f"{native:.3f} / {combined:.3f}"
        if row_hi - 0.5 >= 0.5 - row_lo:
            x, ha = row_lo - 0.004, "right"
        else:
            x, ha = row_hi + 0.004, "left"
        ax.text(x, i, text, va="center", ha=ha, fontsize=6.0, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85))

    ax.axvline(0.5, color=INK, lw=1.0, zorder=4)
    ax.axvline(ceiling, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=4)

    ax.set_yticks(np.arange(len(series)))
    ax.set_yticklabels([r["pretty"] for r in series], fontsize=6.2)
    ax.set_xlabel("AUC", fontsize=8)
    ax.tick_params(labelsize=6.8)
    # Asymmetric padding: a left-placed label is right-aligned at the row's lower bound and grows
    # leftward, so the left margin has to hold the widest "0.000 / 0.000" string or it clips on the
    # spine. The right side only ever holds a whisker cap plus its own label.
    span = data_hi - data_lo
    ax.set_xlim(min(data_lo, 0.5) - 0.20 * span, max(data_hi, ceiling) + 0.10 * span)
    ax.set_ylim(-0.7, len(series) - 0.15)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    # The oracle rule is named here rather than annotated on the plot: no in-axes placement is free.
    # At the bottom a label lands on the sub-chance rows' value labels (which sit right of the null),
    # and at the top it lies across the two longest averaged bars.
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=PRETRAINED_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=SCRATCH_COLOR),
        plt.Line2D([0], [0], color=INK, lw=1.0, ls=(0, (5, 3))),
    ], labels=["Model alone", "Averaged with Class Oracle", f"Class Oracle ({ceiling:.3f})"],
        frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
        handlelength=1.4, borderpad=0.2, columnspacing=1.2)
    ax.set_title("AUC Per Architecture, Alone and Averaged With the Class Oracle\n"
                 f"(95% Building-Clustered Bootstrap, $N_{{test}}$ = {st['population']['n']:,})",
                 fontsize=7.6, loc="center", pad=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_fleet_auc")


# ---- fig 4 -------------------------------------------------------------------------------------
def draw_fig4(st, out_dir):
    """ROC: the best model ON ITS OWN against the Class Oracle, plus the averaged reference.

    Four curves only. The best native model carries the win/loss shading against the Class Oracle,
    because that comparison is the figure's argument -- a standalone model, given only satellite
    imagery, outruns the label-derived oracle; the best averaged score is drawn plain, as the "what
    does adding the class table buy" reference. AUC is a scalar over the whole curve, but under a
    flight budget only the top-left matters, which is what the shaded bands localize.

    Shading runs in BOTH directions -- green where the model leads the oracle, red where the oracle
    leads. Showing only the favourable half would misrepresent the comparison: the oracle's advantage
    comes almost entirely from one steep segment where it dumps `destroyed` to the bottom.
    """
    f = st["fig4"]
    grid = np.array(f["grid"])
    m_i = np.array(f["native_interp"])
    o_i = np.array(f["oracle_interp"])

    fig, ax = plt.subplots(figsize=(COL_W, 3.6))
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, lw=1.0, ls=(0, (4, 3)), zorder=2, label="chance")

    for band in f["bands"]:
        lo, hi, area = band["lo"], band["hi"], band["area"]
        sel = (grid >= lo) & (grid <= hi)
        color = WIN_COLOR if band["sign"] > 0 else LOSS_COLOR
        ax.fill_between(grid[sel], o_i[sel], m_i[sel], color=color, alpha=0.30, lw=0, zorder=1)
        # Every band surviving the width filter is labelled, including the ones the oracle wins. The
        # label sits INSIDE its band with no connector -- a vertical connector terminating at a
        # centred label would draw a line through the text -- and a thin band pushes its label clear
        # rather than masking both curves with its own background box.
        if abs(area) >= 0.004:
            xm = float(np.clip(0.5 * (lo + hi), 0.13, 0.87))
            upper = float(np.interp(xm, grid, np.maximum(m_i, o_i)))
            lower = float(np.interp(xm, grid, np.minimum(m_i, o_i)))
            if upper - lower >= 0.10:
                y_text, va = 0.5 * (upper + lower), "center"
            elif 0.5 * (upper + lower) > 0.5:
                y_text, va = lower - 0.02, "top"
            else:
                y_text, va = upper + 0.02, "bottom"
            ax.text(xm, float(np.clip(y_text, 0.02, 0.98)), f"{area:+.3f} area",
                    ha="center", va=va, fontsize=5.6, color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82))

    ax.plot(f["oracle"]["fpr"], f["oracle"]["tpr"], color=INK, lw=1.8, zorder=5,
            label=f"Satellite Class Oracle ({f['oracle']['auc']:.3f})")
    ax.plot(f["native"]["fpr"], f["native"]["tpr"], color=PRETRAINED_COLOR, lw=2.5, zorder=6,
            label=f"{f['native']['label']} ({f['native']['auc']:.3f})")
    ax.plot(f["combo"]["fpr"], f["combo"]["tpr"], color=COMBO_COLOR, lw=1.6, zorder=4,
            label=f"{f['combo']['label']} ({f['combo']['auc']:.3f})")

    shade_legend = ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=WIN_COLOR, alpha=0.30),
         plt.Rectangle((0, 0), 1, 1, color=LOSS_COLOR, alpha=0.30)],
        [f"{f['native']['label']} ahead", "Oracle ahead"], frameon=False, fontsize=5.8,
        loc="upper left", bbox_to_anchor=(0.02, 0.99), handlelength=1.0, borderpad=0.2,
        labelspacing=0.35)
    ax.add_artist(shade_legend)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate", fontsize=8)
    ax.set_ylabel("True Positive Rate", fontsize=8)
    ax.tick_params(labelsize=6.8)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=4.8, loc="lower right", bbox_to_anchor=(1.0, 0.0),
                    handlelength=1.2, markerfirst=False, labelspacing=0.32, borderpad=0.2,
                    handletextpad=0.4)
    leg._legend_box.align = "right"  # pylint: disable=protected-access
    ax.set_title("Receiver Operating Characteristic Curve\n"
                 f"($N_{{test}}$ = {f['n']:,})", fontsize=8.5, loc="center", pad=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_roc_curves")


# ---- fig 5 -------------------------------------------------------------------------------------
def draw_fig5(st, out_dir):
    """AUC vs the top-1 accuracy of the satellite damage-class detector feeding the Class Oracle.

    Caption caveat: the simulated detector errs symmetrically around the truth. A real model biased
    toward the dominant `no damage` class would give a weaker class branch at the same nominal top-1,
    so treat the accuracy axis as a lower bound on the requirement.
    """
    f = st["fig5"]
    accs = np.array(f["accs"])
    om, osd = np.array(f["oracle_mean"]), np.array(f["oracle_sd"])
    cm, csd = np.array(f["combo_mean"]), np.array(f["combo_sd"])
    perfect = f["perfect_oracle"]

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    ax.axhline(perfect, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=5)
    # Kept short and hard left: the combined curve crosses this rule mid-sweep, so a longer label
    # runs straight into it.
    ax.text(0.012, perfect + 0.004, f"Perfect Class Oracle ({perfect:.3f})",
            fontsize=6.2, color=INK, va="bottom", ha="left")
    ax.axhline(0.5, color=INK_SECONDARY, lw=1.0, zorder=3)
    # Hard right: the sweep starts at accuracy 0, where the oracle curve's origin occupies the
    # left end of the chance rule.
    ax.text(0.988, 0.503, "chance", fontsize=6.2, color=INK_SECONDARY, va="bottom", ha="right")

    ax.fill_between(accs, cm - csd, cm + csd, color=COMBO_COLOR, alpha=0.18, lw=0, zorder=4)
    ax.plot(accs, cm, color=COMBO_COLOR, lw=1.8, zorder=7,
            label=f"Average(Class Oracle, {f['pretty']})")
    ax.fill_between(accs, om - osd, om + osd, color=PRETRAINED_COLOR, alpha=0.18, lw=0, zorder=4)
    ax.plot(accs, om, color=PRETRAINED_COLOR, lw=1.6, zorder=6, label="Class Oracle Alone")

    ax.set_xlabel("Simulated Satellite Damage-Class\nDetector Top-1 Accuracy", fontsize=8)
    ax.set_ylabel("AUC", fontsize=8)
    ax.tick_params(labelsize=6.8)
    ax.set_xlim(0.0, 1.0)
    # Plain data limits: with the legend outside the axes nothing needs reserved space, and the sweep
    # reaches acc 0 where the oracle dips just under the chance rule.
    ax.set_ylim(float((om - osd).min()) - 0.010, float((cm + csd).max()) + 0.010)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    # The legend sits outside the axes, below them: at one column the combined series' label spans
    # most of the axes width, so any in-plot placement lands on either the curves or the chance rule.
    leg = ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                    handlelength=1.5, markerfirst=False, labelspacing=0.4, borderpad=0.2)
    leg._legend_box.align = "right"  # pylint: disable=protected-access
    # The architecture is named in the legend, so it does not also belong in the title. The band
    # definition is included in the title as ±1 SD over N Monte-Carlo draws.
    ax.set_title(f"AUC vs. Damage-Class Detector Accuracy\n(±1 SD Over {f['n_draw']} Monte-Carlo Draws, $N_{{test}}$ = {f['n']:,})",
                 fontsize=8.5, loc="center", pad=12)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_detector_degradation")


# ---- fig 6 -------------------------------------------------------------------------------------
def draw_fig6(st, out_dir):
    """Per-class AUC for every arm, ordered by overall AUC.

    No colourbar: every cell already prints its value, so a colourbar would only restate the
    encoding, and at one column it costs more space than it earns. Define the scale in the caption.
    """
    f = st["fig6"]
    classes = f["classes"]
    cell = np.array([[_nan(v) for v in row] for row in f["cell"]], dtype=float)

    cmap = LinearSegmentedColormap.from_list("auc", ["#8c8a80", "#f2f1ec", PRETRAINED_COLOR])
    span = np.nanmax(np.abs(cell - 0.5))
    norm = TwoSlopeNorm(vmin=0.5 - span, vcenter=0.5, vmax=0.5 + span)

    fig, ax = plt.subplots(figsize=(COL_W, 4.3))
    ax.imshow(cell, cmap=cmap, norm=norm, aspect="auto")
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            v = cell[i, j]
            if not np.isfinite(v):
                continue
            shade = abs(v - 0.5) > 0.62 * span
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.4,
                    color="#ffffff" if shade else INK)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([f"{DAMAGE_LABEL_TIGHT[c]}\nN={n}"
                        for c, n in zip(classes, f["n_by_class"])], fontsize=5.6)
    for tick, c in zip(ax.get_xticklabels(), classes):
        tick.set_color(DAMAGE_COLOR[c])
    ax.set_yticks(np.arange(len(f["row_labels"])))
    ax.set_yticklabels(f["row_labels"], fontsize=6.0)
    ax.set_xlabel("Satellite-Derived Damage Class", fontsize=6.8, labelpad=5)
    ax.set_title("AUC Within Each Satellite Damage Class\n"
                 f"(Ordered by Overall AUC, $N_{{test}}$ = {f['n']:,})",
                 fontsize=7.6, loc="center", pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_auc_by_damage_class")


# ---- fig 7 (supplemental) ----------------------------------------------------------------------
def draw_fig7(st, out_dir):
    """Truth-conditioned predicted-score distributions.

    Motivates the metric choice from the MODEL side (fig2 does it from the data side): the two
    distributions are separated -- which is what AUC measures -- but for most arms both sit in a
    narrow band well below 0.5, so any fixed argmax threshold assigns everything to one class. The
    threshold annotation is data-driven because this does NOT hold for every arm.
    """
    f = st["fig7"]
    edges = np.array(f["bin_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for key, label, color in (("unchanged", "truly UNCHANGED", UNCHANGED_COLOR),
                              ("changed", "truly CHANGED", CHANGED_COLOR)):
        rec = f[key]
        ax.bar(centers, rec["percent"], width=width, color=color, alpha=0.62, zorder=2,
               edgecolor="none", label=f"{label}  (N={rec['n']:,})")
        ax.axvline(rec["mean"], color=color, lw=1.6, ls=(0, (4, 2)), zorder=4)

    ax.axvline(0.5, color=INK, lw=1.3, zorder=5)
    n_right = f["n_above_half"]
    if n_right == 0:
        # Vertical label hugging the threshold line: horizontal text would run off the right edge,
        # and when no score crosses 0.5 the region beside the threshold is free of data.
        ax.text(0.4965, ax.get_ylim()[1] * 0.98,
                "argmax threshold — every building falls left of it",
                fontsize=7.6, color=INK, va="top", ha="right", rotation=90)
    else:
        # With mass on both sides of 0.5, a vertical label at the threshold strikes through the
        # densest bars -- state the split horizontally in the free top-left corner.
        ax.text(0.02, 0.98, f"argmax threshold (black line):\n{f['n'] - n_right:,} of {f['n']:,} "
                            "buildings fall left of it",
                transform=ax.transAxes, fontsize=7.6, color=INK, va="top", ha="left")
    ax.set_xlim(f["score_min"] - 0.008, max(0.515, f["score_max"] + 0.01))
    ax.set_xlabel("Predicted P(CHANGED)")
    ax.set_ylabel("Percent of Class")
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    # Anchored well left of the threshold column so the rotated label above cannot clip the counts.
    ax.legend(frameon=False, fontsize=8.2, loc="upper right", bbox_to_anchor=(0.74, 1.0))
    ax.set_title(f"Score Separation Without Calibration: {f['pretty']}\n"
                 f"(Dashed Lines = Class Means, $N_{{test}}$={f['n']:,})",
                 fontsize=10.5, loc="center", pad=20)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_score_separation")


# ---- fig 8 (supplemental) ----------------------------------------------------------------------
def draw_fig8(st, out_dir):
    """Satellite x sUAS pairwise AUC against each row's agreement cell."""
    f = st["fig8"]
    classes = f["classes"]
    n = len(classes)
    matrix = np.array([[_nan(v) for v in row] for row in f["auc"]], dtype=float)
    n_pos = np.array(f["n_pos"], dtype=int)

    cmap = LinearSegmentedColormap.from_list("auc", ["#8c8a80", "#f2f1ec", PRETRAINED_COLOR])
    span = np.nanmax(np.abs(matrix - 0.5)) if np.isfinite(matrix).any() else 0.1
    norm = TwoSlopeNorm(vmin=0.5 - span, vcenter=0.5, vmax=0.5 + span)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f7f6f2",
                                           edgecolor=INK, lw=1.2, zorder=4))
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7,
                        color=INK_SECONDARY, zorder=5)
            elif not np.isfinite(matrix[i, j]):
                if n_pos[i, j]:
                    ax.text(j, i, f"N={n_pos[i, j]}", ha="center", va="center", fontsize=7,
                            color=INK_SECONDARY, style="italic")
            else:
                lo, hi = f["ci"][f"{i},{j}"]
                ax.text(j, i, f"{matrix[i, j]:.2f}\n[{lo:.2f},{hi:.2f}]\nN={n_pos[i, j]}",
                        ha="center", va="center", fontsize=6.8, color=INK)
    _matrix_axes(ax, classes,
                 f"AUC vs. the Agreement Cell, Per Satellite Class: {f['pretty']}\n"
                 f"($N_{{test}}$={f['n_positives_used']:,} Positives Across "
                 f"{f['cells_computed']} Cells)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("AUC vs the row's agreement cell", fontsize=8.4)
    cb.ax.tick_params(labelsize=7.6)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_transition_matrix")


# ---- orchestration -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"),
                    help="directory holding figure_stats.json (from compute_figure_stats.py)")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "figures"),
                    help="where the figures are written")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.data_dir, "figure_stats.json")
    with open(stats_path, encoding="utf-8") as handle:
        st = json.load(handle)
    apply_rc()

    pop = st["population"]
    print(f"stats = {stats_path}", flush=True)
    print(f"n={pop['n']}  positives={pop['positives']}  buildings={pop['buildings']}  "
          f"arms={len(st['arms'])}", flush=True)
    print(f"out   = {args.out_dir}", flush=True)

    draw_fig2(st, args.out_dir)
    draw_fig3(st, args.out_dir)
    draw_fig4(st, args.out_dir)
    draw_fig5(st, args.out_dir)
    draw_fig6(st, args.out_dir)
    draw_fig7(st, args.out_dir)
    draw_fig8(st, args.out_dir)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
