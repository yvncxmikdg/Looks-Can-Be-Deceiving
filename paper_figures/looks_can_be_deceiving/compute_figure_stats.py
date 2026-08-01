"""Stage 2: figure_data.json -> figure_stats.json.

This module owns every number the paper reports. It computes the AUCs, the bootstrap confidence
intervals, the rate tables, the ROC curves and the detector sweep, then writes them all to one JSON.
make_paper_figures.py consumes that JSON and does nothing but draw.

The split exists so a figure can be re-rendered without re-deriving its statistics, and so the
numbers quoted in the text come from a stored artifact rather than from stdout of a plotting script.
Any value that appears in a figure or in the prose should be findable in figure_stats.json.

Figure numbers here are the paper's numbers:

  fig2  disagreement rate per satellite damage class, training split vs test split (no model)
  fig3  every architecture alone and averaged with the Class Oracle, with 95% intervals
  fig4  ROC: best native arm vs the Class Oracle, plus the best averaged score and chance
  fig5  AUC vs the top-1 accuracy of a simulated satellite damage-class detector
  fig6  per-class AUC for the whole fleet
  fig7  (supplemental) truth-conditioned predicted-score distributions for one arm
  fig8  (supplemental) satellite x sUAS pairwise AUC against each row's agreement cell

The damage taxonomy and its severity order are read from the channels YAML, not restated here, and
every statistical knob is a CLI flag. No path has a default that points outside this directory.

Usage:
    python compute_figure_stats.py \
        --channels_parameters_file <repo>/src/modeling/data_envs/Tasks/BDA/Channels/rgb_mask_no_obscured.yaml
"""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=wrong-import-position
# ortho_loader puts <repo>/src on sys.path as an import side effect, so it must be imported after
# HERE is defined -- it cannot move to the top.
from ortho_loader import add_taxonomy_arg, damage_classes  # noqa: E402
# The AUC primitives -- including the rank-average combiner behind the "averaged" models -- are the
# evaluator's, not the figures'. Importing them is what keeps a number in a figure identical to the
# same number in an evaluation file instead of merely intended to match.
from modeling.evaluate_CHANGE import (  # noqa: E402
    global_auc as auc,
    within_ortho_auc,
    rank_average,
    building_clustered_bootstrap_ci,
)

# Defaults for the statistical knobs; every one is overridable from the CLI. They are module-level
# only so the default appears once in --help rather than being repeated in each function signature.
DEFAULT_N_BOOT = 2000
DEFAULT_MIN_CELL = 20
# Every consumer builds its OWN RandomState from the seed rather than drawing from a shared stream, so
# no published number depends on how many others were computed first. Drawing from one shared stream
# would make every interval a function of figure ordering.
DEFAULT_SEED = 7


@dataclass(frozen=True)
class Config:
    """Statistical knobs plus the taxonomy read from the channels YAML.

    Passed explicitly rather than read from module globals so a caller can see -- and change -- every
    value that moves a published number.
    """
    classes: tuple          # satellite damage classes in severity order
    n_boot: int = DEFAULT_N_BOOT
    min_cell: int = DEFAULT_MIN_CELL
    seed: int = DEFAULT_SEED

    def severity_rank(self, damage_class):
        """Position in the severity ordering, or None for a class outside it.

        The last class in the channel map's index order is the catch-all ("un-classified"): it is not
        a severity, so the detector-confusion model treats it as off-scale rather than adjacent to
        the most severe real class.
        """
        idx = self.classes.index(damage_class)
        return None if idx == len(self.classes) - 1 else idx


# The damage taxonomy is NOT defined here -- it is read from the channels YAML via
# ortho_loader.damage_classes(), which returns the output classes in model-index (severity) order.
# "obscured" never appears in it, which is what we want: on this task the satellite annotator could
# not assess the building at all, so any drone label disagrees by construction and it is
# definitionally CHANGED -- leaving it in hands every ranker a free perfectly-ranked class.


def boot_ci(metric_func, building_ids, cfg):
    """cfg-carrying wrapper over the evaluator's building-clustered bootstrap.

    The shared primitive takes its knobs explicitly (so the evaluator can pass CLI values); the
    figures carry the same knobs on cfg. Returns (lo, hi) as the figure code expects.
    """
    return building_clustered_bootstrap_ci(metric_func, building_ids,
                                           n_iterations=cfg.n_boot, seed=cfg.seed)


def _f(v):
    """numpy scalar -> JSON-safe float, with NaN as None (bare NaN is not valid JSON)."""
    return None if v is None or not np.isfinite(v) else float(v)


# ---- population --------------------------------------------------------------------------------
def load(data_dir, cfg):
    """Load figure_data.json and derive the single analysis population every figure shares."""
    with open(os.path.join(data_dir, "figure_data.json"), encoding="utf-8") as handle:
        d = json.load(handle)
    # Restrict the POPULATION to the taxonomy, not just the plotted classes. Doing it here rather than
    # per figure keeps one population across every figure and the rate-table ceiling; leaving it to a
    # per-figure filter lets the fleet figure and the per-class figures score different populations,
    # which moves the ceiling by ~0.002. In practice this drops satellite-"obscured": the annotator
    # could not assess the building, so any drone label disagrees by construction.
    n_before = len(d["rows"])
    rows = [r for r in d["rows"] if r["sat_class"] in set(cfg.classes)]
    dropped = n_before - len(rows)
    if dropped:
        print(f"dropped {dropped} buildings outside the taxonomy ({n_before} -> {len(rows)})",
              flush=True)

    y = np.array([r["y"] for r in rows])
    orthos = sorted({r["ortho"] for r in rows})
    code = {o: i for i, o in enumerate(orthos)}
    g = np.array([code[r["ortho"]] for r in rows])          # integer codes; see within_ortho_auc
    sat = np.array([r["sat_class"] for r in rows])
    drone = np.array([r["drone_class"] for r in rows])
    # Building id = the hash suffix of the row key ("__<ortho>_<hash>"). The same hash under two
    # satellite views (or two drone scenes) is the same physical building; the bootstrap clusters on it.
    bid = np.array([r["key"].rsplit("_", 1)[-1] for r in rows])
    scores = {a: np.array([r["scores"][a] for r in rows]) for a in d["arms"]}
    # Flatten arms.yaml into the lookups the stats functions expect.
    d["pretty"] = {a: m["pretty"] for a, m in d["arms"].items()}
    d["pretrained"] = {a: m["pretrained"] for a, m in d["arms"].items()}
    d["arms"] = sorted(d["arms"])
    return d, y, g, sat, drone, bid, scores, len(orthos), dropped


def rate_table(class_rates, subset, cfg=None):
    """P(CHANGED | satellite class) for one stripe, as (rate, changed, total) dicts.

    Reads the counts stage 1 emitted. Passing `cfg` restricts the table to the taxonomy's classes,
    which drops "obscured" -- definitionally CHANGED, so leaving it in gives the oracle a free
    perfectly-ranked class. Omit `cfg` for the data-characterisation figure, which reports the
    annotations as collected.
    """
    counts = class_rates[subset]
    keep = set(cfg.classes) if cfg is not None else set(counts)
    changed = {k: v["changed"] for k, v in counts.items() if k in keep}
    total = {k: v["total"] for k, v in counts.items() if k in keep}
    return {k: changed[k] / t for k, t in total.items()}, changed, total


# ---- fig 2: disagreement rate by split ---------------------------------------------------------
def stats_fig2(class_rates, cfg):
    """Training-split vs test-split disagreement rate per satellite damage class. No model.

    "Training events" (Ian + Harvey) and "test event" (Michael) name what each split contains, but
    the comparison is TRAIN vs TEST -- not one individual event against another. Obscured is kept
    here (unlike the model figures) because this figure characterises the annotations as collected.
    """
    _, tr_num, tr_den = rate_table(class_rates, "train")
    _, te_num, te_den = rate_table(class_rates, "test")
    classes = [c for c in cfg.classes
               if tr_den.get(c, 0) >= cfg.min_cell and te_den.get(c, 0) >= cfg.min_cell]
    return {
        "classes": classes,
        "train": [{"class": c, "p": tr_num[c] / tr_den[c], "n": int(tr_den[c]),
                   "changed": int(tr_num[c])} for c in classes],
        "test": [{"class": c, "p": te_num[c] / te_den[c], "n": int(te_den[c]),
                  "changed": int(te_num[c])} for c in classes],
        "n_train": int(sum(tr_den[c] for c in classes)),
        "n_test": int(sum(te_den[c] for c in classes)),
    }


# ---- fig 3: fleet AUC, native and averaged -----------------------------------------------------
def stats_fig3(d, y, scores, cls_score, bid, cfg):
    """Per-arm AUC alone and averaged with the Class Oracle, each with a 95% clustered interval.

    Sorted by the OUTER value -- the better of alone/averaged -- so the rendered panel has no ragged
    outer edge.
    """
    series = []
    for arm in d["arms"]:
        native = auc(y, scores[arm])
        n_lo, n_hi = boot_ci(lambda i, s=scores[arm]: auc(y[i], s[i]), bid, cfg)
        combined_score = rank_average(cls_score, scores[arm])
        combined = auc(y, combined_score)
        c_lo, c_hi = boot_ci(lambda i, c=combined_score: auc(y[i], c[i]), bid, cfg)
        series.append({"arm": arm, "pretty": d["pretty"][arm],
                       "native": _f(native), "native_lo": n_lo, "native_hi": n_hi,
                       "combined": _f(combined), "combined_lo": c_lo, "combined_hi": c_hi})
    series.sort(key=lambda r: max(r["native"], r["combined"]))
    return {"series": series}


# ---- fig 4: ROC --------------------------------------------------------------------------------
def _win_loss_bands(fpr_grid, delta, min_width=0.01):
    """Contiguous FPR bands where `delta` keeps one sign, ignoring hairline flips.

    The empirical ROC is a step function, so the raw sign of (model - oracle) flips ~14 times on
    micro-oscillations. Delineating on every flip would produce a forest of bars; we keep only bands
    at least `min_width` wide, which recovers the substantive structure (one winning band).
    """
    bands = []
    start, cur = 0, np.sign(delta[0])
    for i in range(1, len(delta)):
        s = np.sign(delta[i])
        if s not in (cur, 0):
            if fpr_grid[i] - fpr_grid[start] >= min_width:
                bands.append((fpr_grid[start], fpr_grid[i], cur))
                start = i
            cur = s
    if fpr_grid[-1] - fpr_grid[start] >= min_width:
        bands.append((fpr_grid[start], fpr_grid[-1], cur))
    # merge neighbours that ended up with the same sign after dropping hairline flips
    merged = []
    for lo, hi, s in bands:
        if merged and merged[-1][2] == s:
            merged[-1] = (merged[-1][0], hi, s)
        else:
            merged.append((lo, hi, s))
    return merged


def stats_fig4(y, cls_score, native_score, native_label, combo, combo_label):
    """ROC curves plus the signed win/loss bands between the native arm and the Class Oracle.

    The curves are stored as the arrays the renderer draws, and the bands carry their integrated
    area so the annotation is a stored number rather than something the plot recomputes.
    """
    o_fpr, o_tpr, _ = roc_curve(y, cls_score)
    n_fpr, n_tpr, _ = roc_curve(y, native_score)
    c_fpr, c_tpr, _ = roc_curve(y, combo)

    grid = np.linspace(0, 1, 2001)
    m_i = np.interp(grid, n_fpr, n_tpr)
    o_i = np.interp(grid, o_fpr, o_tpr)
    delta = m_i - o_i
    bands = []
    for lo, hi, sign in _win_loss_bands(grid, delta):
        sel = (grid >= lo) & (grid <= hi)
        bands.append({"lo": float(lo), "hi": float(hi), "sign": int(sign),
                      "area": float(np.trapz(delta[sel], grid[sel]))})
    return {
        "n": int(len(y)),
        "grid": grid.tolist(),
        "native_interp": m_i.tolist(),
        "oracle_interp": o_i.tolist(),
        "oracle": {"fpr": o_fpr.tolist(), "tpr": o_tpr.tolist(), "auc": _f(auc(y, cls_score))},
        "native": {"fpr": n_fpr.tolist(), "tpr": n_tpr.tolist(), "auc": _f(auc(y, native_score)),
                   "label": native_label},
        "combo": {"fpr": c_fpr.tolist(), "tpr": c_tpr.tolist(), "auc": _f(auc(y, combo)),
                  "label": combo_label},
        "bands": bands,
    }


# ---- fig 5: detector degradation sweep ---------------------------------------------------------
def stats_fig5(d, y, sat, scores, arm, tbl, cfg, n_draw=30, step=0.01):
    """AUC vs the top-1 accuracy of the satellite damage-class detector feeding the Class Oracle.

    Everywhere else the Class Oracle is handed the TRUE satellite class, which is an upper bound --
    in deployment a model has to infer it. Rather than bolt the result to one BDA checkpoint (which
    would make it a statement about that checkpoint, not about the task), the detector is SIMULATED:
    at top-1 accuracy `a` the true class survives with probability `a`, otherwise it is resampled
    from a severity-proximity confusion model where adjacent severities are confused more often than
    distant ones. Both the Class Oracle and the combined score are rebuilt from the DETECTED class,
    so the x-axis reads "how good does satellite BDA need to be".

    Caption caveat: this detector errs symmetrically around the truth. A real model biased toward the
    dominant `no damage` class would give a weaker class branch at the same nominal top-1, so treat
    the accuracy axis as a lower bound on the requirement.

    The sweep draws from its own RandomState(cfg.seed) rather than a shared stream, so the curve does not
    depend on what was computed before it.
    """
    local_rng = np.random.RandomState(cfg.seed)
    labels = sorted(tbl)
    trans = {}
    for t in labels:
        w = []
        for x in labels:
            if x == t:
                w.append(0.0)
            elif cfg.severity_rank(t) is None or cfg.severity_rank(x) is None:
                w.append(0.4)                      # un-classified sits outside the severity order
            else:
                w.append(1.0 / (1.0 + abs(cfg.severity_rank(x) - cfg.severity_rank(t))) ** 2)
        trans[t] = np.array(w) / sum(w)
    rate_vec = np.array([tbl[x] for x in labels])
    idx_by_true = {t: np.where(sat == t)[0] for t in labels}
    truth = np.array([tbl.get(c, 0.0) for c in sat])
    model = scores[arm]

    def detect(acc):
        out = truth.copy()
        for t, idx in idx_by_true.items():
            if idx.size == 0:
                continue
            flip = idx[local_rng.rand(idx.size) >= acc]
            if flip.size:
                out[flip] = rate_vec[local_rng.choice(len(labels), size=len(flip), p=trans[t])]
        return out

    # Swept over the full 0-1 range. The left end is NOT "no information": at accuracy 0 the detector
    # never returns the true class but still resamples from its severity neighbourhood, so the
    # emitted class stays correlated with the truth and the curve sits above chance.
    accs = np.round(np.arange(0.0, 1.0 + step / 2, step), 4)
    om, osd, cm, csd = [], [], [], []
    for a in accs:
        o_, c_ = [], []
        for _ in range(1 if a >= 1.0 else n_draw):
            det = detect(a)
            o_.append(auc(y, det))
            c_.append(auc(y, rank_average(det, model)))
        om.append(np.mean(o_))
        osd.append(np.std(o_))
        cm.append(np.mean(c_))
        csd.append(np.std(c_))
    perfect = auc(y, truth)
    cm_arr = np.array(cm)
    hit = np.where(cm_arr >= perfect)[0]
    return {
        "arm": arm, "pretty": d["pretty"][arm], "n": int(len(y)), "n_draw": n_draw,
        "accs": accs.tolist(),
        "oracle_mean": [float(v) for v in om], "oracle_sd": [float(v) for v in osd],
        "combo_mean": [float(v) for v in cm], "combo_sd": [float(v) for v in csd],
        "perfect_oracle": _f(perfect),
        "combo_reaches_perfect_at_acc": float(accs[hit[0]]) if len(hit) else None,
    }


# ---- fig 6: fleet x class AUC ------------------------------------------------------------------
def stats_fig6(d, y, sat, scores, cfg):
    """Per-class AUC for every arm, ordered by overall AUC.

    The Class Oracle + Change Model Average deliberately does NOT appear. Within one satellite class
    every building carries the same rate-table score, so rank(class) is constant there and the
    combination collapses to `const + 0.5*rank(model)` -- a monotone transform of the model's own
    ranking, hence bit-identical per-class AUC. The combination can only pay off ACROSS classes,
    which is what figs 3/4/5 measure; a row here would just duplicate its base model.
    """
    order = sorted(d["arms"], key=lambda a: -auc(y, scores[a]))
    cell = np.full((len(order), len(cfg.classes)), np.nan)
    for i, arm in enumerate(order):
        for j, c in enumerate(cfg.classes):
            m = sat == c
            if len(np.unique(y[m])) > 1:
                cell[i, j] = auc(y[m], scores[arm][m])
    # Summary the prose quotes: among arms clearly above chance overall, how many rank each class.
    above = [i for i, a in enumerate(order) if auc(y, scores[a]) > 0.52]
    per_class = []
    for j, c in enumerate(cfg.classes):
        col = cell[above, j]
        per_class.append({"class": c, "mean": _f(np.nanmean(col)),
                          "n_above_half": int((col > 0.5).sum()), "n_arms": len(above)})
    return {
        "classes": list(cfg.classes),
        "arms": order,
        "row_labels": [d["pretty"][a] for a in order],
        "cell": [[_f(v) for v in row] for row in cell],
        "n_by_class": [int((sat == c).sum()) for c in cfg.classes],
        "n": int(len(y)),
        "summary": per_class,
    }


# ---- fig 7 (supplemental): score separation ----------------------------------------------------
def stats_fig7(d, y, scores, arm, bins=46):
    """Truth-conditioned predicted-score distributions for one arm.

    Each histogram is normalised to its OWN class (bars sum to 100%), not plotted as raw counts. AUC
    depends only on the two conditional distributions and is invariant to class prevalence, so counts
    would make the larger class sit higher across the range for reasons unrelated to separation.
    """
    s = scores[arm]
    edges = np.linspace(s.min(), s.max(), bins)
    out = {"arm": arm, "pretty": d["pretty"][arm], "n": int(len(y)),
           "bin_edges": edges.tolist(),
           "score_min": float(s.min()), "score_max": float(s.max()),
           "n_above_half": int((s > 0.5).sum())}
    for name, mask in (("unchanged", y == 0), ("changed", y == 1)):
        sel = s[mask]
        counts, _ = np.histogram(sel, bins=edges)
        out[name] = {"n": int(mask.sum()),
                     "percent": (100.0 * counts / int(mask.sum())).tolist(),
                     "mean": float(sel.mean())}
    return out


# ---- fig 8 (supplemental): satellite x sUAS pairwise AUC ---------------------------------------
def stats_fig8(d, sat, drone, scores, arm, bid, cfg):
    """Within a satellite row, separate that cell's buildings (positives) from the row's
    diagonal/agreement cell (negatives). Gated at cfg.min_cell on both sides.
    """
    s = scores[arm]
    n = len(cfg.classes)
    matrix = np.full((n, n), np.nan)
    n_pos = np.zeros((n, n), int)
    ci = {}
    for i, sc in enumerate(cfg.classes):
        neg = (sat == sc) & (drone == sc)
        if neg.sum() < cfg.min_cell:
            continue
        for j, dc in enumerate(cfg.classes):
            if dc == sc:
                continue
            pos = (sat == sc) & (drone == dc)
            n_pos[i, j] = pos.sum()
            if pos.sum() < cfg.min_cell:
                continue
            yy = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
            ss = np.r_[s[pos], s[neg]]
            bb = np.r_[bid[pos], bid[neg]]
            matrix[i, j] = auc(yy, ss)
            ci[f"{i},{j}"] = boot_ci(
                lambda k, yv=yy, sv=ss: auc(yv[k], sv[k]) if len(np.unique(yv[k])) > 1 else np.nan,
                bb, cfg)
    finite = np.isfinite(matrix)
    return {
        "arm": arm, "pretty": d["pretty"][arm], "classes": list(cfg.classes),
        "auc": [[_f(v) for v in row] for row in matrix],
        "n_pos": n_pos.tolist(),
        "ci": ci,
        "cells_computed": int(finite.sum()),
        "n_positives_used": int(n_pos[finite].sum()),
    }


# ---- orchestration -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"),
                    help="directory holding figure_data.json; figure_stats.json is written here too")
    add_taxonomy_arg(ap)
    ap.add_argument("--supplemental_arm", default="scalemae-seg",
                    help="arm for the supplemental per-model panels (figs 7 and 8). The main-body "
                         "figures always follow the top-AUC arm.")
    ap.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT,
                    help="bootstrap resamples behind every confidence interval")
    ap.add_argument("--min_cell", type=int, default=DEFAULT_MIN_CELL,
                    help="smallest stratum or matrix cell that gets reported")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="base seed; each consumer derives its own generator from it")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    cfg = Config(classes=tuple(damage_classes(args.channels_parameters_file)),
                 n_boot=args.n_boot, min_cell=args.min_cell, seed=args.seed)
    print(f"data    = {data_dir}", flush=True)
    print(f"classes = {list(cfg.classes)}  (from {os.path.basename(args.channels_parameters_file)})",
          flush=True)

    d, y, g, sat, drone, bid, scores, n_g, dropped = load(data_dir, cfg)
    print(f"n={len(y)}  positives={y.sum()}  arms={len(d['arms'])}  "
          f"buildings={len(np.unique(bid))}", flush=True)

    tbl, _, _ = rate_table(d["class_rates"], "train", cfg)
    cls_score = np.array([tbl.get(c, 0.0) for c in sat])
    ceil_g = auc(y, cls_score)
    ceil_w = within_ortho_auc(y, cls_score, g, n_g)
    print(f"class-table ceiling: global {ceil_g:.4f} | within-ortho {ceil_w:.4f}", flush=True)

    # TWO different "best" arms, and they are not the same model.
    #   top_native -- highest AUC on its own. Drives the per-model panels, which are about how one
    #                 model behaves.
    #   top_combo  -- whose AVERAGE with the Class Oracle scores highest. Drives figs 4 and 5, which
    #                 are about the combination. The best solo ranker need not combine best: what the
    #                 average needs is evidence INDEPENDENT of the class score, and a model that has
    #                 already learned the class structure adds little.
    # Both are selected on the test set. That is a real caveat for the combination claim, since no
    # held-out split is left to choose the arm on -- state it wherever the number is reported.
    top_native = max(d["arms"], key=lambda a: auc(y, scores[a]))
    combo_of = {a: rank_average(cls_score, scores[a]) for a in d["arms"]}
    top_combo = max(d["arms"], key=lambda a: auc(y, combo_of[a]))
    combo = combo_of[top_combo]
    combo_label = f"Average(Class Oracle, {d['pretty'][top_combo]})"
    print(f"top arm alone     : {d['pretty'][top_native]} ({auc(y, scores[top_native]):.4f})",
          flush=True)
    print(f"best averaged arm : {d['pretty'][top_combo]} ({auc(y, combo):.4f})", flush=True)

    gain_lo, gain_hi = boot_ci(lambda i: auc(y[i], combo[i]) - auc(y[i], cls_score[i]),
                               bid, cfg)
    supp = args.supplemental_arm if args.supplemental_arm in d["arms"] else top_native

    stats = {
        "population": {
            "n": int(len(y)), "positives": int(y.sum()),
            "buildings": int(len(np.unique(bid))), "orthomosaics": int(n_g),
            "dropped_outside_taxonomy": int(dropped),
        },
        "arms": list(d["arms"]),
        "pretty": d["pretty"],
        "pretrained": d["pretrained"],
        "rate_table": tbl,
        "damage_classes": list(cfg.classes),
        "bootstrap": {"n_boot": cfg.n_boot, "seed": cfg.seed, "min_cell": cfg.min_cell,
                      "unit": "building (clustered)"},
        "class_oracle": {"auc_global": _f(ceil_g), "auc_within_ortho": _f(ceil_w)},
        "top_native": top_native,
        "top_combo": top_combo,
        "combo_label": combo_label,
        "combo_gain_over_oracle": {
            "value": _f(auc(y, combo) - ceil_g), "lo": gain_lo, "hi": gain_hi,
        },
        "supplemental_arm": supp,
        "fig2": stats_fig2(d["class_rates"], cfg),
        "fig3": stats_fig3(d, y, scores, cls_score, bid, cfg),
        "fig4": stats_fig4(y, cls_score, scores[top_native], d["pretty"][top_native],
                           combo, combo_label),
        "fig5": stats_fig5(d, y, sat, scores, top_combo, tbl, cfg),
        "fig6": stats_fig6(d, y, sat, scores, cfg),
        "fig7": stats_fig7(d, y, scores, supp),
        "fig8": stats_fig8(d, sat, drone, scores, supp, bid, cfg),
    }

    out = os.path.join(data_dir, "figure_stats.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=1)

    print(f"\n{'arm':>24} {'alone':>8} {'95% CI':>16} {'averaged':>9} {'95% CI':>16}", flush=True)
    for r in sorted(stats["fig3"]["series"], key=lambda r: -r["combined"]):
        print(f"{r['pretty']:>24} {r['native']:>8.4f} [{r['native_lo']:.3f},{r['native_hi']:.3f}] "
              f"{r['combined']:>9.4f} [{r['combined_lo']:.3f},{r['combined_hi']:.3f}]", flush=True)
    print(f"\n{combo_label}: AUC {auc(y, combo):.4f}  gain over the Class Oracle "
          f"{auc(y, combo) - ceil_g:+.4f}  95% CI [{gain_lo:+.4f},{gain_hi:+.4f}]", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
