# CHANGE paper figures

Figures for the satellite→drone change-detection study, scored by **AUC**.

Why AUC: a classifier is a ranking plus a threshold. On this task the ranking transfers across
disaster events and the threshold does not — the disagreement rate per satellite damage class shifts
between the training events and the test event (figure 2), so a model that faithfully learns the training
distribution is *mis-thresholded* at test and its macro F1 collapses while its ordering is intact.
AUC scores the ordering alone, which is the part that survives. The deployment number is macro F1
after per-event calibration, which costs ~25–50 labelled buildings from the target event.

Figure numbers here are the **paper's** numbers. Paper figure 1 is not produced by this pipeline.

For how the models behind these figures were trained and scored — the 13 architectures, their launch
configuration, and where the archived runs that `--logs_dir` expects are published — see
[`src/modeling/README-CHANGE.md`](../../src/modeling/README-CHANGE.md).

## Pipeline

Three stages. Stage 1 reads the CRASAR-U-DROIDs annotations through the repo's own `Orthomosaic`
machinery, stage 2 does every statistic, and stage 3 only draws.

**Every path is a required argument.** There are no machine-specific defaults — see the
[conventions](../README.md#conventions) for why. Substitute your own locations below.

```bash
cd paper_figures/looks_can_be_deceiving
REPO=../..                 # repo root
DATA=/path/to/CRASAR-U-DROIDs
LOGS=/path/to/CHANGE_logs  # holds <run>/test_inference/preds.json per arm
CHANNELS=$REPO/src/modeling/data_envs/Tasks/BDA/Channels/rgb_mask_no_obscured.yaml

# 1. annotations + predictions -> data/figure_data.json
python build_figure_data.py \
    --logs_dir "$LOGS" \
    --ortho_stats_file "$DATA/statistics.csv" \
    --dataset_paths_file_path "$REPO/src/modeling/data_envs/Datasets/<your_paths>.yaml" \
    --data_source_config_parameters_file_path "$REPO/src/modeling/data_envs/Sources/train_val_change.yaml" \
    --channels_parameters_file "$CHANNELS"

# 2. all statistics: AUCs, bootstrap CIs, rate tables, ROC curves, detector sweep -- see "Where the
#    AUCs come from" below for what is native to a run vs. derived across the fleet
python compute_figure_stats.py --channels_parameters_file "$CHANNELS"

# 3. figures
python make_paper_figures.py
```

| stage | script | reads | writes |
|---|---|---|---|
| 1 | `build_figure_data.py` | annotations + `*/test_inference/preds.json` | `data/figure_data.json` |
| 2 | `compute_figure_stats.py` | `data/figure_data.json` | `data/figure_stats.json` |
| 3 | `make_paper_figures.py` | `data/figure_stats.json` | `figures/*.{png,pdf}` |

Stage 1 loads the **training** stripe as well as the test one. The training stripe carries no model
predictions, but it is what the Class Oracle is fitted on and what figure 2's train-vs-test comparison
plots, so its per-class disagreement counts ship in the same artifact.

Add `--expect_n 2282 --expect_positives 954` to stage 1 to re-assert the exact population this paper
scored; if a rebuild does not reproduce it, ground truth came from a materially different labeler and
every downstream figure would be silently wrong. The check is opt-in because those numbers describe
one dataset snapshot, not a property of the code.

### Where the AUCs come from

`evaluate_CHANGE.py` defines every AUC quantity in this paper — `global_auc`, `within_ortho_auc`,
`class_oracle_scores`, `rank_average`, `building_clustered_bootstrap_ci`. Run it on one arm's own
predictions and it writes that arm's native `AUC_ROC.global` and `AUC_ROC.within_orthomosaic` into its
`test_inference/metrics.json`, each with a building-clustered 95% CI; pass it
`--class_oracle_rate_table` and it also writes `AUC_ROC.class_oracle` and
`AUC_ROC.averaged_with_class_oracle` for that one arm. Stage 2 imports `global_auc`, `within_ortho_auc`,
`rank_average` and `building_clustered_bootstrap_ci` from `evaluate_CHANGE` rather than keeping copies
of them, so a number here is not merely *intended* to match a run's own metrics file — it is produced
by the same function call.

What stage 2 computes rather than reads is everything that needs more than one arm at a time — something
no single evaluation run has: the fleet comparison across all 13 arms (fig3), which arm's average with the
Class Oracle scores highest (figs 3–5 — not necessarily the arm that tops the native ranking), the ROC
curve behind a scalar AUC (fig4), the detector-accuracy sweep (fig5), and the per-damage-class and
satellite-vs-drone breakdowns (figs 6, 8), none of which any evaluation run groups its metrics by. The
Class Oracle's rate table comes from the training-stripe counts stage 1 already collected (the same
counts fig2 plots); every arm's CHANGED score (`change_probability`, computed in stage 1) is combined
with it through the imported `rank_average`. That combination across the fleet is stage 2's job, not a
second definition of AUC.

