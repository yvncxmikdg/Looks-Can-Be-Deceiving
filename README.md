# The CHANGE task

Binary per-building change detection across sensors: a **satellite** "before" assessment against a
**drone** "after" assessment of the same building. A building is `CHANGED` when its damage label
differs between the two views and `UNCHANGED` when they agree — so the label is *cross-view
disagreement*, not a re-segmentation of the imagery.

This README covers running `train.py` and `infer.py` on CHANGE, and reproducing the figures for
**Looks Can Be Deceiving** from the archived runs. For the general modeling workflow and the
configuration model that the flags below come from, see [`README.md`](README.md).

| | |
| --- | --- |
| Task key | `CHANGE` (the `task:` field in a model hyperparameters YAML) |
| Task module | [`Models/TaskModuleCHANGE.py`](Models/TaskModuleCHANGE.py), dispatched by [`Models/model_registry.py`](Models/model_registry.py) |
| Evaluator | [`evaluate_CHANGE.py`](evaluate_CHANGE.py) |
| Classes | 3 — `CHANGED`, `UNCHANGED`, background ([`change_pred.yaml`](data_envs/Tasks/CHANGE/Channels/change_pred.yaml)) |
| Paper figures | [`paper_figures/looks_can_be_deceiving/`](../../paper_figures/looks_can_be_deceiving/README.md) |

Two properties of the task are fixed in code rather than configuration, and both matter when reading
results:

- **Anchors are post-event satellite tiles.** Sample windows centre on `Satellite`-source
  orthomosaics, and `CHANGE_ANCHOR_EVENT_PHASE` in
  [`utils/initialize_dataset.py`](utils/initialize_dataset.py) restricts them to post-event imagery.
  A pre-event tile carries no damage signal, so anchoring on one would measure priors instead of
  observable change.
- **`change_decision_rule: all`.** A building is `CHANGED` only when *every* parallel view from
  another source disagrees with the anchor's label; same-source views are skipped. The rule is set
  in the generator YAML and passed to `evaluate_CHANGE.py` so the two cannot drift.

## Before you run anything

1. **Environment** — the GPU build. See [Getting Started](../../README.md#getting-started);
   don't `pip install -r requirements.txt` directly.

   ```bash
   ./setup_env.sh cu121
   ```

2. **Dataset** — CRASAR-U-DROIDs, from
   [`CRASAR/CRASAR-U-DROIDs`](https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs). CHANGE needs
   both the **UAS** and the **SATELLITE** stripes; the crewed stripe is unused
   ([`train_val_change.yaml`](data_envs/Sources/train_val_change.yaml)).

3. **A dataset-paths YAML** under [`data_envs/Datasets/`](data_envs/Datasets/) pointing at your copy.
   Copy an existing one and edit the roots. Its top-level `statistics:` key must resolve to the
   dataset's `statistics.csv`, or pass `--ortho_stats_file` to override it.

4. **Pretrained backbones**, for the five arms that use them. Their YAMLs carry an absolute cluster
   path — repoint `model_parameters.encoder_parameters.backbone_path` at your own copy before
   training, or the run dies on load.

## The 13 architectures

Every arm in the paper was launched with the *same* configuration except the model hyperparameters
YAML. Slugs and display names come from
[`arms.yaml`](../../paper_figures/looks_can_be_deceiving/arms.yaml), which the figure pipeline reads;
the archives are the promoted run bundles described under [Archived runs](#archived-runs).

| Slug | Architecture | Model hyperparameters YAML | Pretrained | Archive |
| --- | --- | --- | --- | --- |
| `scalemae` | ScaleMAE + UperNet | [`CHANGE_scalemae_upernet_pretrained.yaml`](Models/UperNet/hyperparameters/CHANGE_scalemae_upernet_pretrained.yaml) | yes | [run](https://huggingface.co/opntesqwrcky/20260724-032317_change-post-512-scalemae_714d2992) |
| `scalemae-seg` | ScaleMAE + Segmenter | [`CHANGE_scalemae_segmenter_pretrained.yaml`](Models/Segmenter/hyperparameters/CHANGE_scalemae_segmenter_pretrained.yaml) | yes | [run](https://huggingface.co/opntesqwrcky/20260724-041133_change-post-512-scalemae-seg_714d2992) |
| `satmae` | SatMAE + UperNet | [`CHANGE_satmae_upernet_pretrained.yaml`](Models/UperNet/hyperparameters/CHANGE_satmae_upernet_pretrained.yaml) | yes | [run](https://huggingface.co/opntesqwrcky/20260724-032351_change-post-512-satmae_714d2992) |
| `convmae` | ConvMAE + UperNet | [`CHANGE_convmae_upernet_pretrained.yaml`](Models/UperNet/hyperparameters/CHANGE_convmae_upernet_pretrained.yaml) | yes | [run](https://huggingface.co/opntesqwrcky/20260724-032351_change-post-512-convmae_714d2992) |
| `dino-m2f` | DINOv3 + Mask2Former | [`CHANGE_dino_m2f_retrain.yaml`](Models/Mask2Former/hyperparameters/CHANGE_dino_m2f_retrain.yaml) | yes | [run](https://huggingface.co/opntesqwrcky/20260728-165500_change-post-512-dino-m2f_47741e7a) |
| `resnet` | ResNet101 + UperNet | [`CHANGE_resnet_upernet.yaml`](Models/UperNet/hyperparameters/CHANGE_resnet_upernet.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-032508_change-post-512-resnet_714d2992) |
| `psp101` | ResNet101 + PSPNet | [`CHANGE_psp_resnet101.yaml`](Models/PSPNet/hyperparameters/CHANGE_psp_resnet101.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-092719_change-post-512-psp101_714d2992) |
| `dlv3p` | ResNet101 + DeepLabV3+ | [`CHANGE_deeplabv3plus.yaml`](Models/DeepLabV3Plus/hyperparameters/CHANGE_deeplabv3plus.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-092836_change-post-512-dlv3p_714d2992) |
| `vitl` | ViT-L + UperNet | [`CHANGE_ViT_large_upernet_nopretrained.yaml`](Models/UperNet/hyperparameters/CHANGE_ViT_large_upernet_nopretrained.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-033218_change-post-512-vitl_714d2992) |
| `vitm` | ViT-M + UperNet | [`CHANGE_ViT_medium_upernet_nopretrained.yaml`](Models/UperNet/hyperparameters/CHANGE_ViT_medium_upernet_nopretrained.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-085141_change-post-512-vitm_714d2992) |
| `vits` | ViT-S + UperNet | [`CHANGE_ViT_small_upernet_nopretrained.yaml`](Models/UperNet/hyperparameters/CHANGE_ViT_small_upernet_nopretrained.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-092600_change-post-512-vits_714d2992) |
| `unet-vanilla` | Vanilla U-Net | [`CHANGE_UNet_noattention.yaml`](Models/MaskedUNet/hyperparameters/CHANGE_UNet_noattention.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-032508_change-post-512-unet-vanilla_714d2992) |
| `unet-attn` | Attention U-Net | [`CHANGE_UNet_attention.yaml`](Models/MaskedUNet/hyperparameters/CHANGE_UNet_attention.yaml) | no | [run](https://huggingface.co/opntesqwrcky/20260724-092908_change-post-512-unet-attn_714d2992) |

`train.py` and `infer.py` dispatch the architecture entirely from this YAML's
`model_type` / `encoder_type` / `decoder_type` keys, which is why one command covers all thirteen.

## Train

Swap `--model_hyperparameters_yaml_path` for a different arm; leave everything else alone. These are
the flags the paper's runs were launched with, transcribed from their experiment manifests.

```bash
python src/modeling/train.py --restart \
  --model_hyperparameters_yaml_path src/modeling/Models/UperNet/hyperparameters/CHANGE_scalemae_upernet_pretrained.yaml \
  --channels_hyperparameters_file_path src/modeling/data_envs/Tasks/CHANGE/Channels/change_pred.yaml \
  --train_datagen_hyperparameters_yaml_path src/modeling/data_envs/Tasks/CHANGE/Generators/weighted_centered_512_pixels_satellite.yaml \
  --val_datagen_hyperparameters_yaml_path src/modeling/data_envs/Tasks/CHANGE/Generators/centered_512_pixels_satellite.yaml \
  --data_source_config_parameters_file_path src/modeling/data_envs/Sources/train_val_change.yaml \
  --dataset_paths_file_path src/modeling/data_envs/Datasets/<your_machine>.yaml \
  --out_path /path/to/run_dir \
  --accelerator gpu --precision 16-mixed --num_gpus 1 --data_gen_workers 4
```

The **`_satellite` generator variants are not optional**. The plain
`weighted_centered_512_pixels.yaml` / `centered_512_pixels.yaml` anchor on sUAS imagery instead, which
is a different experiment — it trains and scores on drone-anchored windows rather than the
satellite-anchored ones the paper reports.

Everything shared across the fleet, and why:

| Setting | Value | Note |
| --- | --- | --- |
| Tiles | 512 px, centred on the anchor building | set in the generator YAMLs |
| Train sampling | `weighted`, `CHANGED:UNCHANGED = 1:1`, 2000 samples/epoch | the raw population is heavily `UNCHANGED` |
| Val sampling | `indexed` — every anchor once | no reweighting, so the monitor tracks the real distribution |
| Checkpoint monitor | `val_macro_f1`, `max`, top 3 kept | from the model YAML's `validation:` block |
| LR schedule monitor | `val_criterion_loss`, `min` | deliberately *not* the checkpoint monitor: an early macro-F1 fluke otherwise sets a high-water mark that decays the LR to zero while the loss is still falling |
| Early stopping | off (`patience: 0`) | same fluke would kill long runs at ~epoch `patience+1` |
| UQ head | none | `--uq_hyperparameters_yaml_path` omitted for every arm |
| Wall clock | 6 h, single A6000 | `max_epochs: 4000` is a bound, never reached |

Training writes TensorBoard logs and checkpoints under `<out_path>/tb_logs/`. Add `--debug` for a
fast smoke test on a tiny subset.

[`scripts/sh_scripts/modeling/train_change_512.sh`](../../scripts/sh_scripts/modeling/train_change_512.sh)
wraps this command and fills the shared paths from `scripts/local.env.sh`; it takes the model YAML as
its first argument. Note its defaults are the sUAS-anchored generators, so pass the `_satellite`
variants explicitly to reproduce a paper arm:

```bash
scripts/sh_scripts/modeling/train_change_512.sh \
  ../../../src/modeling/Models/UperNet/hyperparameters/CHANGE_scalemae_upernet_pretrained.yaml \
  ../../../src/modeling/data_envs/Tasks/CHANGE/Generators/weighted_centered_512_pixels_satellite.yaml \
  ../../../src/modeling/data_envs/Tasks/CHANGE/Generators/centered_512_pixels_satellite.yaml
```

## Infer

Inference runs the selected checkpoint over the **test** stripe and writes a predictions JSON.
Two things change from training: the data source becomes
[`test_change.yaml`](data_envs/Sources/test_change.yaml), and the datagen becomes the `indexed`
`centered_512_pixels_satellite.yaml` — the same generator used for validation, so every anchor is
scored exactly once with no reweighting.

**Pick the checkpoint the same way the paper did**: highest `val_macro_f1` among the top-3 saved. The
runs used [`scripts/slurm/select_best_checkpoint.py`](../../scripts/slurm/select_best_checkpoint.py),
which records its choice and the runners-up in `test_inference/checkpoint_selection.yaml`. Each
archive ships that file, so an arm's exact checkpoint filename — epoch, step and score — is
recoverable even though the weights are not.

```bash
python src/modeling/infer.py \
  --model_path /path/to/run_dir/tb_logs/<model>_CHANGE/version_0/checkpoints/<best>.ckpt \
  --preds_path /path/to/run_dir/test_inference/preds.json \
  --model_hyperparameters_yaml_path src/modeling/Models/UperNet/hyperparameters/CHANGE_scalemae_upernet_pretrained.yaml \
  --channels_hyperparameters_file_path src/modeling/data_envs/Tasks/CHANGE/Channels/change_pred.yaml \
  --infer_datagen_hyperparameters_yaml_path src/modeling/data_envs/Tasks/CHANGE/Generators/centered_512_pixels_satellite.yaml \
  --data_source_config_parameters_file_path src/modeling/data_envs/Sources/test_change.yaml \
  --dataset_paths_file_path src/modeling/data_envs/Datasets/<your_machine>.yaml \
  --dataset_stripe test \
  --accelerator gpu --precision 16-mixed --data_gen_workers 4
```

Then score it. `evaluate_CHANGE.py` rebuilds ground truth from the annotations — it does **not** read
labels out of `preds.json`, whose `label` field is the model's argmax *prediction* — and writes the
run's own `AUC_ROC.global` and `AUC_ROC.within_orthomosaic`, each with a building-clustered 95% CI:

```bash
python src/modeling/evaluate_CHANGE.py \
  --preds_paths /path/to/run_dir/test_inference/preds.json \
  --metrics_file /path/to/run_dir/test_inference/metrics.json \
  --channels_hyperparameters_file_path src/modeling/data_envs/Tasks/CHANGE/Channels/change_pred.yaml \
  --model_hyperparameters_yaml_path src/modeling/Models/UperNet/hyperparameters/CHANGE_scalemae_upernet_pretrained.yaml \
  --data_source_config_parameters_file_path src/modeling/data_envs/Sources/test_change.yaml \
  --dataset_paths_file_path src/modeling/data_envs/Datasets/<your_machine>.yaml \
  --change_decision_rule all
```

[`scripts/sh_scripts/inference/infer_change.sh`](../../scripts/sh_scripts/inference/infer_change.sh)
chains infer + evaluate + plots, taking the checkpoint path as its first argument.

## Reproducing the paper figures

The figure pipeline and every statistic behind it are documented in
**[`paper_figures/looks_can_be_deceiving/README.md`](../../paper_figures/looks_can_be_deceiving/README.md)**
— read that for the three stages, the Class Oracle, the bootstrap design, and what each figure shows.

The one thing that README leaves to you is `--logs_dir`: a directory holding one subdirectory per run,
each containing `test_inference/preds.json`. The archives below *are* that layout, so no retraining is
needed to rebuild the figures.

```bash
# 1. pull all 13 archived runs into one directory, keeping the repo name as the directory name
mkdir -p CHANGE_logs && cd CHANGE_logs
for run in \
  20260724-032317_change-post-512-scalemae_714d2992 \
  20260724-041133_change-post-512-scalemae-seg_714d2992 \
  20260724-032351_change-post-512-satmae_714d2992 \
  20260724-032351_change-post-512-convmae_714d2992 \
  20260728-165500_change-post-512-dino-m2f_47741e7a \
  20260724-032508_change-post-512-resnet_714d2992 \
  20260724-092719_change-post-512-psp101_714d2992 \
  20260724-092836_change-post-512-dlv3p_714d2992 \
  20260724-033218_change-post-512-vitl_714d2992 \
  20260724-085141_change-post-512-vitm_714d2992 \
  20260724-092600_change-post-512-vits_714d2992 \
  20260724-032508_change-post-512-unet-vanilla_714d2992 \
  20260724-092908_change-post-512-unet-attn_714d2992 ; do
  hf download "opntesqwrcky/$run" --local-dir "$run"
done
```

Then run the three stages with `--logs_dir` pointing at that directory — see the figures README for
the full invocation and the remaining arguments.

The directory names carry the arm slug: `build_figure_data.py` parses each one on `change-post-512-`
and drops the trailing commit, so `20260724-032317_change-post-512-scalemae_714d2992` resolves to
`scalemae` and is matched against `arms.yaml`. **Renaming a directory breaks that lookup**, and a slug
that isn't in `arms.yaml` is a hard error rather than a silent drop — which is the intent, since a
quietly missing arm would change every fleet figure.

### Archived runs

Each of the 13 repos above is one promoted run: the configuration, the test predictions and metrics,
the checkpoint-selection report, and image-stripped TensorBoard logs.

| Path | Contents |
| --- | --- |
| `test_inference/preds.json` | per-building predictions — the figure pipeline's only required input |
| `test_inference/metrics.json` | that run's own `evaluate_CHANGE.py` output |
| `test_inference/checkpoint_selection.yaml` | the checkpoint chosen, its `val_macro_f1`, and the runners-up |
| `configs/` | the channels, datagen and model hyperparameters the run actually used, as JSON |
| `tb_logs/` | TensorBoard events, image summaries stripped |
| `experiment_manifest.yaml` | task, model, configs, commit, comment |

**The archives do not contain model weights.** They are enough to rebuild every published figure and
to re-derive every published number, but re-running `infer.py` against a paper arm means training it
first. Reproducibility of the training side rests on the commit id in the run id — twelve arms were
launched at `714d2992`, `dino-m2f` at `47741e7a` — together with the configs bundled beside them.

### Where the numbers live

Deliberately none are restated here. A run's own metrics are in its `test_inference/metrics.json`;
everything the paper reports across the fleet is written to
`paper_figures/looks_can_be_deceiving/data/figure_stats.json` by stage 2, and the renderer derives
nothing beyond it. A number copied into prose is a number that can drift from the artifact that
produced it.
