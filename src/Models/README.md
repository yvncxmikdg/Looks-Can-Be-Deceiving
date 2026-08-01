# Models

This directory is the **model zoo** plus the glue that assembles a runnable model from a
hyperparameters YAML. If you are looking for the training/inference/evaluation workflow, start at
[`../README.md`](../README.md); this file explains how models are structured and how to add one.

## How a model is assembled

Every run is built by [`model_registry.py`](model_registry.py)'s
`parse_and_initialize_segmentation_model(...)`, which reads the model hyperparameters YAML and
combines two independent choices:

1. **A task** (`task:` → a `TaskModule`) that owns the loss, the train/val/predict steps, and the
   task metrics.
2. **A network** (`model_type:` + architecture keys) that produces the raw predictions.

```
model hyperparameters YAML
        │
        ├── task:        BDA | RDA | BDAADJ ──► TaskModule (loss + steps + metrics)
        │
        └── model_type:  Tower | MaskedTower | EncoderDecoder | MaskedEncoderDecoder
                         └── model / encoder_type / decoder_type ──► the neural network
```

The `TaskModule` wraps the network via `initialize_model(...)`, so the same architecture can serve
any task it supports.

### Task modules

| `task:` | Class | Notes |
| --- | --- | --- |
| `BDA` | [`TaskModuleBDA`](TaskModuleBDA.py) | Per-building damage classification; fuses tiled predictions per building. |
| `RDA` | [`TaskModuleRDA`](TaskModuleRDA.py) | Per-road-line damage classification. |
| `BDAADJ` | [`TaskModuleBDAADJ`](TaskModuleBDAADJ.py) | Building alignment; extends `TaskModuleBDA`, consumes a displacement field and warps polygons. |

All three inherit from [`TaskModuleBase`](TaskModuleBase.py), which holds the shared plumbing:
optimizer/scheduler config, checkpointing, class-weight strategies (`uniform`, `manual`, `ipw`,
`log_class_balance`), loss, TensorBoard telemetry, and the batch→model-input formatting.

### Model types

`model_type:` selects how the network is wrapped (defined in [`Model.py`](Model.py)):

| `model_type:` | Meaning |
| --- | --- |
| `Tower` | A self-contained network run end-to-end (e.g. a UNet). |
| `MaskedTower` | A `Tower` that also applies the query **mask** to its logits so predictions are confined to the object of interest (see [`Maskable`](Maskable.py)). |
| `EncoderDecoder` | A backbone **encoder** + a segmentation **decoder** composed at runtime. |
| `MaskedEncoderDecoder` | An `EncoderDecoder` with the same masking behavior as `MaskedTower`. |

Masking (via [`Maskable`](Maskable.py)) forces every non-object pixel to the background class by
driving its logits to the half-precision floor before softmax — this is how per-building /
per-road-line predictions stay inside the queried geometry.

### The architecture catalog

These are the string→class maps in [`model_registry.py`](model_registry.py). Add a new architecture
by registering it here (and, for encoder/decoder pieces, in the corresponding sub-map).

**Towers** (`STR2MODELTOWERCLASS`, keyed by `model:`)

| `model:` | Class | Folder |
| --- | --- | --- |
| `UNetTowerModule` | UNet | [`MaskedUNet/`](MaskedUNet/) |
| `PSPNetResNet` | PSPNet (ResNet backbone) | [`PSPNet/`](PSPNet/) |
| `DeepLabV3PlusResNet` | DeepLabV3+ (ResNet backbone) | [`DeepLabV3Plus/`](DeepLabV3Plus/) |
| `ZampieriEtAl2018` | Cascaded diffeomorphic alignment network ([Zampieri et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Armand_Zampieri_Multimodal_image_alignment_ECCV_2018_paper.pdf)), used for BDA_ADJ | [`ZampieriEtAl2018/`](ZampieriEtAl2018/) |
| `Attention_ZampieriEtAl2018` | Attention-augmented variant of the above | [`Attention_ZampieriEtAl2018/`](Attention_ZampieriEtAl2018/) |
| `RandomBaseline`, `RandomBaselineBDAADJ` | Random-prediction baselines | [`Baselines/`](Baselines/) |

**Masked towers** (`STR2MASKEDMODELTOWERCLASS`): `MaskedUNetTowerModule`, `MaskedPSPNetResNet`,
`MaskedDeepLabV3PlusResNet`, `MaskedUperNet`, `MaskedRandomBaseline`.

**Encoders** (`STR2MODELENCODERCLASS`, keyed by `encoder_type:`): `ViT`, `ScaleMAE`, `Dino`
(implemented in [`Backbones/`](Backbones/); `ResNet`/`Convext`/`Swin` are registered as
placeholders and are not yet wired up).

**Decoders** (`STR2DECODERCLASS`, keyed by `decoder_type:`): `UperNet` ([`UperNet/`](UperNet/)),
`Segmenter` ([`Segmenter/`](Segmenter/)), `Mask2Former` ([`Mask2Former/`](Mask2Former/)).

### The model I/O contract

Networks exchange data through the small typed containers in [`ModelDatum.py`](ModelDatum.py) —
`ModelInput` and `ModelOutput` dicts keyed by well-known field names (`CHANNEL_INPUT`, `MASK`,
`BUILDINGS`, `Y_HAT_SEGMENTATION_UNMASKED`, `Y_HAT_SEGMENTATION_MASKED`,
`Y_HAT_SEGMENTATION_LOGITS`, `DISPLACEMENT_FIELD`, `DO_SOFTMAX`, `GSD`). A segmentation model fills
in the `Y_HAT_SEGMENTATION_*` fields; a BDA_ADJ alignment model fills in `DISPLACEMENT_FIELD`. The
task modules read whichever fields their task needs, which is what lets one wrapping serve multiple
architectures.

## Anatomy of a model folder

A typical architecture folder contains:

- the network implementation (e.g. `UNet.py`, `Segmenter.py`);
- a `*TowerModule.py` / module wrapper that the registry instantiates and that returns the network
  via `get_model()`;
- a [`hyperparameters/`](MaskedUNet/hyperparameters/) folder of ready-to-run YAML configs, named by
  task (e.g. `BDA_UNet_attention.yaml`, `BDAADJ_ZampieriEtAl2018_v1.yaml`).

Backbones shared across architectures (ViT / ScaleMAE / DinoV3) live under
[`Backbones/`](Backbones/).

## Adding a new architecture

1. Create a folder under `Models/` for the model (put a shared backbone under
   [`Backbones/`](Backbones/) if applicable).
2. Implement the network and a module wrapper exposing `get_model()`, following an existing model
   (e.g. [`MaskedUNet/UNetTowerModule.py`](MaskedUNet/UNetTowerModule.py)).
3. Register it in [`model_registry.py`](model_registry.py) under the map that matches its
   `model_type` (`STR2MODELTOWERCLASS`, `STR2MASKEDMODELTOWERCLASS`, `STR2MODELENCODERCLASS`, or
   `STR2DECODERCLASS`).
4. Add a `hyperparameters/<TASK>_<name>.yaml` config. **Copy an existing YAML** and edit it rather
   than generating one — see the schema note in [`../README.md`](../README.md#create-a-new-model-configuration).
5. Train/infer with the shared launchers; only the `--model_hyperparameters_yaml_path` argument
   changes.
