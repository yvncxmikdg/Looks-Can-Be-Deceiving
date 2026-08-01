"""Dataset wiring and taxonomy lookup shared by this paper's pipeline stages.

Deliberately carries **no default paths**. Every location is a required CLI argument, because this
code ships as supplementary material and is expected to run on machines that look nothing like the one
it was written on -- a default pointing at someone's `H:` drive is worse than no default, since it
fails late and confusingly instead of at argument parsing.

The damage taxonomy is likewise not restated here: the class set and its severity order come from the
model's channels YAML, which is the repo's single source of truth for them.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# pylint: disable=wrong-import-position
from modeling.utils.hyperparameters import parse_hyperparameters
from modeling.Orthomosaic import MultisourceOrthomosaicFactory
from modeling.DataMap import color_map_from_channels_parameters_file


def add_dataset_args(ap):
    """Attach the dataset-location flags. All required -- see the module docstring."""
    ap.add_argument("--ortho_stats_file", required=True,
                    help="statistics.csv at the dataset root")
    ap.add_argument("--dataset_paths_file_path", required=True,
                    help="dataset-paths YAML giving the imagery/annotation roots on this machine")
    ap.add_argument("--data_source_config_parameters_file_path", required=True,
                    help="data-source YAML selecting which sensors participate, e.g. "
                         "src/modeling/data_envs/Sources/train_val_change.yaml")
    return ap


def add_taxonomy_arg(ap):
    """Attach the channels YAML flag that defines the satellite damage classes."""
    ap.add_argument("--channels_parameters_file", required=True,
                    help="channels YAML whose output_class_2_idx_map defines the satellite damage "
                         "classes and their severity order, e.g. "
                         "src/modeling/data_envs/Tasks/BDA/Channels/rgb_mask_no_obscured.yaml")
    return ap


def load_orthos(args, subset):
    """Build the CHANGE orthomosaic factory for one dataset stripe."""
    return MultisourceOrthomosaicFactory(
        dataset_paths_dict=parse_hyperparameters(args.dataset_paths_file_path, verbose=False),
        data_source_config_parameters=parse_hyperparameters(
            args.data_source_config_parameters_file_path, verbose=False),
        # The factory reads only ["task"] (to select the BDA annotation paths) and an optional
        # ["encoder"], so a literal avoids depending on a model yaml whose name may move.
        model_hyperparameters={"task": "CHANGE"},
        boundary_folder=None,
        statistics_file_path=args.ortho_stats_file,
        train_validation_test=subset,
        backend="rasterio",
        # train.py/infer.py pass [3, 4] here; the excluded 1-band products are all same-source as the
        # satellite anchors, so they are skipped as parallel views anyway and the labels are identical
        # either way. None keeps the population independent of that filter.
        required_channels=None,
    )


def damage_classes(channels_parameters_file):
    """Satellite damage classes in severity order, read from the channels YAML.

    `ColorMap.getLabelsInIndexOrder()` sorts the output classes by model index and drops the
    transparent background class, which is exactly the order these figures plot. Taking it from the
    channel map rather than restating it means a taxonomy change cannot leave the figures describing
    classes the models no longer emit.

    Note this is the taxonomy only. The channel map's colours are tuned for orthomosaic overlays
    (saturated primaries, chosen to stay visible over imagery) and are not legible in print, so the
    figures keep their own print palette -- see paper_style.py.
    """
    color_map = color_map_from_channels_parameters_file(channels_parameters_file)
    return color_map.getLabelsInIndexOrder()
