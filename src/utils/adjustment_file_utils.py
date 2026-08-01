import os
import json

from modeling.Alignment import AdjustmentVectorFieldFactory


def load_adjustment_subfield_preds_by_id(predictions_paths):
    all_pred_adjustments = {}
    for preds_path in predictions_paths:
        with open(preds_path, "r") as preds_file:
            preds_data = json.load(preds_file)
            for building_id, pred in preds_data["preds"].items():
                all_pred_adjustments[building_id] = AdjustmentVectorFieldFactory(pred["adjustment_subfield"])
    return all_pred_adjustments


def write_ortho_adjustment_file(output_adjustments_folder, ortho, ortho_result):
    # ortho.get_name() already carries the full file name (e.g. "<name>.geo.tif.json"),
    # so just normalize the extension instead of appending another ".geo.tif.json".
    output_name = ortho.get_name()
    if not output_name.endswith(".json"):
        output_name = output_name + ".json"
    with open(os.path.join(output_adjustments_folder, output_name), "w") as output_file:
        output_file.write(json.dumps(ortho_result))


def generate_adjustment_files(orthomosaics, output_adjustments_folder, per_building_fn):
    """Build one adjustment file per orthomosaic.

    For each building, ``per_building_fn(building, start_x, start_y)`` is called with the
    building's pixel-space centroid. A returned value is appended to that ortho's result list;
    returning ``None`` skips the building.
    """
    os.makedirs(output_adjustments_folder, exist_ok=True)
    for ortho in orthomosaics:
        ortho_result = []
        for building in ortho.get_buildings():
            centroid = building.getGeometry("pixels").centroid
            entry = per_building_fn(building, centroid.x, centroid.y)
            if entry is not None:
                ortho_result.append(entry)
        write_ortho_adjustment_file(output_adjustments_folder, ortho, ortho_result)


def add_adjustment_io_args_to_parse_args(arg_parser,
                                         add_buildings_labels_folder=False,
                                         add_output_adjustments_folder=False,
                                         add_preds_path=False):
    if add_buildings_labels_folder:
        arg_parser.add_argument(
            "--buildings_labels_folder",
            type=str,
            help="Path to buildings labels folder"
        )
    if add_output_adjustments_folder:
        arg_parser.add_argument(
            "--output_adjustments_folder",
            type=str,
            help="Path to buildings adjustments folder"
        )
    if add_preds_path:
        arg_parser.add_argument(
            "--preds_path",
            type=str,
            nargs="+",
            help="The path to file that contains the model predicitons.",
        )
    return arg_parser
