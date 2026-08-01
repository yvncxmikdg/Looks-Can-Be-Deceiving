import argparse
import os
import json

from collections import defaultdict
from shapely import Polygon

from dataset.constants import UNCLASSIFIED
from modeling.utils.confidence_utils import confidence_scalar

def displayed_confidence(prediction):
    """The confidence to SURFACE for a building: the model's normalized probability in its
    predicted label (the per-class ``class_preds`` map). The sibling ``confidence`` field holds
    per-class PIXEL-VOTE COUNTS - not comparable across buildings and they saturate the confidence
    color scale/histograms - so it must never be reported as the displayed Confidence. Falls back to
    ``confidence`` only for legacy predictions that predate ``class_preds``."""
    scores = prediction.get("class_preds") or prediction.get("confidence")
    return confidence_scalar(scores, prediction["label"])

def make_prediction_dict(label, confidence, polygon_id, label_source, geometry_source, views_considered, view_strategy):
    return {"Confidence": confidence,
            "Label": label,
            "Label Source": label_source,
            "Geometry Source": geometry_source,
            "ID":polygon_id,
            "Views Considered":views_considered,
            "Multi-View Fusion Strategy": view_strategy,
            "Multi-View Fusion Occured": views_considered > 1}

def match_polygon_id_to_inference_id(predictions_keys, polygon_id):
    for key in predictions_keys:
        if polygon_id in key:
            return key
    return None

def get_predictions_by_file(polygon_data, predictions, id_2_model):
    resulting_predictions_by_file = defaultdict(lambda:{})
    for predictions_file in polygon_data.keys():
        for entry_data in polygon_data[predictions_file]:
            geometry_source = entry_data["geometry_source"] if "geometry_source" in entry_data.keys() else entry_data["source"]
            try:
                pred_key = match_polygon_id_to_inference_id(predictions, entry_data["id"])
                pred = predictions[pred_key]
                d = make_prediction_dict(label=pred["label"],
                                         confidence=displayed_confidence(pred),
                                         polygon_id=pred_key,
                                         label_source=id_2_model[pred_key],
                                         geometry_source=geometry_source,
                                         views_considered=1,
                                         view_strategy="N/A")
                resulting_predictions_by_file[predictions_file][pred_key] = d
            except KeyError:
                pass
    return resulting_predictions_by_file

def get_predictions_and_metadata_grouped_by_polygon(polygon_data_by_file, predictions, id_2_model):
    centroids = defaultdict(lambda:[])
    for filename in polygon_data_by_file.keys():
        for polygon_data in polygon_data_by_file[filename]:
            label = None
            geometry_source = None
            label_source = None
            try:
                pred_key = match_polygon_id_to_inference_id(predictions, polygon_data["id"])
                label = predictions[pred_key]
                geometry_source = polygon_data["source"]
                label_source = id_2_model[pred_key]
            except KeyError:
                pass
            if label:
                centroid = Polygon([(x["lon"], x["lat"]) for x in polygon_data['EPSG:4326']]).centroid.coords[0]
                centroids[centroid].append([polygon_data["id"], geometry_source, label_source, label])

    return list(centroids.values())

def pick_max_confidence(predictions_grouped_by_polygon):
    fused_predictions = {}
    for labels in predictions_grouped_by_polygon:
        # View SELECTION is ranked on the raw pixel-vote counts (the original behavior, kept so the
        # chosen view / fused label is unchanged); only the REPORTED confidence is the normalized
        # class probability.
        polygon_id, geometry_source, label_source, label = max(
            labels,
            key=lambda x: confidence_scalar(x[3]["confidence"], x[3]["label"]) - (1 if x[3]["label"] == UNCLASSIFIED else 0))
        fused_predictions[polygon_id] = make_prediction_dict(label=label["label"],
                                                             confidence=displayed_confidence(label),
                                                             polygon_id=polygon_id,
                                                             label_source=label_source,
                                                             geometry_source=geometry_source,
                                                             views_considered= len(labels),
                                                             view_strategy="Max Confidence" if len(labels) > 1 else "N/A")
    return fused_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fused the predictions made across the different orthomosaics being considered')
    parser.add_argument('--save_path', type=str, help='The path to where the plots should be saved.')
    parser.add_argument('--building_polygons_folder', type=str, help='The folder where the initial building polygons are saved.')
    parser.add_argument('--predictions_folder', type=str, help='The folder where the predictions are saved.')
    args = parser.parse_args()

    print("Loading the predictions from :", args.predictions_folder)
    predictions_data = {}
    id_2_model_map = {}
    for file in os.listdir(args.predictions_folder):
        file_path = os.path.join(args.predictions_folder, file)
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.loads(f.read())
                for predicted_polygon_id in data["preds"].keys():
                    id_2_model_map[predicted_polygon_id] = data["model_name"]
                predictions_data.update(data["preds"])

    print("Reading annotations from: ", args.building_polygons_folder)
    all_data = {}
    for file in os.listdir(args.building_polygons_folder):
        file_path = os.path.join(args.building_polygons_folder, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            all_data[file] = data

    predictions_by_file = get_predictions_by_file(all_data, predictions_data, id_2_model_map)

    predictions_by_polygon = get_predictions_and_metadata_grouped_by_polygon(all_data, predictions_data, id_2_model_map)

    predictions_fused_by_confidence = pick_max_confidence(predictions_by_polygon)

    for file in predictions_by_file:
        with open(os.path.join(args.save_path, file), 'w') as f:
            json.dump(predictions_by_file[file], f)

    with open(os.path.join(args.save_path, "fused_predictions.json"), 'w') as f:
        json.dump(predictions_fused_by_confidence, f)
