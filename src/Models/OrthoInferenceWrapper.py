from collections import defaultdict

from shapely import MultiPolygon, Polygon

from dataset.constants import BDA_DAMAGE_CLASSES
from modeling.Spatial import LabeledBuilding
from modeling.Sample import ConfidenceResult
from modeling.Alignment import AdjustmentVectorSubfield


def joint_file_pred_key(ortho_name, prediction_id, gsd_x=None, gsd_y=None):
    # Keys are GSD-prefixed so multiscale validation/evaluation can bucket predictions by the
    # resolution they were made at. Keys built without a GSD get an empty prefix ("__...") and
    # still round-trip through parse_pred_key.
    gsd_key = ("" if gsd_x is None else str(gsd_x)) + "_" + ("" if gsd_y is None else str(gsd_y))
    return gsd_key + "_" + ortho_name + "_" + prediction_id

def parse_pred_key(pred_key):
    splits = pred_key.split("_")
    gsd_x = splits[0]
    gsd_y = splits[1]
    prediction_id = splits[-1]
    ortho_name = "_".join(splits[2:-1])
    return ortho_name, prediction_id, gsd_x, gsd_y

# Perform fusion using mathematically sound spatial aggregation
# pylint: disable-next=too-many-branches
def fuse_bda_tiled_inference(tiled_preds, run_uq=False, class_labels=None):
    if class_labels is None:
        class_labels = BDA_DAMAGE_CLASSES
    fused_labels = {}

    missed_labels = set()

    for prediction_id, inferences in tiled_preds.items():
        label_totals = defaultdict(float)
        gsd = None
        adjusted = False
        polygon_parts = []
        label_sources = []
        adjustments = []

        # The sum of class probabilities across tiles equals the building's total pixel area,
        # which weights the per-tile mean UQ metrics during fusion.
        total_pixels = 0.0

        # Initialize UQ Accumulators
        uq_maxes = {"max_aleatoric": 0.0, "max_combined": 0.0, "max_epistemic": 0.0}
        uq_entropy_count = 0
        uq_weighted_means = {"mean_aleatoric": 0.0, "mean_combined": 0.0, "mean_epistemic": 0.0}

        for inference in inferences:
            if gsd is None and "gsd" in inference.keys():
                gsd = inference["gsd"]
            if "adjustments" in inference.keys():
                adjustments.extend(inference["adjustments"])
            if "polygon_part" in inference.keys():
                polygon_part = inference["polygon_part"]
                if isinstance(polygon_part, Polygon):
                    polygon_parts.append(polygon_part)
                elif isinstance(polygon_part, MultiPolygon):
                    polygon_parts.extend(polygon_part.geoms)
            if "adjusted" in inference.keys():
                adjusted = adjusted or inference["adjusted"]
            if "label_source" in inference.keys():
                label_sources.append(inference["label_source"])

            tile_area_weight = sum(inference["class_preds"].values())
            total_pixels += tile_area_weight

            # Accumulate Class Probabilities
            for label in class_labels:
                if label in inference["class_preds"]:
                    label_totals[label] += inference["class_preds"][label]
                else:
                    missed_labels.add(label)

            # Accumulate UQ Metrics
            if run_uq and tile_area_weight > 0:
                uq = inference["uq_metrics"]

                # Max the maxes
                uq_maxes["max_aleatoric"] = max(uq_maxes["max_aleatoric"], uq["max_aleatoric"])
                uq_maxes["max_combined"] = max(uq_maxes["max_combined"], uq["max_combined"])
                uq_maxes["max_epistemic"] = max(uq_maxes["max_epistemic"], uq["max_epistemic"])

                # Sum the counts
                uq_entropy_count += uq["high_entropy_count"]

                # Multiply mean by the tile's area weight for later division
                uq_weighted_means["mean_aleatoric"] += uq["mean_aleatoric"] * tile_area_weight
                uq_weighted_means["mean_combined"] += uq["mean_combined"] * tile_area_weight
                uq_weighted_means["mean_epistemic"] += uq["mean_epistemic"] * tile_area_weight

        # Finalize UQ Metrics
        final_uncertainty = None
        if run_uq and total_pixels > 0:
            final_uncertainty = {
                "max_aleatoric": uq_maxes["max_aleatoric"],
                "max_combined": uq_maxes["max_combined"],
                "max_epistemic": uq_maxes["max_epistemic"],
                # Divide by total area to get the true weighted mean
                "mean_aleatoric": uq_weighted_means["mean_aleatoric"] / total_pixels,
                "mean_combined": uq_weighted_means["mean_combined"] / total_pixels,
                "mean_epistemic": uq_weighted_means["mean_epistemic"] / total_pixels,
                "high_entropy_count": uq_entropy_count / total_pixels
            }

        label_source = ", ".join(list(set(label_sources)))

        prediction_confidence = ConfidenceResult(label_totals, True)

        asf = AdjustmentVectorSubfield(adjustments)

        fused_labels[prediction_id] = LabeledBuilding(identifier=prediction_id,
                                                      label=prediction_confidence.getHighestConfidenceLabel(),
                                                      geometry_source="AdjustedMultiPolygon",
                                                      pixel_geom=MultiPolygon(polygon_parts),
                                                      epsg_4326_geom=None,
                                                      adjusted=adjusted,
                                                      adjustment_subfield=asf,
                                                      confidence=prediction_confidence,
                                                      label_source=label_source,
                                                      views_considered=len(inferences)).jsonify()

        # Attach the UQ payload alongside per-building normalized class probabilities so the
        # meta-model and inspection tooling can consume them without re-deriving areas.
        if total_pixels == 0:
            print(f"Warning! fuse_bda_tiled_inference found 0 area for {prediction_id}.")
            total_pixels = 1.0
        fused_labels[prediction_id]["class_preds"] = {key: value/total_pixels for key, value in label_totals.items()}
        fused_labels[prediction_id]["uncertainty"] = final_uncertainty

    if len(missed_labels) > 0:
        print(f"Warning! Found Label(s) '{missed_labels}' but this was not found in class preds.")

    return fused_labels
