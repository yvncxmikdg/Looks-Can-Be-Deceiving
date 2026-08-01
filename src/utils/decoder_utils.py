from collections import defaultdict

import torch
import shapely

from modeling.utils.sample_generator_utils import draw_buildings_on_mask, geometry_in_frame, draw_road_lines_on_mask
from modeling.Spatial import RoadLine, LabeledRoadLine
from modeling.DataMap import DefaultLabel2IdxMap

def divide_road_line_into_sub_segments(road_line, segment_length_pixels):
    segmented_road_line_geometry = road_line.getGeometry("pixels").segmentize(float(segment_length_pixels))
    subsegments = []
    xx, yy = segmented_road_line_geometry.coords.xy
    for i in range(0, len(xx)-1):
        subseg = [(xx[i], yy[i]), (xx[i+1], yy[i+1])]
        subsegments.append(RoadLine(
            identifier=None,
            geometry_source=road_line.getGeometrySource(),
            pixel_geom=shapely.LineString(subseg),
            epsg_4326_geom=None,
            adjusted=road_line.isAdjusted(),
            adjustment_subfield=road_line.getAdjustmentSubfield(),
            label=road_line.getLabel()
        ))

    return subsegments

def compute_masked_pixel_agg(data, mask, keys_map, aggregation_strategies=None):
    """
    data: [C, H, W] tensor (probabilities or UQ maps)
    mask: [1, H, W] tensor
    keys_map: Labels2IdxMap object
    aggregation_strategies: list of strings e.g. ["sum", "mean", "max", "high_entropy_count"]
    """
    if aggregation_strategies is None:
        aggregation_strategies = ["sum"]
    result = {}

    # Convert [1, H, W] to a 2D boolean mask [H, W] to safely drop background pixels
    bool_mask = mask.squeeze(0) > 0

    # keys_map is explicitly a Labels2IdxMap instance
    for label in keys_map.getAllLabels():
        idx = keys_map.getIndex(label)

        # Isolate only the valid pixels for this specific channel
        valid_pixels = data[idx][bool_mask]

        # Handle polygons that are completely off-screen / empty mask
        if valid_pixels.numel() == 0:
            for agg in aggregation_strategies:
                out_key = f"{agg}_{label}" if len(aggregation_strategies) > 1 else label
                result[out_key] = 0.0
            continue

        for agg in aggregation_strategies:
            out_key = f"{agg}_{label}" if len(aggregation_strategies) > 1 else label

            if agg == "sum":
                result[out_key] = float(valid_pixels.sum())
            elif agg == "mean":
                result[out_key] = float(valid_pixels.mean())
            elif agg == "max":
                result[out_key] = float(valid_pixels.max())
            elif agg == "high_entropy_count":
                # specific threshold logic for UQ
                result[out_key] = int((valid_pixels > 1.5).sum())

    return result

#Function to map a preds tensor to labeled road lines via masking and counting/summing
def road_lines_to_labeled_road_line_segments(preds,
                                             road_lines,
                                             x_offset,
                                             y_offset,
                                             label_to_idx_map,
                                             segment_length_pixels=120,
                                             segment_buffer_width_pixels=40):
    resulting_segments = {}

    for road_line in road_lines:
        segments = divide_road_line_into_sub_segments(road_line, segment_length_pixels)
        resulting_segments[road_line] = {"parent":road_line, "segments":[]}
        for segment in segments:
            if geometry_in_frame(segment.getGeometry("pixels"), x_offset, y_offset, preds.shape[1], preds.shape[2]):
                #Generate a query mask where all the pixels that correspond to the segment are 1, and all the others are zero
                query = draw_road_lines_on_mask([segment],
                                                x_offset,
                                                y_offset,
                                                preds.shape[1],
                                                preds.shape[2],
                                                preds.shape[1],
                                                preds.shape[2],
                                                class_color_map=DefaultLabel2IdxMap(1),
                                                road_width_pixels=segment_buffer_width_pixels)

                #Reshape the query into the format of the preds
                mask = torch.tensor(query).reshape(1, preds.shape[1], preds.shape[2])

                #Count the number of pixels under the mask
                segment_pixel_counts = compute_masked_pixel_agg(preds, mask.to(preds.device), label_to_idx_map)

                #Get the label and confidence for this segment
                max_label, max_value = max(segment_pixel_counts.items(), key=lambda x:x[1])
                total_pixels = sum(segment_pixel_counts.values())

                #Store everything in the LabeledRoadLine object
                if total_pixels > 0:
                    resulting_segments[road_line]["segments"].append(LabeledRoadLine(identifier=None,
                                                                                     geometry_source=road_line.getGeometrySource(),
                                                                                     pixel_geom=segment.getGeometry("pixels"),
                                                                                     epsg_4326_geom=None,
                                                                                     adjusted=road_line.isAdjusted(),
                                                                                     adjustment_subfield=segment.getAdjustmentSubfield(),
                                                                                     label=max_label,
                                                                                     confidence=max_value/total_pixels,
                                                                                     parent_road_line_identifier=road_line.getId()))

    return resulting_segments

def buildings_to_pixel_agg(data, buildings, x_offset, y_offset, keys_map, aggregation_strategies=None):
    if aggregation_strategies is None:
        aggregation_strategies = ["sum"]
    labels = {}

    #For every building we have to evaluate
    for building in buildings:
        #Get the x and y coordinate from the current polygon and offset it accordingly
        query = draw_buildings_on_mask([building],
                                       x_offset, y_offset,
                                       data.shape[1], data.shape[2],
                                       data.shape[1], data.shape[2],
                                       class_color_map=DefaultLabel2IdxMap(1))

        # Reshape the query into the format of the preds
        mask = torch.tensor(query).reshape(1, data.shape[1], data.shape[2])

        # Safely compute the aggregations using the new logic
        labels[building.getId()] = compute_masked_pixel_agg(
            data,
            mask.to(data.device),
            keys_map,
            aggregation_strategies
        )

    return labels

def compute_masked_pixel_counts(preds, mask, label_to_idx_map):
    # Backwards-compatible alias for the single-strategy aggregation: per-class probability-mass
    # sums under the mask.
    return compute_masked_pixel_agg(preds, mask, label_to_idx_map)

def buildings_to_pixel_counts(preds, buildings, x_offset, y_offset, label_to_idx_map):
    # Backwards-compatible alias for the single-strategy aggregation: returns per-building
    # per-class probability-mass sums, the contract callers relied on before
    # buildings_to_pixel_agg generalized the aggregation strategies.
    return buildings_to_pixel_agg(preds, buildings, x_offset, y_offset, keys_map=label_to_idx_map)

def combine_gathered_preds_dictionaries(outputs, accessor):
    result = defaultdict(list)
    for output in outputs:
        for k, v in accessor(output).items():
            result[k].extend(v)
    return result

def combine_gathered_labels(outputs, accessor):
    result = {}
    for output in outputs:
        for k, v in accessor(output).items():
            result[k] = v
    return result

def combine_gathered_loss(outputs, accessor):
    result = []
    for output in outputs:
        result.extend(accessor(output))
    return result
