import os
import json
import argparse

from collections import defaultdict

from modeling.Orthomosaic import OrthomosaicFactory
from modeling.Spatial import MultiLabeledRoadLineFactory
from modeling.utils.decoder_utils import divide_road_line_into_sub_segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with Model.")
    parser.add_argument("--test_spatial_folder", type=str, help="Path to polygons.")
    parser.add_argument("--test_adjustments_folder", type=str, help="Path to adjustments.")
    parser.add_argument("--preds_path", type=str, help="The path to file where predicitons will be stored.")
    parser.add_argument("--adjusted", action="store_true", help="Set this flag if you want the road lines to be adjusted.")
    parser.add_argument("--model_name", type=str, default="CRASAR-U-DROIDs", help="The name of the model that will be saved in the output preds file.")
    parser.add_argument("--road_line_segment_len_pixels", type=int, default=120, help="The length of the segments that the road lines will be divided into.")
    args = parser.parse_args()

    # Create the location where the predictions are going to be stored
    if not os.path.exists(args.preds_path):
        os.makedirs(args.preds_path, exist_ok=True)
        print("Created the directory to store the output: " + str(args.preds_path))

    # Load up the orthomosaics
    inference_orthomosaics = OrthomosaicFactory(
        rda_annotation_folder=args.test_spatial_folder,
        rda_adj_annotation_folder=args.test_adjustments_folder
    )

    # Pull the data out from the orthomosaics and save it in the format it based on what the RDA model returns.
    for orthomosaic in inference_orthomosaics:
        preds = defaultdict(list)
        segmented_road_lines = []
        for rl in orthomosaic.get_road_lines():
            segmented_road_lines.extend(divide_road_line_into_sub_segments(rl, args.road_line_segment_len_pixels))
        annotated_road_lines = MultiLabeledRoadLineFactory(segmented_road_lines, orthomosaic.get_road_line_annotation_polygons())
        for parent_road_line in annotated_road_lines:
            for labeled_segment in parent_road_line.get_labeled_sub_lines():
                preds[labeled_segment.getParentRoadLineId()].append(labeled_segment.jsonify(parent_road_line=parent_road_line))
        output = {"model_name":args.model_name, "preds":preds}
        with open(os.path.join(args.preds_path, orthomosaic.get_name() + "_preds" + ("_adjusted" if args.adjusted else "_unadjusted") + ".json"), "w") as f:
            f.write(json.dumps(output))
