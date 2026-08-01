import argparse

from modeling.Orthomosaic import OrthomosaicFactory
from modeling.utils.adjustment_file_utils import (
    load_adjustment_subfield_preds_by_id,
    generate_adjustment_files,
    add_adjustment_io_args_to_parse_args,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    add_adjustment_io_args_to_parse_args(parser,
                                         add_buildings_labels_folder=True,
                                         add_output_adjustments_folder=True,
                                         add_preds_path=True)
    args = parser.parse_args()

    #Compute load the building polygons that we will be using to measure adjustments
    orthomosaics = OrthomosaicFactory(bda_annotation_folder=args.buildings_labels_folder)

    adjustment_vector_fields_by_building = load_adjustment_subfield_preds_by_id(args.preds_path)

    def predicted_adjustment(building, start_x, start_y):
        adjustment_vector_field = adjustment_vector_fields_by_building.get(building.getId())
        if adjustment_vector_field is None:
            return None
        return adjustment_vector_field.getAverageAdjustment(start_x, start_y).jsonify()

    generate_adjustment_files(orthomosaics, args.output_adjustments_folder, predicted_adjustment)
