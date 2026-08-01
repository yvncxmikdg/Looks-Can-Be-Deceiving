import argparse

from modeling.Orthomosaic import OrthomosaicFactory
from modeling.utils.adjustment_file_utils import (
    generate_adjustment_files,
    add_adjustment_io_args_to_parse_args,
)
from modeling.utils.random_adjustment_utils import estimate_sigma_from_percentiles, generate_x_y_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    add_adjustment_io_args_to_parse_args(parser,
                                         add_buildings_labels_folder=True,
                                         add_output_adjustments_folder=True)
    args = parser.parse_args()

    #Compute load the building polygons that we will be using to measure adjustments
    orthomosaics = OrthomosaicFactory(bda_annotation_folder=args.buildings_labels_folder)

    sigma = estimate_sigma_from_percentiles()

    def random_adjustment(_building, start_x, start_y):
        generate_x_y_sample(sigma)
        return [[start_x, start_y], [start_x+1, start_y+1]]

    generate_adjustment_files(orthomosaics, args.output_adjustments_folder, random_adjustment)
