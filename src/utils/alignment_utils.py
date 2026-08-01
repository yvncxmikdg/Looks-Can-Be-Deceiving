import torch

from modeling.Spatial import Building, RoadLine
from modeling.Alignment import Adjustment
from modeling.utils.sample_generator_utils import offset_pixel_coords

# Function to compute the flow of the vector field in the shape (2, H, W) at a set of points (N, 2)
def nearest_flow(vector_field, points):
    # Get the indices of these points in the reference frame that we care about...
    points_indexed = points.round()
    points_x = points_indexed[:, 0].clamp(0, vector_field.shape[1]-1).int()
    points_y = points_indexed[:, 1].clamp(0, vector_field.shape[2]-1).int()

    # Go get the flow at those points.
    flow = vector_field[:, points_x, points_y]

    # Make sure the flow is in the right shape before returning it.
    return flow.permute(1,0)

# Normalize x, y coordinates to [-1, 1]
def normalize_coordinates(coords, height, width):
    return 2 * (coords.float().to(coords.device)) / torch.tensor([width - 1, height - 1]).float().to(coords.device) - 1

def get_adjustment_from_unadjusted_and_adjusted_building(unadjusted_building, adjusted_building, x_p, y_p):
    if not (isinstance(unadjusted_building, Building) and isinstance(adjusted_building, Building)):
        raise ValueError("Both objects passed must be buildings. Got", type(unadjusted_building), "and", type(adjusted_building))

    source_adjustment = (adjusted_building.getAdjustmentSubfield().getAdjustments())[0]
    new_adjustment_geom = offset_pixel_coords(source_adjustment.getGeometry(), x_p, y_p)
    dx = new_adjustment_geom.coords[1][0] - new_adjustment_geom.coords[0][0]
    new_end_x = unadjusted_building.getGeometry("pixels").centroid.x + dx
    dy = new_adjustment_geom.coords[1][1] - new_adjustment_geom.coords[0][1]
    new_end_y = unadjusted_building.getGeometry("pixels").centroid.y + dy
    return Adjustment(unadjusted_building.getGeometry("pixels").centroid.x,
                      unadjusted_building.getGeometry("pixels").centroid.y,
                      new_end_x,
                      new_end_y,
                      source_adjustment.getId(),
                      source_adjustment.isAttributable())

def get_adjustments_from_unadjusted_and_adjusted_road_line(unadjusted_road_line, adjusted_road_line):
    if not (isinstance(unadjusted_road_line, RoadLine) and isinstance(adjusted_road_line, RoadLine)):
        raise ValueError("Both objects passed must be RoadLine objects. Got", type(unadjusted_road_line), "and", type(adjusted_road_line))

    adjustments = []
    for unadjusted_coord, adjusted_coord in zip(unadjusted_road_line.getGeometry("pixels").coords, adjusted_road_line.getGeometry("pixels").coords):
        adjustments.append(Adjustment(unadjusted_coord[0],
                                      unadjusted_coord[1],
                                      adjusted_coord[0],
                                      adjusted_coord[1],
                                      None,
                                      True))
        # adjustments.append(Adjustment(unadjusted_coord.x,
        #                               unadjusted_coord.y,
        #                               adjusted_coord.x,
        #                               adjusted_coord.y,
        #                               None,
        #                               True))

    # NOTE: THIS MAY BE INSUFFICIENT DEPENDING ON HOW THE RDA ADJUSTMENTS LOGIC IS IMPLEMENTED. IT IS POSSIBLE THAT SUCH A MODEL WILL REQUIRE ADJUSTMENTS FOR
    # ALL LABELED SUBSPANS IN ADDITION TO THE TOP LINE LABELS. Likely the best way to get the correct adjustments from the labeled road line segments is to
    # use the relative coordinate transform between the adjusted and unadjusted geoms for each labeled segment. We can assume that the geometry of the base
    # lines, will align, but we will need to convert to relative coords to get them between the adjusted spaces. See the get_relative_span() function on the
    # road line class for more information

    return adjustments

def reconstruct_adjustments_from_unadjusted_adjusted_pairs(unadjusted_spatial_objects, adjusted_spatial_objects, x_p, y_p):
    adjustments = []
    for unadjusted_spatial_object, adjusted_spatial_object in zip(unadjusted_spatial_objects, adjusted_spatial_objects):
        if isinstance(unadjusted_spatial_object, Building) and isinstance(adjusted_spatial_object, Building):
            adjustments.append(get_adjustment_from_unadjusted_and_adjusted_building(unadjusted_spatial_object, adjusted_spatial_object, x_p, y_p))
        elif isinstance(unadjusted_spatial_object, RoadLine) and isinstance(adjusted_spatial_object, RoadLine):
            adjustments.extend(get_adjustments_from_unadjusted_and_adjusted_road_line(unadjusted_spatial_object, adjusted_spatial_object))
    return adjustments
