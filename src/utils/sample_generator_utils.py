import shapely
import numpy as np

from PIL import Image, ImageDraw
from shapely.affinity import translate, scale

from modeling.Alignment import Adjustment

def arg_nearest_basis_polygon(point, unadjusted_polygons, adjusted_polygons, boundary_box):
    nearest_i = -1
    nearest_dist = float("inf")
    nearest_is_contained = False
    for i, adjusted_polygon in enumerate(adjusted_polygons):
        adjusted_polygon_is_in_frame = geometry_contained(adjusted_polygon, boundary_box)
        unadjusted_polygon_is_in_frame = geometry_contained(unadjusted_polygons[i], boundary_box)

        valid_basis = adjusted_polygon_is_in_frame or unadjusted_polygon_is_in_frame
        is_contained = adjusted_polygon_is_in_frame

        if valid_basis:
            if unadjusted_polygons[i].contains(point):
                return i, is_contained
            d = unadjusted_polygons[i].distance(point)
            if d < nearest_dist:
                nearest_dist = d
                nearest_i = i
                nearest_is_contained = is_contained
    return nearest_i, nearest_is_contained

#This function returns 3 grids that are relevant for learning how to adjust building polygons
#The first grid is a grid of the angles for the adjustments vector field
#The second grid is a grid of the magnitudes for the adjustments vector field
#The third is an occupancy grid which indicates if there is a valid adjustment which could be learned in this cell
def get_adjustment_field_grid(unadjusted_polygons,
                              adjusted_polygons,
                              polygon_adjustments,
                              x_offset,
                              y_offset,
                              x_source_dim,
                              y_source_dim,
                              x_target_dim,
                              y_target_dim):

    bbox_poly = shapely.Polygon([(x_offset,y_offset), \
                         (x_offset+x_source_dim,y_offset), \
                         (x_offset+x_source_dim,y_offset+y_source_dim), \
                         (x_offset,y_offset+y_source_dim), \
                         (x_offset,y_offset)])

    x_step = x_source_dim/x_target_dim
    y_step = y_source_dim/y_target_dim

    #Construct the 3 grids which will store our outputs
    adjustments = []

    #For every x/y pair we have
    for x_i in range(0, x_target_dim):
        for y_i in range(0, y_target_dim):
            #Compute the coord for that cell
            x_i_coord = x_i*x_step
            y_i_coord = y_i*y_step

            p = shapely.Point(x_i_coord, y_i_coord)

            #Find the nearest unadjusted polygon to the current cell
            nearest_index, is_contained = arg_nearest_basis_polygon(p,
                                                      [poly.getGeometry("pixels") for poly in unadjusted_polygons],
                                                      [poly.getGeometry("pixels") for poly in adjusted_polygons],
                                                      bbox_poly)
            if nearest_index != -1:
                #Get the adjustment that is associated with the current cell
                adjustment = polygon_adjustments[nearest_index].getAdjustmentForPoint(p)
                new_adjustment = Adjustment(x_i_coord,
                                            y_i_coord,
                                            x_i_coord+adjustment.getDx(),
                                            y_i_coord+adjustment.getDy(),
                                            attributable=is_contained)
            else:
                new_adjustment = Adjustment(x_i_coord,
                                            y_i_coord,
                                            x_i_coord,
                                            y_i_coord,
                                            attributable=False)
            adjustments.append(new_adjustment)

    return adjustments

def offset_pixel_coords(geom, x_offset, y_offset):
    return translate(geom, xoff=-1*x_offset, yoff=-1*y_offset)

def scale_pixel_coords(geom, x_scale_factor, y_scale_factor, origin=(0,0)):
    return scale(geom, xfact=x_scale_factor, yfact=y_scale_factor, origin=origin)

def geometry_contained(geom, polygon):
    return geom.intersects(polygon)

def geometry_in_frame(geom, x_offset, y_offset, x_source_dim, y_source_dim):
    offset_geom = offset_pixel_coords(geom, x_offset, y_offset)
    bbox = shapely.Polygon([[0,0], [0, y_source_dim], [x_source_dim, y_source_dim], [x_source_dim, 0], [0,0]])
    return geometry_contained(offset_geom, bbox)

def translate_road_line(road_line, x, y):
    offset_geom = offset_pixel_coords(road_line.getGeometry("pixels"), x, y)
    road_line.setGeometry(offset_geom, "pixels")
    return road_line

def draw_objects_on_mask(objects_to_draw,
                         x_offset,
                         y_offset,
                         x_source_dim,
                         y_source_dim,
                         x_target_dim,
                         y_target_dim,
                         channels=3,
                         output_format="tensor",
                         geometry_accessor=lambda x: x.getGeometry("pixels"),
                         color_accessor=lambda x:1,
                         initial_mask=None):

    # Determine the dimensions of the thing we are attempting to draw
    if channels > 1:
        argument_dimension = (x_target_dim, y_target_dim, channels)
    else:
        argument_dimension = (x_target_dim, y_target_dim)

    # If we have been passed an initial mask, verify it is the correct dimensions, and then create an image object from it
    # If we havent been passed an initial mask, then create an empty image to work with
    if not initial_mask is None:
        if initial_mask.shape != argument_dimension:
            raise ValueError("Cannot use initial mask with dim " + initial_mask.shape + " with argument dimensions of " + str(argument_dimension))
        debug_image = Image.fromarray(initial_mask)
    else:
        debug_image = Image.fromarray(np.zeros(argument_dimension, dtype=np.uint8))
    debug_image_draw = ImageDraw.Draw(debug_image)

    # For every polygon we have
    for object_to_draw in objects_to_draw:

        # Offset using the passed args
        geom = geometry_accessor(object_to_draw)
        s_offset = offset_pixel_coords(geom, x_offset, y_offset)

        # Scale the pixel coordinates as needed
        s_scaled = scale_pixel_coords(s_offset, x_target_dim/x_source_dim, y_target_dim/y_source_dim, origin=(0,0))

        # If the object is in the frame...
        if geometry_in_frame(s_scaled, 0, 0, x_target_dim, y_target_dim):
            # Draw the polygon on the image using the defined color map

            color = color_accessor(object_to_draw)
            debug_image_draw.polygon(s_scaled.exterior.coords, fill=color)

            interior_fill = tuple([0]*channels) if channels > 1 else 0
            for interior in s_scaled.interiors:
                debug_image_draw.polygon(interior.coords, fill=interior_fill)
        else:
            print("\tSkipping Geom - NOT IN FRAME")
            print("\tInitial Geom", geom)
            print("\tOffset Geom", s_offset)
            print("\tScaled Geom", s_scaled)

    # Return the result in the output_format we care about...
    if output_format == "image":
        return debug_image
    return np.asarray(debug_image)

def get_color_accessor(class_color_map, draw_color):
    color_accessor = None
    if draw_color:
        color_accessor = get_spatial_object_tuple_color_accessor(class_color_map)
    else:
        color_accessor = get_spatial_object_value_color_accessor(class_color_map)
    return color_accessor

def get_spatial_object_value_color_accessor(class_color_map):
    def color_val_accessor(spatial_obj):
        return class_color_map.getIndex(spatial_obj.getLabel())
    return color_val_accessor

def get_spatial_object_tuple_color_accessor(class_color_map):
    def color_tuple_accessor(spatial_obj):
        return tuple(class_color_map.getColor(spatial_obj.getLabel())[3:])
    return color_tuple_accessor


def get_adjustment_geom(adj, adjustment_width_max, adjustment_height_to_width_ratio):
    adj_line = adj.getGeometry().buffer(min(adjustment_width_max, adj.getMagnitude()*adjustment_height_to_width_ratio), cap_style="flat")
    adj_centroid = adj.getStartPoint().buffer(adjustment_width_max*2)
    return adj_line.union(adj_centroid)

def draw_adjustments_on_mask(adjustments,
                             x_offset,
                             y_offset,
                             x_source_dim,
                             y_source_dim,
                             x_target_dim,
                             y_target_dim,
                             draw_color=False,
                             output_format="tensor",
                             adjustment_height_to_width_ratio=0.25,
                             adjustment_width_max=2,
                             initial_mask=None):
    return draw_objects_on_mask(adjustments,
                                x_offset,
                                y_offset,
                                x_source_dim,
                                y_source_dim,
                                x_target_dim,
                                y_target_dim,
                                channels=3 if draw_color else 1,
                                output_format=output_format,
                                geometry_accessor=lambda x: get_adjustment_geom(x, adjustment_width_max, adjustment_height_to_width_ratio),
                                initial_mask=initial_mask)

def draw_buildings_on_mask(buildings,
                           x_offset,
                           y_offset,
                           x_source_dim,
                           y_source_dim,
                           x_target_dim,
                           y_target_dim,
                           class_color_map,
                           draw_color=False,
                           output_format="tensor",
                           initial_mask=None):
    return draw_objects_on_mask(buildings,
                                x_offset,
                                y_offset,
                                x_source_dim,
                                y_source_dim,
                                x_target_dim,
                                y_target_dim,
                                channels=3 if draw_color else 1,
                                output_format=output_format,
                                geometry_accessor=lambda x: x.getGeometry("pixels"),
                                color_accessor=get_color_accessor(class_color_map, draw_color),
                                initial_mask=initial_mask)

def draw_road_lines_on_mask(road_lines,
                            x_offset,
                            y_offset,
                            x_source_dim,
                            y_source_dim,
                            x_target_dim,
                            y_target_dim,
                            class_color_map,
                            draw_color=False,
                            output_format="tensor",
                            road_width_pixels=40,
                            initial_mask=None):
    return draw_objects_on_mask(road_lines,
                                x_offset,
                                y_offset,
                                x_source_dim,
                                y_source_dim,
                                x_target_dim,
                                y_target_dim,
                                channels=3 if draw_color else 1,
                                output_format=output_format,
                                geometry_accessor=lambda x: x.getGeometry("pixels").buffer(road_width_pixels, cap_style="flat"),
                                color_accessor=get_color_accessor(class_color_map, draw_color),
                                initial_mask=initial_mask)

def draw_labeled_road_lines_on_mask(labeled_road_lines,
                                    x_offset,
                                    y_offset,
                                    x_source_dim,
                                    y_source_dim,
                                    x_target_dim,
                                    y_target_dim,
                                    class_color_map,
                                    draw_color=False,
                                    output_format="tensor",
                                    road_width_pixels=40,
                                    initial_mask=None):
    labeled_road_lines_to_draw = []
    default_label = None

    #Get the default label which will be used to sort the label render order
    if len(labeled_road_lines) > 0:
        default_label = labeled_road_lines[0].getLabel()

    #Get the buffered road lines for all of the main road lines themselves
    labeled_road_lines_to_draw.extend(labeled_road_lines)

    #For every labeled road line we have
    for labeled_road_line in labeled_road_lines:

        #Get the sub lines we want to draw...
        labeled_road_lines_to_draw.extend(labeled_road_line.get_labeled_sub_lines())

    #Sort the spatial objects so they are in the correct render order and labels are drawn on top of road lines.
    #Drawing the labels in this way effectively breaks the multiclass paradigm
    labeled_road_lines_to_draw.sort(key=lambda x: 0 if x.getLabel() == default_label else 1)

    return draw_objects_on_mask(labeled_road_lines_to_draw,
                                x_offset,
                                y_offset,
                                x_source_dim,
                                y_source_dim,
                                x_target_dim,
                                y_target_dim,
                                channels=3 if draw_color else 1,
                                output_format=output_format,
                                geometry_accessor=lambda x: x.getGeometry("pixels").buffer(road_width_pixels, cap_style="flat"),
                                color_accessor=get_color_accessor(class_color_map, draw_color),
                                initial_mask=initial_mask)

#This function considers all of the polygons that have been passed
#This function looks for all the polygons in a box of size x_dim and y_dim who's center point is at the passed x, y
def get_valid_buildings(x, y, buildings, x_dim=2048, y_dim=2048, building_intersection_proportion_threshold=0.925, exceptions_to_track=None, center_xy=True):
    window = None
    if center_xy:
        window = shapely.Polygon([(x-x_dim/2, y-y_dim/2), (x+x_dim/2, y-y_dim/2), (x+x_dim/2, y+y_dim/2), (x-x_dim/2, y+y_dim/2), (x-x_dim/2, y-y_dim/2)])
    else:
        window = shapely.Polygon([(x, y), (x+x_dim, y), (x+x_dim, y+y_dim), (x, y+y_dim), (x, y)])

    valid_buildings = []
    for building in buildings:
        try:
            if shapely.intersects(window, building.getGeometry("pixels")):
                intersection_polygon = shapely.intersection(window, building.getGeometry("pixels"))
                iou = intersection_polygon.area/building.getGeometry("pixels").area
                if iou > building_intersection_proportion_threshold:
                    valid_buildings.append(building)
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            if (not exceptions_to_track is None) and (str(e.__class__.__name__) in exceptions_to_track.keys()):
                exceptions_to_track[str(e.__class__.__name__)] += 1
            else:
                print("Error encountered in building polygon bounding box intersection...")
                print(str(e.__class__.__name__), str(e))

    return valid_buildings, exceptions_to_track

def get_valid_lines(x, y, road_lines, x_dim=2048, y_dim=2048, exceptions_to_track=None, center_xy=False):
    window = None
    if center_xy:
        window = shapely.Polygon([(x-x_dim/2, y-y_dim/2), (x+x_dim/2, y-y_dim/2), (x+x_dim/2, y+y_dim/2), (x-x_dim/2, y+y_dim/2), (x-x_dim/2, y-y_dim/2)])
    else:
        window = shapely.Polygon([(x, y), (x+x_dim, y), (x+x_dim, y+y_dim), (x, y+y_dim), (x, y)])

    results = []
    for line in road_lines:
        try:
            if shapely.intersects(window, line.getGeometry("pixels")):
                results.append(line)
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            if (not exceptions_to_track is None) and (str(e.__class__.__name__) in exceptions_to_track.keys()):
                exceptions_to_track[str(e.__class__.__name__)] += 1
            else:
                print("Error encountered in road line bounding box intersection...")
                print(str(e.__class__.__name__), str(e))

    return results, exceptions_to_track

def generate_sample_point(ortho, random_state):

    #Select a point in the ortho that is within the bounds of the imagery
    x = random_state.uniform(0, ortho.get_width())
    y = random_state.uniform(0, ortho.get_height())

    return x, y
