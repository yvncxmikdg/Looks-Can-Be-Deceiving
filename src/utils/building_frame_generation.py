import os
import numpy as np
import PIL
from PIL import Image, ImageDraw
from shapely import Polygon

ID = "id"
NEIGHBORS = "neighbors"
POLYGON = "polygon"
CAPTURED = "captured"
FRAME = "frame"
BUILDINGS = "buildings"

#Compute the combined bounds of an arbitrary number of polygons
def _get_combined_bounds_from_polygons(*polygons):
    if not polygons:
        raise ValueError("At least one polygon must be provided")

    #Initialize bounds using the first polygon
    minx, miny, maxx, maxy = polygons[0].bounds

    #Expand bounds with each additional polygon
    for poly in polygons[1:]:
        px_min, py_min, px_max, py_max = poly.bounds
        minx = min(minx, px_min)
        miny = min(miny, py_min)
        maxx = max(maxx, px_max)
        maxy = max(maxy, py_max)

    #Return the bounds after they have been computed
    return minx, miny, maxx, maxy

#This function takes an uninitialized graph of the buildings in an orthomosaic and computes
#the buildings which have "neighbors" which are buildings that could reasonably be included
#in the same frame as the other building. The expectation is that this data  structure will
#be used for sample generation in the future.
def _init_building_neighbor_graph(uninitialized_buildings_graph, x_dim, y_dim):

    #Get the graph as a list of nodes that we will iterate through
    buildings_graph_list = list(uninitialized_buildings_graph.items())

    #For every building in the graph we will...
    for i, (building_source_id, building_source_node) in enumerate(buildings_graph_list):
        #...look at every other building in the graph that we havent already considered
        for building_target_id, building_target_node in buildings_graph_list[i+1:]:
            #We will then determine if these two buildings could be put in a frame together

            #Start by getting the bounds of each of these polygons
            min_x_combined, min_y_combined, max_x_combined, max_y_combined = _get_combined_bounds_from_polygons(building_source_node[POLYGON], \
                                                                                                                building_target_node[POLYGON])
            min_x_source, min_y_source, max_x_source, max_y_source = building_source_node[POLYGON].bounds
            min_x_target, min_y_target, max_x_target, max_y_target = building_target_node[POLYGON].bounds

            #Determining how many frames these buildings could be spread across
            num_frames_x_source = (max_x_source - min_x_source) // x_dim
            num_frames_y_source = (max_y_source - min_y_source) // y_dim
            num_frames_x_target = (max_x_target - min_x_target) // x_dim
            num_frames_y_target = (max_y_target - min_y_target) // y_dim
            num_frames_x_combined = (max_x_combined - min_x_combined) // x_dim
            num_frames_y_combined = (max_y_combined - min_y_combined) // y_dim

            #If the two buildings are arranged such that considering them together
            #does not increase the number of tiles then they are neighbors
            if num_frames_x_source == num_frames_x_target == num_frames_x_combined and \
                num_frames_y_source == num_frames_y_target == num_frames_y_combined:
                uninitialized_buildings_graph[building_source_id][NEIGHBORS].append(building_target_id)
                uninitialized_buildings_graph[building_target_id][NEIGHBORS].append(building_source_id)
    #After having initialized the graph, then we return it
    return uninitialized_buildings_graph

#This function takes a current building to start at, and an initialized buildings graph and it iterates through
#the graph recursively, and generates frames that contains the buildings that it can include in the frames where possible
def _recursively_generate_framed_buildings(current_building_id, buildings_graph, x_dim, y_dim):
    #Mark the current building as captured since we are looking at it now...
    buildings_graph[current_building_id][CAPTURED] = True

    #Create a list of frames that we will use to store the results
    resulting_frames = []

    #Get the bounds for the building polygon that we are looking at
    min_x_current, min_y_current, max_x_current, max_y_current = buildings_graph[current_building_id][POLYGON].bounds

    #Compute the number of tiles that this building will span...
    num_frames_x = (x_dim + (max_x_current - min_x_current))//x_dim
    num_frames_y = (y_dim + (max_y_current - min_y_current))//y_dim

    #If we are working with a case where the building is bigger than a frame could be...
    # pylint: disable-next=too-many-nested-blocks
    if num_frames_x > 1 or num_frames_y > 1:

        #Then we need to break up the building into many frames, so we do this by...
        #Getting the center point of the buildings bounding box
        center_x = (max_x_current - min_x_current)/2 + min_x_current
        center_y = (max_y_current - min_y_current)/2 + min_y_current

        start_x = center_x - (x_dim*num_frames_x)/2
        start_y = center_y - (y_dim*num_frames_y)/2

        #Then we break the building up into tiles centered on the middle of its bounding box
        for x in np.arange(start_x, start_x+num_frames_x*x_dim, x_dim):
            for y in np.arange(start_y, start_y+num_frames_y*y_dim, y_dim):
                #We get the frame for the part of the building we are going to be looking at
                frame = Polygon([[x, y],
                                 [x+x_dim, y],
                                 [x+x_dim, y+y_dim],
                                 [x, y+y_dim],
                                 [x, y]])
                #Make sure the building still has a part of it in the frame (this is to deal with L shaped, or buildings with other weird geometry)
                if buildings_graph[current_building_id][POLYGON].intersection(frame).area > 0:
                    valid_buildings = [buildings_graph[current_building_id]]
                    #We then take the building, and look at all of the neighbors.
                    for neighbor_id in buildings_graph[current_building_id][NEIGHBORS]:
                        #If the neighbor is in frame, and it has not already been included in another frame...
                        if frame.contains(buildings_graph[neighbor_id][POLYGON]) and not buildings_graph[neighbor_id][CAPTURED]:
                            #Then we add it to the list of buildings to be included in this sample.
                            valid_buildings.append(buildings_graph[neighbor_id])
                            buildings_graph[neighbor_id][CAPTURED] = True
                    #And we store the frame, along with the buildings that should be queried
                    resulting_frames.append({FRAME:frame, BUILDINGS:valid_buildings})
    else:
        #If we can fit the current polygon in a single tile
        working_polygons = [buildings_graph[current_building_id]]

        #Look at the neighnors and see if there are any neighbors that we can add to the frame
        for neighbor_id in buildings_graph[current_building_id][NEIGHBORS]:

            #If the current neighbor hasnt already been captured in a frame
            if not buildings_graph[neighbor_id][CAPTURED]:
                min_x, min_y, max_x, max_y = _get_combined_bounds_from_polygons(*[b[POLYGON] for b in working_polygons], buildings_graph[neighbor_id][POLYGON])

                #If we can add the building without going outside the bounds of the frame, then we cam add it to the list of polygons in frame
                if max_x-min_x <= x_dim and max_y-min_y <= y_dim:
                    working_polygons.append(buildings_graph[neighbor_id])
                    buildings_graph[neighbor_id][CAPTURED] = True

        #Then we get the bounds of all the buildings we want to include
        min_x_combined, min_y_combined, max_x_combined, max_y_combined = _get_combined_bounds_from_polygons(*[b[POLYGON] for b in working_polygons])

        #Then we center the frame around all the buildings that we want to include
        center_x = (max_x_combined - min_x_combined)/2 + min_x_combined
        center_y = (max_y_combined - min_y_combined)/2 + min_y_combined
        x = center_x - x_dim/2
        y = center_y - y_dim/2

        #We make the frame
        frame = Polygon([[x, y],
                         [x+x_dim, y],
                         [x+x_dim, y+y_dim],
                         [x, y+y_dim],
                         [x, y]])

        #And we append it to the list we started with
        resulting_frames.append({FRAME:frame, BUILDINGS:working_polygons})

    #Now we recurse into the neighbors that we werent able to capture with the above logic
    #TODO: This could probably be improved by sorting the neighbors based on their distances to their nearest neighbors
    #   Meaning the one that is most likley to be able to include more than one building in a frame.
    for neighbor_id in buildings_graph[current_building_id][NEIGHBORS]:
        #If we find a neighbor that hasnt been captured, then we recurse into it
        if not buildings_graph[neighbor_id][CAPTURED]:
            #And we add the generated frames into the list we are going to return
            recurse_result = _recursively_generate_framed_buildings(neighbor_id, buildings_graph, x_dim, y_dim)
            resulting_frames.extend(recurse_result)

    return resulting_frames


def get_candidate_samples_center(orthomosaic, x_dim, y_dim, adjustment_buffer_distance_px, adjusted=False, debug=False, debug_output_dir="."):
    # Create a list to store the candidate samples and an empty dictionary to store their relationships
    generated_framed_buildings = []
    uninitialized_buildings_graph = {}

    # Iterate over the buildings and buffer their polygons by the amounts we care about
    for building in orthomosaic.get_buildings(adjusted=adjusted):
        adjustment_buffered_polygon = building.getGeometry("pixels").buffer(adjustment_buffer_distance_px)
        uninitialized_buildings_graph[building.getId()] = {ID:building.getId(), POLYGON:adjustment_buffered_polygon, NEIGHBORS:[], CAPTURED:False}

    #Construct the neighbor graph so we can check which building polygons to include
    buildings_graph = _init_building_neighbor_graph(uninitialized_buildings_graph, x_dim, y_dim)

    #Then iterate through all the buildings
    for building_id, building_graph_node in buildings_graph.items():
        #And if we dont have the building in a frame yet, then we recurse in and generate them
        if not building_graph_node[CAPTURED]:
            generated_framed_buildings.extend(_recursively_generate_framed_buildings(building_id, buildings_graph, x_dim, y_dim))

    #If debug, generate an inspection polygon
    if debug:
        generate_frame_polygon_inspection(generated_framed_buildings, os.path.join(debug_output_dir, "val_samples-" + str(orthomosaic.get_name()) + ".png"))

    #Finally, we return the frames and the associated building ids
    result = []
    for frame_with_buildings in generated_framed_buildings:
        result.append([frame_with_buildings[FRAME], [b[ID] for b in frame_with_buildings[BUILDINGS]]])
    return result

def draw_poly(polygon, draw, min_x, max_y, scale, padding, color):
    coords = list(polygon.exterior.coords)
    # Scale and shift coordinates to fit in image space
    scaled_coords = [
        (int((x - min_x) * scale) + padding,
         int((max_y - y) * scale) + padding)  # Flip Y-axis for image
        for x, y in coords
    ]
    draw.line(scaled_coords + [scaled_coords[0]], fill=color, width=1)

def generate_frame_polygon_inspection(framed_buildings, output_path):
    padding = 20
    scale = 0.25

    frames = [f[FRAME] for f in framed_buildings]
    building_groups = [f[BUILDINGS] for f in framed_buildings]

    # Compute bounding box for all polygons
    min_x = min(p.bounds[0] for p in frames)
    min_y = min(p.bounds[1] for p in frames)
    max_x = max(p.bounds[2] for p in frames)
    max_y = max(p.bounds[3] for p in frames)

    # Determine image dimensions with padding
    width = int((max_x - min_x) * scale) + 2 * padding
    height = int((max_y - min_y) * scale) + 2 * padding

    # Create a blank image
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for i, (frame, buildings) in enumerate(zip(frames, building_groups)):
        color = list(PIL.ImageColor.colormap.keys())[i%len(list(PIL.ImageColor.colormap.keys()))]
        draw_poly(frame, draw, min_x, max_y, scale, padding, color)
        for building in buildings:
            draw_poly(building[POLYGON], draw, min_x, max_y, scale, padding, color)

    img.save(output_path)
