import shapely
import torch
import torch.nn.functional as F

from modeling.utils.data_augmentations import geoms_to_keypoints
from modeling.utils.alignment_utils import nearest_flow, normalize_coordinates
from modeling.utils.sample_generator_utils import scale_pixel_coords

# Create a 2D grid of coordinates for an image of given height and width
def create_image_grid(batch_size, height, width):
    y = torch.linspace(0, height - 1, height)
    x = torch.linspace(0, width - 1, width)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)  # Shape: (height, width, 2)
    return torch.stack([grid]*batch_size, dim=0)

def keypoints_to_geoms(keypoints, geoms):
    keypoints_list = keypoints.cpu().tolist()
    i = 0
    output_geoms = []
    for geom in geoms:
        verts_count = len(geom.exterior.coords)
        verts = keypoints_list[i:i+verts_count]
        output_geoms.append(shapely.Polygon(verts))
        i += verts_count
    return output_geoms

def diffeomorphic_composition_via_sequential_flows(vector_fields, output_grid_x_dim=-1, output_grid_y_dim=-1):

    batch_size, dims, height, width = vector_fields[0].shape

    # Create a temporary set of points that we will update
    combined_flow = torch.zeros([batch_size, dims, height, width]).to(vector_fields[0].get_device())
    current_points = create_image_grid(batch_size,
                                       height if output_grid_x_dim < 0 else output_grid_x_dim,
                                       width if output_grid_y_dim < 0 else output_grid_y_dim).to(vector_fields[0].get_device())

    # For each vectorfield we have
    for vf_i in vector_fields:

        # Get the dimensions of the vector fields we are working with here
        batch_size_i, dims_i, height_i, width_i = vf_i.shape
        try:
            assert batch_size == batch_size_i
            assert dims == dims_i
            assert height == height_i
            assert width == width_i
        except AssertionError as e:
            raise ValueError("Dimensions of current vector field does not equal the dimensions of the initial vector field") from e

        # Normalize the current points
        current_points_normalized = normalize_coordinates(current_points, height, width)

        # Compute the flow at that vector field
        flow = F.grid_sample(vf_i, current_points_normalized)
        flow_permuted = flow.permute(0,2,3,1)

        # Update the combined flow and the current points based on the flow observed at the new points
        current_points += flow_permuted
        combined_flow += flow

    # Return the combined flow
    return combined_flow

def warp_batched_polygons_according_to_vector_fields(batched_polygons, batched_vector_fields, source_dim_x, source_dim_y):
    # Create a list which will store the resulting polygons
    result = []

    # For each batch we have...
    for batch_idx, polygons in enumerate(batched_polygons):
        x_scale_factor = batched_vector_fields.shape[2]/source_dim_x
        y_scale_factor = batched_vector_fields.shape[3]/source_dim_y
        polygons_scaled = [scale_pixel_coords(p, x_scale_factor, y_scale_factor) for p in polygons]

        # Get the points in the polygon as a tensor
        points = torch.tensor(geoms_to_keypoints(polygons_scaled)).to(batched_vector_fields.device)
        if len(points) > 0:
            # Warp the points according to their respective flows
            warped_points = points + nearest_flow(batched_vector_fields[batch_idx, ...], points)
            # Convert the points back to geoms and add them to the result list
            warped_geoms = keypoints_to_geoms(warped_points, polygons)
            warped_geoms_scaled = [scale_pixel_coords(p, 1.0/x_scale_factor, 1.0/y_scale_factor) for p in warped_geoms]
            result.append(warped_geoms_scaled)
        else:
            result.append(polygons)

    # Return the result
    return result
