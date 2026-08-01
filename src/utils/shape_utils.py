import shapely
from shapely import Polygon, LineString


def convert_coords_to_shapely(polygon, shapely_type, accessor=lambda x: (x[0], x[1])):
	coords = []
	for v in polygon:
		coords.append(accessor(v))
	return shapely_type(coords)