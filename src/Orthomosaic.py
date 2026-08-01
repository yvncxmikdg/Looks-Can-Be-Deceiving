import os
import json

import rasterio
import shapely
import tables as tb
import pandas as pd
import numpy as np

from pyproj import Transformer, CRS
from shapely.geometry import Point, shape
from rasterio.transform import AffineTransformer, Affine
from rasterio.windows import Window

from modeling.constants import (
    LAT_LON_CRS,
    EPSG_PREFIX,
    TIMESTAMP_OFFSET_STRATEGIES,
    YEAR,
    MONTH,
)
from modeling.utils.file_management_utils import find_geotif_file_prefix_match
from modeling.Spatial import (
    BuildingFactory,
    RoadLineFactory,
    RoadAnnotationPolygonFactory,
    MultiLabeledRoadLineFactory,
)
from modeling.Alignment import AdjustmentVectorFieldFactory


# This function taskes an orthomosaic and a list of orthomosaic names that should be removed from the
# passed list of orthomosaics. It returns a list of orthomosaics without those that have the passed names
def remove_orthomosaic_from_list_by_name(orthomosaics, names_to_remove, verbose=True):
    result = []
    for ortho in orthomosaics:
        if not ortho.get_name() in names_to_remove:
            result.append(ortho)
        elif verbose:
            print("Removing orthomosaic with name", ortho.get_name())
    return result


class Orthomosaic:  # pylint: disable=too-many-public-methods
    def __init__(
        self,
        name,
        source_rasterio_path=None,
        source_hdf5_path=None,
        transformer=None,
        epsg_integer=None,
        boundary_polygon=None,
        buildings=None,
        road_lines=None,
        road_line_annotation_polygons=None,
        building_adjustment_vector_field=None,
        road_line_adjustment_vector_field=None,
        simple_boundary_tolerance=5,
        width=0,
        height=0,
        channels=0,
        backend="auto",
        mapper_used_for_generation=None,
        platform_used_for_collection=None,
        source_used_for_collection=None,
        is_train=False,
        gsd=None,
        scale_factor=1.0,
        timestamp=None,
        event_phase=None,
    ):

        self.__name = name
        self.__source_hdf5_path = source_hdf5_path
        self.__source_rasterio_path = source_rasterio_path
        self.__transformer = transformer
        self.__boundary_polygon = boundary_polygon
        self.__epsg_integer = epsg_integer
        self.__buildings = buildings
        self.__road_lines = road_lines
        self.__road_line_annotation_polygons = road_line_annotation_polygons
        self.__building_adjustment_vector_field = building_adjustment_vector_field
        self.__road_line_adjustment_vector_field = road_line_adjustment_vector_field
        self.__hdf5_file_handler = None
        self.__hdf5_file_handler_pid = None
        self.__width = width
        self.__height = height
        self.__channels = channels
        self.__mapper_used_for_generation = (
            None if pd.isna(mapper_used_for_generation) else mapper_used_for_generation
        )
        self.__platform_used_for_collection = (
            None
            if pd.isna(platform_used_for_collection)
            else platform_used_for_collection
        )
        self.__source_used_for_collection = (
            None if pd.isna(source_used_for_collection) else source_used_for_collection
        )
        self.__is_train = None if pd.isna(is_train) else is_train
        # "PRE" / "POST" (normalized) from statistics.csv "Pre/Post Event"; None when unknown. The
        # CHANGE task uses this to restrict its satellite anchor imagery to post-event captures.
        self.__event_phase = None if pd.isna(event_phase) else str(event_phase).strip().upper()
        self.__gsd = gsd
        self.__scale_factor = scale_factor
        self.__timestamp = timestamp

        self.set_backend(backend)

        if self.__boundary_polygon:
            self.__simple_boundary_polygon = boundary_polygon.simplify(
                simple_boundary_tolerance
            )
        else:
            self.__simple_boundary_polygon = None

        self.__adjusted_road_lines = None
        if self.__road_line_adjustment_vector_field:
            self.__adjusted_road_lines = (
                self.__road_line_adjustment_vector_field.adjustRoadLines(
                    self.__road_lines
                )
            )

        self.__adjusted_buildings = None
        if self.__building_adjustment_vector_field:
            self.__adjusted_buildings = (
                self.__building_adjustment_vector_field.adjustBuildings(
                    self.__buildings
                )
            )

        self.__buildings_id_to_idx = {
            building.getId(): i for i, building in enumerate(self.__buildings)
        }

        self.__buildings_correlation_id_to_idx = {
            building.getCorrelationId(): i for i, building in enumerate(self.__buildings)
        }

        self.__road_lines_id_to_idx = {
            road_line.getId(): i for i, road_line in enumerate(self.__road_lines)
        }

    def open_hdf5_file(self):
        self.__hdf5_file_handler = tb.open_file(self.__source_hdf5_path, "r")
        self.__hdf5_file_handler_pid = os.getpid()
        return self.__hdf5_file_handler

    def read(self, x_start, y_start, x_dim, y_dim, backend=None, center_xy=True):
        if backend is None or backend == "auto":
            backend = self.__backend

        x_start_scaled = int(x_start * self.__scale_factor)
        y_start_scaled = int(y_start * self.__scale_factor)
        x_dim_scaled = int(x_dim * self.__scale_factor)
        y_dim_scaled = int(y_dim * self.__scale_factor)

        x_start_actual = int(
            x_start_scaled - x_dim_scaled / 2 if center_xy else x_start_scaled
        )
        y_start_actual = int(
            y_start_scaled - y_dim_scaled / 2 if center_xy else y_start_scaled
        )

        left_pad = min(x_dim_scaled, -1 * min(0, x_start_actual))
        right_pad = min(
            x_dim_scaled,
            -1 * min(0, self.get_width() - (x_start_actual + x_dim_scaled)),
        )
        bottom_pad = min(y_dim_scaled, -1 * min(0, y_start_actual))
        top_pad = min(
            y_dim_scaled,
            -1 * min(0, self.get_height() - (y_start_actual + y_dim_scaled)),
        )

        #If we need to pad more than the width of the image, then constrain the padding
        pad_dims = [
            [min(int(bottom_pad), y_dim_scaled), min(int(top_pad), y_dim_scaled)],
            [min(int(left_pad), x_dim_scaled), min(int(right_pad), x_dim_scaled)],
            [0, 0],
        ]

        x_start_offset = min(max(0, x_start_actual), self.get_width())
        y_start_offset = min(max(0, y_start_actual), self.get_height())

        x_dim_actual = x_dim_scaled - (x_start_offset - x_start_actual)
        y_dim_actual = y_dim_scaled - (y_start_offset - y_start_actual)

        x_dim_actual = max(
            0,
            x_dim_actual + min(0, (self.get_width() - (x_start_offset + x_dim_actual))),
        )
        y_dim_actual = max(
            0,
            y_dim_actual
            + min(0, (self.get_height() - (y_start_offset + y_dim_actual))),
        )

        if backend == "rasterio":
            # Initialize a list to store the color arrays
            rgb_array = []
            # Open the rasterio file, and get the file handler...
            with self.open_rasterio_file() as src:
                # Load the red channel
                r = src.read(
                    1,
                    window=Window(
                        x_start_offset, y_start_offset, x_dim_actual, y_dim_actual
                    ),
                )
                # Pad the image with 0s
                r = np.pad(
                    r, pad_width=pad_dims[:2], mode="constant", constant_values=0
                )
                rgb_array.append(r.reshape((x_dim_scaled, y_dim_scaled, 1)))
                r = None

                # Load the green channel
                g = src.read(
                    2,
                    window=Window(
                        x_start_offset, y_start_offset, x_dim_actual, y_dim_actual
                    ),
                )
                # Pad the image with 0s
                g = np.pad(
                    g, pad_width=pad_dims[:2], mode="constant", constant_values=0
                )
                rgb_array.append(g.reshape((x_dim_scaled, y_dim_scaled, 1)))
                g = None

                # Load the blue channel
                b = src.read(
                    3,
                    window=Window(
                        x_start_offset, y_start_offset, x_dim_actual, y_dim_actual
                    ),
                )
                # Pad the image with 0s
                b = np.pad(
                    b, pad_width=pad_dims[:2], mode="constant", constant_values=0
                )
                rgb_array.append(b.reshape((x_dim_scaled, y_dim_scaled, 1)))
                b = None

            # Stack them together into a tensor
            color_data = np.stack(rgb_array, axis=2)
            rgb_array = None
            color_data = color_data.reshape(
                color_data.shape[0], color_data.shape[1], color_data.shape[2]
            )
            return color_data

        if backend == "hdf5":
            f = self.get_hdf5_file_handler(open_hdf5_file=True)
            color_data = f.root.pixel_data.rgb[
                y_start_offset : y_start_offset + y_dim_actual,
                x_start_offset : x_start_offset + x_dim_actual,
                :,
            ]
            color_data.setflags(write=True)
            color_data = np.pad(
                color_data, pad_width=pad_dims, mode="constant", constant_values=0
            )
            return color_data

        raise ValueError(
            'No backend valid was set before reading was attempted. Options are: "rasterio" and "hdf5". Found: '
            + str(backend)
        )

    def close_hdf5_file(self):
        if self.__hdf5_file_handler is not None:
            self.__hdf5_file_handler.close()
        self.__hdf5_file_handler = None
        self.__hdf5_file_handler_pid = None

    def get_width(self):
        return int(self.__width)

    def get_height(self):
        return int(self.__height)

    def get_channels(self):
        return int(self.__channels)

    def get_mapper_used_for_generation(self):
        return self.__mapper_used_for_generation

    def get_platform_used_for_collection(self):
        return self.__platform_used_for_collection

    def get_source_used_for_collection(self):
        return self.__source_used_for_collection

    def get_event_phase(self):
        return self.__event_phase

    def is_train(self):
        return self.__is_train

    def get_gsd(self):
        return self.__gsd

    def get_timestamp(self):
        return self.__timestamp

    def set_backend(self, backend):
        if backend == "auto":
            if self.__source_hdf5_path:
                self.__backend = "hdf5"
            elif self.__source_rasterio_path:
                self.__backend = "rasterio"
            else:
                self.__backend = None
        elif backend == "hdf5" and self.__source_hdf5_path is None:
            raise ValueError(
                "Unable to set backend to hdf5 because there is no hdf5 file supplied for this orthomosaic ("
                + self.get_name()
                + ")"
            )
        elif backend == "rasterio" and self.__source_rasterio_path is None:
            raise ValueError(
                "Unable to set backend to rasterio because there is no rasterio file supplied for this orthomosaic ("
                + self.get_name()
                + ")"
            )
        elif backend == "rasterio" or backend == "hdf5" or backend is None:
            self.__backend = backend
        else:
            raise ValueError(
                "Unable to set backend to "
                + str(backend)
                + ' as that backend is not supported. Options are: "rasterio" and "hdf5"'
            )

    def open_rasterio_file(self):
        return rasterio.open(self.__source_rasterio_path, "r")

    def contains(self, x, y):
        return self.__boundary_polygon.contains(Point([x, y]))

    def contains_simple(self, x, y):
        return self.__simple_boundary_polygon.contains(Point([x, y]))

    def rowcol(self, x, y, coord_system=LAT_LON_CRS):
        coord_transformer = Transformer.from_crs(
            coord_system, EPSG_PREFIX + str(self.__epsg_integer)
        )
        x_t, y_t = coord_transformer.transform(
            y, x
        )  # This swap of x and y is intentional
        x_p, y_p = self.__transformer.rowcol(x_t, y_t)
        return x_p, y_p

    def xy(self, x_p, y_p, coord_system=LAT_LON_CRS):
        x_t, y_t = self.__transformer.xy(x_p, y_p)

        coord_transformer = Transformer.from_crs(
            EPSG_PREFIX + str(self.__epsg_integer), coord_system
        )
        y, x = coord_transformer.transform(
            x_t, y_t
        )  # This swap of x and y is intentional

        return x, y

    def get_hdf5_file_handler(self, open_hdf5_file=False):
        # Open the hdf5 file lazily once per process and reuse the handle. Re-opening on every
        # read (the previous behaviour) leaked the prior pytables File: pytables holds every open
        # file in the global tables.file._open_files registry until it is explicitly closed, so
        # dropping the Python reference does NOT free it. Over a long run this leaks one handle
        # (plus its HDF5 chunk cache / BLOSC buffers) per sample read across every DataLoader
        # worker, exhausting host RAM and file descriptors -> the OOM seen on the cluster.
        #
        # The pid guard keeps this fork-safe: a handle opened before a DataLoader worker is forked
        # must not be reused in the child (a shared HDF5 file descriptor across a fork corrupts
        # reads), so each process opens its own handle the first time it reads.
        if open_hdf5_file and (self.__hdf5_file_handler is None or self.__hdf5_file_handler_pid != os.getpid()):
            self.open_hdf5_file()
        return self.__hdf5_file_handler

    def get_name(self):
        return self.__name

    def get_buildings(self, ids=None, adjusted=False, correlation_ids=None):
        selected_list = None
        if adjusted:
            selected_list = self.__adjusted_buildings
        else:
            selected_list = self.__buildings

        if ids is None and correlation_ids is None:
            return selected_list
        if ids is not None and correlation_ids is not None:
            raise ValueError("Must provide either ids or correlation_ids or neither")
        # An empty id list means "no buildings requested" -> return []. Only when buildings are
        # actually requested from an ortho that has none do we report None.
        if selected_list is None:
            return None if (ids or correlation_ids) else []
        if ids is not None:
            return [selected_list[self.__buildings_id_to_idx[id]] for id in ids]
        # Only correlation_ids remain: earlier branches handled the None/both cases.
        return [selected_list[self.__buildings_correlation_id_to_idx[correlation_id]] for correlation_id in correlation_ids]

    def get_road_lines(self, ids=None, adjusted=False):
        selected_list = None
        if adjusted:
            selected_list = self.__adjusted_road_lines
        else:
            selected_list = self.__road_lines

        if ids is None:
            return selected_list
        return [selected_list[self.__road_lines_id_to_idx[id]] for id in ids]

    def get_road_line_annotation_polygons(self):
        return self.__road_line_annotation_polygons

    def get_annotated_road_lines(self, ids=None, adjusted=False):
        responsive_road_lines = self.get_road_lines(ids, adjusted)
        return MultiLabeledRoadLineFactory(
            responsive_road_lines, self.get_road_line_annotation_polygons()
        )

    def get_building_count(self):
        return len(self.__buildings)

    def get_building_adjustment_vector_field(self):
        return self.__building_adjustment_vector_field

    def get_road_line_adjustment_vector_field(self):
        return self.__road_line_adjustment_vector_field

    def get_source_hdf5_path(self):
        return self.__source_hdf5_path

    def bounds(self):
        return self.__boundary_polygon.bounds

    def get_epsg(self):
        return self.__epsg_integer


def MultisourceOrthomosaicFactory(
    dataset_paths_dict,
    data_source_config_parameters,
    model_hyperparameters,
    boundary_folder=None,
    statistics_file_path=None,
    train_validation_test="train",
    limit=None,
    backend="rasterio",
    fail_on_error=True,
    warnings=True,
    default_epsg_int=None,
    scale_factor=1.0,
    required_channels=None,
):
    # The statistics csv can either be passed explicitly or referenced by the dataset paths yaml
    # under the "statistics" key; an explicit path wins so callers can still override it.
    if statistics_file_path is None:
        statistics_file_path = dataset_paths_dict.get("statistics", None)

    resulting_orthos = []
    for source_name in data_source_config_parameters[train_validation_test]:
        if data_source_config_parameters[train_validation_test][source_name]:

            bda_labels_path = None
            bda_adjustments_path = None
            rda_labels_path = None
            rda_adjustments_path = None
            timeoffset_strategy = None
            if "BDA" in model_hyperparameters["task"] or "CHANGE" in model_hyperparameters["task"]:
                bda_labels_path = dataset_paths_dict[train_validation_test][
                    source_name
                ]["bda"]["annotations_path"]
                bda_adjustments_path = dataset_paths_dict[train_validation_test][
                    source_name
                ]["bda"]["adj_annotations_path"]
            if "RDA" in model_hyperparameters["task"]:
                rda_labels_path = dataset_paths_dict[train_validation_test][
                    source_name
                ]["rda"]["annotations_path"]
                rda_adjustments_path = dataset_paths_dict[train_validation_test][
                    source_name
                ]["rda"]["adj_annotations_path"]

            try:

                timeoffset_strategy = (
                    model_hyperparameters["encoder"]
                    if model_hyperparameters["encoder"] in TIMESTAMP_OFFSET_STRATEGIES
                    else None
                )
            except KeyError:
                pass

            initialized_orthos = OrthomosaicFactory(
                orthomosaic_folder=dataset_paths_dict[train_validation_test][
                    source_name
                ]["imagery_path"],
                bda_annotation_folder=bda_labels_path,
                rda_annotation_folder=rda_labels_path,
                bda_adj_annotation_folder=bda_adjustments_path,
                rda_adj_annotation_folder=rda_adjustments_path,
                boundary_folder=boundary_folder,
                table_folder=None,
                statistics_file_path=statistics_file_path,
                backend=backend,
                limit=limit,
                fail_on_error=fail_on_error,
                warnings=warnings,
                default_epsg_int=default_epsg_int,
                scale_factor=scale_factor,
                timestamp_offset_strategy=timeoffset_strategy,
            )

            for ortho in initialized_orthos:
                if (
                    required_channels is None
                    or ortho.get_channels() in required_channels
                ):
                    resulting_orthos.append(ortho)
                else:
                    print(
                        "Skipping orthomosaic with name",
                        ortho.get_name(),
                        "because it had",
                        ortho.get_channels(),
                        "channels instead of the required",
                        required_channels,
                        "channels",
                    )
    return resulting_orthos


# pylint: disable=too-many-branches
def OrthomosaicFactory(
    orthomosaic_folder=None,
    bda_annotation_folder=None,
    rda_annotation_folder=None,
    bda_adj_annotation_folder=None,
    rda_adj_annotation_folder=None,
    boundary_folder=None,
    table_folder=None,
    statistics_file_path=None,
    backend="auto",
    limit=None,
    fail_on_error=True,
    warnings=True,
    default_epsg_int=None,
    scale_factor=1.0,
    timestamp_offset_strategy=None,
):
    oms = []

    target_files = []
    if orthomosaic_folder:
        orthomosaic_folder_files = os.listdir(orthomosaic_folder)
        target_files = []
        for file in orthomosaic_folder_files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                target_files.append(file)
    elif bda_annotation_folder:
        target_files = os.listdir(bda_annotation_folder)
    elif rda_annotation_folder:
        target_files = os.listdir(rda_annotation_folder)
    elif bda_adj_annotation_folder:
        target_files = os.listdir(bda_adj_annotation_folder)
    elif rda_adj_annotation_folder:
        target_files = os.listdir(rda_adj_annotation_folder)
    elif boundary_folder:
        target_files = os.listdir(boundary_folder)
    elif table_folder:
        target_files = os.listdir(table_folder)

    orthomosaic_stats = None
    if statistics_file_path:
        orthomosaic_stats = pd.read_csv(
            statistics_file_path, header=0, index_col="Orthomosaic"
        )

    for target_file in target_files:  # pylint: disable=too-many-nested-blocks
        geotif_name = target_file.replace(".json", "")
        try:
            source_used_for_collection = (
                None
                if orthomosaic_stats is None
                else orthomosaic_stats.loc[geotif_name]["Source"]
            )
            if limit is None or len(oms) < limit:
                print("Loading orthomosaic: ", geotif_name)
                orthomosaic_file_path = None
                buildings = []
                road_lines = []
                road_line_annotation_polygons = []
                bda_adj_vector_field = None
                rda_adj_vector_field = None
                polygon_boundary = []
                table_path = None
                transformer = None
                epsg_integer = None
                tb_affine_matrix = None
                tb_epsg_integer = None
                rio_epsg_integer = None
                rio_affine_matrix = None
                affine_matrix = None
                width, height, channels = 0, 0, 0
                rio_width, rio_height, rio_channels = 0, 0, 0
                tb_width, tb_height, tb_channels = 0, 0, 0
                gsd_x, gsd_y = None, None
                timestamp = None

                if orthomosaic_folder:
                    orthomosaic_file = find_geotif_file_prefix_match(
                        target_file, os.listdir(orthomosaic_folder)
                    )
                    orthomosaic_file_path = os.path.join(
                        orthomosaic_folder, orthomosaic_file
                    )

                    raster_data = rasterio.open(orthomosaic_file_path, "r")
                    transform = raster_data.transform
                    rio_affine_matrix = [transform[i] for i in range(0, 6)]
                    if raster_data.crs:
                        crs = CRS.from_wkt(raster_data.crs.wkt)
                        rio_epsg_integer = crs.to_epsg()
                    else:
                        rio_epsg_integer = default_epsg_int
                    rio_width = raster_data.width
                    rio_height = raster_data.height
                    rio_channels = raster_data.count
                    raster_data.close()
                # Attempt to load the bda annotations data
                if bda_annotation_folder:
                    buildings_file = find_geotif_file_prefix_match(
                        target_file, os.listdir(bda_annotation_folder)
                    )
                    # Without this guard a missing match reaches os.path.join and raises a confusing TypeError
                    # ("join() argument ... not 'NoneType'"); say what actually happened.
                    if buildings_file is None:
                        print(
                            "No BDA annotation file matches",
                            target_file,
                            "in",
                            bda_annotation_folder,
                            "-- continuing without buildings.",
                        )
                    else:
                        try:
                            with open(
                                os.path.join(bda_annotation_folder, buildings_file), "r"
                            ) as f:

                                buildings = BuildingFactory(json.load(f), source_used_for_collection)
                        except Exception as e:  # pylint: disable=broad-except
                            print(
                                "Exception loading BDA Annotations for",
                                target_file,
                                "Continuing without them. Error:",
                                e,
                            )

                # Attempt to load the rda annotations  data
                if rda_annotation_folder:
                    rda_file = find_geotif_file_prefix_match(
                        target_file, os.listdir(rda_annotation_folder)
                    )
                    try:
                        with open(
                            os.path.join(rda_annotation_folder, rda_file), "r"
                        ) as f:
                            rda_annotations = json.load(f)
                            road_lines = RoadLineFactory(rda_annotations["road_lines"])
                            road_line_annotation_polygons = (
                                RoadAnnotationPolygonFactory(
                                    rda_annotations["polygons"]
                                )
                            )
                    except Exception as e:  # pylint: disable=broad-except
                        print(
                            "Exception loading RDA Annotations for",
                            target_file,
                            "Continuing without them. Error:",
                            e,
                        )

                # Attempt to load the bda adj annotations data
                if bda_adj_annotation_folder:
                    bda_adj_vector_field_file = find_geotif_file_prefix_match(
                        target_file, os.listdir(bda_adj_annotation_folder)
                    )
                    if bda_adj_vector_field_file is None:
                        print(
                            "No BDA adjustment annotation file matches",
                            target_file,
                            "in",
                            bda_adj_annotation_folder,
                            "-- continuing without adjustments.",
                        )
                    else:
                        try:
                            with open(
                                os.path.join(
                                    bda_adj_annotation_folder, bda_adj_vector_field_file
                                ),
                                "r",
                            ) as f:
                                bda_adj_vector_field = AdjustmentVectorFieldFactory(
                                    json.load(f)
                                )
                        except Exception as e:  # pylint: disable=broad-except
                            print(
                                "Exception loading BDA Adjustment Annotations for",
                                target_file,
                                "Continuing without them. Error:",
                                e,
                            )

                # Attempt to load the rda adj annotations data
                if rda_adj_annotation_folder:
                    rda_adj_vector_field_file = find_geotif_file_prefix_match(
                        target_file, os.listdir(rda_adj_annotation_folder)
                    )
                    try:
                        with open(
                            os.path.join(
                                rda_adj_annotation_folder, rda_adj_vector_field_file
                            ),
                            "r",
                        ) as f:
                            rda_adj_vector_field = AdjustmentVectorFieldFactory(
                                json.load(f)
                            )

                    except Exception as e:  # pylint: disable=broad-except
                        print(
                            "Exception loading RDA Adjustment Annotations for",
                            target_file,
                            "Continuing without them. Error:",
                            e,
                        )

                # Attempt to load the boundary polygon, if there is a boundary polygon that is passed
                if boundary_folder:
                    boundary_file = find_geotif_file_prefix_match(
                        target_file, os.listdir(boundary_folder)
                    )
                    if not boundary_file is None:
                        with open(
                            os.path.join(boundary_folder, boundary_file), "r"
                        ) as f:
                            boundary_data = json.load(f)
                        polygon_boundaries = []
                        for boundary in boundary_data:
                            polygon_boundaries.append(shape(boundary["geometry"]))
                        polygon_boundary = shapely.MultiPolygon(polygon_boundaries)

                # Attempt to load the table if there is a table that is passed
                if table_folder:
                    table = find_geotif_file_prefix_match(
                        target_file, os.listdir(table_folder)
                    )
                    if table:
                        table_path = os.path.join(table_folder, table)

                        f = tb.open_file(table_path, "r")
                        tb_height, tb_width, tb_channels = f.root.pixel_data.rgb.shape
                        tb_affine_matrix = f.root.meta.AffineTransform.read()
                        tb_epsg_integer = f.root.meta.OrthoInfo[0][2]
                        f.close()
                    else:
                        if warnings:
                            print(
                                "Warning, a table folder was passed, but no table was found for",
                                target_file,
                            )

                # Compare the affine matrices that are loaded between the HDF5 File and the rasterio file.
                # Raise a warning if there are any differences, since they should be the same.
                if rio_affine_matrix is None and tb_affine_matrix is None:
                    affine_matrix = None
                elif rio_affine_matrix is None and not tb_affine_matrix is None:
                    affine_matrix = tb_affine_matrix
                elif tb_affine_matrix is None and not rio_affine_matrix is None:
                    affine_matrix = rio_affine_matrix
                elif any(
                    float(a) != float(b)
                    for a, b in zip(list(tb_affine_matrix), list(rio_affine_matrix))
                ):
                    if warnings:
                        print(
                            "Warning, affine transform loaded from hdf5 is different from affine transform loaded from the geotiff."
                        )
                        print("Defaulting to the rasterio affine transform")
                        print("HDF5 Affine Matrix:    ", tb_affine_matrix)
                        print("Rasterio Affine Matrix:", rio_affine_matrix)
                    affine_matrix = rio_affine_matrix

                # Compare the affine matrices that are loaded between the HDF5 File and the rasterio file.
                # Raise a warning if there are any differences, since they should be the same.
                if rio_epsg_integer is None and tb_epsg_integer is None:
                    epsg_integer = None
                elif rio_epsg_integer is None and not tb_epsg_integer is None:
                    epsg_integer = tb_epsg_integer
                elif tb_epsg_integer is None and not rio_epsg_integer is None:
                    epsg_integer = rio_epsg_integer
                elif int(tb_epsg_integer) != int(rio_epsg_integer):
                    if warnings:
                        print(
                            "Warning, EPSG Integer loaded from hdf5 is different from the EPSG Integer loaded from the geotiff."
                        )
                        print("Defaulting to the rasterio EPSG Integer")
                        print("HDF5 EPSG Integer:    ", tb_epsg_integer)
                        print("Rasterio EPSG Integer:", rio_epsg_integer)
                    epsg_integer = rio_epsg_integer

                if tb_width != 0 and rio_width != 0 and tb_width == rio_width:
                    width = rio_width
                elif tb_width == 0 and rio_width != 0:
                    width = rio_width
                elif tb_width != 0 and rio_width == 0:
                    width = tb_width
                elif tb_width != rio_width:
                    if warnings:
                        print(
                            "Warning, data width loaded from hdf5 is different from the data width loaded from the geotiff."
                        )
                        print("Defaulting to the rasterio data width")
                        print("HDF5 data width:    ", tb_width)
                        print("Rasterio data width:", rio_width)
                    width = rio_width

                if tb_height != 0 and rio_height != 0 and tb_height == rio_height:
                    height = rio_height
                elif tb_height == 0 and rio_height != 0:
                    height = rio_height
                elif tb_height != 0 and rio_height == 0:
                    height = tb_height
                elif tb_height != rio_height:
                    if warnings:
                        print(
                            "Warning, data height loaded from hdf5 is different from the data height loaded from the geotiff."
                        )
                        print("Defaulting to the rasterio data height")
                        print("HDF5 data height:    ", tb_height)
                        print("Rasterio data height:", rio_height)
                    height = rio_height

                if (
                    tb_channels != 0
                    and rio_channels != 0
                    and tb_channels == rio_channels
                ):
                    channels = rio_channels
                elif tb_channels == 0 and rio_channels != 0:
                    channels = rio_channels
                elif tb_channels != 0 and rio_channels == 0:
                    channels = tb_channels
                elif tb_channels != rio_channels:
                    if warnings:
                        print(
                            "Warning, data channels loaded from hdf5 is different from the data channels loaded from the geotiff."
                        )
                        print("Defaulting to the smaller of the two")
                        print("HDF5 data channels:    ", tb_channels)
                        print("Rasterio data channels:", rio_channels)
                    channels = min(rio_channels, tb_channels)

                # Load the affine transformer and GSD if possible
                if affine_matrix:
                    scaled_affine_matrix = Affine(
                        affine_matrix[0] * scale_factor,  # scale x GSD
                        affine_matrix[1],
                        affine_matrix[2],
                        affine_matrix[3],
                        affine_matrix[4] * scale_factor,  # scale y GSD
                        affine_matrix[5],
                    )
                    transformer = AffineTransformer(scaled_affine_matrix)
                    gsd_x = scaled_affine_matrix[0]
                    gsd_y = -scaled_affine_matrix[4]

                # The statistics csv is the authoritative GSD source when present - some
                # orthomosaics carry affine transforms whose scale does not match the
                # surveyed GSD, so stats override the affine-derived values. The stats
                # file also carries the survey timestamp used by temporal models (SatMAE).
                if not orthomosaic_stats is None:
                    stats_gsd = float(orthomosaic_stats.loc[geotif_name]["GSD (m/px)"])
                    gsd_x = stats_gsd * scale_factor
                    gsd_y = stats_gsd * scale_factor

                    # Extract timestamp for temporal models (e.g. SatMAE)
                    ortho_timestamp = str(
                        orthomosaic_stats.loc[geotif_name]["Date (mm/dd/yyy)"]
                    )
                    month, *_, year = ortho_timestamp.split("/")
                    year = int(year)
                    month = int(month)
                    hour = 14  # TODO: WE ASSUME 2PM LOCAL TIME FOR ALL ORTHOS, THIS NEEDS TO REFLECT THE ACTUAL TIME

                    # SATMAE applies an offset to the timestamps, so the timestamp should be offset similarly
                    if timestamp_offset_strategy:
                        # NOTE: year is indexed on the minimum year from the satmae paper
                        year -= TIMESTAMP_OFFSET_STRATEGIES[timestamp_offset_strategy][
                            YEAR
                        ]
                        # NOTE: month is indexed on 0, based on the satmae paper
                        month -= TIMESTAMP_OFFSET_STRATEGIES[timestamp_offset_strategy][
                            MONTH
                        ]

                    timestamp = np.array([year, month, hour])
                else:
                    print(
                        "Warning there is no timestamp for this ortho, this may cause temporal models to fail!"
                    )

                width = int(width / scale_factor)
                height = int(height / scale_factor)

                m = Orthomosaic(
                    name=target_file,
                    source_rasterio_path=orthomosaic_file_path,
                    source_hdf5_path=table_path,
                    transformer=transformer,
                    epsg_integer=epsg_integer,
                    boundary_polygon=polygon_boundary,
                    buildings=buildings,
                    road_lines=road_lines,
                    road_line_annotation_polygons=road_line_annotation_polygons,
                    building_adjustment_vector_field=bda_adj_vector_field,
                    road_line_adjustment_vector_field=rda_adj_vector_field,
                    width=width,
                    height=height,
                    channels=channels,
                    backend=backend,
                    mapper_used_for_generation=(
                        None
                        if orthomosaic_stats is None
                        else orthomosaic_stats.loc[geotif_name]["Mapper"]
                    ),
                    platform_used_for_collection=(
                        None
                        if orthomosaic_stats is None
                        else orthomosaic_stats.loc[geotif_name]["Platform / Provider"]
                    ),
                    source_used_for_collection=source_used_for_collection,
                    is_train=(
                        False
                        if orthomosaic_stats is None
                        else orthomosaic_stats.loc[geotif_name]["Train/Test"] == "Train"
                    ),
                    gsd=(gsd_x, gsd_y),
                    timestamp=timestamp,
                    event_phase=(
                        None
                        if orthomosaic_stats is None
                        else orthomosaic_stats.loc[geotif_name]["Pre/Post Event"]
                    ),
                )

                oms.append(m)

        except Exception as e:  # pylint: disable=broad-except
            if fail_on_error:
                raise e
            print("Skipping", table, "because of exception", e)

    return oms
