import warnings
import time

from copy import deepcopy
from itertools import chain

import numpy as np
import torch

from modeling.utils.sample_generator_utils import offset_pixel_coords, scale_pixel_coords
from modeling.Sample import Sample, View
from modeling.utils.alignment_utils import reconstruct_adjustments_from_unadjusted_adjusted_pairs
from modeling.utils.fourier_scale import fourier_scale, fourier_reconstruct

from modeling.constants import (
    SAMPLE_METADATA_ATTEMPTS,
    SAMPLE_METADATA_EXCEPTIONS,
    POLYGON_COUNT_PREFIX,
    PIXEL_COUNT_PREFIX,
    SAMPLE_GENERATION_TIMING_PREFIX
)

class WindowedDatasetAdaptor:
    def __init__(
        self,
        orthomosaics,
        label_map,
        sample_location_generation_strategy,
        keypoint_conversion_strategy,
        tile_x=2048,
        tile_y=2048,
        mask_x=None,
        mask_y=None,
        min_gsd_x=0.0,
        min_gsd_y=0.0,
        max_gsd_x=1e10,
        max_gsd_y=1e10,
        rescale_to_interval=False,
        multiscale_steps=1,
        multiscale_step_interval="linspace",
        mask_scale_strategy="pixel",
        tile_scale_strategy="pixel",
        backend="auto",
        rescale_strategy="interpolate",
        generate_adjusted_geoms=False
    ):
        self.orthomosaics = orthomosaics
        self.mask_x = mask_x
        self.mask_y = mask_y
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.min_gsd_x = min_gsd_x
        self.min_gsd_y = min_gsd_y
        self.max_gsd_x = max_gsd_x
        self.max_gsd_y = max_gsd_y
        self.rescale_to_interval = rescale_to_interval
        self.multiscale_steps = multiscale_steps
        self.multiscale_step_interval = multiscale_step_interval
        self.mask_scale_strategy = mask_scale_strategy
        self.tile_scale_strategy = tile_scale_strategy
        self.rescale_strategy = rescale_strategy
        self._backend = backend
        self.label_map = label_map
        self.generate_adjusted_geoms = generate_adjusted_geoms

        if self.multiscale_steps < 1:
            raise ValueError("Cannot have multiscale_steps < 1. Found " + str(self.multiscale_steps))
        if self.multiscale_steps > 1 and (self.min_gsd_x is None or self.min_gsd_y is None or self.max_gsd_x is None or self.max_gsd_y is None):
            raise ValueError("If multiscale_steps is > 1, max and min gsds must be defined for x and y.")

        if self.multiscale_step_interval.lower() == "logspace":
            self.multiscale_steps_gsd_x = np.logspace(np.log10(self.min_gsd_x), np.log10(self.max_gsd_x), num=self.multiscale_steps)
            self.multiscale_steps_gsd_y = np.logspace(np.log10(self.min_gsd_y), np.log10(self.max_gsd_y), num=self.multiscale_steps)
        elif self.multiscale_step_interval.lower() == "linspace":
            self.multiscale_steps_gsd_x = np.linspace(self.min_gsd_x, self.max_gsd_x, num=self.multiscale_steps)
            self.multiscale_steps_gsd_y = np.linspace(self.min_gsd_y, self.max_gsd_y, num=self.multiscale_steps)
        else:
            raise ValueError("Unknown multiscale_step_interval " + str(self.multiscale_step_interval) + " valid options are logspace and linspace.")

        if tile_scale_strategy == "pixel":
            if rescale_strategy != "none" and rescale_strategy is not None:
                raise ValueError("rescale_strategy must be None when set tile_scale_strategy is \"pixel\"")
            if rescale_strategy is None:
                self.rescale_strategy = "none"
        if mask_scale_strategy == "pixel":
            if mask_x is None or mask_y is None:
                raise ValueError("Must set fields mask_x and mask_y when mask_scale_strategy is \"pixel\"")
        if mask_scale_strategy == "spatial":
            if mask_x is not None or mask_y is not None:
                raise ValueError("Cannot set a positive mask value when mask_scale_strategy is set to \"spatial\". \
                                  The fields mask_x and mask_y must be omitted as they will be set dynamically.")

        self._sample_location_generation_strategy = sample_location_generation_strategy
        self.keypoint_conversion_strategy = keypoint_conversion_strategy

        self._sample_location_generation_strategy.initializeLocationGenerationStrategy(self.tile_x, self.tile_y, self.tile_scale_strategy)

        warnings.filterwarnings("ignore")

    def __len__(self) -> int:
        return len(self._sample_location_generation_strategy) * self.multiscale_steps

    def set_backend(self, backend):
        for ortho in self.orthomosaics:
            ortho.set_backend(backend)

    def get_backend(self):
        return self._backend

    def get_gsd_target_from_index(self, index):
        multiscale_step_index = index // len(self._sample_location_generation_strategy)
        return (self.multiscale_steps_gsd_x[multiscale_step_index], self.multiscale_steps_gsd_y[multiscale_step_index])

    def rescale_color_data(self, color_data, rescale_size_x, rescale_size_y):
        # Rescale all imagery according to interpolation
        if self.rescale_strategy.lower() == "interpolate":
            return torch.nn.functional.interpolate(torch.from_numpy(color_data).permute(2,0,1).unsqueeze(0),
                                                         size=(rescale_size_x, rescale_size_y),
                                                         mode='bilinear',
                                                         align_corners=False).squeeze(0).permute(1,2,0).numpy()

        # Rescale all imagery according to fourier bandlimiting
        if self.rescale_strategy.lower() == "fourier":
            spectrogram = fourier_scale(torch.from_numpy(color_data),
                                        output_pixel_dim=(rescale_size_x, rescale_size_y),
                                        projection_mode="visual")
            return fourier_reconstruct(spectrogram,
                                             output_pixel_dim=(rescale_size_x, rescale_size_y),
                                             channels=3)

        # If the strategy is none then just pass the data through
        if self.rescale_strategy.lower() == "none":
            return color_data

        # Raise an error because we don't know the strategy
        raise ValueError("Unknown rescale strategy passed \"" + str(self.rescale_strategy) + "\" options are " + str(["none", "fourier", "interpolate"]))

    def compute_scale_parameters(self, sample_location, orthomosaic, index):
        rescale_size_x = None
        rescale_size_y = None
        gsd_x = None
        gsd_y = None
        mask_x_scale_factor = 1.0
        mask_y_scale_factor = 1.0
        if self.mask_scale_strategy == "pixel":
            # When the passed dimensions are pixels we can use them direclty
            rescale_size_x = sample_location.getXDim()
            rescale_size_y = sample_location.getYDim()
            gsd_x, gsd_y = orthomosaic.get_gsd()
            mask_x_scale_factor = self.mask_x/sample_location.getXDim()
            mask_y_scale_factor = self.mask_y/sample_location.getYDim()
        elif self.mask_scale_strategy == "spatial":
            if self.rescale_to_interval:
                gsd_x, gsd_y = self.get_gsd_target_from_index(index)
                rescale_size_x = int(np.around(self.tile_x / gsd_x))
                rescale_size_y = int(np.around(self.tile_y / gsd_y))
                mask_x_scale_factor = rescale_size_x / sample_location.getXDim()
                mask_y_scale_factor = rescale_size_y / sample_location.getYDim()
            else:
                # If rescaling is necessary
                if self.min_gsd_x is not None and self.tile_x / sample_location.getXDim() < self.min_gsd_x:
                    rescale_size_x = int(np.around(self.tile_x / self.min_gsd_x))
                    mask_x_scale_factor = rescale_size_x / sample_location.getXDim()
                    gsd_x = self.min_gsd_x
                if self.min_gsd_y is not None and self.tile_y / sample_location.getYDim() < self.min_gsd_y:
                    rescale_size_y = int(np.around(self.tile_y / self.min_gsd_y))
                    mask_y_scale_factor = rescale_size_y / sample_location.getYDim()
                    gsd_y = self.min_gsd_y
                if self.max_gsd_x is not None and self.tile_x / sample_location.getXDim() > self.max_gsd_x:
                    rescale_size_x = int(np.around(self.tile_x / self.max_gsd_x))
                    mask_x_scale_factor = rescale_size_x / sample_location.getXDim()
                    gsd_x = self.max_gsd_x
                if self.max_gsd_y is not None and self.tile_y / sample_location.getYDim() > self.max_gsd_y:
                    rescale_size_y = int(np.around(self.tile_y / self.max_gsd_y))
                    mask_y_scale_factor = rescale_size_y / sample_location.getYDim()
                    gsd_y = self.max_gsd_y

                # If rescaling is not necessary
                if rescale_size_x is None:
                    gsd_x = orthomosaic.get_gsd()[0]
                    rescale_size_x = sample_location.getXDim()
                if rescale_size_y is None:
                    gsd_y = orthomosaic.get_gsd()[1]
                    rescale_size_y = sample_location.getYDim()

        return rescale_size_x, rescale_size_y, gsd_x, gsd_y, mask_x_scale_factor, mask_y_scale_factor

    def generate_sample(self, index):
        # Record the time at which sample generation started
        t_select_ortho = time.time()

        # Sample a location that we will present
        sample_location = self._sample_location_generation_strategy.getSampleLocation(index % len(self._sample_location_generation_strategy))
        exceptions = sample_location.getGenerationMetadata().getExceptions()

        # Get the IDs of the spatial objects in the sample
        building_ids = [b.getId() for b in sample_location.getBuildings()]
        road_line_ids = [rl.getId() for rl in sample_location.getRoadLines()]

        # Get the orthomosaic that we will return, and so we can get the details of the sample
        orthomosaic = self.orthomosaics[sample_location.getOrthomosaicIdx()]

        # Get the location of the sample
        x_p = sample_location.getX()
        y_p = sample_location.getY()

        # Record the time at which data loading started
        t_load_data = time.time()

        # Read the data from the orthomosaic
        try:
            color_data = orthomosaic.read(x_p, y_p, sample_location.getXDim(), sample_location.getYDim(), center_xy=False)
        # If an exception occurs, then make the sample blank and record the exception for logging
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            color_data = np.zeros((sample_location.getXDim(), sample_location.getYDim(), 3))
            adjustments = []
            unadjusted_buildings_copy = []
            unadjusted_roadlines_copy = []
            exception_name = str(e.__class__.__name__)
            if exception_name in exceptions:
                exceptions[exception_name] += 1
            else:
                exceptions[exception_name] = 1

        # Record the time that interpolation started
        t_interpolation_time = time.time()

        # Compute all the parameters about the scale of the imagery
        rescale_size_x, rescale_size_y, gsd_x, gsd_y, mask_x_scale_factor, mask_y_scale_factor = self.compute_scale_parameters(sample_location,
                                                                                                                               orthomosaic,
                                                                                                                               index)

        # Resize the color data as needed
        color_data = self.rescale_color_data(color_data, rescale_size_x, rescale_size_y)

        # Record the time that loading and converting spatial data started
        t_load_convert_spatial_time = time.time()

        # Get the adjusted, and unadjusted pairs for both road lines and building polygons.
        # In the event that we are generating adjusted geoms, then we get the adjusted geoms and call them unadjusted.
        # This way the adjustments that we get will all be (0,0)
        adjusted_buildings = orthomosaic.get_buildings(ids=building_ids, adjusted=True)
        adjusted_roadlines = orthomosaic.get_road_lines(ids=road_line_ids, adjusted=True)
        unadjusted_buildings = orthomosaic.get_buildings(ids=building_ids, adjusted=self.generate_adjusted_geoms)
        unadjusted_roadlines = orthomosaic.get_annotated_road_lines(ids=road_line_ids, adjusted=self.generate_adjusted_geoms)

        #Offset the buildings so they are now in the coordinates of the image
        unadjusted_buildings_copy = deepcopy(unadjusted_buildings)
        adjusted_buildings_copy = deepcopy(adjusted_buildings)

        # The buildings fetched above are re-read from the orthomosaic by id, so they carry the
        # ortho's stored labels. The annotator that produced this sample may have assigned different
        # labels (e.g. the CHANGE task relabels each building CHANGED/UNCHANGED by comparing it
        # against its parallel views). Carry those annotated labels back onto the re-fetched copies so
        # the drawn mask and the per-class counts reflect the sample's annotation rather than the
        # ortho's stored label. Today only unadjusted_buildings_copy is drawn and counted -- the
        # adjusted copy feeds only the label-independent adjustment reconstruction below -- but both
        # copies are relabeled so the annotation survives regardless of which copy a task consumes.
        # For tasks whose annotator does not relabel (BDA/BDAADJ/RDA) each label is copied onto
        # itself, so this is a no-op.
        annotated_labels_by_id = {b.getId(): b.getLabel() for b in sample_location.getBuildings()}
        for building in chain(unadjusted_buildings_copy, adjusted_buildings_copy):
            annotated_label = annotated_labels_by_id.get(building.getId())
            if annotated_label is not None:
                building.setLabel(annotated_label)

        for unadjusted_building, adjusted_building in zip(unadjusted_buildings_copy, adjusted_buildings_copy):
            unadjusted_building.setGeometry(
                scale_pixel_coords(
                    offset_pixel_coords(unadjusted_building.getGeometry("pixels"), x_p, y_p),
                    mask_x_scale_factor,
                    mask_y_scale_factor),
                "pixels")
            adjusted_building.setGeometry(
                scale_pixel_coords(
                    offset_pixel_coords(adjusted_building.getGeometry("pixels"), x_p, y_p),
                    mask_x_scale_factor,
                    mask_y_scale_factor),
                "pixels")
        building_adjustments = reconstruct_adjustments_from_unadjusted_adjusted_pairs(unadjusted_buildings_copy, adjusted_buildings_copy, x_p, y_p)

        # Offset the road lines so they are now in the coordinates of the image
        unadjusted_roadlines_copy = deepcopy(unadjusted_roadlines)
        adjusted_roadlines_copy = deepcopy(adjusted_roadlines)
        for unadjusted_roadline, adjusted_roadline in zip(unadjusted_roadlines_copy, adjusted_roadlines_copy):
            unadjusted_roadline.setGeometry(scale_pixel_coords(
                    offset_pixel_coords(unadjusted_roadline.getGeometry("pixels"), x_p, y_p),
                    mask_x_scale_factor,
                    mask_y_scale_factor),
                "pixels")
            for unadjusted_labeled_sub_road_line in unadjusted_roadline.get_labeled_sub_lines():
                unadjusted_labeled_sub_road_line.setGeometry(scale_pixel_coords(
                        offset_pixel_coords(unadjusted_labeled_sub_road_line.getGeometry("pixels"), x_p, y_p),
                        mask_x_scale_factor,
                        mask_y_scale_factor),
                    "pixels")
            adjusted_roadline.setGeometry(scale_pixel_coords(
                    offset_pixel_coords(adjusted_roadline.getGeometry("pixels"), x_p, y_p),
                    mask_x_scale_factor,
                    mask_y_scale_factor),
                "pixels")
        roadline_adjustments = reconstruct_adjustments_from_unadjusted_adjusted_pairs(unadjusted_roadlines_copy, adjusted_roadlines_copy, x_p, y_p)

        # Combine the adjustments into the resulting vector field that will be passed with the sample
        adjustments = building_adjustments + roadline_adjustments

        # Log the details of the attempt to generate a sample
        t_compute_telemetry = time.time()
        sample_metadata = {}
        sample_metadata[SAMPLE_METADATA_EXCEPTIONS] = exceptions
        sample_metadata[SAMPLE_METADATA_ATTEMPTS] = sample_location.getGenerationMetadata().getAttempts()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "select_ortho_time"] = t_load_data - t_select_ortho
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "generate_sample_point_time"] = sample_location.getGenerationMetadata().getGenerationSec()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "get_annotation_time"] = sample_location.getGenerationMetadata().getAnnotationSec()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "validate_sample_time"] = sample_location.getGenerationMetadata().getValidationSec()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "load_convert_spatial_time"] = t_compute_telemetry - t_load_convert_spatial_time
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "rescale_imagery_time"] = t_load_convert_spatial_time - t_interpolation_time
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "load_pixels_time"] = t_interpolation_time - t_load_data

        for label in self.label_map.getAllLabels():
            sample_metadata[POLYGON_COUNT_PREFIX + label] = sum(1 if b.getLabel() == label else 0 for b in unadjusted_buildings_copy)
            sample_metadata[PIXEL_COUNT_PREFIX + label] = sum(b.getGeometry("pixels").area if b.getLabel() == label else 0 for b in unadjusted_buildings_copy)
            for roadline in unadjusted_roadlines_copy:
                sample_metadata[POLYGON_COUNT_PREFIX + label] += sum(1 if b.getLabel() == label else 0 for b in roadline.get_labeled_sub_lines())
                pixel_count = sum(b.getGeometry("pixels").length if b.getLabel() == label else 0 for b in roadline.get_labeled_sub_lines())
                sample_metadata[PIXEL_COUNT_PREFIX + label] += pixel_count

        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "compute_telemetry_time"] = time.time() - t_compute_telemetry

        # Thread the owning ortho's survey timestamp onto the view so temporal encoders (e.g. SatMAE)
        # receive a real [year, month, hour] via getBatchedTimestamp(). Orthos without stats have no
        # timestamp, so get_timestamp() returns None and None propagates unchanged -- models that do
        # not read the timestamp see no difference.
        v = View(
            raw_imagery=color_data,
            adjustments=adjustments,
            orthomosaic=orthomosaic,
            gsd_x=gsd_x,
            gsd_y=gsd_y,
            timestamp=orthomosaic.get_timestamp(),
        )
        return Sample(sample_location=sample_location,
                      views=[v],
                      buildings=unadjusted_buildings_copy,
                      road_lines=unadjusted_roadlines_copy,
                      metadata=sample_metadata,
                      label_map=self.label_map)
