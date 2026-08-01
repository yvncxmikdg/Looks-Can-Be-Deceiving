import os
import copy
import time
from multiprocessing import Pool

import numpy as np

from modeling.utils.sample_generator_utils import generate_sample_point, get_valid_lines, get_valid_buildings
from modeling.utils.building_frame_generation import get_candidate_samples_center
from modeling.utils.sample_presentation import RealTimeSampleLocationPresentationStrategy, PregeneratedSampleLocationPresentationStrategy
from modeling.utils.random_utils import reseed_distributed
from modeling.Spatial import MultiLabeledRoadLineFactory

# This is a class that implements a strategy for generating sample locations in an orthomosaic
# The expectation is that if you want to get different behavior from your data generator, you
# Can pass these different strategies to your code and it it will utilize the different strategy
# as needed. This is the entry point into the class hierarchy.
class SampleLocationGenerationStrategy:
    def __init__(self, strategy_name, sample_location_presentation_strategy, annotator):
        self._strategy_name = strategy_name
        self._sample_location_presentation_strategy = sample_location_presentation_strategy
        self._annotator = annotator
        self._xdim = None
        self._ydim = None
        self._x_gsd = None
        self._y_gsd = None
        self._dim_scale = None
    def initializeLocationGenerationStrategy(self, xdim, ydim, scale="pixel"):
        if scale == "pixel":
            self._xdim = int(xdim)
            self._ydim = int(ydim)
        elif scale == "spatial":
            self._xdim = float(xdim)
            self._ydim = float(ydim)
        else:
            raise ValueError("Unknown dimension scale passed " + str(scale) + ". Options are " + str(["pixel", "spatial"]))
        self._dim_scale = scale
    def _getXDim(self):
        return int(np.around(self._xdim / self._x_gsd))
    def _getYDim(self):
        return int(np.around(self._ydim / self._y_gsd))
    def _prime_dim_scale(self, ortho):
        if self._dim_scale == "pixel":
            self._x_gsd = 1
            self._y_gsd = 1
        elif self._dim_scale == "spatial":
            self._x_gsd, self._y_gsd = ortho.get_gsd()[:2]
    def getStrategyName(self):
        return self._strategy_name
    def getAnnotator(self):
        return self._annotator
    def getSampleLocation(self, index):
        raise NotImplementedError("getSampleLocations must be implemented by a subclass")
    def __len__(self):
        return len(self._sample_location_presentation_strategy)

# This is a class that is specifically for sample generation strategies that involve generating
# Samples in real time when the function getSampleLocation is called.
class RealTimeSampleLocationGenerationStrategy(SampleLocationGenerationStrategy):
    def __init__(self, name, orthomosaics, annotator, sample_location_presentation_strategy):
        if not isinstance(sample_location_presentation_strategy, RealTimeSampleLocationPresentationStrategy):
            raise ValueError("sample_location_presentation_strategy must be an instance of ",
                             RealTimeSampleLocationPresentationStrategy,
                             "instead found",
                             type(sample_location_presentation_strategy))
        super().__init__(name, sample_location_presentation_strategy, annotator)

        self._orthomosaics = orthomosaics
    def getSampleLocation(self, index):
        raise NotImplementedError("getSampleLocations must be implemented by a subclass")

# This is a class that iteratively generates a random sample location until it finds a valid one
class RandomSampleLocationGenerationStrategy(RealTimeSampleLocationGenerationStrategy):
    def __init__(self, orthomosaics, annotator, sample_location_presentation_strategy, sample_acceptance_persistence=25, seed_range=10e6):
        super().__init__("random", orthomosaics, annotator, sample_location_presentation_strategy)
        self._sample_acceptance_persistence = int(sample_acceptance_persistence)
        self.__rs = np.random.RandomState()
        self.__seed_range = int(float(seed_range))
        self._samples_generated = 0

    def getSampleLocation(self, index):
        reseed_distributed(self._samples_generated, self.__rs, self.__seed_range)
        self._samples_generated += 1

        orthomosaic_idx = self.__rs.randint(0, len(self._orthomosaics))
        ortho = self._orthomosaics[orthomosaic_idx]

        attempts = 0
        t_sample_generation = 0
        t_sample_annotation = 0
        t_sample_validation = 0
        accepted = False
        exceptions = {}
        while((not accepted) and attempts < self._sample_acceptance_persistence):
            t_0 = time.time()
            x_p, y_p = generate_sample_point(ortho, self.__rs)
            t_1 = time.time()

            self._prime_dim_scale(ortho)
            validation_call_args = self._annotator.make_sample_annotation_call_args(x_p, y_p, self._getXDim(), self._getYDim(), ortho, orthomosaic_idx, None)
            sample_candidate = self._annotator.annotate_sample(*validation_call_args)

            #Accumulate this candidate's exceptions into the running totals for the sample.
            for exception_name, exception_count in sample_candidate.getGenerationMetadata().getExceptions().items():
                exceptions[exception_name] = exceptions.get(exception_name, 0) + exception_count

            t_2 = time.time()
            if len(sample_candidate.getBuildings()) > 0 or len(sample_candidate.getRoadLines()) > 0:
                accepted = True
            t_3 = time.time()

            t_sample_generation += t_1-t_0
            t_sample_annotation += t_2-t_1
            t_sample_validation += t_3-t_2

            attempts += 1

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(attempts, t_sample_generation, t_sample_annotation, t_sample_validation, exceptions)

        # Return the valid road lines that were generated for the sample.
        result = SampleLocation(x=x_p,
                                y=y_p,
                                x_dim=self._getXDim(),
                                y_dim=self._getYDim(),
                                buildings=sample_candidate.getBuildings(),
                                roadlines=sample_candidate.getRoadLines(),
                                orthomosaic_idx=orthomosaic_idx,
                                generation_meta=generation_meta)

        self._sample_location_presentation_strategy.observeSampleLocation(result)
        return self._sample_location_presentation_strategy.getSampleLocation(index)


# This is a subclass that generates sample locations all at once and validates them for the user
# so that when the user calls the getSampleLocation function, there is a valid sample waiting for
# them to pass to a model.
class PregeneratedSampleLocationGenerationStrategy(SampleLocationGenerationStrategy):
    def __init__(self, name, orthomosaics, annotator, sample_location_presentation_strategy, sample_generator_process_pool_size=6):
        if not isinstance(sample_location_presentation_strategy, PregeneratedSampleLocationPresentationStrategy):
            raise ValueError("sample_location_presentation_strategy must be an instance of ",
                             PregeneratedSampleLocationPresentationStrategy,
                             "instead found",
                             type(sample_location_presentation_strategy))
        super().__init__(name, sample_location_presentation_strategy, annotator)
        self._sample_generator_process_pool_size = sample_generator_process_pool_size
        self._annotator = annotator
        self._samples = []
        self._orthomosaics = orthomosaics

    def initializeLocationGenerationStrategy(self, xdim, ydim, scale="pixel"):
        super().initializeLocationGenerationStrategy(xdim, ydim, scale)
        self._samples = self._pregenerate_sample_locations(self._orthomosaics)
        if len(self._samples) == 0:
            # Without this, an empty pool surfaces much later as an opaque error at the first
            # batch draw (e.g. a ZeroDivisionError inside the weighted presentation strategy on a
            # DataLoader worker). Zero samples at init always means the inputs are broken: no
            # orthomosaic contributed a usable object -- annotation files missing/empty, no
            # adjusted geometries when the annotator samples adjusted locations, or (CHANGE) an
            # anchor_source that matched no orthomosaics.
            raise ValueError(
                f"Sample pregeneration produced 0 valid sample locations across "
                f"{len(self._orthomosaics)} orthomosaic(s). Check that the annotation files for "
                "these orthomosaics exist and actually contain buildings/road lines (including "
                "adjusted geometries if the annotator generates adjusted sample locations), and "
                "that any CHANGE anchor_source matches the orthomosaics' collection source."
            )
        self._sample_location_presentation_strategy.initialize_samples(self._samples)

    def getSampleLocation(self, index):
        return self._sample_location_presentation_strategy.getSampleLocation(index)

    def _pregenerate_sample_locations(self, orthomosaics):
        testing_locations = []
        for i, orthomosaic in enumerate(orthomosaics):
            testing_locations.extend(self._get_sample_locations_to_validate(orthomosaic, i))

        result = []
        with Pool(processes=self._sample_generator_process_pool_size) as pool:
            candidate_samples = pool.starmap(self._annotator.annotate_sample, testing_locations)

        for candidate_sample in candidate_samples:
            if len(candidate_sample.getBuildings()) > 0 or len(candidate_sample.getRoadLines()) > 0:
                result.append(candidate_sample)
        return result

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        raise NotImplementedError("_get_sample_locations_to_validate must be implemented by a subclass")

# This is a subclass that generates samples for building damage assessment and alignment training that
# Attempts to include as many buildings in a frame as possible while keeping them within the range
# of adjustment_buffer_distance_px pixels from the edge of the frame
class CenteredBuildingSampleStrategy(PregeneratedSampleLocationGenerationStrategy):
    def __init__(self, adjustment_buffer_distance_px, annotator, sample_location_presentation_strategy, orthomosaics, sample_generator_process_pool_size=6):
        if not isinstance(annotator, BDASampleAnnotator):
            raise ValueError("CenteredSampleStrategy is only defined for samples that can be validated using the BDASampleAnnotator.")

        self._adjustment_buffer_distance_px = int(adjustment_buffer_distance_px)
        super().__init__("Centered", orthomosaics, annotator, sample_location_presentation_strategy, sample_generator_process_pool_size)

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        bda_sample_validation_calls = []
        self._prime_dim_scale(orthomosaic)
        frames_of_buildings = get_candidate_samples_center(orthomosaic,
                                                           self._getXDim(),
                                                           self._getYDim(),
                                                           self._adjustment_buffer_distance_px,
                                                           adjusted=self._annotator.generatesAdjustedSamples())
        for frame_of_buildings in frames_of_buildings:
            x = frame_of_buildings[0].centroid.x - self._getXDim()/2
            y = frame_of_buildings[0].centroid.y - self._getYDim()/2
            building_ids = frame_of_buildings[1]
            bda_sample_validation_calls.append(self._annotator.make_sample_annotation_call_args(x,
                                                                                                y,
                                                                                                self._getXDim(),
                                                                                                self._getYDim(),
                                                                                                orthomosaic,
                                                                                                orthomosaic_idx,
                                                                                                building_ids))
        return bda_sample_validation_calls

# This is a subclass that generates sample for building and road damage assessment by uniformly tiling
# the image into a grid.
class GridSampleStrategy(PregeneratedSampleLocationGenerationStrategy):
    def __init__(self, adjustment_buffer_distance_px, annotator, sample_location_presentation_strategy, orthomosaics, sample_generator_process_pool_size=6):
        self._adjustment_buffer_distance_px = adjustment_buffer_distance_px
        super().__init__("Grid", orthomosaics, annotator, sample_location_presentation_strategy, sample_generator_process_pool_size)

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        # Create a list to store the candidate samples
        sample_validation_calls = []
        self._prime_dim_scale(orthomosaic)
        # Iterate over the orthomosaic looking for buildings that need to be labeled
        for x in np.arange(0-self._adjustment_buffer_distance_px, orthomosaic.get_width(), self._getXDim()-2*self._adjustment_buffer_distance_px):
            for y in np.arange(0-self._adjustment_buffer_distance_px, orthomosaic.get_height(), self._getYDim()-2*self._adjustment_buffer_distance_px):
                call = self._annotator.make_sample_annotation_call_args(x, y, self._getXDim(), self._getYDim(), orthomosaic, orthomosaic_idx, None)
                sample_validation_calls.append(call)
        return sample_validation_calls

class SampleLocation:
    def __init__(self, x, y, x_dim, y_dim, buildings, roadlines, orthomosaic_idx, generation_meta=None):
        self._x = x
        self._y = y
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._buildings = buildings
        self._roadlines = roadlines
        self._orthomosaic_idx = orthomosaic_idx
        self._generation_meta = generation_meta
        if self._generation_meta is None:
            self._generation_meta = SampleLocationGenerationMetadata()

    def getX(self):
        return self._x
    def getY(self):
        return self._y
    def getXDim(self):
        return self._x_dim
    def getYDim(self):
        return self._y_dim
    def getBuildings(self):
        return self._buildings
    def getRoadLines(self):
        return self._roadlines
    def getOrthomosaicIdx(self):
        return self._orthomosaic_idx
    def getGenerationMetadata(self):
        return self._generation_meta

class SampleLocationGenerationMetadata:
    def __init__(self, attempts=1, generation_sec=0.0, annotation_sec=0.0, validation_sec=0.0, exceptions=None):
        self._exceptions = exceptions
        if self._exceptions is None:
            self._exceptions = {}
        self._attempts = attempts
        self._generation_sec = generation_sec
        self._annotation_sec = annotation_sec
        self._validation_sec = validation_sec
    def getExceptions(self):
        return self._exceptions
    def getAttempts(self):
        return self._attempts
    def getGenerationSec(self):
        return self._generation_sec
    def getAnnotationSec(self):
        return self._annotation_sec
    def getValidationSec(self):
        return self._validation_sec

class SamplePregenerationAnnotator:
    def __init__(self, generate_adjusted_sample_locations, center_xy=False):
        self._generate_adjusted_sample_locations = generate_adjusted_sample_locations
        self._center_xy = center_xy
    def expectsCenteredXY(self):
        return self._center_xy
    def generatesAdjustedSamples(self):
        return self._generate_adjusted_sample_locations

class BDASampleAnnotator(SamplePregenerationAnnotator):
    def __init__(self, generate_adjusted_sample_locations, center_xy=False, building_intersection_proportion_threshold=0.0):
        super().__init__(generate_adjusted_sample_locations, center_xy)
        self._building_intersection_proportion_threshold = float(building_intersection_proportion_threshold)
    def make_sample_annotation_call_args(self, x, y, x_dim, y_dim, orthomosaic, orthomosaic_idx, ids=None):
        return [x,
                y,
                x_dim,
                y_dim,
                orthomosaic.get_buildings(adjusted=self._generate_adjusted_sample_locations, ids=ids),
                orthomosaic_idx]
    def annotate_sample(self, x, y, x_dim, y_dim, buildings, orthomosaic_idx):
        t0 = time.time()
        # Get the valid polygons for this window
        exceptions = {}
        valid_buildings, exceptions = get_valid_buildings(
            x=x,
            y=y,
            buildings=buildings,
            x_dim=x_dim,
            y_dim=y_dim,
            building_intersection_proportion_threshold=self._building_intersection_proportion_threshold,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        t1 = time.time()

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, exceptions)

        # Return the valid buildings that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=valid_buildings,
                              roadlines=[],
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)

class BuildingChangeSampleAnnotator(BDASampleAnnotator):
    """Annotator for the cross-source building-change task.

    Each in-frame building is relabeled CHANGED or UNCHANGED by comparing its own label against the
    label of that same building in its parallel views: the views of the building (matched by
    correlation id) that appear in the other orthomosaics. Orthomosaics that share the building's own
    file are skipped, and -- when ``skip_same_source`` is set -- so are orthomosaics collected from
    the building's own source, so only genuinely independent views drive the decision. The
    ``change_decision_rule`` picks the aggregation: under ``"all"`` (default) a building is CHANGED
    only when its label differs in every parallel view, under ``"any"`` a single disagreeing view is
    enough. A building with no parallel views is dropped from the sample either way.
    """

    CHANGE_DECISION_RULES = {"all": all, "any": any}

    def __init__(self, all_orthos, generate_adjusted_sample_locations, center_xy=False, building_intersection_proportion_threshold=0.0,
                 skip_same_source=True, change_decision_rule="all"):
        super().__init__(generate_adjusted_sample_locations, center_xy, building_intersection_proportion_threshold)
        self._all_orthos = all_orthos
        self._skip_same_source = skip_same_source
        if change_decision_rule not in self.CHANGE_DECISION_RULES:
            raise ValueError("Unknown change_decision_rule " + str(change_decision_rule) + ". Options are " + str(sorted(self.CHANGE_DECISION_RULES.keys())))
        self._change_decision_rule = change_decision_rule

    def annotate_sample(self, x, y, x_dim, y_dim, buildings, orthomosaic_idx):
        t0 = time.time()
        # Get the valid polygons for this window
        exceptions = {}
        in_frame, exceptions = get_valid_buildings(
            x=x,
            y=y,
            buildings=buildings,
            x_dim=x_dim,
            y_dim=y_dim,
            building_intersection_proportion_threshold=self._building_intersection_proportion_threshold,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        t1 = time.time()

        valid_buildings = []
        for building_in_current_frame in in_frame:
            correlation_id = building_in_current_frame.getCorrelationId()
            building_file_name = os.path.splitext(building_in_current_frame.getFile())[0]
            building_source = building_in_current_frame.getLabelSource()

            parallel_views = []
            for ortho in self._all_orthos:
                if building_file_name == ortho.get_name():
                    # Same orthomosaic as the building under inspection, so not a parallel view.
                    continue
                if self._skip_same_source and building_source == ortho.get_source_used_for_collection():
                    # Same collection source, so not an independent view of the building.
                    continue
                try:
                    parallel_buildings = ortho.get_buildings(correlation_ids=[correlation_id])
                except KeyError:
                    # This ortho has buildings but none with this correlation id: no parallel view here.
                    continue
                # get_buildings returns the building indexed by this correlation id (its correlation id
                # therefore already matches), or None when the ortho holds no buildings at all.
                if parallel_buildings is not None:
                    parallel_views.extend(parallel_buildings)

            if not parallel_views:
                # No independent view of this building, so its change state is undetermined; drop it.
                continue

            # Under "all", CHANGED requires the label to differ in EVERY parallel view; under "any",
            # one disagreeing view suffices.
            rule = self.CHANGE_DECISION_RULES[self._change_decision_rule]
            changed_in_parallel_views = rule(
                view.getLabel() != building_in_current_frame.getLabel() for view in parallel_views
            )
            relabeled_building = copy.deepcopy(building_in_current_frame)
            relabeled_building.setLabel("CHANGED" if changed_in_parallel_views else "UNCHANGED")
            valid_buildings.append(relabeled_building)

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, exceptions)

        # Return the valid buildings that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=valid_buildings,
                              roadlines=[],
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)

class RDASampleAnnotator(SamplePregenerationAnnotator):
    def make_sample_annotation_call_args(self, x, y, x_dim, y_dim, orthomosaic, orthomosaic_idx, _):
        return [x,
                y,
                x_dim,
                y_dim,
                orthomosaic.get_road_lines(adjusted=self._generate_adjusted_sample_locations),
                orthomosaic.get_road_line_annotation_polygons(),
                orthomosaic_idx]

    def annotate_sample(self, x, y, x_dim, y_dim, roadlines, annotation_polygons, orthomosaic_idx):
        t0 = time.time()
        exceptions = {}
        valid_road_lines, exceptions = get_valid_lines(
            x,
            y,
            roadlines,
            x_dim=x_dim,
            y_dim=y_dim,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        labeled_road_lines = MultiLabeledRoadLineFactory(valid_road_lines, annotation_polygons)
        t1 = time.time()

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, exceptions)

        # Return the valid road lines that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=[],
                              roadlines=labeled_road_lines,
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)
