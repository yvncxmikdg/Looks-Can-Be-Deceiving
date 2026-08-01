from collections import defaultdict
from shapely import Polygon
import torch


def collate_fn(data):
    return SingleViewBatchedSamples(data)


class BatchedSamples:
    def __init__(self, samples):
        self._batched_buildings = [s.getBuildings() for s in samples]
        self._batched_road_lines = [s.getRoadLines() for s in samples]

        self._batched_metadata = defaultdict(list)
        all_metadata_keys = []
        for s in samples:
            all_metadata_keys.extend(s.getMetadata().keys())
        self._all_metadata_keys = list(set(all_metadata_keys))
        # Iterate the DEDUPLICATED keys: all_metadata_keys contains every key once per sample that
        # carries it, so looping it here appended each value ~batch_size times, inflating every
        # batched-metadata list to O(N^2). Downstream telemetry sums these and normalizes by
        # len(batch), so the "(Normalized)" charts read ~batch_size too high.
        for s in samples:
            for k in self._all_metadata_keys:
                self._batched_metadata[k].append(s.getMetadataEntry(k))

        self._sample_count = len(samples)
        self._batched_raw_imagery = None
        self._batched_input_imagery = None
        self._batched_queries = None
        self._batched_labels = None

        for i, sample_i in enumerate(samples):
            for sample_j in samples[i + 1 :]:
                if sample_i.getLabelMap() != sample_j.getLabelMap():
                    raise ValueError(
                        "Invalid batch. Samples within a batch cannot have different label maps."
                    )

        self._label_map = samples[0].getLabelMap()

    def getBatchedImagery(self):
        return self._batched_input_imagery

    def getBatchedRawImagery(self):
        return self._batched_raw_imagery

    def getBatchedQueries(self):
        return self._batched_queries

    def getBatchedLabels(self):
        return self._batched_labels

    def getBatchedBuildings(self):
        return self._batched_buildings

    def getBatchedRoadLines(self):
        return self._batched_road_lines

    def getBatchedMetadata(self):
        return self._batched_metadata

    def getBatchedMetadataEntry(self, field):
        return self._batched_metadata[field]

    def getMetadataKeys(self):
        return self._all_metadata_keys

    def getLabelMap(self):
        return self._label_map

    def __len__(self):
        return self._sample_count


class SingleViewBatchedSamples(BatchedSamples):
    def __init__(self, samples):
        super().__init__(samples)
        self._batched_raw_imagery = None
        if any((not s.getView(0).getRawImagery() is None) for s in samples):
            self._batched_raw_imagery = torch.stack(
                [s.getView(0).getRawImagery().clone().detach() for s in samples], dim=0
            )

        self._batched_input_imagery = None
        if any((not s.getView(0).getInputImagery() is None) for s in samples):
            self._batched_input_imagery = torch.stack(
                [s.getView(0).getInputImagery().clone().detach() for s in samples],
                dim=0,
            )

        self._batched_queries = None
        if any((not s.getView(0).getQueries() is None) for s in samples):
            self._batched_queries = torch.stack(
                [s.getView(0).getQueries().clone().detach() for s in samples], dim=0
            )

        self._batched_labels = None
        if any((not s.getView(0).getLabels() is None) for s in samples):
            self._batched_labels = torch.stack(
                [s.getView(0).getLabels().clone().detach() for s in samples], dim=0
            )

        self._batched_frame_geoms = [s.getView(0).getFrame() for s in samples]
        self._batched_orthomosaic = [s.getView(0).getOrthomosaic() for s in samples]
        self._label_map = samples[0].getLabelMap()
        self._batched_adjustments = [s.getView(0).getAdjustments() for s in samples]
        self._device = None
        self._batched_xy_p = [
            [s.getView(0).getX(), s.getView(0).getY()] for s in samples
        ]
        self._batched_x = [s.getX() for s in samples]
        self._batched_y = [s.getY() for s in samples]
        self._batched_gsds = [s.getView(0).getGSD() for s in samples]
        self._batched_timestamps = [s.getView(0).getTimestamp() for s in samples]

    def getBatchedX(self):
        return self._batched_x

    def getBatchedY(self):
        return self._batched_y

    def getBatchedXYPixels(self):
        return self._batched_xy_p

    def getBatchedFrameGeometry(self):
        return self._batched_frame_geoms

    def getBatchedOrthomosaic(self):
        return self._batched_orthomosaic

    def getBatchedGSD(self):
        return self._batched_gsds

    def getBatchedTimestamp(self):
        return self._batched_timestamps

    def getBatchedAdjustments(self):
        return self._batched_adjustments

    def getBatchedAdjustmentsTensor(self):
        result = []
        for adjustments in self._batched_adjustments:
            if len(adjustments) > 0:
                result.append(torch.stack([a.asTorchTensor() for a in adjustments]))
            else:
                result.append(None)
        return result

    def moveTo(
        self,
        device,
        move_raw_imagery=True,
        move_input_imagery=True,
        move_queries=True,
        move_labels=True,
    ):
        if not self._batched_raw_imagery is None and move_raw_imagery:
            self._batched_raw_imagery = self._batched_raw_imagery.to(device)
        if not self._batched_input_imagery is None and move_input_imagery:
            self._batched_input_imagery = self._batched_input_imagery.to(device)
        if not self._batched_queries is None and move_queries:
            self._batched_queries = self._batched_queries.to(device)
        if not self._batched_labels is None and move_labels:
            self._batched_labels = self._batched_labels.to(device)
        self._device = device

    def get_device(self):
        return self._device


class Sample:
    def __init__(
        self,
        sample_location=None,
        views=None,
        buildings=None,
        road_lines=None,
        metadata=None,
        label_map=None,
    ):
        self.__views = [] if views is None else views
        self.__sample_location = sample_location
        self.__buildings = [] if buildings is None else buildings
        self.__road_lines = [] if road_lines is None else road_lines
        self.__metadata = {} if metadata is None else metadata
        self.__label_map = label_map

    def getViews(self):
        return self.__views

    def getView(self, index):
        return self.__views[index]

    def getSampleLocation(self):
        return self.__sample_location

    def getX(self):
        if self.__sample_location is None:
            return None
        return self.__sample_location.getX()

    def getY(self):
        if self.__sample_location is None:
            return None
        return self.__sample_location.getY()

    def getPoint(self):
        if self.__sample_location is None:
            return None
        return [self.__sample_location.getX(), self.__sample_location.getY()]

    def getBuildings(self):
        return self.__buildings

    def getRoadLines(self):
        return self.__road_lines

    def getMetadata(self):
        return self.__metadata

    def getMetadataEntry(self, field):
        try:
            return self.__metadata[field]
        except KeyError:
            return None

    def getLabelMap(self):
        return self.__label_map

    def setSampleLocation(self, sample_location):
        self.__sample_location = sample_location

    def setBuildings(self, new_buildings):
        self.__buildings = new_buildings

    def setRoadLines(self, new_road_lines):
        self.__road_lines = new_road_lines

    def setMetadata(self, new_metadata):
        self.__metadata = new_metadata

    def extendMetadata(self, new_metadata):
        for k, v in new_metadata.items():
            self.__metadata[k] = v

    def setMetadataEntry(self, key, value):
        self.__metadata[key] = value

    def __len__(self):
        return len(self.__views)


class View:
    def __init__(
        self,
        raw_imagery=None,
        input_imagery=None,
        labels=None,
        queries=None,
        adjustments=None,
        orthomosaic=None,
        x=0,
        y=0,
        gsd_x=None,
        gsd_y=None,
        timestamp=None,
        pixel_frame_geom=None,
        epsg_4326_frame_geom=None,
    ):
        self.__raw_imagery = None
        if not raw_imagery is None:
            self.__raw_imagery = torch.as_tensor(raw_imagery, dtype=torch.uint8)
        self.__input_imagery = None
        if not input_imagery is None:
            self.__input_imagery = torch.as_tensor(input_imagery, dtype=torch.float32)
        self.__labels = None
        if not labels is None:
            self.__labels = torch.as_tensor(labels, dtype=torch.long)
        self.__queries = None
        if not queries is None:
            self.__queries = torch.as_tensor(queries, dtype=torch.uint8)
        self.__adjustments = adjustments
        self.__orthomosaic = orthomosaic
        self.__x = x
        self.__y = y
        self.__gsd_x = gsd_x
        self.__gsd_y = gsd_y
        self.__timestamp = timestamp

        assert pixel_frame_geom is None or isinstance(pixel_frame_geom, Polygon)
        assert epsg_4326_frame_geom is None or isinstance(epsg_4326_frame_geom, Polygon)

        self.__pixel_frame_geom = pixel_frame_geom
        self.__epsg_4326_frame_geom = epsg_4326_frame_geom

    def getInputImagery(self):
        return self.__input_imagery

    def getRawImagery(self):
        return self.__raw_imagery

    def getQueries(self):
        return self.__queries

    def getLabels(self):
        return self.__labels

    def getAdjustments(self):
        return self.__adjustments

    def getOrthomosaic(self):
        return self.__orthomosaic

    def getGSD(self):
        # Views carry their own GSD because the sample generator may have rescaled the imagery
        # away from the orthomosaic's native resolution (multiscale training). Views built
        # without an explicit GSD fall back to the orthomosaic's native GSD.
        if self.__gsd_x is None and self.__gsd_y is None and self.__orthomosaic is not None:
            return self.__orthomosaic.get_gsd()
        return (self.__gsd_x, self.__gsd_y)

    def getTimestamp(self):
        return self.__timestamp

    def getX(self):
        return self.__x

    def getY(self):
        return self.__y

    def getFrame(self):
        if len(self.__raw_imagery.shape) == 4:
            _, x_dim, y_dim, _ = self.__raw_imagery.shape
        elif len(self.__raw_imagery.shape) == 3:
            x_dim, y_dim, _ = self.__raw_imagery.shape
        else:
            raise ValueError(
                "Raw Image Tensor is an unexpected shape:", self.__raw_imagery.shape
            )
        coords = [
            [self.__x - x_dim / 2, self.__y - y_dim / 2],
            [self.__x + x_dim / 2, self.__y - y_dim / 2],
            [self.__x + x_dim / 2, self.__y + y_dim / 2],
            [self.__x - x_dim / 2, self.__y + y_dim / 2],
            [self.__x - x_dim / 2, self.__y - y_dim / 2],
        ]
        return Polygon(coords)

    def getGeometry(self, axes):
        if "pixel" in axes:
            return self.__pixel_frame_geom
        if "4326" in axes:
            return self.__epsg_4326_frame_geom
        raise ValueError("Passed value for axes \"" + str(axes) + "\" is not defined. Valid options are \"pixels\" and \"EPSG:4326\"")

    def setInputImagery(self, new_input_imagery):
        self.__input_imagery = torch.as_tensor(new_input_imagery, dtype=torch.float32)

    def setRawImagery(self, new_raw_imagery):
        self.__raw_imagery = torch.as_tensor(new_raw_imagery, dtype=torch.uint8)

    def setQueries(self, new_queries):
        self.__queries = torch.as_tensor(new_queries, dtype=torch.uint8)

    def setLabels(self, new_labels):
        self.__labels = torch.as_tensor(new_labels, dtype=torch.long)

    def setAdjustments(self, new_adjustments):
        self.__adjustments = new_adjustments

    def setOrthomosaic(self, new_orthomosaic):
        self.__orthomosaic = new_orthomosaic

class ConfidenceResult:
    def __init__(self, label_confidence_map, force_normalize_confidence_map=True):
        self._label_confidence_map = {}

        confidence_denominator = 0 if force_normalize_confidence_map else 1.0
        for label, confidence in label_confidence_map.items():
            assert(isinstance(label, str) and isinstance(confidence, (float, int)))
            confidence_denominator += confidence

        # Prevent divide by 0
        is_null_prediction = confidence_denominator == 0
        if confidence_denominator == 0:
            confidence_denominator = 1

        for label, confidence in label_confidence_map.items():
            conf_val = 1.0/len(label_confidence_map) if is_null_prediction else confidence
            self._label_confidence_map[label] = conf_val

    def getConfidence(self, label):
        return self._label_confidence_map[label]

    def jsonify(self):
        return self._label_confidence_map

    def getHighestConfidenceLabel(self):
        return max(self._label_confidence_map.items(), key=lambda x: x[1])[0]

    def getConfidenceArray(self, order):
        result = []
        for o in order:
            result.append(self._label_confidence_map[o])
        return result
