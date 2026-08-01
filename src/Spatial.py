import shapely
from shapely import MultiPolygon, Polygon, LineString

from modeling.utils.shape_utils import convert_coords_to_shapely
from modeling.constants import ROAD_LINE

class CalibratedObject:
    def __init__(self, confidence=1.0):
        self._confidence = confidence
    def getConfidence(self):
        return self._confidence
    def jsonify(self):
        confidence = self.getConfidence()
        # BDA predictions carry a per-class ConfidenceResult (jsonify -> dict); RDA predictions
        # carry a plain scalar. Serialize whichever was provided.
        return {"confidence": confidence.jsonify() if hasattr(confidence, "jsonify") else confidence}

class SpatialObject:
    def __init__(self,
                 identifier=None,
                 correlation_identifier=None,
                 label=None,
                 geometry_source=None,
                 pixel_geom=None,
                 epsg_4326_geom=None,
                 adjusted=False,
                 adjustment_subfield=None,
                 label_source=None,
                 boundary=None,
                 filename=None
                 ):
        self._identifier = identifier
        self._correlation_identifier = correlation_identifier
        self._label = label
        self._label_source = label_source
        self._geometry_source = geometry_source
        self._pixel_geom = pixel_geom
        self._epsg_geom = epsg_4326_geom
        self._adjusted = adjusted
        self._adjustment_subfield = adjustment_subfield
        self._boundary = boundary
        self._filename = filename

    def getGeometry(self, axes):
        if "pixel" in axes:
            return self._pixel_geom
        if "4326" in axes:
            return self._epsg_geom
        raise ValueError("Passed value for axes \"" + str(axes) + "\" is not defined. Valid options are \"pixels\" and \"EPSG:4326\"")

    def setGeometry(self, new_geometry, axes):
        if "pixel" in axes:
            self._pixel_geom = new_geometry
        elif "4326" in axes:
            self._epsg_geom = new_geometry
        else:
            raise ValueError("Passed value for axes \"" + str(axes) + "\" is not defined. Valid options are \"pixels\" and \"EPSG:4326\"")

    def getId(self):
        return self._identifier

    def getCorrelationId(self):
        if self._correlation_identifier is None:
            return self._identifier
        return self._correlation_identifier

    def getLabel(self):
        return self._label

    def setLabel(self, new_label):
        self._label = new_label

    def getFile(self):
        return self._filename

    def getLabelSource(self):
        return self._label_source

    def getGeometrySource(self):
        return self._geometry_source

    def isAdjusted(self):
        return self._adjusted

    def getAdjustmentSubfield(self):
        return self._adjustment_subfield

    def jsonify(self):
        pixel_data = self.getGeometry("pixels")
        epsg_data = self.getGeometry("EPSG:4326")
        pixel_data_formatted = []
        epsg_data_formatted = []
        if not pixel_data is None:
            geoms = []
            if isinstance(pixel_data, Polygon):
                geoms.append(pixel_data)
            elif isinstance(pixel_data, MultiPolygon):
                geoms.extend(list(pixel_data.geoms))
            for sub_pixel_polygon in geoms:
                xx, yy = sub_pixel_polygon.exterior.coords.xy
                pixel_data_formatted.append([{"x":float(x), "y":float(y)} for x, y in zip(xx, yy)])
        if not epsg_data is None:
            geoms = []
            if isinstance(pixel_data, Polygon):
                geoms.append(pixel_data)
            elif isinstance(pixel_data, MultiPolygon):
                geoms.extend(list(pixel_data.geoms))
            for sub_epsg_data in geoms:
                xx, yy = sub_epsg_data.exterior.coords.xy
                epsg_data_formatted.append([{"lat":float(x), "lon":float(y)} for x, y in zip(xx, yy)])

        adj_source = self.getAdjustmentSubfield()

        result = {
            "label":self.getLabel(),
            "label_source":self.getLabelSource(),
            "geometry_source":self.getGeometrySource(),
            "pixels": None if pixel_data is None else pixel_data_formatted,
            "EPSG:4326": None if epsg_data is None else epsg_data_formatted,
            "adjusted":self.isAdjusted(),
            "adjustment_subfield": None if (adj_source is None) else adj_source.jsonify(),
            "id":self.getId()
        }

        return result


class Building(SpatialObject):
    pass

class LabeledBuilding(Building, CalibratedObject):
    def __init__(self,
                 identifier=None,
                 label=None,
                 geometry_source=None,
                 pixel_geom=None,
                 epsg_4326_geom=None,
                 adjusted=False,
                 adjustment_subfield=None,
                 confidence=1.0,
                 label_source=None,
                 views_considered=None,
                 gsd=None):
        Building.__init__(self,
                          identifier=identifier,
                          geometry_source=geometry_source,
                          pixel_geom=pixel_geom,
                          epsg_4326_geom=epsg_4326_geom,
                          adjusted=adjusted,
                          adjustment_subfield=adjustment_subfield,
                          label=label,
                          label_source=label_source)
        CalibratedObject.__init__(self, confidence=confidence)
        self._views_considered=views_considered
        self._gsd=gsd

    def jsonify(self):
        result_building = Building.jsonify(self)
        result_calibrated = CalibratedObject.jsonify(self)
        result = result_building | result_calibrated
        result["gsd"] = None if self._gsd is None else self._gsd.jsonify()
        result["views_considered"] = self._views_considered
        return result

class RoadLine(SpatialObject):
    def __init__(self,
                 identifier=None,
                 geometry_source=None,
                 pixel_geom=None,
                 epsg_4326_geom=None,
                 adjusted=False,
                 adjustment_subfield=None,
                 label=None,
                 label_source=None):
        super().__init__(identifier=identifier,
                         geometry_source=geometry_source,
                         pixel_geom=pixel_geom,
                         epsg_4326_geom=epsg_4326_geom,
                         adjusted=adjusted,
                         adjustment_subfield=adjustment_subfield,
                         label="Road Line" if label is None else label,
                         label_source=label_source)

    def getGeometry(self, axes, parent_road_line=None):
        if axes == "relative":
            if parent_road_line is None:
                raise ValueError("In order to compute relative geometry, a parent road line must be passed.")
            par_xx, par_yy = parent_road_line.getGeometry("pixels").coords.xy
            start = (par_xx[0], par_yy[0])
            end = (par_xx[-1], par_yy[-1])

            cur_xx, cur_yy = self.getGeometry("pixels").coords.xy
            sub_start = (cur_xx[0], cur_yy[0])
            sub_end = (cur_xx[-1], cur_yy[-1])

            x_denom = 1 if (end[0]-start[0]) == 0 else (end[0]-start[0])
            y_denom = 1 if (end[1]-start[1]) == 0 else (end[1]-start[1])

            start_span_x = (sub_start[0] - start[0])/x_denom
            start_span_y = (sub_start[1] - start[1])/y_denom

            end_span_x = (sub_end[0] - end[0])/x_denom
            end_span_y = (sub_end[1] - end[1])/y_denom

            #Assuming the lines are ~actually~ conincident then these two values should be equal
            #so we simply take the average
            start_span = (start_span_x+start_span_y)/2
            end_span = (end_span_x+end_span_y)/2

            return LineString([[start_span, 0], [1+end_span, 0]])
        return super().getGeometry(axes)

    def get_relative_span(self, relative_start, relative_end, axes):
        if axes == "relative":
            return LineString([[relative_start, 0], [relative_end, 0]])
        if axes in ("pixels", "EPSG:4326"):
            xx, yy = self.getGeometry(axes).coords.xy
            dx = xx[1] - xx[0]
            dy = yy[1] - yy[0]

            start_x = xx[0] + dx*relative_start
            start_y = yy[0] + dy*relative_start

            end_x = xx[0] + dx*relative_end
            end_y = yy[0] + dy*relative_end

            return LineString([[start_x, start_y], [end_x, end_y]])
        raise ValueError("Axes \"" + str(axes) + "\" is not valid. Options are relative, pixels, and EPSG:3426")


    def jsonify(self, parent_road_line=None):
        pixel_data = self.getGeometry("pixels")
        epsg_data = self.getGeometry("EPSG:4326")
        relative_data = self.getGeometry("relative", parent_road_line) if parent_road_line else None

        if not pixel_data is None:
            xx, yy = pixel_data.coords.xy
            pixel_data = [{"x":float(x), "y":float(y)} for x, y in zip(xx, yy)]
        if not relative_data is None:
            xx, yy = relative_data.coords.xy
            relative_data = [{"x":float(x), "y":float(y)} for x, y in zip(xx, yy)]
        if not epsg_data is None:
            xx, yy = epsg_data.coords.xy
            epsg_data = [{"lat":float(x), "lon":float(y)} for x, y in zip(xx, yy)]

        adj_source = self.getAdjustmentSubfield()

        result = {
            "label":self.getLabel(),
            "label_source":self.getLabelSource(),
            "geometry_source":self.getGeometrySource(),
            "pixels":pixel_data,
            "relative": relative_data,
            "EPSG:4326":epsg_data,
            "adjusted":self.isAdjusted(),
            "adjustment_subfield": None if (adj_source is None) else adj_source.jsonify(),
            "id":self.getId()
        }

        return result

class LabeledRoadLine(RoadLine, CalibratedObject):
    def __init__(self,
                 identifier=None,
                 geometry_source=None,
                 label=None,
                 pixel_geom=None,
                 epsg_4326_geom=None,
                 adjusted=False,
                 adjustment_subfield=None,
                 confidence=1.0,
                 parent_road_line_identifier=None,
                 label_source=None):
        RoadLine.__init__(self, identifier=identifier,
                                geometry_source=geometry_source,
                                pixel_geom=pixel_geom,
                                epsg_4326_geom=epsg_4326_geom,
                                adjusted=adjusted,
                                adjustment_subfield=adjustment_subfield,
                                label=label,
                                label_source=label_source)
        CalibratedObject.__init__(self, confidence=confidence)

        if len(pixel_geom.coords.xy[0]) != 2:
            raise ValueError("Passed segment must have 2 verticies. Found " + str(len(pixel_geom.coords.xy)) + " verticies.")
        self._parent_road_line_identifier = parent_road_line_identifier

    def getParentRoadLineId(self):
        return self._parent_road_line_identifier

    def __str__(self):
        return str(self.jsonify())

    def jsonify(self, parent_road_line=None):
        pixel_data = self.getGeometry("pixels")
        epsg_data = self.getGeometry("EPSG:4326")
        relative_data = self.getGeometry("relative", parent_road_line) if parent_road_line else None

        if not pixel_data is None:
            xx, yy = pixel_data.coords.xy
            pixel_data = [{"x":float(x), "y":float(y)} for x, y in zip(xx, yy)]
        if not relative_data is None:
            xx, yy = relative_data.coords.xy
            relative_data = [{"x":float(x), "y":float(y)} for x, y in zip(xx, yy)]
        if not epsg_data is None:
            xx, yy = epsg_data.coords.xy
            epsg_data = [{"lat":float(x), "lon":float(y)} for x, y in zip(xx, yy)]

        adj_source = self.getAdjustmentSubfield()

        result_roadline = {
            "label":self.getLabel(),
            "label_source":self.getLabelSource(),
            "geometry_source":self.getGeometrySource(),
            "pixels":pixel_data,
            "relative":relative_data,
            "EPSG:4326":epsg_data,
            "parent_road_line_id":self._parent_road_line_identifier,
            "adjusted":self.isAdjusted(),
            "adjustment_subfield":None if (adj_source is None) else adj_source.jsonify(),
            "id":self.getId()
        }

        result_calibrated = CalibratedObject.jsonify(self)
        result = result_roadline | result_calibrated
        return result

class MultiLabeledRoadLine(RoadLine):
    def __init__(self,
                 identifier=None,
                 geometry_source=None,
                 label=None,
                 labeled_road_lines=None,
                 pixel_geom=None,
                 epsg_4326_geom=None,
                 adjusted=False,
                 adjustment_subfield=None,
                 label_source=None):
        super().__init__(identifier=identifier,
                         geometry_source=geometry_source,
                         pixel_geom=pixel_geom,
                         epsg_4326_geom=epsg_4326_geom,
                         adjusted=adjusted,
                         adjustment_subfield=adjustment_subfield,
                         label=label,
                         label_source=label_source)
        self._labeled_road_lines = labeled_road_lines

    def get_labeled_sub_lines(self):
        return self._labeled_road_lines

    def __str__(self):
        return str(self.jsonify()) + "->" + str([str(x) for x in self._labeled_road_lines])


class RoadAnnotationPolygon(SpatialObject):
    pass

def BuildingFactory(bda_annotations, label_source=None):
    buildings = []
    for building in bda_annotations:
        pixel_geom = convert_coords_to_shapely(building["pixels"], Polygon, lambda x:(x["x"], x["y"]))
        epsg_4326_geom = convert_coords_to_shapely(building["EPSG:4326"], Polygon, lambda x:(x["lat"], x["lon"]))

        #For backwards compatibility
        geometry_source = building["geometry_source"] if "geometry_source" in building.keys() else building["source"]

        if "view_id" not in building.keys() and "id" in building.keys():
            buildings.append(Building(identifier=building["id"],
                                      label=building["label"],
                                      geometry_source=geometry_source,
                                      pixel_geom=pixel_geom,
                                      epsg_4326_geom=epsg_4326_geom,
                                      adjusted=False,
                                      adjustment_subfield=None,
                                      label_source=label_source))
        elif "view_id" in building.keys() and "building_id" in building.keys():
            buildings.append(Building(identifier=building["view_id"],
                                      correlation_identifier=building["building_id"],
                                      label=building["label"],
                                      geometry_source=geometry_source,
                                      pixel_geom=pixel_geom,
                                      epsg_4326_geom=epsg_4326_geom,
                                      adjusted=False,
                                      adjustment_subfield=None,
                                      label_source=label_source,
                                      boundary=building["boundary"],
                                      filename=building["filename"]))
        else:
            raise ValueError("Invalid JSON payload for building. Required ID fields are missing.")

    return buildings

def RoadLineFactory(rda_road_lines, label_source=None):
    road_lines = []
    for road_line in rda_road_lines:
        pixel_geom = convert_coords_to_shapely(road_line["pixels"], LineString, lambda x:(x["x"], x["y"]))
        epsg_4326_geom = convert_coords_to_shapely(road_line["EPSG:4326"], LineString, lambda x:(x["lat"], x["lon"]))

        #For backwards compatibility
        geometry_source = road_line["geometry_source"] if "geometry_source" in road_line.keys() else road_line["source"]

        road_lines.append(RoadLine(identifier=road_line["id"],
                                   label=road_line["label"],
                                   geometry_source=geometry_source,
                                   pixel_geom=pixel_geom,
                                   epsg_4326_geom=epsg_4326_geom,
                                   adjusted=False,
                                   adjustment_subfield=None,
                                   label_source=label_source))
    return road_lines

def RoadAnnotationPolygonFactory(rda_annotation_polygons, label_source=None):
    road_annotation_polygons = []
    for rda_annotation_polygon in rda_annotation_polygons:
        pixel_geom = convert_coords_to_shapely(rda_annotation_polygon["pixels"], Polygon, lambda x:(x["x"], x["y"]))
        epsg_4326_geom = convert_coords_to_shapely(rda_annotation_polygon["EPSG:4326"], Polygon, lambda x:(x["lat"], x["lon"]))

        #For backwards compatibility
        geometry_source = rda_annotation_polygon["geometry_source"] if "geometry_source" in rda_annotation_polygon.keys() else rda_annotation_polygon["source"]

        road_annotation_polygons.append(RoadAnnotationPolygon(identifier=None,
                                                              label=rda_annotation_polygon["label"],
                                                              geometry_source=geometry_source,
                                                              pixel_geom=pixel_geom,
                                                              epsg_4326_geom=epsg_4326_geom,
                                                              adjusted=False,
                                                              adjustment_subfield=None,
                                                              label_source=label_source))
    return road_annotation_polygons

def MultiLabeledRoadLineFactory(road_lines, annotation_polygons, label_source=None):
    results = []

    annotation_area = shapely.unary_union([ap.getGeometry("pixels") for ap in annotation_polygons])

    #For every line that we have been passed
    for line in road_lines:
        labeled_road_lines = []
        for annotation_polygon in annotation_polygons:
            #If there is an intersection, then we track it, ortherwise we ignore it and move on
            if shapely.intersects(line.getGeometry("pixels"), annotation_polygon.getGeometry("pixels")):
                labeled_road_line_segment = shapely.intersection(line.getGeometry("pixels"), annotation_polygon.getGeometry("pixels"))
                labeled_segments = []
                if isinstance(labeled_road_line_segment, shapely.LineString):
                    labeled_segments = [labeled_road_line_segment]
                if isinstance(labeled_road_line_segment, shapely.MultiLineString):
                    labeled_segments = list(labeled_road_line_segment.geoms)

                for labeled_segment in labeled_segments:
                    if labeled_segment.length > 0:
                        labeled_road_lines.append(LabeledRoadLine(label=annotation_polygon.getLabel(),
                                                                  pixel_geom=labeled_segment,
                                                                  adjusted=line.isAdjusted(),
                                                                  adjustment_subfield=None,
                                                                  parent_road_line_identifier=line.getId(),
                                                                  label_source=label_source))

        clear_road_line_sections = shapely.difference(line.getGeometry("pixels"), annotation_area)
        clear_segments = []
        if isinstance(clear_road_line_sections, shapely.LineString):
            clear_segments = [clear_road_line_sections]
        if isinstance(clear_road_line_sections, shapely.MultiLineString):
            clear_segments = list(clear_road_line_sections.geoms)

        for clear_segment in clear_segments:
            if clear_segment.length > 0:
                labeled_road_lines.append(LabeledRoadLine(label=ROAD_LINE,
                                                          pixel_geom=clear_segment,
                                                          adjusted=line.isAdjusted(),
                                                          adjustment_subfield=None,
                                                          parent_road_line_identifier=line.getId(),
                                                          label_source=label_source))

        results.append(MultiLabeledRoadLine(identifier=line.getId(),
                                            label=line.getLabel(),
                                            geometry_source=line.getGeometrySource(),
                                            labeled_road_lines=labeled_road_lines,
                                            pixel_geom=line.getGeometry("pixels"),
                                            epsg_4326_geom=line.getGeometry("EPSG:4326"),
                                            adjusted=line.isAdjusted(),
                                            adjustment_subfield=line.getAdjustmentSubfield(),
                                            label_source=label_source))
    return results
