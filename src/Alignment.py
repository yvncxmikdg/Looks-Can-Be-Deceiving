import math
import torch
from shapely import Polygon, LineString, Point

from modeling.Spatial import Building, RoadLine
from modeling.constants import SOURCE_CUSTOM


class Adjustment:
    def __init__(self, x1, y1, x2, y2, identifier=None, attributable=True, dx_std=0.0, dy_std=0.0):
        self._x1 = x1
        self._x2 = x2
        self._y1 = y1
        self._y2 = y2
        self._dx_std = dx_std
        self._dy_std = dy_std
        self._identifier = identifier
        self._attributable = attributable
        self._adj = LineString([[self._x1, self._y1], [self._x2, self._y2]])

    def getId(self):
        return self._identifier

    def getStartPoint(self):
        return Point(self._x1, self._y1)

    def getEndPoint(self):
        return Point(self._x2, self._y2)

    def getDx(self):
        return self._x2 - self._x1

    def getDy(self):
        return self._y2 - self._y1

    def getDxStd(self):
        return self._dx_std

    def getDyStd(self):
        return self._dy_std

    def getGeometry(self):
        return self._adj

    def getAngle(self):
        return math.degrees(math.atan2(self.getDx(), self.getDy()))

    def getMagnitude(self):
        return math.sqrt(self.getDx() ** 2 + self.getDy() ** 2)

    def isAttributable(self):
        return self._attributable

    def jsonify(self):
        return [[self._x1, self._y1], [self._x2, self._y2]]

    def applyToPoint(self, p):
        return Point(p.x + self.getDx(), p.y + self.getDy())

    def asTorchTensor(self, end_point_coords="delta"):
        if end_point_coords == "delta":
            return torch.tensor([self._x1,
                                 self._y1,
                                 self.getDx(),
                                 self.getDy(),
                                 self.getDxStd(),
                                 self.getDyStd(),
                                 1.0 if self.isAttributable() else 0.0])
        if end_point_coords == "coords":
            return torch.tensor([self._x1,
                                 self._y1,
                                 self._x2,
                                 self._y2,
                                 self.getDxStd(),
                                 self.getDyStd(),
                                 1.0 if self.isAttributable() else 0.0])
        if end_point_coords == "polar":
            return torch.tensor([self._x1,
                                 self._y1,
                                 self.getMagnitude(),
                                 self.getAngle(),
                                 self.getDxStd(),
                                 self.getDyStd(),
                                 1.0 if self.isAttributable() else 0.0])
        raise ValueError("Unknown value \"" + end_point_coords + "\" passed to asTorchVector() options are: coords, delta, polar")

    def __str__(self):
        return (
            "[Adjustment ("
            + str(self._x1)
            + ", "
            + str(self._y1)
            + ") -> ("
            + str(self._x2)
            + ", "
            + str(self._y2)
            + ") | "
            + str(self.getAngle())
            + " "
            + str(self.getMagnitude())
            + "]"
        )


class AdjustmentVectorField:
    def __init__(self, adjustments):
        self._adjustments = adjustments
        for a in self._adjustments:
            assert isinstance(a, Adjustment)

    def getAdjustments(self):
        return self._adjustments

    def jsonify(self):
        result = []
        for adj in self._adjustments:
            result.append(adj.jsonify())
        return result

    def getAdjustmentForPoint(self, point):
        min_dist = float("inf")
        best_adjustment = None
        for adjustment in self.getAdjustments():
            start_point = adjustment.getStartPoint()
            dist = math.dist([point.x, point.y], [start_point.x, start_point.y])
            if dist < min_dist:
                min_dist = dist
                best_adjustment = adjustment
        return best_adjustment

    def getAdjustmentForBuilding(self, building):
        min_dist = float("inf")
        thresh_distance = float("inf")
        best_adjustment = None
        candidate_adjustments = self.getAdjustments()
        building_geometry = building.getGeometry("pixels")
        for x, y in building_geometry.exterior.coords:
            next_candidates = []
            for adjustment in candidate_adjustments:
                start_point = adjustment.getStartPoint()
                dist = math.dist([x, y], [start_point.x, start_point.y])
                if dist < thresh_distance:
                    if dist < min_dist:
                        min_dist = dist
                        best_adjustment = adjustment
                    thresh_distance = min_dist + building_geometry.length / 2
                    next_candidates.append(adjustment)
            candidate_adjustments = next_candidates
        return best_adjustment

    def adjustRoadLine(self, road_line):
        adjusted_road_line_pixel_coordinates = []
        adjustments_used = []
        # Does not adjust custom road lines
        if road_line.getGeometrySource() != SOURCE_CUSTOM:
            for x, y in road_line.getGeometry("pixels").coords:
                point = Point(x, y)
                best_adjustment = self.getAdjustmentForPoint(point)
                point_adjusted = best_adjustment.applyToPoint(point)
                adjusted_road_line_pixel_coordinates.append(point_adjusted)
                adjustments_used.append(best_adjustment)

            return RoadLine(
                identifier=road_line.getId(),
                label=road_line.getLabel(),
                geometry_source=road_line.getGeometrySource(),
                pixel_geom=LineString(adjusted_road_line_pixel_coordinates),
                epsg_4326_geom=road_line.getGeometry("EPSG:4326"),
                adjusted=True,
                adjustment_subfield=AdjustmentVectorSubfield(adjustments_used),
            )

        for x, y in road_line.getGeometry("pixels").coords:
            adjusted_road_line_pixel_coordinates.append(Point(x, y))

        return RoadLine(
            identifier=road_line.getId(),
            label=road_line.getLabel(),
            geometry_source=road_line.getGeometrySource(),
            pixel_geom=LineString(adjusted_road_line_pixel_coordinates),
            epsg_4326_geom=road_line.getGeometry("EPSG:4326"),
            adjusted=True,
            adjustment_subfield=AdjustmentVectorSubfield([Adjustment(0,0,0,0)]),
        )

    def adjustBuilding(self, building):
        # Does not adjust custom buildings
        if building.getGeometrySource() != SOURCE_CUSTOM:
            adj = self.getAdjustmentForBuilding(building)

            adjusted_coordinates = []
            for x, y in building.getGeometry("pixels").exterior.coords:
                adjusted_coordinates.append([x + adj.getDx(), y + adj.getDy()])
            adjusted_polygon = Polygon(adjusted_coordinates)

            return Building(
                identifier=building.getId(),
                correlation_identifier=building.getCorrelationId(),
                label=building.getLabel(),
                geometry_source=building.getGeometrySource(),
                pixel_geom=adjusted_polygon,
                epsg_4326_geom=building.getGeometry("EPSG:4326"),
                adjusted=True,
                adjustment_subfield=AdjustmentVectorSubfield([adj]),
                label_source=building.getLabelSource(),
                filename=building.getFile(),
            )
        return Building(
            identifier=building.getId(),
            correlation_identifier=building.getCorrelationId(),
            label=building.getLabel(),
            geometry_source=building.getGeometrySource(),
            pixel_geom=Polygon(building.getGeometry("pixels").exterior.coords),
            epsg_4326_geom=building.getGeometry("EPSG:4326"),
            adjusted=True,
            adjustment_subfield=AdjustmentVectorSubfield([Adjustment(0,0,0,0)]),
            label_source=building.getLabelSource(),
            filename=building.getFile(),
        )

    def adjustBuildings(self, buildings):
        if buildings is None:
            return None
        adjusted_buildings = []
        for building in buildings:
            adjusted_buildings.append(self.adjustBuilding(building))
        return adjusted_buildings

    def adjustRoadLines(self, road_lines):
        if road_lines is None:
            return None
        adjusted_road_lines = []
        for road_line in road_lines:
            adjusted_road_lines.append(self.adjustRoadLine(road_line))
        return adjusted_road_lines

    def getAverageAdjustment(self, start_x=0, start_y=0):
        dx_avg = sum(a.getDx() for a in self._adjustments) / len(self._adjustments)
        dy_avg = sum(a.getDy() for a in self._adjustments) / len(self._adjustments)
        return Adjustment(start_x, start_y, start_x+dx_avg, start_y+dy_avg)


class AdjustmentVectorSubfield(AdjustmentVectorField):
    pass

def AdjustmentVectorFieldFactory(adjustments_data):
    adjustments = []
    if len(adjustments_data) == 0:
        adjustments.append(Adjustment(0, 0, 0, 0))
    else:
        for adj in adjustments_data:
            adjustments.append(Adjustment(adj[0][0], adj[0][1], adj[1][0], adj[1][1]))
    return AdjustmentVectorField(adjustments)
