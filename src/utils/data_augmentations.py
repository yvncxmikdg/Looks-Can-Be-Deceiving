import torch
import shapely
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from modeling.Spatial import Building, LabeledRoadLine, MultiLabeledRoadLine
from modeling.Alignment import Adjustment

NORMALIZATION_MEAN = torch.tensor([0.4915, 0.4823, 0.4468])
NORMALIZATION_STD = torch.tensor([0.2470, 0.2435, 0.2616])

# This class contains the logic to convert BDA and RDA data into keypoints for faster data augmentation
# Working with keypoint data allows us to work with points (cheap) compared the pixel masks (expensive)
# These functions are used to convert the data into the keypoint space, and then update it again when
# After data augmentation has been applied to the data.
class KeyPointConversionStrategy:
    def __init__(self):
        pass
    @staticmethod
    def get_keypoints_from_sample(sample):
        raise NotImplementedError("get_keypoints_from_sample must be implemented by a subclass.")
    @staticmethod
    def apply_keypoint_augmentations_to_sample(keypoints, sample):
        raise NotImplementedError("apply_keypoint_augmentations_to_sample must be implemented by a subclass.")

# Keypoint conversion logic that is specific to the RDA data
class KeyPointConversionStrategyRDA(KeyPointConversionStrategy):
    @staticmethod
    def get_keypoints_from_sample(sample):
        keypoints = []
        for multilabeled_roadline in sample.getRoadLines():
            keypoints.extend(geoms_to_keypoints([multilabeled_roadline.getGeometry("pixels")]))
            keypoints.extend(geoms_to_keypoints([b.getGeometry("pixels") for b in multilabeled_roadline.get_labeled_sub_lines()]))
        return keypoints

    @staticmethod
    def apply_keypoint_augmentations_to_sample(keypoints, sample):
        augmented_geoms, _ = keypoints_to_labeled_roadline_geoms(keypoints, sample)
        augmented_roadlines = update_roadlines_with_augmented_geometry(sample.getRoadLines(), augmented_geoms)
        sample.setRoadLines(augmented_roadlines)

# Keypoint conversion logic that is specific to the BDA data
class KeyPointConversionStrategyBDA(KeyPointConversionStrategy):
    @staticmethod
    def get_keypoints_from_sample(sample):
        keypoints = []

        buildings_keypoints = geoms_to_keypoints([b.getGeometry("pixels") for b in sample.getBuildings()])
        keypoints.extend(buildings_keypoints)

        for view in sample.getViews():
            adjustment_keypoints = geoms_to_keypoints([adj.getGeometry() for adj in view.getAdjustments()])
            keypoints.extend(adjustment_keypoints)

        return keypoints

    @staticmethod
    def apply_keypoint_augmentations_to_sample(keypoints, sample):
        augmented_geoms, adjustment_start_i = keypoints_to_building_geoms(keypoints, sample)
        augmented_buildings = update_buildings_with_augmented_geometry(sample.getBuildings(), augmented_geoms)
        sample.setBuildings(augmented_buildings)

        delta_i = 0
        for view in sample.getViews():
            augmented_adjustment_geoms, view_adjustments_end = keypoints_to_adjustment_geoms(keypoints[adjustment_start_i+delta_i:], view)
            delta_i += view_adjustments_end
            unaugmented_adjustments = view.getAdjustments()
            augmented_adjustments = []
            for augmented_adjustment_geom, unagumented_adjustment in zip(augmented_adjustment_geoms, unaugmented_adjustments):
                a = Adjustment(*augmented_adjustment_geom.coords[0],
                               *augmented_adjustment_geom.coords[1],
                               unagumented_adjustment.getId(),
                               unagumented_adjustment.isAttributable())
                augmented_adjustments.append(a)

            view.setAdjustments(augmented_adjustments)

def geoms_to_keypoints(geoms):
    keypoints = []
    for geom in geoms:
        k = []
        iterator = None
        if isinstance(geom, shapely.LineString):
            iterator = geom.coords
        else:
            iterator = geom.exterior.coords
        for x, y in iterator:
            k.append([x,y])
        keypoints.extend(k)
    return keypoints

def keypoints_to_building_geoms(keypoints, sample):
    i = 0
    geoms = []
    for building in sample.getBuildings():
        verts_in_building_polygon = len(building.getGeometry("pixels").exterior.coords)
        verts = keypoints[i:i+verts_in_building_polygon]
        geoms.append(shapely.Polygon(verts))
        i += verts_in_building_polygon
    return geoms, i

def keypoints_to_labeled_roadline_geoms(keypoints, sample):
    i = 0
    lines = []
    for multilabeled_roadline in sample.getRoadLines():
        verts_in_line = len(multilabeled_roadline.getGeometry("pixels").coords)
        verts = keypoints[i:i+verts_in_line]
        multilabeled_geom = {"line":shapely.LineString(verts), "sublines":[]}
        i += verts_in_line
        for labeled_sub_line in multilabeled_roadline.get_labeled_sub_lines():
            verts_in_sub_line = len(labeled_sub_line.getGeometry("pixels").coords)
            verts = keypoints[i:i+verts_in_sub_line]
            multilabeled_geom["sublines"].append(shapely.LineString(verts))
            i += verts_in_sub_line
        lines.append(multilabeled_geom)
    return lines, i

def keypoints_to_adjustment_geoms(keypoints, view):
    i = 0
    lines = []
    for _ in view.getAdjustments():
        lines.append(shapely.LineString(keypoints[i:i+2]))
        i += 2
    return lines, i

def update_buildings_with_augmented_geometry(buildings, augmented_building_geoms):
    result = []
    for building, augmented_building_geom in zip(buildings, augmented_building_geoms):
        b = Building(identifier=building.getId(),
                     label=building.getLabel(),
                     geometry_source=building.getGeometrySource(),
                     pixel_geom=augmented_building_geom,
                     epsg_4326_geom=building.getGeometry("EPSG:4326"),
                     adjusted=building.isAdjusted(),
                     adjustment_subfield=building.getAdjustmentSubfield())
        result.append(b)
    return result

def update_roadlines_with_augmented_geometry(roadlines, augmented_roadline_geoms):
    result = []
    for roadline, augmented_roadline_lines in zip(roadlines, augmented_roadline_geoms):
        augmented_sub_roadline_objects = []
        for subline_object, augmented_subline_geom in zip(roadline.get_labeled_sub_lines(), augmented_roadline_lines["sublines"]):
            augmented_sub_roadline_objects.append(LabeledRoadLine(identifier=subline_object.getId(),
                                                                  label=subline_object.getLabel(),
                                                                  geometry_source=subline_object.getGeometrySource(),
                                                                  pixel_geom=augmented_subline_geom,
                                                                  epsg_4326_geom=subline_object.getGeometry("EPSG:4326"),
                                                                  adjusted=subline_object.isAdjusted(),
                                                                  adjustment_subfield=subline_object.getAdjustmentSubfield()))
        result.append(MultiLabeledRoadLine(identifier=roadline.getId(),
                                           geometry_source=roadline.getGeometrySource(),
                                           label=roadline.getLabel(),
                                           labeled_road_lines=augmented_sub_roadline_objects,
                                           pixel_geom=augmented_roadline_lines["line"],
                                           epsg_4326_geom=None,
                                           adjusted=roadline.isAdjusted(),
                                           adjustment_subfield=roadline.getAdjustmentSubfield()))
    return result

def get_normalize_transform():
    return A.Normalize(NORMALIZATION_MEAN.tolist(), NORMALIZATION_STD.tolist())
def get_unnormalize_transform():
    def unnorm(image):
        return ((image * NORMALIZATION_STD.tolist()) + NORMALIZATION_MEAN.tolist())
    return unnorm

def get_tensor_transform():
    return ToTensorV2(p=1)

def get_train_transforms():
    augmentations = A.Compose(
        [
            A.D4(p=1.0),
            # pylint: disable-next=no-member
            A.Perspective(scale=(0.00, 0.04), keep_size=True, fit_output=True, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.6, p=1.0),
            A.RandomToneCurve(scale=0.3, per_channel=False, p=1.0),
            A.ColorJitter(hue=0.03, saturation=[0.3, 4], p=1.0),
            A.GaussNoise(std_range=[0, 0.03], p=1.0),
            A.ToGray(method="average", p=0.1),
            A.MedianBlur(blur_limit=[3,5], p=0.1)
        ],
        p=1.0,
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    return augmentations


def get_valid_transforms():
    return A.Compose([], p=1.0)

def get_inference_transforms():
    return A.Compose([], p=1.0)
