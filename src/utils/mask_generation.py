import numpy as np

from modeling.utils.sample_generator_utils import draw_labeled_road_lines_on_mask, draw_buildings_on_mask

class MaskingStrategy:
    def __init__(self, mask_x=None, mask_y=None, tile_x=None, tile_y=None):
        self.mask_x = None
        self.mask_y = None
        self.tile_x = None
        self.tile_y = None
        self.initialize_masking_strategy(mask_x, mask_y, tile_x, tile_y)

    def initialize_masking_strategy(self, mask_x, mask_y, tile_x, tile_y):
        self.mask_x = mask_x
        self.mask_y = mask_y
        self.tile_x = tile_x
        self.tile_y = tile_y

    def compute_mask_from_sample(self, sample):
        raise NotImplementedError("compute_mask_from_sample must be implemented by a subclass")

class MaskingStrategyBDA(MaskingStrategy):
    def compute_mask_from_sample(self, sample):
        label_data = draw_buildings_on_mask(
            sample.getBuildings(),
            0,
            0,
            self.tile_x,
            self.tile_y,
            self.mask_x,
            self.mask_y,
            class_color_map=sample.getLabelMap(),
            draw_color=False
        )
        #And get their labels/queries from them
        label_data = np.copy(label_data)
        query_data = np.minimum(label_data, 1)

        return label_data, query_data

class MaskingStrategyRDA(MaskingStrategy):
    def __init__(self, road_width_pixels):
        super().__init__()
        self._road_width_pixels = road_width_pixels
    def compute_mask_from_sample(self, sample):
        label_data = draw_labeled_road_lines_on_mask(
            sample.getRoadLines(),
            0,
            0,
            self.tile_x,
            self.tile_y,
            self.mask_x,
            self.mask_y,
            class_color_map=sample.getLabelMap(),
            draw_color=False,
            road_width_pixels=self._road_width_pixels
        )
        #And get their labels/queries from them
        label_data = np.copy(label_data)
        query_data = np.minimum(label_data, 1)

        return label_data, query_data
