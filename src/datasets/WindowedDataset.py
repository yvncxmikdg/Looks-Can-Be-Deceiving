import time

from torch.utils.data import Dataset, get_worker_info

from modeling.utils.data_augmentations import (
    get_valid_transforms,
    get_normalize_transform,
    get_tensor_transform,
)
from modeling.utils.system_telemetry import get_worker_heap_stats

from modeling.constants import SAMPLE_GENERATION_TIMING_PREFIX, WORKER_MEMORY_PREFIX


class WindowedDataset(Dataset):
    def __init__(
        self,
        dataset_adaptor,
        mask_generation_strategy,
        transforms=get_valid_transforms(),
        normalize_transform=get_normalize_transform(),
        tensor_transform=get_tensor_transform(),
        normalize=True,
    ):
        self.ds = dataset_adaptor
        self.mask_generation_strategy = mask_generation_strategy
        self.transforms = transforms
        self.normalize_transform = normalize_transform
        self.tensor_transform = tensor_transform
        self.__normalize_images = normalize

    def set_normalization(self, normalize_images):
        self.__normalize_images = bool(normalize_images)

    def is_normalized(self):
        return self.__normalize_images

    def __getitem__(self, index):
        t_start = time.time()

        # Get the sample object from the dataset adaptor
        sample_object = self.ds.generate_sample(index)

        # Get the view from this object so we can augment it
        view = sample_object.getViews()[0]

        t_get_view = time.time()

        # Convert the building polygons to keypoints, and get their indicies and types
        keypoints = self.ds.keypoint_conversion_strategy.get_keypoints_from_sample(sample_object)

        # Transform the imagery, along side the keypoints
        non_normalized_sample_tfm = self.transforms(image=view.getRawImagery().numpy(), keypoints=keypoints)

        # Reconstruct the keypoints into their original spatial objects
        self.ds.keypoint_conversion_strategy.apply_keypoint_augmentations_to_sample(non_normalized_sample_tfm["keypoints"], sample_object)

        t_agument = time.time()

        # The normalize the imagery, only and convert it to a tensor
        sample_tfm = non_normalized_sample_tfm
        if self.__normalize_images:
            sample_tfm = self.normalize_transform(image=sample_tfm["image"])
        sample_tfm = self.tensor_transform(image=sample_tfm["image"])

        t_tensor_transform = time.time()

        # Now, with the spatial objects in the sample augmented, draw them on the masks
        if self.ds.mask_scale_strategy == "pixel":
            self.mask_generation_strategy.initialize_masking_strategy(self.ds.mask_x,
                                                                      self.ds.mask_y,
                                                                      sample_tfm["image"].shape[1],
                                                                      sample_tfm["image"].shape[2])
        elif self.ds.mask_scale_strategy == "spatial":
            self.mask_generation_strategy.initialize_masking_strategy(sample_tfm["image"].shape[1],
                                                                      sample_tfm["image"].shape[2],
                                                                      sample_tfm["image"].shape[1],
                                                                      sample_tfm["image"].shape[2])
        label_data, query_data = self.mask_generation_strategy.compute_mask_from_sample(sample_object)

        t_draw_polygons = time.time()

        # Populate the target objects with the data we have just generated
        view.setInputImagery(sample_tfm["image"])
        view.setRawImagery(non_normalized_sample_tfm["image"])
        view.setQueries(query_data)
        view.setLabels(label_data)

        sample_object.setMetadataEntry(SAMPLE_GENERATION_TIMING_PREFIX + "apply_augmentations_time", t_agument - t_get_view)
        sample_object.setMetadataEntry(SAMPLE_GENERATION_TIMING_PREFIX + "convert_to_tensor_time", t_tensor_transform - t_agument)
        sample_object.setMetadataEntry(SAMPLE_GENERATION_TIMING_PREFIX + "generate_label_masks_time", t_draw_polygons - t_tensor_transform)

        t_done = time.time()
        sample_object.setMetadataEntry(SAMPLE_GENERATION_TIMING_PREFIX + "total_time", t_done - t_start)

        # Stamp this worker's allocator gauges (in-use vs retained-free heap) onto the sample so
        # they flow through the batch metadata to TensorBoard. This is the discriminator for the
        # worker-RSS climb: in-use rising = a genuine leak; retained-free rising = allocator
        # retention. Keyed by worker id so each worker charts as its own series. Linux-only
        # (returns {} elsewhere) and cheap relative to sample generation.
        worker_info = get_worker_info()
        worker_key = f"w{worker_info.id}" if worker_info is not None else "main"
        for stat_name, value in get_worker_heap_stats().items():
            sample_object.setMetadataEntry(WORKER_MEMORY_PREFIX + worker_key + "_" + stat_name, value)

        # Return the object
        return sample_object

    def __len__(self):
        return len(self.ds)
