import time
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from shapely.ops import unary_union
from shapely.affinity import translate

from dataset.constants import BDA_DAMAGE_CLASSES
from modeling.Models.TaskModuleBDA import TaskModuleBDA
from modeling.Models.OrthoInferenceWrapper import fuse_bda_tiled_inference
from modeling.Alignment import Adjustment, AdjustmentVectorField
from modeling.utils.sample_generator_utils import draw_buildings_on_mask, draw_adjustments_on_mask, draw_objects_on_mask, \
                                                  scale_pixel_coords, offset_pixel_coords
from modeling.utils.data_augmentations import geoms_to_keypoints
from modeling.utils.inspection_utils import inspect_image, inspect_labels, inspect_polygons_on_image
from modeling.utils.alignment_utils import nearest_flow
from modeling.Models.ModelDatum import Y_HAT_SEGMENTATION_UNMASKED, DISPLACEMENT_FIELD, BUILDINGS, DO_SOFTMAX

from modeling.constants import (
    TRAINING_STEP_METADATA_TIME_INTER_STEP,
    TRAINING_STEP_METADATA_TIME_INTRA_STEP,
    TRAINING_STEP_METADATA_TIME_PREPROCESS,
    TRAINING_STEP_METADATA_TIME_FORWARD,
    TRAINING_STEP_METADATA_TIME_LOSS,
    TRAINING_STEP_METADATA_TIME_LOG,
    TRAINING_STEP_METADATA_TIME_INIT
)

def getBatchedAdjustments(predicted_batched_adjustments):
    result = []
    for adjustments in predicted_batched_adjustments:
        result.append([])
        if not adjustments is None:
            for adjustment in adjustments:
                result[-1].append(Adjustment(float(adjustment[0]),
                                             float(adjustment[1]),
                                             float(adjustment[0]+adjustment[2]),
                                             float(adjustment[1]+adjustment[3])))
    return result

def plot_batched_alignment_vectors_on_mask(individual_adjustments, initial_mask):
    _, x_dim, y_dim = initial_mask.shape

    batched_vector_field_and_individual_adjustment_masks = []
    for i, b_adj in enumerate(individual_adjustments):
        mask = torch.tensor(draw_adjustments_on_mask(b_adj,
                                                     0,
                                                     0,
                                                     x_dim,
                                                     y_dim,
                                                     x_dim,
                                                     y_dim,
                                                     initial_mask=initial_mask[i]*0.5,
                                                     adjustment_width_max=2))
        batched_vector_field_and_individual_adjustment_masks.append(mask)

    return torch.stack(batched_vector_field_and_individual_adjustment_masks)

def compute_batched_building_iou(batched_buildings_a, batched_buildings_b):
    batched_ious = []
    for batch_a, batch_b in zip(batched_buildings_a, batched_buildings_b):
        ious = []
        for building_a, building_b in zip(batch_a, batch_b):
            building_a_polygon = building_a.getGeometry("pixels")
            building_b_polygon = building_b.getGeometry("pixels")
            intersect = building_a_polygon.intersection(building_b_polygon).area
            union = building_a_polygon.union(building_b_polygon).area
            ious.append(intersect / union)
        if len(ious) > 0:
            batched_ious.append(ious)
        else:
            batched_ious.append(None)
    return batched_ious

def add_offset_to_batched_buildings(batched_buildings, batched_frames, x_dim, y_dim):
    result = []
    for frame_geom, buildings in zip(batched_frames, batched_buildings):
        result.append([])
        for building in buildings:
            #Multiply by -1 because we want to add these offsets instead of subtract them
            offset_building_geom = offset_pixel_coords(building.getGeometry("pixels"), -1*(frame_geom.centroid.x-x_dim), -1*(frame_geom.centroid.y-y_dim))
            building_copy = deepcopy(building)
            building_copy.setGeometry(offset_building_geom, "pixels")
            result[-1].append(building_copy)
    return result

def get_adjustment_from_mask(vector_field, building, building_mask, source_dim_x, source_dim_y):
    count_relevant_vectors = torch.sum(building_mask, dim=None)
    if count_relevant_vectors == 0:
        count_relevant_vectors = 1.0

    #Compute the predicted adjustment
    centroid = building.getGeometry("pixels").centroid
    masked_adjustments = vector_field.mul(building_mask.to(vector_field.device))
    dx = torch.sum(masked_adjustments[0, :, :], dim=None)/count_relevant_vectors
    dy = torch.sum(masked_adjustments[1, :, :], dim=None)/count_relevant_vectors
    std_dx = torch.sum(masked_adjustments[0, :, :] * (masked_adjustments[0, :, :] - dx)**2)/count_relevant_vectors
    std_dy = torch.sum(masked_adjustments[1, :, :] * (masked_adjustments[1, :, :] - dy)**2)/count_relevant_vectors
    adjustment = torch.stack([torch.tensor(centroid.x).to(dx.device),
                              torch.tensor(centroid.y).to(dy.device),
                              dx*source_dim_x,
                              dy*source_dim_y,
                              std_dx.detach().to(dx.device),
                              std_dy.detach().to(dx.device),
                              torch.tensor(1.0).to(dx.device)])
    # Todo, consider having the model output quantiles instead of trying to infer them from the adjustment distribtuion

    return adjustment

def get_adjustment_from_vertices(vector_field, building, source_dim_x, source_dim_y):
    centroid = building.getGeometry("pixels").centroid

    x_scale_factor = vector_field.shape[1]/source_dim_x
    y_scale_factor = vector_field.shape[2]/source_dim_y
    polygon_scaled = scale_pixel_coords(building.getGeometry("pixels"), x_scale_factor, y_scale_factor)
    points_list = geoms_to_keypoints([polygon_scaled])

    points = torch.tensor(points_list).to(vector_field.device)
    count_relevant_vectors = len(points_list)

    flow_tensor = nearest_flow(vector_field, points)

    dx = torch.sum(flow_tensor[:, 0], dim=None)/count_relevant_vectors
    dy = torch.sum(flow_tensor[:, 1], dim=None)/count_relevant_vectors
    adjustment = torch.stack([torch.tensor(centroid.x).to(dx.device),
                              torch.tensor(centroid.y).to(dy.device),
                              dx*source_dim_x,
                              dy*source_dim_y,
                              torch.tensor(0.0).to(dx.device),
                              torch.tensor(0.0).to(dx.device),
                              torch.tensor(1.0).to(dx.device)])
    return adjustment

def get_batched_building_label_weights(batched_buildings, weights_dict):
    result = []
    for buildings in batched_buildings:
        weights = []
        for building in buildings:
            weights.append(weights_dict[building.getLabel()])
        if len(weights) > 0:
            result.append(torch.tensor(weights))
        else:
            result.append(None)
    return result

class TaskModuleBDAADJ(TaskModuleBDA):

    def get_building_adjustment(self,
                                batched_vector_field,
                                batched_buildings,
                                batched_building_masks,
                                strategy="mask",
                                source_dim_x=2048,
                                source_dim_y=2048):
        batched_adjustments = []
        for vf, buildings, building_masks in zip(batched_vector_field, batched_buildings, batched_building_masks):
            adjustments = []
            for building, building_mask in zip(buildings, building_masks):
                adjustment = None
                if "mask" in strategy:
                    adjustment = get_adjustment_from_mask(vf, building, building_mask, source_dim_x, source_dim_y)
                elif "vert" in strategy:
                    adjustment = get_adjustment_from_vertices(vf, building, source_dim_x, source_dim_y)

                adjustments.append(adjustment)
            if len(adjustments) > 0:
                batched_adjustments.append(torch.stack(adjustments))
            else:
                batched_adjustments.append(None)
        return batched_adjustments

    def get_tensor_building_masks(self,
                                  batched_buildings,
                                  source_dim_x,
                                  source_dim_y,
                                  target_scale_x,
                                  target_scale_y,
                                  target_device=None):
        actual_target_device = self._device if not target_device is None else target_device
        result = []
        for buildings in batched_buildings:
            result.append([])
            for building in buildings:
                building_mask = draw_objects_on_mask([building],
                                                       0,
                                                       0,
                                                       source_dim_x,
                                                       source_dim_y,
                                                       target_scale_x,
                                                       target_scale_y,
                                                       channels=1,
                                                       geometry_accessor=lambda x: x.getGeometry("pixels"))
                building_mask_tensor = torch.tensor(building_mask).to(actual_target_device)
                result[-1].append(building_mask_tensor)
        return result

    def apply_adjustment_tensors_to_buildings(self, batched_adjustment_tensors, batched_buildings):
        result = []
        for adjustment_tensors, buildings in zip(batched_adjustment_tensors, batched_buildings):
            aligned_buildings = []
            if len(buildings) > 0:
                for adjustment_tensor, building in zip(adjustment_tensors, buildings):
                    if not adjustment_tensor is None:
                        start_x = float(adjustment_tensor[0])
                        start_y = float(adjustment_tensor[1])
                        dx = float(adjustment_tensor[2])
                        dy = float(adjustment_tensor[3])
                        a = Adjustment(start_x, start_y, start_x+dx, start_y+dy)
                        avf = AdjustmentVectorField([a])
                        aligned_buildings.append(avf.adjustBuilding(building))
                    else:
                        aligned_buildings.append(building)
            result.append(aligned_buildings)
        return result

    def get_query_label_tensor(self, batched_adjusted_buildings, mask_x, mask_y):
        labels = []
        queries = []
        for buildings in batched_adjusted_buildings:
            label_mask = draw_buildings_on_mask(buildings,
                                                0,
                                                0,
                                                mask_x,
                                                mask_y,
                                                mask_x,
                                                mask_y,
                                                class_color_map=self.output_label_map,
                                                draw_color=False)
            labels.append(torch.tensor(label_mask))
            queries.append(torch.tensor(np.minimum(label_mask, 1)))
        labels_tensor = torch.stack(labels).to(self._device).long()
        query_tensor = torch.stack(queries).to(self._device).long()

        return labels_tensor, query_tensor

    def log_timing_pixel_polygon_stats(self, batch):
        # TODO: telemetry hook referenced by training_step but never implemented on main.
        # Stubbed as a no-op so the training loop runs; implement proper logging if desired.
        pass

    def on_validation_epoch_start(self):
        # BDAADJ accumulates a list of polygon parts per building id, so both
        # accumulators must be defaultdict(list) (the base initializes labels as a plain dict).
        self.validation_step_outputs = defaultdict(list)
        self.validation_step_labels = defaultdict(list)
        self.validation_loss = []

    def on_predict_epoch_start(self):
        self.predict_step_outputs = defaultdict(list)
        self.predicted_labels = {}

    def format_bdaadj_model_input(self, batch):
        # BDAADJ alignment models consume the shared ModelInput contract: the rgb+mask channels
        # (CHANNEL_INPUT) plus the building polygons they warp through the displacement chain.
        model_input = self.format_batched_data_to_model_input(batch)
        model_input.setField(BUILDINGS, batch.getBatchedBuildings())
        model_input.setField(DO_SOFTMAX, True)
        return model_input

    # pylint: disable-next=arguments-differ, unused-argument
    def predict_step(self, batch, batch_idx):
        # For every building, average the predicted displacement field over its mask to get an
        # adjustment, anchor it to the original (unadjusted) ortho building, and store both the
        # adjustment vector (fusion strategy 1) and the adjusted polygon (fusion strategy 2) so
        # the per-tile predictions can be fused at epoch end. The displacement is converted from
        # tile-pixel space to ortho-pixel space via the GSD ratio (tile gsd / ortho gsd), so the
        # output is in the annotation coordinate space (ratio is 1 today, correct under tiling).
        _, _, width, height = batch.getBatchedImagery().shape

        model_output = self.model.forward(self.format_bdaadj_model_input(batch))
        if model_output.contains(DISPLACEMENT_FIELD):
            y_hat_adj = model_output[DISPLACEMENT_FIELD]
            adjustment_building_masks = self.get_tensor_building_masks(batch.getBatchedBuildings(), width, height, y_hat_adj.shape[2], y_hat_adj.shape[3])
            pred_adjustments = self.get_building_adjustment(y_hat_adj, batch.getBatchedBuildings(), adjustment_building_masks)
        else:
            pred_adjustments = [None for _ in batch.getBatchedBuildings()]

        for adjustment_tensors, buildings, ortho, tile_gsd in zip(pred_adjustments,
                                                                  batch.getBatchedBuildings(),
                                                                  batch.getBatchedOrthomosaic(),
                                                                  batch.getBatchedGSD()):
            if adjustment_tensors is None:
                continue
            ortho_gsd = ortho.get_gsd()
            scale_x = tile_gsd[0] / ortho_gsd[0]
            scale_y = tile_gsd[1] / ortho_gsd[1]
            for adjustment_tensor, building in zip(adjustment_tensors, buildings):
                if adjustment_tensor is None:
                    continue
                unadjusted = ortho.get_buildings(ids=[building.getId()], adjusted=False)
                if not unadjusted:
                    continue
                unadjusted_geom = unadjusted[0].getGeometry("pixels")
                centroid = unadjusted_geom.centroid

                # Predicted displacement, converted from tile-pixel space to ortho-pixel space.
                dx = float(adjustment_tensor[2]) * scale_x
                dy = float(adjustment_tensor[3]) * scale_y

                self.predict_step_outputs[building.getId()].append({
                    "class_preds": {label: (1.0 if label == building.getLabel() else 0.0) for label in BDA_DAMAGE_CLASSES},
                    "gsd": tile_gsd,
                    "adjusted": True,
                    "polygon_part": translate(unadjusted_geom, xoff=dx, yoff=dy),
                    "adjustments": [Adjustment(centroid.x, centroid.y, centroid.x + dx, centroid.y + dy)],
                    "label_source": ortho.get_name(),
                })

    def on_predict_epoch_end(self):
        self.predicted_labels = fuse_bda_tiled_inference(self.predict_step_outputs)

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        # Set the model to train mode
        self.model.train()

        init_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INIT] = init_time - start_time

        # Get the class losses from the model...
        batch_size, _, width, height = batch.getBatchedImagery().shape
        # The number of target classes. channel_maps lives in the channel parameters, not
        # model_hyperparameters; the base class already parsed input_dataset_class_2_idx_map into
        # self.dataset_label_map, so getAllLabels() gives that class count.
        num_target_classes = len(self.dataset_label_map.getAllLabels())

        #Before we run the model...
        #We get the the adjustments for that vectorfield based on the ground truth data
        gt_adjustments_scaled = batch.getBatchedAdjustmentsTensor()
        gt_building_weights = get_batched_building_label_weights(batch.getBatchedBuildings(),
                                                                 self._model_hyperparameters["input"]["training_parameters"]["adjustment_class_loss_weights"])

        #We adjust all of the buildings based on the ground turth data
        batched_gt_buildings = self.apply_adjustment_tensors_to_buildings(batch.getBatchedAdjustmentsTensor(), batch.getBatchedBuildings())
        preprocess_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_PREPROCESS] = preprocess_time - init_time

        #Run the model and get the unmasked output for BDA, and adjustments
        model_output = self.model.forward(self.format_bdaadj_model_input(batch))

        # If the model generates a displacement field
        if model_output.contains(DISPLACEMENT_FIELD):
            # Get the displacement field from the model output
            y_hat_adj = model_output[DISPLACEMENT_FIELD]

            #Generate adjustment masks for all of the buildings in the shape of the output vector field
            adjustment_building_masks = self.get_tensor_building_masks(batch.getBatchedBuildings(), width, height, y_hat_adj.shape[2], y_hat_adj.shape[3])

            forward_time = time.time()
            self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_FORWARD] = forward_time - preprocess_time

            #Get the relevant adjustments for the buildings based on the predicted and ground truth vector fields
            pred_adjustments = self.get_building_adjustment(y_hat_adj, batch.getBatchedBuildings(), adjustment_building_masks)

            compute_adjustments_time = time.time()
            self._step_metadata.scalars["Timing/Training Step Timings"]["compute_adjustments_time"] = compute_adjustments_time - forward_time
        else:
            compute_adjustments_time = time.time()
            pred_adjustments = torch.zeros((batch_size, 2, width, height)).to(batch.get_device())

        #Apply the relevant adjustments to the buiildings and select which building polygons to use for BDA training based on if we are teacher forcing
        batched_pred_buildings = self.apply_adjustment_tensors_to_buildings(pred_adjustments, batch.getBatchedBuildings())
        buildings_to_use = batched_gt_buildings if self._model_hyperparameters["input"]["training_parameters"]["adj_teacher_force"] else batched_pred_buildings

        apply_adjustment = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"]["apply_adjustment"] = apply_adjustment - compute_adjustments_time

        # If the model output contains BDA predictions...
        if model_output.contains(Y_HAT_SEGMENTATION_UNMASKED):
            # Get the values from the model output...
            y_hat_segmentation_unmasked = model_output[Y_HAT_SEGMENTATION_UNMASKED]

            # Generate label and query tensors based on the adjusted locations
            y_bda, y_bda_query = self.get_query_label_tensor(buildings_to_use, y_hat_segmentation_unmasked.shape[2], y_hat_segmentation_unmasked.shape[3])
            label_query_generation_time = time.time()
            self._step_metadata.scalars["Timing/Training Step Timings"]["label_query_generation_time"] = label_query_generation_time - apply_adjustment

            # Mask the output of the bda model based on the query
            y_hat_bda_masked = self.model.mask(y_hat_segmentation_unmasked, y_bda_query)
            mask_application_time = time.time()
            self._step_metadata.scalars["Timing/Training Step Timings"]["mask_application_time"] = mask_application_time - label_query_generation_time

            # Compute the criterion loss
            criterion_loss = self.criterion(y_hat_segmentation_unmasked, y_bda).mean()
        else:
            mask_application_time = time.time()
            criterion_loss = torch.zeros((batch_size,)).mean().to(batch.get_device())
            y_hat_bda_masked = torch.zeros((batch_size, num_target_classes, width, height)).to(batch.get_device())
            y_hat_segmentation_unmasked = torch.zeros((batch_size, num_target_classes, width, height)).to(batch.get_device())

        batched_pred_ious = compute_batched_building_iou(batched_pred_buildings, batched_gt_buildings)
        batched_unadjusted_ious = compute_batched_building_iou(batch.getBatchedBuildings(), batched_gt_buildings)

        # Compute the losses for both the entire vector field and for the individual adjustments
        pred_adjustments_without_none = [adj_batch for adj_batch in pred_adjustments if adj_batch is not None]
        gt_adjustments_without_none = [adj_batch for adj_batch in gt_adjustments_scaled if adj_batch is not None]
        adjustment_weights_without_none = [adj_batch for adj_batch in gt_building_weights if adj_batch is not None]
        adjustment_ious = [torch.tensor(iou_batch) for iou_batch in batched_pred_ious if iou_batch is not None]
        noadjustment_ious = [torch.tensor(iou_batch) for iou_batch in batched_unadjusted_ious if iou_batch is not None]

        has_samples = False
        if len(pred_adjustments_without_none) > 0 and len(pred_adjustments_without_none) == len(gt_adjustments_without_none):
            magnitude_normalization = torch.tensor([1, 1, width, height, 1, 1, 1]).to(batch.get_device())
            adjustment_weight_concatenated = torch.cat(adjustment_weights_without_none).to(batch.get_device())
            pred_adjustment_concatenated = torch.cat(pred_adjustments_without_none).to(batch.get_device()).to(torch.float32) / magnitude_normalization
            gt_adjustment_concatenated = torch.cat(gt_adjustments_without_none).to(batch.get_device()) / magnitude_normalization
            adjustment_ious_concatenated = torch.cat(adjustment_ious)
            noadjustment_ious_concatenated = torch.cat(noadjustment_ious)
            adjustment_iou_gains = adjustment_ious_concatenated - noadjustment_ious_concatenated

            #Add epsilon=1e8 to the denominator so we don't div by zero and get NaN
            adjustment_iou_relative_gains = (adjustment_ious_concatenated-noadjustment_ious_concatenated)/(1.0 - noadjustment_ious_concatenated + 1e-8)

            pred_adjustment_magnitudes = (pred_adjustment_concatenated[:, 2]**2.0 + pred_adjustment_concatenated[:, 3]**2.0)**0.5
            gt_adjustment_magnitudes = (gt_adjustment_concatenated[:, 2]**2.0 + gt_adjustment_concatenated[:, 3]**2.0)**0.5

            adj_indiv_error_x = F.l1_loss(pred_adjustment_concatenated[:, 2], gt_adjustment_concatenated[:, 2], reduction="none")
            adj_indiv_mse_x = F.mse_loss(pred_adjustment_concatenated[:, 2], gt_adjustment_concatenated[:, 2], reduction="none")
            adj_indiv_error_y = F.l1_loss(pred_adjustment_concatenated[:, 3], gt_adjustment_concatenated[:, 3], reduction="none")
            adj_indiv_mse_y = F.mse_loss(pred_adjustment_concatenated[:, 3], gt_adjustment_concatenated[:, 3], reduction="none")
            adj_indiv_magnitude_error = F.l1_loss(pred_adjustment_magnitudes, gt_adjustment_magnitudes, reduction="none")

            adjustment_cos_similarities = F.cosine_embedding_loss(pred_adjustment_concatenated[:, 2:4],
                                                                  gt_adjustment_concatenated[:, 2:4],
                                                                  target=torch.ones((gt_adjustment_concatenated.shape[0])).to(batch.get_device()),
                                                                  reduction="none")

            has_samples = True
            bda_scaled_adj_indiv_mse_x = (adjustment_weight_concatenated * adj_indiv_mse_x).mean()
            bda_scaled_adj_indiv_mse_y = (adjustment_weight_concatenated * adj_indiv_mse_y).mean()

        else:
            bda_scaled_adj_indiv_mse_x = 0.0
            bda_scaled_adj_indiv_mse_y = 0.0
            adjustment_iou_relative_gains = torch.tensor(0.0)
            adj_indiv_error_x = torch.tensor(0.0)
            adj_indiv_error_y = torch.tensor(0.0)
            adj_indiv_magnitude_error = torch.tensor(0.0)
            adjustment_cos_similarities = torch.tensor(0.0)
            adjustment_iou_gains = torch.tensor(0.0)

        # Compute the L1 Loss
        l1_reg_loss = self.get_l1_loss()

        # Compute the total loss for the model
        loss = bda_scaled_adj_indiv_mse_x + bda_scaled_adj_indiv_mse_y + l1_reg_loss + criterion_loss

        loss_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOSS] = loss_time - mask_application_time

        # Update the dictionaries that contain our tracking data...
        self.log_timing_pixel_polygon_stats(batch)

        if has_samples:
             # If we need to log out images...
            if self.should_log_images() and not self.images_have_been_logged():
                color_data = batch.getBatchedImagery()[:, :3, :, :].permute((0,2,3,1))
                polygon_inspection_payload = (color_data, batch.getBatchedBuildings(), batched_pred_buildings, batched_gt_buildings)
                self._add_batched_images_to_step_metadata(color_data,
                                                          "Visuals | 1) Image",
                                                          inspect_image,
                                                          image_limit=2)
                self._add_batched_images_to_step_metadata(polygon_inspection_payload,
                                                          "Visuals | 2) Polygons (Red - Input, Blue - Pred, Green - GT)",
                                                          lambda x: inspect_polygons_on_image(*x),
                                                          image_limit=2)
                if model_output.contains(Y_HAT_SEGMENTATION_UNMASKED):
                    self._add_batched_images_to_step_metadata(torch.argmax(y_hat_segmentation_unmasked, 1),
                                                              "Visuals | 3) Unmasked Preds",
                                                              lambda x: inspect_labels(x, self.idx2color_map),
                                                              image_limit=2)
                    self._add_batched_images_to_step_metadata(torch.argmax(y_hat_bda_masked, 1),
                                                              "Visuals | 4) Masked Preds",
                                                              lambda x: inspect_labels(x, self.idx2color_map),
                                                              image_limit=2)
                self.mark_images_logged()

            buildings_in_batch = sum(len(b) for b in batch.getBatchedBuildings())

            self._step_metadata.scalar["Statistics/Samples in Batch"] += len(batch)
            self._step_metadata.quantiles["Statistics/True Adjustment Magnitudes"].extend(gt_adjustment_magnitudes.tolist())
            self._step_metadata.quantiles["Statistics/Predicted Adjustment Magnitudes"].extend(pred_adjustment_magnitudes.tolist())
            self._step_metadata.quantiles["Statistics/Sample GSDs"].extend(batch.getBatchedGSD())
            self._step_metadata.quantiles["Statistics/Count of Buildings"].extend([len(b) for b in batch.getBatchedBuildings()])
            self._step_metadata.quantiles["Model Perf/Adjustment Pixel Errors (Relative x)"].extend((adj_indiv_error_x).tolist())
            self._step_metadata.quantiles["Model Perf/Adjustment Pixel Errors (Relative y)"].extend((adj_indiv_error_y).tolist())
            self._step_metadata.quantiles["Model Perf/Adjustment Pixel Errors (Relative Magnitude)"].extend((adj_indiv_magnitude_error).tolist())
            self._step_metadata.quantiles["Model Perf/Adjustment Cos Similiarity"].extend(adjustment_cos_similarities.tolist())
            self._step_metadata.quantiles["Model Perf/Adjusted Building IoUs"].extend(adjustment_ious_concatenated.tolist())
            self._step_metadata.quantiles["Model Perf/Delta Building IoU"].extend(adjustment_iou_gains.tolist())
            self._step_metadata.quantiles["Model Perf/Relative Delta Building IoU"].extend(adjustment_iou_relative_gains.tolist())

            for i in range(0, 256, 8):
                self._step_metadata.scalars["Model Perf/Adj Error (x) Threshold"]["Px Threshold = " + str(i)] += torch.sum((adj_indiv_error_x * width) < i)
                self._step_metadata.scalars["Model Perf/Adj Error (y) Threshold"]["Px Threshold = " + str(i)] += torch.sum((adj_indiv_error_y * height) < i)
            self._step_metadata.normalizations["Model Perf/Adjustment Correctness by Pixel Threshold (y)"] += buildings_in_batch
            self._step_metadata.normalizations["Model Perf/Adjustment Correctness by Pixel Threshold (x)"] += buildings_in_batch

            self._step_metadata.scalar["Loss/BDA Scaled MSE Loss x"] += bda_scaled_adj_indiv_mse_x
            self._step_metadata.scalar["Loss/BDA Scaled MSE Loss y"] += bda_scaled_adj_indiv_mse_y
            self._step_metadata.scalar["Loss/BDA Criterion Loss"] += criterion_loss
            self._step_metadata.normalizations["Loss/BDA Scaled MSE Loss x"] += buildings_in_batch
            self._step_metadata.normalizations["Loss/BDA Scaled MSE Loss y"] += buildings_in_batch
            self._step_metadata.normalizations["Loss/BDA Criterion Loss"] += len(batch)

        self._step_metadata.scalar["Loss/Final Loss"] += loss
        self._step_metadata.scalar["Loss/L1 Regularization Loss"] += l1_reg_loss
        self._step_metadata.normalizations["Loss/Final Loss"] += len(batch)
        self._step_metadata.normalizations["Loss/L1 Regularization Loss"] += len(batch)

        self._cur_iter += 1

        log_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOG] = log_time - loss_time
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INTER_STEP] = log_time - start_time
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INTRA_STEP] = start_time - self._prev_start_time
        self._step_metadata.normalizations["Timing/Training Step Timings"] += len(batch)

        self._prev_start_time = start_time

        return loss

    def validation_step(self, batch, batch_idx):
         # Get the class losses from the model...
        # Get the shape of the data we will be working with
        batch_size, _, width, height = batch.getBatchedImagery().shape
        # The number of target classes. channel_maps lives in the channel parameters, not
        # model_hyperparameters; the base class already parsed input_dataset_class_2_idx_map into
        # self.dataset_label_map, so getAllLabels() gives that class count.
        num_target_classes = len(self.dataset_label_map.getAllLabels())

        #We adjust all of the buildings based on the ground turth data
        batched_gt_buildings = self.apply_adjustment_tensors_to_buildings(batch.getBatchedAdjustmentsTensor(), batch.getBatchedBuildings())

        #Run the model and get the unmasked output for BDA, and adjustments
        model_output = self.model.forward(self.format_bdaadj_model_input(batch))

        # If the model generates a displacement field
        if model_output.contains(DISPLACEMENT_FIELD):
            # Get the displacement field from the model output
            y_hat_adj = model_output[DISPLACEMENT_FIELD]

            #Generate adjustment masks for all of the buildings in the shape of the output vector field
            adjustment_building_masks = self.get_tensor_building_masks(batch.getBatchedBuildings(), width, height, y_hat_adj.shape[2], y_hat_adj.shape[3])

            #Get the relevant adjustments for the buildings based on the predicted and ground truth vector fields
            pred_adjustments = self.get_building_adjustment(y_hat_adj, batch.getBatchedBuildings(), adjustment_building_masks)
        else:
            pred_adjustments = torch.zeros((batch_size, 2, width, height)).to(batch.get_device())

        batched_pred_buildings = self.apply_adjustment_tensors_to_buildings(pred_adjustments, batch.getBatchedBuildings())

        # If the model output contains BDA predictions...
        if model_output.contains(Y_HAT_SEGMENTATION_UNMASKED):
            # Get the values from the model output...
            y_hat_segmentation_unmasked = model_output[Y_HAT_SEGMENTATION_UNMASKED]

            # Generate label and query tensors based on the adjusted locations
            y_bda, _ = self.get_query_label_tensor(batched_pred_buildings, y_hat_segmentation_unmasked.shape[2], y_hat_segmentation_unmasked.shape[3])

            # Compute the criterion loss
            # TODO: Update this logic once we start doing joint training...
            # TODO: compute class preds for the buildings we are inspecting
            criterion_loss = self.criterion(y_hat_segmentation_unmasked, y_bda)
        else:
            criterion_loss = torch.zeros((batch_size,)).to(batch.get_device())
            y_hat_segmentation_unmasked = torch.zeros((batch_size, num_target_classes, width, height)).to(batch.get_device())

        #Offset the buildings to pixel global pixel coordinates
        batched_frame_geoms = batch.getBatchedFrameGeometry()
        batched_offset_gt_buildings = add_offset_to_batched_buildings(batched_gt_buildings, batched_frame_geoms, width/2, height/2)
        batched_offset_pred_buildings = add_offset_to_batched_buildings(batched_pred_buildings, batched_frame_geoms, width/2, height/2)

        for gt_buildings, pred_buildings, frame_geom in zip(batched_offset_gt_buildings, batched_offset_pred_buildings, batched_frame_geoms):
            for gt_building, pred_building in zip(gt_buildings, pred_buildings):
                gt_intersected_geom = frame_geom.intersection(gt_building.getGeometry("pixels"))
                pred_intersected_geom = frame_geom.intersection(pred_building.getGeometry("pixels"))

                self.validation_step_labels[gt_building.getId()].append(gt_intersected_geom)
                self.validation_step_outputs[gt_building.getId()].append(pred_intersected_geom)

        self.validation_loss.extend(criterion_loss.tolist())


    def on_validation_epoch_end(self):
        ious = []
        for building_id, gt_building_polygon_parts in self.validation_step_labels.items():

            pred_building_polygon_parts = self.validation_step_outputs[building_id]

            gt_multi_polygon = unary_union(gt_building_polygon_parts)
            pred_multi_polygon = unary_union(pred_building_polygon_parts)

            try:
                iou_val = gt_multi_polygon.intersection(pred_multi_polygon).area/gt_multi_polygon.union(pred_multi_polygon).area
            except ZeroDivisionError:
                print("Warning, when attempting to log a validation IoU, got divide by zero!")
                iou_val = 0
            ious.append(iou_val)

        self.log("val_macro_f1", np.mean(self.validation_loss))
        self.log("val_ortho_inference_avg_IoU", np.mean(ious))
