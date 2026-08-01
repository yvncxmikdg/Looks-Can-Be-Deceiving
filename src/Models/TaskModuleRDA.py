import numpy as np

import torch.nn.functional as F

from modeling.utils.decoder_utils import road_lines_to_labeled_road_line_segments
from modeling.formatters.plot_metrics_RDA import generate_confusion_matrix_plot
from modeling.Models.TaskModuleBase import TaskModuleBase
from modeling.evaluate_RDA import load_multi_labeled_road_lines_from_preds, get_metrics_bundle, \
                                  get_ground_truth_multilabeled_road_lines, get_road_line_to_gsd_map
from modeling.utils.sample_generator_utils import translate_road_line

class TaskModuleRDA(TaskModuleBase):
    def __init__(self, channel_parameters=None, model_hyperparameters=None, val_orthomosaics=None, device="cuda", uq_hyperparameters=None, alerter=None):
        super().__init__(channel_parameters, model_hyperparameters, val_orthomosaics, device, uq_hyperparameters=uq_hyperparameters, alerter=alerter)
        self._road_line_buffer_width_pixels = model_hyperparameters["input"]["road_line_buffer_width_pixels"]
        self._road_line_segment_length_pixels = model_hyperparameters["input"]["road_line_segment_length_pixels"]

    # pylint: disable-next=arguments-differ, unused-argument
    def predict_step(self, batch, batch_idx):
        y_hat_logits = self._compute_y_hat_inference(batch)
        y_hat_preds = F.softmax(y_hat_logits, dim=1) # Obtain preds via softmax

        for road_lines, x_offset, y_offset, y_hat_i in zip(batch.getBatchedRoadLines(), batch.getBatchedX(), batch.getBatchedY(), y_hat_preds):
            if len(road_lines) > 0:
                labeled_road_line_segments = road_lines_to_labeled_road_line_segments(
                    y_hat_i,
                    road_lines,
                    0,
                    0,
                    label_to_idx_map=self.output_label_map,
                    segment_length_pixels=self._road_line_segment_length_pixels,
                    segment_buffer_width_pixels=self._road_line_buffer_width_pixels,
                )
                for parent_road_line, segment_payload in labeled_road_line_segments.items():
                    translated_parent = translate_road_line(parent_road_line, x_offset, y_offset)
                    for labeled_segment in segment_payload["segments"]:
                        translated_labeled_segment = translate_road_line(labeled_segment, -1*x_offset, -1*y_offset)
                        labeled_segment_json = labeled_segment.jsonify(parent_road_line=translated_parent)
                        self.predict_step_outputs[translated_labeled_segment.getParentRoadLineId()].append(labeled_segment_json)

    def on_predict_epoch_end(self):
        self.predicted_labels = self.predict_step_outputs

    # pylint: disable-next=arguments-differ, unused-argument
    def validation_step(self, batch, batch_idx):
        y_hat_logits = self._compute_y_hat_inference(batch)
        y_hat_preds = F.softmax(y_hat_logits, dim=1) # Obtain preds via softmax

        for road_lines, x_offset, y_offset, y_hat_i in zip(batch.getBatchedRoadLines(), batch.getBatchedX(), batch.getBatchedY(), y_hat_preds):
            if len(road_lines) > 0:
                labeled_road_line_segments = road_lines_to_labeled_road_line_segments(
                    y_hat_i,
                    road_lines,
                    0,
                    0,
                    label_to_idx_map=self.output_label_map,
                    segment_length_pixels=self._road_line_segment_length_pixels,
                    segment_buffer_width_pixels=self._road_line_buffer_width_pixels,
                )

                for parent_road_line, segment_payload in labeled_road_line_segments.items():
                    translated_parent = translate_road_line(parent_road_line, x_offset, y_offset)
                    for labeled_segment in segment_payload["segments"]:
                        translated_labeled_segment = translate_road_line(labeled_segment, -1*x_offset, -1*y_offset)
                        labeled_segment_json = labeled_segment.jsonify(parent_road_line=translated_parent)
                        labeled_segment_px_len = float(labeled_segment.getGeometry("pixels").length)
                        self.validation_step_outputs[translated_labeled_segment.getParentRoadLineId()].append(labeled_segment_json)
                        self._step_metadata.scalars["val/Predicted_Pixel_Counts"][labeled_segment.getLabel()] += labeled_segment_px_len
                self._step_metadata.normalizations["val/Predicted_Pixel_Counts"] += len(road_lines)

        # Compute the criterion loss
        criterion_loss = self.criterion(y_hat_logits, batch.getBatchedLabels()) * self.criterion_scale_factor
        self.validation_loss.append(criterion_loss.mean().cpu().tolist())

    def on_validation_epoch_end(self):
        val_loss = np.mean(self.validation_loss)
        self.validation_step_labels = self.validation_step_outputs
        self.log("val_criterion_loss", val_loss)
        self.get_tb_logger().add_scalar("val/criterion_loss", val_loss, self.current_epoch)

        # Load all of the ground truth multilabeled roadlines
        gt_multilabeled_road_lines = get_ground_truth_multilabeled_road_lines(self.val_orthomosaics)
        road_lines_to_gsd = get_road_line_to_gsd_map(self.val_orthomosaics)
        pred_multilabeled_road_lines = load_multi_labeled_road_lines_from_preds(self.validation_step_labels, gt_multilabeled_road_lines)
        metrics = get_metrics_bundle(pred_multilabeled_road_lines,
                                     gt_multilabeled_road_lines,
                                     self.default_label,
                                     self.dataset_label_map,
                                     self.output_label_map,
                                     road_lines_to_gsd,
                                     self.getName(),
                                     False)

        logger = self.get_tb_logger()
        self.log("val_macro_f1", metrics["metrics"]["F1"]["macro"])
        self.log("val_macro_iou", metrics["metrics"]["IoU"]["macro"])
        logger.add_scalar("val/macro_f1", metrics["metrics"]["F1"]["macro"], self.current_epoch)
        logger.add_scalar("val/macro_precision", metrics["metrics"]["Precision"]["macro"], self.current_epoch)
        logger.add_scalar("val/macro_recall", metrics["metrics"]["Recall"]["macro"], self.current_epoch)
        logger.add_scalar("val/macro_iou", metrics["metrics"]["IoU"]["macro"], self.current_epoch)

        np_image_px = generate_confusion_matrix_plot([metrics], [self.getName()], return_np=True, key="Confusion_Matrix_pixels")
        np_image_km = generate_confusion_matrix_plot([metrics], [self.getName()], return_np=True, key="Confusion_Matrix_km")
        logger.add_image("val/ConfusionMatrix_Pixels",np_image_px,self.current_epoch,dataformats="HWC")
        logger.add_image("val/ConfusionMatrix_KM", np_image_km, self.current_epoch, dataformats="HWC")

        # Assemble the alert that will be sent
        alert_messages = [
            f"\nValidation #{self.current_epoch} on step {self.global_step}",
            f"\tval_global/macro_f1: {(metrics['metrics']['F1']['macro']):0.5f}",
            f"\tval_global/macro_iou: {(metrics['metrics']['IoU']['macro']):0.5f}",
        ]
        self.sendAlert("\n".join(alert_messages))
