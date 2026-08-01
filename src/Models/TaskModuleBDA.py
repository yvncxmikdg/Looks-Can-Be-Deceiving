import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from modeling.DataMap import Labels2IdxMap
from modeling.Models.TaskModuleBase import TaskModuleBase
from modeling.Models.OrthoInferenceWrapper import fuse_bda_tiled_inference, joint_file_pred_key
from modeling.utils.multiscale_metrics import evaluate_multiscale_predictions, AUC_multiscale_metrics
from modeling.formatters.plot_metrics_BDA import generate_confusion_matrix_plot
from modeling.utils.decoder_utils import buildings_to_pixel_agg, buildings_to_pixel_counts, \
                                         combine_gathered_labels, combine_gathered_preds_dictionaries, \
                                         combine_gathered_loss
from modeling.Models.ModelDatum import Y_HAT_SEGMENTATION_MASKED, DO_SOFTMAX, SIGMA, UQ_PREDICTION

# Fallback MC-dropout sample count, used only when the uq hyperparameters (uq.yaml) don't set
# `epistemic_samples`. Read per-model through _epistemic_samples() rather than referenced directly.
DEFAULT_EPISTEMIC_SAMPLES = 4

class TaskModuleBDA(TaskModuleBase):

    def _epistemic_samples(self):
        # Number of MC-dropout forward passes for epistemic UQ: 0 when UQ is off, otherwise the
        # uq.yaml `epistemic_samples` (falling back to DEFAULT_EPISTEMIC_SAMPLES). Keeping the count
        # a config value lets a run trade UQ fidelity for speed without a code change.
        if not self.uq_hyperparameters:
            return 0
        return int(self.uq_hyperparameters.get("epistemic_samples", DEFAULT_EPISTEMIC_SAMPLES))

    # pylint: disable-next=arguments-differ, unused-argument
    def predict_step(self, batch, batch_idx, verbose=True):
        num_samples = self._epistemic_samples()

        # Standard deterministic path: mirrors the non-UQ mainline behavior exactly.
        if num_samples <= 1:
            y_hat_logits = self._compute_y_hat_inference(batch)
            y_hat_preds = F.softmax(y_hat_logits, dim=1) # Obtain preds via softmax
            for buildings, ortho, gsd, y_hat_i in zip(
                batch.getBatchedBuildings(),
                batch.getBatchedOrthomosaic(),
                batch.getBatchedGSD(),
                y_hat_preds,
            ):
                if len(buildings) > 0:
                    class_preds = buildings_to_pixel_counts(y_hat_i, buildings, 0, 0, label_to_idx_map=self.output_label_map)
                    for b in buildings:
                        joint_id = joint_file_pred_key(ortho.get_name(), b.getId(), gsd[0], gsd[1])
                        self.predict_step_outputs[joint_id].append({"class_preds": class_preds[b.getId()]})
            return

        # UQ / Bayesian code path: MC-dropout epistemic sampling + aleatoric sigma.
        x = self.format_batched_data_to_model_input(batch)
        x.setField(DO_SOFTMAX, False)
        self.model.eval()
        self.enable_dropout()

        logits_sum = 0
        prob_sum = 0
        sigma_sum = 0
        expected_entropy_sum = 0

        with torch.no_grad():
            for _ in range(num_samples):
                out = self.model(x)
                logits_sum += out[Y_HAT_SEGMENTATION_MASKED]
                sigma_sum += out[SIGMA]

                uq_log_probs = out[UQ_PREDICTION]
                prob = torch.exp(uq_log_probs)
                prob_sum += prob
                expected_entropy_sum += -(prob * uq_log_probs).sum(dim=1)

        mean_logits = logits_sum / num_samples
        mean_sigma = sigma_sum / num_samples
        prob_ave = prob_sum / num_samples
        expected_entropy = expected_entropy_sum / num_samples

        aleatoric_map = mean_sigma.mean(dim=1)
        combined_map = -(prob_ave * torch.log(prob_ave + 1e-8)).sum(dim=1)
        epistemic_map = torch.relu(combined_map - expected_entropy)

        # The sigma head can emit the aleatoric map at the encoder's patch-grid resolution (e.g.
        # ScaleMAE/UperNet: input/patch = 1024/16 = 64) while the softmax-derived maps are at full
        # input resolution. Upsample it to match so the three UQ maps stack, and so the stacked
        # tensor later concatenates with the full-resolution segmentation logits. No-op for models
        # whose sigma is already full resolution (e.g. the UNet).
        if aleatoric_map.shape[-2:] != combined_map.shape[-2:]:
            aleatoric_map = F.interpolate(
                aleatoric_map.unsqueeze(1), size=combined_map.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        # Softmax for final building prediction extraction
        y_hat_probs = torch.softmax(mean_logits, dim=1)

        for batch_i, (buildings, ortho, gsd, y_hat_i) in enumerate(zip(
            batch.getBatchedBuildings(),
            batch.getBatchedOrthomosaic(),
            batch.getBatchedGSD(),
            y_hat_probs,
        )):
            if len(buildings) > 0:
                # Combine probabilities and UQ maps into a single tensor
                uq_tensor = torch.stack([
                    aleatoric_map[batch_i],
                    combined_map[batch_i],
                    epistemic_map[batch_i]
                ], dim=0)

                omni_tensor = torch.cat([y_hat_i, uq_tensor], dim=0)

                c_offset = len(self.output_label_map)
                omni_dict = {label: self.output_label_map.getIndex(label) for label in self.output_label_map.getAllLabels()}
                omni_dict.update({"aleatoric": c_offset + 0, "combined": c_offset + 1, "epistemic": c_offset + 2})
                omni_map = Labels2IdxMap(omni_dict)

                strategies = ["sum", "mean", "max", "high_entropy_count"]

                # Run a single aggregation pass to avoid redundant CPU rasterization
                all_metrics = buildings_to_pixel_agg(omni_tensor, buildings, 0, 0, keys_map=omni_map, aggregation_strategies=strategies)

                # Assemble the results into the final payload format
                for b in buildings:
                    building_id = b.getId()
                    joint_id = joint_file_pred_key(ortho.get_name(), building_id, gsd[0], gsd[1])

                    b_metrics = all_metrics[building_id]

                    class_preds = {
                        label: b_metrics[f"sum_{label}"]
                        for label in self.output_label_map.getAllLabels()
                    }

                    payload = {
                        "class_preds": class_preds,
                        "uq_metrics": {
                            "mean_aleatoric": b_metrics["mean_aleatoric"],
                            "max_aleatoric": b_metrics["max_aleatoric"],
                            "mean_combined": b_metrics["mean_combined"],
                            "max_combined": b_metrics["max_combined"],
                            "mean_epistemic": b_metrics["mean_epistemic"],
                            "max_epistemic": b_metrics["max_epistemic"],
                            "high_entropy_count": b_metrics["high_entropy_count_combined"]
                        }
                    }

                    self.predict_step_outputs[joint_id].append(payload)

    def _fusion_class_labels(self):
        # None -> fuse_bda_tiled_inference falls back to the BDA damage classes. Subclasses whose
        # buildings carry a different label set (TaskModuleCHANGE) override this.
        return None

    def on_predict_epoch_end(self):
        local_dict = {"preds": self.predict_step_outputs}
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gathered, local_dict)
            global_dict = {"preds": combine_gathered_preds_dictionaries(gathered, lambda x:x["preds"])}
        else:
            global_dict = local_dict

        self.predicted_labels = fuse_bda_tiled_inference(global_dict["preds"], run_uq=self._epistemic_samples() > 1,
                                                         class_labels=self._fusion_class_labels())

    # pylint: disable-next=arguments-differ, unused-argument
    def validation_step(self, batch, batch_idx):
        num_samples = self._epistemic_samples()

        if num_samples > 1:
            # UQ / Bayesian code path: MC-dropout epistemic sampling + aleatoric sigma.
            x = self.format_batched_data_to_model_input(batch)
            x.setField(DO_SOFTMAX, False) # MUST be False for criterion loss
            self.model.eval()
            self.enable_dropout() # Turn ONLY Dropout on

            logits_sum = 0
            prob_sum = 0
            sigma_sum = 0
            expected_entropy_sum = 0

            with torch.no_grad():
                for _ in range(num_samples):
                    out = self.model(x)

                    logits_sum += out[Y_HAT_SEGMENTATION_MASKED]
                    sigma_sum += out[SIGMA]

                    # UQ_PREDICTION returns log-probs when DO_SOFTMAX=False, so we reverse it
                    uq_log_probs = out[UQ_PREDICTION]
                    prob = torch.exp(uq_log_probs)
                    prob_sum += prob

                    # Calculate entropy of this specific model state
                    expected_entropy_sum += -(prob * uq_log_probs).sum(dim=1)

            # Average the ensembles
            mean_logits = logits_sum / num_samples
            mean_sigma = sigma_sum / num_samples
            prob_ave = prob_sum / num_samples
            expected_entropy = expected_entropy_sum / num_samples

            del x
            y_hat_logits = mean_logits
        else:
            y_hat_logits = self._compute_y_hat_inference(batch)

        y_hat_preds = F.softmax(y_hat_logits, dim=1) # Obtain preds via softmax
        for buildings, ortho, gsd, y_hat_i in zip( batch.getBatchedBuildings(), batch.getBatchedOrthomosaic(), batch.getBatchedGSD(), y_hat_preds):
            if len(buildings) > 0:
                class_preds = buildings_to_pixel_counts(y_hat_i, buildings, 0, 0, label_to_idx_map=self.output_label_map)
                for b in buildings:
                    joint_id = joint_file_pred_key(ortho.get_name(), b.getId(), gsd[0], gsd[1])
                    self.validation_step_outputs[joint_id].append({"class_preds": class_preds[b.getId()]})
                    self.validation_step_labels[joint_id] = b.getLabel()
                    for c in self.output_label_map.getAllLabels():
                        self._step_metadata.scalars["val/Predicted_Pixel_Counts"][c] += float(class_preds[b.getId()][c])
                self._step_metadata.normalizations["val/Predicted_Pixel_Counts"] += len(buildings)

        # Compute the crtierion loss on the (possibly UQ-averaged) logits
        criterion_loss = self.criterion(y_hat_logits, batch.getBatchedLabels()) * self.criterion_scale_factor
        self.validation_loss.append(criterion_loss.detach().cpu().tolist())

        if num_samples > 1:
            aleatoric_uq = mean_sigma.mean(dim=1)

            # Total Predictive Uncertainty (Entropy of average probabilities)
            combined_uq = -(prob_ave * torch.log(prob_ave + 1e-8)).sum(dim=1)

            # Epistemic Uncertainty (Mutual Information)
            # Epistemic = Total Predictive Uncertainty - Expected Aleatoric Entropy
            # Use ReLU because floating point math can occasionally make this infinitesimally negative
            epistemic_uq = torch.relu(combined_uq - expected_entropy)

            # Log the metrics
            self._step_metadata.scalars["val/UQ"]["Aleatoric_Uncertainty"] += aleatoric_uq.mean().item()
            self._step_metadata.scalars["val/UQ"]["Combined_Uncertainty"] += combined_uq.mean().item()
            self._step_metadata.scalars["val/UQ"]["Epistemic_Uncertainty"] += epistemic_uq.mean().item()
            self._step_metadata.normalizations["val/UQ"] += 1

    def on_validation_epoch_end(self):
        # The pre-training sanity pass runs the FULL validation set (num_sanity_val_steps=-1 unless
        # --skip_sanity), and its metrics are reported like any other epoch's on purpose: they are the
        # untrained-model reference point the training curves are read against.
        local_dict = {
            "labels": self.validation_step_labels,
            "preds": self.validation_step_outputs,
            "loss": self.validation_loss,
        }

        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gathered, local_dict)
            global_dict = {
                "labels": combine_gathered_labels(gathered, lambda x:x["labels"]),
                "preds": combine_gathered_preds_dictionaries(gathered, lambda x:x["preds"]),
                "loss": combine_gathered_loss(gathered, lambda x:x["loss"])
            }
        else:
            global_dict = local_dict

        # Validation fusion only needs the per-building label for the reported metrics; the
        # validation_step stores class_preds without per-building uq_metrics (UQ is logged as
        # aggregate scalars during the step), so fuse without the UQ path to avoid a KeyError.
        fused_preds = fuse_bda_tiled_inference(global_dict["preds"], class_labels=self._fusion_class_labels())
        for c in self.output_label_map.getAllLabels():
            self._step_metadata.scalars["val/Predicted_Class_Counts"][c] = 0
        for pred in fused_preds.values():
            self._step_metadata.scalars["val/Predicted_Class_Counts"][pred["label"]] += 1

        actual_labels = []
        preds_labels = []
        for building_id, actual_label in global_dict["labels"].items():
            actual_labels.append(actual_label)
            preds_labels.append(fused_preds[building_id]["label"])

        macro_f1 = f1_score(actual_labels, preds_labels, average="macro")
        micro_f1 = f1_score(actual_labels, preds_labels, average="micro")
        macro_precision = precision_score(actual_labels, preds_labels, average="macro")
        micro_precision = precision_score(actual_labels, preds_labels, average="micro")
        macro_recall = recall_score(actual_labels, preds_labels, average="macro")
        micro_recall = recall_score(actual_labels, preds_labels, average="micro")

        bda_confusion_matrix = confusion_matrix(y_true=actual_labels, y_pred=preds_labels, labels=list(self.output_label_map.getAllLabels()))
        matrix_data = {"Confusion_Matrix": {"matrix": bda_confusion_matrix.tolist(), "class_labels": self.output_label_map.getAllLabels()}}

        np_image = generate_confusion_matrix_plot(
            [{"metrics": matrix_data, "samples": {"total": len(global_dict["labels"].keys())}, "step": self.global_step}],
            [self.getName()],
            return_np=True,
        )

        # evaluate_multiscale_predictions calls bucket_func(bucket_preds, bucket_actuals) - preds
        # first (see test_bucket_func_receives_preds_then_actuals) - so each lambda takes
        # (preds, actuals) and forwards them to sklearn as (y_true=actuals, y_pred=preds).
        gsd_bucketed_macro_f1 = evaluate_multiscale_predictions(fused_preds,
                                                                global_dict["labels"],
                                                                lambda preds, actuals: f1_score(actuals, preds, average="macro"))
        gsd_bucketed_micro_f1 = evaluate_multiscale_predictions(fused_preds,
                                                                global_dict["labels"],
                                                                lambda preds, actuals: f1_score(actuals, preds, average="micro"))
        gsd_bucketed_macro_precision = evaluate_multiscale_predictions(fused_preds,
                                                                global_dict["labels"],
                                                                lambda preds, actuals: precision_score(actuals, preds, average="macro"))
        gsd_bucketed_micro_precision = evaluate_multiscale_predictions(fused_preds,
                                                                global_dict["labels"],
                                                                lambda preds, actuals: precision_score(actuals, preds, average="micro"))
        gsd_bucketed_macro_recall = evaluate_multiscale_predictions(fused_preds,
                                                                global_dict["labels"],
                                                                lambda preds, actuals: recall_score(actuals, preds, average="macro"))
        gsd_bucketed_micro_recall = evaluate_multiscale_predictions(fused_preds,
                                                                global_dict["labels"],
                                                                lambda preds, actuals: recall_score(actuals, preds, average="micro"))

        auc_macro_f1 = AUC_multiscale_metrics(gsd_bucketed_macro_f1, log_space_area=True, normalize=True)
        auc_micro_f1 = AUC_multiscale_metrics(gsd_bucketed_micro_f1, log_space_area=True, normalize=True)
        auc_macro_precision = AUC_multiscale_metrics(gsd_bucketed_macro_precision, log_space_area=True, normalize=True)
        auc_micro_precision = AUC_multiscale_metrics(gsd_bucketed_micro_precision, log_space_area=True, normalize=True)
        auc_macro_recall = AUC_multiscale_metrics(gsd_bucketed_macro_recall, log_space_area=True, normalize=True)
        auc_micro_recall = AUC_multiscale_metrics(gsd_bucketed_micro_recall, log_space_area=True, normalize=True)

        self.log("val_macro_f1", macro_f1)
        self.log("auc_macro_f1", auc_macro_f1)
        # Same name RDA logs; monitorable by the plateau scheduler (validation_scheduler_monitor)
        # unlike the raw val_global/criterion_loss TB scalar below, which bypasses Lightning.
        self.log("val_criterion_loss", np.mean(global_dict["loss"]))

        if self.trainer.is_global_zero:
            logger = self.get_tb_logger()
            logger.add_image("val_global/ConfusionMatrix", np_image, self.global_step, dataformats="HWC")
            logger.add_scalar("val_global/criterion_loss", np.mean(global_dict["loss"]), self.global_step)
            logger.add_scalar("val_global/macro_f1", macro_f1, self.global_step)
            logger.add_scalar("val_global/micro_f1", micro_f1, self.global_step)
            logger.add_scalar("val_global/macro_precision", macro_precision, self.global_step)
            logger.add_scalar("val_global/micro_precision", micro_precision, self.global_step)
            logger.add_scalar("val_global/macro_recall", macro_recall, self.global_step)
            logger.add_scalar("val_global/micro_recall", micro_recall, self.global_step)
            logger.add_scalar("val_global/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

            logger.add_scalar("val_multiscale/auc_macro_f1", auc_macro_f1, self.global_step)
            logger.add_scalar("val_multiscale/auc_micro_f1", auc_micro_f1, self.global_step)
            logger.add_scalar("val_multiscale/auc_macro_precision", auc_macro_precision, self.global_step)
            logger.add_scalar("val_multiscale/auc_micro_precision", auc_micro_precision, self.global_step)
            logger.add_scalar("val_multiscale/auc_macro_recall", auc_macro_recall, self.global_step)
            logger.add_scalar("val_multiscale/auc_micro_recall", auc_micro_recall, self.global_step)

            logger.add_scalars("val_multiscale/macro_f1", {str(k):v for k,v in gsd_bucketed_macro_f1.items()}, self.global_step)
            logger.add_scalars("val_multiscale/micro_f1", {str(k):v for k,v in gsd_bucketed_micro_f1.items()}, self.global_step)
            logger.add_scalars("val_multiscale/macro_precision", {str(k):v for k,v in gsd_bucketed_macro_precision.items()}, self.global_step)
            logger.add_scalars("val_multiscale/micro_precision", {str(k):v for k,v in gsd_bucketed_micro_precision.items()}, self.global_step)
            logger.add_scalars("val_multiscale/macro_recall", {str(k):v for k,v in gsd_bucketed_macro_recall.items()}, self.global_step)
            logger.add_scalars("val_multiscale/micro_recall", {str(k):v for k,v in gsd_bucketed_micro_recall.items()}, self.global_step)

            #Assemble the alert that will be sent
            alert_messages = [
                f"\nValidation #{self.current_epoch} on step {self.global_step}",
                f"\tval_global/macro_f1: {macro_f1:0.5f}",
                f"\tval_multiscale/auc_macro_f1: {auc_macro_f1:0.5f}",
                "\tval_multiscale/macro_f1:"
            ]
            for k, v in gsd_bucketed_macro_f1.items():
                alert_messages.append(f"\t\tmacro_f1: {v:0.5f} | gsd: {k:0.5f}")

            self.sendAlert("\n".join(alert_messages))
