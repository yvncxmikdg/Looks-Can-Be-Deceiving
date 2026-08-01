import time
from collections import defaultdict
from torch import optim

import pytorch_lightning as L
import torch.nn.functional as F
import torch
import torch.distributed as dist
import numpy as np

from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_optimizer import AdaFactor

from modeling.utils.data_augmentations import (
    get_tensor_transform,
    get_normalize_transform,
)
from modeling.utils.system_telemetry import get_host_memory_stats, get_gpu_memory_stats
from modeling.DataMap import Labels2IdxMap, ColorMap, Channel2IdxMap
from modeling.utils.inspection_utils import inspect_image, inspect_labels, inspect_grad_flow
from modeling.utils.loss_utils import WeightedLoss, get_ipw_weights_from_class_counts, get_log_class_balanced_weights_from_class_counts
from modeling.ModelStepMetadata import ModelStepMetadata
from modeling.Models.ModelDatum import UQ_PREDICTION, ModelInput, Y_HAT_SEGMENTATION_MASKED, CHANNEL_INPUT, MASK, GSD, DO_SOFTMAX, TIMESTAMP
from modeling.constants import POLYGON_COUNT_PREFIX, PIXEL_COUNT_PREFIX, SAMPLE_GENERATION_TIMING_PREFIX, WORKER_MEMORY_PREFIX, \
                               SAMPLE_METADATA_ATTEMPTS, SAMPLE_METADATA_EXCEPTIONS, TRAINING_STEP_METADATA_TIME_INIT, \
                               TRAINING_STEP_METADATA_TIME_PREPROCESS, TRAINING_STEP_METADATA_TIME_FORWARD, TRAINING_STEP_METADATA_TIME_LOSS, \
                               TRAINING_STEP_METADATA_TIME_INTER_STEP, TRAINING_STEP_METADATA_TIME_INTRA_STEP, TRAINING_STEP_METADATA_TIME_LOG

# How often (in optimizer steps) to sample host/GPU memory gauges. Sampling is throttled because
# the dependency-free worker-RSS probe walks /proc, which is cheap but not free on busy nodes.
MEMORY_TELEMETRY_EVERY_N_STEPS = 20


# pylint: disable-next=too-many-public-methods
class TaskModuleBase(L.LightningModule):
    # pylint: disable-next=too-many-branches
    def __init__(self,
                 channel_parameters=None,
                 model_hyperparameters=None,
                 val_orthomosaics=None,
                 device="cuda",
                 quantiles=None,
                 uq_hyperparameters=None,
                 alerter=None):
        super().__init__()

        self._device = device
        self._qs = quantiles if not quantiles is None else [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        self.model_hyperparameters = model_hyperparameters
        # A dict when the run opted into UQ (see model_registry normalization), otherwise None.
        self.uq_hyperparameters = uq_hyperparameters

        self.dataset_label_map = Labels2IdxMap(
            channel_parameters["channel_maps"]["input_dataset_class_2_idx_map"],
            channel_parameters["channel_maps"]["background_class_idx"],
        )
        self.input_channel_map = Channel2IdxMap(channel_parameters["channel_maps"]["input_channels"])
        self.input_background_idx = self.input_channel_map.getIdx("mask")
        self.output_background_idx = channel_parameters["channel_maps"]["background_class_idx"]

        self.output_label_map = Labels2IdxMap(
            channel_parameters["channel_maps"]["output_class_2_idx_map"],
            channel_parameters["channel_maps"]["background_class_idx"],
        )
        self.idx2color_map = ColorMap(
            channel_parameters["channel_maps"]["model_class_2_color_map"],
            channel_parameters["channel_maps"]["output_class_2_idx_map"],
        )

        self.default_label = channel_parameters["channel_maps"]["default_label"]
        self.register_buffer(
            "running_class_counts",
            torch.zeros(len(self.output_label_map.getAllLabels()), dtype=torch.float64)
        )


        try:
            self._uq_loss_weight = self.model_hyperparameters["training"]["training_parameters"]["loss_parameters"]["uq_loss_weight"]
        except KeyError:
            self._uq_loss_weight = 1.0
        try:
            self._normalized_inputs = model_hyperparameters["input"]["normalized_inputs"]
        except KeyError:
            self._normalized_inputs = False
        try:
            self.l1_reg = float(model_hyperparameters["training"]["training_parameters"]["l1_reg"])
        except KeyError:
            self.l1_reg = 0.0
        try:
            self.l2_reg = float(model_hyperparameters["training"]["training_parameters"]["l2_reg"])
        except KeyError:
            self.l2_reg = 0.0
        try:
            self.gamma = float(model_hyperparameters["training"]["training_parameters"]["gamma"])
        except KeyError:
            self.gamma = None
        try:
            self.alpha = float(model_hyperparameters["training"]["training_parameters"]["alpha"])
        except KeyError:
            self.alpha = None
        try:
            self.lr = float(model_hyperparameters["training"]["training_parameters"]["optimizer_parameters"]["learning_rate"])
        except KeyError:
            self.lr = None
        try:
            self._log_images_every_n_steps = model_hyperparameters["training"]["training_parameters"]["log_images_every_n_steps"]
        except KeyError:
            self._log_images_every_n_steps = None
        try:
            self.criterion = WeightedLoss(model_hyperparameters["training"]["training_parameters"]["loss_parameters"],
                                          None,
                                          ignore_index=self.output_background_idx)

        except KeyError:
            self.criterion = None
        try:
            self._include_gsd = model_hyperparameters["model_parameters"]["encoder_parameters"]["backbone"] == "scalemae"
        except KeyError:
            self._include_gsd = False
        try:
            self._include_mask_input = model_hyperparameters["input"]["mask_input"]
        except KeyError:
            self._include_mask_input = True # default to passing in mask as an input to the model
        try:
            self.criterion_scale_factor = model_hyperparameters["training"]["training_parameters"]["loss_parameters"]["scale_factor"]
        except KeyError:
            self.criterion_scale_factor = 1.0

        self._alerter = alerter

        self._cur_iter = 0
        self._cur_step = 0
        self._prev_start_time = time.time()
        self._training_step_end_time = time.time()
        self._on_after_backward_time = time.time()
        self._on_train_batch_end_time = time.time()

        self._logger = None

        self._step_metadata = ModelStepMetadata(self.global_step)
        self._images_logged = False
        self._reset_aggregation_step_metadata()

        self.tensor_transform = get_tensor_transform()
        self.normalize_transform = get_normalize_transform()

        self.val_orthomosaics = val_orthomosaics

        self.validation_step_outputs = defaultdict(list)
        self.validation_step_labels = {}
        self.validation_loss = []
        self.predict_step_outputs = defaultdict(list)
        self.predicted_labels = {}

        self._name = model_hyperparameters["name"]
        self._model_hyperparameters = model_hyperparameters

        # Initialize Model
        self.model = None
        self.model_arg_prep_func = None
        self.class_weights = None
        self.lr_scheduler = None
        self.optimizer = None

    def on_validation_epoch_start(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()
        self.validation_loss.clear()

        for c in self.output_label_map.getAllLabels():
            self._step_metadata.scalars["val/Predicted_Pixel_Counts"][c] = 0

        self.model.eval()

    def get_l2_loss(self):
        # Get the L2 losses from the model
        l2_reg_term = 0
        for param in self.model.parameters():
            l2_reg_term += torch.sum(param**2)
        l2_reg_loss = self.l2_reg * l2_reg_term
        return l2_reg_loss

    @staticmethod
    def _criterion_loss_is_unusable(loss_term):
        # Whether to drop a non-finite loss term. Under DDP this decision MUST be unanimous: the
        # drop changes the autograd graph, and a rank whose graph differs from its peers'
        # desynchronises gradient reduction, which manifests as a silent, indefinite hang rather
        # than an error. So every rank votes and any rank's non-finite term makes all of them drop
        # it. The all_reduce is issued unconditionally (never inside the branch it decides) so it
        # can never itself become the mismatched collective.
        unusable = (~torch.isfinite(loss_term)).any().to(dtype=torch.uint8)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(unusable, op=dist.ReduceOp.MAX)
        return bool(unusable.item())

    def get_l1_loss(self):
        # Get the L1 losses from the model
        l1_reg_term = 0
        for param in self.model.parameters():
            l1_reg_term += torch.sum(torch.abs(param))
        l1_reg_loss = self.l1_reg * l1_reg_term
        return l1_reg_loss

    def log_batch_telemetry(self, batch):
        #Log all the data from fields where will have multiple scalars
        for key in batch.getMetadataKeys():
            # Worker heap stats are GAUGES (a snapshot of the producing worker's allocator), not
            # per-sample quantities: take the latest value rather than summing/normalizing.
            if WORKER_MEMORY_PREFIX in key:
                gauge_values = [v for v in batch.getBatchedMetadataEntry(key) if v is not None]
                if gauge_values:
                    self._step_metadata.scalars["System/Worker Memory (GB)"][key.replace(WORKER_MEMORY_PREFIX, "")] = gauge_values[-1]
                continue
            scalars_collection = None
            if POLYGON_COUNT_PREFIX in key:
                scalars_collection = "Statistics/Polygon Statistics"
            elif PIXEL_COUNT_PREFIX in key:
                scalars_collection = "Statistics/Pixel Statistics"
            elif SAMPLE_GENERATION_TIMING_PREFIX in key:
                scalars_collection = "Timing/Sample Generating Times"
            if scalars_collection:
                self._step_metadata.scalars[scalars_collection][key] += sum(batch.getBatchedMetadataEntry(key))
        self._step_metadata.normalizations["Timing/Sample Generating Times"] += len(batch)
        self._step_metadata.normalizations["Statistics/Polygon Statistics"] += len(batch)
        self._step_metadata.normalizations["Statistics/Pixel Statistics"] += len(batch)

        #Then we can work through the fields with the single values
        self._step_metadata.scalar["Statistics/Sample Generator Attempts"] += sum(batch.getBatchedMetadataEntry(SAMPLE_METADATA_ATTEMPTS))
        self._step_metadata.normalizations["Statistics/Sample Generator Attempts"] += len(batch)

        #Then we log the GSDs of the samples that were passed to the model for training
        for gsd in batch.getBatchedGSD():
            self._step_metadata.scalars["Statistics/GSDs"][str(gsd)] += 1

        #Finally, we can log the exceptions
        for entry in batch.getBatchedMetadataEntry(SAMPLE_METADATA_EXCEPTIONS):
            for exception_name, count in entry.items():
                self._step_metadata.scalars["Statistics/Monitored Exceptions"][exception_name] += count

    def log_loss_telemetry(self, loss_dict, batch):
        for name, value in loss_dict.items():
            self._step_metadata.scalar["Loss/"+name] += value
            self._step_metadata.normalizations["Loss/"+name] += len(batch)
        if getattr(self.trainer, "scaler", None) is not None:
            self._step_metadata.scalar["Loss/GradScaler_Scale"] += self.trainer.scaler.get_scale()
            self._step_metadata.normalizations["Loss/GradScaler_Scale"] += 1

    def sendAlert(self, alert):
        print("====== ALERT ======")
        print(alert)
        if self._alerter:
            self._alerter.sendAlert(alert)
        else:
            print("Warning: Alert fired without a defined alerter")

    def getName(self):
        return self._name

    def initialize_model(self, model):
        self.model = model

    def forward(self, batch):  # pylint: disable=arguments-differ
        return self.model.forward(batch)

    def get_predicted_labels(self):
        return self.predicted_labels

    def on_predict_epoch_start(self):
        self.predict_step_outputs.clear()
        self.predicted_labels.clear()
        self.model.eval()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.moveTo(device, move_raw_imagery=False)
        return batch

    def format_batched_data_to_model_input(self, batch):
        batched_model_input = ModelInput()
        rgb = batch.getBatchedImagery()
        mask = batch.getBatchedQueries().unsqueeze(1)

        hyp_channels = self.input_channel_map.dict()
        actual_channels = ["red", "green", "blue", "mask"]

        try:
            if not self._include_mask_input:
                hyp_channels.pop("mask")
        except KeyError:
            pass

        channels_sorted = sorted(hyp_channels.items(), key=lambda x: x[1])
        permute = [actual_channels.index(c[0]) for c in channels_sorted]
        if any(p < 0 for p in permute) or any(p > 3 for p in permute):
            raise ValueError("Hyperparamters passed field that is not available in the batch.")

        #Get the GSD info we care about
        gsd_batched = []
        for gsd in batch.getBatchedGSD():
            gsd_batched.append(gsd[0] / 100)
        batched_model_input.setField(GSD, gsd_batched)

        #Get the timestamp info we care about
        timestamps_batched = []
        for timestamp in batch.getBatchedTimestamp():
            timestamps_batched.append(timestamp)
        batched_model_input.setField(TIMESTAMP, timestamps_batched)

        #Get the channels that we are about
        channel_model_input = torch.cat((rgb, mask), dim=1)
        batched_model_input.setField(CHANNEL_INPUT, channel_model_input[:, permute])

        #Pass the mask that we care about
        batched_model_input.setField(MASK, mask)

        #Return the object that will be used to run the model
        return batched_model_input

    def load(self, path, strict=True):
        self.model.load(path, strict)

    def _reset_aggregation_step_metadata(self):
        self._step_metadata = ModelStepMetadata(self.global_step)
        self.mark_images_logged(False)

    def _add_batched_images_to_step_metadata(self, batched_images, name, inspection_func=inspect_image, image_limit=-1):
        #Count up the number of occurances of this name so we dont overwrite data in the object
        cur_keys_in_meta = self._step_metadata.images.keys()
        cur_id = 0
        for field_key in cur_keys_in_meta:
            if f"{name}/" in field_key:
                cur_id += 1

        #With the correct ID now known, now we can add the image
        images = inspection_func(batched_images)
        for img_idx, image in enumerate(images):
            if image_limit == -1 or (img_idx + cur_id) < image_limit:
                field_idx = cur_id + img_idx
                self._step_metadata.images[f"{name}/{field_idx}"] = image

     # pylint: disable-next=too-many-branches
    def _log_labels_update_loss(self, batched_labels=None):
        if not batched_labels is None:
            unique_values, counts = torch.unique(batched_labels, return_counts=True)
        else:
            unique_values, counts = [], []

        class_counts = {}
        sum_counts = sum(counts)
        for unique_value, count in zip(unique_values, counts):
            labels = self.output_label_map.getLabels(int(unique_value))
            if len(labels) > 0:
                class_counts[labels[0]] = count/sum_counts
            else:
                print("Warning: Found an index without a Label", labels, unique_value, count)

        for key, count in class_counts.items():
            self.running_class_counts[self.output_label_map.getIndex(key)] += count

        cw = [1] * len(self.output_label_map)
        strategy_name = self.model_hyperparameters["training"]["training_parameters"]["output_class_weights_strategy"].lower()

        if strategy_name == "uniform":
            self.class_weights = torch.tensor(cw).to(self._device)

        elif strategy_name == "manual":
            if len(self.model_hyperparameters["training"]["training_parameters"]["output_class_weights"]) != len(self.output_label_map):
                print(
                    "Warning: Found a different number of class weights vs output indicies in model hyperparameter."
                    + "This may result in some output classes being weighted incorrectly during training."
                )
            for label, weight in self.model_hyperparameters["training"]["training_parameters"]["output_class_weights"].items():
                cw[self.output_label_map.getIndex(label)] = weight

            self.class_weights = torch.tensor(cw).to(self._device)

        elif strategy_name in ["ipw", "log_class_balance"]:
            for class_idx, count in enumerate(self.running_class_counts):
                cw[class_idx] = count
            if strategy_name == "ipw":
                self.class_weights = torch.tensor(get_ipw_weights_from_class_counts(cw)).to(self._device)
            else:
                self.class_weights = torch.tensor(get_log_class_balanced_weights_from_class_counts(cw)).to(self._device)

        else:
            raise ValueError("Unknown value passed as output_class_weights_strategy, options are " + str(["uniform", "manual", "ipw", "log_class_balance"]))

        # Update the criterion with the new weights
        for i, weight in enumerate(self.class_weights):
            self._step_metadata.scalars["Loss/Class Weights"][self.output_label_map.getLabels(i)[0]] = weight
            count = self.running_class_counts[i]
            self._step_metadata.scalars["Loss/Running Class Counts"][self.output_label_map.getLabels(i)[0]] = count
        self.criterion.set_class_weights(self.class_weights, normalize=self.model_hyperparameters["training"]["training_parameters"]["normalize_weights"])

    def _log_metadata(self, logger):
        for name, scalars_collection in self._step_metadata.scalars.items():
            if scalars_collection.is_normalizable() and name in self._step_metadata.normalizations.keys():
                normalization_constant = self._step_metadata.normalizations[name]
                normalization_constant = 1 if normalization_constant == 0 else normalization_constant
                logger.add_scalars(
                    name + " (Normalized)",
                    {
                        k: v / normalization_constant
                        for k, v in scalars_collection.as_dict().items()
                    },
                    self._step_metadata.get_step(),
                )
            else:
                logger.add_scalars(name, scalars_collection.as_dict(), self._step_metadata.get_step())

        for name, scalar_collection in self._step_metadata.scalar.items():
            if self._step_metadata.scalar.is_normalizable(name) and name in self._step_metadata.normalizations.keys():
                normalization_constant = self._step_metadata.normalizations[name]
                normalization_constant = 1 if normalization_constant == 0 else normalization_constant
                logger.add_scalar(
                    name + " (Normalized)",
                    scalar_collection / normalization_constant,
                    self._step_metadata.get_step(),
                )
            else:
                logger.add_scalar(name, scalar_collection, self._step_metadata.get_step())

        for name, image in self._step_metadata.images.items():
            logger.add_image(name, image, self._step_metadata.get_step(), dataformats="HWC")

        for name, quantile_collection in self._step_metadata.quantiles.items():
            logger.add_scalars(
                name,
                {
                    "Quantile " + str(q): np.quantile(quantile_collection, q=q)
                    for i, q in enumerate(self._qs)
                },
                self._step_metadata.get_step(),
            )

    def get_tb_logger(self):
        try:
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    self._logger = logger.experiment
                    return self._logger
        except:  # pylint: disable=bare-except
            pass

        if self._logger is None:
            self._logger = TensorBoardLogger("tb_logs", name="debug_log").experiment
        return self._logger

    def mark_images_logged(self, mark=True):
        self._images_logged = mark

    def images_have_been_logged(self):
        return self._images_logged

    def should_log_images(self):
        is_new_step = self._cur_step != self.global_step
        images_requested_on_step = (
            self._log_images_every_n_steps > 0
            and self.global_step % self._log_images_every_n_steps == 0
        )
        return is_new_step and images_requested_on_step

    def _record_system_memory_telemetry(self):
        # Instantaneous host-RAM / GPU-VRAM gauges, sampled every MEMORY_TELEMETRY_EVERY_N_STEPS
        # optimizer steps from rank 0. Host memory is split into system-wide / this-process /
        # DataLoader-worker RSS so a worker-side leak (which the training process's own RSS would
        # hide) shows up as a rising line. These are gauges, so their group names are deliberately
        # kept out of self._step_metadata.normalizations (no per-batch division). The step gate
        # keeps the per-sample /proc worker-RSS scan off the hot path on busy shared nodes.
        if self.global_step % MEMORY_TELEMETRY_EVERY_N_STEPS != 0:
            return
        for key, value in get_host_memory_stats().items():
            if key.endswith("_gb"):
                self._step_metadata.scalars["System/Host Memory (GB)"][key] = value
            else:
                self._step_metadata.scalars["System/Host Process Counts"][key] = value
        for key, value in get_gpu_memory_stats(self.device).items():
            self._step_metadata.scalars["System/GPU Memory (GB)"][key] = value

    def on_after_backward(self):
        if self._cur_step != self.global_step:
            if self.trainer.is_global_zero:
                if self.should_log_images() and self.images_have_been_logged():
                    self._step_metadata.images["Gradient Flow"] = inspect_grad_flow(
                        self.model.named_parameters(), " | Step=" + str(self.global_step)
                    )
                self._record_system_memory_telemetry()
                self._log_metadata(self.get_tb_logger())
            self._reset_aggregation_step_metadata()
            self._cur_step = self.global_step

        # Measure time from end of training_step to after backward pass
        backward_overhead_time = time.time() - self._training_step_end_time
        self._step_metadata.scalars["Timing/Training Step Overhead"]["backward_pass"] += backward_overhead_time
        self._on_after_backward_time = time.time()

    def isNormalizedInput(self):
        return self._normalized_inputs

    def on_fit_start(self):
        print("Restored running class counts:", self.running_class_counts)

    def _build_optimizer_parameter_groups(self, optimizer_parameters):
        # Opt-in discriminative learning rate: "backbone_lr_scale" under encoder_parameters gives
        # the pretrained backbone its own, typically lower, LR. Fine-tuning pretrained weights at
        # the same LR as a freshly-initialized decoder generally destroys them. Absent the key
        # this returns a plain parameter iterator, so every other model is unaffected.
        encoder_parameters = self._model_hyperparameters.get("model_parameters", {}).get("encoder_parameters", {})
        scale = encoder_parameters.get("backbone_lr_scale")
        base_lr = optimizer_parameters.get("lr")
        encoder = getattr(self.model, "backbone", None)
        if scale is None or base_lr is None or encoder is None:
            return self.model.parameters()

        # An encoder that wraps a pretrained model in trainable machinery (the DINOv3 adapter)
        # reports only the pretrained subset; a plain pretrained encoder is itself the subset.
        if hasattr(encoder, "pretrained_parameters"):
            pretrained = list(encoder.pretrained_parameters())
        else:
            pretrained = list(encoder.parameters())

        trainable_pretrained = [p for p in pretrained if p.requires_grad]
        if not trainable_pretrained:
            print("Backbone LR scale set but the backbone is frozen; using a single parameter group.")
            return self.model.parameters()

        # The two groups must partition model.parameters(): anything the scaled group does not
        # claim goes in the base group, so no parameter is dropped from the optimizer.
        scaled_ids = {id(p) for p in trainable_pretrained}
        remaining = [p for p in self.model.parameters() if id(p) not in scaled_ids]

        print(f"Backbone LR scale {scale}: backbone lr={base_lr * scale}, remainder lr={base_lr}")
        return [
            {"params": remaining},
            {"params": trainable_pretrained, "lr": base_lr * scale},
        ]

    def get_optimizer(self):
        optimizer_parameters = self._model_hyperparameters["training"]["training_parameters"]["optimizer_parameters"].copy()
        optimizer_name = optimizer_parameters.pop("name")

        optimizer_cls = None
        for name in dir(optim):
            if name.lower() == optimizer_name.lower():
                optimizer_cls = getattr(optim, name)
        if optimizer_cls is None:
            if optimizer_name.lower() == "adafactor":
                optimizer_cls = AdaFactor
            else:
                raise ValueError(f"Optimizer '{optimizer_name}' not found.")
        self.optimizer = optimizer_cls(
            self._build_optimizer_parameter_groups(optimizer_parameters), **optimizer_parameters
        )
        return self.optimizer

    def configure_optimizers(self):
        validation_parameters = self._model_hyperparameters["validation"]["validation_parameters"]
        factor = validation_parameters.get("factor", 0.1)
        # The plateau scheduler may follow a different validation signal than checkpointing/early
        # stopping (validation_scheduler_monitor / validation_scheduler_monitor_mode; both default
        # to the shared validation_monitor keys). CHANGE points it at val_criterion_loss: an
        # early-epoch macro-F1 fluke otherwise sets a high-water mark nothing re-passes, and the
        # plateau schedule decays the LR to zero while the loss is still improving.
        scheduler_monitor = validation_parameters.get("validation_scheduler_monitor", validation_parameters["validation_monitor"])
        scheduler_mode = validation_parameters.get("validation_scheduler_monitor_mode", validation_parameters["validation_monitor_mode"])
        optimizer = self.get_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=validation_parameters["validation_reduce_lr_on_plateau_patience"],
            cooldown=validation_parameters["validation_reduce_lr_on_plateau_cooldown"],
            mode=scheduler_mode,
            factor=factor,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": scheduler_monitor,
            },
        }

    def configure_checkpoint(self):
        monitor = self._model_hyperparameters["validation"]["validation_parameters"]["validation_monitor"]
        naming_pattern = str(self.getName()) + "-{epoch:02d}-{step}-{" + monitor + ":.5f}"
        return ModelCheckpoint(
            monitor=monitor,
            save_top_k=self._model_hyperparameters["validation"]["validation_parameters"]["validation_checkpoint_save_top_k"],
            mode=self._model_hyperparameters["validation"]["validation_parameters"]["validation_monitor_mode"],
            filename=naming_pattern,
        )

    def configure_early_stopping(self):
        validation_parameters = self._model_hyperparameters["validation"]["validation_parameters"]
        patience = validation_parameters.get("validation_early_stopping_patience", 12)
        if patience is None or patience <= 0:
            return None
        return EarlyStopping(
            monitor=validation_parameters["validation_monitor"],
            mode=validation_parameters["validation_monitor_mode"],
            patience=patience,
        )

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def _compute_y_hat_inference(self, batch):
        # Return masked logits: the criterion (F.cross_entropy) expects logits, and callers
        # apply softmax explicitly when they need probabilities. This keeps validation and
        # inference consistent with the training path, which also runs with DO_SOFTMAX off.
        x = self.format_batched_data_to_model_input(batch)
        x.setField(DO_SOFTMAX, False)
        model_output = self.model(x) # Obtain the model outputs
        y_hat = model_output[Y_HAT_SEGMENTATION_MASKED]
        del x
        return y_hat

    def on_train_start(self):
        #We are having some issues with bad values showing up in gradients when runs are resumed. This is an attempt to resolve it.
        if self.trainer.scaler is not None:
            scaler_state = self.trainer.scaler.state_dict()
            scaler_state['scale'] = 65536.0
            scaler_state['growth_tracker'] = 0
            self.trainer.scaler.load_state_dict(scaler_state)

    def _train_preamble_forward(self, batch):
        start_time = time.time()

        # Set the model to train mode
        self.model.train()

        init_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INIT] += init_time - start_time

        # Get the class losses from the model...
        x = self.format_batched_data_to_model_input(batch)
        x.setField(DO_SOFTMAX, False)

        self._sync()
        preprocess_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_PREPROCESS] += preprocess_time - init_time

        y_hat = self.model(x)
        del x

        self._sync()
        forward_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_FORWARD] += forward_time - preprocess_time

        return y_hat

    def _train_loss_timing_telemetry_cleanup(self, y_hat, loss_dict, batch):
        log_start_time = time.time()
        # If we need to log out data...
        if self.should_log_images() and not self.images_have_been_logged():
            self._add_batched_images_to_step_metadata(batch.getBatchedRawImagery(), "Image", inspect_image)
            self._add_batched_images_to_step_metadata(batch.getBatchedLabels(), "Label",lambda x: inspect_labels(x, self.idx2color_map))
            self._add_batched_images_to_step_metadata(torch.argmax(y_hat, 1), "Preds", lambda x: inspect_labels(x, self.idx2color_map))
            self.mark_images_logged()

        # Update the dictionaries that contain our tracking data...
        self._log_labels_update_loss(batch.getBatchedLabels())
        self.log_batch_telemetry(batch)
        self.log_loss_telemetry(loss_dict, batch)

        log_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOG] += log_time - log_start_time

    def _train_compute_loss(self, y_hat, batch):
        self._sync()
        loss_start_time = time.time()

        # Compute the criterion loss
        criterion_loss = self.criterion(y_hat, batch.getBatchedLabels()) * self.criterion_scale_factor

        # Compute the L2 Loss
        l1_reg_loss = self.get_l1_loss()

        # Compute the L1 Loss
        l2_reg_loss = self.get_l2_loss()

        # Compute the total loss for the model
        # Warning for nan/inf loss values
        if self._criterion_loss_is_unusable(criterion_loss):
            print("Warning: criterion_loss contains NaN or Inf! Ignoring sample...")
            loss = l2_reg_loss + l1_reg_loss
            criterion_loss = torch.zeros_like(criterion_loss)
        else:
            loss = l2_reg_loss + l1_reg_loss + criterion_loss

        loss_dict = {
            "Final Loss":float(loss.detach().cpu()),
            "Criterion Loss":float(criterion_loss.mean().detach().cpu()),
            "L1 Regularization Loss":float(l1_reg_loss.detach().cpu()),
            "L2 Regularization Loss":float(l2_reg_loss.detach().cpu())
        }

        self._sync()
        loss_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOSS] += loss_time - loss_start_time
        return loss, loss_dict

    # add seperate function for added uq loss
    def _train_compute_uq_loss(self, y_hat, y_hat_uq, batch):
        loss_start_time = time.time()

        # Compute the criterion loss
        criterion_loss = self.criterion(y_hat, batch.getBatchedLabels()) * self.criterion_scale_factor

        # Compute the L1 Loss
        l1_reg_loss = self.get_l1_loss()

        # Compute UQ loss: negative log likelihood over the sampled UQ probabilities, following
        # the reference implementation of Kendall & Gal's combined-uncertainty classification:
        # https://github.com/ShellingFord221/My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision
        uq_loss = self._uq_loss_weight * F.nll_loss(y_hat_uq, batch.getBatchedLabels(), ignore_index=self.output_background_idx)

        # Compte ECE
        ece = self.compute_ece(y_hat_uq, batch)

        # Compute the total loss for the model
        # Warning for nan/inf loss values; a non-finite term is dropped so it cannot poison the
        # gradients of the healthy terms.
        if self._criterion_loss_is_unusable(criterion_loss):
            print("Warning: criterion_loss contains NaN or Inf! Ignoring sample...")
            criterion_loss = torch.zeros_like(criterion_loss)
        if self._criterion_loss_is_unusable(uq_loss):
            print("Warning: uq_loss contains NaN or Inf! Ignoring sample...")
            uq_loss = torch.zeros_like(uq_loss)
        loss = l1_reg_loss + criterion_loss + uq_loss

        loss_dict = {
            "Final Loss":float(loss.item()),
            "Criterion Loss":float(criterion_loss.mean().item()),
            "L1 Regularization Loss":float(l1_reg_loss.item()),
            "Uncertainty Quant Loss":float(uq_loss.item()),
            "ECE": float(ece.item())
        }

        loss_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOSS] += loss_time - loss_start_time
        return loss, loss_dict

    # pylint: disable-next=arguments-differ, unused-argument
    def training_step(self, batch, batch_idx):
        self._step_metadata.scalars["Timing/Training Step Overhead"]["backward_to_next_start"] += time.time() - self._on_after_backward_time
        start_time = time.time()

        model_output = self._train_preamble_forward(batch)

        if self.uq_hyperparameters:
            y_hat = model_output[Y_HAT_SEGMENTATION_MASKED]
            y_hat_uq = model_output[UQ_PREDICTION]
            loss, loss_dict = self._train_compute_uq_loss(y_hat, y_hat_uq, batch) # computes total loss (loss + uq_loss) and returns loss_dict
        else:
            y_hat = model_output[Y_HAT_SEGMENTATION_MASKED]
            loss, loss_dict = self._train_compute_loss(y_hat, batch)

        self._train_loss_timing_telemetry_cleanup(y_hat, loss_dict, batch)

        done_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INTER_STEP] += done_time - start_time
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INTRA_STEP] += start_time - self._prev_start_time
        self._step_metadata.normalizations["Timing/Training Step Timings"] += len(batch)

        self._cur_iter += 1
        self._prev_start_time = start_time
        self._training_step_end_time = done_time

        return loss

    def compute_ece(self, y_hat_uq, batch, num_bins=10):
        # Compute Expected Calibration Error (ECE)
        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=y_hat_uq.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        probs = torch.exp(y_hat_uq)
        confidences, predictions = probs.max(dim=1)

        # Flatten everything to 1D
        confidences = confidences.flatten()
        predictions = predictions.flatten()
        labels = batch.getBatchedLabels().to(y_hat_uq.device).view(-1)

        # We have to ignore the background mask from the ece because those pixels are fixed values.
        valid_mask = labels != self.output_background_idx
        confidences = confidences[valid_mask]
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]

        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=y_hat_uq.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                acc = accuracies[in_bin].float().mean()
                conf = confidences[in_bin].mean()
                ece += torch.abs(conf - acc) * prop_in_bin

        return ece
