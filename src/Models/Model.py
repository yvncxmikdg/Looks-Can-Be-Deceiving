import os

import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as L

from modeling.Models.Maskable import Maskable
from modeling.Models.ModelDatum import (
    ModelOutput,
    Y_HAT_SEGMENTATION_UNMASKED,
    Y_HAT_SEGMENTATION_MASKED,
    Y_HAT_SEGMENTATION_LOGITS,
    UQ_PREDICTION,
    MU,
    SIGMA,
    DO_SOFTMAX,
    MASK,
    CHANNEL_INPUT,
)

NUM_UQ_LOOPS = 4
NUM_VECTORIZED_SAMPLES = 3  # Total samples = 4 * 3 = 12

def _state_dict_backwards_compat(state_dict):
    for key in list(state_dict.keys()):
        # Task-module-only buffers (e.g. the running class counts used to resume class-weight
        # strategies) have no counterpart in the bare inference model - drop them.
        if "running_class_counts" in key:
            del state_dict[key]
        elif "model.model." in key:
            state_dict[key.replace("model.model.", "model.")] = state_dict.pop(key)
        else:
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


class SegmentationModelOutputModule(nn.Module):
    def __init__(self, model, uq_hyperparameters=None, n_cls=None):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.uq_hyperparameters = None
        self.mu_sigma_layer = None
        self._uq_n_cls = None
        if uq_hyperparameters is not None and n_cls is not None:
            self.initialize_uq(uq_hyperparameters, n_cls)

    def initialize_uq(self, uq_hyperparameters, n_cls):
        # Attach the aleatoric mu/sigma head. The head consumes the tower's class logits, so its
        # input width and its 2*n_cls output (a mu and a sigma per class) are sized by the task's
        # output label map rather than the config, which only shapes the hidden layer.
        self.uq_hyperparameters = uq_hyperparameters
        self._uq_n_cls = n_cls
        self.mu_sigma_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=n_cls,
                out_channels=uq_hyperparameters["conv1"]["out_channels"],
                kernel_size=uq_hyperparameters["conv1"]["kernel_size"],
                stride=uq_hyperparameters["conv1"]["stride"],
                padding=uq_hyperparameters["conv1"]["padding"],
            ),
            nn.ReLU(),
            nn.Dropout2d(uq_hyperparameters["dropout"]),
            nn.Conv2d(
                in_channels=uq_hyperparameters["conv1"]["out_channels"],
                out_channels=2 * n_cls,
                kernel_size=uq_hyperparameters["conv2"]["kernel_size"],
                stride=uq_hyperparameters["conv2"]["stride"],
                padding=uq_hyperparameters["conv2"]["padding"],
            )
        )

    def load(self, path, strict=True, assign=True):
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"]

        # Probe leniently first: a legacy checkpoint stores the tower under a "model.model."
        # prefix, so every one of its keys comes back as "unexpected" here. That remap has to run
        # regardless of the caller's `strict` preference - a silent strict=False load would match
        # zero keys and leave the model on its random init (it "loads" fine but predicts garbage).
        probe = self.load_state_dict(state_dict, strict=False, assign=assign)
        if probe.unexpected_keys:
            self.load_state_dict(_state_dict_backwards_compat(state_dict), strict=strict, assign=assign)
        elif strict and probe.missing_keys:
            # No prefix mismatch, but the caller asked for a strict load and the lenient probe
            # swallowed missing keys - redo it strictly so they still get the error they asked for.
            self.load_state_dict(state_dict, strict=True, assign=assign)

    def _compute_segmentation_output(self, model_input):
        # Subclasses own the "prepared input -> raw segmentation logits" step. forward() below
        # wraps this with the shared UQ head and interpolation tail and is deliberately NOT
        # overridden by subclasses, so no code path can accidentally bypass the UQ pathway
        # (which is exactly how the EncodeDecode path used to silently drop the mu/sigma head).
        # Tower runs a single end-to-end model; EncodeDecode runs its backbone then its decoder.
        raise NotImplementedError(
            "_compute_segmentation_output() must be implemented by a subclass."
        )

    def forward(self, model_input):
        result = ModelOutput()

        segmentation_output = self._compute_segmentation_output(model_input)
        result.setField(Y_HAT_SEGMENTATION_UNMASKED, segmentation_output)

        if self.mu_sigma_layer is not None:
            mu, sigma = self.mu_sigma_layer(segmentation_output).split(self._uq_n_cls, dim=1)
            # pylint: disable-next=not-callable
            sigma = F.softplus(sigma) + 1e-6 # ensure that sigma is positive

            # Initialize a running sum tensor with the same shape as mu: [B, C, H, W]
            prob_sum = torch.zeros_like(mu)

            for _ in range(NUM_UQ_LOOPS):

                # Create epsilon tensor for just this chunk: [VECTOR_SAMPLES, B, C, H, W]
                sample_shape = (NUM_VECTORIZED_SAMPLES, *mu.shape)
                epsilon = torch.randn(sample_shape, device=mu.device)

                # Broadcast mu and sigma [1, B, C, H, W] to multiply with epsilon
                logits = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon

                # Softmax along the class dimension (dim=2)
                probs = F.softmax(logits, dim=2)

                # Sum across the vector dimension (dim=0) to squash it back to [B, C, H, W]
                # and add it to our running total.
                # This immediately frees the massive [4, B, C, H, W] tensors from VRAM for the next loop.
                prob_sum += probs.sum(dim=0)

            # Average across the total number of samples generated
            total_samples = NUM_UQ_LOOPS * NUM_VECTORIZED_SAMPLES
            expected_probs = prob_sum / total_samples

            result.setField(UQ_PREDICTION, expected_probs)
            result.setField(MU, mu)
            result.setField(SIGMA, sigma)

        scaled_result = self._interpolate_and_softmax(
            result, model_input[CHANNEL_INPUT].shape[-2:], model_input[DO_SOFTMAX]
        )
        return scaled_result

    def _interpolate_and_softmax(self, result, target_shape, do_softmax):
        result.setField(DO_SOFTMAX, do_softmax)

        # Perform the interpolation for the predictions
        preds = F.interpolate(
            result[Y_HAT_SEGMENTATION_UNMASKED],
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
        # Always store the pre-softmax logits: Maskable.mask() must operate on these (see
        # Y_HAT_SEGMENTATION_LOGITS docstring), not on the possibly-already-softmaxed UNMASKED field.
        result.setField(Y_HAT_SEGMENTATION_LOGITS, preds)

        if do_softmax:
            result.setField(Y_HAT_SEGMENTATION_UNMASKED, self.softmax(preds))
        else:
            result.setField(Y_HAT_SEGMENTATION_UNMASKED, preds)

        # Also perform the interpolation for the UQ logic so we return calibrated labels per output pixel.
        if result.contains(UQ_PREDICTION):
            uq_preds = F.interpolate(
                result[UQ_PREDICTION],
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
            # Ensure interpolation didn't break sum-to-1
            uq_preds = uq_preds / uq_preds.sum(dim=1, keepdim=True)

            if not do_softmax:
                # Log-probs behave exactly like logits for downstream loss functions
                uq_preds = torch.log(uq_preds.clamp(min=1e-8))

            result.setField(UQ_PREDICTION, uq_preds)

        return result

    def prepare_model_input_func(self, model_input):
        return (model_input[CHANNEL_INPUT],)


# Towers are containers for models that we expect to run end to end as a single model and return ModelOutputs
# These are things like UNets which are not generally broken down into composite components except for the separate layers.
class Tower(SegmentationModelOutputModule):
    def _compute_segmentation_output(self, model_input):
        args = self.prepare_model_input_func(model_input)
        return self.model(*args)


# A Masked tower is an object that can be created that extends the existing tower logic and adds the masking logic to it also
class MaskedTower(Tower, Maskable):
    def __init__(
        self,
        model,
        n_cls,
        input_channel_mask_index=-1,
        output_channel_background_index=-1,
    ):
        Tower.__init__(self, model)
        Maskable.__init__(
            self, n_cls, input_channel_mask_index, output_channel_background_index
        )

    def forward(self, model_input):
        result = super().forward(model_input)
        if self.mask_output:
            masked_logits = self.mask(result[Y_HAT_SEGMENTATION_LOGITS], model_input[MASK])
            if model_input[DO_SOFTMAX]:
                masked_output = self.softmax(masked_logits)
            else:
                masked_output = masked_logits
            result.setField(Y_HAT_SEGMENTATION_MASKED, masked_output)
        return result


class TowerModule(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()
        self._model = self._load_tower_model(
            hyperparameters, input_channel_map, output_label_map
        )

    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        raise NotImplementedError(
            "_load_tower_model() has not yet been implemented by a subclass."
        )

    def get_model(self):
        return self._model

    def on_load_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace("model.model.", "_model.model.")] = (
                checkpoint["state_dict"].pop(key)
            )


class MaskedTowerModule(TowerModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__(hyperparameters, input_channel_map, output_label_map)

        # Initialize Tower
        model = self._load_tower_model(
            hyperparameters, input_channel_map, output_label_map
        )

        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
            output_channel_background_index = output_label_map.getBackgroundClassIdx()
        except KeyError:
            print("Warning: There is no mask channel. Continuing without it...")
            input_channel_mask_index = -1
            output_channel_background_index = -1

        # Initalize MaskedUperNet with backbone
        self._model = MaskedTower(
            model,
            len(output_label_map),
            input_channel_mask_index=input_channel_mask_index,
            output_channel_background_index=output_channel_background_index,
        )

    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        raise NotImplementedError(
            "_load_tower_model() has not yet been implemented by a subclass."
        )


class EncoderModule:
    def __init__(self):
        pass

    def load_encoder_model(self, hyperparameters, output_label_map):
        raise NotImplementedError(
            "load_encoder_model() has not yet been implemented by a subclass."
        )

    def is_encoder(self):
        return True

    def is_decoder(self):
        return False


class DecoderModule:
    def __init__(self):
        pass

    def load_decoder_model(self, hyperparameters, output_label_map):
        raise NotImplementedError(
            "load_decoder_model() has not yet been implemented by a subclass."
        )

    def is_encoder(self):
        return False

    def is_decoder(self):
        return True


class EncodeDecode(SegmentationModelOutputModule):
    def __init__(self, encoder, decoder, prepare_model_input_func):
        super().__init__(None)
        self.softmax = nn.Softmax(dim=1)
        self.backbone = encoder
        self.decoder = decoder
        self.prepare_model_input_func = prepare_model_input_func

    def _compute_segmentation_output(self, model_input):
        args = self.prepare_model_input_func(model_input)
        intermediate = self.backbone(*args)
        return self.decoder(intermediate)


class MaskedEncodeDecode(EncodeDecode, Maskable):
    def __init__(
        self,
        encoder,
        decoder,
        prepare_model_input_func,
        n_cls,
        input_channel_mask_index=-1,
        output_channel_background_index=-1,
    ):
        EncodeDecode.__init__(self, encoder, decoder, prepare_model_input_func)
        Maskable.__init__(
            self, n_cls, input_channel_mask_index, output_channel_background_index
        )

    def forward(self, model_input):
        result = super().forward(model_input)
        if self.mask_output:
            masked_logits = self.mask(result[Y_HAT_SEGMENTATION_LOGITS], model_input[MASK])
            if model_input[DO_SOFTMAX]:
                masked_output = self.softmax(masked_logits)
            else:
                masked_output = masked_logits
            result.setField(Y_HAT_SEGMENTATION_MASKED, masked_output)
        return result


class EncoderDecoderModule(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        hyperparameters=None,
        input_channel_map=None,
        output_label_map=None,
    ):
        super().__init__()

        # Initialize Encoder
        _backbone = encoder.load_encoder_model(hyperparameters, input_channel_map)

        # Initialize Decoder
        _decoder = decoder.load_decoder_model(hyperparameters, output_label_map)

        # Get the function to prepare inputs for the loaded encoder
        self._prep_model_input_func = encoder.prepare_model_input

        try:
            if hyperparameters["input"]["model_parameters"]["encoder_parameters"][
                "freeze_backbone"
            ]:
                print(
                    "NOTICE: Freezing Weights for backbone, this assumes pretrained backbone..\n"
                )
                for param in _backbone.parameters():
                    param.requires_grad = False
        except KeyError:
            pass

        # Initalize EncodeDecode with the models we have loaded
        self._model = EncodeDecode(_backbone, _decoder, self._prep_model_input_func)

    def get_model(self):
        return self._model

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint["model"])

    def on_load_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("model."):
                checkpoint["state_dict"][key.replace("model.", "_model.")] = checkpoint[
                    "state_dict"
                ].pop(key)


class MaskedEncoderDecoderModule(EncoderDecoderModule):
    def __init__(
        self,
        encoder,
        decoder,
        hyperparameters=None,
        input_channel_map=None,
        output_label_map=None,
    ):
        super().__init__(
            encoder, decoder, hyperparameters, input_channel_map, output_label_map
        )

        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
            output_channel_background_index = output_label_map.getBackgroundClassIdx()
        except KeyError:
            print("Warning: There is no mask channel....Continuing without it...")
            input_channel_mask_index = -1
            output_channel_background_index = -1

        # Initalize MaskedEncodeDecode with the models we have loaded
        self._model = MaskedEncodeDecode(
            self._model.backbone,
            self._model.decoder,
            self._prep_model_input_func,
            len(output_label_map),
            input_channel_mask_index=input_channel_mask_index,
            output_channel_background_index=output_channel_background_index,
        )
