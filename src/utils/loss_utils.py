import torch
from torch import nn
import torch.nn.functional as F

class CCE(nn.Module):
    def __init__(self, weights, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.weights = weights
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, preds, target):
        # reduction="mean" divides by the number of NON-ignored pixels, so a tile in which every
        # pixel is ignore_index (a sample location that caught no labelled building) evaluates 0/0
        # and returns NaN. The correct contribution from a tile with nothing to learn from is zero.
        # Returning a graph-connected zero rather than a bare constant keeps the autograd graph the
        # same shape as every other step, which matters under DDP: a rank whose graph differs from
        # its peers' desynchronises gradient reduction.
        if self.ignore_index is not None and not (target != self.ignore_index).any():
            return preds.sum() * 0.0
        return F.cross_entropy(
            preds,
            target,
            weight=self.weights,
            reduction="mean",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, target):
        batch_size, num_classes, _, _ = preds.shape
        device = preds.device
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(batch_size, -1, num_classes)
        target = target.view(-1)

        targets_one_hot = (
            # pylint: disable-next=not-callable
            F.one_hot(target.long(), num_classes=num_classes)
            .float()
            .to(device)
        )
        targets_one_hot = targets_one_hot.view(batch_size, -1, num_classes)

        intersection = torch.sum(preds * targets_one_hot, dim=1)
        union = torch.sum(preds + targets_one_hot, dim=1)

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coeff.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weights=None, ignore_index=-100):
        super().__init__()
        if weights is None:
            self.alpha = alpha
        else:
            self.alpha = weights
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        # Honor ignore_index like CCE: cross_entropy zeroes the ignored pixels' contribution and we
        # average over only the valid pixels below, so the background/mask class does not dilute the loss.
        ce_loss = F.cross_entropy(preds, target, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, torch.Tensor):
            # Ignored pixels can carry an out-of-range target (e.g. -100); clamp before indexing
            # alpha and drop their contribution via valid_mask so the lookup can't go out of bounds.
            safe_target = target.clamp(min=0, max=self.alpha.shape[0] - 1)
            alpha_t = self.alpha.to(preds.device)[safe_target]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return focal_loss.sum() * 0.0
        return focal_loss[valid_mask].mean()


# GHM-C Loss implementation based on (Li et. al, 2019) https://arxiv.org/abs/1811.05181
# modified from https://github.com/libuyu/mmdetection/blob/master/mmdet/models/losses/ghm_loss.py
class GHMCLoss(nn.Module):
    def __init__(self, bins=10, momentum=0.9):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.linspace(0, 1, steps=bins + 1)
        self.edges[-1] += 1e-6

        if self.momentum > 0:
            self.acc_sum = torch.zeros(bins)

    def forward(self, preds, target):
        device = preds.device
        # pylint: disable-next=not-callable
        target_one_hot = F.one_hot(target.long(), num_classes=preds.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        p_t = (preds * target_one_hot).sum(dim=1)
        gradients = torch.abs(p_t - 1.0).view(-1)

        inds = torch.bucketize(gradients, self.edges.to(device)) - 1
        inds = torch.clamp(inds, min=0, max=self.bins - 1)

        bin_count = torch.bincount(inds, minlength=self.bins).float()

        if self.momentum > 0:
            self.acc_sum = (
                self.momentum * self.acc_sum.to(device) + (1 - self.momentum) * bin_count
            )
            bin_count = self.acc_sum

        weights = 1.0 / (bin_count[inds] + 1e-6)
        weights = weights.view(-1)

        ce_loss = F.cross_entropy(preds, target, reduction="none").view(-1)

        weighted_loss = ce_loss * weights

        return weighted_loss.mean()

class WeightedLoss(nn.Module):
    def __init__(self, loss_parameters, class_weights=None, ignore_index=-100, eps=1.0e-15):
        super().__init__()
        print("\tInitial class_weights", class_weights)
        self._ignore_index = ignore_index
        self._eps = eps
        self.set_class_weights(class_weights)

        self.loss_functions = {}
        for loss_name, weight in loss_parameters["loss"]:
            self.loss_functions[self._initialize_loss(loss_name, loss_parameters)] = float(weight)

    def set_class_weights(self, new_class_weight, normalize=False):
        if not new_class_weight is None:
            if normalize:
                max_weight = new_class_weight.max()
                min_weight = new_class_weight.min()
                self.class_weights = ((new_class_weight - min_weight) + self._eps)/((max_weight - min_weight) + self._eps)
            else:
                self.class_weights = new_class_weight
        else:
            print("Class Weights are None...")
            self.class_weights = None

    def _initialize_loss(self, loss_function_name, loss_parameters):
        if loss_function_name == "cross entropy":
            ls = loss_parameters.get("label_smoothing", 0.0) # Get "label_smoothing" if it exists, else default to 0.0
            loss_function = CCE(self.class_weights, self._ignore_index, label_smoothing=ls)
        elif loss_function_name == "dice":
            loss_function = DiceLoss(loss_parameters["smooth"])
        elif loss_function_name == "focal":
            loss_function = FocalLoss(loss_parameters["alpha"], loss_parameters["gamma"], self.class_weights,
                                      ignore_index=self._ignore_index)
        elif loss_function_name == "ghm-c":
            loss_function = GHMCLoss(loss_parameters["bins"], loss_parameters["momentum"])
        else:
            raise ValueError(f"Unknown loss function: {loss_function_name}")
        return loss_function

    def forward(self, preds, target):
        total_loss = 0
        # pylint: disable-next=consider-using-dict-items
        for loss in self.loss_functions:
            total_loss += loss(preds, target) * self.loss_functions[loss]
        return total_loss

def get_ipw_weights_from_class_counts(label_counts, eps=1.0e-6):
    class_counts_tensor = torch.tensor(label_counts)
    class_weights = 1.0 / (class_counts_tensor + eps)
    return class_weights / class_weights.sum()

def get_log_class_balanced_weights_from_class_counts(label_counts):
    #See details: https://arxiv.org/abs/1901.05555
    #Useful for long tail distributions
    class_counts_tensor = torch.tensor(label_counts)
    class_weights = 1.0 / torch.log(1.02 + class_counts_tensor)
    return class_weights / class_weights.sum()
