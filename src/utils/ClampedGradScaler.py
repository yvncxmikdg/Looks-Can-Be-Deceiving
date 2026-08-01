from torch.cuda.amp import GradScaler


class ClampedGradScaler(GradScaler):
    def __init__(self, *args, min_scale=64.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_scale = min_scale

    def update(self, new_scale=None):
        super().update(new_scale)

        # Intercept the scale tensor and clamp it if it falls below the floor
        if self._scale is not None and self._scale.item() < self.min_scale:
            self._scale.fill_(self.min_scale)
