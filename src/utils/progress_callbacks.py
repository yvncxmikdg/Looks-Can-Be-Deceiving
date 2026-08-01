import time

from pytorch_lightning.callbacks import Callback

class SimpleTextProgressCallback(Callback):
    """
    A custom callback that prints a simple text string instead of
    animating a progress bar. Includes an ETA. Safe for Tkinter and GUI redirects.
    """
    # pylint: disable=unused-argument

    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_predict_start(self, trainer, pl_module):
        # Record the exact time inference begins
        self.start_time = time.time()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        total_batches = trainer.num_predict_batches[dataloader_idx]
        current_batch = batch_idx + 1

        # Calculate time elapsed and estimate time remaining
        elapsed_seconds = time.time() - self.start_time

        # Avoid division by zero on the extremely fast/first batch
        if current_batch > 0:
            time_per_batch = elapsed_seconds / current_batch
            batches_remaining = total_batches - current_batch
            eta_seconds = time_per_batch * batches_remaining
        else:
            eta_seconds = 0

        elapsed_str = self._format_time(elapsed_seconds)
        eta_str = self._format_time(eta_seconds)
        percent_complete = (current_batch / total_batches) * 100

        # Print a simple, clean line with flush=True to guarantee it hits the GUI instantly
        print(f"Inference Progress: {percent_complete:.1f}% - Batch {current_batch}/{total_batches} | Elapsed: {elapsed_str} | ETA: {eta_str}", flush=True)

    def _format_time(self, seconds):
        """Helper function to format seconds into MM:SS or HH:MM:SS"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
