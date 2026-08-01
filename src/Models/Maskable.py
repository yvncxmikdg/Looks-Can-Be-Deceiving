import torch

class Maskable:
    def __init__(self, n_classes, input_channel_mask_index, output_channel_background_index):
        self.n_maskable_classes = n_classes
        self.input_channel_mask_index = input_channel_mask_index
        self.output_channel_background_index = output_channel_background_index
        self.mask_output = input_channel_mask_index >= 0 and output_channel_background_index >= 0

        if output_channel_background_index >= 0 or input_channel_mask_index >= 0:
            if not self.mask_output:
                raise ValueError("Both output_channel_background_index and input_channel_mask_index must be specified when masking.")

    def mask(self, masked_logits, mask):
        binary_mask_background = 1-mask == 1
        binary_mask_objects = mask == 1
        stack = [binary_mask_background]*self.n_maskable_classes
        stack[self.output_channel_background_index] = binary_mask_objects
        stacked_mask = torch.stack(stack, dim=1).squeeze(dim=2)
        output_masked_logits = masked_logits.clone()
        output_masked_logits[stacked_mask] = -65504 #Underflow limit for half precision floats
        return output_masked_logits
    def get_classes_count(self):
        return self.n_maskable_classes
    def get_input_channel_mask_index(self):
        return self.input_channel_mask_index
    def get_output_channel_background_index(self):
        return self.output_channel_background_index
