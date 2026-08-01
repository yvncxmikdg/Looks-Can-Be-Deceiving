import torch

def adapt_checkpoint(checkpoint, num_channels=4):
# We added to handel 4 channels RGB + Polygons/Roadlines
    # Load the original model weights for patch embedding
    original_weight = checkpoint['model']['patch_embed.proj.weight']
    if original_weight.size(1) != num_channels:
        # Initialize the new weights
        new_weight = torch.zeros((original_weight.size(0), num_channels, original_weight.size(2), original_weight.size(3)))
        # Copy the existing weights for the first three channels
        new_weight[:, :3, :, :] = original_weight
        # Copy from the first channel
        new_weight[:, 3, :, :] = original_weight[:, 0, :, :]  
        # Update the checkpoint
        checkpoint['model']['patch_embed.proj.weight'] = new_weight
    return checkpoint