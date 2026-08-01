import os
import glob

def parse_continued_model_checkpoint(out_path, model_hyperparameters):
    ckpt_path = None
    path = os.path.join(
        os.path.join(out_path, "tb_logs"),
        str(model_hyperparameters["name"]) + "_" + str(model_hyperparameters["task"]),
        "version_*",
    )
    print("Searching", path)
    # Find latest version for the checkpoint
    version_folders = sorted(
        glob.glob(path),
        key=os.path.getctime,
    )

    # If we were able to find previous model runs that have been started....
    if version_folders:
        latest_version = version_folders[-1]
        print(latest_version)
        checkpoint_dir = os.path.join(latest_version, "checkpoints")
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getctime)
        # ...and if there are checkpoint files stored in those previous runs
        if checkpoint_files:
            # Get the checkpoint file that is the farthest along in training
            ckpt_path = checkpoint_files[-1]
            print("Found checkpoint, Resuming Training...")
            print("\tCheckpoint found at", ckpt_path)
    if ckpt_path is None:
        print("Did Not Find Checkpoint, Restarting Training...")
    return ckpt_path
