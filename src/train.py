import os
import json
import argparse
import torch
import tables as tb

import matplotlib

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.precision import MixedPrecision
from lightning.pytorch.loggers import TensorBoardLogger

from modeling.utils.hyperparameters import parse_hyperparameters, add_hyperparameters_files_to_parse_args
from modeling.utils.data_augmentations import get_valid_transforms, get_train_transforms
from modeling.utils.alerts import Alerter

from modeling.data_modules.TrainValPredictDataModule import TrainValPredictDataModule
from modeling.Orthomosaic import MultisourceOrthomosaicFactory, remove_orthomosaic_from_list_by_name
from modeling.Models.model_registry import parse_and_initialize_segmentation_model
from modeling.utils.initialize_dataset import initialize_windowed_dataset
from modeling.utils.parse_continued_model_checkpoint import parse_continued_model_checkpoint
from modeling.utils.provenance import build_training_metadata, write_run_manifest
from modeling.utils.run_metadata import current_timestamp
from modeling.utils.ClampedGradScaler import ClampedGradScaler
from modeling.utils.accelerator import resolve_torch_device

matplotlib.use('Agg')
local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"Running process {os.getpid()} on local rank {local_rank} and __name__ = {__name__}")

tb.parameters.MAX_BLOSC_THREADS = 1
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the specified model on the predefined task.")
    add_hyperparameters_files_to_parse_args(parser,
                                            add_dataset_paths_file_path=True,
                                            add_data_source_config_parameters_file_path=True,
                                            add_channels_hyperparameters_file_path=True,
                                            add_train_datagen_hyperparameters_yaml_path=True,
                                            add_val_datagen_hyperparameters_yaml_path=True,
                                            add_model_hyperparameters_yaml_path=True,
                                            add_uq_hyperparameters_yaml_path=True
                                            )
    parser.add_argument(
        "--out_path",
        type=str,
        help="The path to the folder where the metrics, logs, and checkpoints will be saved.",
        default="./",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="The path to the model checkpoint from which training should resume.",
        default=None,
    )
    parser.add_argument(
        "--data_gen_workers",
        type=int,
        help="The number of worker processes that will be used for data generation",
        default=12,
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="The floating point precision with which the model should be trained.",
        default="16-mixed",
    )
    parser.add_argument(
        "--accelerator",
        help="Which hardware accelerator should be used for model training (cpu, gpu, tpu, mps)",
        default="cpu",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When set, use simplified parameters to initialize things faster.",
    )
    parser.add_argument(
        "--train_backend",
        type=str,
        help="The backend used to read imagery during training",
        default="auto",
    )
    parser.add_argument(
        "--val_backend",
        type=str,
        help="The backend used to read imagery during validation",
        default="auto",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="The number of GPUs to use for training",
        default=1,
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        help="The number of nodes to use for training",
        default=1,
    )
    parser.add_argument(
        "--alert_api_key_path",
        type=str,
        help="The path to a file contianing the API key that will be used to send alerts"
    )
    parser.add_argument(
        "--alert_channel_id",
        type=str,
        help="The id of the channel that will be used to post alerts.",
        default="#monitors"
    )
    parser.add_argument(
        "--ortho_stats_file",
        type=str,
        help="The path to the statistics.csv file included with the dataset. Overrides the \"statistics\" key in the dataset paths yaml.",
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--skip_sanity", action="store_true")
    parser.add_argument(
        "--experiment_comment",
        type=str,
        help="Free-text note describing what this run is testing. Recorded in the provenance manifest and TensorBoard.",
        default=None,
    )
    args = parser.parse_args()
    print(args)

    # Capture provenance (git commit/branch/dirty, command line, environment, SLURM job info) before
    # any heavy lifting, so even runs that die during data loading leave a record. Shares the
    # run_metadata vocabulary with the inference/eval stages; stamped as the "training" stage.
    run_metadata = build_training_metadata(
        experiment_comment=args.experiment_comment,
        args_dict=vars(args),
        start_time=current_timestamp(),
        model_path=args.model_checkpoint,
    )
    if local_rank == 0:
        print("Wrote provenance manifest to", write_run_manifest(args.out_path, run_metadata))

    # Load the hyperparameters file
    print("Reading hyperparameters...")
    channel_parameters = parse_hyperparameters(args.channels_hyperparameters_file_path, verbose=False)
    dataset_paths = parse_hyperparameters(args.dataset_paths_file_path, verbose=False)
    data_source_config_parameters = parse_hyperparameters(args.data_source_config_parameters_file_path, verbose=False)

    model_hyperparameters = parse_hyperparameters(args.model_hyperparameters_yaml_path, verbose=False)
    train_datagen_hyperparameters = parse_hyperparameters(args.train_datagen_hyperparameters_yaml_path, verbose=False)
    val_datagen_hyperparameters = parse_hyperparameters(args.val_datagen_hyperparameters_yaml_path, verbose=False)
    uq_hyperparameters = parse_hyperparameters(args.uq_hyperparameters_yaml_path, verbose=False) if args.uq_hyperparameters_yaml_path else None

    # Get the orthomsaics that will be used for validation
    # pylint: disable=duplicate-code
    print("Loading Validation Orthomosaics...")
    validation_orthomosaics = MultisourceOrthomosaicFactory(
        dataset_paths_dict=dataset_paths,
        data_source_config_parameters=data_source_config_parameters,
        model_hyperparameters=model_hyperparameters,
        boundary_folder=None,
        statistics_file_path=args.ortho_stats_file,
        train_validation_test="validation",
        limit=(1 if args.debug else None),
        backend=args.val_backend,
        required_channels=[3,4]
    )
    # pylint: enable=duplicate-code
    print("\tDone")

    # Get the orthomsaics that will be used for training
    # pylint: disable=duplicate-code
    print("Loading Train Orthomosaics...")
    train_orthomosaics = MultisourceOrthomosaicFactory(
        dataset_paths_dict=dataset_paths,
        data_source_config_parameters=data_source_config_parameters,
        model_hyperparameters=model_hyperparameters,
        boundary_folder=None,
        statistics_file_path=args.ortho_stats_file,
        train_validation_test="train",
        limit=(3 if args.debug else None),
        backend=args.train_backend,
        required_channels=[3,4]
    )
    # pylint: enable=duplicate-code
    print("\tMaking sure that there are no validation orthomosaics in the train set")
    train_orthomosaics = remove_orthomosaic_from_list_by_name(train_orthomosaics, [vo.get_name() for vo in validation_orthomosaics], True)
    print("\tDone")

    print("Initializing Validation Dataset...")
    val_dataset = initialize_windowed_dataset(validation_orthomosaics,
                                              channel_parameters,
                                              model_hyperparameters,
                                              val_datagen_hyperparameters,
                                              get_valid_transforms())

    print("Initializing Train Dataset...")
    train_dataset = initialize_windowed_dataset(train_orthomosaics,
                                                channel_parameters,
                                                model_hyperparameters,
                                                train_datagen_hyperparameters,
                                                get_train_transforms())

    print("Initializing Train & Validation Data Module")
    dm = TrainValPredictDataModule(train_dataset = train_dataset,
                                   valid_dataset = val_dataset,
                                   num_workers = 1 if args.debug else args.data_gen_workers,
                                   train_batch_size = model_hyperparameters["training"]["training_parameters"]["batch_size"],
                                   valid_batch_size = model_hyperparameters["validation"]["validation_parameters"]["batch_size"])

    # Initialize Alerting System
    alerter_obj = None
    if args.alert_api_key_path and args.alert_channel_id:
        print("Initializing Alerting System...")
        msg_prefix = str(model_hyperparameters["name"]) + " running on " + str(os.path.split(args.dataset_paths_file_path)[-1].split(".")[0]) + ": "
        alerter_obj = Alerter(args.alert_api_key_path, args.alert_channel_id, message_prefix=msg_prefix)
    else:
        print("Alerting API key and group ID key were not passed. Continuing without alerts...")

    # Initialize the model
    print("Initializing Trainable Model...")
    print("\tTask:", model_hyperparameters["task"], "Model Type:", model_hyperparameters["model_type"])
    model = parse_and_initialize_segmentation_model(channel_parameters, model_hyperparameters, validation_orthomosaics, uq_hyperparameters, alerter_obj)
    print("\tDone")

    # To handle resume training when preempted...
    ckpt_path = None
    if args.restart:
        print("Restarting Training...")
    elif not args.model_checkpoint is None:
        ckpt_path = args.model_checkpoint
        print("Resuming from passed checkpoint:", ckpt_path)
    else:
        ckpt_path = parse_continued_model_checkpoint(args.out_path, model_hyperparameters)
        print("Resuming from found checkpoint:", ckpt_path)

    # Initialize the model and set it to checkpoint on the best observed valid_loss
    print("Initializing Logger...")
    logger = TensorBoardLogger(os.path.join(args.out_path, "tb_logs"), name=str(model.getName()) + "_" + str(model_hyperparameters["task"]))
    if local_rank == 0:
        logger.experiment.add_text("model_hyperparameters", json.dumps(model_hyperparameters), global_step=0)
        logger.experiment.add_text("channel_parameters", json.dumps(channel_parameters), global_step=0)
        logger.experiment.add_text("val_datagen_hyperparameters", json.dumps(val_datagen_hyperparameters), global_step=0)
        logger.experiment.add_text("train_datagen_hyperparameters", json.dumps(train_datagen_hyperparameters), global_step=0)
        logger.experiment.add_text("dataset_paths", json.dumps(dataset_paths), global_step=0)
        logger.experiment.add_text("data_source_config_parameters", json.dumps(data_source_config_parameters), global_step=0)
        logger.experiment.add_text("provenance", json.dumps(run_metadata), global_step=0)
        logger.experiment.add_text("CUDA_is_available", str(torch.cuda.is_available()), global_step=0)
        if torch.cuda.is_available():
            for i in range(0, torch.cuda.device_count()):
                logger.experiment.add_text("CUDA_device_"+str(i), torch.cuda.get_device_name(i), global_step=0)
    checkpoint_callback = model.configure_checkpoint()
    callbacks = [checkpoint_callback]
    early_stopping_callback = model.configure_early_stopping()
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)

    print("Initializing Training Parameters...")
    # Parse the gradient clipping value if one was passed
    gradient_clip_val = None
    if "gradient_clip_val" in model_hyperparameters["training"]["training_parameters"].keys():
        gradient_clip_val = model_hyperparameters["training"]["training_parameters"]["gradient_clip_val"]

    trainer_args = {
        "gradient_clip_val":gradient_clip_val,
        "max_epochs":model_hyperparameters["training"]["training_parameters"]["max_epochs"],
        "callbacks":callbacks,
        "default_root_dir":args.out_path,
        "logger":logger,
        "accelerator":args.accelerator,
        "profiler":None,
        "accumulate_grad_batches":model_hyperparameters["training"]["training_parameters"]["grad_accumulation"],
        "num_sanity_val_steps":0 if args.skip_sanity else -1
    }

    plugins=[]
    if "grad_scaler_floor" in model_hyperparameters["training"]["training_parameters"].keys():
        grad_scaler_floor = model_hyperparameters["training"]["training_parameters"]["grad_scaler_floor"]
        # MixedPrecision wants a torch device string, not the Lightning "--accelerator" spelling the
        # Trainer takes above, so translate it (gpu -> cuda) rather than hardcoding "cuda". See #215.
        mixed_precision_device = resolve_torch_device(args.accelerator)
        plugins.append(MixedPrecision(precision=args.precision, device=mixed_precision_device, scaler=ClampedGradScaler(min_scale=grad_scaler_floor)))
        trainer_args = trainer_args | {"plugins": plugins}
    else:
        trainer_args = trainer_args | {"precision": args.precision}

    # If multi-GPU training is invoked, then do that
    if args.num_gpus > 1:
        print("Enabling MultiGPU Training...")
        multi_gpu_args = {"strategy":"ddp", "devices":args.num_gpus, "num_nodes":args.num_nodes}
        trainer_args = trainer_args | multi_gpu_args

    # Initialize the trainer with all the parsed args
    trainer = Trainer(**trainer_args)

    # Fit the model.
    if local_rank == 0:
        model.sendAlert("Training started!")
    try:
        print("Training...")
        trainer.fit(model, dm, ckpt_path=ckpt_path)
        if local_rank == 0:
            model.sendAlert("Training finished!")
    except Exception as e:
        model.sendAlert("Training failed!\n\t" + str(e.__class__.__name__) + " occured...\n\tCheck logs for more information.")
        raise e
