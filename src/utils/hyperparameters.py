import yaml

def parse_hyperparameters(file_path, verbose=True):
    with open(file_path) as stream:
        try:
            hyperparameters = yaml.safe_load(stream)
            if verbose:
                print(hyperparameters)
            return hyperparameters
        except yaml.YAMLError as exc:
            print("Encountered error parsing hyperparameters...")
            print(exc)
            return {}

def add_hyperparameters_files_to_parse_args(arg_parser,
                                            add_dataset_paths_file_path=False,
                                            add_data_source_config_parameters_file_path=False,
                                            add_channels_hyperparameters_file_path=False,
                                            add_train_datagen_hyperparameters_yaml_path=False,
                                            add_val_datagen_hyperparameters_yaml_path=False,
                                            add_infer_datagen_hyperparameters_yaml_path=False,
                                            add_model_hyperparameters_yaml_path=False,
                                            add_uq_hyperparameters_yaml_path=False):
    if add_dataset_paths_file_path:
        arg_parser.add_argument(
            "--dataset_paths_file_path",
            type=str,
            help="The path to the yaml file containing the file paths for all of the different data sources that could be used.",
        )
    if add_data_source_config_parameters_file_path:
        arg_parser.add_argument(
            "--data_source_config_parameters_file_path",
            type=str,
            help="The path to the yaml file containing the configuration for the data that should be used for training.",
        )
    if add_channels_hyperparameters_file_path:
        arg_parser.add_argument(
            "--channels_hyperparameters_file_path",
            type=str,
            help="The path to the file containing the information for mapping channels to inputs and outputs",
        )
    if add_train_datagen_hyperparameters_yaml_path:
        arg_parser.add_argument(
            "--train_datagen_hyperparameters_yaml_path",
            type=str,
            help="The path to the file containing the parameters that describe how train samples should be generated",
        )
    if add_val_datagen_hyperparameters_yaml_path:
        arg_parser.add_argument(
            "--val_datagen_hyperparameters_yaml_path",
            type=str,
            help="The path to the file containing the parameters that describe how validation samples should be generated",
        )
    if add_infer_datagen_hyperparameters_yaml_path:
        arg_parser.add_argument(
            "--infer_datagen_hyperparameters_yaml_path",
            type=str,
            help="The path to the file containing the parameters that describe how train samples should be generated",
        )
    if add_model_hyperparameters_yaml_path:
        arg_parser.add_argument(
            "--model_hyperparameters_yaml_path",
            type=str,
            help="Path to the hyperparameters yaml file path.",
        )
    if add_uq_hyperparameters_yaml_path:
        arg_parser.add_argument(
            "--uq_hyperparameters_yaml_path",
            type=str,
            help="Path to the hyperparameters yaml file path.",
        )
    return arg_parser
