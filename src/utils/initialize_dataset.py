from modeling.utils.data_augmentations import get_normalize_transform, get_tensor_transform

from modeling.adaptors.WindowedDatasetAdaptor import WindowedDatasetAdaptor
from modeling.datasets.WindowedDataset import WindowedDataset
from modeling.DataMap import Labels2IdxMap

from modeling.Models.model_registry import (
    LOCATIONSTRATEGY2MODULEMAPPING,
    MASKINGSTRATEGY2MODULEMAPPING,
    KEYPOINTSTRATEGY2MODULEMAPPING,
    SAMPLEANNOTATORTRATEGY2MODULEMPAPPING,
    PRESENTATIONTRATEGY2MODULEMAPPING,
)

# CHANGE sample windows only ever anchor on post-disaster imagery; enforced in
# initialize_windowed_dataset (not a datagen knob) so the pre-event confound can't creep back in.
CHANGE_ANCHOR_EVENT_PHASE = "POST"

def initialize_windowed_dataset(orthomosaics, channel_parameters, model_hyperparameters, datagen_hyperparameters, augmentation_transform):

    # Parse the dataset label map from the channel parameters
    input_dataset_label_map = Labels2IdxMap(
        channel_parameters["channel_maps"]["input_dataset_class_2_idx_map"],
        channel_parameters["channel_maps"]["background_class_idx"],
    )

    # Parse and initialize the masking strategy that should be used by the dataset
    print("\tInitializing Masking Strategy")
    masking_strategy_args = {}
    if "masking_strategy_parameters" in datagen_hyperparameters.keys():
        masking_strategy_args = datagen_hyperparameters["masking_strategy_parameters"]
    masking_strat = MASKINGSTRATEGY2MODULEMAPPING[model_hyperparameters["task"]](**masking_strategy_args)

    # Parse and initialize the keypoint augmentation strategy that should be used by the dataset
    print("\tInitializing Keypoint Strategy")
    keypoint_strat = KEYPOINTSTRATEGY2MODULEMAPPING[model_hyperparameters["task"]]()

    # Initialize the dataset based on task...
    print("\tInitializing Sample Location Generation Strategy")
    presentation_strategy_args = {}
    if "presentation_strategy_parameters" in datagen_hyperparameters.keys():
        presentation_strategy_args.update(datagen_hyperparameters["presentation_strategy_parameters"])

    # Initialize the location selection strategy that should be used to select image frames
    loc_pres_strat = PRESENTATIONTRATEGY2MODULEMAPPING[datagen_hyperparameters["presentation_strategy"]](**presentation_strategy_args)

    # The CHANGE annotator labels each building by comparing it against that building's views in
    # every other orthomosaic, so it is constructed over the full orthomosaic list. The sample
    # windows themselves are anchored on a single collection source when the datagen yaml sets
    # anchor_source (e.g. "Satellite"), and are ALWAYS restricted to post-event imagery: a pre-event
    # satellite tile carries no damage signal, so anchoring on it would train the model on priors
    # rather than observable change. The post-event restriction is hardwired (CHANGE_ANCHOR_EVENT_PHASE)
    # rather than a datagen knob so no datagen can reintroduce it; sUAS/crewed orthos are all
    # post-event already, so it only ever drops pre-event satellite anchors. Other tasks sample every
    # orthomosaic.
    sample_orthomosaics = orthomosaics
    if model_hyperparameters["task"] == "CHANGE":
        annotator = SAMPLEANNOTATORTRATEGY2MODULEMPAPPING[model_hyperparameters["task"]](orthomosaics, **datagen_hyperparameters["annotator_parameters"])
        anchor_source = datagen_hyperparameters.get("anchor_source")
        if anchor_source is not None:
            sample_orthomosaics = [o for o in sample_orthomosaics if o.get_source_used_for_collection() == anchor_source]
        sample_orthomosaics = [o for o in sample_orthomosaics if o.get_event_phase() == CHANGE_ANCHOR_EVENT_PHASE]
        # Sample windows can only come from anchor orthos that actually carry buildings (the centered
        # strategy walks get_buildings(adjusted=...)), so surface those counts here -- an anchor with
        # orthos but zero buildings means the annotation files are missing or empty for this stripe,
        # which otherwise only shows up as an empty sample pool later.
        unadjusted_buildings = sum(len(o.get_buildings() or []) for o in sample_orthomosaics)
        adjusted_buildings = sum(len(o.get_buildings(adjusted=True) or []) for o in sample_orthomosaics)
        print(f"\tCHANGE anchors (source={anchor_source}, post-event): matched {len(sample_orthomosaics)} of "
              f"{len(orthomosaics)} orthomosaics ({unadjusted_buildings} buildings, {adjusted_buildings} adjusted)")
        if len(sample_orthomosaics) == 0:
            observed_sources = sorted({str(o.get_source_used_for_collection()) for o in orthomosaics})
            observed_phases = sorted({str(o.get_event_phase()) for o in orthomosaics})
            raise ValueError(f"CHANGE anchor_source '{anchor_source}' + post-event filter matched no orthomosaics; "
                             f"observed sources: {observed_sources}, phases: {observed_phases}")
    else:
        annotator = SAMPLEANNOTATORTRATEGY2MODULEMPAPPING[model_hyperparameters["task"]](**datagen_hyperparameters["annotator_parameters"])

    sample_location_strategy_args = {
        "annotator":annotator,
        "sample_location_presentation_strategy":loc_pres_strat,
        "orthomosaics":sample_orthomosaics
    }

    if "location_parameters" in datagen_hyperparameters.keys():
        sample_location_strategy_args.update(datagen_hyperparameters["location_parameters"])
    location_strat = LOCATIONSTRATEGY2MODULEMAPPING[datagen_hyperparameters["location_strategy"]](**sample_location_strategy_args)

    # Initialize the adaptor that will consume all of the strategies we have initialized
    print("\tInitializing Dataset Adaptor. This may take a moment as it can involve generating samples to send to the model...")
    dataset_adaptor_args = {
        "orthomosaics": sample_orthomosaics,
        "label_map": input_dataset_label_map,
        "sample_location_generation_strategy": location_strat,
        "keypoint_conversion_strategy": keypoint_strat,
    }

    dataset_adaptor_args.update(datagen_hyperparameters["dataset_adaptor_parameters"])
    adaptor = WindowedDatasetAdaptor(**dataset_adaptor_args)

    # Initialize the dataset with all the transforms that we care about
    print("\tInitializing Dataset")
    dataset = WindowedDataset(adaptor,
                              masking_strat,
                              augmentation_transform,
                              get_normalize_transform(),
                              get_tensor_transform(),
                              model_hyperparameters["input"]["normalized_inputs"])

    print("\tDone")

    return dataset
