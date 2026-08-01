from modeling.Models.TaskModuleBDA import TaskModuleBDA
from modeling.Models.TaskModuleRDA import TaskModuleRDA
from modeling.Models.TaskModuleBDAADJ import TaskModuleBDAADJ
from modeling.Models.TaskModuleCHANGE import TaskModuleCHANGE

from modeling.utils.sample_presentation import (
    IndexSampleLocationPresentationStrategy,
    MostRecentlyObservedSampleLocationPresentationStrategy,
    WeightedSampleLocationPresentationStrategy,
)

from modeling.utils.sample_location_generator import (
    RandomSampleLocationGenerationStrategy,
    CenteredBuildingSampleStrategy,
    GridSampleStrategy,
    BDASampleAnnotator,
    RDASampleAnnotator,
    BuildingChangeSampleAnnotator,
)
from modeling.utils.mask_generation import MaskingStrategyBDA, MaskingStrategyRDA

from modeling.utils.data_augmentations import (
    KeyPointConversionStrategyRDA,
    KeyPointConversionStrategyBDA,
)

from modeling.Models.Model import EncoderDecoderModule, MaskedEncoderDecoderModule

from modeling.Models.MaskedUNet.UNetTowerModule import UNetTowerModule, MaskedUNetTowerModule
from modeling.Models.ZampieriEtAl2018.LitZampieriEtAl2018 import LitZampieriEtAl2018
from modeling.Models.ZampieriEtAl2018.ZampieriEtAl2018TowerModule import ZampieriEtAl2018TowerModule
from modeling.Models.Attention_ZampieriEtAl2018.Attention_ZampieriEtAl2018TowerModule import Attention_ZampieriEtAl2018TowerModule
from modeling.Models.Segmenter.Segmenter import Segmenter
from modeling.Models.UperNet.UperNet import UperNet, MaskedUperNet
from modeling.Models.Mask2Former.Mask2Former import Mask2Former
from modeling.Models.Baselines.RandomBaselineModel import RandomBaselineModel, MaskedRandomBaselineModel
from modeling.Models.Baselines.RandomBaselineBDAADJ import RandomBaselineBDAADJTowerModule
from modeling.Models.PSPNet.PSPNetResNet import PSPNetResNet, MaskedPSPNetResNet
from modeling.Models.DeepLabV3Plus.DeepLabV3PlusResNet import DeepLabV3PlusResNet, MaskedDeepLabV3PlusResNet
from modeling.Models.Backbones.ViTBackbone import ViTBackboneEncoder, ScaleMAEBackboneEncoder, SatMAEBackboneEncoder

# TODO: Need to think about how this transfers for train and inference
# TODO: We may want to further extend this to different datasets
LOCATIONSTRATEGY2MODULEMAPPING = {
    "random": RandomSampleLocationGenerationStrategy,
    "centered": CenteredBuildingSampleStrategy,
    "grid": GridSampleStrategy,
}

PRESENTATIONTRATEGY2MODULEMAPPING = {
    "indexed": IndexSampleLocationPresentationStrategy,
    "most_recently_observed": MostRecentlyObservedSampleLocationPresentationStrategy,
    "weighted": WeightedSampleLocationPresentationStrategy,
}

MASKINGSTRATEGY2MODULEMAPPING = {
    "BDA": MaskingStrategyBDA,
    "BDAADJ": MaskingStrategyBDA,
    "CHANGE": MaskingStrategyBDA,
    "RDA": MaskingStrategyRDA,
    "RDAADJ": MaskingStrategyRDA,
}

KEYPOINTSTRATEGY2MODULEMAPPING = {
    "BDA": KeyPointConversionStrategyBDA,
    "BDAADJ": KeyPointConversionStrategyBDA,
    "CHANGE": KeyPointConversionStrategyBDA,
    "RDA": KeyPointConversionStrategyRDA,
    "RDAADJ": None,
}

SAMPLEANNOTATORTRATEGY2MODULEMPAPPING = {
    "BDA": BDASampleAnnotator,
    "BDAADJ": BDASampleAnnotator,
    "CHANGE": BuildingChangeSampleAnnotator,
    "RDA": RDASampleAnnotator,
    "RDAADJ": None,
}

STR2TASKMODELCLASS = {
    "BDA": TaskModuleBDA,
    "CHANGE": TaskModuleCHANGE,
    "RDA": TaskModuleRDA,
    "BDAADJ": TaskModuleBDAADJ,
}

STR2MODELCLASS = {
    "ZampieriEtAl2018": LitZampieriEtAl2018,
}

STR2MODELENCODERCLASS = {
    "ViT": ViTBackboneEncoder,
    "ScaleMAE": ScaleMAEBackboneEncoder,
    "SatMAE": SatMAEBackboneEncoder,
    "ConvMAE": ViTBackboneEncoder,
    "ResNet": None,
    "Convext": None,
    "Swin": None,
    "Dino": ViTBackboneEncoder,
}

STR2DECODERCLASS = {
    "UperNet": UperNet,
    "Segmenter": Segmenter,
    "Mask2Former": Mask2Former,
}

STR2MODELTOWERCLASS = {
    "PSPNetResNet": PSPNetResNet,
    "DeepLabV3PlusResNet": DeepLabV3PlusResNet,
    "UNetTowerModule": UNetTowerModule,
    "RandomBaseline": RandomBaselineModel,
    "RandomBaselineBDAADJ": RandomBaselineBDAADJTowerModule,
    "ZampieriEtAl2018": ZampieriEtAl2018TowerModule,
    "Attention_ZampieriEtAl2018": Attention_ZampieriEtAl2018TowerModule
}

STR2MASKEDMODELTOWERCLASS = {
    "MaskedPSPNetResNet": MaskedPSPNetResNet,
    "MaskedDeepLabV3PlusResNet": MaskedDeepLabV3PlusResNet,
    "MaskedUNetTowerModule": MaskedUNetTowerModule,
    "MaskedRandomBaseline": MaskedRandomBaselineModel,
    "MaskedUperNet": MaskedUperNet
}

MODELTYPE2MODELDICT = {
    "Tower": STR2MODELTOWERCLASS,
    "MaskedTower": STR2MASKEDMODELTOWERCLASS,
    "EncoderDecoder": {"Encoder": STR2MODELENCODERCLASS, "Decoder": STR2DECODERCLASS},
    "MaskedEncoderDecoder": {
        "Encoder": STR2MODELENCODERCLASS,
        "Decoder": STR2DECODERCLASS,
    },
}


# pylint: disable-next=too-many-branches
def parse_and_initialize_segmentation_model(
    channel_parameters,
    model_hyperparameters,
    validation_orthomosaics,
    uq_hyperparameters=None,
    alerter=None,
    model_path=None,
    strict_load=True,
):
    # UQ is opt-in: only a config that explicitly sets "uq: true" turns the UQ pathway on. The
    # normalized value (a dict or None) gates both the task module's UQ loss and the model head.
    if not (uq_hyperparameters and uq_hyperparameters.get("uq", False)):
        uq_hyperparameters = None

    task_model = STR2TASKMODELCLASS[model_hyperparameters["task"]](
        channel_parameters=channel_parameters,
        model_hyperparameters=model_hyperparameters,
        val_orthomosaics=validation_orthomosaics,
        uq_hyperparameters=uq_hyperparameters,
        alerter=alerter,
    )
    model_type = model_hyperparameters["model_type"]
    if "EncoderDecoder" in model_type:
        encoder = MODELTYPE2MODELDICT[model_type]["Encoder"][
            model_hyperparameters["encoder_type"]
        ]()
        decoder = MODELTYPE2MODELDICT[model_type]["Decoder"][
            model_hyperparameters["decoder_type"]
        ]()
        if model_type == "EncoderDecoder":
            if model_path is None:
                encoder_decoder_task_model = EncoderDecoderModule(
                    encoder,
                    decoder,
                    model_hyperparameters,
                    task_model.input_channel_map,
                    task_model.output_label_map,
                )
            else:
                print("Loading model from provided checkpoint (This is for Inference)")
                encoder_decoder_task_model = EncoderDecoderModule.load_from_checkpoint(
                    checkpoint_path=model_path,
                    map_location="cpu",
                    strict=strict_load,
                    encoder=encoder,
                    decoder=decoder,
                    hyperparameters=model_hyperparameters,
                    input_channel_map=task_model.input_channel_map,
                    output_label_map=task_model.output_label_map,
                )
                encoder_decoder_task_model.set_model(encoder_decoder_task_model)
            task_model.initialize_model(encoder_decoder_task_model.get_model())
        elif model_type == "MaskedEncoderDecoder":
            if model_path is None:
                encoder_decoder_task_model = MaskedEncoderDecoderModule(
                    encoder,
                    decoder,
                    model_hyperparameters,
                    task_model.input_channel_map,
                    task_model.output_label_map,
                )
            else:
                print("Loading model from provided checkpoint (This is for Inference)")
                encoder_decoder_task_model = (
                    MaskedEncoderDecoderModule.load_from_checkpoint(
                        checkpoint_path=model_path,
                        map_location="cpu",
                        strict=strict_load,
                        encoder=encoder,
                        decoder=decoder,
                        hyperparameters=model_hyperparameters,
                        input_channel_map=task_model.input_channel_map,
                        output_label_map=task_model.output_label_map,
                    )
                )

            task_model.initialize_model(encoder_decoder_task_model.get_model())
    elif "Tower" in model_type:
        model_name = model_hyperparameters["model"]
        model_class = MODELTYPE2MODELDICT[model_type][model_name]
        if model_path is None:
            m = model_class(
                model_hyperparameters,
                task_model.input_channel_map,
                task_model.output_label_map,
            ).get_model()
        else:
            print("Loading model from provided checkpoint (This is for Inference)")
            m = model_class.load_from_checkpoint(
                checkpoint_path=model_path,
                map_location="cpu",
                strict=strict_load,
                hyperparameters=model_hyperparameters,
                input_channel_map=task_model.input_channel_map,
                output_label_map=task_model.output_label_map,
            ).get_model()

        task_model.initialize_model(m)

    else:
        raise ValueError(
            'Unknown Model Type "'
            + str(model_type)
            + '" available options are...'
            + str(list(MODELTYPE2MODELDICT.keys()))
        )

    if uq_hyperparameters is not None:
        if not hasattr(task_model.model, "initialize_uq"):
            raise ValueError(
                'UQ hyperparameters were provided but model "'
                + str(model_hyperparameters["model_type"])
                + '" does not expose a SegmentationModelOutputModule to attach the UQ head to.'
            )
        task_model.model.initialize_uq(
            uq_hyperparameters, len(task_model.output_label_map)
        )

    return task_model
