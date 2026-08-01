Y_HAT_SEGMENTATION_UNMASKED = "y_hat_segmentation_unmasked"
Y_HAT_SEGMENTATION_MASKED = "y_hat_segmentation_masked"
Y_HAT_SEGMENTATION_LOGITS = "y_hat_segmentation_logits"
DISPLACEMENT_FIELD = "displacement_field"
DO_SOFTMAX = "do_softmax"
UQ_PREDICTION = "uq_prediction"
UQ_COMBINED = "uq_combined"
MU = "mu"
SIGMA = "sigma"

CHANNEL_INPUT = "channels"
MASK = "mask"
GSD = "gsd"
BUILDINGS = "buildings"
TIMESTAMP = "timestamp"

class ModelDatum:
    def __init__(self):
        self.__fields = {}
    def __getitem__(self, field):
        return self.__fields[field]
    def setField(self, field, value):
        self.__fields[field] = value
    def contains(self, field):
        return field in self.__fields

class ModelInput(ModelDatum):
    pass

class ModelOutput(ModelDatum):
    pass
