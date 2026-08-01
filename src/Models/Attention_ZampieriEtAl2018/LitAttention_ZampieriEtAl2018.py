import pytorch_lightning as L

from modeling.Models.Attention_ZampieriEtAl2018.Attention_ZampieriEtAl2018 import Attention_ZampieriEtAl2018

class LitAttention_ZampieriEtAl2018(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()
        self._hyperparamters = hyperparameters
        self._model = (
            Attention_ZampieriEtAl2018(
                len(input_channel_map),
                len(output_label_map),
                output_channel_background_index=output_label_map.getBackgroundClassIdx(),
            )
            .cpu()
            .to(self._device)
        )

    def get_model(self):
        return self._model
    def get_hyperparamters(self):
        return self._hyperparamters
