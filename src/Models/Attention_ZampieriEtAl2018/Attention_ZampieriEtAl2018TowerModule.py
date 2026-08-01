from modeling.Models.Attention_ZampieriEtAl2018.Attention_ZampieriEtAl2018 import Attention_ZampieriEtAl2018
from modeling.Models.ZampieriEtAl2018.ZampieriEtAl2018TowerModule import ZampieriEtAl2018TowerModule


# Tower wrapper so the model registry can build the attention-augmented Zampieri
# alignment model (same displacement-field output as ZampieriEtAl2018, with added
# cross/self-attention blocks). Used by the BDAADJ task.
class Attention_ZampieriEtAl2018TowerModule(ZampieriEtAl2018TowerModule):
    def _alignment_model_class(self):
        return Attention_ZampieriEtAl2018
