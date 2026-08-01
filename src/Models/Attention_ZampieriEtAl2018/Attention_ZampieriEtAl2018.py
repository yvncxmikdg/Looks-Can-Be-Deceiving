from modeling.Models.ZampieriEtAl2018.ZampieriEtAl2018 import ZampieriEtAl2018, ZampieriEtAl2018_block

from .CrossAttention import CrossAttention
from .MultiAttention import MultiHeadSelfAttention2D


class Attention_ZampieriEtAl2018(ZampieriEtAl2018):
    def _build_block(self):
        return Attention_ZampieriEtAl2018_block(self.mask_channel)


class Attention_ZampieriEtAl2018_block(ZampieriEtAl2018_block):
    def __init__(self, mask_channel=3):
        super().__init__(mask_channel)

        self.cross_attention_1 = CrossAttention(
            in_channels=32,
            reduction=16,
            downsample=True,
            dropout=0.2,
            drop_path_rate=0.1,
            use_positional_encoding=True
        )

        self.attn_dec_1 = MultiHeadSelfAttention2D(
            in_channels=128,
            num_heads=4,
            dropout=0.2,
            drop_path_rate=0.1,
            use_positional_encoding=True
        )

        self.bottleneck_attn = MultiHeadSelfAttention2D(
            in_channels=64,
            num_heads=4,
            dropout=0.2,
            drop_path_rate=0.1,
            use_positional_encoding=True
        )

    def _attend_visual(self, a6, b6):
        return self.cross_attention_1(a6, b6)

    def _attend_bottleneck(self, down_3_output):
        return self.bottleneck_attn(down_3_output)

    def _attend_decoder(self, down_2_complete_intermediate):
        return self.attn_dec_1(down_2_complete_intermediate)
