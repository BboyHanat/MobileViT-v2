from typing import Optional, Tuple, Union

import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F  # noqa
from mobilevitv2.mobilevit_utils.activation import Swish
from mobilevitv2.mobilevit_utils.math_utils import make_divisible
from mobilevitv2.mobilevit_utils.mobilevit_transformer import LinearAttnFFN
from mobilevitv2.mobilevit_utils.norm import LayerNorm, LayerNorm2DNCHW


class InvertedResidual(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 stride=1,
                 norm_layer: Optional[Union[nn.Module, dict]] = nn.BatchNorm2d,
                 conv_act: Optional[Union[nn.Module, dict]] = nn.ReLU,
                 inplace=False,
                 skip_connection=True):
        super(InvertedResidual, self).__init__()

        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        block = []
        if expand_ratio != 1:
            block.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, bias=False),
                norm_layer(hidden_dim) if isinstance(norm_layer, nn.Module) else eval(
                    norm_layer['name'])(hidden_dim),
                conv_act(inplace=inplace) if isinstance(conv_act, nn.Module) else eval(
                    conv_act['name'])(**conv_act['param'])])

        block.extend([
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                      groups=hidden_dim, stride=stride, padding=1, bias=False),
            norm_layer(hidden_dim) if isinstance(norm_layer, nn.Module) else eval(
                norm_layer['name'])(hidden_dim),
            conv_act(inplace=inplace) if isinstance(conv_act, nn.Module) else eval(
                conv_act['name'])(**conv_act['param']),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=False),
            norm_layer(hidden_dim) if isinstance(norm_layer, nn.Module) else eval(
                norm_layer['name'])(out_channels)])

        self.block = nn.Sequential(*block)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels and skip_connection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlockV2(nn.Module):

    def __init__(self,
                 in_channels: int,
                 attn_unit_dim: int,
                 ffn_multiplier: float = 2.0,
                 n_attn_blocks: int = 2,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.0,
                 ffn_dropout: Optional[float] = 0.0,
                 patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8,
                 conv_norm: Optional[Union[nn.Module, dict]] = nn.BatchNorm2d,
                 conv_act: Optional[Union[nn.Module, dict]] = nn.ReLU,
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1,
                 attn_norm_layer: Optional[Union[nn.Module, dict]] = nn.LayerNorm,
                 attn_act: Optional[Union[nn.Module, dict]] = nn.ReLU,
                 ):
        super(MobileViTBlockV2, self).__init__()

        local_rep = [
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=conv_ksize,
                      stride=1, dilation=dilation, groups=in_channels, padding=1, bias=False),
            conv_norm(in_channels) if isinstance(conv_norm, nn.Module) else eval(
                conv_norm['name'])(in_channels),
            conv_act(inplace=False) if isinstance(conv_act, nn.Module) else eval(
                conv_act['name'])(**conv_act['param']),
            nn.Conv2d(in_channels=in_channels, out_channels=attn_unit_dim, kernel_size=1,
                      stride=1, bias=False)
        ]

        # local representation
        self.local_rep = nn.Sequential(*local_rep)

        # global representation
        self.global_rep, d_model = self._build_attention_layer(d_model=attn_unit_dim,
                                                               ffn_mult=ffn_multiplier,
                                                               n_layers=n_attn_blocks,
                                                               dropout=dropout,
                                                               attn_dropout=attn_dropout,
                                                               ffn_dropout=ffn_dropout,
                                                               attn_norm_layer=attn_norm_layer,
                                                               attn_act=attn_act)

        self.conv_proj = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=in_channels,
                                                 kernel_size=1, stride=1, bias=False),
                                       conv_norm(in_channels) if isinstance(conv_norm, nn.Module) else eval(
                                           conv_norm['name'])(in_channels),
                                       )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

    def _folding(self, patches, output_size: Tuple[int, int]):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(patches, output_size=output_size,
                             kernel_size=(self.patch_h, self.patch_w),
                             stride=(self.patch_h, self.patch_w))
        return feature_map

    def _unfolding(self, x):
        batch_size, in_channels, img_h, img_w = x.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(x, kernel_size=(self.patch_h, self.patch_w), stride=(self.patch_h, self.patch_w))
        patches = patches.reshape(batch_size, in_channels, self.patch_area, -1)

        return patches, (img_h, img_w)

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)
        return x

    @staticmethod
    def _build_attention_layer(d_model: int,
                               ffn_mult: float,
                               n_layers: int,
                               dropout: float,
                               attn_dropout: float,
                               ffn_dropout: float,
                               attn_norm_layer: Optional[Union[nn.Module, dict]] = nn.LayerNorm,
                               attn_act: Optional[Union[nn.Module, dict]] = nn.ReLU,
                               ):
        ffn_dims = [ffn_mult * d_model] * n_layers
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]
        global_rep = [
            LinearAttnFFN(embed_dim=d_model,
                          ffn_latent_dim=ffn_dims[block_idx],
                          dropout=dropout,
                          attn_dropout=attn_dropout,
                          ffn_dropout=ffn_dropout,
                          attn_norm_layer=attn_norm_layer,
                          attn_act=attn_act) for block_idx in range(n_layers)]
        global_rep.append(
            attn_norm_layer(d_model) if isinstance(attn_norm_layer, nn.Module) else eval(
                attn_norm_layer['name'])(d_model)
        )
        return nn.Sequential(*global_rep), d_model

    def forward(self, x: torch.Tensor):
        x = self.resize_input_if_needed(x)
        fm = self.local_rep(x)
        patches, output_size = self._unfolding(fm)
        patches = self.global_rep(patches)
        fm = self._folding(patches, output_size)
        fm = self.conv_proj(fm)
        return fm
