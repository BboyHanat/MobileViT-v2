from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F  # noqa
from mobilevitv2.mobilevit_utils.activation import Swish
from mobilevitv2.mobilevit_utils.norm import LayerNorm, LayerNorm2DNCHW


class LinearAttnFFN(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 ffn_latent_dim: int,
                 dropout: Optional[float] = 0.1,
                 attn_dropout: Optional[float] = 0.0,
                 ffn_dropout: Optional[float] = 0.0,
                 attn_act: Optional[Union[nn.Module, dict]] = nn.ReLU,
                 attn_norm_layer: Optional[Union[nn.Module, dict]] = nn.LayerNorm,
                 inplace: Optional[bool] = False):
        super(LinearAttnFFN, self).__init__()

        attn_unit = LinearSelfAttention(embed_dim=embed_dim,
                                        attn_dropout=attn_dropout,
                                        inplace=inplace,
                                        bias=True)
        self.pre_norm_attn = nn.Sequential(
            attn_norm_layer(embed_dim) if isinstance(attn_norm_layer, nn.Module) else eval(
                attn_norm_layer['name'])(embed_dim),
            attn_unit,
            nn.Dropout(dropout, inplace=inplace))

        self.pre_norm_ffn = nn.Sequential(
            attn_norm_layer(embed_dim) if isinstance(attn_norm_layer, nn.Module) else eval(attn_norm_layer['name'])(
                embed_dim),
            nn.Conv2d(in_channels=embed_dim, out_channels=ffn_latent_dim, kernel_size=1, stride=1, bias=True),
            attn_act() if isinstance(attn_act, nn.Module) else eval(attn_act['name'])(**attn_act['param']),
            nn.Dropout(ffn_dropout, inplace=inplace),
            nn.Conv2d(in_channels=ffn_latent_dim, out_channels=embed_dim, kernel_size=1, stride=1, bias=True),
            nn.Dropout(dropout, inplace=inplace)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.pre_norm_attn(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class LinearSelfAttention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            inplace: Optional[bool] = False,
            bias: Optional[bool] = True):
        super().__init__()

        self.inplace = inplace
        self.qkv_proj = nn.Conv2d(in_channels=embed_dim, out_channels=1 + (2 * embed_dim), bias=bias, kernel_size=1)
        self.attn_dropout = nn.Dropout(p=attn_dropout, inplace=inplace)
        self.out_proj = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, bias=bias, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(qkv,
                                        split_size_or_sections=[1, self.embed_dim, self.embed_dim],
                                        dim=1)
        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)
        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value, inplace=self.inplace) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out
