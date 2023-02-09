#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Dict
from mobilevitv2.mobilevit_utils.math_utils import make_divisible, bound_fn


def _get_configuration(width_multiplier) -> Dict:

    ffn_multiplier = (
        2  # bound_fn(min_val=2.0, max_val=4.0, value=2.0 * width_multiplier)
    )
    mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))
    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    classifier_dropout = 0.1

    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
            'conv_norm': {'name': 'nn.BatchNorm2d', 'param': {}},
            'conv_act': {'name': 'Swish', 'param': {}},
        },
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            'conv_norm': {'name': 'nn.BatchNorm2d', 'param': {}},
            'conv_act': {'name': 'Swish', 'param': {}},
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            'conv_norm': {'name': 'nn.BatchNorm2d', 'param': {}},
            'conv_act': {'name': 'Swish', 'param': {}},
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "block_type": "mobilevit",
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "n_attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            'conv_norm': {'name': 'nn.BatchNorm2d', 'param': {}},
            'conv_ksize': 3,
            'conv_act': {'name': 'Swish', 'param': {}},
            'attn_norm_layer': {'name': 'LayerNorm2DNCHW', 'param': {}},
            'attn_act': {'name': 'Swish', 'param': {}},
            'dropout': 0.0,
            'attn_dropout': 0.0,
            'ffn_dropout': 0.0
        },
        "layer4": {  # 14x14
            "block_type": "mobilevit",
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "n_attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            'conv_norm': {'name': 'nn.BatchNorm2d', 'param': {}},
            'conv_ksize': 3,
            'conv_act': {'name': 'Swish', 'param': {}},
            'attn_norm_layer': {'name': 'LayerNorm2DNCHW', 'param': {}},
            'attn_act': {'name': 'Swish', 'param': {}},
            'dropout': 0.0,
            'attn_dropout': 0.0,
            'ffn_dropout': 0.0
        },
        "layer5": {  # 7x7
            "block_type": "mobilevit",
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "n_attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            'conv_norm': {'name': 'nn.BatchNorm2d', 'param': {}},
            'conv_ksize': 3,
            'conv_act': {'name': 'Swish', 'param': {}},
            'attn_norm_layer': {'name': 'LayerNorm2DNCHW', 'param': {}},
            'attn_act': {'name': 'Swish', 'param': {}},
            'dropout': 0.0,
            'attn_dropout': 0.0,
            'ffn_dropout': 0.0,
        },
        "last_layer_exp_factor": 4,
    }

    return config


def get_mobilevit_v2_w0_5():     # noqa
    return _get_configuration(0.5)

def get_mobilevit_v2_w0_75():    # noqa
    return _get_configuration(0.75)

def get_mobilevit_v2_w1_0():    # noqa
    return _get_configuration(1)

def get_mobilevit_v2_w1_25():    # noqa
    return _get_configuration(1.25)

def get_mobilevit_v2_w1_5():    # noqa
    return _get_configuration(1.5)

def get_mobilevit_v2_w1_75():    # noqa
    return _get_configuration(1.75)

def get_mobilevit_v2_w2_0():    # noqa
    return _get_configuration(2)
