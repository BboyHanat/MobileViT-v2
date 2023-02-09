import torch
from torch import nn
from mobilevitv2.mobilevit_v2_cfg import *
from mobilevitv2.mobilevit_utils.activation import *
from mobilevitv2.mobilevit_utils.mobile_vit_blocks import InvertedResidual, MobileViTBlockV2


class MobileViTv2(nn.Module):

    def __init__(self, cfg, classifier_num=1000):
        super(MobileViTv2, self).__init__()
        self.inplace = False
        image_channels = cfg["layer0"]["img_channels"]
        out_channels = cfg["layer0"]["out_channels"]
        conv_norm = cfg["layer0"]["conv_norm"]
        conv_act = cfg["layer0"]["conv_act"]
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1, bias=False),
            eval(conv_norm['name'])(out_channels),
            eval(conv_act['name'])(**conv_act['param']))

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(cfg=cfg['layer1'], input_channel=in_channels)
        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(cfg=cfg['layer2'], input_channel=in_channels)
        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(cfg=cfg['layer3'], input_channel=in_channels)
        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(cfg=cfg['layer4'], input_channel=in_channels)
        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(cfg=cfg['layer5'], input_channel=in_channels)

        self.classifier_layer = nn.Linear(out_channels, classifier_num)

    def _make_layer(self, cfg, input_channel):
        if cfg["block_type"] == 'mobilevit':
            return self._make_mit_layer(cfg, input_channel)
        elif cfg["block_type"] == 'mv2':
            return self._make_mobilenet_v2_layer(cfg, input_channel)
        else:
            raise TypeError('Unknown block')

    @staticmethod
    def _make_mit_layer(cfg, input_channel):
        block = []
        stride = cfg.get("stride", 1)
        if stride == 2:  # stride=2 when down-sampling
            block.append(InvertedResidual(in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                          stride=stride, expand_ratio=cfg["mv_expand_ratio"],
                                          norm_layer=cfg['conv_norm'], conv_act=cfg['conv_act']))
            input_channel = cfg.get("out_channels")

        block.append(MobileViTBlockV2(in_channels=input_channel, attn_unit_dim=cfg['attn_unit_dim'],
                                      ffn_multiplier=cfg['ffn_multiplier'], n_attn_blocks=cfg['n_attn_blocks'],
                                      attn_dropout=cfg['attn_dropout'], dropout=cfg['dropout'],
                                      ffn_dropout=cfg['ffn_dropout'], patch_h=cfg['patch_h'],
                                      patch_w=cfg['patch_w'], conv_norm=cfg['conv_norm'],
                                      conv_act=cfg['conv_act'], conv_ksize=cfg['conv_ksize'],
                                      dilation=1, attn_norm_layer=cfg['attn_norm_layer'],
                                      attn_act=cfg['attn_act']))
        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mobilenet_v2_layer(cfg, input_channel):

        block = []
        output_channels = cfg['out_channels']
        expand_ratio = cfg['expand_ratio']
        for i in range(cfg['num_blocks']):
            stride = cfg["stride"] if i == 0 else 1
            block.append(InvertedResidual(in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                          stride=stride, expand_ratio=expand_ratio,
                                          norm_layer=cfg['conv_norm'], conv_act=cfg['conv_act']))
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = torch.mean(x, dim=-1).squeeze(dim=-1)
        x = self.classifier_layer(x)
        return x


def run_model(width='w0_75', orig_pth=''):
    cfg = eval("get_mobilevit_v2_" + width)()       # noqa
    model = MobileViTv2(cfg=cfg)
    model.eval()
    in_tensor = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        out_tensor = model(in_tensor)
    print(out_tensor.size())

    state_dict_origin = torch.load(orig_pth, map_location='cpu')
    state_dict_custom = model.state_dict()
    origin_keys = list(state_dict_origin.keys())
    custom_keys = list(state_dict_custom.keys())

    for key_idx in range(len(custom_keys)):
        origin_model_key = origin_keys[key_idx]
        custom_model_key = custom_keys[key_idx]
        print(custom_model_key, ' ', state_dict_custom[custom_model_key].size(),
              state_dict_origin[origin_model_key].size())
        state_dict_custom.update({custom_model_key: state_dict_origin[origin_model_key]})

    pth_name = "../weights/mobilevit-{}.pth".format(width.replace('_', '.'))    # noqa
    print(pth_name)
    torch.save(state_dict_custom, f=pth_name)


if __name__ == "__main__":
    # w0_5, w0_75, w1_0, w1_25, w1_5, w1_75, w2_0 分别对应 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 网络宽度factor
    # 这主要是用来转化官方模型的部分代码，orig_pth_path指的是下载好的官方预训练模型路径，部分转好的模型放在百度云了。
    model_width = 'w2_0'
    orig_pth_path = "../weights/mobilevitv2-{}.pt".format(      # noqa
        model_width.strip('w').replace('_', '.'))
    run_model(model_width, orig_pth_path)
