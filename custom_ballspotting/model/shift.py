import math

import timm
import torch
import torchvision
from torch import nn

from custom_ballspotting.model.layers import _GSF, _GSM


def make_temporal_shift(net, clip_len, mode="gsf"):
    def _build_shift(module):
        if mode == "gsm" or mode == "gsf":
            return GatedShift(module, n_segment=clip_len, n_div=4, mode=mode)
        raise NotImplementedError(f"Unsupported shift mode: {mode}")

    if isinstance(net, torchvision.models.ResNet):
        n_round = 2 if len(list(net.layer3.children())) >= 23 else 1

        def make_block_temporal(stage):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = _build_shift(block.conv1)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1)
        net.layer2 = make_block_temporal(net.layer2)
        net.layer3 = make_block_temporal(net.layer3)
        net.layer4 = make_block_temporal(net.layer4)
    elif isinstance(net, timm.models.regnet.RegNet):
        for stage in [net.s3, net.s4]:
            for block in stage.children():
                block.conv1 = _build_shift(block.conv1)
    else:
        raise NotImplementedError(f"Unsupported architecture: {type(net)}")


class GatedShift(nn.Module):
    def __init__(self, net, n_segment, n_div, mode="gsm"):
        super().__init__()
        if isinstance(net, torchvision.models.resnet.BasicBlock):
            channels = net.conv1.in_channels
        elif isinstance(net, torchvision.ops.misc.ConvNormActivation):
            channels = net[0].in_channels
        elif isinstance(net, timm.layers.conv_bn_act.ConvBnAct):
            channels = net.conv.in_channels
        elif isinstance(net, nn.Conv2d):
            channels = net.in_channels
        else:
            raise NotImplementedError(type(net))

        self.fold_dim = math.ceil(channels // n_div / 4) * 4
        self.gs = (
            _GSM(self.fold_dim, n_segment)
            if mode == "gsm"
            else _GSF(self.fold_dim, n_segment, 100)
        )
        self.net = net

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:, : self.fold_dim, :, :] = self.gs(x[:, : self.fold_dim, :, :])
        y[:, self.fold_dim :, :, :] = x[:, self.fold_dim :, :, :]
        return self.net(y)
