import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import (
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
    SetBlockWrapper,
    ParallelBN1d,
)
from timm.models.layers import trunc_normal_

from einops import rearrange

import torch.nn.functional as F

from ..modules import BasicBlock2D, BasicBlockP3D

import os.path as osp
from ..backbones.spikformer import SpikformerBlock, SPS_64
from utils import get_valid_args, get_attr_from
import cv2
import os


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SNNGait(BaseModel):
    def __init__(self, cfgs, training):
        super(SNNGait, self).__init__(cfgs, training=training)

    def build_network(self, model_cfg):
        channels = model_cfg["Backbone"]["channels"]
        layers = model_cfg["Backbone"]["layers"]
        in_c = model_cfg["Backbone"]["in_channels"]

        self.inplanes = channels[0]
        # self.layer0 = SetBlockWrapper(nn.Sequential(
        #     conv3x3(in_c, self.inplanes, 1),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True)
        # ))
        # self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], stride=[1, 1], blocks_num=layers[0], mode='2d'))
        # self.layer2 = self.make_layer(BasicBlockP3D, channels[1], stride=[2, 2], blocks_num=layers[1], mode='p3d')

        # self.ulayer = SetBlockWrapper(nn.UpsamplingBilinear2d(size=(30, 20)))
        self.SPS = SPS_64(in_c, embed_dims=256)
        self.transformer = SpikformerBlock(
            in_channels=1,
            embed_dim=256,
            num_heads=[16, 32],
            depths=[layers[2], layers[3]],
            mlp_ratios=[4, 4],
            drop_path_rate=0.1,
        )

        self.FCs = SeparateFCs(
            model_cfg["SeparateBNNecks"]["parts_num"], in_channels=256, out_channels=256
        )
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

    def make_layer(self, block, planes, stride, blocks_num, mode="2d"):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == "3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=[1, 1, 1],
                        stride=stride,
                        padding=[0, 0, 0],
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            elif mode == "2d":
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif mode == "p3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=[1, 1, 1],
                        stride=[1, *stride],
                        padding=[0, 0, 0],
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                raise TypeError("xxx")
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ["2d", "p3d"] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=s))
        return nn.Sequential(*layers)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0].unsqueeze(1)

        del ipts

        out = self.SPS(sils)
        # out0 = self.layer0(sils) # [n, 64, s, h, w]
        # out1 = self.layer1(out0) # [n, 64, s, h, w]
        # out2 = self.layer2(out1) # [n, 128, s, h/2, w/2]

        # out2 = self.ulayer(out2) # [n, 128, s, h/2-2=30 , w/2-2=20]
        # out4 = self.transformer(out2) # [n, 512,s, 15, 10]

        out4 = self.transformer(out)

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c,15, 10]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        feat = torch.cat([feat, feat[:, :, -1].clone().detach().unsqueeze(-1)], dim=-1)
        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed_1 = embed_1.contiguous()[:, :, :-1]  # [n, p, c]
        embed_2 = embed_2.contiguous()[:, :, :-1]  # [n, p, c]
        logits = logits.contiguous()[:, :, :-1]  # [n, p, c]

        embed = embed_1

        retval = {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {
                "image/sils": rearrange(sils, "n c s h w -> (n s) c h w")
            },
            "inference_feat": {"embeddings": embed},
        }
        return retval
