import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
    conv1x1,
    conv3x3,
    BasicBlock2D,
    BasicBlockP3D,
    BasicBlock3D,
)

from einops import rearrange

blocks_map = {"2d": BasicBlock2D, "p3d": BasicBlockP3D, "3d": BasicBlock3D}


class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels, dropout=0.1, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.norm = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [b, c, t, h, w]
        b, c, t, h, w = x.shape
        x_flat = x.view(b, c, t * h * w).permute(2, 0, 1)  # [t*h*w, b, c]
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)  # [t*h*w, b, c]
        attn_output = attn_output.permute(1, 2, 0).view(b, c, t, h, w)
        attn_output = self.norm(x + self.dropout(attn_output))
        return attn_output


class AxialAttention(nn.Module):
    def __init__(self, channels, dropout=0.1, num_heads=4):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads
        )
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads
        )
        self.norm = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [b, c, t, h, w]
        b, c, t, h, w = x.shape
        # 时间注意力
        x_temporal = x.permute(2, 0, 3, 4, 1).reshape(t, b * h * w, c)  # [t, b*h*w, c]
        temporal_output, _ = self.temporal_attn(
            x_temporal, x_temporal, x_temporal
        )  # [t, b*h*w, c]
        temporal_output = temporal_output.view(t, b, h, w, c).permute(
            1, 4, 0, 2, 3
        )  # [b, c, t, h, w]
        # 空间注意力
        x_spatial = x.permute(3, 0, 2, 4, 1).reshape(h, b * t * w, c)  # [h, b*t*w, c]
        spatial_output, _ = self.spatial_attn(
            x_spatial, x_spatial, x_spatial
        )  # [h, b*t*w, c]
        spatial_output = spatial_output.view(h, b, t, w, c).permute(
            1, 4, 2, 0, 3
        )  # [b, c, t, h, w]
        # 融合
        output = self.norm(self.dropout(temporal_output + spatial_output))
        return output


class AdaptiveMixPooling(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveMixPooling, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # 自适应平均池化
        avg_pooled = F.adaptive_avg_pool2d(
            x, self.output_size
        )  # [n, c, output_size[0], output_size[1]]

        # 自适应最大池化
        max_pooled = F.adaptive_max_pool2d(
            x, self.output_size
        )  # [n, c, output_size[0], output_size[1]]

        # 将两者相加
        return avg_pooled + max_pooled  # [n, c, output_size[0], output_size[1]]


class DeepGaitV2Atten(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg["Backbone"]["mode"]
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg["Backbone"]["in_channels"]
        layers = model_cfg["Backbone"]["layers"]
        channels = model_cfg["Backbone"]["channels"]

        if mode == "3d":
            strides = [[1, 1], [1, 2, 2], [1, 2, 2], [1, 1, 1]]
        else:
            strides = [[1, 1], [2, 2], [2, 2], [1, 1]]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(in_channels, self.inplanes, 1),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
            )
        )
        self.layer1 = SetBlockWrapper(
            self.make_layer(
                BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode
            )
        )

        self.layer2 = self.make_layer(
            block, channels[1], strides[1], blocks_num=layers[1], mode=mode
        )
        self.layer3 = self.make_layer(
            block, channels[2], strides[2], blocks_num=layers[2], mode=mode
        )
        self.layer4 = self.make_layer(
            block, channels[3], strides[3], blocks_num=layers[3], mode=mode
        )
        if mode == "2d":
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)
        #
        self.attn_layer_3 = AxialAttention(256)
        self.attn_layer_4 = AxialAttention(512)
        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(
            16, channels[2], class_num=model_cfg["SeparateBNNecks"]["class_num"]
        )
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

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

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(1)
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)  # out0 [n,64,s,64,44]
        out1 = self.layer1(out0)  # out1 [n,64,s,64,44]
        out2 = self.layer2(out1)  # out2 [n,128,s,32,22]
        out3 = self.layer3(out2)  # out3 [n,256,s,16,11]
        out3_fusion = self.attn_layer_3(out3) + out3
        out4 = self.layer4(out3_fusion)  # [n, c, s, h, w] # out4 [n,512,s,16,11]
        out4_fusion = self.attn_layer_4(out4) + out4
        # Temporal Pooling, TP
        outs_1 = self.TP(out4_fusion, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat_1 = self.HPP(outs_1)  # [n, c, p]

        embed_1 = self.FCs(feat_1)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        embed = embed_1

        retval = {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {
                "image/sils": rearrange(sils, "n c s h w -> (n s) c h w"),
            },
            "inference_feat": {"embeddings": embed},
        }

        return retval
