import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..backbones.simvp import Decoder, Encoder, Mid_Xnet
from ..backbones.sttn import Discriminator
from ..base_model import BaseModel
from ..modules import (BasicBlock2D, BasicBlock3D, BasicBlockP3D,
                       HorizontalPoolingPyramid, PackSequenceWrapper,
                       SeparateBNNecks, SeparateFCs, SetBlockWrapper, conv1x1,
                       conv3x3)

blocks_map = {"2d": BasicBlock2D, "p3d": BasicBlockP3D, "3d": BasicBlock3D}


class DeepGaitV2(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg["Backbone"]["mode"]
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg["Backbone"]["in_channels"]
        layers = model_cfg["Backbone"]["layers"]
        channels = model_cfg["Backbone"]["channels"]
        self.inference_use_emb2 = (
            model_cfg["use_emb2"] if "use_emb2" in model_cfg else False
        )

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

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
            embed = embed_2
        else:
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


class DeepGaitV2_NO_Base_V3(nn.Module):

    def __init__(self, model_cfg):
        super(DeepGaitV2_NO_Base_V3, self).__init__()
        mode = model_cfg["Backbone"]["mode"]
        assert mode in blocks_map.keys()
        # block = blocks_map[mode]

        # in_channels = model_cfg['Backbone']['in_channels']
        # layers      = model_cfg['Backbone']['layers']
        channels = model_cfg["Backbone"]["channels"]

        # if mode == '3d':
        #     strides = [
        #         [1, 1],
        #         [1, 2, 2],
        #         [1, 2, 2],
        #         [1, 1, 1]
        #     ]
        # else:
        #     strides = [
        #         [1, 1],
        #         [2, 2],
        #         [2, 2],
        #         [1, 1]
        #     ]

        self.inplanes = channels[0]
        # self.layer0 = SetBlockWrapper(nn.Sequential(
        #     conv3x3(in_channels, self.inplanes, 1),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True)
        # ))
        # self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        # self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        # self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        # self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        self.fea_map = nn.Sequential(
            conv1x1(128, channels[3]),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
        )

        # if mode == '2d':
        #     self.layer2 = SetBlockWrapper(self.layer2)
        #     self.layer3 = SetBlockWrapper(self.layer3)
        #     self.layer4 = SetBlockWrapper(self.layer4)

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

    def forward(self, inputs, enc_feat):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(1)

        n, c, s, h, w = sils.size()

        assert sils.size(-1) in [44, 88]

        del ipts

        # out0 = self.layer0(sils) # [b,64,t,h,w]
        # out1 = self.layer1(out0) # [b,64,t,h,w]
        # out2 = self.layer2(out1) # [b,128,t,h/2,w/2]
        # out3 = self.layer3(out2) # [b,256,t,h/2,w/2]
        # out4 = self.layer4(out3) # [b,512,t,h/4,w/4]

        enc_feat = self.fea_map(enc_feat)

        output_size = enc_feat.size()

        enc_feat = enc_feat.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()

        # Temporal Pooling, TP
        outs = self.TP(enc_feat, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p] , [n, class_num, p]

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


class DeepGaitV2_NO_Base_V2(nn.Module):

    def __init__(self, model_cfg):
        super(DeepGaitV2_NO_Base_V2, self).__init__()
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

        self.fea_map = nn.Sequential(
            conv1x1(128, channels[3]),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
        )
        # self.fea_map = nn.Linear(channels[1], channels[3])

        if mode == "2d":
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

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

    def forward(self, inputs, enc_feat):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(1)

        n, c, s, h, w = sils.size()

        assert sils.size(-1) in [44, 88]

        del ipts

        out0 = self.layer0(sils)  # [b,64,t,h,w]
        out1 = self.layer1(out0)  # [b,64,t,h,w]
        out2 = self.layer2(out1)  # [b,128,t,h/2,w/2]
        out3 = self.layer3(out2)  # [b,256,t,h/2,w/2]
        out4 = self.layer4(out3)  # [b,512,t,h/4,w/4]
        enc_feat = self.fea_map(enc_feat)

        output_size = enc_feat.size()

        enc_feat = enc_feat.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()

        # Temporal Pooling, TP
        outs_1 = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        outs_2 = self.TP(enc_feat, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat_1 = self.HPP(outs_1)  # [n, c, p]
        feat_2 = self.HPP(outs_2)  # [n, c, p]

        embed_1 = self.FCs(torch.add(feat_1, feat_2))  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p] , [n, class_num, p]

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


class SimVP(nn.Module):
    def __init__(self, model_cfg):
        super(SimVP, self).__init__()
        T, C, H, W = model_cfg["shape_in"]
        hid_S = model_cfg["hid_S"]
        N_S = model_cfg["N_S"]
        hid_T = model_cfg["hid_T"]
        N_T = model_cfg["N_T"]
        incep_ker = model_cfg["incep_ker"]
        groups = model_cfg["groups"]
        # self.netGait = GaitSet_Nobase(model_cfg['GaitSet'])
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)
        self.dis = Discriminator(model_cfg["Dis"], use_sigmoid=True)
        self._relu = torch.nn.ReLU1(True)

    # def forward(self, inputs):
    #     ipts, labs, _, _, seqL = inputs
    #     # ipts[0 or 1] : [b,t,h,w]
    #     gt_sils = ipts[0].unsqueeze(2)
    #     occ_sils = ipts[1].unsqueeze(2)
    #     b, t, c, h, w = occ_sils.shape
    #     x = occ_sils.view(b*t, c, h, w)

    #     embed, skip = self.enc(x)
    #     _, C_, H_, W_ = embed.shape

    #     z = embed.view(b, t, C_, H_, W_)
    #     hid = self.hid(z)
    #     hid = hid.reshape(b*t, C_, H_, W_)

    #     pred_y  = self.dec(hid, skip)
    #     pred_y = self._relu(pred_y)
    #     # pred_y  = pred_y .reshape(b, t, c, h, w)

    #     recovered_sils = pred_y.view(b*t, c, h, w)
    #     gt_sils = gt_sils.view(b*t, c, h, w)

    #     gen_vid_feat = self.dis(recovered_sils)

    #     real_sils_embs = self.dis(gt_sils)
    #     fake_sils_embs = self.dis(recovered_sils.detach())

    #     retval = {
    #         'training_feat': {
    #             'adv': {'logits': fake_sils_embs, 'labels': real_sils_embs},
    #             'gan': {'pred_silt_video':pred_y,'gt_silt_video':gt_sils,'gen_vid_feat':gen_vid_feat},
    #         },
    #         'visual_summary': {
    #             'image/gt_sils': gt_sils.view(b*t, c, h, w), 'image/occ_sils': occ_sils.view(b*t, c, h, w), "image/rec_sils": pred_y.view(b*t, c, h, w)
    #         },
    #         'inference_feat': {
    #             'gt': gt_sils.view(b*t, c, h, w),
    #             'pred': pred_y.view(b*t, c, h, w)
    #         }
    #     }

    #     # retval_gait = self.netGait([[recovered_sils.view(b,t, h, w)], labs, None, None, seqL])
    #     # real_gait = self.netGait([[gt_sils.view(b,t, h, w)], labs, None, None, seqL])
    #     # retval['training_feat']['triplet'] = retval_gait['training_feat']['triplet']
    #     # retval['inference_feat']['embeddings'] = retval_gait['inference_feat']['embeddings']
    #     return retval

    def inference_(self, ipts):
        # ipts[0 or 1] : [b,t,h,w]
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.shape
        x = occ_sils.view(b * t, c, h, w)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(b, t, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(b * t, C_, H_, W_)

        pred_y = self.dec(hid, skip)
        pred_y = self._relu(pred_y)
        pred_y = pred_y.reshape(b, t, h, w)

        recovered_sils = pred_y

        return recovered_sils
