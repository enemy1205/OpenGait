import torch.optim as optim
import torch.nn as nn
import torch
import os.path as osp
from ..base_model import BaseModel
from ..backbones.fuse_former import (
    deconv,
    Discriminator,
    TransformerBlock,
    SoftComp,
    SoftSplit,
    AddPosEmb,
    Encoder,
)
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

from utils import get_valid_args, get_attr_from
import numpy as np
from einops import rearrange

blocks_map = {"2d": BasicBlock2D, "p3d": BasicBlockP3D, "3d": BasicBlock3D}


class FFormerGaitDis(BaseModel):

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

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(
            16, channels[2], class_num=model_cfg["SeparateBNNecks"]["class_num"]
        )

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        ss_channel = model_cfg["ss_channel"]
        ts_hidden = model_cfg["hidden"]
        st_stack_num = model_cfg["stack_num"]
        num_head = model_cfg["num_head"]
        kernel_size = (model_cfg["kernel_size"][0], model_cfg["kernel_size"][1])
        padding = (model_cfg["padding"][0], model_cfg["padding"][1])
        soft_stride = (model_cfg["stride"][0], model_cfg["stride"][1])
        output_size = (model_cfg["output_size"][0], model_cfg["output_size"][1])

        blocks = []
        dropout = model_cfg["dropout"]
        t2t_params = {
            "kernel_size": kernel_size,
            "stride": soft_stride,
            "padding": padding,
            "output_size": output_size,
        }
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int(
                (output_size[i] + 2 * padding[i] - (d - 1) - 1) / soft_stride[i] + 1
            )
        for _ in range(st_stack_num):
            blocks.append(
                TransformerBlock(
                    hidden=ts_hidden,
                    num_head=num_head,
                    dropout=dropout,
                    n_vecs=n_vecs,
                    t2t_params=t2t_params,
                )
            )
        self.transformer = nn.Sequential(*blocks)
        self.ss = SoftSplit(
            ss_channel // 2,
            ts_hidden,
            kernel_size,
            soft_stride,
            padding,
            dropout=dropout,
        )
        self.add_pos_emb = AddPosEmb(n_vecs, ts_hidden)
        self.sc = SoftComp(
            ss_channel // 2, ts_hidden, output_size, kernel_size, soft_stride, padding
        )
        self.encoder = Encoder()
        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(ss_channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

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

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg["solver"])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ["solver"])
        transformer_params = (
            list(self.transformer.parameters())
            + list(self.ss.parameters())
            + list(self.add_pos_emb.parameters())
            + list(self.sc.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        params_list = [
            {
                "params": transformer_params,
                "lr": optimizer_cfg["lr"],
                "weight_decay": optimizer_cfg["weight_decay"],
            },
            {
                "params": self.FCs.parameters(),
                "lr": optimizer_cfg["lr"] * 1,
                "weight_decay": optimizer_cfg["weight_decay"],
            },
            {
                "params": self.BNNecks.parameters(),
                "lr": optimizer_cfg["lr"] * 1,
                "weight_decay": optimizer_cfg["weight_decay"],
            },
        ]
        for i in range(5):
            if hasattr(self, "layer%d" % i):
                params_list.append(
                    {
                        "params": getattr(self, "layer%d" % i).parameters(),
                        "lr": optimizer_cfg["lr"],
                        "weight_decay": optimizer_cfg["weight_decay"],
                    }
                )
        optimizer = optimizer(params_list, **valid_arg)
        return optimizer

    # def finetune_parameters(self):
    #     dis_tune_params = list()
    #     others_params = list()
    #     for name, p in self.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         if 'dis' in name:
    #             dis_tune_params.append(p)
    #         else:
    #             others_params.append(p)
    #     return [{'params': dis_tune_params, 'lr': self.dis_lr},{'params': others_params}]

    # def get_optimizer(self, optimizer_cfg):
    #     self.msg_mgr.log_info(optimizer_cfg)
    #     optimizer = get_attr_from([optim], optimizer_cfg['solver'])
    #     valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
    #     optimizer = optimizer(self.finetune_parameters(), **valid_arg)
    #     return optimizer

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.size()
        enc_feat = self.encoder(occ_sils.view(b * t, c, h, w))
        trans_feat = self.ss(enc_feat, b)
        trans_feat = self.add_pos_emb(trans_feat)
        trans_feat = self.transformer(trans_feat)
        trans_feat = self.sc(trans_feat, t)
        enc_feat = enc_feat + trans_feat  # [b*t,c ,h/4 ,w/4]
        rec_sil = self.decoder(enc_feat)
        rec_sil = torch.tanh(rec_sil)

        rec_out0 = self.layer0(rec_sil.view(b, c, t, h, w))  # [b,64,t,h,w]
        rec_out1 = self.layer1(rec_out0)  # [b,64,t,h,w]
        rec_out2 = self.layer2(rec_out1)  # [b,128,t,h/2,w/2]
        rec_out3 = self.layer3(rec_out2)  # [b,256,t,h/2,w/2]
        rec_out4 = self.layer4(rec_out3)  # [b,512,t,h/4,w/4]

        gt_out0 = self.layer0(gt_sils.transpose(1, 2))  # [b,64,t,h,w]
        gt_out1 = self.layer1(gt_out0)  # [b,64,t,h,w]
        gt_out2 = self.layer2(gt_out1)  # [b,128,t,h/2,w/2]
        gt_out3 = self.layer3(gt_out2)  # [b,256,t,h/2,w/2]
        gt_out4 = self.layer4(gt_out3)  # [b,512,t,h/4,w/4]
        # Temporal Pooling, TP
        outs_rec = self.TP(rec_out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_gt = self.TP(gt_out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat_rec = self.HPP(outs_rec)  # [n, c, p]
        feat_gt = self.HPP(outs_gt)  # [n, c, p]
        # feat_2 = self.HPP(outs_2)

        # embed_1 = self.FCs(torch.add(feat_1,feat_2))
        embed_1 = self.FCs(feat_rec)
        embed_1_gt = self.FCs(feat_gt)
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p] , [n, class_num, p]
        embed = embed_1
        retval = {
            "training_feat": {
                "adv": {"logits": embed_1, "labels": embed_1_gt},
                "gan": {
                    "pred_silt_video": rec_sil,
                    "gt_silt_video": gt_sils.view(b * t, c, h, w),
                },
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {
                "image/gt_sils": gt_sils.view(b * t, c, h, w),
                "image/occ_sils": occ_sils.view(b * t, c, h, w),
                "image/rec_sils": rec_sil.view(b * t, c, h, w),
            },
            "inference_feat": {
                "gt": gt_sils.view(b * t, c, h, w),
                "pred": rec_sil.view(b * t, c, h, w),
                "embeddings": embed,
            },
        }
        return retval
