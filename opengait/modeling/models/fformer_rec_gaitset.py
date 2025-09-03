import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import (
    SeparateFCs,
    BasicConv2d,
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
)

from ..backbones.fuse_former import (
    deconv,
    Discriminator,
    TransformerBlock,
    SoftComp,
    SoftSplit,
    AddPosEmb,
    Encoder,
)

from utils import get_valid_args, get_attr_from
import numpy as np
import torch.optim as optim


class GaitSet_FFormer_rec(BaseModel):

    def build_network(self, model_cfg):

        in_c = model_cfg["in_channels"]
        self.set_block1 = nn.Sequential(
            BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.set_block2 = nn.Sequential(
            BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.set_block3 = nn.Sequential(
            BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.Head = SeparateFCs(**model_cfg["SeparateFCs"])

        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

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

        self.netDis = Discriminator(model_cfg["Dis"], use_sigmoid=True)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        gt_sils = ipts[0].unsqueeze(1)
        occ_sils = ipts[1].unsqueeze(1)
        b, c, t, h, w = occ_sils.size()
        outs = self.set_block1(occ_sils)
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)

        enc_feat = outs
        enc_feat = enc_feat.view(b * t, -1, h // 4, w // 4)
        trans_feat = self.ss(enc_feat, b)
        trans_feat = self.add_pos_emb(trans_feat)
        trans_feat = self.transformer(trans_feat)
        trans_feat = self.sc(trans_feat, t)
        enc_feat = enc_feat + trans_feat  # [b*t,c ,h/4 ,w/4]
        rec_sil = self.decoder(enc_feat)
        rec_sil = torch.tanh(rec_sil)
        gt_sils = gt_sils.view(b * t, c, h, w)
        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(rec_sil.detach())
        gen_vid_feat = self.netDis(rec_sil)

        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)

        retval = {
            "training_feat": {
                "adv": {"logits": fake_sils_embs, "labels": real_sils_embs},
                "gan": {
                    "pred_silt_video": rec_sil,
                    "gt_silt_video": gt_sils,
                    "gen_vid_feat": gen_vid_feat,
                },
                "gan": {"pred_silt_video": rec_sil, "gt_silt_video": gt_sils},
                "triplet": {"embeddings": embs, "labels": labs},
            },
            "visual_summary": {
                "image/gt_sils": gt_sils,
                "image/occ_sils": occ_sils.view(b * t, c, h, w),
                "image/rec_sils": rec_sil.view(b * t, c, h, w),
            },
            "inference_feat": {
                "gt": gt_sils.view(b * t, c, h, w),
                "pred": rec_sil.view(b * t, c, h, w),
                "embeddings": embs,
            },
        }
        return retval
