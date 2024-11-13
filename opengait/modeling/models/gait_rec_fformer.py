import torch.optim as optim
import os.path as osp
from ..base_model import BaseModel
from .deepgaitv2 import DeepGaitV2_NO_Base_V2
from ..backbones.fuse_former import InpaintGenerator, Discriminator
from utils import get_valid_args, get_attr_from
import numpy as np
import torch


class FFormer_Rec(BaseModel):
    ###
    #  单一的步态补全网络
    ###
    def build_network(self, model_cfg):
        self.netGen = InpaintGenerator(model_cfg["Gen"])
        self.netDis = Discriminator(model_cfg["Dis"], use_sigmoid=True)
        # self._relu = torch.nn.ReLU1(True)
        self.dis_lr = model_cfg["lr_D"]

    def finetune_parameters(self):
        dis_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "dis" in name:
                dis_tune_params.append(p)
            else:
                others_params.append(p)
        return [
            {"params": dis_tune_params, "lr": self.dis_lr},
            {"params": others_params},
        ]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg["solver"])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ["solver"])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)

        b, t, c, h, w = gt_sils.size()
        # NetG input : [b,x,c,h,w]
        # recovered_sils : [b*t,c,h,w]
        recovered_sils = self.netGen(occ_sils)

        gt_sils = gt_sils.view(b * t, c, h, w)

        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(recovered_sils.detach())

        gen_vid_feat = self.netDis(recovered_sils)

        retval = {
            "training_feat": {
                "adv": {"logits": fake_sils_embs, "labels": real_sils_embs},
                "gan": {
                    "pred_silt_video": recovered_sils,
                    "gt_silt_video": gt_sils,
                    "gen_vid_feat": gen_vid_feat,
                },
            },
            "visual_summary": {
                "image/gt_sils": gt_sils,
                "image/occ_sils": occ_sils.view(b * t, c, h, w),
                "image/rec_sils": recovered_sils.view(b * t, c, h, w),
            },
            "inference_feat": {
                "gt": gt_sils.view(b * t, c, h, w),
                "pred": recovered_sils.view(b * t, c, h, w),
            },
        }

        return retval


class FFormer_E2E_Sopt(BaseModel):
    ###
    # 端到端的步态识别，直接拼接，单优化器
    ###
    def build_network(self, model_cfg):
        self.netGait = DeepGaitV2_NO_Base_V2(model_cfg["Gait"])
        self.netGen = InpaintGenerator(model_cfg["Gen"])
        self.netDis = Discriminator(model_cfg["Dis"], use_sigmoid=True)
        self.dis_lr = model_cfg["lr_D"]
        self.gait_lr = model_cfg["lr_gait"]

    def finetune_parameters(self):
        dis_tune_params = list()
        gait_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "netDis" in name:
                dis_tune_params.append(p)
            elif "netGait" in name:
                gait_tune_params.append(p)
            else:
                others_params.append(p)
        return [
            {"params": dis_tune_params, "lr": self.dis_lr},
            {"params": gait_tune_params, "lr": self.gait_lr},
            {"params": others_params},
        ]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg["solver"])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ["solver"])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.size()
        # NetG input : [b,t,c,h,w]
        recovered_sils,enc_feat = self.netGen(occ_sils)
        recovered_sils = recovered_sils.view(b * t, c, h, w)
        gt_sils = gt_sils.view(b * t, c, h, w)
        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(recovered_sils.detach())
        gen_vid_feat = self.netDis(recovered_sils)
        retval = {
            "training_feat": {
                "adv": {"logits": fake_sils_embs, "labels": real_sils_embs},
                "gan": {
                    "pred_silt_video": recovered_sils,
                    "gt_silt_video": gt_sils,
                    "gen_vid_feat": gen_vid_feat,
                },
            },
            "visual_summary": {
                "image/gt_sils": gt_sils,
                "image/occ_sils": occ_sils.view(b * t, c, h, w),
                "image/rec_sils": recovered_sils.view(b * t, c, h, w),
            },
            "inference_feat": {
                "gt": gt_sils.view(b * t, c, h, w),
                "pred": recovered_sils.view(b * t, c, h, w),
            },
        }
        retval_gait = self.netGait(
            [[recovered_sils.view(b, t, h, w)], labs, None, None, seqL],enc_feat
        )
        retval["training_feat"]["triplet"] = retval_gait["training_feat"]["triplet"]
        retval["training_feat"]["softmax"] = retval_gait["training_feat"]["softmax"]
        retval["inference_feat"]["embeddings"] = retval_gait["inference_feat"][
            "embeddings"
        ]
        return retval
