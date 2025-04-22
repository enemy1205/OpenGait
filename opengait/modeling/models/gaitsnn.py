import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..backbones.snn_model import Spiking_vit_MetaFormer
from ..modules import SetBlockWrapper,HorizontalPoolingPyramid,PackSequenceWrapper,SeparateFCs,SeparateBNNecks,conv3x3
from utils import clones
from functools import partial
from einops import rearrange
from spikingjelly.clock_driven import functional


class SwinSnnGait(BaseModel):
    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"]
        channels = model_cfg["channels"]
        self.layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(in_channels, channels[0], 1),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
            )
        )
        self.Backbone = Spiking_vit_MetaFormer(
            num_heads=4,
            mlp_ratios=2,
            embed_dim=channels,
            qkv_bias=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=8,
            sr_ratios=1,
        )
        self.FCs = SeparateFCs(16, channels[2], channels[2])
        self.BNNecks = SeparateBNNecks(
            16, channels[2], class_num=model_cfg["SeparateBNNecks"]["class_num"]
        )

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        self.inference_use_emb2 = (
            model_cfg["use_emb2"] if "use_emb2" in model_cfg else False
        )
        
    def forward(self, inputs):
        functional.reset_net(self.Backbone)
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]

        sils = sils.unsqueeze(1)

        del ipts

        out0 = self.layer0(sils)
        
        feat = out0     # [n, c, s, h, w]

        out = self.Backbone(feat.permute(2, 0, 1, 3, 4))  # [s, n, c, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out, seqL,dim=0,options={"dim": 0})[0]  # [n, c, h, w]

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
