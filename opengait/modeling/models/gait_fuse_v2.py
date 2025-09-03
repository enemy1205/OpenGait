from einops import rearrange
from ..base_model import BaseModel
from ..backbones.fuse_former import (
    TransformerBlock,
    SoftSplit,
    AddPosEmb,
    SoftComp,
    Encoder,
)
from utils import get_valid_args, get_attr_from
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
import torch
import numpy as np
import cv2
import os
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

blocks_map = {"2d": BasicBlock2D, "p3d": BasicBlockP3D, "3d": BasicBlock3D}


class CatFusion(nn.Module):
    def __init__(self, in_channels=64):
        super(CatFusion, self).__init__()
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, 2 * in_channels),
            )
        )

    def forward(self, sil_feat, map_feat):
        """
        sil_feat: [n, c, s, h, w]
        map_feat: [n, c, s, h, w]
        """
        feats = torch.cat([sil_feat, map_feat], 1)
        retun = self.conv(feats)
        return retun


class FuseGait_V2(BaseModel):
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

        # self.fusion = CatFusion(channels[3]//2)

        channel = model_cfg["Transformer"]["channel"]
        hidden = model_cfg["Transformer"]["hidden"]
        stack_num = model_cfg["Transformer"]["stack_num"]
        num_head = model_cfg["Transformer"]["num_head"]
        kernel_size = (
            model_cfg["Transformer"]["kernel_size"][0],
            model_cfg["Transformer"]["kernel_size"][1],
        )
        padding = (
            model_cfg["Transformer"]["padding"][0],
            model_cfg["Transformer"]["padding"][1],
        )
        stride = (
            model_cfg["Transformer"]["stride"][0],
            model_cfg["Transformer"]["stride"][1],
        )
        output_size = (
            model_cfg["Transformer"]["output_size"][0],
            model_cfg["Transformer"]["output_size"][1],
        )

        blocks = []
        dropout = model_cfg["Transformer"]["dropout"]
        t2t_params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "output_size": output_size,
        }
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int(
                (output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1
            )
        for _ in range(stack_num):
            blocks.append(
                TransformerBlock(
                    hidden=hidden,
                    num_head=num_head,
                    dropout=dropout,
                    n_vecs=n_vecs,
                    t2t_params=t2t_params,
                )
            )
        self.transformer = nn.Sequential(*blocks)
        self.ss = SoftSplit(
            channel // 2, hidden, kernel_size, stride, padding, dropout=dropout
        )

        self.add_pos_emb = AddPosEmb(n_vecs, hidden)
        self.sc = SoftComp(
            channel // 2, hidden, output_size, kernel_size, stride, padding
        )
        self.relu = nn.ReLU(inplace=True)
        self.encoder = Encoder()

    def visual_heatmap(self, input_tensor, sils):
        n, c, s, height, width = sils.size()
        outputs = (input_tensor**2).sum(1)
        # print(type(outputs))
        outputs = outputs.squeeze()

        # print(outputs.size())
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        images = {}
        sils, outputs = sils.cpu(), outputs.cpu()
        sils = sils.permute(0, 2, 1, 3, 4)
        n, s, c, height, width = sils.size()
        sils = sils.squeeze(dim=0)
        ratio = sils.size(0) // outputs.size(0)
        for j in range(outputs.size(0)):
            # img = sils[j, 0 , ...]
            img = sils[j * ratio, ...]
            img_mean = IMAGENET_MEAN
            img_std = IMAGENET_STD
            # for t, m, s in zip(img, img_mean, img_std):
            #         t.mul_(s).add_(m).clamp_(0, 1)

            img_np = np.uint8(np.floor(img.numpy() * 255))
            # img_np = np.uint8(np.floor(img.numpy()))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)
            new_img_np = np.zeros((64, 44, 3))
            new_img_np[:, :, :2] = img_np
            new_img_np[:, :, 2] = img_np[:, :, 0]

            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
            # overlapped
            overlapped = new_img_np * 0.3 + am * 0.7
            # overlapped = am
            # overlapped = new_img_np
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)
            # H, W, C = overlapped.shape
            # crop_h = int(0.1 * H)
            # crop_w = int(0.1 * W)

            # # 裁剪操作
            # cropped_image = overlapped[crop_h:-crop_h, crop_w:-crop_w, :]
            # cropped_image = cv2.resize(cropped_image, (width, height))
            images.update({j * ratio: overlapped})

        return images

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

        n, c, s, h, w = sils.size()

        assert sils.size(-1) in [44, 88]

        del ipts

        out0 = self.layer0(sils)  # [b,64,t,h,w]
        out1 = self.layer1(out0)  # [b,64,t,h,w]
        out2 = self.layer2(out1)  # [b,128,t,h/2,w/2]
        out3 = self.layer3(out2)  # [b,256,t,h/4,w/4]

        enc_feat = self.encoder(
            rearrange(sils, "n c s h w -> n s c h w").view(n * s, c, h, w)
        )  # [t,128,h/4,w/4]

        trans_feat = self.ss(enc_feat, n)  # [b,1050,512]
        trans_feat = self.add_pos_emb(trans_feat)  # [b,1050,512]
        trans_feat = self.transformer(trans_feat)  # [b,1050,512]
        trans_feat = self.sc(trans_feat, s)  # [t*b,256,h/4,w/4]

        enc_feat = enc_feat + trans_feat

        output_size = enc_feat.size()

        enc_feat = enc_feat.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()

        out3 = self.relu(out3 + enc_feat)
        # out3 = out3+enc_feat
        # out4 = self.fusion(out3,enc_feat)
        out4 = self.layer4(out3)  # [b,512,t,h/4,w/4]

        dataset = "gait_fuse_v2_out3"
        images = self.visual_heatmap(out3, sils)
        ipts, labs1, var1, var2, seqL1 = inputs

        for lab, sub, view in zip(labs1, var1, var2):
            for index, image in images.items():
                ske_save_path = (
                    "/home/sp/sp/projects/OpenGait/vis2/{}/{:03d}/{}/{}".format(
                        dataset, lab, sub, view
                    )
                )
                if osp.exists(ske_save_path) is False:
                    os.makedirs(ske_save_path)
                ske_save_name = "{}/{}-sils.png".format(ske_save_path, index)
                cv2.imwrite(ske_save_name, image)

        # Temporal Pooling, TP
        outs_1 = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat_1 = self.HPP(outs_1)  # [n, c, p]

        embed_1 = self.FCs(feat_1)  # [n, c, p]

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
