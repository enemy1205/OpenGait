
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import cv2
import os
import os.path as osp
import numpy as np
import cv2
import os
import os.path as osp
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from ..backbones.LG_Modules import CorrBlock


def gem(x, p=6.5, eps=1e-6):
    # print('x-',x.shape)
    # print('xpow-',x.clamp(min=eps).pow(p).shape)
    # print(F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).shape)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1. / p)


class GeM(nn.Module):

    def __init__(self, p=6.5, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # print('p-',self.p)
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def gem1(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM_1(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_1, self).__init__()
        # self.p = Parameter(torch.ones(1)*p)
        self.p = 1
        self.eps = eps

    def forward(self, x):
        return gem1(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(self.eps) + ')'


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class LagrangeGait_New(BaseModel):
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
        ratio = sils.size(0)//outputs.size(0)
        for j in range(outputs.size(0)):
            # img = sils[j, 0 , ...]
            img = sils[j*ratio, ...]
            img_mean = IMAGENET_MEAN
            img_std = IMAGENET_STD
            # for t, m, s in zip(img, img_mean, img_std):
            #         t.mul_(s).add_(m).clamp_(0, 1)

            img_np = np.uint8(np.floor(img.numpy() * 255))
            # img_np = np.uint8(np.floor(img.numpy()))
            img_np = img_np.transpose((1, 2, 0))# (c, h, w) -> (h, w, c)
            new_img_np = np.zeros((64, 44, 3))  
            new_img_np[:, :, :2] = img_np
            new_img_np[:, :, 2] = img_np[:, :, 0]

            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
            # overlapped
            overlapped = new_img_np*0.3 + am*0.7
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
            images.update({j*ratio:overlapped})

        return images
    
    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        num_classes = model_cfg['class_num']
        view_nums = model_cfg['view_nums']
        radius = model_cfg['radius']

        # self.bin_num = model_cfg['bin_num']
        self.bin_numgl = [32*2]
        self.bin_numgl_motion = [4]

        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )
        self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ########################################Level-1###########################################################
        self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.Gem = GeM()
        # self.avgpool = GeM_1()
        # self.cls = nn.Linear(in_features=in_c[2], out_features=view_nums)

        # self.view_embedding_64 = nn.Parameter(torch.randn(view_nums, in_c[0], 1))
        # self.view_embedding_64_motion = nn.Parameter(torch.randn(view_nums, in_c[0], 1))
        # self.view_embedding_64_motion2 = nn.Parameter(torch.randn(view_nums, in_c[0], 1))

        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(sum(self.bin_numgl), in_c[2] , in_c[3])))

        self.fc_bin_motion = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(sum(self.bin_numgl_motion), in_c[2] , in_c[3])))

        # self.fc_bin0 = SeparateFCs(sum(self.bin_numgl), in_c[2] + in_c[0], in_c[3])
        # self.fc_bin1 = SeparateFCs(sum(self.bin_numgl_motion), in_c[2] + in_c[0], in_c[3])

        self.bn2 = nn.BatchNorm1d(in_c[3])
        self.fc2 = SeparateFCs(sum(self.bin_numgl + self.bin_numgl_motion), in_c[3], num_classes)

        self.motion_extract = CorrBlock(num_levels=1, radius=radius, input_dim=in_c[0])
        self.motion_conv1 = BasicConv3d(1 * (radius * 2 + 1) ** 2, in_c[1], kernel_size=3)
        self.motion_conv2 = BasicConv3d(in_c[1], in_c[2], kernel_size=3)
        self.pool_motion = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # view = [int(int(i) / 15) if int(i) <= 90 else int(int(i) - 90 + 15) / 15 for i in view]
        # view = torch.tensor(view).long().cuda()
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        x2d_motion = outs.clone()
        x2d_motion = self.MaxPool0(x2d_motion)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        b, c, t, h, w = x2d_motion.shape
        x2d_motion = x2d_motion.view(b * c, t, h, w)
        x2d_motion = F.avg_pool2d(x2d_motion, kernel_size=(2, 2))
        x2d_motion = x2d_motion.view(b, c, t, h // 2, w // 2)
        x2d_motion = self.motion_extract(x2d_motion)  # b (2*r+1)**2*2 t 16 11
        x2d_motion = self.motion_conv1(x2d_motion)
        x2d_motion = self.motion_conv2(x2d_motion)  # b 256 t//3 16 11
        # b, c, t, h, w = x2d_motion.shape
        x2d_motion = x2d_motion.mean(2)
        #
        outs = self.GLConvA1(outs)
        
        
        if outs.shape[2]>1:
            dataset = 'lagrangegait_outs_GLConvA1'
            images = self.visual_heatmap(outs, sils)
            ipts, labs1, var1, var2, seqL1 = inputs
    
            for lab,sub,view in zip(labs1,var1,var2):
                for index,image in images.items():
                    ske_save_path = "/home/sp/sp/projects/OpenGait/vis2/{}/{:03d}/{}/{}".format(dataset,lab, sub,view)
                    if osp.exists(ske_save_path) is False:
                        os.makedirs(ske_save_path)
                    ske_save_name = "{}/{}-sils.png".format(ske_save_path,index)
                    cv2.imwrite(ske_save_name,image)

        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

                 
        x2db3d = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # n, c2d, _, _ = x2db3d.size()

        n, c2d, hh, ww = x2d_motion.shape
        feature_motion = []
        for num_bin in self.bin_numgl_motion:
            z = x2d_motion.view(n, c2d, num_bin, hh // num_bin, ww).contiguous()
            z2 = self.pool_motion(z)
            z2 = z2.view(n, c2d, num_bin)
            # z2 = torch.cat([z.view(n, c2d, num_bin), self.view_embedding_64_motion[angle].expand(-1, -1, num_bin)], 1)

            feature_motion.append(z2)

        feature_motion = torch.cat(feature_motion, 2).permute(2, 0, 1).contiguous()  # 8 n 256
        feature_motion = feature_motion.matmul(self.fc_bin_motion)  # 8 n 256
        feature_motion = feature_motion.permute(1, 2, 0).contiguous()

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            z2 = self.Gem(z).squeeze(-1)
            # z2 = torch.cat((self.Gem(z).squeeze(-1), self.view_embedding_64[angle].expand(-1, -1, num_bin)), 1)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()  # 64 n 256
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin)  # 64 n 256
        feature = feature.permute(1, 2, 0).contiguous()

        feature = torch.cat([feature_motion, feature], 2)


        embed = self.bn2(feature)  # ([B, 256, 128])
        # print("embed",embed.shape)

        logi = self.fc2(embed)  # ([B, num_classes, 128])

        # Since this may have been repeated, we need to get n,s again
        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
                # 'view_softmax': {'logits': angle_probe.unsqueeze(1), 'labels': view}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
