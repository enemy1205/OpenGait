''' Spatial-Temporal Transformer Networks
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .spectral_norm import spectral_norm as _spectral_norm
from .sparse_transformer import TemporalSparseTransformerBlock,SoftSplit,SoftComp

class InpaintGenerator(nn.Module):
    def __init__(self,model_cfg):
        super(InpaintGenerator, self).__init__()
        # the default param reference
        if model_cfg is None:
            channel = 256
            hidden = 512
        else:
            channel = model_cfg['channel']
            hidden = model_cfg['hidden']
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.transformer = TemporalSparseTransformerBlock(dim=hidden,n_head=1,window_size=(5,9),pool_size=(4,4),depths=8,t2t_params=t2t_params)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, masks):
        if len(masks.size()) == 4:
            masks = masks.unsqueeze(2)
        # extracting features
        b, t, c, h, w = masks.size()
        masks = masks.view(b*t, c, h, w)
        enc_feat = self.encoder(masks)
        fold_feat_size = (enc_feat.size()[2], enc_feat.size()[3])
        trans_feat = self.ss(enc_feat, b, fold_feat_size)
        trans_feat = self.transformer(trans_feat, fold_feat_size)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        output = self.decoder(trans_feat)
        # return output+masks
        return output


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self,model_cfg, use_sigmoid=False, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        if model_cfg is None:
            in_channels = 1
            nf = 64.
            self.use_sigmoid = use_sigmoid
        else:
            in_channels = model_cfg['input_c']
            nf = model_cfg['hidden_size']
        self.use_sigmoid = model_cfg['use_sigmoid'] if model_cfg['use_sigmoid'] is not None else use_sigmoid
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out
    


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


class Inpainter(nn.Module):
    def __init__(self,model_cfg):
        super(Inpainter, self).__init__()
        self.netGen = InpaintGenerator(model_cfg['Gen'])
        self.netDis = Discriminator(model_cfg['Dis'],use_sigmoid=True)
        self._relu = torch.nn.ReLU1(True)

    
    def forward(self,gt_sils,occ_sils):
        b, t, c, h, w = occ_sils.size()
        # NetG input : [b,t,c,h,w]
        recovered_sils = self.netGen(occ_sils)
        recovered_sils = self._relu(recovered_sils)
        
        recovered_sils = recovered_sils.view(b*t, c, h, w)
        gt_sils = gt_sils.view(b*t, c, h, w)
        
        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(recovered_sils.detach())
  
        Tensor = torch.cuda.FloatTensor
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((gt_sils.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * gt_sils + ((1 - alpha) * recovered_sils)).requires_grad_(True)
        d_interpolates = self.netDis(interpolates)
                
        gen_vid_feat = self.netDis(recovered_sils)
        
        return recovered_sils,real_sils_embs,fake_sils_embs,interpolates,d_interpolates,gen_vid_feat
    

