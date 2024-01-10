''' Spatial-Temporal Transformer Networks
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .spectral_norm import spectral_norm as _spectral_norm



class BinaryActivation(nn.Module):
    def __init__(self, threshold=0.):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x):
        out = torch.where(x>self.threshold, torch.ones_like(x), torch.zeros_like(x))
        return out
    
    def backward(self, grad_output):
        # 反向传播时不执行二值化
        grad_input = grad_output.clone() 
        return grad_input

class ThresholdBinary(torch.nn.Module):
    def __init__(self, threshold):
        super(ThresholdBinary, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        sigmoid_output = torch.sigmoid(x)
        binary_output = torch.where(sigmoid_output >= self.threshold, torch.tensor(1), torch.tensor(0))
        return binary_output


class InpaintGenerator(nn.Module):
    def __init__(self,model_cfg):
        super(InpaintGenerator, self).__init__()
        # the default param reference
        if model_cfg is None:
            channel = 256
            stack_num = 4
            patchsize = [(4,4),(2,2)]
        else:
            channel = model_cfg['channel']
            stack_num = model_cfg['stack_num']
            patchsize = model_cfg['patchsize']
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)

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
        _, c, h, w = enc_feat.size()
        # masks = F.interpolate(masks, scale_factor=1.0/4)
        enc_feat = self.transformer(
            {'x': enc_feat,'b': b, 'c': c})['x']
        output = self.decoder(enc_feat)
        # return output+masks
        return output
        
    def infer(self, feat, masks):
        t, c, h, w = masks.size()
        masks = masks.view(t, c, h, w)
        masks = F.interpolate(masks, scale_factor=1.0/4)
        t, c, _, _ = feat.size()
        enc_feat = self.transformer(
            {'x': feat, 'b': 1, 'c': c})['x']
        return enc_feat


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, b, c = x['x'], x['b'], x['c']
        x = x + self.attention(x, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}


# ######################################################################
# ######################################################################


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
    
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    