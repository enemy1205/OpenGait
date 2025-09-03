import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks, SeparateFCs
from einops import rearrange

from ..base_model import BaseModel
from data.transform import TubeMaskingGenerator
from utils import (
    get_valid_args,
    is_list,
    is_dict,
    np2var,
    ts2np,
    list2var,
    get_attr_from,
)

def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class DeepGaitV2_MAE(BaseModel):
    def build_network(self, model_cfg):
        # 获取配置参数
        img_size = model_cfg["img_size"]
        self.patch_size = model_cfg["patch_size"]
        in_chans = model_cfg.get("in_chans", 1)
        embed_dim = model_cfg.get("embed_dim", 512)
        encoder_depth = model_cfg.get("encoder_depth", 2)
        num_heads = model_cfg.get("num_heads", 4)
        self.tubelet_size = model_cfg.get("tubelet_size", 2)

        self.inference_use_emb2 = model_cfg.get("use_emb2", False)

        # Patch Embedding 层
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=30,
            tubelet_size=self.tubelet_size,
        )
        num_patches = self.patch_embed.num_patches

        # Positional Encoding
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        # Encoder Blocks
        dpr = [x.item() for x in torch.linspace(0, 0.1, encoder_depth)]
        self.encoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., drop=0., drop_path=dpr[i])
            for i in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder 相关（仅训练时用）
        decoder_embed_dim = model_cfg.get("decoder_embed_dim", 512)
        decoder_depth = model_cfg.get("decoder_depth", 2)

        self.decoder_to_latent = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=4., drop=0., drop_path=dpr[i])
            for i in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.tubelet_size*(self.patch_size[0] * self.patch_size[1]) * in_chans, bias=True)
        # 分类相关模块
        class_num = 64

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.FCs = SeparateFCs(16, embed_dim, 512)
        self.BNNecks = SeparateBNNecks(16, 512, class_num=class_num)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()

        B, C, T, H, W = sils.shape

        # Patch Embedding
        x = self.patch_embed(sils)  # shape: [B*T, num_patches, embed_dim]

        # 添加 position embedding
        x = x + self.pos_embed.type_as(x).to(x.device).detach()

        # 随机生成 mask（每个视频帧独立随机遮盖）
        mask_generator = TubeMaskingGenerator((T//self.tubelet_size, H // self.patch_size[0], W // self.patch_size[1]), mask_ratio=0.25)
        mask = torch.from_numpy(mask_generator(B)).bool().to(x.device)  # shape: [B*T, num_patches]

        # 将被遮盖位置替换为 mask_token
        x_vis = x[~mask].reshape(B, -1, x.shape[-1])  # 可见部分
        mask_num = mask.sum(dim=-1)  # 每个样本的遮盖数量
        x_mask = self.mask_token.expand(B, mask_num.max(), -1)  # 扩展到最大遮盖数
        x_full = torch.cat([x_vis, x_mask], dim=1)

        # 加上 position embedding（注意要复制 mask token 的 pos_emb）
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, x.shape[-1])
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, x.shape[-1])
        x_full = x_full + torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        # Encoder
        for blk in self.encoder_blocks:
            x_full = blk(x_full)
        x_full = self.encoder_norm(x_full)

        # 提取 cls token 或所有可见 token 用于身份识别
        feat = x_full[:, :x_vis.shape[1]]  # 只取可见 token
        feat = feat.mean(dim=1)  # 平均池化
        feat = feat.view(B, feat.shape[-1], 1)  # [B, embed_dim, 1]

        # Temporal Pooling
        outs = self.TP(feat.unsqueeze(-1), seqL, options={"dim": 2})[0]  # [B, embed_dim, 1]

        # Horizontal Pooling Matching
        feat = self.HPP(outs)  # [B, embed_dim, p]

        embed_1 = self.FCs(feat)  # [B, embed_dim, p]
        embed_2, logits = self.BNNecks(embed_1)  # [B, embed_dim, p], [B, class_num, p]

        embed = embed_1 if not self.inference_use_emb2 else embed_2

        retval = {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {
                "image/sils": rearrange(sils, "(b t) c h w -> b c t h w", b=B, t=T),
            },
            "inference_feat": {"embeddings": embed},
        }

        # 如果是训练阶段，则启用 decoder 做图像重建
        if self.training:
            # 解码器输入
            x_dec = self.decoder_to_latent(x_full)

            # 解码器处理
            for blk in self.decoder_blocks:
                x_dec = blk(x_dec)
            x_dec = self.decoder_norm(x_dec)

            # 预测被遮盖的 patch
            pred_x = self.decoder_pred(x_dec)  # [B, N, P*p*C]

            # 构建 target（真实 patch）
            with torch.no_grad():
                target_x = sils.unfold(3, self.patch_size[0], self.patch_size[0]).unfold(4, self.patch_size[1], self.patch_size[1])  # [B, C, T, H/p1, W/p2, p1, p2]
    
                # Step 2: 在时间维度上划分 tubelet
                target_x = target_x.unfold(2, self.tubelet_size, self.tubelet_size)  # [B, C, T/tubelet_size, H/p1, W/p2, p1, p2]

                # Step 3: 展平
                target_x = target_x.reshape(B, -1, self.tubelet_size * self.patch_size[0] * self.patch_size[1] * C)

            # 只计算被遮盖的部分
            pred_x_mask = pred_x[:, -mask.sum(dim=-1).min():]
            target_x_mask = target_x[mask].reshape(pred_x_mask.shape[0], -1, pred_x_mask.shape[-1])

            # 添加 reconstruction loss 到 training_feat
            retval["training_feat"]["reconstruction"] = {
                "pred": pred_x_mask,
                "target": target_x_mask,
            }

        return retval