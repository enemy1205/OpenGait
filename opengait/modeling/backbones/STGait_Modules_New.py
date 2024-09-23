import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

#####################################################################################################################
def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'LeakyReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims//2, 1, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ElementScale_G(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale_G, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

    """A learnable element-wise scaler."""


#######################################################################################################################
class Local_SKTA_FD(nn.Module):
    def __init__(self, in_channels, kernel_size=13, dilation=3):
        super(Local_SKTA_FD, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        d_k0 = 2 * dilation - 1
        d_p0 = (d_k0 - 1) // 2

        self.DWConv = nn.Conv3d(in_channels, in_channels, (d_k0, 1, 1), padding=(d_p0, 0, 0), groups=in_channels)
        self.PWConv = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        attn = self.DWConv(x)  # depth-wise conv  #([32, 1, 30, 64, 22])
        attn = self.PWConv(attn)

        return attn

class Local_SKSA_FD(nn.Module):
    def __init__(self, in_channels, kernel_size=13, dilation=3):
        super(Local_SKSA_FD, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        d_k0 = 2 * dilation - 1
        d_p0 = (d_k0 - 1) // 2

        self.DWConv = nn.Conv3d(in_channels, in_channels, (1, d_k0, d_k0), padding=(0, d_p0, d_p0), groups=in_channels)
        self.PWConv = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        attn = self.DWConv(x)  # depth-wise conv  #([32, 1, 30, 64, 22])
        attn = self.PWConv(attn)

        return attn
#######################################################################################################################
class Dynamic_TF(nn.Module):
    def __init__(self, in_channels, kernel_size=13):
        super(Dynamic_TF, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.TF = nn.Sequential(nn.Conv3d(in_channels, in_channels, (3, 1, 1), padding=(1, 0, 0), groups=in_channels),
                                nn.Conv3d(in_channels, in_channels, 1)
                                )

    def forward(self, x):

        x_m = x.mean(dim = 2, keepdim = True)
        x_mt = x - x_m

        x_t = self.TF(x)
        x_mt = self.TF(x_mt)

        x_f = x_t + x_mt

        return x_f

class Dynamic_SF(nn.Module):
    def __init__(self, in_channels, kernel_size=13):
        super(Dynamic_SF, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.SF = nn.Sequential(nn.Conv3d(in_channels, in_channels, (1, 3, 3), padding=(0, 1, 1), groups=in_channels),
                                nn.Conv3d(in_channels, in_channels, 1)
                                )

    def forward(self, x):
        x_m = x.mean(dim=2, keepdim=True)
        x_mt = x - x_m

        x_s = self.SF(x)
        x_ms = self.SF(x_mt)

        x_f = x_s + x_ms

        return x_f

#######################################################################################################################

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()

        self.fc1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        # attention_weights = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        x_c = F.adaptive_avg_pool3d(x, output_size=1)
        attention_weights = F.relu(self.fc1(x_c))
        attention_weights = torch.sigmoid(self.fc2(attention_weights))

        return attention_weights

class TAFL(nn.Module):
    def __init__(self, in_channels, attn_act_type = 'LeakyReLU', attn_force_fp32=False, attn_aggregation=True):
        super(TAFL, self).__init__()

        self.gate = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.act_value = build_act_layer(attn_act_type)
        self.value = Local_SKTA_FD(in_channels)
        self.conv1 = nn.Conv3d(in_channels,in_channels,kernel_size=1)
        self.proj_2 = nn.Conv3d(in_channels,in_channels,kernel_size=1)
        self.attn_force_fp32 = attn_force_fp32
        self.sigma = ElementScale_G(
            in_channels, init_value=1e-5, requires_grad=True)
        self.act_gate = build_act_layer(attn_act_type)
        self.attn_aggregation = attn_aggregation
        self.attention = Attention(in_channels)

    def feature_decompose_C(self, x):
        # b,c,t,h,w = x.size()

        x = self.conv1(x)
        x_c = F.adaptive_avg_pool3d(x, output_size=1)
        x = x + self.sigma(x - x_c)
        x = self.act_value(x)
        return x

    # def feature_decompose_T(self, x):
    #     # b,c,t,h,w = x.size()
    #
    #     x = self.conv1(x)
    #     x_t = F.adaptive_avg_pool3d(x, output_size=2)
    #     x = x + self.sigma(x - x_t)
    #     x = self.act_value(x)
    #     return x

    # def forward_gating(self, g, v):
    #     with torch.autocast(device_type='cuda', enabled=False):
    #         g = g.to(torch.float32)
    #         v = v.to(torch.float32)
    #         return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self,x):
        shortcut = x.clone()

        b,c,t,h,w = x.size()

        h = x.size(3)
        split_size = int(h // 2 ** 1)
        lcl_feat = x.split(split_size, 3)

        x = self.feature_decompose_C(x)
        g = self.gate(x)
        v = torch.cat([self.value(_) for _ in lcl_feat], 3)
        # v = self.value(x)

        if self.attn_aggregation:
            w_g = self.attention(g) * g
            w_v = self.attention(v) * v
            #
            x = self.proj_2(self.act_gate(w_g) * self.act_gate(w_v))
        else:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
            # x = self.forward_gating(self.act_gate(g), self.act_gate(v))

        x = x + shortcut

        return x

class SAFL(nn.Module):
    def __init__(self, in_channels, attn_act_type = 'LeakyReLU', attn_force_fp32=False, attn_aggregation=True):
        super(SAFL, self).__init__()

        self.gate = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.act_value = build_act_layer(attn_act_type)
        self.value = Local_SKSA_FD(in_channels)
        self.conv1 = nn.Conv3d(in_channels,in_channels,kernel_size=1)
        self.proj_2 = nn.Conv3d(in_channels,in_channels,kernel_size=1)
        self.attn_force_fp32 = attn_force_fp32
        self.sigma = ElementScale(
            in_channels, init_value=1e-5, requires_grad=True)
        self.act_gate = build_act_layer(attn_act_type)
        self.attn_aggregation = attn_aggregation
        self.attention = Attention(in_channels)

    def feature_decompose_C(self, x):
        # b,c,t,h,w = x.size()

        x = self.conv1(x)
        x_c = F.adaptive_avg_pool3d(x, output_size=1)
        x = x + self.sigma(x - x_c)
        x = self.act_value(x)
        return x

    def forward(self,x):
        shortcut = x.clone()

        b,c,t,h,w = x.size()

        h = x.size(3)
        split_size = int(h // 2 ** 1)
        lcl_feat = x.split(split_size, 3)

        x = self.feature_decompose_C(x)
        g = self.gate(x)
        v = torch.cat([self.value(_) for _ in lcl_feat], 3)
        # v = self.value(x)

        if self.attn_aggregation:
            w_g = self.attention(g) * g
            w_v = self.attention(v) * v
            #
            x = self.proj_2(self.act_gate(w_g) * self.act_gate(w_v))
        else:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
            # x = self.forward_gating(self.act_gate(g), self.act_gate(v))

        x = x + shortcut

        return x
####################################################################################################################
class GTAFL(nn.Module):
    def __init__(self, in_channels, num_groups = 2, attn_act_type = 'LeakyReLU', attn_force_fp32=False, attn_aggregation=True):
        super(GTAFL, self).__init__()

        self.num_groups = num_groups
        assert in_channels % num_groups == 0

        self.gate = nn.Conv3d(in_channels//2, in_channels//2, kernel_size=1)
        self.act_value = build_act_layer(attn_act_type)
        self.value1 = Local_SKTA_FD(in_channels // 2)
        self.value2 = Local_SKSA_FD(in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels//2,in_channels//2,kernel_size=1)
        self.proj_2 = nn.Conv3d(in_channels//2,in_channels//2,kernel_size=1)
        self.attn_force_fp32 = attn_force_fp32
        self.sigma = ElementScale(
            in_channels, init_value=1e-5, requires_grad=True)
        self.act_gate = build_act_layer(attn_act_type)
        self.attn_aggregation = attn_aggregation
        self.attention = Attention(in_channels//2)

    def feature_decompose_C(self, x):

        x = self.conv1(x)
        x_c = F.adaptive_avg_pool3d(x, output_size=1)
        x = x + self.sigma(x - x_c)
        x = self.act_value(x)
        return x

    def forward(self,x):
        shortcut = x.clone()

        groups = torch.chunk(x, self.num_groups, dim=1)

        x1 = groups[0]
        x2 = groups[1]

        h1 = x1.size(3)
        split_size = int(h1 // 2 ** 1)
        lcl_feat_g1 = x1.split(split_size, 3)

        x1 = self.feature_decompose_C(x1)
        g1 = self.gate(x1)
        v1 = torch.cat([self.value1(_) for _ in lcl_feat_g1], 3)

        h2 = x2.size(3)
        split_size = int(h2 // 2 ** 1)
        lcl_feat_g2 = x2.split(split_size, 3)

        x2 = self.feature_decompose_C(x2)
        g2 = self.gate(x2)
        v2 = torch.cat([self.value2(_) for _ in lcl_feat_g2], 3)
        # v = self.value(x)

        if self.attn_aggregation:
            w_g1 = self.attention(g1) * g1
            w_v1 = self.attention(v1) * v1
            x1 = self.proj_2(self.act_gate(w_g1) * self.act_gate(w_v1))

            w_g2 = self.attention(g2) * g2
            w_v2 = self.attention(v2) * v2
            x2 = self.proj_2(self.act_gate(w_g2) * self.act_gate(w_v2))
            xc = torch.cat((x1, x2), dim = 1)

        else:
            x1 = self.proj_2(self.act_gate(g1) * self.act_gate(v1))
            x2 = self.proj_2(self.act_gate(g2) * self.act_gate(v2))
            xc = torch.cat((x1, x2), dim=1)

        x = xc + shortcut

        return x

class GDTAFL(nn.Module):
    def __init__(self, in_channels, num_groups=2, attn_act_type='LeakyReLU', attn_force_fp32=False, attn_aggregation=True):
        super(GDTAFL, self).__init__()

        self.num_groups = num_groups
        assert in_channels % num_groups == 0

        self.gate = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.act_value = build_act_layer(attn_act_type)
        self.value1 = Dynamic_TF(in_channels // 2)
        self.value2 = Dynamic_SF(in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.proj_2 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.attn_force_fp32 = attn_force_fp32
        self.sigma = ElementScale(
            in_channels, init_value=1e-5, requires_grad=True)
        self.act_gate = build_act_layer(attn_act_type)
        self.attn_aggregation = attn_aggregation
        self.attention = Attention(in_channels // 2)

    def feature_decompose_C(self, x):

        x = self.conv1(x)
        x_c = F.adaptive_avg_pool3d(x, output_size=1)
        x = x + self.sigma(x - x_c)
        x = self.act_value(x)
        return x

    def forward(self, x):
        shortcut = x.clone()

        groups = torch.chunk(x, self.num_groups, dim=1)

        x1 = groups[0]
        x2 = groups[1]

        h1 = x1.size(3)
        split_size = int(h1 // 2 ** 1)
        lcl_feat_g1 = x1.split(split_size, 3)

        x1 = self.feature_decompose_C(x1)
        g1 = self.gate(x1)
        v1 = torch.cat([self.value1(_) for _ in lcl_feat_g1], 3)

        h2 = x2.size(3)
        split_size = int(h2 // 2 ** 1)
        lcl_feat_g2 = x2.split(split_size, 3)

        x2 = self.feature_decompose_C(x2)
        g2 = self.gate(x2)
        v2 = torch.cat([self.value2(_) for _ in lcl_feat_g2], 3)
        # v = self.value(x)

        if self.attn_aggregation:
            w_g1 = self.attention(g1) * g1
            w_v1 = self.attention(v1) * v1
            x1 = self.proj_2(self.act_gate(w_g1) * self.act_gate(w_v1))

            w_g2 = self.attention(g2) * g2
            w_v2 = self.attention(v2) * v2
            x2 = self.proj_2(self.act_gate(w_g2) * self.act_gate(w_v2))
            xc = torch.cat((x1, x2), dim=1)

        else:
            x1 = self.proj_2(self.act_gate(g1) * self.act_gate(v1))
            x2 = self.proj_2(self.act_gate(g2) * self.act_gate(v2))
            xc = torch.cat((x1, x2), dim=1)

        x = xc + shortcut

        return x

#############################################################################################################
class SCGait_SCGTST_AA(nn.Module):
    def __init__(self,in_channels, num_groups=2, attn_agg = True):
        super(SCGait_SCGTST_AA,self).__init__()

        self.num_groups = num_groups
        assert in_channels % num_groups == 0
        self.attn_agg = attn_agg

        self.conv = nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.TG = nn.Sequential(nn.Conv3d(in_channels//2, in_channels//2, kernel_size = (3,1,1), stride=1, padding=(1,0,0), dilation=1, groups=in_channels//2, bias=False),
                                nn.BatchNorm3d(in_channels // 2),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv3d(in_channels//2,in_channels//2,kernel_size=1),
                                nn.BatchNorm3d(in_channels//2),
                                nn.LeakyReLU(inplace=True))


        self.SG = nn.Sequential(nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1, groups=in_channels // 2, bias=False),
                                 nn.BatchNorm3d(in_channels // 2),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=1),
                                 nn.BatchNorm3d(in_channels // 2),
                                 nn.LeakyReLU(inplace=True))


        self.Cross_Att = Spatio_Temporal_Cross_Attention(in_channels, in_channels)

    def forward(self,x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        gs = x.size(1) // self.num_groups
        groups = torch.chunk(x, self.num_groups, dim=1)

        w_t = torch.sigmoid(torch.add(groups[0], self.TG(groups[0])))
        f_t = w_t * self.TG(groups[0])

        w_st = torch.sigmoid(torch.add(groups[1], self.SG(groups[1])))
        f_st = w_st * self.SG(groups[1])

        f_tst = self.Cross_Att(f_t,f_st)

        f_stt = torch.cat((f_tst,f_st),dim=1)

        return f_stt

#####################################################################################################################
class Spatio_Temporal_Cross_Attention(nn.Module):
    def __init__(self, temp_fea, stemp_fea):
        super(Spatio_Temporal_Cross_Attention, self).__init__()

        self.avgpool_t = nn.AdaptiveAvgPool2d(1)
        self.avgpool_st = nn.AdaptiveAvgPool2d(1)

        self.Q1 = nn.Conv1d(1, 1, kernel_size=1)
        self.K1 = nn.Conv1d(1, 1, kernel_size=1)
        self.V1 = nn.Conv1d(1, 1, kernel_size=1)

        self.s = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, temp_fea, stemp_fea):

        tf = torch.mean(temp_fea, dim=1)
        stf = torch.mean(stemp_fea, dim=1)

        z0 = self.avgpool_t(tf)
        z0 = z0.squeeze(-1).transpose(-1, -2)

        z1 = self.avgpool_st(stf)
        z1 = z1.squeeze(-1).transpose(-1, -2)

        query_fea1 = self.Q1(z0).transpose(-1, -2)
        key_fea1 = self.K1(z1)
        Val_fea1 = self.V1(z1)

        attention1 = torch.bmm(query_fea1, key_fea1)
        attention1 = self.s(attention1)
        out1 = torch.bmm(Val_fea1, attention1).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        out1 = out1.permute(0, 2, 1, 3, 4)
        out1 = self.sigmoid(out1)
        temp_fea_F = temp_fea * out1.expand_as(temp_fea)

        return temp_fea_F
########################################################################################################################
