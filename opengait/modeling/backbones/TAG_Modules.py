import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os


class Bidirectional_Temporal_Self_Attention(nn.Module):
    def __init__(self, in_channels):
        super(Bidirectional_Temporal_Self_Attention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.Q1 = nn.Conv1d(
            1, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False
        )
        self.K1 = nn.Conv1d(
            1, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False
        )
        self.V1 = nn.Conv1d(
            1, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False
        )

        self.Q2 = nn.Conv1d(
            1, 1, kernel_size=5, stride=1, padding=(5 - 1) // 2, bias=False
        )
        self.K2 = nn.Conv1d(
            1, 1, kernel_size=5, stride=1, padding=(5 - 1) // 2, bias=False
        )
        self.V2 = nn.Conv1d(
            1, 1, kernel_size=5, stride=1, padding=(5 - 1) // 2, bias=False
        )

        self.Q3 = nn.Conv1d(
            1, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False
        )
        self.K3 = nn.Conv1d(
            1, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False
        )
        self.V3 = nn.Conv1d(
            1, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False
        )

        self.s = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xc = torch.mean(x, dim=1)
        y = self.avgpool(xc)
        y = y.squeeze(-1).transpose(-1, -2)

        query_fea1 = self.Q1(y).transpose(-1, -2)
        key_fea1 = self.K1(y)
        Val_fea1 = self.V1(y)

        attention1 = torch.bmm(query_fea1, key_fea1)
        attention1 = self.s(attention1)
        out1 = (
            torch.bmm(Val_fea1, attention1)
            .transpose(-1, -2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        out1 = out1.permute(0, 2, 1, 3, 4)
        out1 = self.sigmoid(out1)
        x1 = x * out1.expand_as(x)

        query_fea2 = self.Q2(y).transpose(-1, -2)
        key_fea2 = self.K2(y)
        Val_fea2 = self.V2(y)

        attention2 = torch.bmm(query_fea2, key_fea2)
        attention2 = self.s(attention2)
        out2 = (
            torch.bmm(Val_fea2, attention2)
            .transpose(-1, -2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        out2 = out2.permute(0, 2, 1, 3, 4)
        out2 = self.sigmoid(out2)
        x2 = x * out2.expand_as(x)

        query_fea3 = self.Q3(y).transpose(-1, -2)
        key_fea3 = self.K3(y)
        Val_fea3 = self.V3(y)

        attention3 = torch.bmm(query_fea3, key_fea3)
        attention3 = self.s(attention3)
        out3 = (
            torch.bmm(Val_fea3, attention3)
            .transpose(-1, -2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        out3 = out3.permute(0, 2, 1, 3, 4)
        out3 = self.sigmoid(out3)
        x3 = x * out3.expand_as(x)

        return x1 + x2 + x3


################################################################################################################
def ST_feature(in_channels, out_channels, kernel_size, **kwargs):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs),
        nn.BatchNorm3d(out_channels),
    )


class Short_term_Temporal_Feature_V2(nn.Module):
    def __init__(self, in_channels):
        super(Short_term_Temporal_Feature_V2, self).__init__()

        self.ST = nn.ModuleList(
            [
                ST_feature(
                    in_channels, in_channels, (3, 1, 1), stride=1, padding=(1, 0, 0)
                ),
                ST_feature(
                    in_channels, in_channels, (3, 1, 1), stride=1, padding=(1, 0, 0)
                ),
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, t, h, w = x.size()

        temp = self.ST[0](x)
        ST_fea = temp + self.ST[1](temp)
        ST_attn = self.sigmoid(ST_fea)

        return ST_attn * x


##################################################################################################################
class AttentionModule_3D_Temporal_MK_V2_New(nn.Module):
    def __init__(self, in_channels, kernel_size=[13, 21, 31], dilation=[3, 5, 7]):
        super(AttentionModule_3D_Temporal_MK_V2_New, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        d_k0 = 2 * dilation[0] - 1
        d_p0 = (d_k0 - 1) // 2
        dd_k0 = kernel_size[0] // dilation[0] + (
            (kernel_size[0] // dilation[0]) % 2 - 1
        )
        dd_p0 = dilation[0] * (dd_k0 - 1) // 2

        d_k1 = 2 * dilation[1] - 1
        d_p1 = (d_k1 - 1) // 2
        dd_k1 = kernel_size[1] // dilation[1] + (
            (kernel_size[1] // dilation[1]) % 2 - 1
        )
        dd_p1 = dilation[1] * (dd_k1 - 1) // 2

        d_k2 = 2 * dilation[2] - 1
        d_p2 = (d_k2 - 1) // 2
        dd_k2 = kernel_size[2] // dilation[2] + (
            (kernel_size[2] // dilation[2]) % 2 - 1
        )
        dd_p2 = dilation[2] * (dd_k2 - 1) // 2

        # self.conv_spatio = nn.Conv3d(in_channels, in_channels, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1), bias = False)
        # self.bns = nn.BatchNorm3d(in_channels)
        # self.relu = nn.ReLU(inplace=True)

        self.conv01 = nn.Conv3d(2, 1, (d_k0, 1, 1), padding=(d_p0, 0, 0), groups=1)
        self.conv_spatial01 = nn.Conv3d(
            1,
            1,
            (dd_k0, 1, 1),
            stride=1,
            padding=(dd_p0, 0, 0),
            groups=1,
            dilation=dilation[0],
        )
        # self.conv1 = nn.Conv3d(1, in_channels, 1)

        self.conv02 = nn.Conv3d(2, 1, (d_k1, 1, 1), padding=(d_p1, 0, 0), groups=1)
        self.conv_spatial02 = nn.Conv3d(
            1,
            1,
            (dd_k1, 1, 1),
            stride=1,
            padding=(dd_p1, 0, 0),
            groups=1,
            dilation=dilation[1],
        )
        # self.conv2 = nn.Conv3d(1, in_channels, 1)

        self.conv03 = nn.Conv3d(2, 1, (d_k2, 1, 1), padding=(d_p2, 0, 0), groups=1)
        self.conv_spatial03 = nn.Conv3d(
            1,
            1,
            (dd_k2, 1, 1),
            stride=1,
            padding=(dd_p2, 0, 0),
            groups=1,
            dilation=dilation[2],
        )
        # self.conv3 = nn.Conv3d(1, in_channels, 1)

    def forward(self, x):
        u = x.clone()
        b, c, t, h, w = x.size()
        identity = x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)

        attn = self.conv01(x1)  # depth-wise conv  #([32, 1, 30, 64, 22])
        attn = self.conv_spatial01(attn)
        f_g1 = attn.expand(b, c, t, h, w)
        # print("f_g1",f_g1.shape)

        # f_g1 = self.conv1(attn)  # ([32, 128, 30, 64, 22])

        attn2 = self.conv02(x1)  # depth-wise conv  #([32, 1, 30, 64, 22])
        attn2 = self.conv_spatial02(attn2)
        f_g2 = attn2.expand(b, c, t, h, w)
        # print("f_g2", f_g2.shape)
        # f_g2 = self.conv2(attn2)  # ([32, 128, 30, 64, 22])

        attn3 = self.conv03(x1)  # depth-wise conv  #([32, 1, 30, 64, 22])
        attn3 = self.conv_spatial03(attn3)
        f_g3 = attn3.expand(b, c, t, h, w)
        # print("f_g3", f_g3.shape)
        # f_g3 = self.conv3(attn3)  # ([32, 128, 30, 64, 22])

        f_g123 = torch.sigmoid(f_g1 + f_g2 + f_g3) * identity

        return f_g123


class Part_Level_Temporal_Attention_GL(nn.Module):
    def __init__(self, in_channels, kernel_size=[13, 21, 31], dilation=[3, 5, 7]):
        super(Part_Level_Temporal_Attention_GL, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.PTA = AttentionModule_3D_Temporal_MK_V2_New(
            in_channels, kernel_size=[13, 21, 31], dilation=[3, 5, 7]
        )

    def forward(self, x):
        u = x.clone()
        b, c, t, h, w = x.size()
        identity = x
        h = x.size(3)

        split_size = int(h // 2**2)
        XL = x.split(split_size, 3)  ##############[B,C,T,H/4,W]

        lcl_TA = F.leaky_relu(torch.cat([self.PTA(_) for _ in XL], 3))
        gl_TA = self.PTA(x)

        TF_GL = lcl_TA + gl_TA

        return TF_GL


#####################################################################################################################
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Temporal_Mixer_V2(nn.Module):
    def __init__(self, in_channels):
        super(Temporal_Mixer_V2, self).__init__()

        self.convDW1 = DepthwiseSeparableConv(
            in_channels, in_channels, kernel_size=(31, 1, 1), padding=(15, 0, 0)
        )
        self.convDW2 = DepthwiseSeparableConv(
            in_channels, in_channels, kernel_size=(13, 1, 1), padding=(6, 0, 0)
        )
        self.act = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channels, eps=0.001)

    def forward(self, x):
        residual = x

        x1 = self.convDW1(x)
        x1 = self.act(x1)
        x1 = self.bn(x1)

        x2 = self.convDW2(x)
        x2 = self.act(x2)
        x2 = self.bn(x2)

        f_x12 = torch.sigmoid(x1 + x2) * residual

        return f_x12
