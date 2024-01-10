import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from swin_transformer_3d import SwinTransformer3D
from resnet_3d import ResNet_1_Stage,P3DResidualUint,Basic3DBlock
from thop import profile

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


#-------------------------resnet-3d---------------------

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=(1,stride,stride),
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=(1,stride,stride),
                     bias=False)
    
def conv1x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(1,3,3),
                     stride=(1,stride,stride),
                     padding=(0,1,1),
                     bias=False)

def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(3,1,1),
                     stride=(1,stride,stride),
                     padding=(1,0,0),
                     bias=False)


# class BasicBlock3d(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()

#         self.conv1 = conv3x3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# def make_layer(in_planes,planes,stride,blocks):
#     downsample = None
#     if stride != 1 or in_planes != planes:
#         downsample = nn.Sequential(
#                 conv1x1x1(in_planes, planes, stride),
#                 nn.BatchNorm3d(planes))
#     layers = nn.Sequential()
#     layers.append(
#         BasicBlock3d(in_planes=in_planes,
#               planes=planes,
#               stride=stride,
#               downsample=downsample))
#     for i in range(1, blocks):
#         layers.append(BasicBlock3d(planes, planes))
#     return layers


#-------------------------resnet-2d---------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def make_layer(in_planes,planes,stride,blocks):
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                nn.BatchNorm2d(planes))
    layers = nn.Sequential()
    layers.append(
        BasicBlock(inplanes=in_planes,
              planes=planes,
              stride=stride,
              downsample=downsample))
    for i in range(1, blocks):
        layers.append(BasicBlock(planes, planes))
    return layers


#---------------------p3d net------------------------

# def conv_S(in_planes,out_planes,stride=1,padding=1):
#     # as is descriped, conv S is 1x3x3
#     return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=(1,stride,stride),
#                      padding=padding,bias=False)

# def conv_T(in_planes,out_planes,stride=1,padding=1):
#     # conv T is 3x1x1
#     return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=stride,
#                      padding=padding,bias=False)

# def downsample_basic_block(x, planes, stride):
#     out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#     zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
#                              out.size(2), out.size(3),
#                              out.size(4)).zero_()
#     if isinstance(out.data, torch.cuda.FloatTensor):
#         zero_pads = zero_pads.cuda()

#     out = Variable(torch.cat([out.data, zero_pads], dim=1))

#     return out

# # according to deepgait_v2_p3d
# class BottleBlock(nn.Module):

#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#         self.downsample = downsample
#         self.conv1 = conv_S(in_planes, planes,stride=stride,padding=(0,1,1))
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = conv_S(planes,planes, stride=1,padding=(0,1,1))
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = conv_T(planes,planes, stride=1,padding=(1,0,0))
#         self.bn3 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)

#     def ST_D(self,x):
#         temp = x.clone()
#         out = self.conv3(x)
#         out = self.bn3(out)
#         out = self.relu(out)

#         out = self.conv2(out+temp)
#         out = self.bn2(out)
#         out = self.relu(out)

#         return out
    
#     def forward(self,x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out =self.ST_D(out)
        
#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out =out+ residual
#         out = self.relu(out)
#         return out

# def make_layer(in_planes,planes,stride,blocks):
#     downsample = None
#     if stride != 1 or in_planes != planes:
#         downsample = nn.Sequential(
#             nn.Conv3d(in_planes,planes,kernel_size=1,stride=(1,stride,stride),bias=False),
#                 nn.BatchNorm3d(planes))
#     layers = nn.Sequential()
#     layers.append(
#         BottleBlock(in_planes=in_planes,
#               planes=planes,
#               stride=stride,
#               downsample=downsample))
#     for i in range(1, blocks):
#         layers.append(BottleBlock(planes, planes))
#     return layers

class TestModel(nn.Module):
    def __init__(self) -> None:
        super(TestModel,self).__init__()
        self.in_planes = 64
        self.backbone = nn.Sequential(
            BasicConv2d(1, self.in_planes, 3, 1, 1),
            nn.BatchNorm2d(self.in_planes),
            nn.LeakyReLU(inplace=True),
            BasicBlock(self.in_planes,self.in_planes,stride=1)
        )
        
        T=self.in_planes
        self.backbone.append(make_layer(T,2*T,2,4))
        self.backbone.append(make_layer(2*T,4*T,2,4))
        self.backbone.append(make_layer(4*T,8*T,1,1))
        self.backbone = nn.Sequential(SetBlockWrapper(self.backbone))
    
    def forward(self, sils):
        outs = self.backbone(sils)
        return outs


class SwinGait3D(nn.Module):
    def __init__(self) -> None:
        super(SwinGait3D,self).__init__()
        self.in_planes = 64
        self.conv_block = nn.Sequential(
            BasicConv2d(1, self.in_planes, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.backbone = nn.Sequential(
            SetBlockWrapper(self.conv_block),
            SetBlockWrapper(BasicBlock(self.in_planes,self.in_planes,stride=1)),
            ResNet_1_Stage(Basic3DBlock,2,2*self.in_planes,self.in_planes)
        )
        self.swin3d=SwinTransformer3D(in_chans=128,patch_size=(2,2,2),window_size=(3,3,5),embed_dim=128,depths=[2,2],num_heads=[4,4])

    def forward(self, sils):
        outs = self.backbone(sils)
        outs = self.swin3d(outs)
        return outs


if __name__ == '__main__':
    net = TestModel().cuda()
    dummy_input = torch.randn(1, 1,30,64, 44).cuda()
    flops, params = profile(net, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))