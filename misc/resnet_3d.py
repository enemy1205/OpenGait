import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F



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


class P3DResidualUint(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = conv1x3x3(in_planes,planes,stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1x1(planes,planes,stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x3x3(planes,planes,stride=1)
        self.bn3 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        temp = out.clone()
        out = self.conv2(out)
        out = self.bn2(out)
        out +=temp
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out
        
class Basic3DBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 1, 1),
                               padding=(conv1_t_size // 2, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def make_3d_layer(in_planes,planes,stride,blocks):
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = nn.Sequential(
                conv1x1x1(in_planes, planes, stride),
                nn.BatchNorm3d(planes))
    layers = nn.Sequential()
    layers.append(
        Basic3DBlock(in_planes=in_planes,
              planes=planes,
              stride=stride,
              downsample=downsample))
    for i in range(1, blocks):
        layers.append(Basic3DBlock(planes, planes))
    return layers

class ResNet_SP(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3):
        super().__init__()

        self.in_planes = n_input_channels

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],stride=2)
        self.layer2 = self._make_layer(block,block_inplanes[1],layers[1],stride=2)
        self.layer3 = self._make_layer(block,block_inplanes[2], layers[2])


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class ResNet_1_Stage(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3):
        super().__init__()
        self.in_planes = n_input_channels

        self.layer1 = self._make_layer(block, block_inplanes, layers,stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        res = torch.cat([F.interpolate(x[i],size=(30,20),mode='bilinear').unsqueeze(0) for i in range(x.shape[0])],0)
        return res

def generate_model_sp(model_depth,inplanes, **kwargs):
    assert model_depth in [10,14,22, 34]

    if model_depth == 10:
        model = ResNet_SP(P3DResidualUint, [ 1, 1, 1], inplanes, **kwargs)
    elif model_depth == 14:
        model = ResNet_SP(P3DResidualUint, [ 2, 2, 1], inplanes, **kwargs)
    elif model_depth == 22:
        model = ResNet_SP(P3DResidualUint, [ 4, 4, 1], inplanes, **kwargs)
    elif model_depth == 34:
        model = ResNet_SP(P3DResidualUint, [4, 6, 3], inplanes, **kwargs)
    return model


def generate_model(model_depth,inplanes, **kwargs):
    assert model_depth in [10,14, 18,22, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(Basic3DBlock, [1, 1, 1, 1], inplanes, **kwargs)
    elif model_depth == 14:
        model = ResNet(Basic3DBlock, [1, 2, 2, 1], inplanes, **kwargs)
    elif model_depth == 18:
        model = ResNet(Basic3DBlock, [2, 2, 2, 2], inplanes, **kwargs)
    elif model_depth == 22:
        model = ResNet(Basic3DBlock, [1, 4, 4, 1], inplanes, **kwargs)
    elif model_depth == 34:
        model = ResNet(Basic3DBlock, [3, 4, 6, 3], inplanes, **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck3D, [3, 4, 6, 3], inplanes, **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck3D, [3, 4, 23, 3], inplanes, **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck3D, [3, 8, 36, 3], inplanes, **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck3D, [3, 24, 36, 3], inplanes, **kwargs)

    return model


if __name__ == '__main__':
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 64 
    model  = generate_model(10,[T,2*T,4*T,8*T],n_input_channels=1)
    model.to(device=device)
    # print(model)
    from torchsummary import summary
    print(summary(model,(1,100,64,64)))