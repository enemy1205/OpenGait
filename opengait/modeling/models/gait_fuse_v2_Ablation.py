from einops import rearrange
from ..base_model import BaseModel
from ..backbones.fuse_former import TransformerBlock,SoftSplit,AddPosEmb,SoftComp,Encoder
from utils import get_valid_args, get_attr_from
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
import torch
import torch.nn as nn
import torch.optim as optim
    
blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class FuseGait_Ablation(BaseModel):
    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(1)
        
        n, c, s, h, w = sils.size()
        
        assert sils.size(-1) in [44, 88]

        del ipts        
        
        out0 = self.layer0(sils) # [b,64,t,h,w]
        out1 = self.layer1(out0) # [b,64,t,h,w]
        out2 = self.layer2(out1) # [b,128,t,h/2,w/2]
        out3 = self.layer3(out2) # [b,256,t,h/4,w/4]

        out4 = self.layer4(out3) # [b,512,t,h/4,w/4]     
        # Temporal Pooling, TP
        outs_1 = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        
        # Horizontal Pooling Matching, HPM
        feat_1 = self.HPP(outs_1)  # [n, c, p]
        
        embed_1 = self.FCs(feat_1)  # [n, c, p]

        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p] , [n, class_num, p]

        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval