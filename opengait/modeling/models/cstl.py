import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from ..base_model import BaseModel
from einops import rearrange


from ..modules import SetBlockWrapper
from ..backbones.cstl_module import MSTE, ATA, SSFL,SetBlock

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv2d(x)
        return F.leaky_relu(x, inplace=True)

class CSTL(BaseModel):
        
    def build_network(self, model_cfg):
        self.hidden_dim = model_cfg['hidden_dim']
        part_num = model_cfg['part_num']
        div = model_cfg['div']
        class_num = model_cfg['class_num']
        _in_channels = 1
        _channels = [32, 64, 128]

        # 2D Convolution
        self.conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1),True)
        self.conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))
        # three modules
        self.multi_scale = MSTE(_channels[2], _channels[2], part_num)
        self.adaptive_aggregation = ATA(_channels[2], part_num, div)
        self.salient_learning = SSFL(_channels[2], _channels[2], part_num, class_num)

        # separate FC
        self.fc_bin = nn.Parameter(
                init.xavier_uniform_(
                    torch.zeros(part_num, _channels[2]*3, self.hidden_dim)))

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0].unsqueeze(2)
        del ipts

        x = self.conv2d_1(sils) # [n,s,c,w,h]
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x) # [n,s,c,w/2,h/2]
        x = x.max(-1)[0] + x.mean(-1)  # [n,s,c,w/2]

        
        
        t_f, t_s, t_l = self.multi_scale(x)

        aggregated_feature = self.adaptive_aggregation(t_f, t_s, t_l)

        part_classification, weighted_part_feature, selected_part_feature = self.salient_learning(t_f, t_s, t_l)

        feature = torch.cat([aggregated_feature, weighted_part_feature, selected_part_feature], -1)
        feature = feature.matmul(self.fc_bin)
        feature = feature.permute(1, 0, 2).contiguous()
                
        retval = {
            'training_feat': {
                'triplet': {'embeddings': feature, 'labels': labs},
                'softmax': {'logits': part_classification, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': feature
            }
        }

        return retval